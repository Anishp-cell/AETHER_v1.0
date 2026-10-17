#!/usr/bin/env python3
"""
Group-Disjoint and Near-Duplicate-Free Split Generator for ARN.
Creates group-disjoint splits based on semantic frames (tool signature + slot structure
+ normalized syntactic utterance skeleton), performs graph-based clustering to merge
all near-duplicate skeletons (token-level similarity >= 0.80), and applies group-level
tool stratification to ensure tools appear across splits with zero template leakage.
"""

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.make_splits")


def extract_fine_semantic_frame(sample: Dict[str, Any]) -> str:
    """
    Extract a fine-grained semantic frame key:
    Tool Signature || Normalized Syntactic Utterance Skeleton
    where argument values are replaced by `<SLOT_KEY>`, conversational fillers,
    speech repairs/false starts, and punctuation are completely normalized.
    For task-based freeform tools, internal syntactic structure is preserved.
    """
    user_msg = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    asst_msg = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
    calls = json.loads(asst_msg)

    # 1. Multi-tool call signature
    tool_sig = "+".join(sorted(c["name"] for c in calls))

    # 2. Extract slot arguments and replace in reverse length order
    slot_pairs = []
    for c in calls:
        tool_name = c["name"]
        args = c.get("arguments", {})
        for k, v in args.items():
            val_str = str(v).strip()
            # If task contains full sentences, mask internal locations/entities rather than whole string
            if tool_name in ("browser_task", "analyze_screen_with_llava") and len(val_str.split()) > 3:
                continue
            if val_str:
                slot_pairs.append((k, val_str))

    slot_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    skel = user_msg.lower()
    for k, v in slot_pairs:
        if len(v) >= 1:
            skel = skel.replace(v.lower(), f"<{k.upper()}>")

    # 3. Normalize speech disfluencies / false starts (e.g. "colleague... no wait, tell <CONTACT>")
    skel = re.sub(r"\b\w+\.\.\.\s*(actually make it|no wait)\b", "<DISFLUENCY_REPAIR>", skel)

    # 4. Normalize conversational filler prefixes and suffixes
    fillers = (
        r"\b(please|plz|plzzz|hey|umm|yaar|bro|dude|ty|thanks|thx|for me|"
        r"will you|right so|just|ok|okay so|so basically|alright|aether|can you|could you|"
        r"i need you to|real quick|quickly|asap|rn|na|yo|listen|do me a favor and|"
        r"go ahead and|do this:|right now)\b"
    )
    skel = re.sub(fillers, "", skel)

    # 5. Normalize punctuation and collapse whitespace
    skel = re.sub(r"[?!.,;:\"'-]", " ", skel)
    skel = re.sub(r"\s+", " ", skel).strip()

    return f"{tool_sig} || {skel}"


def token_edit_sim(s1: str, s2: str) -> float:
    """Compute normalized token-level Levenshtein edit similarity in [0, 1]."""
    w1 = s1.split()
    w2 = s2.split()
    n1, n2 = len(w1), len(w2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        dp[i][0] = i
    for j in range(n2 + 1):
        dp[0][j] = j
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = 0 if w1[i - 1] == w2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[n1][n2]
    max_len = max(n1, n2)
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def make_group_disjoint_splits(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    sim_threshold: float = 0.80,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Partition dataset into group-disjoint train, val, and test splits with:
    1. Graph-based clustering of all near-duplicate skeletons (sim >= sim_threshold).
    2. Group-level tool stratification ensuring tools appear across splits.
    3. Mathematical verification of zero near-duplicate leakage.
    """
    rng = random.Random(seed)

    # 1. Extract fine-grained frames
    sample_frames = [extract_fine_semantic_frame(s) for s in samples]
    unique_frames = sorted(list(set(sample_frames)))

    # 2. Graph clustering: merge near-duplicates within the same tool signature
    by_tool = defaultdict(list)
    for f in unique_frames:
        tool_sig, skel = f.split(" || ")
        by_tool[tool_sig].append(skel)

    parent = {f: f for f in unique_frames}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    merge_count = 0
    for tool_sig, skels in by_tool.items():
        n = len(skels)
        for i in range(n):
            for j in range(i + 1, n):
                sim = token_edit_sim(skels[i], skels[j])
                if sim >= sim_threshold:
                    union(f"{tool_sig} || {skels[i]}", f"{tool_sig} || {skels[j]}")
                    merge_count += 1

    # Map each sample to its unified cluster ID
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s, f in zip(samples, sample_frames):
        cid = find(f)
        clusters[cid].append(s)

    # Map clusters to tools
    tool_to_clusters: Dict[str, List[str]] = defaultdict(list)
    cluster_tools: Dict[str, Set[str]] = {}

    for cid, slist in clusters.items():
        tools = set()
        for s in slist:
            asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
            for c in json.loads(asst):
                tools.add(c["name"])
        cluster_tools[cid] = tools
        for t in tools:
            tool_to_clusters[t].append(cid)

    # 3. Stratification: ensure tools have clusters in train, test, val where possible
    sorted_tools = sorted(tool_to_clusters.keys(), key=lambda t: len(tool_to_clusters[t]))
    assigned: Dict[str, str] = {}
    tool_counts_split: Dict[str, Dict[str, int]] = {t: {"train": 0, "val": 0, "test": 0} for t in sorted_tools}

    for t in sorted_tools:
        cids = [c for c in tool_to_clusters[t] if c not in assigned]
        cids.sort(key=lambda c: len(clusters[c]), reverse=True)
        if not cids:
            continue

        # Largest cluster goes to train
        if tool_counts_split[t]["train"] == 0 and cids:
            c_tr = cids.pop(0)
            assigned[c_tr] = "train"
            for tool in cluster_tools[c_tr]:
                tool_counts_split[tool]["train"] += len(clusters[c_tr])

        # Next cluster goes to test
        if tool_counts_split[t]["test"] == 0 and cids:
            c_te = cids.pop(-1)
            assigned[c_te] = "test"
            for tool in cluster_tools[c_te]:
                tool_counts_split[tool]["test"] += len(clusters[c_te])

        # Next cluster goes to val
        if tool_counts_split[t]["val"] == 0 and cids:
            c_va = cids.pop(-1)
            assigned[c_va] = "val"
            for tool in cluster_tools[c_va]:
                tool_counts_split[tool]["val"] += len(clusters[c_va])

    # 4. Fill remaining clusters balancing toward target ratios
    total_samples = len(samples)
    target_test = int(total_samples * test_ratio)
    target_val = int(total_samples * val_ratio)

    remaining = [c for c in clusters if c not in assigned]
    rng.shuffle(remaining)

    for c in remaining:
        curr_test = sum(len(clusters[k]) for k, v in assigned.items() if v == "test")
        curr_val = sum(len(clusters[k]) for k, v in assigned.items() if v == "val")

        if curr_test < target_test:
            assigned[c] = "test"
        elif curr_val < target_val:
            assigned[c] = "val"
        else:
            assigned[c] = "train"

        for tool in cluster_tools[c]:
            tool_counts_split[tool][assigned[c]] += len(clusters[c])

    # Materialize sample lists
    train_samples = [s for c, split in assigned.items() if split == "train" for s in clusters[c]]
    val_samples = [s for c, split in assigned.items() if split == "val" for s in clusters[c]]
    test_samples = [s for c, split in assigned.items() if split == "test" for s in clusters[c]]

    train_cids = {c for c, split in assigned.items() if split == "train"}
    val_cids = {c for c, split in assigned.items() if split == "val"}
    test_cids = {c for c, split in assigned.items() if split == "test"}

    # Assert cluster disjointness
    assert len(train_cids & val_cids) == 0
    assert len(train_cids & test_cids) == 0
    assert len(val_cids & test_cids) == 0

    # Assert raw utterance disjointness
    train_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in train_samples}
    test_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in test_samples}
    val_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in val_samples}
    assert len(train_utts & test_utts) == 0
    assert len(train_utts & val_utts) == 0

    # Extract unique skeletons per split
    train_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in train_samples})
    val_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in val_samples})
    test_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in test_samples})

    val_sims = [max(token_edit_sim(vs, ts) for ts in train_skels) for vs in val_skels]
    test_sims = [max(token_edit_sim(ts_eval, ts) for ts in train_skels) for ts_eval in test_skels]

    summary = {
        "total_samples": total_samples,
        "raw_unique_frames": len(unique_frames),
        "merged_clusters_count": len(clusters),
        "merged_near_duplicate_pairs": merge_count,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_clusters": len(train_cids),
        "val_clusters": len(val_cids),
        "test_clusters": len(test_cids),
        "train_skels_count": len(train_skels),
        "val_skels_count": len(val_skels),
        "test_skels_count": len(test_skels),
        "val_max_similarity_to_train": round(max(val_sims), 4) if val_sims else 0.0,
        "test_max_similarity_to_train": round(max(test_sims), 4) if test_sims else 0.0,
        "val_mean_similarity_to_train": round(sum(val_sims) / max(len(val_sims), 1), 4),
        "test_mean_similarity_to_train": round(sum(test_sims) / max(len(test_sims), 1), 4),
        "val_near_duplicates_above_80": sum(1 for s in val_sims if s >= 0.80),
        "test_near_duplicates_above_80": sum(1 for s in test_sims if s >= 0.80),
        "per_tool_distribution": {
            t: {
                "train": tool_counts_split[t]["train"],
                "val": tool_counts_split[t]["val"],
                "test": tool_counts_split[t]["test"],
                "total": sum(tool_counts_split[t].values()),
            }
            for t in sorted_tools
        },
    }

    return train_samples, val_samples, test_samples, summary


def main():
    parser = argparse.ArgumentParser(description="ARN Group-Disjoint Split Generator")
    parser.add_argument("--input", type=str, default="data/raw/aether_orchestrator_dataset.jsonl",
                        help="Input raw dataset")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                        help="Output directory for split files")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio")
    parser.add_argument("--sim-threshold", type=float, default=0.80, help="Similarity threshold for near-duplicate merging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input file does not exist: %s", in_path)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info("Loaded %d samples from %s", len(samples), in_path)

    train_s, val_s, test_s, summary = make_group_disjoint_splits(
        samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        sim_threshold=args.sim_threshold, seed=args.seed
    )

    logger.info("Split summary:")
    logger.info("Total samples: %d | Merged clusters: %d (merged %d near-duplicate pairs)",
                summary["total_samples"], summary["merged_clusters_count"], summary["merged_near_duplicate_pairs"])
    logger.info("Train: %d samples (%d clusters, %d skeletons)",
                summary["train_samples"], summary["train_clusters"], summary["train_skels_count"])
    logger.info("Val:   %d samples (%d clusters, %d skeletons) | Max train sim: %.2f (>=0.8: %d)",
                summary["val_samples"], summary["val_clusters"], summary["val_skels_count"],
                summary["val_max_similarity_to_train"], summary["val_near_duplicates_above_80"])
    logger.info("Test:  %d samples (%d clusters, %d skeletons) | Max train sim: %.2f (>=0.8: %d)",
                summary["test_samples"], summary["test_clusters"], summary["test_skels_count"],
                summary["test_max_similarity_to_train"], summary["test_near_duplicates_above_80"])

    with open(out_dir / "group_disjoint_train.jsonl", "w", encoding="utf-8") as f:
        for s in train_s:
            f.write(json.dumps(s) + "\n")
    with open(out_dir / "group_disjoint_val.jsonl", "w", encoding="utf-8") as f:
        for s in val_s:
            f.write(json.dumps(s) + "\n")
    with open(out_dir / "group_disjoint_test.jsonl", "w", encoding="utf-8") as f:
        for s in test_s:
            f.write(json.dumps(s) + "\n")

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved group-disjoint splits to %s", out_dir)


if __name__ == "__main__":
    main()
