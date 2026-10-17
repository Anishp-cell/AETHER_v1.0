#!/usr/bin/env python3
"""
Group-Disjoint and Random Split Generator for ARN.
Creates group-disjoint splits based on semantic frames (tool signature + slot structure
+ normalized syntactic utterance skeleton) with group-level tool stratification
to ensure every tool appears in all splits without train/test template leakage.
"""

import argparse
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.make_splits")


def extract_semantic_frame(sample: Dict[str, Any]) -> str:
    """
    Extract a formal semantic frame key:
    Tool Signature || Normalized Syntactic Utterance Skeleton
    where argument values are replaced by `<SLOT_KEY>` and superficial fillers are stripped.
    """
    user_msg = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    asst_msg = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
    calls = json.loads(asst_msg)

    # 1. Multi-tool call signature
    tool_sig = "+".join(sorted(c["name"] for c in calls))

    # 2. Extract slot arguments and replace in reverse length order
    slot_pairs = []
    for c in calls:
        for k, v in c.get("arguments", {}).items():
            if str(v).strip():
                slot_pairs.append((k, str(v)))
    slot_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    skel = user_msg.lower()
    for k, v in slot_pairs:
        if len(v) >= 1:
            skel = skel.replace(v.lower(), f"<{k.upper()}>")

    # 3. Normalize conversational filler prefixes and suffixes
    fillers = (
        r"\b(please|plz|plzzz|hey|umm|yaar|bro|dude|ty|thanks|thx|for me|"
        r"will you|right so|just|ok|alright|aether|can you|could you|"
        r"i need you to|real quick|quickly|asap|rn|na)\b"
    )
    skel = re.sub(fillers, "", skel)
    skel = re.sub(r"\s+", " ", skel).strip()

    return f"{tool_sig} || {skel}"


def make_group_disjoint_splits(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Partition dataset into group-disjoint train, val, and test splits with
    group-level tool stratification ensuring all 18 tools are present in all splits.
    """
    rng = random.Random(seed)

    # 1. Group samples by semantic frame
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        frame = extract_semantic_frame(sample)
        groups.setdefault(frame, []).append(sample)

    # Map groups to tools and tools to groups
    tool_to_groups: Dict[str, List[str]] = {}
    group_tools: Dict[str, Set[str]] = {}

    for g, slist in groups.items():
        tools = set()
        for s in slist:
            asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
            for c in json.loads(asst):
                tools.add(c["name"])
        group_tools[g] = tools
        for t in tools:
            tool_to_groups.setdefault(t, []).append(g)

    # Sort tools by frequency of groups (rarest first, e.g. open_url with 3 groups)
    sorted_tools = sorted(tool_to_groups.keys(), key=lambda t: len(tool_to_groups[t]))

    assigned: Dict[str, str] = {}
    tool_counts_split: Dict[str, Dict[str, int]] = {t: {"train": 0, "val": 0, "test": 0} for t in sorted_tools}

    # 2. Stratification: ensure every tool has groups in train, test, val
    for t in sorted_tools:
        grps = [g for g in tool_to_groups[t] if g not in assigned]
        grps.sort(key=lambda g: len(groups[g]), reverse=True)
        if not grps:
            continue

        # Largest available group goes to train
        if tool_counts_split[t]["train"] == 0 and grps:
            g_train = grps.pop(0)
            assigned[g_train] = "train"
            for tool in group_tools[g_train]:
                tool_counts_split[tool]["train"] += len(groups[g_train])

        # Next available group goes to test
        if tool_counts_split[t]["test"] == 0 and grps:
            g_test = grps.pop(-1)
            assigned[g_test] = "test"
            for tool in group_tools[g_test]:
                tool_counts_split[tool]["test"] += len(groups[g_test])

        # Next available group goes to val
        if tool_counts_split[t]["val"] == 0 and grps:
            g_val = grps.pop(-1)
            assigned[g_val] = "val"
            for tool in group_tools[g_val]:
                tool_counts_split[tool]["val"] += len(groups[g_val])

    # 3. Fill remaining groups balancing toward target ratios
    total_samples = len(samples)
    target_test = int(total_samples * test_ratio)
    target_val = int(total_samples * val_ratio)

    remaining = [g for g in groups if g not in assigned]
    rng.shuffle(remaining)

    for g in remaining:
        curr_test = sum(len(groups[k]) for k, v in assigned.items() if v == "test")
        curr_val = sum(len(groups[k]) for k, v in assigned.items() if v == "val")

        if curr_test < target_test:
            assigned[g] = "test"
        elif curr_val < target_val:
            assigned[g] = "val"
        else:
            assigned[g] = "train"

        for tool in group_tools[g]:
            tool_counts_split[tool][assigned[g]] += len(groups[g])

    # Materialize sample lists
    train_samples = [s for g, split in assigned.items() if split == "train" for s in groups[g]]
    val_samples = [s for g, split in assigned.items() if split == "val" for s in groups[g]]
    test_samples = [s for g, split in assigned.items() if split == "test" for s in groups[g]]

    train_groups = {g for g, split in assigned.items() if split == "train"}
    val_groups = {g for g, split in assigned.items() if split == "val"}
    test_groups = {g for g, split in assigned.items() if split == "test"}

    # Verification: Assert disjoint groups
    assert len(train_groups & val_groups) == 0, "Train and Val groups overlap!"
    assert len(train_groups & test_groups) == 0, "Train and Test groups overlap!"
    assert len(val_groups & test_groups) == 0, "Val and Test groups overlap!"

    # Verification: Assert zero raw utterance text leakage
    train_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in train_samples}
    test_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in test_samples}
    leakage = train_utts & test_utts

    summary = {
        "total_samples": total_samples,
        "distinct_semantic_groups": len(groups),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "utterance_leakage_count": len(leakage),
        "leakage_clean": len(leakage) == 0,
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
        samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    logger.info("Group-disjoint split summary:")
    logger.info("Total: %d samples, %d groups", summary["total_samples"], summary["distinct_semantic_groups"])
    logger.info("Train: %d samples (%d groups)", summary["train_samples"], summary["train_groups"])
    logger.info("Val:   %d samples (%d groups)", summary["val_samples"], summary["val_groups"])
    logger.info("Test:  %d samples (%d groups)", summary["test_samples"], summary["test_groups"])
    logger.info("Utterance leakage: %d (clean=%s)", summary["utterance_leakage_count"], summary["leakage_clean"])

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
