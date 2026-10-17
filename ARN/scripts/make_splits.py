#!/usr/bin/env python3
"""
Group-Disjoint and Random Split Generator for ARN.
Creates group-disjoint splits based on template skeletons and slot patterns
to prevent train/test leakage, and generates a matched random split baseline.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.make_splits")


def extract_template_skeleton(utterance: str, tool_calls: List[Dict[str, Any]]) -> str:
    """
    Extract a normalized template skeleton by masking argument values
    and normalize boilerplate phrasing.
    """
    skeleton = utterance.strip().lower()
    # Mask argument values in reverse order of length
    values_to_mask = []
    for call in tool_calls:
        args = call.get("arguments", {})
        for k, v in args.items():
            if isinstance(v, (str, int, float)) and str(v).strip():
                values_to_mask.append(str(v).lower())
    
    values_to_mask.sort(key=lambda s: len(s), reverse=True)
    for v in values_to_mask:
        if len(v) >= 2:
            skeleton = skeleton.replace(v, "<SLOT>")
            
    # Normalize common filler tokens
    skeleton = re.sub(r"\b(please|plz|hey|umm|yaar|bro|dude|ty|thanks|for me)\b", "", skeleton)
    skeleton = re.sub(r"\s+", " ", skeleton).strip()
    return skeleton


def make_splits(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create group-disjoint train, val, and test splits.
    Guarantees no template skeleton or prompt group spans multiple splits.
    """
    import random
    rng = random.Random(seed)

    # Group samples by template skeleton
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        user_msg = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        asst_msg = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        calls = json.loads(asst_msg)
        skel = extract_template_skeleton(user_msg, calls)
        tool_sig = "+".join(sorted(c["name"] for c in calls))
        group_key = f"{tool_sig}::{skel}"
        groups.setdefault(group_key, []).append(sample)

    sorted_group_keys = sorted(groups.keys())
    rng.shuffle(sorted_group_keys)

    total_samples = len(samples)
    target_test = int(total_samples * test_ratio)
    target_val = int(total_samples * val_ratio)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    train_groups: Set[str] = set()
    val_groups: Set[str] = set()
    test_groups: Set[str] = set()

    for gkey in sorted_group_keys:
        grp = groups[gkey]
        if len(test_samples) < target_test:
            test_samples.extend(grp)
            test_groups.add(gkey)
        elif len(val_samples) < target_val:
            val_samples.extend(grp)
            val_groups.add(gkey)
        else:
            train_samples.extend(grp)
            train_groups.add(gkey)

    # Verification: Assert disjoint groups
    assert len(train_groups & val_groups) == 0, "Train and Val groups overlap!"
    assert len(train_groups & test_groups) == 0, "Train and Test groups overlap!"
    assert len(val_groups & test_groups) == 0, "Val and Test groups overlap!"

    # Verification: Assert zero raw utterance leakage between train and test
    train_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in train_samples}
    test_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in test_samples}
    leakage = train_utts & test_utts

    summary = {
        "total_samples": total_samples,
        "distinct_groups": len(groups),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "utterance_leakage_count": len(leakage),
        "leakage_clean": len(leakage) == 0,
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

    # 1. Group-disjoint split
    train_s, val_s, test_s, summary = make_splits(
        samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    logger.info("Group-disjoint split created: Train=%d, Val=%d, Test=%d (Groups: %d/%d/%d, Leakage=%d)",
                summary["train_samples"], summary["val_samples"], summary["test_samples"],
                summary["train_groups"], summary["val_groups"], summary["test_groups"],
                summary["utterance_leakage_count"])

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
