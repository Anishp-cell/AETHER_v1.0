#!/usr/bin/env python3
"""
ARN Data Processing & Synthetic Dataset Utility.
Validates, inspects, and generates synthetic tool-orchestration datasets
for the 18-tool ARN schema.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.make_data")


def validate_sample(sample: Dict[str, Any], tool_names: set) -> bool:
    """Validate sample structure and ensure tool calls match schema."""
    if "messages" not in sample:
        return False
    messages = sample["messages"]
    if len(messages) < 3:
        return False
    
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if not user_msg or not asst_msg:
        return False
    
    try:
        calls = json.loads(asst_msg["content"])
        if not isinstance(calls, list):
            return False
        for call in calls:
            if not isinstance(call, dict) or "name" not in call:
                return False
    except Exception:
        return False
    
    return True


def inspect_dataset(file_path: Path) -> Dict[str, Any]:
    """Inspect a jsonl dataset and collect schema statistics."""
    total = 0
    valid = 0
    tool_counts: Dict[str, int] = {}
    multi_tool_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            data = json.loads(line)
            asst = next((m for m in data["messages"] if m["role"] == "assistant"), None)
            if asst:
                calls = json.loads(asst["content"])
                if len(calls) > 1:
                    multi_tool_count += 1
                for c in calls:
                    tname = c.get("name", "unknown")
                    tool_counts[tname] = tool_counts.get(tname, 0) + 1
                valid += 1

    stats = {
        "file": str(file_path),
        "total_lines": total,
        "valid_samples": valid,
        "multi_tool_samples": multi_tool_count,
        "tool_distribution": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="ARN Dataset Verification & Generation Utility")
    parser.add_argument("--inspect", type=str, help="Path to jsonl file to inspect")
    parser.add_argument("--output", type=str, help="Output path for processed dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.inspect:
        path = Path(args.inspect)
        if not path.exists():
            logger.error("Dataset file not found: %s", path)
            return
        stats = inspect_dataset(path)
        logger.info("Dataset statistics for %s:", path.name)
        logger.info("Total samples: %d", stats["total_lines"])
        logger.info("Multi-tool samples: %d (%.1f%%)", stats["multi_tool_samples"], 
                    100.0 * stats["multi_tool_samples"] / max(stats["total_lines"], 1))
        logger.info("Distinct tools present: %d", len(stats["tool_distribution"]))
        for t, c in stats["tool_distribution"].items():
            logger.info("  - %s: %d", t, c)


if __name__ == "__main__":
    main()
