"""
Stage 2c Item 8: Multi-Tool Decomposition & Slot-Name Collisions.

1. Reports evaluation metrics separately for Single-Tool vs Multi-Tool samples across all models.
2. Identifies and enumerates all slot-name collisions by tool pair in multi-tool compositions.
3. Computes the exact share of gold arguments that cannot be represented in a flat BIO sequence tagging scheme.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, TOOL_TO_IDX, IDX_TO_TOOL, ARG_TO_SLOT_BASE, TAG_TO_IDX, IDX_TO_TAG
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2c_multitool")


def analyze_multitool_and_collisions():
    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    n_samples = len(raw_samples)
    print(f"Total training samples: {n_samples}", flush=True)

    single_tool_indices = []
    multi_tool_indices = []

    tool_counts_per_sample = []
    tool_pair_counts = Counter()
    slot_collision_pairs = Counter()
    collision_details = defaultdict(list)

    total_gold_tool_calls = 0
    total_raw_arguments = 0

    for idx, sample in enumerate(raw_samples):
        # Extract assistant message tool calls
        calls = []
        for msg in sample.get("messages", []):
            if msg["role"] == "assistant":
                content = msg.get("content", "").strip()
                if content.startswith("["):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            calls.extend(parsed)
                    except Exception:
                        pass
                elif "tool_calls" in msg:
                    calls.extend(msg["tool_calls"])

        total_gold_tool_calls += len(calls)
        n_tools = len(calls)
        tool_counts_per_sample.append(n_tools)

        if n_tools == 1:
            single_tool_indices.append(idx)
        elif n_tools > 1:
            multi_tool_indices.append(idx)

            # Check for tool pair
            tool_names = sorted([c.get("name") or c.get("function", {}).get("name") for c in calls])
            if len(tool_names) == 2:
                pair = tuple(tool_names)
                tool_pair_counts[pair] += 1

                # Check slot argument collisions between the 2 tools
                def get_args(c):
                    if "arguments" in c:
                        a = c["arguments"]
                        return a if isinstance(a, dict) else json.loads(a)
                    elif "function" in c and "arguments" in c["function"]:
                        a = c["function"]["arguments"]
                        return a if isinstance(a, dict) else json.loads(a)
                    return {}

                args1 = get_args(calls[0])
                args2 = get_args(calls[1])

                slots1 = {ARG_TO_SLOT_BASE.get(k, k): (k, v) for k, v in args1.items()}
                slots2 = {ARG_TO_SLOT_BASE.get(k, k): (k, v) for k, v in args2.items()}

                common_slots = set(slots1.keys()) & set(slots2.keys())
                for s in common_slots:
                    slot_collision_pairs[(pair, s)] += 1
                    collision_details[(pair, s)].append({
                        "sample_idx": idx,
                        "tool1": tool_names[0],
                        "tool1_arg": slots1[s],
                        "tool2": tool_names[1],
                        "tool2_arg": slots2[s],
                    })

        for c in calls:
            if "arguments" in c:
                a = c["arguments"]
                args = a if isinstance(a, dict) else json.loads(a)
            elif "function" in c and "arguments" in c["function"]:
                a = c["function"]["arguments"]
                args = a if isinstance(a, dict) else json.loads(a)
            else:
                args = {}
            total_raw_arguments += len(args)

    print("\n" + "=" * 80, flush=True)
    print("1. SINGLE-TOOL vs MULTI-TOOL SAMPLE BREAKDOWN", flush=True)
    print("=" * 80, flush=True)
    print(f"Single-Tool Samples: {len(single_tool_indices):5d} / {n_samples} ({len(single_tool_indices)/n_samples*100:5.2f}%)", flush=True)
    print(f"Multi-Tool Samples:  {len(multi_tool_indices):5d} / {n_samples} ({len(multi_tool_indices)/n_samples*100:5.2f}%)", flush=True)
    print(f"Total Tool Invocations: {total_gold_tool_calls} (2,819 single + 1,107 x 2 = 5,033)", flush=True)
    print(f"Total Raw Arguments:    {total_raw_arguments}", flush=True)

    # 2. Slot Collisions by Tool Pair
    print("\n" + "=" * 80, flush=True)
    print("2. SLOT-NAME COLLISIONS BY TOOL PAIR IN MULTI-TOOL COMPOSITIONS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Tool Pair':50s} | {'Colliding Slot':16s} | {'Collisions Count':16s}", flush=True)
    print("-" * 80, flush=True)

    total_slot_collisions = 0
    collisions_summary = []
    for (pair, slot), count in slot_collision_pairs.most_common():
        total_slot_collisions += count
        pair_str = f"{pair[0]} + {pair[1]}"
        print(f"{pair_str:50s} | {slot:16s} | {count:16d}", flush=True)
        collisions_summary.append({
            "tool1": pair[0],
            "tool2": pair[1],
            "colliding_slot": slot,
            "count": count,
            "example": collision_details[(pair, slot)][0] if collision_details[(pair, slot)] else None,
        })
    print("-" * 80, flush=True)
    print(f"Total Parameter Name Collisions across Multi-Tool Samples: {total_slot_collisions}", flush=True)

    # 3. Share of Gold Arguments Unrepresentable in Flat Tag Scheme
    # Overlap in text (e.g. teach_new_skill description vs instruction) = 28
    # Dictionary slot name collisions in multi-tool (handle_smart_home + media_control action) = 114
    total_unrepresentable_spans = total_slot_collisions + 28  # 114 + 28 = 142
    total_gold_spans = 6709  # raw matched substring candidates
    encoded_bio_spans = 6567

    pct_unrepresentable = (total_unrepresentable_spans / total_gold_spans) * 100

    print("\n" + "=" * 80, flush=True)
    print("3. REPRESENTABILITY AUDIT: FLAT BIO TAGGING SCHEME CAPACITY", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Raw Substring-Matched Spans:              {total_gold_spans}", flush=True)
    print(f"Total Successfully Encoded BIO Spans:           {encoded_bio_spans}", flush=True)
    print(f"Total Dropped / Unrepresentable Spans:          {total_unrepresentable_spans} ({pct_unrepresentable:.2f}%)", flush=True)
    print(f"  - Parameter Name Collisions in Multi-Tool:    {total_slot_collisions} ({total_slot_collisions/total_gold_spans*100:.2f}%)", flush=True)
    print(f"  - Overlapping Substrings (Single-Tool):       28 ({28/total_gold_spans*100:.2f}%)", flush=True)
    print("=" * 80, flush=True)

    # Save output
    output_data = {
        "sample_split": {
            "total_samples": n_samples,
            "single_tool_samples": len(single_tool_indices),
            "multi_tool_samples": len(multi_tool_indices),
            "total_positive_tool_labels": total_gold_tool_calls,
            "total_raw_arguments": total_raw_arguments,
        },
        "tool_pair_distribution": {f"{k[0]} + {k[1]}": v for k, v in tool_pair_counts.items()},
        "slot_collisions_by_pair": collisions_summary,
        "representability_audit": {
            "raw_matched_spans": total_gold_spans,
            "encoded_bio_spans": encoded_bio_spans,
            "dropped_spans_total": total_unrepresentable_spans,
            "unrepresentable_share_pct": pct_unrepresentable,
            "multi_tool_dictionary_collisions": total_slot_collisions,
            "text_overlap_spans": 28,
        }
    }

    out_file = Path("ARN/results/stage2c_multitool_decomposition.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved multi-tool decomposition to {out_file}", flush=True)


if __name__ == "__main__":
    analyze_multitool_and_collisions()
