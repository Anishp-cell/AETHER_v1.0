import json
import logging
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL, ARG_TO_SLOT_BASE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.audit")

def extract_spans_from_bio(tags):
    spans = set()
    curr_type = None
    start = -1
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if curr_type is not None:
                spans.add((curr_type, start, i))
            curr_type = tag[2:]
            start = i
        elif tag.startswith("I-"):
            ttype = tag[2:]
            if curr_type != ttype:
                if curr_type is not None:
                    spans.add((curr_type, start, i))
                curr_type = ttype
                start = i
        else:
            if curr_type is not None:
                spans.add((curr_type, start, i))
                curr_type = None
                start = -1
    if curr_type is not None:
        spans.add((curr_type, start, len(tags)))
    return spans

def run_label_audit():
    print("="*90)
    print("ITEM 1: COMPREHENSIVE LABEL AUDIT (RAW JSON VS FIXED TAG CONVERSION)")
    print("="*90)

    audit_results = {}

    for split_name, split_path in [
        ("train", "ARN/data/splits/group_disjoint_train.jsonl"),
        ("val", "ARN/data/splits/group_disjoint_val.jsonl"),
    ]:
        tool_sample_count = Counter()
        tool_samples_with_gold_spans = Counter()
        tool_total_gold_spans = Counter()
        tool_raw_args_counter = defaultdict(Counter)
        tool_arg_matched_counter = defaultdict(Counter)
        slot_type_counter = Counter()

        total_samples = 0
        total_raw_args = 0
        total_matched_spans = 0

        # Load raw JSONL
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_samples += 1
                obj = json.loads(line)
                user_text = ""
                assistant_content = ""
                for msg in obj.get("messages", []):
                    if msg.get("role") == "user":
                        user_text = msg.get("content", "").strip()
                    elif msg.get("role") == "assistant":
                        assistant_content = msg.get("content", "").strip()

                tool_calls = []
                if assistant_content:
                    try:
                        parsed = json.loads(assistant_content)
                        if isinstance(parsed, list):
                            tool_calls = parsed
                        elif isinstance(parsed, dict):
                            tool_calls = [parsed]
                    except Exception:
                        pass

                words = user_text.split()
                words_clean = [w.lower().strip("',.\"?;:") for w in words]

                sample_has_span = False
                for tc in tool_calls:
                    tname = tc.get("name", "")
                    tool_sample_count[tname] += 1
                    args = tc.get("arguments", {})
                    if not isinstance(args, dict):
                        continue

                    spans_for_tool = 0
                    for k, v in args.items():
                        total_raw_args += 1
                        val_str = str(v).strip()
                        tool_raw_args_counter[tname][k] += 1
                        val_words = val_str.split()
                        if not val_words:
                            continue
                        val_clean = [w.lower().strip("',.\"?;:") for w in val_words]
                        val_len = len(val_clean)

                        matched = False
                        for i in range(len(words_clean) - val_len + 1):
                            if words_clean[i:i+val_len] == val_clean:
                                matched = True
                                break

                        if matched:
                            base_slot = ARG_TO_SLOT_BASE.get(k, k)
                            slot_type_counter[base_slot] += 1
                            tool_arg_matched_counter[tname][k] += 1
                            spans_for_tool += 1
                            total_matched_spans += 1

                    tool_total_gold_spans[tname] += spans_for_tool
                    if spans_for_tool > 0:
                        tool_samples_with_gold_spans[tname] += 1

        audit_results[split_name] = {
            "total_samples": total_samples,
            "total_raw_args": total_raw_args,
            "total_matched_spans": total_matched_spans,
            "tool_sample_count": dict(tool_sample_count),
            "tool_samples_with_gold_spans": dict(tool_samples_with_gold_spans),
            "tool_total_gold_spans": dict(tool_total_gold_spans),
            "tool_raw_args": {t: dict(c) for t, c in tool_raw_args_counter.items()},
            "tool_arg_matched": {t: dict(c) for t, c in tool_arg_matched_counter.items()},
            "slot_type_counts": dict(slot_type_counter),
        }

    with open("ARN/results/stage1c_label_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    return audit_results

if __name__ == "__main__":
    run_label_audit()
