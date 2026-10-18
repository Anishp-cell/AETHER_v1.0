"""
Diagnostic script to reconcile:
1. Per-tool gold spans: 6,709 (train) and 704 (val) vs 6,567 and 696.
   - List every span dropped by overlap/collision by tool pair and argument pair.
2. List arguments never tagged because they are not verbatim in the text.
3. Fix the TP+FN (4,872) vs 5,033 positive label mismatch.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TAG, TOOL_TO_IDX, ARG_TO_SLOT_BASE, TAG_TO_IDX


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


def analyze_collisions(split_name, jsonl_path):
    print(f"\n{'='*80}\nANALYZING SPAN COLLISIONS FOR: {split_name} ({jsonl_path})\n{'='*80}")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        raw_lines = [json.loads(line) for line in f if line.strip()]

    ds = ARNDataset(jsonl_path, tokenizer=TOKENIZER)

    raw_matches_total = 0
    bio_spans_total = 0
    collision_events = []
    tool_pair_collisions = Counter()
    arg_pair_collisions = Counter()
    non_verbatim_args = defaultdict(lambda: {"total": 0, "matched": 0})

    for idx, obj in enumerate(raw_lines):
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

        # Find all independent substring matches
        all_candidate_spans = []  # list of (tool_name, arg_name, base_slot, start_word, end_word, val_str)
        for tc in tool_calls:
            tname = tc.get("name", "")
            args = tc.get("arguments", {})
            if not isinstance(args, dict):
                continue
            for k, v in args.items():
                val_str = str(v).strip()
                non_verbatim_args[f"{tname}.{k}"]["total"] += 1
                val_words = val_str.split()
                if not val_words:
                    continue
                val_clean = [w.lower().strip("',.\"?;:") for w in val_words]
                val_len = len(val_clean)

                for w_idx in range(len(words_clean) - val_len + 1):
                    if words_clean[w_idx : w_idx + val_len] == val_clean:
                        base_slot = ARG_TO_SLOT_BASE.get(k, k)
                        all_candidate_spans.append({
                            "tool": tname,
                            "arg": k,
                            "slot": base_slot,
                            "start": w_idx,
                            "end": w_idx + val_len,
                            "val": val_str
                        })
                        non_verbatim_args[f"{tname}.{k}"]["matched"] += 1
                        break

        raw_matches_total += len(all_candidate_spans)

        # Get actual BIO spans from dataset
        item = ds[idx]
        mask = item["attention_mask"].numpy()
        tags = item["tags"].numpy()
        length = int(mask.sum())
        t_strings = [IDX_TO_TAG.get(t, "O") for t in tags[:length]]
        bio_spans = extract_spans_from_bio(t_strings)
        bio_spans_total += len(bio_spans)

        # Check if raw_matches > bio_spans
        if len(all_candidate_spans) > len(bio_spans):
            # Collision occurred in this sample!
            # Find which spans collided / overlapped
            for i in range(len(all_candidate_spans)):
                for j in range(i + 1, len(all_candidate_spans)):
                    s1 = all_candidate_spans[i]
                    s2 = all_candidate_spans[j]
                    # Check overlap in word intervals [start, end)
                    if max(s1["start"], s2["start"]) < min(s1["end"], s2["end"]):
                        pair = tuple(sorted([s1["tool"], s2["tool"]]))
                        tool_pair_collisions[pair] += 1
                        arg_pair = tuple(sorted([f"{s1['tool']}.{s1['arg']}", f"{s2['tool']}.{s2['arg']}"]))
                        arg_pair_collisions[arg_pair] += 1
                        collision_events.append({
                            "sample_id": idx,
                            "text": user_text,
                            "span1": s1,
                            "span2": s2
                        })

    print(f"Total Raw Substring Matches : {raw_matches_total}")
    print(f"Total Encoded BIO Spans     : {bio_spans_total}")
    print(f"Total Spans Dropped by Overlap: {raw_matches_total - bio_spans_total}")
    print(f"Detected Overlapping Pairs  : {len(collision_events)}")
    print("\nCollisions by Tool Pair:")
    for pair, cnt in tool_pair_collisions.most_common():
        print(f"  {pair[0]} <-> {pair[1]}: {cnt}")

    print("\nCollisions by Argument Pair:")
    for pair, cnt in arg_pair_collisions.most_common(15):
        print(f"  {pair[0]} <-> {pair[1]}: {cnt}")

    return {
        "raw_matches": raw_matches_total,
        "bio_spans": bio_spans_total,
        "dropped_spans": raw_matches_total - bio_spans_total,
        "tool_pair_collisions": dict(tool_pair_collisions),
        "arg_pair_collisions": {f"{k[0]} <-> {k[1]}": v for k, v in arg_pair_collisions.items()},
        "non_verbatim_args": non_verbatim_args,
        "sample_collision_examples": collision_events[:5]
    }


def analyze_positive_labels():
    print(f"\n{'='*80}\nANALYZING POSITIVE LABELS (4,872 vs 5,033)\n{'='*80}")
    train_ds = ARNDataset("ARN/data/splits/group_disjoint_train.jsonl", tokenizer=TOKENIZER)
    total_pos = int(sum(item["tool_labels"].sum().item() for item in train_ds))
    print(f"Ground truth total positive labels in group_disjoint_train.jsonl: {total_pos}")

    # Check stage1b_train_metrics_and_overfit.py checkpoint evaluation
    ckpt_path = Path("ARN/results/checkpoints/stage1_checkpoint.pt")
    fixed_ckpt_path = Path("ARN/results/checkpoints/arn/stage1c_fixed_model.pt")

    print(f"Stage 1 checkpoint exists: {ckpt_path.exists()}")
    print(f"Stage 1c fixed checkpoint exists: {fixed_ckpt_path.exists()}")

    # Check stage1c_fixed_train_metrics.json
    with open("ARN/results/stage1c_fixed_train_metrics.json", "r") as f:
        fixed_metrics = json.load(f)
    print("Stage 1c fixed train metrics content:", json.dumps(fixed_metrics, indent=2))


if __name__ == "__main__":
    train_res = analyze_collisions("Train", "ARN/data/splits/group_disjoint_train.jsonl")
    val_res = analyze_collisions("Val", "ARN/data/splits/group_disjoint_val.jsonl")

    # Non-verbatim arguments across both splits
    print(f"\n{'='*80}\nARGUMENTS NEVER OR RARELY TAGGED (NON-VERBATIM IN TEXT)\n{'='*80}")
    all_args = train_res["non_verbatim_args"]
    never_tagged = []
    rarely_tagged = []
    for arg_name, counts in sorted(all_args.items()):
        tot = counts["total"]
        mat = counts["matched"]
        rate = mat / max(tot, 1)
        if mat == 0:
            never_tagged.append((arg_name, tot, mat))
            print(f"  NEVER TAGGED: {arg_name:40s} | Total: {tot:4d} | Matched: {mat:4d} (0.0%)")
        elif rate < 0.5:
            rarely_tagged.append((arg_name, tot, mat, rate))
            print(f"  RARELY TAGGED (<50%): {arg_name:30s} | Total: {tot:4d} | Matched: {mat:4d} ({rate*100:.1f}%)")

    analyze_positive_labels()

    out = {
        "train": {k: v for k, v in train_res.items() if k != "non_verbatim_args"},
        "val": {k: v for k, v in val_res.items() if k != "non_verbatim_args"},
        "never_tagged_arguments": never_tagged,
        "rarely_tagged_arguments": rarely_tagged
    }
    with open("ARN/results/stage2b_collision_and_label_reconciliation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved reconciliation to ARN/results/stage2b_collision_and_label_reconciliation.json")
