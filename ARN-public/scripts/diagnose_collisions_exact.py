import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TAG, ARG_TO_SLOT_BASE

def extract_spans(tags):
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

def analyze_dataset_diffs(split_name, path):
    print(f"\n{'='*70}\nAnalyzing: {split_name} ({path})\n{'='*70}")
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [json.loads(line) for line in f if line.strip()]
    ds = ARNDataset(path, tokenizer=TOKENIZER)

    reasons = Counter()
    tool_collisions = Counter()
    arg_collisions = Counter()
    diff_records = []

    for idx, (raw, item) in enumerate(zip(raw_lines, ds)):
        user_text = ""
        asst_text = ""
        for m in raw.get("messages", []):
            if m.get("role") == "user":
                user_text = m.get("content", "").strip()
            elif m.get("role") == "assistant":
                asst_text = m.get("content", "").strip()

        tcs = []
        if asst_text:
            try:
                parsed = json.loads(asst_text)
                if isinstance(parsed, list): tcs = parsed
                elif isinstance(parsed, dict): tcs = [parsed]
            except Exception:
                pass

        words = user_text.split()
        words_clean = [w.lower().strip("',.\"?;:") for w in words]

        raw_matches = []
        for tc in tcs:
            tname = tc.get("name", "")
            for k, v in tc.get("arguments", {}).items():
                v_words = [w.lower().strip("',.\"?;:") for w in str(v).split()]
                v_len = len(v_words)
                if v_len == 0: continue
                for i in range(len(words_clean) - v_len + 1):
                    if words_clean[i : i + v_len] == v_words:
                        base_slot = ARG_TO_SLOT_BASE.get(k, k)
                        raw_matches.append({
                            "tool": tname,
                            "arg": k,
                            "slot": base_slot,
                            "start": i,
                            "end": i + v_len,
                            "val": str(v)
                        })
                        break

        # Check BIO spans
        mask = item["attention_mask"].numpy()
        tags = item["tags"].numpy()
        length = int(mask.sum())
        t_strings = [IDX_TO_TAG.get(t, "O") for t in tags[:length]]
        bio_spans = extract_spans(t_strings)

        if len(raw_matches) != len(bio_spans):
            diff = len(raw_matches) - len(bio_spans)
            # Find pairwise overlaps in raw_matches
            overlap_pairs = []
            for i in range(len(raw_matches)):
                for j in range(i + 1, len(raw_matches)):
                    m1 = raw_matches[i]
                    m2 = raw_matches[j]
                    if max(m1["start"], m2["start"]) < min(m1["end"], m2["end"]):
                        overlap_pairs.append((m1, m2))
                        tp = tuple(sorted([m1["tool"], m2["tool"]]))
                        tool_collisions[f"{tp[0]} <-> {tp[1]}"] += 1
                        ap = tuple(sorted([f"{m1['tool']}.{m1['arg']}", f"{m2['tool']}.{m2['arg']}"]))
                        arg_collisions[f"{ap[0]} <-> {ap[1]}"] += 1

            diff_records.append({
                "sample_idx": idx,
                "text": user_text,
                "raw_matches_count": len(raw_matches),
                "bio_spans_count": len(bio_spans),
                "diff": diff,
                "raw_matches": raw_matches,
                "bio_spans": list(bio_spans),
                "overlaps": [
                    {"m1": f"{o[0]['tool']}.{o[0]['arg']} ({o[0]['val']})",
                     "m2": f"{o[1]['tool']}.{o[1]['arg']} ({o[1]['val']})"}
                    for o in overlap_pairs
                ]
            })

    total_diff = sum(r["diff"] for r in diff_records)
    print(f"Total diff samples: {len(diff_records)}")
    print(f"Total dropped spans: {total_diff}")
    print("\nCollisions by Tool Pair:")
    for k, v in tool_collisions.most_common():
        print(f"  {k:50s}: {v}")
    print("\nCollisions by Argument Pair:")
    for k, v in arg_collisions.most_common():
        print(f"  {k:50s}: {v}")

    # Inspect samples where diff occurred without a simple 2-span overlap
    no_overlap_diffs = [r for r in diff_records if len(r["overlaps"]) == 0]
    print(f"\nSamples with diff but no simple pairwise word overlap: {len(no_overlap_diffs)}")
    for r in no_overlap_diffs[:5]:
        print(f"  Sample {r['sample_idx']}: diff={r['diff']}, text={r['text']}")
        print(f"    Raw: {[(m['slot'], m['val']) for m in r['raw_matches']]}")
        print(f"    BIO: {r['bio_spans']}")

    return {
        "total_diff": total_diff,
        "tool_collisions": dict(tool_collisions),
        "arg_collisions": dict(arg_collisions),
        "records": diff_records
    }

if __name__ == "__main__":
    train_diffs = analyze_dataset_diffs("Train", "ARN/data/splits/group_disjoint_train.jsonl")
    val_diffs = analyze_dataset_diffs("Val", "ARN/data/splits/group_disjoint_val.jsonl")
