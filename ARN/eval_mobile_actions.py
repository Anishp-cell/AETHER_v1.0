import os
import sys
import re
import json
import time
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout/stderr on Windows to avoid CP1252 encoding errors
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from transformers import BertTokenizerFast, logging as hf_logging
hf_logging.set_verbosity_error()
from sklearn.metrics import f1_score, precision_score, recall_score

from mobile_actions_config import (
    ARNMobileConfig, MOBILE_TOOL_TO_IDX, MOBILE_TAG_TO_IDX,
    IDX_TO_MOBILE_TOOL, IDX_TO_MOBILE_TAG, MOBILE_SLOT_TYPES
)
from mobile_actions_dataset import MobileActionsDataset
from mobile_actions_model import ARNMobileModel

# Tool-to-argument schema mapping for deterministic argument assignment
TOOL_ARG_SCHEMA = {
    "create_calendar_event": ["title", "datetime"],
    "send_email": ["to", "subject", "body"],
    "create_contact": ["first_name", "last_name", "phone_number", "email"],
    "show_map": ["query"],
    "open_wifi_settings": [],
    "turn_on_flashlight": [],
    "turn_off_flashlight": []
}

def parse_reference_datetime(sys_content: str):
    """Extract reference datetime from the system message."""
    match = re.search(r'format:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', sys_content)
    if match:
        try:
            return datetime.fromisoformat(match.group(1))
        except Exception:
            pass
    return datetime(2025, 1, 1, 0, 0, 0)

def extract_datetime_span_regex(text: str):
    pattern = re.compile(
        r'(?:(?:on|for|at|this|next|upcoming|coming)\s+)?'
        r'(?:(?:this|next|upcoming|coming)\s+)?'
        r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?'
        r'[\s,]*'
        r'(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
        r'[\s,]*'
        r'(?:at\s+)?'
        r'(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}\s*(?:am|pm)|noon|midnight)?',
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    if matches:
        return max(matches, key=lambda m: len(m.group(0))).group(0).strip()
    return text

def resolve_datetime_span(raw_span: str, ref_dt: datetime, full_text: str):
    """Resolves relative and absolute natural language temporal spans into standard ISO-8601 datetimes."""
    if not ref_dt:
        ref_dt = datetime(2025, 1, 1, 0, 0, 0)
    base_ref = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Clean span
    clean_span = re.sub(r'^(?:on|for|at)\s+', '', raw_span, flags=re.IGNORECASE).strip()
    
    # Try parsing extracted span
    try:
        dt = date_parser.parse(clean_span, fuzzy=True, default=base_ref)
        return dt.replace(second=0, microsecond=0)
    except Exception:
        pass
        
    # Try regex fallback on full utterance
    try:
        extracted = extract_datetime_span_regex(full_text)
        dt = date_parser.parse(extracted, fuzzy=True, default=base_ref)
        return dt.replace(second=0, microsecond=0)
    except Exception:
        pass
        
    # Final fallback: full text fuzzy parse
    try:
        dt = date_parser.parse(full_text, fuzzy=True, default=base_ref)
        return dt.replace(second=0, microsecond=0)
    except Exception:
        return None

def extract_slots_clean(user_text, tokenizer, tag_preds):
    """
    Extracts argument values by mapping subword BIO tags directly to character offsets
    in the raw user text, avoiding tokenization whitespace artifacts.
    """
    tokenized = tokenizer(
        user_text,
        return_offsets_mapping=True,
        max_length=64,
        truncation=True,
        padding="max_length"
    )
    offsets = tokenized["offset_mapping"]

    slots = {}
    current_tag = None
    span_start = None
    span_end = None

    for i, tag_id in enumerate(tag_preds):
        tag = IDX_TO_MOBILE_TAG.get(tag_id, "O")
        start_char, end_char = offsets[i]

        if start_char == end_char or tag == "<PAD>":
            continue

        if tag.startswith("B-"):
            if current_tag is not None and span_start is not None:
                val = user_text[span_start:span_end].strip('\'",. ')
                slots[current_tag] = val
            current_tag = tag.split("-")[1]
            span_start = start_char
            span_end = end_char
        elif tag.startswith("I-"):
            tag_type = tag.split("-")[1]
            if current_tag == tag_type:
                span_end = end_char
        elif tag == "O":
            if current_tag is not None and span_start is not None:
                val = user_text[span_start:span_end].strip('\'",. ')
                slots[current_tag] = val
            current_tag = None
            span_start = None
            span_end = None

    if current_tag is not None and span_start is not None:
        val = user_text[span_start:span_end].strip('\'",. ')
        slots[current_tag] = val

    return slots

def get_tool_text_position(tool_name: str, user_text: str):
    """Determines the mention order of a tool in the user utterance."""
    keywords = {
        "turn_on_flashlight": ["turn on", "flashlight on", "on the flashlight", "torch on", "light on"],
        "turn_off_flashlight": ["turn off", "flashlight off", "off the flashlight", "torch off", "light off"],
        "open_wifi_settings": ["wifi", "wi-fi", "network", "settings"],
        "create_calendar_event": ["reminder", "calendar", "event", "meeting", "schedule", "appointment"],
        "send_email": ["email", "mail", "send", "write to"],
        "create_contact": ["contact", "save", "add", "phone number"],
        "show_map": ["map", "location", "directions", "where is", "route to", "find"]
    }
    user_lower = user_text.lower()
    for kw in keywords.get(tool_name, []):
        pos = user_lower.find(kw)
        if pos != -1:
            return pos
    return 9999

def evaluate_mobile_actions(checkpoint_path: str = None):
    config = ARNMobileConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "arn_mobile_best.pt")

    print(f"\n{'='*80}")
    print(f"  ARN-MOBILE HEAD-TO-HEAD BENCHMARK (google/mobile-actions eval split)")
    print(f"{'='*80}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")

    # 1. Load Model
    model = ARNMobileModel(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Load Evaluation Split (961 rows)
    eval_dataset = MobileActionsDataset("eval")
    print(f"  Total Eval Rows: {len(eval_dataset)}")

    tokenizer = eval_dataset.tokenizer

    # Metrics Accumulators
    strict_exact_matches = 0
    name_only_matches = 0
    relaxed_exact_matches = 0
    
    single_tool_total = 0
    single_tool_strict = 0
    multi_tool_total = 0
    multi_tool_strict = 0

    per_tool_gt = {t: 0 for t in MOBILE_TOOL_TO_IDX}
    per_tool_pred = {t: 0 for t in MOBILE_TOOL_TO_IDX}
    per_tool_correct = {t: 0 for t in MOBILE_TOOL_TO_IDX}

    latencies = []

    for i in range(len(eval_dataset)):
        item = eval_dataset[i]
        raw_row = eval_dataset.ds[i]
        msgs = raw_row["messages"]
        user_text = msgs[1]["content"]
        ref_dt = parse_reference_datetime(msgs[0]["content"])
        gt_tool_calls = msgs[-1].get("tool_calls") or []

        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)

        # Measure latency
        start_time = time.perf_counter()
        with torch.no_grad():
            tool_preds, tag_preds, tool_confs, _ = model(input_ids, attention_mask)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        # 1. Active tools sorted by text mention position
        active_indices = torch.where(tool_preds[0])[0].tolist()
        sorted_indices = sorted(active_indices, key=lambda idx: get_tool_text_position(IDX_TO_MOBILE_TOOL[idx], user_text))
        pred_tool_names = [IDX_TO_MOBILE_TOOL[idx] for idx in sorted_indices]

        # 2. Extracted slot spans directly from user text character offsets
        extracted_slots = extract_slots_clean(user_text, tokenizer, tag_preds[0])

        # 3. Assemble Predicted Tool Calls with Schema Routing
        pred_tool_calls = []
        for tool_name in pred_tool_names:
            expected_args = TOOL_ARG_SCHEMA.get(tool_name, [])
            call_args = {}
            for arg_name in expected_args:
                if arg_name == "datetime":
                    raw_dt_span = extracted_slots.get("datetime", "")
                    resolved_dt = resolve_datetime_span(raw_dt_span, ref_dt, user_text)
                    call_args["datetime"] = resolved_dt
                else:
                    raw_val = extracted_slots.get(arg_name, None)
                    call_args[arg_name] = raw_val

            pred_tool_calls.append({
                "name": tool_name,
                "arguments": call_args
            })

        # 4. Extract Ground Truth Calls
        normalized_gt_calls = []
        for tc in gt_tool_calls:
            fn_name = tc["function"]["name"]
            gt_args = {k: v for k, v in tc["function"]["arguments"].items() if v is not None}
            normalized_gt_calls.append({
                "name": fn_name,
                "arguments": gt_args
            })

        # --- SCORING: Ordered Strict Exact Match ---
        is_strict_match = False
        is_name_match = False
        is_relaxed_match = False

        # Name only check
        gt_names = [c["name"] for c in normalized_gt_calls]
        if pred_tool_names == gt_names:
            is_name_match = True
            name_only_matches += 1

        # Strict Exact Match check
        if len(pred_tool_calls) == len(normalized_gt_calls):
            calls_match_strict = True
            calls_match_relaxed = True

            for p_call, g_call in zip(pred_tool_calls, normalized_gt_calls):
                if p_call["name"] != g_call["name"]:
                    calls_match_strict = False
                    calls_match_relaxed = False
                    break
                
                p_args = p_call["arguments"]
                g_args = g_call["arguments"]

                for k, g_val in g_args.items():
                    p_val = p_args.get(k)
                    if p_val is None:
                        calls_match_strict = False
                        calls_match_relaxed = False
                        break
                    
                    if k == "datetime":
                        # Compare formatted ISO timestamps
                        g_str = g_val.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(g_val, 'strftime') else str(g_val)
                        p_str = p_val.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(p_val, 'strftime') else str(p_val)
                        if p_str != g_str:
                            calls_match_strict = False
                        # Relaxed: date matches even if time differs slightly
                        if hasattr(p_val, 'date') and hasattr(g_val, 'date') and p_val.date() != g_val.date():
                            calls_match_relaxed = False
                    else:
                        # String comparison (lowercase, stripped)
                        p_clean = str(p_val).lower().strip().strip('\'".')
                        g_clean = str(g_val).lower().strip().strip('\'".')
                        if p_clean != g_clean:
                            calls_match_strict = False
                            calls_match_relaxed = False
                            break

            if calls_match_strict:
                is_strict_match = True
                strict_exact_matches += 1
            if calls_match_relaxed:
                is_relaxed_match = True
                relaxed_exact_matches += 1

        # Complexity stratification
        is_multi = len(normalized_gt_calls) > 1
        if is_multi:
            multi_tool_total += 1
            if is_strict_match:
                multi_tool_strict += 1
        else:
            single_tool_total += 1
            if is_strict_match:
                single_tool_strict += 1

        # Per-tool tracking
        for g_name in gt_names:
            per_tool_gt[g_name] = per_tool_gt.get(g_name, 0) + 1
        for p_name in pred_tool_names:
            per_tool_pred[p_name] = per_tool_pred.get(p_name, 0) + 1
        for p_name in pred_tool_names:
            if p_name in gt_names:
                per_tool_correct[p_name] = per_tool_correct.get(p_name, 0) + 1

    total_eval = len(eval_dataset)
    strict_acc = (strict_exact_matches / total_eval) * 100
    name_acc = (name_only_matches / total_eval) * 100
    relaxed_acc = (relaxed_exact_matches / total_eval) * 100
    
    single_acc = (single_tool_strict / single_tool_total * 100) if single_tool_total > 0 else 0
    multi_acc = (multi_tool_strict / multi_tool_total * 100) if multi_tool_total > 0 else 0
    
    mean_latency = float(sum(latencies) / len(latencies))

    print(f"\n{'-'*80}")
    print(f"  BENCHMARK RESULTS (N={total_eval} eval rows)")
    print(f"{'-'*80}")
    print(f"  Ordered Strict Exact Match: {strict_acc:.2f}%  ({strict_exact_matches}/{total_eval})")
    print(f"  Relaxed Exact Match:        {relaxed_acc:.2f}%  ({relaxed_exact_matches}/{total_eval})")
    print(f"  Tool Name-Only Accuracy:    {name_acc:.2f}%  ({name_only_matches}/{total_eval})")
    print(f"  Single-Tool Accuracy (1-call): {single_acc:.2f}%  ({single_tool_strict}/{single_tool_total})")
    print(f"  Multi-Tool Accuracy (2-call):  {multi_acc:.2f}%  ({multi_tool_strict}/{multi_tool_total})")
    print(f"  Mean Latency (CPU):         {mean_latency:.2f} ms")

    print(f"\n  Per-Tool Performance Breakdown:")
    for tool_name in MOBILE_TOOL_TO_IDX:
        gt_cnt = per_tool_gt.get(tool_name, 0)
        pr_cnt = per_tool_pred.get(tool_name, 0)
        corr = per_tool_correct.get(tool_name, 0)
        prec = (corr / pr_cnt * 100) if pr_cnt > 0 else 0
        rec = (corr / gt_cnt * 100) if gt_cnt > 0 else 0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        print(f"    {tool_name:<28s}  F1: {f1:5.1f}%  (P={prec:5.1f}%, R={rec:5.1f}%, n={gt_cnt})")

    # 5. Build Comparative Leaderboard
    leaderboard = [
        {"Model": "LFM2.5 (Liquid AI)", "Params": "230M", "Quant": "f16", "RAM": ">400 MB", "Accuracy": 69.1, "Routing_Acc": 93.0},
        {"Model": "FunctionGemma (Google)", "Params": "270M", "Quant": "f16", "RAM": ">500 MB", "Accuracy": 64.0, "Routing_Acc": 87.3},
        {"Model": "Needle 2 (Cactus)", "Params": "45M", "Quant": "CQ2-bit", "RAM": "28 MB", "Accuracy": 63.7, "Routing_Acc": 98.3},
        {"Model": "Apple FM (Apple)", "Params": "~3B", "Quant": "Native", "RAM": ">1 GB", "Accuracy": 57.6, "Routing_Acc": 94.2},
        {"Model": "ARN-Mobile (Ours)", "Params": "4.4M", "Quant": "FP32 / INT8", "RAM": "5.5 MB", "Accuracy": round(strict_acc, 1), "Routing_Acc": round(name_acc, 1)}
    ]

    print(f"\n{'+'*80}")
    print(f"  HEAD-TO-HEAD COMPARISON: Google Mobile-Actions (961 eval rows)")
    print(f"{'+'*80}")
    print(f"  | {'Model':<25s} | {'Params':>8s} | {'RAM':>9s} | {'Exact Match':>11s} | {'Routing Acc':>11s} |")
    print(f"  |{'-'*27}|{'-'*10}|{'-'*11}|{'-'*13}|{'-'*13}|")
    for row in leaderboard:
        bold = "*" if "ARN" in row["Model"] else " "
        print(f" {bold}| {row['Model']:<25s} | {row['Params']:>8s} | {row['RAM']:>9s} | {row['Accuracy']:10.1f}% | {row['Routing_Acc']:10.1f}% |")
    print(f"  +{'-'*78}+")

    # 6. Save JSON and LaTeX
    results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, "mobile_actions_benchmark_results.json")
    results_payload = {
        "benchmark": "google/mobile-actions (eval split)",
        "eval_rows": total_eval,
        "ordered_strict_exact_match": round(strict_acc, 2),
        "relaxed_exact_match": round(relaxed_acc, 2),
        "name_only_accuracy": round(name_acc, 2),
        "single_tool_accuracy": round(single_acc, 2),
        "multi_tool_accuracy": round(multi_acc, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "leaderboard": leaderboard
    }
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n  [Output] Saved JSON results -> {json_path}")

    # LaTeX Table
    tex_path = os.path.join(results_dir, "mobile_actions_paper_table.tex")
    with open(tex_path, "w") as f:
        f.write("% === Auto-generated Mobile-Actions Comparison Table ===\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Head-to-Head Comparison on Google Mobile-Actions ($N$=961 Eval Split)}\n")
        f.write("\\label{tab:mobile_actions_comparison}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Params} & \\textbf{Quant} & \\textbf{RAM} & \\textbf{Exact Match} & \\textbf{Routing Acc} \\\\\n\\hline\n")
        for row in leaderboard:
            if "ARN" in row["Model"]:
                f.write(f"\\textbf{{{row['Model']}}} & \\textbf{{{row['Params']}}} & \\textbf{{{row['Quant']}}} & \\textbf{{{row['RAM']}}} & \\textbf{{{row['Accuracy']:.1f}\\%}} & \\textbf{{{row['Routing_Acc']:.1f}\\%}} \\\\\n")
            else:
                f.write(f"{row['Model']} & {row['Params']} & {row['Quant']} & {row['RAM']} & {row['Accuracy']:.1f}\\% & {row['Routing_Acc']:.1f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  [Output] Saved LaTeX table  -> {tex_path}")

    return results_payload

if __name__ == "__main__":
    evaluate_mobile_actions()
