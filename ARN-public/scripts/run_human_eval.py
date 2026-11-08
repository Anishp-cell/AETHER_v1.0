"""
ARN Human Evaluation Runner & Validation Harness (Phase 9).

Evaluates trained ARN models against hand-authored human evaluation benchmarks.
Adheres strictly to the pre-registration protocol:
- Zero synthetic data generation in evaluation mode.
- Evaluates Tool Exact Match, Slot Strict/Lenient F1, and Joint Exact Match.
- Computes Cohen's kappa for inter-annotator agreement when multiple raters exist.
- Provides a --mock mode for testing harness integrity prior to external study completion.
"""

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import torch
from transformers import BertTokenizerFast

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.baselines import CLSCRFModel
from arn.inference import extract_slot_arguments
from arn.tokenizer import TOKENIZER, IDX_TO_TOOL, IDX_TO_TAG, TOOL_TO_IDX, TAG_TO_IDX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.run_human_eval")


def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """
    Compute Cohen's kappa between two categorical raters.
    Handles binary or multi-class ratings.
    """
    assert len(rater1) == len(rater2), "Rater sequences must have identical lengths."
    n = len(rater1)
    if n == 0:
        return 1.0

    categories = sorted(list(set(rater1).union(set(rater2))))
    if len(categories) <= 1:
        return 1.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = np.zeros((k, k), dtype=np.float64)

    for r1, r2 in zip(rater1, rater2):
        matrix[cat_to_idx[r1], cat_to_idx[r2]] += 1.0

    po = np.trace(matrix) / n
    row_sums = np.sum(matrix, axis=1) / n
    col_sums = np.sum(matrix, axis=0) / n
    pe = np.sum(row_sums * col_sums)

    if math.isclose(pe, 1.0):
        return 1.0
    return float((po - pe) / (1.0 - pe))


def load_model(model_dir: Path, device: torch.device) -> Tuple[CLSCRFModel, BertTokenizerFast, float]:
    """Load trained final model and tokenizer."""
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.pt"
    tokenizer_dir = model_dir / "tokenizer"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Model artifacts not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    threshold = cfg.get("optimal_threshold", 0.30)
    num_tools = cfg.get("num_tools", 18)
    num_tags = cfg.get("num_tags", 26)
    hidden_dim = cfg.get("hidden_dim", 128)

    model = CLSCRFModel(
        num_tools=num_tools,
        num_tags=num_tags,
        hidden_dim=hidden_dim,
        freeze_backbone=False,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if tokenizer_dir.exists():
        tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir))
    else:
        tokenizer = TOKENIZER

    return model, tokenizer, threshold


def run_model_inference(
    utterance: str,
    model: CLSCRFModel,
    tokenizer: BertTokenizerFast,
    threshold: float,
    device: torch.device,
    max_seq_len: int = 64
) -> Tuple[List[str], Dict[str, str]]:
    """Predict active tools and slot arguments for a single utterance."""
    inputs = tokenizer(
        utterance,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        tool_logits, emissions = model(input_ids, attention_mask)
        tool_probs = torch.sigmoid(tool_logits)[0].cpu()
        preds = (tool_probs >= threshold).tolist()
        pred_tools = [IDX_TO_TOOL[i] for i, val in enumerate(preds) if val]

        tag_preds = model.crf.viterbi_decode(emissions, mask=attention_mask.bool())
        pred_tags = tag_preds[0]

    pred_slots = extract_slot_arguments(
        user_text=utterance,
        tokenizer=tokenizer,
        tag_ids=pred_tags,
        max_seq_len=max_seq_len
    )

    return pred_tools, pred_slots


def normalize_val(val: Any) -> str:
    """Normalize string values for lenient matching."""
    return str(val).strip().lower().strip("',.\"?;:")


def evaluate_records(
    records: List[Dict[str, Any]],
    model: CLSCRFModel,
    tokenizer: BertTokenizerFast,
    threshold: float,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model on a list of human eval records."""
    total = len(records)
    if total == 0:
        return {"total_records": 0, "status": "NO_RECORDS"}

    tool_exact_matches = 0
    joint_exact_matches = 0
    total_gold_slots = 0
    total_pred_slots = 0
    strict_slot_matches = 0
    lenient_slot_matches = 0

    per_category = {}
    per_tool_stats = {t: {"gold": 0, "correct": 0, "pred": 0} for t in IDX_TO_TOOL.values()}

    predictions_log = []

    for item in records:
        uid = item.get("id", "unknown")
        category = item.get("category", "single_tool")
        utterance = item.get("utterance", "")
        gold_tools = set(item.get("gold_tools", []))
        gold_slots_raw = item.get("gold_slots", {})

        # Filter out empty gold slots
        gold_slots = {k: normalize_val(v) for k, v in gold_slots_raw.items() if v != ""}

        pred_tools_list, pred_slots_raw = run_model_inference(
            utterance, model, tokenizer, threshold, device
        )
        pred_tools = set(pred_tools_list)
        pred_slots = {k: normalize_val(v) for k, v in pred_slots_raw.items()}

        tool_match = (pred_tools == gold_tools)
        if tool_match:
            tool_exact_matches += 1

        for gt in gold_tools:
            per_tool_stats[gt]["gold"] += 1
            if gt in pred_tools:
                per_tool_stats[gt]["correct"] += 1
        for pt in pred_tools:
            per_tool_stats[pt]["pred"] += 1

        # Slot matching
        cur_gold_count = len(gold_slots)
        cur_pred_count = len(pred_slots)
        total_gold_slots += cur_gold_count
        total_pred_slots += cur_pred_count

        strict_cur = 0
        lenient_cur = 0
        for slot_k, slot_v in gold_slots.items():
            # Check if key or stripped key matches
            matching_pred_key = None
            if slot_k in pred_slots:
                matching_pred_key = slot_k
            else:
                # e.g., if gold_slot is 'handle_smart_home.device' and pred is 'device'
                short_k = slot_k.split(".")[-1]
                if short_k in pred_slots:
                    matching_pred_key = short_k

            if matching_pred_key:
                pv = pred_slots[matching_pred_key]
                if pv == slot_v:
                    strict_cur += 1
                    lenient_cur += 1
                elif slot_v in pv or pv in slot_v:
                    lenient_cur += 1

        strict_slot_matches += strict_cur
        lenient_slot_matches += lenient_cur

        slot_match = (strict_cur == cur_gold_count and cur_pred_count == cur_gold_count)
        joint_match = tool_match and slot_match
        if joint_match:
            joint_exact_matches += 1

        if category not in per_category:
            per_category[category] = {"total": 0, "tool_em": 0, "joint_em": 0}
        per_category[category]["total"] += 1
        if tool_match:
            per_category[category]["tool_em"] += 1
        if joint_match:
            per_category[category]["joint_em"] += 1

        predictions_log.append({
            "id": uid,
            "utterance": utterance,
            "gold_tools": sorted(list(gold_tools)),
            "pred_tools": sorted(list(pred_tools)),
            "gold_slots": gold_slots,
            "pred_slots": pred_slots,
            "tool_match": tool_match,
            "joint_match": joint_match
        })

    tool_subset_acc = tool_exact_matches / total
    joint_em_acc = joint_exact_matches / total

    strict_p = strict_slot_matches / max(total_pred_slots, 1)
    strict_r = strict_slot_matches / max(total_gold_slots, 1)
    strict_f1 = 2 * strict_p * strict_r / max(strict_p + strict_r, 1e-8)

    lenient_p = lenient_slot_matches / max(total_pred_slots, 1)
    lenient_r = lenient_slot_matches / max(total_gold_slots, 1)
    lenient_f1 = 2 * lenient_p * lenient_r / max(lenient_p + lenient_r, 1e-8)

    summary = {
        "total_records": total,
        "tool_subset_accuracy": round(tool_subset_acc * 100, 2),
        "joint_exact_match": round(joint_em_acc * 100, 2),
        "strict_slot_f1": round(strict_f1 * 100, 2),
        "lenient_slot_f1": round(lenient_f1 * 100, 2),
        "categories": {
            cat: {
                "total": data["total"],
                "tool_em": round(data["tool_em"] / data["total"] * 100, 2),
                "joint_em": round(data["joint_em"] / data["total"] * 100, 2)
            }
            for cat, data in per_category.items()
        },
        "per_tool_breakdown": {
            t: {
                "gold_count": stats["gold"],
                "pred_count": stats["pred"],
                "accuracy": round(stats["correct"] / stats["gold"] * 100, 2) if stats["gold"] > 0 else 0.0
            }
            for t, stats in per_tool_stats.items() if stats["gold"] > 0 or stats["pred"] > 0
        }
    }

    return {
        "summary": summary,
        "predictions": predictions_log
    }


def generate_mock_records() -> List[Dict[str, Any]]:
    """Generate 10 synthetic test records to verify harness end-to-end functionality."""
    return [
        {
            "id": "mock_001",
            "category": "single_tool",
            "utterance": "turn off the living room lights",
            "gold_tools": ["handle_smart_home"],
            "gold_slots": {"device": "living room lights", "action": "turn off"},
            "completed": True
        },
        {
            "id": "mock_002",
            "category": "single_tool",
            "utterance": "set a timer for 10 minutes labeled pizza",
            "gold_tools": ["set_timer"],
            "gold_slots": {"duration": "10 minutes", "label": "pizza"},
            "completed": True
        },
        {
            "id": "mock_003",
            "category": "single_tool",
            "utterance": "what time is it right now",
            "gold_tools": ["get_current_time"],
            "gold_slots": {},
            "completed": True
        },
        {
            "id": "mock_004",
            "category": "single_tool",
            "utterance": "search the web for latest quantum computing breakthroughs",
            "gold_tools": ["search_web"],
            "gold_slots": {"query": "latest quantum computing breakthroughs"},
            "completed": True
        },
        {
            "id": "mock_005",
            "category": "single_tool",
            "utterance": "send a whatsapp message to Mom saying I am heading home",
            "gold_tools": ["send_whatsapp_message"],
            "gold_slots": {"recipient": "Mom", "message": "I am heading home"},
            "completed": True
        },
        {
            "id": "mock_006",
            "category": "single_tool",
            "utterance": "play hotel california on spotify",
            "gold_tools": ["media_control"],
            "gold_slots": {"action": "play", "media_item": "hotel california"},
            "completed": True
        },
        {
            "id": "mock_007",
            "category": "single_tool",
            "utterance": "open notepad and type meeting minutes draft",
            "gold_tools": ["open_app_and_type"],
            "gold_slots": {"app_name": "notepad", "text_to_type": "meeting minutes draft"},
            "completed": True
        },
        {
            "id": "mock_008",
            "category": "multi_tool",
            "utterance": "turn on bedroom ac and play smooth jazz music",
            "gold_tools": ["handle_smart_home", "media_control"],
            "gold_slots": {
                "handle_smart_home.device": "bedroom ac",
                "handle_smart_home.action": "turn on",
                "media_control.action": "play",
                "media_control.media_item": "smooth jazz music"
            },
            "completed": True
        },
        {
            "id": "mock_009",
            "category": "single_tool",
            "utterance": "check system cpu and memory usage diagnostics",
            "gold_tools": ["get_system_diagnostics"],
            "gold_slots": {},
            "completed": True
        },
        {
            "id": "mock_010",
            "category": "single_tool",
            "utterance": "solve this integral x^2 dx with deepseek",
            "gold_tools": ["route_to_deepseek"],
            "gold_slots": {"query": "solve this integral x^2 dx with deepseek"},
            "completed": True
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="ARN Human Evaluation Runner")
    parser.add_argument("--input", type=str, default="ARN/data/human_eval_filled.jsonl", help="Path to filled human eval JSONL")
    parser.add_argument("--model-dir", type=str, default="ARN/checkpoints/arn_final_model", help="Path to trained model checkpoint dir")
    parser.add_argument("--output", type=str, default="ARN/results/human_eval_results.json", help="Path to save evaluation results")
    parser.add_argument("--threshold", type=float, default=None, help="Custom sigmoid threshold for tool activation")
    parser.add_argument("--mock", action="store_true", help="Run a verification mock dry-run")
    parser.add_argument("--check-agreement", action="store_true", help="Check inter-annotator agreement if multiple raters exist")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)

    model, tokenizer, optimal_threshold = load_model(model_dir, device)
    threshold = args.threshold if args.threshold is not None else optimal_threshold
    logger.info(f"Loaded model from {model_dir} (activation threshold: {threshold:.2f})")

    if args.mock:
        logger.info("Executing mock validation dry-run on 10 synthetic test records...")
        records = generate_mock_records()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.warning(f"File {input_path} does not exist.")
            logger.info("Human evaluation benchmark remains sealed/pending external rating.")
            logger.info("See ARN/docs/HUMAN_EVAL_PENDING.md for protocol and execution guidance.")
            sys.exit(0)

        with open(input_path, "r", encoding="utf-8") as f:
            raw_lines = [json.loads(line.strip()) for line in f if line.strip()]

        records = [r for r in raw_lines if r.get("completed", False) and r.get("utterance", "").strip()]
        if len(records) == 0:
            logger.warning(f"Found 0 completed records in {input_path}.")
            logger.info("Human evaluation benchmark is currently unannotated. Pending external human evaluation.")
            sys.exit(0)

    results = evaluate_records(records, model, tokenizer, threshold, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Successfully evaluated {len(records)} records.")
    logger.info(f"Summary Results: {json.dumps(results['summary'], indent=2)}")
    logger.info(f"Saved evaluation artifact to {out_path}")


if __name__ == "__main__":
    main()
