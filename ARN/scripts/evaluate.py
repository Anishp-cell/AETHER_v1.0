#!/usr/bin/env python3
"""
ARN Evaluation Suite.
Computes multi-label tool routing metrics, token/span slot F1, joint frame accuracy,
and probability calibration (ECE) across test splits and out-of-distribution probes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from arn import (
    ARNConfig,
    AetherRoutingNetwork,
    JointBERT,
    CLSCRFModel,
    FARNoCRFModel,
    ARNDataset,
    TOKENIZER,
    IDX_TO_TAG,
    IDX_TO_TOOL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.evaluate")


def compute_calibration_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over multi-label sigmoid probabilities."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = probs.size

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs > bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(targets[mask])
            ece += (bin_size / total_samples) * np.abs(avg_confidence - avg_accuracy)

    return float(ece)


def extract_spans_from_bio(tags: List[str]) -> Set[Tuple[str, int, int]]:
    """Extract named entity spans (type, start_idx, end_idx) from BIO tag list."""
    spans = set()
    current_type = None
    start_idx = -1

    for idx, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.add((current_type, start_idx, idx))
            current_type = tag[2:]
            start_idx = idx
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type != tag_type:
                if current_type is not None:
                    spans.add((current_type, start_idx, idx))
                current_type = tag_type
                start_idx = idx
        else:  # O or PAD
            if current_type is not None:
                spans.add((current_type, start_idx, idx))
                current_type = None
                start_idx = -1

    if current_type is not None:
        spans.add((current_type, start_idx, len(tags)))

    return spans


def evaluate_dataset(model, dataset, device) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_tool_probs = []
    all_tool_preds = []
    all_tool_targets = []

    gold_spans_all: List[Set[Tuple[str, int, int]]] = []
    pred_spans_all: List[Set[Tuple[str, int, int]]] = []

    joint_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tool_labels = batch["tool_labels"].cpu().numpy()
            tags = batch["tags"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            tool_probs = outputs[0].cpu().numpy()
            tool_preds = outputs[1].cpu().numpy()
            tag_preds = outputs[2]  # List of lists

            for b in range(input_ids.size(0)):
                length = int(attention_mask[b].sum().item())
                gold_tag_ids = tags[b][:length].tolist()
                pred_tag_ids = tag_preds[b][:length]

                gold_tags = [IDX_TO_TAG.get(i, "O") for i in gold_tag_ids]
                pred_tags = [IDX_TO_TAG.get(i, "O") for i in pred_tag_ids]

                g_spans = extract_spans_from_bio(gold_tags)
                p_spans = extract_spans_from_bio(pred_tags)

                gold_spans_all.append(g_spans)
                pred_spans_all.append(p_spans)

                # Check strict exact match: both tool selection and slot spans match
                tool_match = np.array_equal(tool_preds[b], tool_labels[b])
                slot_match = (g_spans == p_spans)
                if tool_match and slot_match:
                    joint_correct += 1
                total_samples += 1

            all_tool_probs.append(tool_probs)
            all_tool_preds.append(tool_preds)
            all_tool_targets.append(tool_labels)

    all_tool_probs = np.vstack(all_tool_probs)
    all_tool_preds = np.vstack(all_tool_preds)
    all_tool_targets = np.vstack(all_tool_targets)

    # 1. Tool Routing Metrics
    tool_exact_matches = np.all(all_tool_preds == all_tool_targets, axis=1)
    tool_em_count = int(np.sum(tool_exact_matches))
    tool_em_acc = float(tool_em_count / max(total_samples, 1))

    # Precision, Recall, F1 (micro & macro)
    tp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 1)))
    fp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 0)))
    fn = int(np.sum((all_tool_preds == 0) & (all_tool_targets == 1)))

    micro_prec = tp / max(tp + fp, 1)
    micro_rec = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-8) if (micro_prec + micro_rec) > 0 else 0.0

    # Per-tool macro & per-tool details
    per_tool_results = {}
    macro_f1s = []
    macro_precs = []
    macro_recs = []

    for k in range(all_tool_targets.shape[1]):
        tool_name = IDX_TO_TOOL.get(k, f"tool_{k}")
        t_tp = int(np.sum((all_tool_preds[:, k] == 1) & (all_tool_targets[:, k] == 1)))
        t_fp = int(np.sum((all_tool_preds[:, k] == 1) & (all_tool_targets[:, k] == 0)))
        t_fn = int(np.sum((all_tool_preds[:, k] == 0) & (all_tool_targets[:, k] == 1)))
        t_support = int(np.sum(all_tool_targets[:, k] == 1))

        p = t_tp / max(t_tp + t_fp, 1) if (t_tp + t_fp) > 0 else 0.0
        r = t_tp / max(t_tp + t_fn, 1) if (t_tp + t_fn) > 0 else 0.0
        f = 2 * p * r / max(p + r, 1e-8) if (p + r) > 0 else 0.0

        macro_precs.append(p)
        macro_recs.append(r)
        macro_f1s.append(f)

        support_status = "sufficient" if t_support >= 10 else "insufficient support"
        per_tool_results[tool_name] = {
            "support": t_support,
            "tp": t_tp,
            "fp": t_fp,
            "fn": t_fn,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "status": support_status,
        }

    macro_prec = float(np.mean(macro_precs))
    macro_rec = float(np.mean(macro_recs))
    macro_f1 = float(np.mean(macro_f1s))

    # 2. Span F1 Metrics
    total_gold_spans = sum(len(s) for s in gold_spans_all)
    total_pred_spans = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))

    span_prec = correct_spans / max(total_pred_spans, 1) if total_pred_spans > 0 else 0.0
    span_rec = correct_spans / max(total_gold_spans, 1) if total_gold_spans > 0 else 0.0
    span_f1 = 2 * span_prec * span_rec / max(span_prec + span_rec, 1e-8) if (span_prec + span_rec) > 0 else 0.0

    # 3. Joint Frame Accuracy
    joint_frame_acc = joint_correct / max(total_samples, 1)

    # 4. Calibration ECE
    ece = compute_calibration_ece(all_tool_probs, all_tool_targets)

    return {
        "num_samples": total_samples,
        "tool_routing": {
            "exact_match_count": tool_em_count,
            "exact_match_total": total_samples,
            "exact_match_accuracy": tool_em_acc,
            "micro": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(micro_prec),
                "recall": float(micro_rec),
                "f1": float(micro_f1),
            },
            "macro": {
                "precision": macro_prec,
                "recall": macro_rec,
                "f1": macro_f1,
            },
            "calibration_ece": ece,
            "per_tool": per_tool_results,
        },
        "slot_tagging": {
            "correct_spans": correct_spans,
            "total_pred_spans": total_pred_spans,
            "total_gold_spans": total_gold_spans,
            "span_precision": float(span_prec),
            "span_recall": float(span_rec),
            "span_f1": float(span_f1),
        },
        "joint_frame": {
            "exact_match_count": joint_correct,
            "exact_match_total": total_samples,
            "exact_match_accuracy": float(joint_frame_acc),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="ARN Model Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data", type=str, required=True, help="Path to test jsonl dataset")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Evaluation dataset not found: %s", data_path)
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=device)

    model_type = ckpt.get("model_type", "arn")
    config_dict = ckpt.get("config", {})
    config = ARNConfig(**{k: v for k, v in config_dict.items() if hasattr(ARNConfig, k)}) if config_dict else ARNConfig()

    logger.info("Loaded checkpoint %s (type=%s) on %s", ckpt_path.name, model_type, device)

    if model_type == "arn":
        model = AetherRoutingNetwork(config)
    elif model_type == "jointbert":
        model = JointBERT(config)
    elif model_type == "cls_crf":
        model = CLSCRFModel(config)
    elif model_type == "far_no_crf":
        model = FARNoCRFModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    dataset = ARNDataset(data_path, TOKENIZER, max_length=config.max_seq_len)
    logger.info("Evaluating on %d samples from %s...", len(dataset), data_path.name)

    results = evaluate_dataset(model, dataset, device)
    results["checkpoint"] = str(ckpt_path)
    results["dataset"] = str(data_path)
    results["model_type"] = model_type

    logger.info("=== Evaluation Results ===")
    tr = results["tool_routing"]
    logger.info("Tool Exact Match Accuracy (Subset Acc): %d of %d (%.2f%%)",
                tr["exact_match_count"], tr["exact_match_total"], 100.0 * tr["exact_match_accuracy"])
    logger.info("Tool Micro Metrics: Prec=%.2f%%, Rec=%.2f%%, F1=%.2f%% (TP=%d, FP=%d, FN=%d)",
                100.0 * tr["micro"]["precision"], 100.0 * tr["micro"]["recall"], 100.0 * tr["micro"]["f1"],
                tr["micro"]["tp"], tr["micro"]["fp"], tr["micro"]["fn"])
    logger.info("Tool Macro Metrics: Prec=%.2f%%, Rec=%.2f%%, F1=%.2f%%",
                100.0 * tr["macro"]["precision"], 100.0 * tr["macro"]["recall"], 100.0 * tr["macro"]["f1"])
    logger.info("Tool Calibration (ECE): %.4f", tr["calibration_ece"])

    st = results["slot_tagging"]
    logger.info("Slot Span Metrics: Prec=%d/%d (%.2f%%), Rec=%d/%d (%.2f%%), F1=%.2f%%",
                st["correct_spans"], max(st["total_pred_spans"], 1), 100.0 * st["span_precision"],
                st["correct_spans"], max(st["total_gold_spans"], 1), 100.0 * st["span_recall"],
                100.0 * st["span_f1"])

    jf = results["joint_frame"]
    logger.info("Joint Exact Match Accuracy: %d of %d (%.2f%%)",
                jf["exact_match_count"], jf["exact_match_total"], 100.0 * jf["exact_match_accuracy"])

    logger.info("--- Per-Tool Breakdown ---")
    logger.info("%-28s | %-8s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %s",
                "Tool Name", "Support", "TP", "FP", "FN", "Prec", "Rec", "F1", "Status")
    logger.info("-" * 105)
    for t_name, m in tr["per_tool"].items():
        status_flag = "[insufficient support]" if m["status"] == "insufficient support" else "[OK]"
        logger.info("%-28s | %-8d | %-6d | %-6d | %-6d | %-7.2f%% | %-7.2f%% | %-7.2f%% | %s",
                    t_name, m["support"], m["tp"], m["fp"], m["fn"],
                    100.0 * m["precision"], 100.0 * m["recall"], 100.0 * m["f1"], status_flag)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved metrics to %s", out_path)


if __name__ == "__main__":
    main()
