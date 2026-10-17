#!/usr/bin/env python3
"""
ARN Evaluation Suite.
Computes multi-label tool routing metrics, token/span slot F1, joint frame accuracy,
and probability calibration (ECE) across test splits and out-of-distribution probes.
Includes:
1. Cluster bootstrap (>= 1,000 resamples over test clusters) for 95% Confidence Intervals.
2. Per-cluster evaluation results for all test clusters.
3. Stratified accuracy tables by nearest-training-skeleton similarity bins.
4. Per-tool performance breakdown with N < 10 marked "insufficient support".
5. Integer counts (k of N) and percentages everywhere.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from scripts.make_splits import extract_fine_semantic_frame, token_edit_sim
from scripts.tune_lambda_cv import build_canonical_clusters

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


def compute_metrics_subset(
    indices: List[int],
    all_tool_preds: np.ndarray,
    all_tool_targets: np.ndarray,
    gold_spans_all: List[Set[Tuple[str, int, int]]],
    pred_spans_all: List[Set[Tuple[str, int, int]]],
) -> Dict[str, Any]:
    """Compute metrics for a subset of samples specified by indices."""
    n_sub = len(indices)
    if n_sub == 0:
        return {
            "tool_em": 0.0,
            "micro_prec": 0.0,
            "micro_rec": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "span_prec": 0.0,
            "span_rec": 0.0,
            "slot_span_f1": 0.0,
            "joint_em": 0.0,
            "tool_em_count": 0,
            "joint_em_count": 0,
            "total_samples": 0,
        }

    sub_preds = all_tool_preds[indices]
    sub_targets = all_tool_targets[indices]

    tool_matches = np.all(sub_preds == sub_targets, axis=1)
    tool_em_count = int(np.sum(tool_matches))
    tool_em = float(tool_em_count / max(n_sub, 1))

    # Micro F1
    tp = int(np.sum((sub_preds == 1) & (sub_targets == 1)))
    fp = int(np.sum((sub_preds == 1) & (sub_targets == 0)))
    fn = int(np.sum((sub_preds == 0) & (sub_targets == 1)))
    micro_prec = tp / max(tp + fp, 1)
    micro_rec = tp / max(tp + fn, 1)
    micro_f1 = float(2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-8)) if (micro_prec + micro_rec) > 0 else 0.0

    # Macro F1
    macro_f1s = []
    for k in range(sub_targets.shape[1]):
        t_tp = int(np.sum((sub_preds[:, k] == 1) & (sub_targets[:, k] == 1)))
        t_fp = int(np.sum((sub_preds[:, k] == 1) & (sub_targets[:, k] == 0)))
        t_fn = int(np.sum((sub_preds[:, k] == 0) & (sub_targets[:, k] == 1)))
        p = t_tp / max(t_tp + t_fp, 1) if (t_tp + t_fp) > 0 else 0.0
        r = t_tp / max(t_tp + t_fn, 1) if (t_tp + t_fn) > 0 else 0.0
        f = 2 * p * r / max(p + r, 1e-8) if (p + r) > 0 else 0.0
        macro_f1s.append(f)
    macro_f1 = float(np.mean(macro_f1s))

    # Span F1
    sub_gold = [gold_spans_all[i] for i in indices]
    sub_pred = [pred_spans_all[i] for i in indices]
    tot_gold = sum(len(s) for s in sub_gold)
    tot_pred = sum(len(s) for s in sub_pred)
    correct_spans = sum(len(g & p) for g, p in zip(sub_gold, sub_pred))

    span_prec = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
    span_rec = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
    slot_span_f1 = float(2 * span_prec * span_rec / max(span_prec + span_rec, 1e-8)) if (span_prec + span_rec) > 0 else 0.0

    # Joint Exact Match
    joint_correct = sum(1 for m, g, p in zip(tool_matches, sub_gold, sub_pred) if m and (g == p))
    joint_em = float(joint_correct / max(n_sub, 1))

    return {
        "tool_em": tool_em,
        "tool_em_count": tool_em_count,
        "micro_prec": float(micro_prec),
        "micro_rec": float(micro_rec),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "span_prec": float(span_prec),
        "span_rec": float(span_rec),
        "slot_span_f1": slot_span_f1,
        "joint_em": joint_em,
        "joint_em_count": joint_correct,
        "total_samples": n_sub,
    }


def run_cluster_bootstrap(
    cluster_to_indices: Dict[str, List[int]],
    all_tool_preds: np.ndarray,
    all_tool_targets: np.ndarray,
    gold_spans_all: List[Set[Tuple[str, int, int]]],
    pred_spans_all: List[Set[Tuple[str, int, int]]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Resample clusters with replacement to compute 95% bootstrap confidence intervals."""
    cluster_keys = sorted(list(cluster_to_indices.keys()))
    n_clusters = len(cluster_keys)
    rng = np.random.RandomState(seed)

    boot_tool_em = []
    boot_micro_f1 = []
    boot_macro_f1 = []
    boot_slot_f1 = []
    boot_joint_em = []

    for _ in range(n_bootstrap):
        sampled_cids = rng.choice(cluster_keys, size=n_clusters, replace=True)
        sample_indices = [idx for cid in sampled_cids for idx in cluster_to_indices[cid]]
        res = compute_metrics_subset(sample_indices, all_tool_preds, all_tool_targets, gold_spans_all, pred_spans_all)
        boot_tool_em.append(res["tool_em"])
        boot_micro_f1.append(res["micro_f1"])
        boot_macro_f1.append(res["macro_f1"])
        boot_slot_f1.append(res["slot_span_f1"])
        boot_joint_em.append(res["joint_em"])

    def ci(arr):
        return {
            "p2_5": float(np.percentile(arr, 2.5)),
            "median": float(np.percentile(arr, 50.0)),
            "p97_5": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_resamples": n_bootstrap,
        "tool_subset_accuracy": ci(boot_tool_em),
        "tool_micro_f1": ci(boot_micro_f1),
        "tool_macro_f1": ci(boot_macro_f1),
        "slot_span_f1": ci(boot_slot_f1),
        "joint_exact_match": ci(boot_joint_em),
    }


def evaluate_dataset(
    model,
    dataset: ARNDataset,
    device: torch.device,
    test_samples: List[Dict[str, Any]] = None,
    train_samples: List[Dict[str, Any]] = None,
    frame_to_cluster: Dict[str, str] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_tool_probs = []
    all_tool_preds = []
    all_tool_targets = []

    gold_spans_all: List[Set[Tuple[str, int, int]]] = []
    pred_spans_all: List[Set[Tuple[str, int, int]]] = []

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

            all_tool_probs.append(tool_probs)
            all_tool_preds.append(tool_preds)
            all_tool_targets.append(tool_labels)

    all_tool_probs = np.vstack(all_tool_probs)
    all_tool_preds = np.vstack(all_tool_preds)
    all_tool_targets = np.vstack(all_tool_targets)
    total_samples = all_tool_targets.shape[0]

    all_indices = list(range(total_samples))
    overall = compute_metrics_subset(all_indices, all_tool_preds, all_tool_targets, gold_spans_all, pred_spans_all)

    # Per-tool macro & per-tool details
    per_tool_results = {}
    macro_precs = []
    macro_recs = []
    macro_f1s = []

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

    # Calibration ECE
    ece = compute_calibration_ece(all_tool_probs, all_tool_targets)

    results: Dict[str, Any] = {
        "num_samples": total_samples,
        "tool_routing": {
            "exact_match_count": overall["tool_em_count"],
            "exact_match_total": total_samples,
            "exact_match_accuracy": overall["tool_em"],
            "micro": {
                "tp": int(np.sum((all_tool_preds == 1) & (all_tool_targets == 1))),
                "fp": int(np.sum((all_tool_preds == 1) & (all_tool_targets == 0))),
                "fn": int(np.sum((all_tool_preds == 0) & (all_tool_targets == 1))),
                "precision": overall["micro_prec"],
                "recall": overall["micro_rec"],
                "f1": overall["micro_f1"],
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
            "correct_spans": sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all)),
            "total_pred_spans": sum(len(s) for s in pred_spans_all),
            "total_gold_spans": sum(len(s) for s in gold_spans_all),
            "span_precision": overall["span_prec"],
            "span_recall": overall["span_rec"],
            "span_f1": overall["slot_span_f1"],
        },
        "joint_frame": {
            "exact_match_count": overall["joint_em_count"],
            "exact_match_total": total_samples,
            "exact_match_accuracy": overall["joint_em"],
        },
    }

    # 1. Cluster Bootstrap & Per-Cluster Results (if cluster mapping available)
    if test_samples and frame_to_cluster:
        cluster_to_indices = defaultdict(list)
        for idx, s in enumerate(test_samples):
            frame = extract_fine_semantic_frame(s)
            cid = frame_to_cluster[frame]
            cluster_to_indices[cid].append(idx)

        logger.info("Computing cluster bootstrap across %d test clusters (%d resamples)...", len(cluster_to_indices), n_bootstrap)
        boot_ci = run_cluster_bootstrap(
            cluster_to_indices, all_tool_preds, all_tool_targets, gold_spans_all, pred_spans_all,
            n_bootstrap=n_bootstrap, seed=42
        )
        results["cluster_bootstrap_95ci"] = boot_ci

        # Per-cluster breakdown
        per_cluster = []
        for cid in sorted(cluster_to_indices.keys()):
            idxs = cluster_to_indices[cid]
            c_res = compute_metrics_subset(idxs, all_tool_preds, all_tool_targets, gold_spans_all, pred_spans_all)
            per_cluster.append({
                "cluster_id": cid,
                "num_samples": len(idxs),
                "tool_subset_accuracy_count": c_res["tool_em_count"],
                "tool_subset_accuracy_total": c_res["total_samples"],
                "tool_subset_accuracy": c_res["tool_em"],
                "slot_span_f1": c_res["slot_span_f1"],
                "joint_exact_match_count": c_res["joint_em_count"],
                "joint_exact_match_total": c_res["total_samples"],
                "joint_exact_match_accuracy": c_res["joint_em"],
            })
        results["per_cluster_results"] = per_cluster

    # 2. Similarity Stratification (if train samples available)
    if test_samples and train_samples:
        train_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in train_samples})
        test_skels = {extract_fine_semantic_frame(s).split(" || ")[1] for s in test_samples}

        skel_to_sim = {skel: max(token_edit_sim(skel, ts) for ts in train_skels) for skel in test_skels}

        bin_names = ["[0.0, 0.4)", "[0.4, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)"]
        bin_skels = {b: set() for b in bin_names}
        for skel, sim in skel_to_sim.items():
            if sim < 0.4: bin_skels["[0.0, 0.4)"].add(skel)
            elif sim < 0.6: bin_skels["[0.4, 0.6)"].add(skel)
            elif sim < 0.7: bin_skels["[0.6, 0.7)"].add(skel)
            else: bin_skels["[0.7, 0.8)"].add(skel)

        bin_indices = {b: [] for b in bin_names}
        for idx, s in enumerate(test_samples):
            skel = extract_fine_semantic_frame(s).split(" || ")[1]
            for b in bin_names:
                if skel in bin_skels[b]:
                    bin_indices[b].append(idx)
                    break

        stratified_results = {}
        for b in bin_names:
            idxs = bin_indices[b]
            res = compute_metrics_subset(idxs, all_tool_preds, all_tool_targets, gold_spans_all, pred_spans_all)
            stratified_results[b] = {
                "num_skeletons": len(bin_skels[b]),
                "num_samples": len(idxs),
                "tool_subset_accuracy_count": res["tool_em_count"],
                "tool_subset_accuracy_total": res["total_samples"],
                "tool_subset_accuracy": res["tool_em"],
                "slot_span_f1": res["slot_span_f1"],
                "joint_exact_match_count": res["joint_em_count"],
                "joint_exact_match_total": res["total_samples"],
                "joint_exact_match_accuracy": res["joint_em"],
            }
        results["similarity_stratification"] = stratified_results

    return results


def main():
    parser = argparse.ArgumentParser(description="ARN Model Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data", type=str, required=True, help="Path to test jsonl dataset")
    parser.add_argument("--raw-data", type=str, default="data/raw/aether_orchestrator_dataset.jsonl",
                        help="Path to full raw dataset for canonical cluster mapping")
    parser.add_argument("--train-data", type=str, default="data/splits/group_disjoint_train.jsonl",
                        help="Path to train split for nearest skeleton similarity stratification")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of cluster bootstrap resamples")
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

    # Load raw JSON lines from test and train splits
    with open(data_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path(args.raw_data)
    frame_to_cluster = build_canonical_clusters(raw_path) if raw_path.exists() else None

    train_path = Path(args.train_data)
    train_samples = None
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_samples = [json.loads(line) for line in f if line.strip()]

    results = evaluate_dataset(
        model=model,
        dataset=dataset,
        device=device,
        test_samples=test_samples,
        train_samples=train_samples,
        frame_to_cluster=frame_to_cluster,
        n_bootstrap=args.n_bootstrap,
    )
    results["checkpoint"] = str(ckpt_path)
    results["dataset"] = str(data_path)
    results["model_type"] = model_type

    logger.info("=== Primary Evaluation Results ===")
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

    if "cluster_bootstrap_95ci" in results:
        ci = results["cluster_bootstrap_95ci"]
        logger.info("=== Cluster Bootstrap (1,000 resamples over %d clusters) 95%% CIs ===", len(results.get("per_cluster_results", [])))
        logger.info("Tool Subset Accuracy: %.2f%% [95%% CI: %.2f%% - %.2f%%]",
                    100.0 * ci["tool_subset_accuracy"]["median"], 100.0 * ci["tool_subset_accuracy"]["p2_5"], 100.0 * ci["tool_subset_accuracy"]["p97_5"])
        logger.info("Tool Micro F1:        %.2f%% [95%% CI: %.2f%% - %.2f%%]",
                    100.0 * ci["tool_micro_f1"]["median"], 100.0 * ci["tool_micro_f1"]["p2_5"], 100.0 * ci["tool_micro_f1"]["p97_5"])
        logger.info("Tool Macro F1:        %.2f%% [95%% CI: %.2f%% - %.2f%%]",
                    100.0 * ci["tool_macro_f1"]["median"], 100.0 * ci["tool_macro_f1"]["p2_5"], 100.0 * ci["tool_macro_f1"]["p97_5"])
        logger.info("Slot Span F1:         %.2f%% [95%% CI: %.2f%% - %.2f%%]",
                    100.0 * ci["slot_span_f1"]["median"], 100.0 * ci["slot_span_f1"]["p2_5"], 100.0 * ci["slot_span_f1"]["p97_5"])
        logger.info("Joint Exact Match:    %.2f%% [95%% CI: %.2f%% - %.2f%%]",
                    100.0 * ci["joint_exact_match"]["median"], 100.0 * ci["joint_exact_match"]["p2_5"], 100.0 * ci["joint_exact_match"]["p97_5"])

    if "similarity_stratification" in results:
        logger.info("=== Test Accuracy Stratified by Nearest Training Skeleton Similarity ===")
        logger.info("%-14s | %-10s | %-10s | %-18s | %-14s | %-18s",
                    "Similarity Bin", "Skeletons", "Samples", "Tool Subset Acc", "Slot Span F1", "Joint Exact Match")
        logger.info("-" * 95)
        for b, m in results["similarity_stratification"].items():
            logger.info("%-14s | %-10d | %-10d | %4d/%-4d (%6.2f%%) | %12.2f%% | %4d/%-4d (%6.2f%%)",
                        b, m["num_skeletons"], m["num_samples"],
                        m["tool_subset_accuracy_count"], m["tool_subset_accuracy_total"], 100.0 * m["tool_subset_accuracy"],
                        100.0 * m["slot_span_f1"],
                        m["joint_exact_match_count"], m["joint_exact_match_total"], 100.0 * m["joint_exact_match_accuracy"])

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
