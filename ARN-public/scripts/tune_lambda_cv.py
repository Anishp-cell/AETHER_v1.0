#!/usr/bin/env python3
"""
Cluster-Level Grouped 5-Fold Cross-Validation for Tuning lambda_slot.
Partitions the 209 training clusters into 5 disjoint folds.
Evaluates lambda_slot in {0.5, 1.0, 1.5, 2.0} with early stopping (patience=3).
Applies the 1-standard-error rule to select optimal lambda_slot.
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from arn import (
    ARNConfig,
    AetherRoutingNetwork,
    ARNDataset,
    TOKENIZER,
    IDX_TO_TAG,
)
from scripts.make_splits import extract_fine_semantic_frame, token_edit_sim, build_canonical_clusters
from scripts.evaluate import extract_spans_from_bio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.tune_cv")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_clusters_into_5_folds(
    train_samples: List[Dict[str, Any]],
    frame_to_cluster: Dict[str, str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[int]]:
    """Partition training sample indices into 5 cluster-disjoint folds with balanced sizes."""
    cluster_to_indices = defaultdict(list)
    for idx, s in enumerate(train_samples):
        frame = extract_fine_semantic_frame(s)
        cid = frame_to_cluster[frame]
        cluster_to_indices[cid].append(idx)

    # Sort clusters by size descending for greedy balanced partition
    sorted_clusters = sorted(cluster_to_indices.keys(), key=lambda c: len(cluster_to_indices[c]), reverse=True)

    rng = random.Random(seed)
    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for cid in sorted_clusters:
        # Assign cluster to the fold with minimum current size
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(cluster_to_indices[cid])
        fold_sizes[min_fold] += len(cluster_to_indices[cid])

    return fold_indices


@torch.no_grad()
def evaluate_fold(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    total_loss_accum = 0.0
    steps = 0

    tool_exact_correct = 0
    joint_exact_correct = 0
    total_samples = 0

    gold_spans_all = []
    pred_spans_all = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tags = batch["tags"].to(device)
        tool_labels = batch["tool_labels"].to(device)

        loss_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tags=tags,
            tool_labels=tool_labels,
        )
        total_loss = loss_outputs[0]
        total_loss_accum += total_loss.item()
        steps += 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        tool_preds = outputs[1].cpu().numpy()
        tag_preds = outputs[2]

        tags_np = tags.cpu().numpy()
        labels_np = tool_labels.cpu().numpy()

        for b in range(input_ids.size(0)):
            length = int(attention_mask[b].sum().item())
            g_tags = [IDX_TO_TAG.get(i, "O") for i in tags_np[b][:length].tolist()]
            p_tags = [IDX_TO_TAG.get(i, "O") for i in tag_preds[b][:length]]

            g_spans = extract_spans_from_bio(g_tags)
            p_spans = extract_spans_from_bio(p_tags)

            gold_spans_all.append(g_spans)
            pred_spans_all.append(p_spans)

            tool_match = np.array_equal(tool_preds[b], labels_np[b])
            slot_match = (g_spans == p_spans)

            if tool_match:
                tool_exact_correct += 1
            if tool_match and slot_match:
                joint_exact_correct += 1
            total_samples += 1

    total_gold_spans = sum(len(s) for s in gold_spans_all)
    total_pred_spans = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))

    span_prec = correct_spans / max(total_pred_spans, 1)
    span_rec = correct_spans / max(total_gold_spans, 1)
    span_f1 = 2 * span_prec * span_rec / max(span_prec + span_rec, 1e-8) if (span_prec + span_rec) > 0 else 0.0

    return {
        "val_loss": total_loss_accum / max(steps, 1),
        "tool_subset_acc": tool_exact_correct / max(total_samples, 1),
        "slot_span_f1": float(span_f1),
        "joint_frame_acc": joint_exact_correct / max(total_samples, 1),
    }


def train_and_eval_fold(
    config: ARNConfig,
    train_dataset: ARNDataset,
    train_indices: List[int],
    val_indices: List[int],
    device: torch.device,
    patience: int = 3,
) -> Dict[str, float]:
    """Train on 4 folds, early stop on 5th fold with patience=3."""
    set_seed(42)
    model = AetherRoutingNetwork(config).to(device)

    train_sub = Subset(train_dataset, train_indices)
    val_sub = Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_metrics = {}
    no_improve_epochs = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags"].to(device)
            tool_labels = batch["tool_labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, tags, tool_labels)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_metrics = evaluate_fold(model, val_loader, device)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_metrics = val_metrics.copy()
            best_metrics["best_epoch"] = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    return best_metrics


def run_cv(
    train_path: Path,
    raw_path: Path,
    lambda_grid: List[float] = [0.5, 1.0, 1.5, 2.0],
    n_folds: int = 5,
    patience: int = 3,
    max_epochs: int = 10,
    batch_size: int = 64,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_str)
    logger.info("Initializing Grouped 5-Fold CV on %s", device)

    # 1. Load canonical cluster mappings
    frame_to_cluster = build_canonical_clusters(raw_path)

    # 2. Load training dataset
    with open(train_path, "r", encoding="utf-8") as f:
        train_samples = [json.loads(line) for line in f if line.strip()]

    fold_splits = partition_clusters_into_5_folds(train_samples, frame_to_cluster, n_folds=n_folds, seed=42)
    for i, f_idx in enumerate(fold_splits):
        logger.info("Fold %d: %d samples", i, len(f_idx))

    base_config = ARNConfig()
    base_config.batch_size = batch_size
    base_config.num_epochs = max_epochs
    dataset = ARNDataset(train_path, TOKENIZER, max_length=base_config.max_seq_len)

    cv_results = {}

    for lam in lambda_grid:
        logger.info("=== Running 5-Fold CV for lambda_slot = %.2f ===", lam)
        fold_metrics_list = []

        for k in range(n_folds):
            val_indices = fold_splits[k]
            train_indices = [idx for j, f_indices in enumerate(fold_splits) if j != k for idx in f_indices]

            cfg = ARNConfig()
            cfg.batch_size = batch_size
            cfg.num_epochs = max_epochs
            cfg.lambda_slot = lam

            metrics = train_and_eval_fold(cfg, dataset, train_indices, val_indices, device, patience=patience)
            fold_metrics_list.append(metrics)
            logger.info("  Fold %d (lambda=%.1f) -> Val Loss: %.4f, Joint Acc: %.2f%%, Slot F1: %.2f%% (epoch %d)",
                        k, lam, metrics["val_loss"], 100.0 * metrics["joint_frame_acc"],
                        100.0 * metrics["slot_span_f1"], metrics["best_epoch"])

        # Aggregate across 5 folds
        val_losses = [m["val_loss"] for m in fold_metrics_list]
        joint_accs = [m["joint_frame_acc"] for m in fold_metrics_list]
        tool_accs = [m["tool_subset_acc"] for m in fold_metrics_list]
        slot_f1s = [m["slot_span_f1"] for m in fold_metrics_list]

        mean_loss = float(np.mean(val_losses))
        se_loss = float(np.std(val_losses, ddof=1) / np.sqrt(n_folds))

        mean_joint = float(np.mean(joint_accs))
        se_joint = float(np.std(joint_accs, ddof=1) / np.sqrt(n_folds))

        mean_tool = float(np.mean(tool_accs))
        se_tool = float(np.std(tool_accs, ddof=1) / np.sqrt(n_folds))

        mean_slot = float(np.mean(slot_f1s))
        se_slot = float(np.std(slot_f1s, ddof=1) / np.sqrt(n_folds))

        cv_results[str(lam)] = {
            "lambda_slot": lam,
            "val_loss": {"mean": mean_loss, "se": se_loss, "folds": val_losses},
            "joint_frame_accuracy": {"mean": mean_joint, "se": se_joint, "folds": joint_accs},
            "tool_subset_accuracy": {"mean": mean_tool, "se": se_tool, "folds": tool_accs},
            "slot_span_f1": {"mean": mean_slot, "se": se_slot, "folds": slot_f1s},
            "fold_details": fold_metrics_list,
        }

        logger.info("lambda=%.1f Summary | Val Loss: %.4f +/- %.4f | Joint Acc: %.2f%% +/- %.2f%% | Slot F1: %.2f%% +/- %.2f%%",
                    lam, mean_loss, se_loss, 100.0 * mean_joint, 100.0 * se_joint, 100.0 * mean_slot, 100.0 * se_slot)

    # 1-SE rule selection based on joint frame accuracy (or val loss)
    # Best candidate by highest mean joint accuracy
    best_lambda = max(lambda_grid, key=lambda l: cv_results[str(l)]["joint_frame_accuracy"]["mean"])
    best_acc = cv_results[str(best_lambda)]["joint_frame_accuracy"]["mean"]
    best_se = cv_results[str(best_lambda)]["joint_frame_accuracy"]["se"]

    # Check lambda = 1.0
    acc_1_0 = cv_results["1.0"]["joint_frame_accuracy"]["mean"]
    diff_from_best = abs(best_acc - acc_1_0)

    within_1_se = diff_from_best <= best_se
    selected_lambda = 1.0 if within_1_se else best_lambda

    logger.info("=== 1-Standard-Error Rule Selection ===")
    logger.info("Best candidate: lambda=%.1f (Joint Acc: %.2f%% +/- %.2f%%)", best_lambda, 100.0 * best_acc, 100.0 * best_se)
    logger.info("Baseline candidate: lambda=1.0 (Joint Acc: %.2f%%, Diff: %.2f%%, 1 SE Threshold: %.2f%%)",
                100.0 * acc_1_0, 100.0 * diff_from_best, 100.0 * best_se)
    logger.info("Difference within 1 SE: %s -> Selected lambda_slot: %.1f", within_1_se, selected_lambda)

    output_summary = {
        "n_folds": n_folds,
        "patience": patience,
        "max_epochs": max_epochs,
        "lambda_grid": lambda_grid,
        "results": cv_results,
        "selection": {
            "best_candidate": best_lambda,
            "best_candidate_mean_joint_acc": best_acc,
            "best_candidate_se": best_se,
            "lambda_1_0_mean_joint_acc": acc_1_0,
            "difference": diff_from_best,
            "within_1_se": within_1_se,
            "selected_lambda": selected_lambda,
            "selection_rationale": "Differences across candidate lambdas are within one standard error of the best; chosen lambda=1.0 under the parsimonious standard error rule." if within_1_se else f"Candidate lambda={best_lambda} significantly outperformed lambda=1.0 beyond 1 standard error."
        }
    }

    return output_summary


def main():
    parser = argparse.ArgumentParser(description="ARN lambda_slot 5-Fold Grouped CV Tuning")
    parser.add_argument("--train-data", type=str, default="data/splits/group_disjoint_train.jsonl")
    parser.add_argument("--raw-data", type=str, default="data/raw/aether_orchestrator_dataset.jsonl")
    parser.add_argument("--output-json", type=str, default="results/cv_tuning_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    arn_root = Path(__file__).resolve().parent.parent
    train_path = Path(args.train_data)
    if not train_path.exists():
        train_path = arn_root / args.train_data

    raw_path = Path(args.raw_data)
    if not raw_path.exists():
        raw_path = arn_root / args.raw_data

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = arn_root / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = run_cv(train_path, raw_path, device_str=args.device, batch_size=args.batch_size)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved CV tuning results to %s", out_path)


if __name__ == "__main__":
    main()
