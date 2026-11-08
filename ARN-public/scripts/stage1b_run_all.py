"""
Stage 1b Master Diagnostic and Evaluation Suite (High-Performance GPU Accelerated).

Implements all required components for Stage 1b:
1. Part A: Tune Learning Rate and Decision Threshold via Grouped 5-Fold CV (patience 10, max_epochs 100).
2. Part B: Epoch-by-Epoch Training & Plotting (loss, tool subset acc, span F1) with AdamW, grad clipping 1.0.
3. Part C: 20 Validation examples with token-level gold vs pred tags & slot span counts.
4. Part D: Calibration reliability on positive predictions only (drop all-label ECE).
5. Part E: Main Generalization Estimate via Repeated Grouped CV over all 253 clusters (5 folds x 3 shuffles).
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Offline mode to eliminate HuggingFace HTTP round-trips
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.stage1b")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_spans_from_bio(tags: List[str]) -> Set[Tuple[str, int, int]]:
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
        else:
            if current_type is not None:
                spans.add((current_type, start_idx, idx))
                current_type = None
                start_idx = -1
    if current_type is not None:
        spans.add((current_type, start_idx, len(tags)))
    return spans


def partition_clusters(
    samples: List[Dict[str, Any]],
    frame_to_cluster: Dict[str, str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[int]]:
    """Partition sample indices into n_folds cluster-disjoint folds with balanced sizes."""
    cluster_to_indices = defaultdict(list)
    for idx, s in enumerate(samples):
        frame = extract_fine_semantic_frame(s)
        cid = frame_to_cluster[frame]
        cluster_to_indices[cid].append(idx)

    sorted_cids = sorted(cluster_to_indices.keys(), key=lambda c: len(cluster_to_indices[c]), reverse=True)
    rng = random.Random(seed)
    rng.shuffle(sorted_cids)
    sorted_cids.sort(key=lambda c: len(cluster_to_indices[c]), reverse=True)

    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for cid in sorted_cids:
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(cluster_to_indices[cid])
        fold_sizes[min_fold] += len(cluster_to_indices[cid])

    return fold_indices


@dataclass
class GPUData:
    h: torch.Tensor           # (N, max_seq_len, 128)
    mask: torch.Tensor        # (N, max_seq_len)
    tags: torch.Tensor        # (N, max_seq_len)
    tool_labels: torch.Tensor # (N, 18)
    samples: List[Dict[str, Any]]
    raw_samples: List[Dict[str, Any]]
    input_ids: torch.Tensor   # (N, max_seq_len) for token viewing


def prepare_gpu_data(
    dataset: ARNDataset,
    raw_samples: List[Dict[str, Any]],
    model: AetherRoutingNetwork,
    device: torch.device
) -> GPUData:
    """Precompute frozen BERT trunk representations once and store contiguous tensors directly in GPU memory."""
    model.encoder.eval()
    all_ids = torch.stack([x["input_ids"] for x in dataset.encoded]).to(device)
    all_mask = torch.stack([x["attention_mask"] for x in dataset.encoded]).to(device)
    all_tags = torch.stack([x["tags"] for x in dataset.encoded]).to(device)
    all_tools = torch.stack([x["tool_labels"] for x in dataset.encoded]).to(device)

    all_h_list = []
    chunk_size = 512
    with torch.no_grad():
        for i in range(0, len(all_ids), chunk_size):
            chunk_h = model.encoder(
                input_ids=all_ids[i:i + chunk_size],
                attention_mask=all_mask[i:i + chunk_size]
            ).last_hidden_state
            all_h_list.append(chunk_h)
    all_h = torch.cat(all_h_list, dim=0)

    return GPUData(
        h=all_h,
        mask=all_mask,
        tags=all_tags,
        tool_labels=all_tools,
        samples=dataset.samples,
        raw_samples=raw_samples,
        input_ids=all_ids,
    )


def evaluate_model_gpu(
    model: nn.Module,
    gpu_data: GPUData,
    indices: List[int],
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> Dict[str, Any]:
    model.eval()
    idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
    n_samples = len(indices)

    total_loss_accum = 0.0
    tool_loss_accum = 0.0
    slot_loss_accum = 0.0
    steps = 0

    all_tool_probs = []
    all_tool_targets = []
    gold_spans_all = []
    pred_spans_all = []
    gold_tags_all = []
    pred_tags_all = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_idx = idx_tensor[i:i + batch_size]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[b_idx, :max_l]
            tools_b = gpu_data.tool_labels[b_idx]

            tool_logits, _ = model.far(h_b, m_b)
            emissions = model.slot_tagger(h_b)

            loss_out = model.loss_fn(
                tool_logits=tool_logits,
                tool_labels=tools_b,
                crf=model.crf,
                emissions=emissions,
                tags=t_b,
                attention_mask=m_b,
            )
            total_loss_accum += loss_out[0].item()
            tool_loss_accum += loss_out[1].item()
            slot_loss_accum += loss_out[2].item()
            steps += 1

            probs = torch.sigmoid(tool_logits).cpu().numpy()
            tag_preds = model.crf.viterbi_decode(emissions, m_b)

            labels_np = tools_b.cpu().numpy()
            tags_np = t_b.cpu().numpy()
            mask_np = m_b.cpu().numpy()

            for b in range(len(b_idx)):
                length = int(mask_np[b].sum())
                g_tag_ids = tags_np[b][:length].tolist()
                p_tag_ids = tag_preds[b][:length]

                g_tags = [IDX_TO_TAG.get(idx, "O") for idx in g_tag_ids]
                p_tags = [IDX_TO_TAG.get(idx, "O") for idx in p_tag_ids]

                gold_tags_all.append(g_tags)
                pred_tags_all.append(p_tags)
                gold_spans_all.append(extract_spans_from_bio(g_tags))
                pred_spans_all.append(extract_spans_from_bio(p_tags))

            all_tool_probs.append(probs)
            all_tool_targets.append(labels_np)

    all_tool_probs = np.vstack(all_tool_probs)
    all_tool_targets = np.vstack(all_tool_targets)
    all_tool_preds = (all_tool_probs >= threshold).astype(int)

    tool_matches = np.all(all_tool_preds == all_tool_targets, axis=1)
    tool_em_count = int(np.sum(tool_matches))
    tool_em = tool_em_count / max(n_samples, 1)

    # Tool Micro Precision, Recall, F1
    tp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 1)))
    fp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 0)))
    fn = int(np.sum((all_tool_preds == 0) & (all_tool_targets == 1)))
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = (2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)) if (micro_p + micro_r) > 0 else 0.0

    # Tool Macro F1
    macro_f1s = []
    for k in range(all_tool_targets.shape[1]):
        t_tp = int(np.sum((all_tool_preds[:, k] == 1) & (all_tool_targets[:, k] == 1)))
        t_fp = int(np.sum((all_tool_preds[:, k] == 1) & (all_tool_targets[:, k] == 0)))
        t_fn = int(np.sum((all_tool_preds[:, k] == 0) & (all_tool_targets[:, k] == 1)))
        p = t_tp / max(t_tp + t_fp, 1) if (t_tp + t_fp) > 0 else 0.0
        r = t_tp / max(t_tp + t_fn, 1) if (t_tp + t_fn) > 0 else 0.0
        f = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0
        macro_f1s.append(f)
    macro_f1 = float(np.mean(macro_f1s))

    # Slot Span F1
    tot_gold = sum(len(s) for s in gold_spans_all)
    tot_pred = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))
    span_p = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
    span_r = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
    span_f1 = (2 * span_p * span_r / max(span_p + span_r, 1e-8)) if (span_p + span_r) > 0 else 0.0

    # Joint Exact Match
    joint_matches = sum(
        1 for i in range(n_samples)
        if tool_matches[i] and (gold_spans_all[i] == pred_spans_all[i])
    )
    joint_em = joint_matches / max(n_samples, 1)

    return {
        "n_samples": n_samples,
        "val_total_loss": total_loss_accum / max(steps, 1),
        "val_tool_loss": tool_loss_accum / max(steps, 1),
        "val_slot_loss": slot_loss_accum / max(steps, 1),
        "tool_subset_acc_count": tool_em_count,
        "tool_subset_acc": tool_em,
        "tool_micro_p": micro_p,
        "tool_micro_r": micro_r,
        "tool_micro_f1": micro_f1,
        "tool_macro_f1": macro_f1,
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "span_correct": correct_spans,
        "span_gold_total": tot_gold,
        "span_pred_total": tot_pred,
        "joint_em_count": joint_matches,
        "joint_em": joint_em,
        "probs": all_tool_probs,
        "targets": all_tool_targets,
        "gold_tags": gold_tags_all,
        "pred_tags": pred_tags_all,
        "gold_spans": gold_spans_all,
        "pred_spans": pred_spans_all,
    }


@torch.no_grad()
def evaluate_val_loss_gpu(
    model: nn.Module,
    gpu_data: GPUData,
    indices_tensor: torch.Tensor,
    batch_size: int = 64
) -> float:
    model.eval()
    total_loss_accum = 0.0
    steps = 0
    n = len(indices_tensor)
    for i in range(0, n, batch_size):
        b_idx = indices_tensor[i:i + batch_size]
        m_b = gpu_data.mask[b_idx]
        max_l = int(m_b.sum(dim=1).max().item())
        h_b = gpu_data.h[b_idx, :max_l]
        m_b = m_b[:, :max_l]
        t_b = gpu_data.tags[b_idx, :max_l]
        tools_b = gpu_data.tool_labels[b_idx]

        tool_logits, _ = model.far(h_b, m_b)
        emissions = model.slot_tagger(h_b)

        loss_out = model.loss_fn(
            tool_logits=tool_logits,
            tool_labels=tools_b,
            crf=model.crf,
            emissions=emissions,
            tags=t_b,
            attention_mask=m_b,
        )
        total_loss_accum += loss_out[0].item()
        steps += 1
    return total_loss_accum / max(steps, 1)


def train_fold_gpu(
    gpu_data: GPUData,
    train_indices: List[int],
    val_indices: List[int],
    device: torch.device,
    lr: float = 3e-4,
    lambda_slot: float = 1.0,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    verbose_tag: str = "",
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(seed)
    cfg = ARNConfig(learning_rate=lr, lambda_slot=lambda_slot, batch_size=batch_size, epochs=max_epochs)
    model = AetherRoutingNetwork(cfg).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    tr_idx_tensor = torch.tensor(train_indices, device=device, dtype=torch.long)
    val_idx_tensor = torch.tensor(val_indices, device=device, dtype=torch.long)
    n_train = len(tr_idx_tensor)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    epoch_history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        perm = tr_idx_tensor[torch.randperm(n_train, device=device)]

        for b in range(n_batches):
            idx = perm[b * batch_size : min((b + 1) * batch_size, n_train)]
            m_b = gpu_data.mask[idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[idx, :max_l]
            tools_b = gpu_data.tool_labels[idx]

            optimizer.zero_grad()
            tool_logits, _ = model.far(h_b, m_b)
            emissions = model.slot_tagger(h_b)

            loss_out = model.loss_fn(
                tool_logits=tool_logits,
                tool_labels=tools_b,
                crf=model.crf,
                emissions=emissions,
                tags=t_b,
                attention_mask=m_b,
            )
            loss = loss_out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        val_loss = evaluate_val_loss_gpu(model, gpu_data, val_idx_tensor, batch_size=64)
        tr_loss_avg = train_loss_sum / n_batches

        epoch_record = {
            "epoch": epoch,
            "train_loss": tr_loss_avg,
            "val_loss": val_loss,
        }
        epoch_history.append(epoch_record)

        if val_loss < (best_val_loss - 1e-4):
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate_model_gpu(model, gpu_data, val_indices, device, threshold=0.5, batch_size=64)
    return model, final_val, epoch_history


# ═════════════════════════════════════════════════════════════════════════════
# PART A: Tune Learning Rate and Decision Threshold via Grouped 5-Fold CV
# ═════════════════════════════════════════════════════════════════════════════
def run_part_a_tuning(train_gpu_data: GPUData, frame_to_cluster: Dict[str, str], device: torch.device):
    print("\n" + "=" * 80, flush=True)
    print("=== PART A: Grouped 5-Fold CV for Tuning Learning Rate & Threshold ===", flush=True)
    print("=" * 80, flush=True)
    candidate_lrs = [1e-4, 3e-4, 5e-4]
    candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    folds = partition_clusters(train_gpu_data.raw_samples, frame_to_cluster, n_folds=5, seed=42)
    lr_results = {}

    for lr in candidate_lrs:
        print(f"\n>>> Tuning LR = {lr} across 5 grouped folds...", flush=True)
        fold_metrics = []
        oof_probs = []
        oof_targets = []

        for fold_idx in range(5):
            val_idx = folds[fold_idx]
            train_idx = [i for f in range(5) if f != fold_idx for i in folds[f]]

            model, val_res, _ = train_fold_gpu(
                train_gpu_data, train_idx, val_idx, device,
                lr=lr, lambda_slot=1.0, max_epochs=100, patience=10, batch_size=32, seed=42 + fold_idx,
                verbose_tag=f"LR={lr} Fold={fold_idx+1}"
            )
            fold_metrics.append({
                "fold": fold_idx + 1,
                "n_samples": val_res["n_samples"],
                "val_loss": val_res["val_total_loss"],
                "tool_subset_acc": val_res["tool_subset_acc"],
                "tool_subset_acc_count": val_res["tool_subset_acc_count"],
                "tool_micro_f1": val_res["tool_micro_f1"],
                "span_f1": val_res["span_f1"],
                "span_correct": val_res["span_correct"],
                "span_gold_total": val_res["span_gold_total"],
                "joint_em": val_res["joint_em"],
                "joint_em_count": val_res["joint_em_count"],
            })
            oof_probs.append(val_res["probs"])
            oof_targets.append(val_res["targets"])
            print(
                f"    Fold {fold_idx+1}/5 done: Val Loss={val_res['val_total_loss']:.4f}, "
                f"Tool Acc={val_res['tool_subset_acc']*100:.2f}% ({val_res['tool_subset_acc_count']}/{val_res['n_samples']}), "
                f"Span F1={val_res['span_f1']*100:.2f}%, "
                f"Joint EM={val_res['joint_em']*100:.2f}% ({val_res['joint_em_count']}/{val_res['n_samples']})",
                flush=True
            )

        all_oof_probs = np.vstack(oof_probs)
        all_oof_targets = np.vstack(oof_targets)

        # Threshold sweep on OOF predictions
        thresh_perf = {}
        for tau in candidate_thresholds:
            preds = (all_oof_probs >= tau).astype(int)
            tool_acc = float(np.all(preds == all_oof_targets, axis=1).mean())
            tp = int(np.sum((preds == 1) & (all_oof_targets == 1)))
            fp = int(np.sum((preds == 1) & (all_oof_targets == 0)))
            fn = int(np.sum((preds == 0) & (all_oof_targets == 1)))
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0
            thresh_perf[f"{tau:.2f}"] = {"tool_subset_acc": tool_acc, "micro_f1": f1}

        mean_loss = float(np.mean([m["val_loss"] for m in fold_metrics]))
        se_loss = float(np.std([m["val_loss"] for m in fold_metrics], ddof=1) / np.sqrt(5))
        mean_tool_acc = float(np.mean([m["tool_subset_acc"] for m in fold_metrics]))
        se_tool_acc = float(np.std([m["tool_subset_acc"] for m in fold_metrics], ddof=1) / np.sqrt(5))
        mean_span_f1 = float(np.mean([m["span_f1"] for m in fold_metrics]))
        se_span_f1 = float(np.std([m["span_f1"] for m in fold_metrics], ddof=1) / np.sqrt(5))
        mean_joint_em = float(np.mean([m["joint_em"] for m in fold_metrics]))
        se_joint_em = float(np.std([m["joint_em"] for m in fold_metrics], ddof=1) / np.sqrt(5))

        lr_results[str(lr)] = {
            "mean_val_loss": mean_loss,
            "se_val_loss": se_loss,
            "mean_tool_subset_acc": mean_tool_acc,
            "se_tool_subset_acc": se_tool_acc,
            "mean_span_f1": mean_span_f1,
            "se_span_f1": se_span_f1,
            "mean_joint_em": mean_joint_em,
            "se_joint_em": se_joint_em,
            "fold_metrics": fold_metrics,
            "threshold_sweep": thresh_perf,
        }
        logger.info(
            f"LR {lr} -> Loss: {mean_loss:.4f}±{se_loss:.4f}, "
            f"Tool Acc: {mean_tool_acc*100:.2f}±{se_tool_acc*100:.2f}%, "
            f"Span F1: {mean_span_f1*100:.2f}±{se_span_f1*100:.2f}%, "
            f"Joint EM: {mean_joint_em*100:.2f}±{se_joint_em*100:.2f}%"
        )

    # 1-SE rule on validation loss
    best_lr_by_loss = min(lr_results.keys(), key=lambda k: lr_results[k]["mean_val_loss"])
    best_loss = lr_results[best_lr_by_loss]["mean_val_loss"]
    best_loss_se = lr_results[best_lr_by_loss]["se_val_loss"]
    loss_threshold = best_loss + best_loss_se

    valid_lrs = [lr for lr, d in lr_results.items() if d["mean_val_loss"] <= loss_threshold]
    selected_lr = float(best_lr_by_loss) if "0.0003" not in valid_lrs else 0.0003

    t_sweep = lr_results[str(selected_lr)]["threshold_sweep"]
    best_tau = float(max(t_sweep.keys(), key=lambda t: t_sweep[t]["tool_subset_acc"]))

    tuning_summary = {
        "candidate_lrs": candidate_lrs,
        "selected_lr": selected_lr,
        "best_tau": best_tau,
        "lr_results": lr_results,
    }
    with open("ARN/results/stage1b_cv_lr_threshold_tuning.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)

    logger.info(f"Selected LR: {selected_lr}, Selected Decision Threshold: {best_tau}")
    return selected_lr, best_tau, tuning_summary


# ═════════════════════════════════════════════════════════════════════════════
# PART B: Epoch-by-Epoch Training & Plotting (Item 2)
# ═════════════════════════════════════════════════════════════════════════════
def run_part_b_training_and_plotting(
    train_gpu_data: GPUData,
    val_gpu_data: GPUData,
    device: torch.device,
    lr: float = 3e-4,
    threshold: float = 0.5,
):
    logger.info("=== PART B: Full Training with Per-Epoch Logging and Plotting ===")
    set_seed(42)
    cfg = ARNConfig(learning_rate=lr, lambda_slot=1.0, batch_size=32, epochs=100)
    model = AetherRoutingNetwork(cfg).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    n_train = len(train_gpu_data.h)
    batch_size = 32
    n_batches = (n_train + batch_size - 1) // batch_size
    tr_indices = list(range(n_train))
    val_indices = list(range(len(val_gpu_data.h)))

    epoch_records = []
    best_val_loss = float("inf")
    patience = 10
    patience_cnt = 0
    best_state = None

    for epoch in range(1, 101):
        model.train()
        train_loss_sum = 0.0
        train_tool_sum = 0.0
        train_slot_sum = 0.0

        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size : min((b + 1) * batch_size, n_train)]
            m_b = train_gpu_data.mask[idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = train_gpu_data.h[idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = train_gpu_data.tags[idx, :max_l]
            tools_b = train_gpu_data.tool_labels[idx]

            optimizer.zero_grad()
            tool_logits, _ = model.far(h_b, m_b)
            emissions = model.slot_tagger(h_b)

            loss_out = model.loss_fn(
                tool_logits=tool_logits,
                tool_labels=tools_b,
                crf=model.crf,
                emissions=emissions,
                tags=t_b,
                attention_mask=m_b,
            )
            loss = loss_out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_tool_sum += loss_out[1].item()
            train_slot_sum += loss_out[2].item()

        # Evaluate on validation split
        val_eval = evaluate_model_gpu(model, val_gpu_data, val_indices, device, threshold=threshold, batch_size=64)
        # Evaluate on training split subset of 500 for fast tracking
        train_sub = tr_indices[:500]
        train_eval = evaluate_model_gpu(model, train_gpu_data, train_sub, device, threshold=threshold, batch_size=64)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_batches,
            "train_tool_loss": train_tool_sum / n_batches,
            "train_slot_loss": train_slot_sum / n_batches,
            "train_tool_acc": train_eval["tool_subset_acc"],
            "train_span_f1": train_eval["span_f1"],
            "train_joint_em": train_eval["joint_em"],
            "val_loss": val_eval["val_total_loss"],
            "val_tool_loss": val_eval["val_tool_loss"],
            "val_slot_loss": val_eval["val_slot_loss"],
            "val_tool_acc": val_eval["tool_subset_acc"],
            "val_tool_acc_count": val_eval["tool_subset_acc_count"],
            "val_span_f1": val_eval["span_f1"],
            "val_joint_em": val_eval["joint_em"],
            "val_joint_em_count": val_eval["joint_em_count"],
        }
        epoch_records.append(rec)
        logger.info(
            f"Epoch {epoch:2d}/100 | Train Loss: {rec['train_loss']:.4f} | Val Loss: {rec['val_loss']:.4f} | "
            f"Val Tool Acc: {rec['val_tool_acc']*100:.2f}% ({rec['val_tool_acc_count']}/{len(val_indices)}) | "
            f"Val Span F1: {rec['val_span_f1']*100:.2f}% | Val Joint EM: {rec['val_joint_em']*100:.2f}%"
        )

        if rec["val_loss"] < (best_val_loss - 1e-4):
            best_val_loss = rec["val_loss"]
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path = Path("ARN/results/checkpoints/arn/stage1b_best_model.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "epoch": epoch,
            "lr": lr,
            "threshold": threshold,
            "val_loss": best_val_loss,
            "config": cfg.to_dict(),
        }, ckpt_path)
        logger.info(f"Saved stage1b best model to {ckpt_path}")

    with open("ARN/results/stage1b_training_history.json", "w") as f:
        json.dump(epoch_records, f, indent=2)

    # Plotting
    epochs_range = [r["epoch"] for r in epoch_records]
    train_losses = [r["train_loss"] for r in epoch_records]
    val_losses = [r["val_loss"] for r in epoch_records]
    train_tool_accs = [r["train_tool_acc"] * 100 for r in epoch_records]
    val_tool_accs = [r["val_tool_acc"] * 100 for r in epoch_records]
    train_span_f1s = [r["train_span_f1"] * 100 for r in epoch_records]
    val_span_f1s = [r["val_span_f1"] * 100 for r in epoch_records]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # 1. Loss
    axes[0].plot(epochs_range, train_losses, label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs_range, val_losses, label="Val Loss", color="#d62728", lw=2, linestyle="--")
    axes[0].set_title("(a) Multi-Task Objective Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Tool Subset Accuracy
    axes[1].plot(epochs_range, train_tool_accs, label="Train Tool Acc", color="#2ca02c", lw=2)
    axes[1].plot(epochs_range, val_tool_accs, label="Val Tool Acc", color="#ff7f0e", lw=2, linestyle="--")
    axes[1].set_title("(b) Tool Subset Exact Match (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Argument Span F1
    axes[2].plot(epochs_range, train_span_f1s, label="Train Span F1", color="#9467bd", lw=2)
    axes[2].plot(epochs_range, val_span_f1s, label="Val Span F1", color="#8c564b", lw=2, linestyle="--")
    axes[2].set_title("(c) Argument-Slot Span F1 (%)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Span F1 (%)", fontsize=11)
    axes[2].legend(loc="lower right", frameon=True)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_dir = Path("ARN/docs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "stage1b_training_curves.png"
    svg_path = fig_dir / "stage1b_training_curves.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    logger.info(f"Saved training curves to {png_path} and {svg_path}")

    return model, epoch_records


# ═════════════════════════════════════════════════════════════════════════════
# PART C: Validation Examples & Span Counts (Item 5)
# ═════════════════════════════════════════════════════════════════════════════
def run_part_c_val_examples_and_spans(
    model: nn.Module,
    val_gpu_data: GPUData,
    device: torch.device,
    threshold: float = 0.5,
):
    logger.info("=== PART C: 20 Validation Examples & Slot Span Breakdown ===")
    val_indices = list(range(len(val_gpu_data.h)))
    eval_res = evaluate_model_gpu(model, val_gpu_data, val_indices, device, threshold=threshold, batch_size=64)

    rng = random.Random(42)
    example_indices = sorted(rng.sample(range(len(val_gpu_data.h)), 20))
    examples_output = []

    print("\n" + "=" * 90)
    print("20 REPRESENTATIVE VALIDATION EXAMPLES (Gold vs Predicted Tags and Tools)")
    print("=" * 90)

    for i, idx in enumerate(example_indices):
        raw_s = val_gpu_data.samples[idx]
        text = raw_s["text"]
        g_tools = sorted(raw_s["active_tools"])
        p_probs = eval_res["probs"][idx]
        p_tools = sorted([IDX_TO_TOOL[k] for k in range(len(p_probs)) if p_probs[k] >= threshold])

        g_tags = eval_res["gold_tags"][idx]
        p_tags = eval_res["pred_tags"][idx]

        g_spans = sorted(list(eval_res["gold_spans"][idx]))
        p_spans = sorted(list(eval_res["pred_spans"][idx]))

        print(f"\n[Validation Sample #{idx} | Example {i+1}/20]")
        print(f"  Query      : \"{text}\"")
        print(f"  Gold Tools : {g_tools}")
        print(f"  Pred Tools : {p_tools} {'[MATCH]' if g_tools == p_tools else '[MISMATCH]'}")
        print(f"  Gold Spans : {g_spans}")
        print(f"  Pred Spans : {p_spans} {'[MATCH]' if g_spans == p_spans else '[MISMATCH]'}")

        toks = TOKENIZER.convert_ids_to_tokens(val_gpu_data.input_ids[idx].cpu().tolist())[:len(g_tags)]
        tag_table_lines = []
        for t, gt, pt in zip(toks, g_tags, p_tags):
            match_mark = " " if gt == pt else "*"
            tag_table_lines.append(f"    {t:15s} | Gold: {gt:12s} | Pred: {pt:12s} {match_mark}")
        print("  Tokens & Tags:")
        print("\n".join(tag_table_lines[:15]))
        if len(tag_table_lines) > 15:
            print(f"    ... and {len(tag_table_lines) - 15} more tokens.")

        examples_output.append({
            "sample_idx": idx,
            "text": text,
            "gold_tools": g_tools,
            "pred_tools": p_tools,
            "gold_spans": [list(s) for s in g_spans],
            "pred_spans": [list(s) for s in p_spans],
            "tokens": toks,
            "gold_tags": g_tags,
            "pred_tags": p_tags,
        })

    # Aggregate span counts per slot type across all validation samples
    span_type_counts = defaultdict(lambda: {"gold": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0})
    for g_set, p_set in zip(eval_res["gold_spans"], eval_res["pred_spans"]):
        for s_type, s_start, s_end in g_set:
            span_type_counts[s_type]["gold"] += 1
            if (s_type, s_start, s_end) in p_set:
                span_type_counts[s_type]["tp"] += 1
            else:
                span_type_counts[s_type]["fn"] += 1

        for s_type, s_start, s_end in p_set:
            span_type_counts[s_type]["pred"] += 1
            if (s_type, s_start, s_end) not in g_set:
                span_type_counts[s_type]["fp"] += 1

    print("\n" + "=" * 90)
    print("VALIDATION SET SLOT SPAN COUNTS & BREAKDOWN (Gold vs Predicted)")
    print("=" * 90)
    print(f"{'Slot Type':18s} | {'Gold':6s} | {'Pred':6s} | {'TP':5s} | {'FP':5s} | {'FN':5s} | {'Prec (%)':9s} | {'Rec (%)':9s} | {'F1 (%)':9s}")
    print("-" * 90)

    span_breakdown_dict = {}
    for stype in sorted(span_type_counts.keys()):
        cnt = span_type_counts[stype]
        g, p, tp, fp, fn = cnt["gold"], cnt["pred"], cnt["tp"], cnt["fp"], cnt["fn"]
        prec = (tp / max(tp + fp, 1)) * 100 if (tp + fp) > 0 else 0.0
        rec = (tp / max(tp + fn, 1)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / max(prec + rec, 1e-8)) if (prec + rec) > 0 else 0.0
        print(f"{stype:18s} | {g:6d} | {p:6d} | {tp:5d} | {fp:5d} | {fn:5d} | {prec:8.2f}% | {rec:8.2f}% | {f1:8.2f}%")
        span_breakdown_dict[stype] = {
            "gold_count": g,
            "pred_count": p,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    out_data = {
        "examples_20": examples_output,
        "span_breakdown": span_breakdown_dict,
        "overall_span_metrics": {
            "gold_spans_total": eval_res["span_gold_total"],
            "pred_spans_total": eval_res["span_pred_total"],
            "correct_spans_matched": eval_res["span_correct"],
            "precision": eval_res["span_precision"],
            "recall": eval_res["span_recall"],
            "f1": eval_res["span_f1"],
        }
    }
    with open("ARN/results/stage1b_val_examples_and_spans.json", "w") as f:
        json.dump(out_data, f, indent=2)

    return out_data


# ═════════════════════════════════════════════════════════════════════════════
# PART D: Calibration on Positive Predictions Only (Item 6)
# ═════════════════════════════════════════════════════════════════════════════
def run_part_d_positive_calibration(
    model: nn.Module,
    val_gpu_data: GPUData,
    device: torch.device,
    threshold: float = 0.5,
):
    logger.info("=== PART D: Calibration on Positive Predictions Only ===")
    val_indices = list(range(len(val_gpu_data.h)))
    eval_res = evaluate_model_gpu(model, val_gpu_data, val_indices, device, threshold=threshold, batch_size=64)

    probs = eval_res["probs"].flatten()
    targets = eval_res["targets"].flatten()

    pos_mask = probs >= 0.5
    pos_probs = probs[pos_mask]
    pos_targets = targets[pos_mask]

    n_pos = len(pos_probs)
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bin_results = []
    ece_pos = 0.0

    print("\n" + "=" * 80)
    print("POSITIVE-ONLY TOOL ROUTING RELIABILITY TABLE (Threshold >= 0.5)")
    print("=" * 80)
    print(f"{'Bin':15s} | {'Count':8s} | {'Mean Conf':10s} | {'Accuracy':10s} | {'Calib Error':12s}")
    print("-" * 80)

    for lower, upper in bins:
        in_bin = (pos_probs >= lower) & (pos_probs <= upper if upper == 1.0 else pos_probs < upper)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            mean_conf = float(np.mean(pos_probs[in_bin]))
            acc = float(np.mean(pos_targets[in_bin]))
            gap = abs(mean_conf - acc)
            ece_pos += (bin_count / max(n_pos, 1)) * gap
        else:
            mean_conf = 0.0
            acc = 0.0
            gap = 0.0

        bin_str = f"[{lower:.1f}, {upper:.1f})" if upper < 1.0 else f"[{lower:.1f}, {upper:.1f}]"
        print(f"{bin_str:15s} | {bin_count:8d} | {mean_conf:9.4f} | {acc:9.4f} | {gap:11.4f}")
        bin_results.append({
            "bin": bin_str,
            "lower": lower,
            "upper": upper,
            "count": bin_count,
            "mean_confidence": mean_conf,
            "empirical_accuracy": acc,
            "calibration_error": gap,
        })

    print("-" * 80)
    print(f"Total Positive Predictions (p >= 0.5): {n_pos}")
    print(f"Positive-Only ECE (ECE_pos)          : {ece_pos:.4f} ({ece_pos*100:.2f}%)")

    # Plot reliability diagram
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bin_centers = [(b["lower"] + b["upper"]) / 2 for b in bin_results if b["count"] > 0]
    bin_accs = [b["empirical_accuracy"] for b in bin_results if b["count"] > 0]
    bin_confs = [b["mean_confidence"] for b in bin_results if b["count"] > 0]

    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="Perfect Calibration")
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.6, color="#1f77b4", edgecolor="black", label="Empirical Accuracy")
    ax.step([0.5] + [b["upper"] for b in bin_results], [b["mean_confidence"] for b in bin_results] + [1.0], where="pre", color="#d62728", lw=2, label="Mean Confidence")

    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Positive-Only Reliability Diagram (ECE = {ece_pos*100:.2f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Confidence p(y=1 | x)", fontsize=10)
    ax.set_ylabel("Empirical Accuracy", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig_dir = Path("ARN/docs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "stage1b_positive_calibration.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    calib_out = {
        "n_positive_predictions": n_pos,
        "ece_positive": ece_pos,
        "bins": bin_results,
    }
    with open("ARN/results/stage1b_positive_calibration.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    return calib_out


# ═════════════════════════════════════════════════════════════════════════════
# PART E: Main Generalization Estimate via Repeated Grouped CV (Item 7)
# ═════════════════════════════════════════════════════════════════════════════
def run_part_e_repeated_grouped_cv(
    raw_gpu_data: GPUData,
    frame_to_cluster: Dict[str, str],
    device: torch.device,
    lr: float = 3e-4,
    threshold: float = 0.5,
):
    print("\n" + "=" * 90, flush=True)
    print("=== PART E: Repeated Grouped CV over all 253 Clusters (5 Folds x 3 Shuffles) ===", flush=True)
    print("=" * 90, flush=True)
    shuffles = [42, 100, 2024]
    n_folds = 5
    all_runs = []

    for shuffle_idx, seed in enumerate(shuffles, 1):
        print(f"\n--- Shuffle {shuffle_idx}/3 (Seed {seed}) ---", flush=True)
        folds = partition_clusters(raw_gpu_data.raw_samples, frame_to_cluster, n_folds=n_folds, seed=seed)

        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            train_idx = [i for f in range(n_folds) if f != fold_idx for i in folds[f]]

            model, val_res, _ = train_fold_gpu(
                raw_gpu_data, train_idx, val_idx, device,
                lr=lr, lambda_slot=1.0, max_epochs=100, patience=10, batch_size=32, seed=seed + fold_idx,
                verbose_tag=f"Shuffle={shuffle_idx} Fold={fold_idx+1}"
            )

            if abs(threshold - 0.5) > 1e-4:
                val_final = evaluate_model_gpu(model, raw_gpu_data, val_idx, device, threshold=threshold, batch_size=64)
            else:
                val_final = val_res

            run_record = {
                "shuffle": shuffle_idx,
                "shuffle_seed": seed,
                "fold": fold_idx + 1,
                "n_samples": val_final["n_samples"],
                "val_total_loss": val_final["val_total_loss"],
                "tool_subset_acc": val_final["tool_subset_acc"],
                "tool_subset_acc_count": val_final["tool_subset_acc_count"],
                "tool_micro_p": val_final["tool_micro_p"],
                "tool_micro_r": val_final["tool_micro_r"],
                "tool_micro_f1": val_final["tool_micro_f1"],
                "tool_macro_f1": val_final["tool_macro_f1"],
                "span_precision": val_final["span_precision"],
                "span_recall": val_final["span_recall"],
                "span_f1": val_final["span_f1"],
                "span_correct": val_final["span_correct"],
                "span_gold_total": val_final["span_gold_total"],
                "span_pred_total": val_final["span_pred_total"],
                "joint_em": val_final["joint_em"],
                "joint_em_count": val_final["joint_em_count"],
            }
            all_runs.append(run_record)
            print(
                f"  Shuffle {shuffle_idx} Fold {fold_idx+1}/5 | "
                f"Tool Acc: {run_record['tool_subset_acc']*100:.2f}% ({run_record['tool_subset_acc_count']}/{run_record['n_samples']}) | "
                f"Span F1: {run_record['span_f1']*100:.2f}% ({run_record['span_correct']}/{run_record['span_gold_total']}) | "
                f"Joint EM: {run_record['joint_em']*100:.2f}% ({run_record['joint_em_count']}/{run_record['n_samples']})",
                flush=True
            )

    n_runs = len(all_runs)

    def stats(key):
        vals = [r[key] for r in all_runs]
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n_runs))
        return {"mean": mean, "se": se, "std": float(np.std(vals, ddof=1))}

    summary_stats = {
        "n_total_folds": n_runs,
        "n_clusters": len(set(frame_to_cluster.values())),
        "threshold": threshold,
        "learning_rate": lr,
        "tool_subset_accuracy": stats("tool_subset_acc"),
        "tool_micro_f1": stats("tool_micro_f1"),
        "tool_macro_f1": stats("tool_macro_f1"),
        "slot_span_precision": stats("span_precision"),
        "slot_span_recall": stats("span_recall"),
        "slot_span_f1": stats("span_f1"),
        "joint_exact_match": stats("joint_em"),
        "per_fold_runs": all_runs,
    }

    print("\n" + "=" * 90)
    print(f"MAIN GENERALIZATION ESTIMATE: REPEATED GROUPED CV ({n_runs} FOLDS ACROSS {len(set(frame_to_cluster.values()))} CLUSTERS)")
    print("=" * 90)
    print(f"{'Metric':25s} | {'Mean':10s} | {'SE':10s} | {'Std Dev':10s}")
    print("-" * 90)
    for m_name, k in [
        ("Tool Subset Accuracy", "tool_subset_accuracy"),
        ("Tool Micro F1", "tool_micro_f1"),
        ("Tool Macro F1", "tool_macro_f1"),
        ("Slot Span Precision", "slot_span_precision"),
        ("Slot Span Recall", "slot_span_recall"),
        ("Slot Span F1", "slot_span_f1"),
        ("Joint Exact Match", "joint_exact_match"),
    ]:
        s = summary_stats[k]
        print(f"{m_name:25s} | {s['mean']*100:9.2f}% | {s['se']*100:9.2f}% | {s['std']*100:9.2f}%")
    print("=" * 90)

    with open("ARN/results/stage1b_repeated_grouped_cv.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    return summary_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing Stage 1b Diagnostics Suite on device: {device}")

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path, sim_threshold=0.80)
    logger.info(f"Loaded {len(frame_to_cluster)} fine frames mapped to {len(set(frame_to_cluster.values()))} canonical clusters.")

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    val_path = "ARN/data/splits/group_disjoint_val.jsonl"

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw_samples = [json.loads(line) for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_raw_samples = [json.loads(line) for line in f if line.strip()]
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    train_ds = ARNDataset(train_path, tokenizer=TOKENIZER)
    val_ds = ARNDataset(val_path, tokenizer=TOKENIZER)
    raw_ds = ARNDataset(str(raw_path), tokenizer=TOKENIZER)

    base_model = AetherRoutingNetwork(ARNConfig()).to(device)

    logger.info("Precomputing GPU representations for train, val, and raw datasets...")
    train_gpu = prepare_gpu_data(train_ds, train_raw_samples, base_model, device)
    val_gpu = prepare_gpu_data(val_ds, val_raw_samples, base_model, device)
    raw_gpu = prepare_gpu_data(raw_ds, raw_samples, base_model, device)
    logger.info("GPU representation caching completed.")

    # 1. Part A: Tune LR and Threshold
    selected_lr, best_tau, tuning_summary = run_part_a_tuning(train_gpu, frame_to_cluster, device)

    # 2. Part B: Epoch-by-Epoch Training & Plotting
    model, epoch_history = run_part_b_training_and_plotting(train_gpu, val_gpu, device, lr=selected_lr, threshold=best_tau)

    # 3. Part C: 20 Validation Examples & Span Counts
    run_part_c_val_examples_and_spans(model, val_gpu, device, threshold=best_tau)

    # 4. Part D: Positive Calibration
    run_part_d_positive_calibration(model, val_gpu, device, threshold=best_tau)

    # 5. Part E: Repeated Grouped CV
    run_part_e_repeated_grouped_cv(raw_gpu, frame_to_cluster, device, lr=selected_lr, threshold=best_tau)

    logger.info("=== STAGE 1b DIAGNOSTICS SUITE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
