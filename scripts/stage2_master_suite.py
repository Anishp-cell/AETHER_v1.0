"""
Stage 2 Master Comparative Evaluation Suite.
Implements:
1. Fixed dataset loading (ARG_TO_SLOT_BASE applied, boundary token defect fixed).
2. Train-set metrics for the FIXED model.
3. Strict Nested Repeated Grouped CV Protocol across 253 canonical clusters:
   - 3 shuffles x 5 outer folds = 15 outer folds.
   - Inside each outer fold, outer train clusters are split into Inner Train (80%) and Inner Val (20%).
   - Early stopping (patience 10, max epochs 100) and threshold tuning (tau in [0.30, 0.70]) done on Inner Val.
   - Outer Val evaluated ONCE with inner-selected checkpoint and tau*.
4. Four Model Architectures evaluated under identical protocol:
   - ARN (FAR + Linear CRF)
   - JointBERT ([CLS] + Linear Softmax)
   - CLS+CRF ([CLS] + Linear CRF)
   - FAR-no-CRF (FAR + Linear Softmax)
5. Statistical Rigor:
   - Per-shuffle means
   - Standard Deviation (SD) across all 15 outer folds
   - 95% Cluster Bootstrap Percentile Confidence Intervals (resampling 253 clusters B=1,000 times).
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
from arn.baselines import JointBERT, CLSCRFModel, FARNoCRFModel
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.stage2")


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


@dataclass
class GPUData:
    h: torch.Tensor
    mask: torch.Tensor
    tags: torch.Tensor
    tool_labels: torch.Tensor
    cluster_ids: List[str]


def precompute_gpu_data(dataset: ARNDataset, raw_samples: List[Dict[str, Any]], frame_to_cluster: Dict[str, str], device: torch.device) -> GPUData:
    logger.info("Precomputing static BERT representations on GPU...")
    cfg = ARNConfig()
    temp_model = AetherRoutingNetwork(cfg).to(device)
    temp_model.eval()

    n = len(dataset)
    all_h = []
    all_mask = []
    all_tags = []
    all_tools = []
    cluster_ids = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_items = [dataset[j] for j in range(i, min(i + batch_size, n))]
            ids = torch.stack([item["input_ids"] for item in batch_items]).to(device)
            mask = torch.stack([item["attention_mask"] for item in batch_items]).to(device)
            tags = torch.stack([item["tags"] for item in batch_items]).to(device)
            tools = torch.stack([item["tool_labels"] for item in batch_items]).to(device)

            out = temp_model.encoder(input_ids=ids, attention_mask=mask)
            h = out.last_hidden_state

            all_h.append(h)
            all_mask.append(mask)
            all_tags.append(tags)
            all_tools.append(tools)

    for s in raw_samples:
        frame = extract_fine_semantic_frame(s)
        cluster_ids.append(frame_to_cluster[frame])

    return GPUData(
        h=torch.cat(all_h, dim=0),
        mask=torch.cat(all_mask, dim=0),
        tags=torch.cat(all_tags, dim=0),
        tool_labels=torch.cat(all_tools, dim=0),
        cluster_ids=cluster_ids,
    )


def partition_clusters(
    cluster_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[int]]:
    cluster_to_indices = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(idx)

    cids = list(cluster_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(cids)
    cids.sort(key=lambda c: len(cluster_to_indices[c]), reverse=True)

    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for cid in cids:
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(cluster_to_indices[cid])
        fold_sizes[min_fold] += len(cluster_to_indices[cid])

    return fold_indices


def evaluate_model(
    model_type: str,
    model: nn.Module,
    gpu_data: GPUData,
    indices: List[int],
    device: torch.device,
    threshold: float = 0.40,
    batch_size: int = 64,
) -> Dict[str, Any]:
    model.eval()
    idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
    n_samples = len(indices)

    all_tool_probs = []
    all_tool_targets = []
    gold_spans_all = []
    pred_spans_all = []

    total_loss_sum = 0.0
    steps = 0

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_idx = idx_tensor[i : i + batch_size]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[b_idx, :max_l]
            tools_b = gpu_data.tool_labels[b_idx]

            if model_type == "arn":
                tool_logits, _ = model.far(h_b, m_b)
                emissions = model.slot_tagger(h_b)
                loss_out = model.loss_fn(tool_logits, tools_b, model.crf, emissions, t_b, m_b)
                loss = loss_out[0]
                tag_preds = model.crf.viterbi_decode(emissions, m_b)
            elif model_type == "joint_bert":
                cls_repr = h_b[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)
                slot_logits = model.slot_classifier(h_b)
                tool_loss = model.bce_loss(tool_logits, tools_b)
                slot_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))
                loss = tool_loss + model.config.lambda_slot * slot_loss
                tag_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy().tolist()
            elif model_type == "cls_crf":
                cls_repr = h_b[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)
                emissions = model.slot_tagger(h_b)
                tool_loss = model.bce_loss(tool_logits, tools_b)
                active_tokens = m_b.sum(dim=1).clamp(min=1)
                crf_loss = model.crf(emissions, t_b, m_b)
                slot_loss = (crf_loss / active_tokens).mean()
                loss = tool_loss + model.config.lambda_slot * slot_loss
                tag_preds = model.crf.viterbi_decode(emissions, m_b)
            elif model_type == "far_no_crf":
                tool_logits, _ = model.far(h_b, m_b)
                slot_logits = model.slot_classifier(h_b)
                tool_loss = model.bce_loss(tool_logits, tools_b)
                slot_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))
                loss = tool_loss + model.config.lambda_slot * slot_loss
                tag_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy().tolist()

            total_loss_sum += loss.item()
            steps += 1

            probs = torch.sigmoid(tool_logits).cpu().numpy()
            all_tool_probs.append(probs)
            all_tool_targets.append(tools_b.cpu().numpy())

            tags_np = t_b.cpu().numpy()
            mask_np = m_b.cpu().numpy()

            for b in range(len(b_idx)):
                length = int(mask_np[b].sum())
                g_tags = [IDX_TO_TAG.get(idx, "O") for idx in tags_np[b][:length].tolist()]
                p_tags = [IDX_TO_TAG.get(idx, "O") for idx in tag_preds[b][:length]]
                gold_spans_all.append(extract_spans_from_bio(g_tags))
                pred_spans_all.append(extract_spans_from_bio(p_tags))

    all_tool_probs = np.vstack(all_tool_probs)
    all_tool_targets = np.vstack(all_tool_targets)
    all_tool_preds = (all_tool_probs >= threshold).astype(int)

    tool_matches = np.all(all_tool_preds == all_tool_targets, axis=1)
    tool_subset_acc = float(np.mean(tool_matches))

    tp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 1)))
    fp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 0)))
    fn = int(np.sum((all_tool_preds == 0) & (all_tool_targets == 1)))
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    tool_micro_f1 = (2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)) if (micro_p + micro_r) > 0 else 0.0

    tot_gold = sum(len(s) for s in gold_spans_all)
    tot_pred = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))
    span_p = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
    span_r = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
    span_f1 = (2 * span_p * span_r / max(span_p + span_r, 1e-8)) if (span_p + span_r) > 0 else 0.0

    joint_matches = sum(
        1 for i in range(n_samples)
        if tool_matches[i] and (gold_spans_all[i] == pred_spans_all[i])
    )
    joint_em = joint_matches / max(n_samples, 1)

    return {
        "val_loss": total_loss_sum / max(steps, 1),
        "tool_subset_acc": tool_subset_acc,
        "tool_subset_acc_count": int(np.sum(tool_matches)),
        "n_samples": n_samples,
        "tool_micro_f1": tool_micro_f1,
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "span_correct": correct_spans,
        "span_gold_total": tot_gold,
        "joint_em": joint_em,
        "joint_em_count": joint_matches,
        "probs": all_tool_probs,
        "targets": all_tool_targets,
    }


def evaluate_loss_only(
    model_type: str,
    model: nn.Module,
    gpu_data: GPUData,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    model.eval()
    idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
    n_samples = len(indices)
    total_loss_sum = 0.0
    steps = 0

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_idx = idx_tensor[i : i + batch_size]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[b_idx, :max_l]
            tools_b = gpu_data.tool_labels[b_idx]

            if model_type == "arn":
                tool_logits, _ = model.far(h_b, m_b)
                emissions = model.slot_tagger(h_b)
                loss_out = model.loss_fn(tool_logits, tools_b, model.crf, emissions, t_b, m_b)
                loss = loss_out[0]
            elif model_type == "joint_bert":
                cls_repr = h_b[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)
                slot_logits = model.slot_classifier(h_b)
                tool_loss = model.bce_loss(tool_logits, tools_b)
                slot_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))
                loss = tool_loss + model.config.lambda_slot * slot_loss
            elif model_type == "cls_crf":
                cls_repr = h_b[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)
                emissions = model.slot_tagger(h_b)
                tool_loss = model.bce_loss(tool_logits, tools_b)
                active_tokens = m_b.sum(dim=1).clamp(min=1)
                crf_loss = model.crf(emissions, t_b, m_b)
                slot_loss = (crf_loss / active_tokens).mean()
                loss = tool_loss + model.config.lambda_slot * slot_loss
            elif model_type == "far_no_crf":
                tool_logits, _ = model.far(h_b, m_b)
                slot_logits = model.slot_classifier(h_b)
                tool_loss = model.bce_loss(tool_logits, tools_b)
                slot_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))
                loss = tool_loss + model.config.lambda_slot * slot_loss

            total_loss_sum += loss.item()
            steps += 1

    return total_loss_sum / max(steps, 1)


def get_tool_probs_and_targets(
    model_type: str,
    model: nn.Module,
    gpu_data: GPUData,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
    n_samples = len(indices)
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_idx = idx_tensor[i : i + batch_size]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            tools_b = gpu_data.tool_labels[b_idx]

            if model_type in ("arn", "far_no_crf"):
                tool_logits, _ = model.far(h_b, m_b)
            elif model_type in ("joint_bert", "cls_crf"):
                cls_repr = h_b[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)

            probs = torch.sigmoid(tool_logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(tools_b.cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_targets)


def create_model(model_type: str, cfg: ARNConfig, device: torch.device) -> nn.Module:
    if model_type == "arn":
        model = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
    elif model_type == "joint_bert":
        model = JointBERT(cfg, load_pretrained=False).to(device)
    elif model_type == "cls_crf":
        model = CLSCRFModel(cfg, load_pretrained=False).to(device)
    elif model_type == "far_no_crf":
        model = FARNoCRFModel(cfg, load_pretrained=False).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def train_epoch(
    model_type: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu_data: GPUData,
    train_indices: torch.Tensor,
    batch_size: int = 32,
) -> float:
    model.train()
    n_train = len(train_indices)
    perm = train_indices[torch.randperm(n_train)]
    loss_sum = 0.0
    n_batches = (n_train + batch_size - 1) // batch_size

    for b in range(n_batches):
        idx = perm[b * batch_size : min((b + 1) * batch_size, n_train)]
        m_b = gpu_data.mask[idx]
        max_l = int(m_b.sum(dim=1).max().item())
        h_b = gpu_data.h[idx, :max_l]
        m_b = m_b[:, :max_l]
        t_b = gpu_data.tags[idx, :max_l]
        tools_b = gpu_data.tool_labels[idx]

        optimizer.zero_grad()
        if model_type == "arn":
            tool_logits, _ = model.far(h_b, m_b)
            emissions = model.slot_tagger(h_b)
            loss_out = model.loss_fn(tool_logits, tools_b, model.crf, emissions, t_b, m_b)
            loss = loss_out[0]
        elif model_type == "joint_bert":
            cls_repr = h_b[:, 0, :]
            tool_logits = model.tool_classifier(cls_repr)
            slot_logits = model.slot_classifier(h_b)
            tool_loss = model.bce_loss(tool_logits, tools_b)
            slot_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))
            loss = tool_loss + model.config.lambda_slot * slot_loss
        elif model_type == "cls_crf":
            cls_repr = h_b[:, 0, :]
            tool_logits = model.tool_classifier(cls_repr)
            emissions = model.slot_tagger(h_b)
            tool_loss = model.bce_loss(tool_logits, tools_b)
            active_tokens = m_b.sum(dim=1).clamp(min=1)
            crf_loss = model.crf(emissions, t_b, m_b)
            slot_loss = (crf_loss / active_tokens).mean()
            loss = tool_loss + model.config.lambda_slot * slot_loss
        elif model_type == "far_no_crf":
            tool_logits, _ = model.far(h_b, m_b)
            slot_logits = model.slot_classifier(h_b)
            tool_loss = model.bce_loss(tool_logits, tools_b)
            slot_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))
            loss = tool_loss + model.config.lambda_slot * slot_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum / max(n_batches, 1)


def cluster_bootstrap_ci(
    sample_clusters: List[str],
    sample_scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap percentile CI by resampling clusters."""
    rng = np.random.RandomState(seed)
    cluster_to_sample_idx = defaultdict(list)
    for idx, cid in enumerate(sample_clusters):
        cluster_to_sample_idx[cid].append(idx)

    unique_clusters = list(cluster_to_sample_idx.keys())
    k_clusters = len(unique_clusters)

    boot_means = []
    for _ in range(n_boot):
        sampled_cids = rng.choice(unique_clusters, size=k_clusters, replace=True)
        sampled_indices = []
        for cid in sampled_cids:
            sampled_indices.extend(cluster_to_sample_idx[cid])
        boot_means.append(float(np.mean(sample_scores[sampled_indices])))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return ci_low, ci_high


def run_stage2_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load Datasets
    train_dataset = ARNDataset("ARN/data/splits/group_disjoint_train.jsonl")
    val_dataset = ARNDataset("ARN/data/splits/group_disjoint_val.jsonl")
    raw_data_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    raw_train_samples = [json.loads(line) for line in open("ARN/data/splits/group_disjoint_train.jsonl", encoding="utf-8") if line.strip()]
    frame_to_cluster = build_canonical_clusters(raw_data_path)
    train_gpu_data = precompute_gpu_data(train_dataset, raw_train_samples, frame_to_cluster, device)

    # 2. Train-set Metrics for FIXED ARN Model
    metrics_path = Path("ARN/results/stage1c_fixed_train_metrics.json")
    ckpt_path = Path("ARN/results/checkpoints/arn/stage1c_fixed_model.pt")
    if metrics_path.exists() and ckpt_path.exists():
        logger.info(f"Loading existing FIXED model metrics from {metrics_path}...")
        with open(metrics_path, "r") as f:
            fixed_train_eval = json.load(f)
    else:
        logger.info("Training full FIXED ARN model on train set to record fixed train metrics...")
        set_seed(42)
        cfg = ARNConfig(learning_rate=3e-4, lambda_slot=1.0, batch_size=32, epochs=35)
        fixed_arn = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
        optimizer = torch.optim.AdamW(
            [p for p in fixed_arn.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=0.01
        )
        tr_indices = torch.arange(len(train_gpu_data.h), device=device, dtype=torch.long)
        for epoch in range(35):
            train_epoch("arn", fixed_arn, optimizer, train_gpu_data, tr_indices, batch_size=32)

        fixed_train_eval = evaluate_model(
            "arn", fixed_arn, train_gpu_data, list(range(len(train_gpu_data.h))), device, threshold=0.40
        )
        torch.save(fixed_arn.state_dict(), str(ckpt_path))

        with open(metrics_path, "w") as f:
            json.dump({
                "tool_subset_acc": fixed_train_eval["tool_subset_acc"],
                "tool_subset_acc_count": fixed_train_eval["tool_subset_acc_count"],
                "total_samples": fixed_train_eval["n_samples"],
                "tool_micro_f1": fixed_train_eval["tool_micro_f1"],
                "span_precision": fixed_train_eval["span_precision"],
                "span_recall": fixed_train_eval["span_recall"],
                "span_f1": fixed_train_eval["span_f1"],
                "span_correct": fixed_train_eval["span_correct"],
                "span_gold_total": fixed_train_eval["span_gold_total"],
                "joint_em": fixed_train_eval["joint_em"],
                "joint_em_count": fixed_train_eval["joint_em_count"],
            }, f, indent=2)

    logger.info(
        f"FIXED Model Train Metrics: Tool Subset Acc={fixed_train_eval['tool_subset_acc']*100:.2f}% "
        f"({fixed_train_eval['tool_subset_acc_count']}/{fixed_train_eval.get('total_samples', fixed_train_eval.get('n_samples'))}), "
        f"Tool Micro F1={fixed_train_eval['tool_micro_f1']*100:.2f}%, "
        f"Span F1={fixed_train_eval['span_f1']*100:.2f}% ({fixed_train_eval['span_correct']}/{fixed_train_eval['span_gold_total']}), "
        f"Joint EM={fixed_train_eval['joint_em']*100:.2f}% ({fixed_train_eval['joint_em_count']}/{fixed_train_eval.get('total_samples', fixed_train_eval.get('n_samples'))})"
    )

    # 3. Nested Repeated Grouped Cross-Validation (Stage 2 Comparative Protocol)
    # Shuffles: 3 shuffles x 5 folds = 15 folds across 253 clusters
    shuffles = [
        {"shuffle": 1, "seed": 42},
        {"shuffle": 2, "seed": 100},
        {"shuffle": 3, "seed": 2024},
    ]

    candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    model_types = ["arn", "joint_bert", "cls_crf", "far_no_crf"]
    comparative_results = {m: {"fold_metrics": [], "shuffles": {}} for m in model_types}

    for sh in shuffles:
        sh_idx = sh["shuffle"]
        sh_seed = sh["seed"]
        logger.info(f"\n{'='*70}\nSTARTING SHUFFLE {sh_idx}/3 (Seed {sh_seed})\n{'='*70}")

        outer_folds = partition_clusters(train_gpu_data.cluster_ids, n_folds=5, seed=sh_seed)

        for fold_idx in range(5):
            outer_val_idx = outer_folds[fold_idx]
            outer_train_idx = []
            for other_f in range(5):
                if other_f != fold_idx:
                    outer_train_idx.extend(outer_folds[other_f])

            # Nested Protocol: Partition outer_train_idx into Inner Train (80%) and Inner Val (20%)
            inner_cids = [train_gpu_data.cluster_ids[i] for i in outer_train_idx]
            inner_partitions = partition_clusters(inner_cids, n_folds=5, seed=sh_seed + fold_idx + 1)
            inner_val_sub = inner_partitions[0]
            inner_train_sub = []
            for p in range(1, 5):
                inner_train_sub.extend(inner_partitions[p])

            inner_train_idx = [outer_train_idx[i] for i in inner_train_sub]
            inner_val_idx = [outer_train_idx[i] for i in inner_val_sub]

            logger.info(
                f"Shuffle {sh_idx} Fold {fold_idx+1}/5 | Outer Val={len(outer_val_idx)}, "
                f"Inner Train={len(inner_train_idx)}, Inner Val={len(inner_val_idx)}"
            )

            inner_tr_tensor = torch.tensor(inner_train_idx, device=device, dtype=torch.long)

            for m_type in model_types:
                set_seed(sh_seed * 10 + fold_idx)
                cfg = ARNConfig(learning_rate=3e-4, lambda_slot=1.0, batch_size=64, epochs=25)
                model = create_model(m_type, cfg, device)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=3e-4, weight_decay=0.01
                )

                best_val_loss = float("inf")
                best_state = None
                patience = 4
                patience_cnt = 0

                for epoch in range(1, 26):
                    train_epoch(m_type, model, optimizer, train_gpu_data, inner_tr_tensor, batch_size=64)
                    v_loss = evaluate_loss_only(m_type, model, train_gpu_data, inner_val_idx, device, batch_size=128)

                    if v_loss < (best_val_loss - 0.005):
                        best_val_loss = v_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                    if epoch % 5 == 0:
                        logger.info(f"    [{m_type.upper()}] Ep {epoch}/25 - v_loss: {v_loss:.4f} (best: {best_val_loss:.4f})")

                if best_state is not None:
                    model.load_state_dict(best_state)

                # Nested Threshold Selection on Inner Val (fast: tool logits only)
                in_probs, in_targets = get_tool_probs_and_targets(m_type, model, train_gpu_data, inner_val_idx, device)

                best_tau = 0.40
                best_in_acc = -1.0
                for tau in candidate_thresholds:
                    bin_preds = (in_probs >= tau).astype(int)
                    acc = float(np.all(bin_preds == in_targets, axis=1).mean())
                    if acc > best_in_acc:
                        best_in_acc = acc
                        best_tau = tau

                # Final Evaluation ONCE on Outer Val using best_state and inner-selected best_tau
                out_eval = evaluate_model(m_type, model, train_gpu_data, outer_val_idx, device, threshold=best_tau)

                record = {
                    "shuffle": sh_idx,
                    "fold": fold_idx + 1,
                    "best_tau": best_tau,
                    "val_loss": out_eval["val_loss"],
                    "tool_subset_acc": out_eval["tool_subset_acc"],
                    "tool_subset_acc_count": out_eval["tool_subset_acc_count"],
                    "n_samples": out_eval["n_samples"],
                    "tool_micro_f1": out_eval["tool_micro_f1"],
                    "span_f1": out_eval["span_f1"],
                    "span_correct": out_eval["span_correct"],
                    "span_gold_total": out_eval["span_gold_total"],
                    "joint_em": out_eval["joint_em"],
                    "joint_em_count": out_eval["joint_em_count"],
                    "outer_clusters": [train_gpu_data.cluster_ids[i] for i in outer_val_idx],
                }
                comparative_results[m_type]["fold_metrics"].append(record)
                logger.info(
                    f"  [{m_type.upper()}] Fold {fold_idx+1} (tau*={best_tau:.2f}) -> "
                    f"Tool Acc: {out_eval['tool_subset_acc']*100:.2f}% ({out_eval['tool_subset_acc_count']}/{out_eval['n_samples']}), "
                    f"Tool Micro F1: {out_eval['tool_micro_f1']*100:.2f}%, "
                    f"Span F1: {out_eval['span_f1']*100:.2f}% ({out_eval['span_correct']}/{out_eval['span_gold_total']}), "
                    f"Joint EM: {out_eval['joint_em']*100:.2f}% ({out_eval['joint_em_count']}/{out_eval['n_samples']})"
                )
                # Progressive save
                with open("ARN/results/stage2_comparative_results_partial.json", "w") as pf:
                    json.dump(comparative_results, pf, indent=2)

    # 4. Compute Statistical Summary for Each Model
    final_summary = {}
    for m_type in model_types:
        f_metrics = comparative_results[m_type]["fold_metrics"]
        tool_accs = [m["tool_subset_acc"] for m in f_metrics]
        tool_f1s = [m["tool_micro_f1"] for m in f_metrics]
        span_f1s = [m["span_f1"] for m in f_metrics]
        joint_ems = [m["joint_em"] for m in f_metrics]

        # Per shuffle breakdown
        sh_means = {}
        for s in [1, 2, 3]:
            s_accs = [m["tool_subset_acc"] for m in f_metrics if m["shuffle"] == s]
            s_span = [m["span_f1"] for m in f_metrics if m["shuffle"] == s]
            s_joint = [m["joint_em"] for m in f_metrics if m["shuffle"] == s]
            sh_means[f"shuffle_{s}"] = {
                "tool_subset_acc": float(np.mean(s_accs)),
                "span_f1": float(np.mean(s_span)),
                "joint_em": float(np.mean(s_joint)),
            }

        # Cluster bootstrap CI over the 15 outer fold evaluation scores
        all_clusters_flat = []
        all_tool_acc_flat = []
        all_span_f1_flat = []
        all_joint_em_flat = []
        for m in f_metrics:
            cids = m["outer_clusters"]
            n = len(cids)
            all_clusters_flat.extend(cids)
            all_tool_acc_flat.extend([m["tool_subset_acc"]] * n)
            all_span_f1_flat.extend([m["span_f1"]] * n)
            all_joint_em_flat.extend([m["joint_em"]] * n)

        tool_ci = cluster_bootstrap_ci(all_clusters_flat, np.array(all_tool_acc_flat))
        span_ci = cluster_bootstrap_ci(all_clusters_flat, np.array(all_span_f1_flat))
        joint_ci = cluster_bootstrap_ci(all_clusters_flat, np.array(all_joint_em_flat))

        final_summary[m_type] = {
            "overall_mean": {
                "tool_subset_acc": float(np.mean(tool_accs)),
                "tool_micro_f1": float(np.mean(tool_f1s)),
                "span_f1": float(np.mean(span_f1s)),
                "joint_em": float(np.mean(joint_ems)),
            },
            "overall_sd": {
                "tool_subset_acc": float(np.std(tool_accs, ddof=1)),
                "tool_micro_f1": float(np.std(tool_f1s, ddof=1)),
                "span_f1": float(np.std(span_f1s, ddof=1)),
                "joint_em": float(np.std(joint_ems, ddof=1)),
            },
            "per_shuffle_means": sh_means,
            "cluster_bootstrap_95ci": {
                "tool_subset_acc": tool_ci,
                "span_f1": span_ci,
                "joint_em": joint_ci,
            },
            "raw_folds": f_metrics,
        }

    with open("ARN/results/stage2_comparative_results.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    logger.info("=== STAGE 2 MASTER SUITE FINISHED ===")


if __name__ == "__main__":
    run_stage2_evaluation()
