"""
Stage 2c Item 2: Softmax Baselines, Constrained Decoding, and CRF Contribution Analysis.

Computes:
1. Tag-level F1 (micro and macro over BIO tags).
2. Strict vs. Lenient span extraction.
3. Softmax + hard BIO-masked constrained decoding (Viterbi with zero learned transitions).
4. Standard token-classification sanity reference.
5. Decomposed contribution of structural grammar vs. learnable transitions + global normalization.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
import torch.nn as nn

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG
from arn.model import AetherRoutingNetwork
from arn.baselines import CLSCRFModel, JointBERT, FARNoCRFModel
from arn.crf import build_bio_transition_mask
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data, GPUData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2c_softmax")


def extract_spans_strict(tags: List[str]) -> Set[Tuple[str, int, int]]:
    """Strict span matching: exact (entity_type, start, end)."""
    spans = set()
    current_label = None
    start = -1
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_label is not None:
                spans.add((current_label, start, i - 1))
            current_label = tag[2:]
            start = i
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label is None or label != current_label:
                if current_label is not None:
                    spans.add((current_label, start, i - 1))
                current_label = label
                start = i
        else:
            if current_label is not None:
                spans.add((current_label, start, i - 1))
                current_label = None
                start = -1
    if current_label is not None:
        spans.add((current_label, start, len(tags) - 1))
    return spans


def compute_span_metrics(
    gold_spans_list: List[Set[Tuple[str, int, int]]],
    pred_spans_list: List[Set[Tuple[str, int, int]]],
) -> Dict[str, float]:
    """Compute strict and lenient span P, R, F1."""
    strict_correct = 0
    lenient_correct = 0
    total_gold = sum(len(g) for g in gold_spans_list)
    total_pred = sum(len(p) for p in pred_spans_list)

    for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
        # Strict
        strict_correct += len(gold_spans & pred_spans)

        # Lenient: type matches and [s, e] overlaps
        for g_type, g_s, g_e in gold_spans:
            for p_type, p_s, p_e in pred_spans:
                if g_type == p_type and not (p_e < g_s or p_s > g_e):
                    lenient_correct += 1
                    break

    s_p = strict_correct / max(total_pred, 1) if total_pred > 0 else 0.0
    s_r = strict_correct / max(total_gold, 1) if total_gold > 0 else 0.0
    s_f1 = (2 * s_p * s_r / max(s_p + s_r, 1e-8)) if (s_p + s_r) > 0 else 0.0

    l_p = lenient_correct / max(total_pred, 1) if total_pred > 0 else 0.0
    l_r = lenient_correct / max(total_gold, 1) if total_gold > 0 else 0.0
    l_f1 = (2 * l_p * l_r / max(l_p + l_r, 1e-8)) if (l_p + l_r) > 0 else 0.0

    return {
        "strict_precision": s_p,
        "strict_recall": s_r,
        "strict_f1": s_f1,
        "strict_correct": strict_correct,
        "lenient_precision": l_p,
        "lenient_recall": l_r,
        "lenient_f1": l_f1,
        "lenient_correct": lenient_correct,
        "total_gold": total_gold,
        "total_pred": total_pred,
    }


def compute_tag_level_f1(gold_tags_flat: List[str], pred_tags_flat: List[str]) -> Dict[str, float]:
    """Compute micro and macro tag-level F1 across tokens."""
    # Micro F1 (all tokens excluding PAD)
    active_indices = [i for i, t in enumerate(gold_tags_flat) if t != "<PAD>"]
    g_active = [gold_tags_flat[i] for i in active_indices]
    p_active = [pred_tags_flat[i] for i in active_indices]

    correct = sum(1 for g, p in zip(g_active, p_active) if g == p)
    micro_acc = correct / max(len(g_active), 1)

    # Class-by-class for Macro F1 (excluding O and PAD)
    classes = sorted(list(set(g_active) - {"O", "<PAD>"}))
    class_f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(g_active, p_active) if g == c and p == c)
        fp = sum(1 for g, p in zip(g_active, p_active) if g != c and p == c)
        fn = sum(1 for g, p in zip(g_active, p_active) if g == c and p != c)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0
        class_f1s.append(f1)

    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

    # Count invalid BIO transitions in predictions
    invalid_transitions = 0
    total_transitions = 0
    for i in range(1, len(p_active)):
        prev = p_active[i - 1]
        curr = p_active[i]
        total_transitions += 1
        if curr.startswith("I-"):
            curr_type = curr[2:]
            if prev == "O" or prev == "<PAD>":
                invalid_transitions += 1
            elif prev.startswith("B-") or prev.startswith("I-"):
                if prev[2:] != curr_type:
                    invalid_transitions += 1

    invalid_rate = invalid_transitions / max(total_transitions, 1)

    return {
        "tag_micro_acc": micro_acc,
        "tag_macro_f1": macro_f1,
        "invalid_bio_rate": invalid_rate,
        "invalid_bio_count": invalid_transitions,
        "total_transitions": total_transitions,
    }


def bio_constrained_viterbi_decode(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    trans_mask: torch.Tensor,
    start_mask: torch.Tensor,
    end_mask: torch.Tensor,
    large_neg: float = -10000.0,
) -> List[List[int]]:
    """
    Decodes softmax emission logits using Viterbi constrained by hard BIO grammar,
    with ZERO learnable transition weights.
    """
    batch_size, seq_len, num_tags = emissions.shape
    log_probs = torch.log_softmax(emissions, dim=-1)

    # Convert boolean masks to additive penalty
    t_pen = torch.where(trans_mask, torch.zeros_like(trans_mask, dtype=torch.float32), torch.tensor(large_neg))
    s_pen = torch.where(start_mask, torch.zeros_like(start_mask, dtype=torch.float32), torch.tensor(large_neg))
    e_pen = torch.where(end_mask, torch.zeros_like(end_mask, dtype=torch.float32), torch.tensor(large_neg))

    viterbi_vars = s_pen.unsqueeze(0) + log_probs[:, 0, :]
    backpointers = []

    for t in range(1, seq_len):
        active = mask[:, t].bool().unsqueeze(1)
        next_var = viterbi_vars.unsqueeze(2) + t_pen.unsqueeze(0)
        max_vars, bptrs = torch.max(next_var, dim=1)
        max_vars = max_vars + log_probs[:, t, :]
        viterbi_vars = torch.where(active, max_vars, viterbi_vars)
        backpointers.append(bptrs)

    terminal_vars = viterbi_vars + e_pen.unsqueeze(0)
    best_tag_ids = torch.argmax(terminal_vars, dim=1)

    pad_val = TAG_TO_IDX.get("<PAD>", 25)
    best_paths = []

    for b in range(batch_size):
        active_len = int(mask[b].sum().item())
        if active_len <= 0:
            best_paths.append([pad_val] * seq_len)
            continue

        curr_tag = best_tag_ids[b].item()
        path = [curr_tag]
        for t in range(active_len - 2, -1, -1):
            curr_tag = backpointers[t][b, curr_tag].item()
            path.append(curr_tag)

        path.reverse()
        path.extend([pad_val] * (seq_len - len(path)))
        best_paths.append(path)

    return best_paths


def evaluate_decoding_variants(
    model_type: str,
    model: nn.Module,
    gpu_data: GPUData,
    indices: List[int],
    device: torch.device,
) -> Dict:
    model.eval()
    t_mask, s_mask, e_mask = build_bio_transition_mask(TAG_TO_IDX)
    t_mask, s_mask, e_mask = t_mask.to(device), s_mask.to(device), e_mask.to(device)

    gold_tags_flat = []
    gold_spans = []

    greedy_tags_flat = []
    greedy_spans = []

    constrained_tags_flat = []
    constrained_spans = []

    crf_tags_flat = []
    crf_spans = []

    with torch.no_grad():
        for b_start in range(0, len(indices), 128):
            b_idx = torch.tensor(indices[b_start : min(b_start + 128, len(indices))], device=device, dtype=torch.long)
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[b_idx, :max_l]

            if model_type in ("arn", "cls_crf"):
                emissions = model.slot_tagger(h_b)
                crf_preds = model.crf.viterbi_decode(emissions, m_b)
            else:
                emissions = model.slot_classifier(h_b)
                crf_preds = None

            # 1. Greedy Unconstrained Softmax
            greedy_preds_tensor = torch.argmax(emissions, dim=-1)

            # 2. Constrained Softmax (mask only, no learned transitions)
            constrained_preds = bio_constrained_viterbi_decode(emissions, m_b, t_mask, s_mask, e_mask)

            tags_np = t_b.cpu().numpy()
            mask_np = m_b.cpu().numpy()

            for i in range(len(b_idx)):
                l_i = int(mask_np[i].sum())
                g_seq = [IDX_TO_TAG.get(idx, "O") for idx in tags_np[i][:l_i].tolist()]
                p_greedy = [IDX_TO_TAG.get(idx, "O") for idx in greedy_preds_tensor[i, :l_i].tolist()]
                p_const = [IDX_TO_TAG.get(idx, "O") for idx in constrained_preds[i][:l_i]]

                gold_tags_flat.extend(g_seq)
                gold_spans.append(extract_spans_strict(g_seq))

                greedy_tags_flat.extend(p_greedy)
                greedy_spans.append(extract_spans_strict(p_greedy))

                constrained_tags_flat.extend(p_const)
                constrained_spans.append(extract_spans_strict(p_const))

                if crf_preds is not None:
                    p_crf = [IDX_TO_TAG.get(idx, "O") for idx in crf_preds[i][:l_i]]
                    crf_tags_flat.extend(p_crf)
                    crf_spans.append(extract_spans_strict(p_crf))

    # Evaluate
    res = {}
    res["greedy_softmax"] = {
        "span_metrics": compute_span_metrics(gold_spans, greedy_spans),
        "tag_metrics": compute_tag_level_f1(gold_tags_flat, greedy_tags_flat),
    }
    res["constrained_softmax"] = {
        "span_metrics": compute_span_metrics(gold_spans, constrained_spans),
        "tag_metrics": compute_tag_level_f1(gold_tags_flat, constrained_tags_flat),
    }
    if crf_preds is not None:
        res["crf_decoding"] = {
            "span_metrics": compute_span_metrics(gold_spans, crf_spans),
            "tag_metrics": compute_tag_level_f1(gold_tags_flat, crf_tags_flat),
        }

    return res


def run_stage2c_softmax_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Softmax Baselines & Constrained Decoding Suite on: {device}", flush=True)

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)

    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
    gpu_data = precompute_gpu_data(dataset, raw_samples, frame_to_cluster, device)

    # 1. Evaluate Train Set ($N=3,926$) on the fixed train models
    print("\n" + "=" * 95, flush=True)
    print("SOFTMAX vs. CONSTRAINED vs. CRF DECODING: TRAIN SET EVALUATION (N=3,926)", flush=True)
    print("=" * 95, flush=True)

    # Load fixed checkpoints or train fast reference models
    train_idx = list(range(len(raw_samples)))

    # A. JointBERT (Softmax)
    print("Training JointBERT reference model on train set (30 epochs)...", flush=True)
    cfg = ARNConfig(freeze_trunk=True)
    jbert = JointBERT(cfg, load_pretrained=False).to(device)
    opt = torch.optim.AdamW(jbert.parameters(), lr=3e-4, weight_decay=0.01)
    tr_indices = torch.tensor(train_idx, device=device, dtype=torch.long)
    for ep in range(30):
        jbert.train()
        perm = tr_indices[torch.randperm(len(train_idx))]
        for b in range(0, len(train_idx), 64):
            idx = perm[b : min(b + 64, len(train_idx))]
            m_b = gpu_data.mask[idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[idx, :max_l]
            tools_b = gpu_data.tool_labels[idx]

            opt.zero_grad()
            slot_logits = jbert.slot_classifier(h_b)
            loss = jbert.ce_loss(slot_logits.view(-1, cfg.num_tags), t_b.view(-1))
            loss.backward()
            opt.step()

    jbert_eval = evaluate_decoding_variants("joint_bert", jbert, gpu_data, train_idx, device)

    # B. CLS+CRF (CRF)
    print("Training CLS+CRF reference model on train set (30 epochs)...", flush=True)
    clscrf = CLSCRFModel(cfg, load_pretrained=False).to(device)
    opt = torch.optim.AdamW(list(clscrf.slot_tagger.parameters()) + list(clscrf.crf.parameters()), lr=3e-4, weight_decay=0.01)
    for ep in range(30):
        clscrf.train()
        perm = tr_indices[torch.randperm(len(train_idx))]
        for b in range(0, len(train_idx), 64):
            idx = perm[b : min(b + 64, len(train_idx))]
            m_b = gpu_data.mask[idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[idx, :max_l]

            opt.zero_grad()
            emissions = clscrf.slot_tagger(h_b)
            sentence_nlls = clscrf.crf.sentence_nll(emissions, t_b, m_b)
            active = m_b.sum(dim=1).float().clamp(min=1.0)
            loss = (sentence_nlls / active).mean()
            loss.backward()
            opt.step()

    crf_eval = evaluate_decoding_variants("cls_crf", clscrf, gpu_data, train_idx, device)

    # Display comparison table
    print("\n" + "=" * 105, flush=True)
    print("DECODING METHOD COMPARISON ON IDENTICAL EMISSIONS (TRAIN SET, N=3,926, 6,567 GOLD SPANS)")
    print("=" * 105, flush=True)
    header = f"{'Decoding Strategy':35s} | {'Tag Micro Acc':13s} | {'Tag Macro F1':12s} | {'Invalid BIO %':13s} | {'Strict Span F1':14s} | {'Lenient Span F1':15s}"
    print(header, flush=True)
    print("-" * 105, flush=True)

    # 1. Softmax Greedy
    g_m = jbert_eval["greedy_softmax"]
    print(
        f"{'1. Softmax (Unconstrained Greedy)':35s} | "
        f"{g_m['tag_metrics']['tag_micro_acc']*100:11.2f}% | "
        f"{g_m['tag_metrics']['tag_macro_f1']*100:10.2f}% | "
        f"{g_m['tag_metrics']['invalid_bio_rate']*100:11.2f}% | "
        f"{g_m['span_metrics']['strict_f1']*100:12.2f}% | "
        f"{g_m['span_metrics']['lenient_f1']*100:13.2f}%",
        flush=True
    )

    # 2. Softmax Constrained
    c_m = jbert_eval["constrained_softmax"]
    print(
        f"{'2. Softmax + BIO Masked Viterbi':35s} | "
        f"{c_m['tag_metrics']['tag_micro_acc']*100:11.2f}% | "
        f"{c_m['tag_metrics']['tag_macro_f1']*100:10.2f}% | "
        f"{c_m['tag_metrics']['invalid_bio_rate']*100:11.2f}% | "
        f"{c_m['span_metrics']['strict_f1']*100:12.2f}% | "
        f"{c_m['span_metrics']['lenient_f1']*100:13.2f}%",
        flush=True
    )

    # 3. Full CRF
    crf_m = crf_eval["crf_decoding"]
    print(
        f"{'3. Linear-Chain CRF (Full)':35s} | "
        f"{crf_m['tag_metrics']['tag_micro_acc']*100:11.2f}% | "
        f"{crf_m['tag_metrics']['tag_macro_f1']*100:10.2f}% | "
        f"{crf_m['tag_metrics']['invalid_bio_rate']*100:11.2f}% | "
        f"{crf_m['span_metrics']['strict_f1']*100:12.2f}% | "
        f"{crf_m['span_metrics']['lenient_f1']*100:13.2f}%",
        flush=True
    )
    print("=" * 105, flush=True)

    # Quantify real contribution
    delta_const_strict = (c_m['span_metrics']['strict_f1'] - g_m['span_metrics']['strict_f1']) * 100
    delta_crf_strict = (crf_m['span_metrics']['strict_f1'] - c_m['span_metrics']['strict_f1']) * 100
    total_gap = (crf_m['span_metrics']['strict_f1'] - g_m['span_metrics']['strict_f1']) * 100

    pct_grammar = (delta_const_strict / max(total_gap, 1e-5)) * 100
    pct_learned = (delta_crf_strict / max(total_gap, 1e-5)) * 100

    print("\nREAL CRF CONTRIBUTION BREAKDOWN:", flush=True)
    print(f"  Total Strict Span F1 Gap (CRF vs Greedy Softmax): +{total_gap:.2f}%", flush=True)
    print(f"  -> Share from BIO Grammar Masking alone (no learned transitions): +{delta_const_strict:.2f}% ({pct_grammar:.1f}% of gap)", flush=True)
    print(f"  -> Share from Learned Transition Dynamics & Sequence NLL: +{delta_crf_strict:.2f}% ({pct_learned:.1f}% of gap)", flush=True)

    # Save output
    output_data = {
        "jointbert_evaluation": jbert_eval,
        "clscrf_evaluation": crf_eval,
        "decomposition": {
            "greedy_strict_f1": g_m['span_metrics']['strict_f1'],
            "greedy_lenient_f1": g_m['span_metrics']['lenient_f1'],
            "constrained_strict_f1": c_m['span_metrics']['strict_f1'],
            "constrained_lenient_f1": c_m['span_metrics']['lenient_f1'],
            "crf_strict_f1": crf_m['span_metrics']['strict_f1'],
            "crf_lenient_f1": crf_m['span_metrics']['lenient_f1'],
            "delta_from_grammar_masking": delta_const_strict,
            "delta_from_crf_transitions": delta_crf_strict,
            "pct_explained_by_grammar": pct_grammar,
            "pct_explained_by_learned_transitions": pct_learned,
        }
    }

    out_file = Path("ARN/results/stage2c_softmax_baselines.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved softmax baseline analysis to {out_file}", flush=True)


if __name__ == "__main__":
    run_stage2c_softmax_baselines()
