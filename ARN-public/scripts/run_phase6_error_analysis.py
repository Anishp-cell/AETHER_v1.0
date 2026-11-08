"""
ARN Phase 6 Error Analysis Runner.

Performs detailed error diagnosis on the selected model (`unfrozen_cls_crf`)
and baseline (`frozen_arn`) using outer validation fold predictions:
  1. Per-tool classification metrics (Precision, Recall, F1, TP, FP, FN, Support).
  2. Failure analysis by cluster: Top 10 hardest clusters with representative error cases.
  3. Slot error taxonomy:
     - Exact span matches
     - Boundary errors (lenient overlap, boundary mismatch)
     - Label type mismatches (exact boundary, wrong slot type)
     - False positive spans (hallucinated spans)
     - False negative spans (completely missed gold spans)
  4. Saves output to results/error_analysis.json.
"""

from collections import Counter, defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from arn.baselines import CLSCRFModel
from scripts.run_master_protocol import (
    extract_spans_strict,
    compute_span_f1,
    MasterProtocolExperimentRunner,
    get_git_hash,
)
from scripts.make_splits import extract_fine_semantic_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.phase6_error_analysis")


def evaluate_detailed_errors(
    runner: MasterProtocolExperimentRunner
) -> Dict[str, Any]:
    """Run full out-of-fold inference on shuffle_1 for unfrozen_cls_crf to extract error taxonomy."""
    sh_folds = runner.manifest["folds"]["shuffle_1"]
    raw_samples = [json.loads(line) for line in open(runner.dataset.path, "r", encoding="utf-8") if line.strip()]

    num_tools = len(TOOL_TO_IDX)
    num_tags = len(TAG_TO_IDX)

    tool_counts = {t: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for t in TOOL_TO_IDX.keys()}
    slot_taxonomy = Counter({
        "exact_match": 0,
        "boundary_error": 0,
        "label_mismatch": 0,
        "false_positive": 0,
        "false_negative": 0,
    })

    cluster_stats = defaultdict(lambda: {"total": 0, "tool_correct": 0, "joint_correct": 0, "samples": []})

    # Run outer evaluation across the 5 folds of shuffle_1
    for fold_name, f_data in sh_folds.items():
        logger.info(f"Analyzing errors on {fold_name}...")
        in_tr_idx = f_data["inner_train_indices"]
        in_val_idx = f_data["inner_val_indices"]
        val_idx = f_data["outer_val_indices"]
        N_val = len(val_idx)

        # Train unfrozen CLS+CRF model
        fold_seed = 420 + int(fold_name.split("_")[1])
        model, best_tau = runner.train_unfrozen_model("unfrozen_cls_crf", in_tr_idx, in_val_idx, seed=fold_seed)
        model.eval()

        val_tensor = torch.tensor(val_idx, device=runner.device, dtype=torch.long)
        gold_tools_np = runner.gpu_data.tool_labels[val_tensor].cpu().numpy()
        gold_tags_np = runner.gpu_data.tags[val_tensor].cpu().numpy()
        mask_np = runner.gpu_data.mask[val_tensor].cpu().numpy()
        val_data = [runner.dataset.encoded[i] for i in val_idx]

        pred_tool_probs = []
        pred_tag_seqs: List[List[str]] = []

        with torch.no_grad():
            for b in range(0, N_val, 128):
                batch = val_data[b : min(b + 128, N_val)]
                max_l = min(max(item["attention_mask"].sum().item() for item in batch), 64)
                ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(runner.device)
                m_b = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(runner.device)

                t_logits, emissions = model(ids, m_b)
                pred_tool_probs.append(torch.sigmoid(t_logits).cpu().numpy())
                tag_ids = model.crf.viterbi_decode(emissions, m_b)
                for j, ids_row in enumerate(tag_ids):
                    l_j = int(m_b[j].sum().item())
                    pred_tag_seqs.append([IDX_TO_TAG.get(x, "O") for x in ids_row[:l_j]])

        pred_tool_probs = np.vstack(pred_tool_probs)
        pred_tools = (pred_tool_probs >= best_tau).astype(np.float32)

        for i in range(N_val):
            sample_idx = val_idx[i]
            raw_s = raw_samples[sample_idx]
            c_id = extract_fine_semantic_frame(raw_s)

            # Per-tool tracking
            g_t = gold_tools_np[i]
            p_t = pred_tools[i]
            is_tool_correct = bool(np.all(g_t == p_t))

            for t_name, t_idx in TOOL_TO_IDX.items():
                g_active = bool(g_t[t_idx] == 1.0)
                p_active = bool(p_t[t_idx] == 1.0)
                if g_active:
                    tool_counts[t_name]["support"] += 1
                if g_active and p_active:
                    tool_counts[t_name]["tp"] += 1
                elif not g_active and p_active:
                    tool_counts[t_name]["fp"] += 1
                elif g_active and not p_active:
                    tool_counts[t_name]["fn"] += 1

            # Spans extraction
            l_i = int(mask_np[i].sum())
            gold_seq = [IDX_TO_TAG.get(x, "O") for x in gold_tags_np[i][:l_i]]
            pred_seq = pred_tag_seqs[i]

            gold_spans = extract_spans_strict(gold_seq)
            pred_spans = extract_spans_strict(pred_seq)

            is_slot_correct = bool(gold_spans == pred_spans)
            is_joint_correct = bool(is_tool_correct and is_slot_correct)

            # Cluster tracking
            cluster_stats[c_id]["total"] += 1
            if is_tool_correct:
                cluster_stats[c_id]["tool_correct"] += 1
            if is_joint_correct:
                cluster_stats[c_id]["joint_correct"] += 1

            if not is_joint_correct and len(cluster_stats[c_id]["samples"]) < 3:
                cluster_stats[c_id]["samples"].append({
                    "sample_idx": sample_idx,
                    "text": raw_s["messages"][0]["content"] if raw_s.get("messages") else "",
                    "gold_tools": [IDX_TO_TOOL[k] for k, v in enumerate(g_t) if v == 1.0],
                    "pred_tools": [IDX_TO_TOOL[k] for k, v in enumerate(p_t) if v == 1.0],
                    "gold_spans": [f"{s[0]}:{s[1]}-{s[2]}" for s in gold_spans],
                    "pred_spans": [f"{s[0]}:{s[1]}-{s[2]}" for s in pred_spans],
                })

            # Slot error taxonomy categorization
            # Match each gold span to predicted spans
            matched_preds = set()
            for g_slot, g_s, g_e in gold_spans:
                matched = False
                for p_idx, (p_slot, p_s, p_e) in enumerate(pred_spans):
                    if g_s == p_s and g_e == p_e:
                        if g_slot == p_slot:
                            slot_taxonomy["exact_match"] += 1
                        else:
                            slot_taxonomy["label_mismatch"] += 1
                        matched_preds.add(p_idx)
                        matched = True
                        break
                    elif max(g_s, p_s) < min(g_e, p_e):
                        slot_taxonomy["boundary_error"] += 1
                        matched_preds.add(p_idx)
                        matched = True
                        break
                if not matched:
                    slot_taxonomy["false_negative"] += 1

            for p_idx in range(len(pred_spans)):
                if p_idx not in matched_preds:
                    slot_taxonomy["false_positive"] += 1

    # Compute per-tool metrics
    tool_performance = {}
    for t_name, counts in tool_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        supp = counts["support"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / max(prec + rec, 1e-8)) if (prec + rec) > 0 else 0.0
        tool_performance[t_name] = {
            "support": supp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    # Hardest clusters (rank by lowest Joint EM with min support >= 5)
    ranked_clusters = []
    for c_id, stats in cluster_stats.items():
        if stats["total"] >= 5:
            tool_acc = stats["tool_correct"] / stats["total"]
            joint_em = stats["joint_correct"] / stats["total"]
            ranked_clusters.append({
                "cluster_id": c_id,
                "n_samples": stats["total"],
                "tool_acc": float(tool_acc),
                "joint_em": float(joint_em),
                "failure_examples": stats["samples"],
            })

    ranked_clusters.sort(key=lambda x: (x["joint_em"], x["tool_acc"]))

    total_slot_events = sum(slot_taxonomy.values())
    slot_shares = {k: float(v / max(total_slot_events, 1)) for k, v in slot_taxonomy.items()}

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": "unfrozen_cls_crf",
        "num_val_samples_analyzed": len(raw_samples),
        "per_tool_performance": tool_performance,
        "slot_error_taxonomy": {
            "counts": dict(slot_taxonomy),
            "shares": slot_shares,
            "total_spans_evaluated": total_slot_events,
        },
        "top_10_hardest_clusters": ranked_clusters[:10],
    }


def main():
    repo_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Phase 6 Error Analysis on {device}...")

    runner = MasterProtocolExperimentRunner(device)
    error_analysis = evaluate_detailed_errors(runner)

    out_path = repo_dir / "results" / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    logger.info(f"Phase 6 Error Analysis Complete! Saved report to {out_path}")


if __name__ == "__main__":
    main()
