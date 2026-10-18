"""
Evaluate Single-Tool vs Multi-Tool samples performance.
Reports Tool Subset Accuracy, Tool Micro F1, Span F1, and Joint EM separately.
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TAG
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data
from scripts.stage2c_head_independence import extract_spans_from_bio


def evaluate_single_vs_multitool():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running single vs multi-tool evaluation on {device}...", flush=True)

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)
    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    gpu_data = precompute_gpu_data(dataset, raw_samples, frame_to_cluster, device)

    # Identify single vs multi tool samples
    single_indices = []
    multi_indices = []
    for idx, sample in enumerate(raw_samples):
        calls = []
        for msg in sample.get("messages", []):
            if msg["role"] == "assistant":
                content = msg.get("content", "").strip()
                if content.startswith("["):
                    try:
                        p = json.loads(content)
                        if isinstance(p, list):
                            calls.extend(p)
                    except Exception:
                        pass
                elif "tool_calls" in msg:
                    calls.extend(msg["tool_calls"])
        if len(calls) == 1:
            single_indices.append(idx)
        elif len(calls) > 1:
            multi_indices.append(idx)

    print(f"Total: {len(raw_samples)} | Single: {len(single_indices)} | Multi: {len(multi_indices)}", flush=True)

    # We evaluate CLS+CRF, ARN, JointBERT on the dataset
    # Train each model on full train set with early stopping or standard epochs
    from arn.model import AetherRoutingNetwork
    from arn.baselines import CLSCRFModel, JointBERT

    models = {
        "arn": AetherRoutingNetwork(ARNConfig(freeze_trunk=True), load_pretrained=False).to(device),
        "cls_crf": CLSCRFModel(ARNConfig(freeze_trunk=True), load_pretrained=False).to(device),
        "joint_bert": JointBERT(ARNConfig(freeze_trunk=True), load_pretrained=False).to(device),
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---", flush=True)
        # Train tool and slot
        if name in ("arn", "cls_crf"):
            # Tool params
            t_params = list(model.far.parameters()) if hasattr(model, "far") else list(model.tool_classifier.parameters())
            s_params = list(model.slot_tagger.parameters()) + list(model.crf.parameters())
        else:
            t_params = list(model.tool_classifier.parameters())
            s_params = list(model.slot_classifier.parameters())

        t_opt = torch.optim.AdamW(t_params, lr=3e-4, weight_decay=0.01)
        s_opt = torch.optim.AdamW(s_params, lr=1e-3, weight_decay=0.01)
        bce = nn.BCEWithLogitsLoss()

        model.train()
        n_samples = len(raw_samples)
        for ep in range(1, 11):
            perm = torch.randperm(n_samples)
            for b in range(0, n_samples, 64):
                b_idx = perm[b : min(b + 64, n_samples)]
                m_b = gpu_data.mask[b_idx]
                max_l = int(m_b.sum(dim=1).max().item())
                h_b = gpu_data.h[b_idx, :max_l]
                m_b = m_b[:, :max_l]
                t_b = gpu_data.tags[b_idx, :max_l]
                tools_b = gpu_data.tool_labels[b_idx]

                t_opt.zero_grad()
                if hasattr(model, "far"):
                    t_logits, _ = model.far(h_b, m_b)
                else:
                    t_logits = model.tool_classifier(h_b[:, 0, :])
                t_loss = bce(t_logits, tools_b)
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(t_params, 1.0)
                t_opt.step()

                s_opt.zero_grad()
                if hasattr(model, "crf"):
                    emissions = model.slot_tagger(h_b)
                    nlls = model.crf.sentence_nll(emissions, t_b, m_b)
                    active = m_b.sum(dim=1).float().clamp(min=1.0)
                    s_loss = (nlls / active).mean()
                else:
                    s_logits = model.slot_classifier(h_b)
                    s_loss = model.ce_loss(s_logits.view(-1, model.config.num_tags), t_b.view(-1))
                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(s_params, 1.0)
                s_opt.step()

        # Evaluate separately on Single and Multi
        model.eval()
        with torch.no_grad():
            # Get predictions for all samples
            all_tool_preds = []
            all_slot_preds = []
            for b in range(0, n_samples, 128):
                b_idx = torch.arange(b, min(b + 128, n_samples), device=device)
                m_b = gpu_data.mask[b_idx]
                max_l = int(m_b.sum(dim=1).max().item())
                h_b = gpu_data.h[b_idx, :max_l]
                m_b = m_b[:, :max_l]

                if hasattr(model, "far"):
                    t_logits, _ = model.far(h_b, m_b)
                else:
                    t_logits = model.tool_classifier(h_b[:, 0, :])
                probs = torch.sigmoid(t_logits).cpu().numpy()
                all_tool_preds.append((probs >= 0.5).astype(int))

                if hasattr(model, "crf"):
                    emissions = model.slot_tagger(h_b)
                    tags = model.crf.viterbi_decode(emissions, m_b)
                else:
                    s_logits = model.slot_classifier(h_b)
                    pred_tags = torch.argmax(s_logits, dim=-1)
                    tags = []
                    for i in range(len(b_idx)):
                        l_i = int(m_b[i].sum().item())
                        tags.append(pred_tags[i, :l_i].tolist())
                all_slot_preds.extend(tags)

            all_tool_preds = np.vstack(all_tool_preds)
            gold_tools = gpu_data.tool_labels.cpu().numpy()
            gold_tags = gpu_data.tags.cpu().numpy()
            masks = gpu_data.mask.cpu().numpy()

            def eval_subset(indices, subset_name):
                t_matches = np.all(all_tool_preds[indices] == gold_tools[indices], axis=1)
                t_acc = float(np.mean(t_matches))

                pred_pos = all_tool_preds[indices] == 1
                gold_pos = gold_tools[indices] == 1
                tp = int(np.sum(pred_pos & gold_pos))
                fp = int(np.sum(pred_pos & (~gold_pos)))
                fn = int(np.sum((~pred_pos) & gold_pos))
                p = tp / max(tp + fp, 1)
                r = tp / max(tp + fn, 1)
                t_f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0

                # Slot metrics
                gold_spans_list = []
                pred_spans_list = []
                for idx in indices:
                    l_i = int(masks[idx].sum())
                    g_t = [IDX_TO_TAG.get(x, "O") for x in gold_tags[idx, :l_i]]
                    p_t = [IDX_TO_TAG.get(x, "O") for x in all_slot_preds[idx][:l_i]]
                    gold_spans_list.append(extract_spans_from_bio(g_t))
                    pred_spans_list.append(extract_spans_from_bio(p_t))

                tot_gold_spans = sum(len(s) for s in gold_spans_list)
                tot_pred_spans = sum(len(s) for s in pred_spans_list)
                corr_spans = sum(len(g & p) for g, p in zip(gold_spans_list, pred_spans_list))

                s_p = corr_spans / max(tot_pred_spans, 1) if tot_pred_spans > 0 else 0.0
                s_r = corr_spans / max(tot_gold_spans, 1) if tot_gold_spans > 0 else 0.0
                s_f1 = (2 * s_p * s_r / max(s_p + s_r, 1e-8)) if (s_p + s_r) > 0 else 0.0

                # Joint EM: tool exact match AND all slot spans exact match
                joint_em_matches = 0
                for i_pos, idx in enumerate(indices):
                    tool_ok = t_matches[i_pos]
                    slot_ok = (gold_spans_list[i_pos] == pred_spans_list[i_pos])
                    if tool_ok and slot_ok:
                        joint_em_matches += 1
                joint_em = joint_em_matches / len(indices)

                print(f"  [{name:10s}] {subset_name:12s} (N={len(indices)}): Tool Acc={t_acc*100:5.2f}%, Tool F1={t_f1*100:5.2f}%, Span F1={s_f1*100:5.2f}%, Joint EM={joint_em*100:5.2f}%", flush=True)

                return {
                    "n_samples": len(indices),
                    "tool_subset_acc": t_acc,
                    "tool_subset_acc_count": int(np.sum(t_matches)),
                    "tool_micro_f1": t_f1,
                    "tool_tp": tp,
                    "tool_fp": fp,
                    "tool_fn": fn,
                    "span_f1": s_f1,
                    "span_correct": corr_spans,
                    "span_gold_total": tot_gold_spans,
                    "joint_em": joint_em,
                    "joint_em_count": joint_em_matches,
                }

            results[name] = {
                "single_tool": eval_subset(single_indices, "Single-Tool"),
                "multi_tool": eval_subset(multi_indices, "Multi-Tool"),
            }

    # Update stage2c_multitool_decomposition.json
    out_file = Path("ARN/results/stage2c_multitool_decomposition.json")
    with open(out_file, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing["model_performance_by_tool_count"] = results
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\nSuccessfully updated {out_file} with single-tool vs multi-tool performance breakdowns!", flush=True)


if __name__ == "__main__":
    evaluate_single_vs_multitool()
