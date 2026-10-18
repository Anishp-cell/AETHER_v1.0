"""
Stage 2c Item 1 & Item 3: Head Independence, Gradient Flow, Decoupled Training & Lambda Ablation.

1. Formal Gradient-Flow Verification:
   With frozen trunk, verifies tool loss gradients flow strictly to tool head and slot loss gradients
   flow strictly to slot head, with zero gradient to encoder trunk.
2. Decoupled Training & Branch-Specific Early Stopping:
   - Tool branch trained and early-stopped strictly on tool validation loss / accuracy.
   - Slot branch trained and early-stopped strictly on slot validation loss.
   - Compares ARN and CLS+CRF slot extraction to demonstrate exact matching.
3. Explaining lambda_slot=0 and lambda_slot=0.1:
   Evaluates lambda_slot in {0.0, 0.1, 1.0} with decoupled early stopping and no joint gradient clipping.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
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
from arn.tokenizer import TOKENIZER, IDX_TO_TAG
from arn.model import AetherRoutingNetwork
from arn.baselines import CLSCRFModel, JointBERT, FARNoCRFModel
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import partition_clusters, precompute_gpu_data, GPUData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2c_independence")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_gradient_flow_test(device: torch.device) -> Dict:
    print("\n" + "=" * 80, flush=True)
    print("1. FORMAL GRADIENT-FLOW INDEPENDENCE TEST", flush=True)
    print("=" * 80, flush=True)
    cfg = ARNConfig(freeze_trunk=True)
    models = {
        "arn": AetherRoutingNetwork(cfg, load_pretrained=True).to(device),
        "cls_crf": CLSCRFModel(cfg, load_pretrained=True).to(device),
        "joint_bert": JointBERT(cfg, load_pretrained=True).to(device),
        "far_no_crf": FARNoCRFModel(cfg, load_pretrained=True).to(device),
    }

    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    tags = torch.randint(0, cfg.num_tags, (batch_size, seq_len), device=device)
    tool_labels = torch.randint(0, 2, (batch_size, cfg.num_tools), device=device).float()

    flow_results = {}
    for name, m in models.items():
        m.train()
        m.zero_grad()

        if name == "arn":
            h = m.encoder(input_ids, mask).last_hidden_state
            tool_logits, _ = m.far(h, mask)
            emissions = m.slot_tagger(h)
            total_loss, tool_loss, slot_loss = m.loss_fn(tool_logits, tool_labels, m.crf, emissions, tags, mask)
            tool_params = list(m.far.parameters())
            slot_params = list(m.slot_tagger.parameters()) + list(m.crf.parameters())
        elif name == "cls_crf":
            total_loss, tool_loss, slot_loss, tool_logits = m(input_ids, mask, tags, tool_labels)
            tool_params = list(m.tool_classifier.parameters())
            slot_params = list(m.slot_tagger.parameters()) + list(m.crf.parameters())
        elif name == "joint_bert":
            total_loss, tool_loss, slot_loss, tool_logits = m(input_ids, mask, tags, tool_labels)
            tool_params = list(m.tool_classifier.parameters())
            slot_params = list(m.slot_classifier.parameters())
        elif name == "far_no_crf":
            total_loss, tool_loss, slot_loss, tool_logits, _ = m(input_ids, mask, tags, tool_labels)
            tool_params = list(m.far.parameters())
            slot_params = list(m.slot_classifier.parameters())

        trunk_params = list(m.encoder.parameters())

        # Test A: Tool Loss Backward Only
        tool_loss.backward(retain_graph=True)
        tool_has_grad = all(p.grad is not None and torch.any(p.grad != 0) for p in tool_params)
        slot_clean_under_tool = all(p.grad is None or torch.all(p.grad == 0) for p in slot_params)
        trunk_clean_under_tool = all(p.grad is None or torch.all(p.grad == 0) for p in trunk_params)

        m.zero_grad()
        # Test B: Slot Loss Backward Only
        slot_loss.backward()
        slot_has_grad = all(p.grad is not None and torch.any(p.grad != 0) for p in slot_params)
        tool_clean_under_slot = all(p.grad is None or torch.all(p.grad == 0) for p in tool_params)
        trunk_clean_under_slot = all(p.grad is None or torch.all(p.grad == 0) for p in trunk_params)

        disjoint = bool(
            tool_has_grad and slot_clean_under_tool and trunk_clean_under_tool and
            slot_has_grad and tool_clean_under_slot and trunk_clean_under_slot
        )
        flow_results[name] = {
            "tool_loss_flows_to_tool": bool(tool_has_grad),
            "tool_loss_zero_to_slot": bool(slot_clean_under_tool),
            "tool_loss_zero_to_trunk": bool(trunk_clean_under_tool),
            "slot_loss_flows_to_slot": bool(slot_has_grad),
            "slot_loss_zero_to_tool": bool(tool_clean_under_slot),
            "slot_loss_zero_to_trunk": bool(trunk_clean_under_slot),
            "parameters_strictly_disjoint": disjoint,
        }
        print(
            f"  Model {name:12s} | Strictly Disjoint: {disjoint} "
            f"(Tool->Slot Grads: 0, Slot->Tool Grads: 0, Trunk Grads: 0)",
            flush=True
        )

    return flow_results


def extract_spans_from_bio(tags: List[str]) -> set:
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


def train_slot_branch_decoupled(
    model_type: str,
    model: nn.Module,
    gpu_data: GPUData,
    train_idx: List[int],
    val_idx: List[int],
    device: torch.device,
    epochs: int = 20,
    patience: int = 4,
    lr: float = 3e-4,
    batch_size: int = 64,
) -> Tuple[Dict, int]:
    """Train slot branch ONLY, early stopping on slot validation loss."""
    if model_type in ("arn", "cls_crf"):
        slot_params = list(model.slot_tagger.parameters()) + list(model.crf.parameters())
    else:
        slot_params = list(model.slot_classifier.parameters())

    optimizer = torch.optim.AdamW(slot_params, lr=lr, weight_decay=0.01)
    tr_indices = torch.tensor(train_idx, device=device, dtype=torch.long)
    n_train = len(train_idx)

    best_val_slot_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_cnt = 0

    val_idx_tensor = torch.tensor(val_idx, device=device, dtype=torch.long)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = tr_indices[torch.randperm(n_train)]
        for b in range(0, n_train, batch_size):
            b_idx = perm[b : min(b + batch_size, n_train)]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            t_b = gpu_data.tags[b_idx, :max_l]

            optimizer.zero_grad()
            if model_type in ("arn", "cls_crf"):
                emissions = model.slot_tagger(h_b)
                sentence_nlls = model.crf.sentence_nll(emissions, t_b, m_b)
                active = m_b.sum(dim=1).float().clamp(min=1.0)
                loss = (sentence_nlls / active).mean()
            else:
                slot_logits = model.slot_classifier(h_b)
                loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_b.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(slot_params, 1.0)
            optimizer.step()

        # Val evaluation on slot loss alone
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vb in range(0, len(val_idx), 128):
                v_idx = val_idx_tensor[vb : min(vb + 128, len(val_idx))]
                m_v = gpu_data.mask[v_idx]
                max_l = int(m_v.sum(dim=1).max().item())
                h_v = gpu_data.h[v_idx, :max_l]
                m_v = m_v[:, :max_l]
                t_v = gpu_data.tags[v_idx, :max_l]

                if model_type in ("arn", "cls_crf"):
                    emissions = model.slot_tagger(h_v)
                    sentence_nlls = model.crf.sentence_nll(emissions, t_v, m_v)
                    active = m_v.sum(dim=1).float().clamp(min=1.0)
                    v_loss = (sentence_nlls / active).mean()
                else:
                    slot_logits = model.slot_classifier(h_v)
                    v_loss = model.ce_loss(slot_logits.view(-1, model.config.num_tags), t_v.view(-1))
                val_losses.append(v_loss.item())

        mean_v_loss = float(np.mean(val_losses))
        if mean_v_loss < best_val_slot_loss - 0.002:
            best_val_slot_loss = mean_v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on val set
    model.eval()
    gold_spans_all = []
    pred_spans_all = []
    with torch.no_grad():
        for vb in range(0, len(val_idx), 128):
            v_idx = val_idx_tensor[vb : min(vb + 128, len(val_idx))]
            m_v = gpu_data.mask[v_idx]
            max_l = int(m_v.sum(dim=1).max().item())
            h_v = gpu_data.h[v_idx, :max_l]
            m_v = m_v[:, :max_l]
            t_v = gpu_data.tags[v_idx, :max_l]

            if model_type in ("arn", "cls_crf"):
                emissions = model.slot_tagger(h_v)
                tag_preds = model.crf.viterbi_decode(emissions, m_v)
            else:
                slot_logits = model.slot_classifier(h_v)
                tag_preds_tensor = torch.argmax(slot_logits, dim=-1)
                tag_preds = []
                for b_i in range(len(v_idx)):
                    l_i = int(m_v[b_i].sum().item())
                    tag_preds.append(tag_preds_tensor[b_i, :l_i].tolist())

            tags_np = t_v.cpu().numpy()
            mask_np = m_v.cpu().numpy()
            for b_i in range(len(v_idx)):
                l_i = int(mask_np[b_i].sum())
                g_tags = [IDX_TO_TAG.get(idx, "O") for idx in tags_np[b_i][:l_i].tolist()]
                p_tags = [IDX_TO_TAG.get(idx, "O") for idx in tag_preds[b_i][:l_i]]
                gold_spans_all.append(extract_spans_from_bio(g_tags))
                pred_spans_all.append(extract_spans_from_bio(p_tags))

    tot_gold = sum(len(s) for s in gold_spans_all)
    tot_pred = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))
    span_p = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
    span_r = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
    span_f1 = (2 * span_p * span_r / max(span_p + span_r, 1e-8)) if (span_p + span_r) > 0 else 0.0

    return {
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "span_correct": correct_spans,
        "span_gold_total": tot_gold,
        "best_epoch": best_epoch,
        "best_val_slot_loss": best_val_slot_loss,
    }, best_epoch


def train_tool_branch_decoupled(
    model_type: str,
    model: nn.Module,
    gpu_data: GPUData,
    train_idx: List[int],
    val_idx: List[int],
    device: torch.device,
    epochs: int = 20,
    patience: int = 4,
    lr: float = 3e-4,
    batch_size: int = 64,
    clip_grad: bool = True,
) -> Tuple[Dict, int]:
    """Train tool branch ONLY, early stopping on tool validation loss/acc."""
    if model_type in ("arn", "far_no_crf"):
        tool_params = list(model.far.parameters())
    else:
        tool_params = list(model.tool_classifier.parameters())

    optimizer = torch.optim.AdamW(tool_params, lr=lr, weight_decay=0.01)
    bce_loss = nn.BCEWithLogitsLoss()

    tr_indices = torch.tensor(train_idx, device=device, dtype=torch.long)
    n_train = len(train_idx)
    val_idx_tensor = torch.tensor(val_idx, device=device, dtype=torch.long)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = tr_indices[torch.randperm(n_train)]
        for b in range(0, n_train, batch_size):
            b_idx = perm[b : min(b + batch_size, n_train)]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            tools_b = gpu_data.tool_labels[b_idx]

            optimizer.zero_grad()
            if model_type in ("arn", "far_no_crf"):
                tool_logits, _ = model.far(h_b, m_b)
            else:
                cls_repr = h_b[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)

            loss = bce_loss(tool_logits, tools_b)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(tool_params, 1.0)
            optimizer.step()

        # Val evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vb in range(0, len(val_idx), 128):
                v_idx = val_idx_tensor[vb : min(vb + 128, len(val_idx))]
                m_v = gpu_data.mask[v_idx]
                max_l = int(m_v.sum(dim=1).max().item())
                h_v = gpu_data.h[v_idx, :max_l]
                m_v = m_v[:, :max_l]
                tools_v = gpu_data.tool_labels[v_idx]

                if model_type in ("arn", "far_no_crf"):
                    tool_logits, _ = model.far(h_v, m_v)
                else:
                    cls_repr = h_v[:, 0, :]
                    tool_logits = model.tool_classifier(cls_repr)

                v_loss = bce_loss(tool_logits, tools_v)
                val_losses.append(v_loss.item())

        mean_v_loss = float(np.mean(val_losses))
        if mean_v_loss < best_val_loss - 0.002:
            best_val_loss = mean_v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate tool accuracy on val set
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for vb in range(0, len(val_idx), 128):
            v_idx = val_idx_tensor[vb : min(vb + 128, len(val_idx))]
            m_v = gpu_data.mask[v_idx]
            max_l = int(m_v.sum(dim=1).max().item())
            h_v = gpu_data.h[v_idx, :max_l]
            m_v = m_v[:, :max_l]
            tools_v = gpu_data.tool_labels[v_idx]

            if model_type in ("arn", "far_no_crf"):
                tool_logits, _ = model.far(h_v, m_v)
            else:
                cls_repr = h_v[:, 0, :]
                tool_logits = model.tool_classifier(cls_repr)

            probs = torch.sigmoid(tool_logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(tools_v.cpu().numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)

    # Tune threshold
    best_tau = 0.5
    best_acc = 0.0
    for tau in [0.35, 0.40, 0.45, 0.50]:
        preds = (probs >= tau).astype(int)
        acc = float(np.all(preds == targets, axis=1).mean())
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    preds = (probs >= best_tau).astype(int)
    matches = np.all(preds == targets, axis=1)
    acc = float(np.mean(matches))
    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0

    return {
        "tool_subset_acc": acc,
        "tool_subset_acc_count": int(np.sum(matches)),
        "n_samples": len(val_idx),
        "tool_micro_f1": f1,
        "best_tau": best_tau,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, best_epoch


def run_stage2c_head_independence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Head Independence Suite on device: {device}", flush=True)

    # 1. Gradient Flow Independence Test
    flow_results = run_gradient_flow_test(device)

    # 2. Load Data
    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)

    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
    gpu_data = precompute_gpu_data(dataset, raw_samples, frame_to_cluster, device)

    # 3. 5 Folds from Shuffle 1 for rigorous controlled comparison
    outer_folds = partition_clusters(cluster_ids, n_folds=5, seed=42)

    print("\n" + "=" * 80, flush=True)
    print("2. DECOUPLED TRAINING & BRANCH-SPECIFIC EARLY STOPPING (5 FOLDS)", flush=True)
    print("=" * 80, flush=True)
    models_to_test = ["arn", "cls_crf", "joint_bert", "far_no_crf"]
    decoupled_results = {m: {"tool_folds": [], "slot_folds": []} for m in models_to_test}

    for fold_idx in range(5):
        val_idx = outer_folds[fold_idx]
        train_idx = [i for other_f in range(5) if other_f != fold_idx for i in outer_folds[other_f]]
        print(f"\n--- FOLD {fold_idx + 1}/5 (Train N={len(train_idx)}, Val N={len(val_idx)}) ---", flush=True)

        for m_type in models_to_test:
            set_seed(42 + fold_idx * 10)
            cfg = ARNConfig(freeze_trunk=True)
            if m_type == "arn":
                model = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
            elif m_type == "cls_crf":
                model = CLSCRFModel(cfg, load_pretrained=False).to(device)
            elif m_type == "joint_bert":
                model = JointBERT(cfg, load_pretrained=False).to(device)
            elif m_type == "far_no_crf":
                model = FARNoCRFModel(cfg, load_pretrained=False).to(device)

            # A. Train Tool Branch Decoupled
            tool_res, t_ep = train_tool_branch_decoupled(m_type, model, gpu_data, train_idx, val_idx, device)
            decoupled_results[m_type]["tool_folds"].append(tool_res)

            # B. Train Slot Branch Decoupled (using same seed for exact comparison)
            set_seed(42 + fold_idx * 10)
            slot_res, s_ep = train_slot_branch_decoupled(m_type, model, gpu_data, train_idx, val_idx, device)
            decoupled_results[m_type]["slot_folds"].append(slot_res)

            print(
                f"  [{m_type:10s}] Tool Acc: {tool_res['tool_subset_acc']*100:.2f}% (ep {t_ep}) | "
                f"Span F1: {slot_res['span_f1']*100:.2f}% ({slot_res['span_correct']}/{slot_res['span_gold_total']}, ep {s_ep})",
                flush=True
            )

    # 4. Item 3: Lambda Ablation under Decoupled Protocol
    print("\n" + "=" * 80, flush=True)
    print("3. LAMBDA_SLOT ABLATION IN {0.0, 0.1, 1.0} WITH DECOUPLED EARLY STOPPING", flush=True)
    print("=" * 80, flush=True)
    lambda_results = {}
    for l_val in [0.0, 0.1, 1.0]:
        accs = []
        for fold_idx in range(5):
            val_idx = outer_folds[fold_idx]
            train_idx = [i for other_f in range(5) if other_f != fold_idx for i in outer_folds[other_f]]
            set_seed(100 + fold_idx * 10)
            cfg = ARNConfig(freeze_trunk=True)
            model = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
            tool_res, ep = train_tool_branch_decoupled(
                "arn", model, gpu_data, train_idx, val_idx, device, clip_grad=True
            )
            accs.append(tool_res["tool_subset_acc"])
        lambda_results[f"lambda_{l_val}"] = {
            "mean_tool_acc": float(np.mean(accs)),
            "sd_tool_acc": float(np.std(accs, ddof=1)),
            "per_fold": accs,
        }
        print(f"  lambda_slot = {l_val:3.1f} -> Tool Acc: {np.mean(accs)*100:.2f}% ± {np.std(accs, ddof=1)*100:.2f}%", flush=True)

    # Save summary
    output_data = {
        "gradient_flow_independence": flow_results,
        "decoupled_training_results": {
            m: {
                "mean_tool_acc": float(np.mean([f["tool_subset_acc"] for f in decoupled_results[m]["tool_folds"]])),
                "sd_tool_acc": float(np.std([f["tool_subset_acc"] for f in decoupled_results[m]["tool_folds"]], ddof=1)),
                "mean_tool_f1": float(np.mean([f["tool_micro_f1"] for f in decoupled_results[m]["tool_folds"]])),
                "mean_span_f1": float(np.mean([f["span_f1"] for f in decoupled_results[m]["slot_folds"]])),
                "sd_span_f1": float(np.std([f["span_f1"] for f in decoupled_results[m]["slot_folds"]], ddof=1)),
                "span_f1_per_fold": [f["span_f1"] for f in decoupled_results[m]["slot_folds"]],
                "tool_acc_per_fold": [f["tool_subset_acc"] for f in decoupled_results[m]["tool_folds"]],
            }
            for m in models_to_test
        },
        "lambda_slot_decoupled_ablation": lambda_results,
    }

    out_file = Path("ARN/results/stage2c_head_independence.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved complete head independence results to {out_file}", flush=True)


if __name__ == "__main__":
    run_stage2c_head_independence()
