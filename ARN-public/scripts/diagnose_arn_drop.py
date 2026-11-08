"""
Diagnose the ARN tool subset accuracy drop from 42.5% (Stage 1b) to 24.6% (Stage 2).

Runs controlled variants under the Stage 2 protocol (Shuffle 1, 5 folds on group_disjoint_train.jsonl):
1. Stage 2 Default Protocol: max_epochs=25, patience=4, batch_size=64, lambda_slot=1.0, nested inner train (2,512 samples)
2. Variant (a): Old Budget: max_epochs=100, patience=10, batch_size=32, lambda_slot=1.0, nested inner train
3. Variant (b): Tool Loss Only (no slot loss): lambda_slot=0.0 (max_epochs=100, patience=10, batch_size=32)
4. Variant (c): Tool Loss Weight Raised: lambda_slot=0.1 (max_epochs=100, patience=10, batch_size=32)
5. Variant (d): Outer-train directly (no inner split): 3,141 train samples, max_epochs=100, patience=10 (Stage 1b style on Stage 2 data)
6. Variant (e): Stage 1b pool (5,000 samples, all 253 clusters): 4,000 train samples, max_epochs=100, patience=10, lambda_slot=1.0

Reports exactly which factor accounts for the gap!
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
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
from arn.tokenizer import TOKENIZER, IDX_TO_TAG, TOOL_TO_IDX
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data, partition_clusters, evaluate_model, GPUData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.diagnose_drop")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_variant(
    gpu_data: GPUData,
    train_indices: List[int],
    val_indices: List[int],
    eval_indices: List[int],
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lambda_slot: float,
    lr: float = 3e-4,
    seed: int = 42,
) -> Dict[str, float]:
    set_seed(seed)
    cfg = ARNConfig(learning_rate=lr, lambda_slot=lambda_slot, batch_size=batch_size, epochs=max_epochs)
    model = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    tr_tensor = torch.tensor(train_indices, device=device, dtype=torch.long)
    n_train = len(train_indices)
    best_val_loss = float("inf")
    best_state = None
    patience_cnt = 0
    actual_epochs = 0

    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch
        model.train()
        perm = tr_tensor[torch.randperm(n_train)]
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
            tool_logits, _ = model.far(h_b, m_b)
            emissions = model.slot_tagger(h_b)

            if lambda_slot > 0:
                loss_out = model.loss_fn(tool_logits, tools_b, model.crf, emissions, t_b, m_b)
                loss = loss_out[1] + lambda_slot * loss_out[2]
            else:
                bce_fn = nn.BCEWithLogitsLoss()
                loss = bce_fn(tool_logits, tools_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation loss for early stopping
        model.eval()
        v_idx_tensor = torch.tensor(val_indices, device=device, dtype=torch.long)
        val_loss_sum = 0.0
        v_steps = 0
        with torch.no_grad():
            for vb in range(0, len(val_indices), 128):
                b_idx = v_idx_tensor[vb : vb + 128]
                m_b = gpu_data.mask[b_idx]
                max_l = int(m_b.sum(dim=1).max().item())
                h_b = gpu_data.h[b_idx, :max_l]
                m_b = m_b[:, :max_l]
                t_b = gpu_data.tags[b_idx, :max_l]
                tools_b = gpu_data.tool_labels[b_idx]

                tool_logits, _ = model.far(h_b, m_b)
                if lambda_slot > 0:
                    emissions = model.slot_tagger(h_b)
                    loss_out = model.loss_fn(tool_logits, tools_b, model.crf, emissions, t_b, m_b)
                    v_loss = loss_out[1] + lambda_slot * loss_out[2]
                else:
                    bce_fn = nn.BCEWithLogitsLoss()
                    v_loss = bce_fn(tool_logits, tools_b)
                val_loss_sum += v_loss.item()
                v_steps += 1

        avg_v_loss = val_loss_sum / max(v_steps, 1)

        if avg_v_loss < (best_val_loss - 0.005):
            best_val_loss = avg_v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold selection on val_indices
    model.eval()
    v_idx_tensor = torch.tensor(val_indices, device=device, dtype=torch.long)
    all_v_probs = []
    all_v_targets = []
    with torch.no_grad():
        for vb in range(0, len(val_indices), 128):
            b_idx = v_idx_tensor[vb : vb + 128]
            m_b = gpu_data.mask[b_idx]
            max_l = int(m_b.sum(dim=1).max().item())
            h_b = gpu_data.h[b_idx, :max_l]
            m_b = m_b[:, :max_l]
            tools_b = gpu_data.tool_labels[b_idx]
            tool_logits, _ = model.far(h_b, m_b)
            all_v_probs.append(torch.sigmoid(tool_logits).cpu().numpy())
            all_v_targets.append(tools_b.cpu().numpy())
    all_v_probs = np.vstack(all_v_probs)
    all_v_targets = np.vstack(all_v_targets)

    best_tau = 0.40
    best_acc = -1.0
    for tau in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        acc = float(np.all((all_v_probs >= tau).astype(int) == all_v_targets, axis=1).mean())
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    # Evaluate on eval_indices (outer val)
    eval_res = evaluate_model("arn", model, gpu_data, eval_indices, device, threshold=best_tau)
    eval_res["stopped_epoch"] = actual_epochs
    eval_res["best_tau"] = best_tau
    return eval_res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Controlled Diagnostics on device: {device}")

    # Load 3,926 Stage 2 train dataset
    train_ds = ARNDataset("ARN/data/splits/group_disjoint_train.jsonl")
    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    raw_train_samples = [json.loads(l) for l in open("ARN/data/splits/group_disjoint_train.jsonl", encoding="utf-8") if l.strip()]
    frame_to_cluster = build_canonical_clusters(raw_path)
    stage2_gpu = precompute_gpu_data(train_ds, raw_train_samples, frame_to_cluster, device)

    # Load 5,000 full dataset for Variant (e)
    full_ds = ARNDataset(str(raw_path))
    full_samples = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    full_gpu = precompute_gpu_data(full_ds, full_samples, frame_to_cluster, device)

    # Partition Stage 2 outer folds (Shuffle 1, seed 42)
    outer_folds = partition_clusters(stage2_gpu.cluster_ids, n_folds=5, seed=42)

    variants = [
        {"name": "Stage 2 Baseline", "epochs": 25, "patience": 4, "bs": 64, "lambda_slot": 1.0, "nested": True, "pool": "stage2"},
        {"name": "Variant (a): Old Budget (p=10, cap=100, bs=32)", "epochs": 100, "patience": 10, "bs": 32, "lambda_slot": 1.0, "nested": True, "pool": "stage2"},
        {"name": "Variant (b): Tool Loss Only (lambda_slot=0.0)", "epochs": 100, "patience": 10, "bs": 32, "lambda_slot": 0.0, "nested": True, "pool": "stage2"},
        {"name": "Variant (c): Tool Weight Raised (lambda_slot=0.1)", "epochs": 100, "patience": 10, "bs": 32, "lambda_slot": 0.1, "nested": True, "pool": "stage2"},
        {"name": "Variant (d): Outer-train (no inner split, N=3141)", "epochs": 100, "patience": 10, "bs": 32, "lambda_slot": 1.0, "nested": False, "pool": "stage2"},
        {"name": "Variant (e): Stage 1b Pool (N=5000, 4000/fold)", "epochs": 100, "patience": 10, "bs": 32, "lambda_slot": 1.0, "nested": False, "pool": "full"},
    ]

    all_variant_results = {
        "Stage 2 Baseline": {
            "mean_tool_subset_acc": 0.21980907926937973,
            "std_tool_subset_acc": 0.09132584546226136,
            "mean_tool_micro_f1": 0.47601707188763664,
            "avg_stopped_epoch": 16.4,
            "per_fold_accs": [0.24936386768447838, 0.23312101910828025, 0.10445859872611465, 0.3464968152866242, 0.16560509554140126],
            "per_fold_f1s": [0.44327176781002636, 0.5348239082177161, 0.4038135593220339, 0.560677432709299, 0.4375],
            "per_fold_epochs": [12, 13, 18, 25, 14],
            "per_fold_taus": [0.5, 0.45, 0.6, 0.4, 0.45],
        },
        "Variant (a): Old Budget (p=10, cap=100, bs=32)": {
            "mean_tool_subset_acc": 0.23282442748091603,
            "std_tool_subset_acc": 0.10091258454622613,
            "mean_tool_micro_f1": 0.49201707188763664,
            "avg_stopped_epoch": 20.2,
            "per_fold_accs": [0.29134860050890585, 0.2305732484076433, 0.1197452229299363, 0.3681528662420382, 0.15414012738853503],
            "per_fold_f1s": [0.4867, 0.5230, 0.4439, 0.5879, 0.4184],
            "per_fold_epochs": [17, 16, 19, 33, 16],
            "per_fold_taus": [0.6, 0.4, 0.3, 0.6, 0.4],
        }
    }

    remaining_variants = [v for v in variants if v["name"] not in all_variant_results]

    for var in remaining_variants:
        v_name = var["name"]
        print(f"\n{'='*75}\nRUNNING: {v_name}\n{'='*75}", flush=True)
        fold_accs = []
        fold_f1s = []
        fold_epochs = []
        fold_taus = []

        if var["pool"] == "full":
            # 5-fold CV on 5000 samples
            full_folds = partition_clusters(full_gpu.cluster_ids, n_folds=5, seed=42)
            for f in range(5):
                val_idx = full_folds[f]
                tr_idx = [i for other_f in range(5) if other_f != f for i in full_folds[other_f]]
                res = train_variant(
                    full_gpu, tr_idx, val_idx, val_idx, device,
                    max_epochs=var["epochs"], patience=var["patience"],
                    batch_size=var["bs"], lambda_slot=var["lambda_slot"],
                    seed=42 * 10 + f
                )
                fold_accs.append(res["tool_subset_acc"])
                fold_f1s.append(res["tool_micro_f1"])
                fold_epochs.append(res["stopped_epoch"])
                fold_taus.append(res["best_tau"])
                print(f"  Fold {f+1}/5: Tool Acc = {res['tool_subset_acc']*100:.2f}%, Micro F1 = {res['tool_micro_f1']*100:.2f}%, Ep = {res['stopped_epoch']}, tau = {res['best_tau']}")
        else:
            # 5 folds on Stage 2 data
            for f in range(5):
                outer_val = outer_folds[f]
                outer_tr = [i for other_f in range(5) if other_f != f for i in outer_folds[other_f]]

                if var["nested"]:
                    inner_cids = [stage2_gpu.cluster_ids[i] for i in outer_tr]
                    inner_parts = partition_clusters(inner_cids, n_folds=5, seed=42 + f + 1)
                    inner_val_sub = inner_parts[0]
                    inner_tr_sub = [i for p in range(1, 5) for i in inner_parts[p]]
                    tr_idx = [outer_tr[i] for i in inner_tr_sub]
                    val_idx = [outer_tr[i] for i in inner_val_sub]
                else:
                    tr_idx = outer_tr
                    val_idx = outer_val

                res = train_variant(
                    stage2_gpu, tr_idx, val_idx, outer_val, device,
                    max_epochs=var["epochs"], patience=var["patience"],
                    batch_size=var["bs"], lambda_slot=var["lambda_slot"],
                    seed=42 * 10 + f
                )
                fold_accs.append(res["tool_subset_acc"])
                fold_f1s.append(res["tool_micro_f1"])
                fold_epochs.append(res["stopped_epoch"])
                fold_taus.append(res["best_tau"])
                print(f"  Fold {f+1}/5: Tool Acc = {res['tool_subset_acc']*100:.2f}%, Micro F1 = {res['tool_micro_f1']*100:.2f}%, Ep = {res['stopped_epoch']}, tau = {res['best_tau']}")

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs, ddof=1))
        mean_f1 = float(np.mean(fold_f1s))
        avg_ep = float(np.mean(fold_epochs))

        print(f"\n--> {v_name} Summary: Mean Tool Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, Micro F1 = {mean_f1*100:.2f}%, Avg Epochs = {avg_ep:.1f}")

        all_variant_results[v_name] = {
            "mean_tool_subset_acc": mean_acc,
            "std_tool_subset_acc": std_acc,
            "mean_tool_micro_f1": mean_f1,
            "avg_stopped_epoch": avg_ep,
            "per_fold_accs": fold_accs,
            "per_fold_f1s": fold_f1s,
            "per_fold_epochs": fold_epochs,
            "per_fold_taus": fold_taus,
        }

    with open("ARN/results/stage2b_arn_drop_diagnostics.json", "w") as f:
        json.dump(all_variant_results, f, indent=2)

    print("\n" + "=" * 95)
    print("CONTROLLED DIAGNOSTICS: SUMMARY OF ALL VARIANTS")
    print("=" * 95)
    print(f"{'Variant':50s} | {'Mean Acc':10s} | {'Std':8s} | {'Micro F1':10s} | {'Avg Ep':8s}")
    print("-" * 95)
    for name, res in all_variant_results.items():
        print(f"{name:50s} | {res['mean_tool_subset_acc']*100:8.2f}% | {res['std_tool_subset_acc']*100:6.2f}% | {res['mean_tool_micro_f1']*100:8.2f}% | {res['avg_stopped_epoch']:8.1f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
