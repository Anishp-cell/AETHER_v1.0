"""
Stage 2c Item 6: Learning Curve with Cluster-Bootstrap Confidence Bands.

1. Fixes a held-out evaluation set of 42 clusters (20% of 209).
2. Evaluates training cluster subsets: 20%, 40%, 60%, 80%, 100% of the remaining 167 clusters.
3. Evaluates 4 models:
   - ARN (frozen trunk + FAR + CRF)
   - CLS+CRF (frozen trunk + CLS + CRF)
   - TF-IDF kNN (k=5)
   - Fine-Tuned BERT-tiny (unfrozen trunk + CLS head)
4. Computes 95% cluster-bootstrap confidence bands on the held-out evaluation set.
5. Outputs structured metrics and formatted ASCII curves.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TAG
from arn.model import AetherRoutingNetwork
from arn.baselines import CLSCRFModel
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data, GPUData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2c_learning_curve")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class BertTinyCLSHead(nn.Module):
    def __init__(self, num_tools: int = 18):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.classifier = nn.Linear(128, num_tools)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_repr)


def evaluate_tool_subset(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.all(preds == targets, axis=1).astype(float)


def cluster_bootstrap_bands(
    sample_scores: np.ndarray,
    sample_clusters: List[str],
    n_boot: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    cluster_to_idx = defaultdict(list)
    for i, c in enumerate(sample_clusters):
        cluster_to_idx[c].append(i)
    unique_clusters = list(cluster_to_idx.keys())
    k = len(unique_clusters)

    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=k, replace=True)
        idx_flat = []
        for c in sampled:
            idx_flat.extend(cluster_to_idx[c])
        boot_means.append(float(np.mean(sample_scores[idx_flat])))

    mean_val = float(np.mean(sample_scores))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_low, ci_high


def run_learning_curve():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Learning Curve Suite on device: {device}", flush=True)

    # 1. Load Data
    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)

    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
    utterances = [s["messages"][0]["content"] if s["messages"][0]["role"] == "user" else s["messages"][1]["content"] for s in raw_samples]
    targets = np.array([item["tool_labels"].numpy() for item in dataset])
    gpu_data = precompute_gpu_data(dataset, raw_samples, frame_to_cluster, device)

    # 2. Partition Clusters: Fixed 42 held-out clusters (20%), 167 training clusters (80%)
    unique_clusters = sorted(list(set(cluster_ids)))
    rng = np.random.RandomState(42)
    shuffled_clusters = rng.permutation(unique_clusters).tolist()
    held_out_clusters = set(shuffled_clusters[:42])
    train_pool_clusters = shuffled_clusters[42:]

    eval_indices = [i for i, c in enumerate(cluster_ids) if c in held_out_clusters]
    eval_clusters = [cluster_ids[i] for i in eval_indices]
    eval_targets = targets[eval_indices]
    eval_utterances = [utterances[i] for i in eval_indices]
    n_eval = len(eval_indices)
    print(f"Fixed Evaluation Split: {n_eval} samples across {len(held_out_clusters)} clusters", flush=True)

    # Fractions of 167 training clusters: 20%, 40%, 60%, 80%, 100%
    fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
    curve_results = {m: [] for m in ["arn", "cls_crf", "tfidf_knn", "bert_tiny_full_finetune"]}

    for frac in fractions:
        k_train = max(1, int(round(frac * len(train_pool_clusters))))
        cur_clusters = set(train_pool_clusters[:k_train])
        train_idx = [i for i, c in enumerate(cluster_ids) if c in cur_clusters]
        n_tr = len(train_idx)
        print(f"\n--- Training Fraction: {frac*100:3.0f}% ({k_train}/{len(train_pool_clusters)} clusters, N={n_tr} samples) ---", flush=True)

        y_tr = targets[train_idx]
        u_tr = [utterances[i] for i in train_idx]

        # -----------------------------------------------------------------
        # Model 1: TF-IDF kNN (k=5)
        # -----------------------------------------------------------------
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_tr = vec.fit_transform(u_tr)
        X_val = vec.transform(eval_utterances)
        knn = NearestNeighbors(n_neighbors=min(5, n_tr), metric="cosine")
        knn.fit(X_tr)
        _, indices = knn.kneighbors(X_val)
        knn_probs = np.array([y_tr[indices[i]].mean(axis=0) for i in range(n_eval)])
        knn_preds = (knn_probs >= 0.5).astype(int)
        knn_scores = evaluate_tool_subset(knn_preds, eval_targets)
        mean_knn, low_knn, high_knn = cluster_bootstrap_bands(knn_scores, eval_clusters)
        curve_results["tfidf_knn"].append({
            "fraction": frac,
            "k_clusters": k_train,
            "n_samples": n_tr,
            "mean": mean_knn,
            "ci_low": low_knn,
            "ci_high": high_knn,
        })
        print(f"  [TF-IDF kNN]         Tool Acc: {mean_knn*100:5.2f}% [{low_knn*100:5.2f}%, {high_knn*100:5.2f}%]", flush=True)

        # -----------------------------------------------------------------
        # Model 2: ARN (Frozen Trunk + FAR)
        # -----------------------------------------------------------------
        set_seed(42 + int(frac * 100))
        cfg = ARNConfig(freeze_trunk=True)
        arn_model = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
        arn_opt = torch.optim.AdamW(arn_model.far.parameters(), lr=3e-4, weight_decay=0.01)
        bce = nn.BCEWithLogitsLoss()

        tr_tensor = torch.tensor(train_idx, device=device, dtype=torch.long)
        for ep in range(20):
            arn_model.train()
            perm = tr_tensor[torch.randperm(n_tr)]
            for b in range(0, n_tr, 64):
                b_idx = perm[b : min(b + 64, n_tr)]
                m_b = gpu_data.mask[b_idx]
                max_l = int(m_b.sum(dim=1).max().item())
                h_b = gpu_data.h[b_idx, :max_l]
                m_b = m_b[:, :max_l]
                tools_b = gpu_data.tool_labels[b_idx]

                arn_opt.zero_grad()
                logits, _ = arn_model.far(h_b, m_b)
                loss = bce(logits, tools_b)
                loss.backward()
                arn_opt.step()

        # Evaluate ARN
        arn_model.eval()
        arn_probs_all = []
        with torch.no_grad():
            for vb in range(0, n_eval, 128):
                v_idx = torch.tensor(eval_indices[vb : min(vb + 128, n_eval)], device=device, dtype=torch.long)
                m_v = gpu_data.mask[v_idx]
                max_l = int(m_v.sum(dim=1).max().item())
                h_v = gpu_data.h[v_idx, :max_l]
                m_v = m_v[:, :max_l]
                logits, _ = arn_model.far(h_v, m_v)
                arn_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        arn_probs = np.vstack(arn_probs_all)
        arn_preds = (arn_probs >= 0.40).astype(int)
        arn_scores = evaluate_tool_subset(arn_preds, eval_targets)
        mean_arn, low_arn, high_arn = cluster_bootstrap_bands(arn_scores, eval_clusters)
        curve_results["arn"].append({
            "fraction": frac,
            "k_clusters": k_train,
            "n_samples": n_tr,
            "mean": mean_arn,
            "ci_low": low_arn,
            "ci_high": high_arn,
        })
        print(f"  [ARN (FAR)]          Tool Acc: {mean_arn*100:5.2f}% [{low_arn*100:5.2f}%, {high_arn*100:5.2f}%]", flush=True)

        # -----------------------------------------------------------------
        # Model 3: CLS+CRF (Frozen Trunk + [CLS])
        # -----------------------------------------------------------------
        set_seed(42 + int(frac * 100))
        cls_model = CLSCRFModel(cfg, load_pretrained=False).to(device)
        cls_opt = torch.optim.AdamW(cls_model.tool_classifier.parameters(), lr=3e-4, weight_decay=0.01)

        for ep in range(20):
            cls_model.train()
            perm = tr_tensor[torch.randperm(n_tr)]
            for b in range(0, n_tr, 64):
                b_idx = perm[b : min(b + 64, n_tr)]
                h_b = gpu_data.h[b_idx, 0, :]
                tools_b = gpu_data.tool_labels[b_idx]

                cls_opt.zero_grad()
                logits = cls_model.tool_classifier(h_b)
                loss = bce(logits, tools_b)
                loss.backward()
                cls_opt.step()

        # Evaluate CLS
        cls_model.eval()
        cls_probs_all = []
        with torch.no_grad():
            for vb in range(0, n_eval, 128):
                v_idx = torch.tensor(eval_indices[vb : min(vb + 128, n_eval)], device=device, dtype=torch.long)
                h_v = gpu_data.h[v_idx, 0, :]
                logits = cls_model.tool_classifier(h_v)
                cls_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        cls_probs = np.vstack(cls_probs_all)
        cls_preds = (cls_probs >= 0.40).astype(int)
        cls_scores = evaluate_tool_subset(cls_preds, eval_targets)
        mean_cls, low_cls, high_cls = cluster_bootstrap_bands(cls_scores, eval_clusters)
        curve_results["cls_crf"].append({
            "fraction": frac,
            "k_clusters": k_train,
            "n_samples": n_tr,
            "mean": mean_cls,
            "ci_low": low_cls,
            "ci_high": high_cls,
        })
        print(f"  [CLS+CRF ([CLS])]    Tool Acc: {mean_cls*100:5.2f}% [{low_cls*100:5.2f}%, {high_cls*100:5.2f}%]", flush=True)

        # -----------------------------------------------------------------
        # Model 4: Fine-Tuned BERT-tiny (Unfrozen Trunk)
        # -----------------------------------------------------------------
        set_seed(42 + int(frac * 100))
        ft_model = BertTinyCLSHead(num_tools=18).to(device)
        ft_opt = torch.optim.AdamW([
            {"params": ft_model.encoder.parameters(), "lr": 2e-5, "weight_decay": 0.01},
            {"params": ft_model.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01}
        ])

        tr_items = [dataset[i] for i in train_idx]
        val_items = [dataset[i] for i in eval_indices]

        tr_ids = torch.stack([x["input_ids"] for x in tr_items]).to(device)
        tr_mask = torch.stack([x["attention_mask"] for x in tr_items]).to(device)
        tr_tools = torch.stack([x["tool_labels"] for x in tr_items]).to(device)

        val_ids = torch.stack([x["input_ids"] for x in val_items]).to(device)
        val_mask = torch.stack([x["attention_mask"] for x in val_items]).to(device)

        for ep in range(10):
            ft_model.train()
            perm = torch.randperm(n_tr)
            for b in range(0, n_tr, 32):
                b_idx = perm[b : min(b + 32, n_tr)]
                ft_opt.zero_grad()
                logits = ft_model(tr_ids[b_idx], tr_mask[b_idx])
                loss = ft_model.bce_loss(logits, tr_tools[b_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
                ft_opt.step()

        # Evaluate FT BERT-tiny
        ft_model.eval()
        ft_probs_all = []
        with torch.no_grad():
            for vb in range(0, n_eval, 64):
                logits = ft_model(val_ids[vb : vb + 64], val_mask[vb : vb + 64])
                ft_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        ft_probs = np.vstack(ft_probs_all)
        ft_preds = (ft_probs >= 0.40).astype(int)
        ft_scores = evaluate_tool_subset(ft_preds, eval_targets)
        mean_ft, low_ft, high_ft = cluster_bootstrap_bands(ft_scores, eval_clusters)
        curve_results["bert_tiny_full_finetune"].append({
            "fraction": frac,
            "k_clusters": k_train,
            "n_samples": n_tr,
            "mean": mean_ft,
            "ci_low": low_ft,
            "ci_high": high_ft,
        })
        print(f"  [BERT-tiny Full FT]  Tool Acc: {mean_ft*100:5.2f}% [{low_ft*100:5.2f}%, {high_ft*100:5.2f}%]", flush=True)

    # Summary table
    print("\n" + "=" * 105, flush=True)
    print("LEARNING CURVE SUMMARY TABLE (EVALUATED ON FIXED 42 HELD-OUT CLUSTERS, N=784)")
    print("=" * 105, flush=True)
    header = f"{'Fraction (Clusters)':20s} | {'TF-IDF kNN [95% CI]':24s} | {'ARN [95% CI]':22s} | {'CLS+CRF [95% CI]':22s} | {'BERT-tiny FT [95% CI]':24s}"
    print(header, flush=True)
    print("-" * 105, flush=True)

    for i, frac in enumerate(fractions):
        knn_res = curve_results["tfidf_knn"][i]
        arn_res = curve_results["arn"][i]
        cls_res = curve_results["cls_crf"][i]
        ft_res = curve_results["bert_tiny_full_finetune"][i]

        f_str = f"{frac*100:3.0f}% ({knn_res['k_clusters']} clusters)"
        knn_str = f"{knn_res['mean']*100:5.2f}% [{knn_res['ci_low']*100:4.1f}%, {knn_res['ci_high']*100:4.1f}%]"
        arn_str = f"{arn_res['mean']*100:5.2f}% [{arn_res['ci_low']*100:4.1f}%, {arn_res['ci_high']*100:4.1f}%]"
        cls_str = f"{cls_res['mean']*100:5.2f}% [{cls_res['ci_low']*100:4.1f}%, {cls_res['ci_high']*100:4.1f}%]"
        ft_str = f"{ft_res['mean']*100:5.2f}% [{ft_res['ci_low']*100:4.1f}%, {ft_res['ci_high']*100:4.1f}%]"

        print(f"{f_str:20s} | {knn_str:24s} | {arn_str:22s} | {cls_str:22s} | {ft_str:24s}", flush=True)
    print("=" * 105, flush=True)

    out_file = Path("ARN/results/stage2c_learning_curve.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(curve_results, f, indent=2)
    print(f"\nSaved learning curve results to {out_file}", flush=True)


if __name__ == "__main__":
    run_learning_curve()
