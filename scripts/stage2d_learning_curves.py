"""
Stage 2d Item 5: Comprehensive Learning Curves.

Protocol:
  - 3 distinct held-out cluster test sets (~42 clusters each).
  - For each test set, subsample the remaining ~167 pool clusters at 20%, 40%, 60%, 80%, 100%.
  - 5 random cluster subsamples per fraction (3 test splits x 5 subsamples = 15 runs per fraction).
  - Evaluates:
      1. Frozen ARN
      2. Frozen CLS+CRF
      3. Unfrozen CLS+CRF
      4. TF-IDF kNN
  - Computes Mean, SD, and paired 95% CIs.
"""

import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from transformers import AutoModel
from arn.crf import CRF
from arn.tokenizer import TOKENIZER, TAG_TO_IDX
from arn.baselines import CLSCRFModel
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data


class UnfrozenBertCLS_CRF(nn.Module):
    def __init__(self, num_tools: int = 18, num_tags: int = 26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.tool_classifier = nn.Linear(128, num_tools)
        self.slot_tagger = nn.Linear(128, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls_repr = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_repr)
        emissions = self.slot_tagger(h)
        return tool_logits, emissions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_unfrozen(batch, device):
    max_l = max(b["attention_mask"].sum().item() for b in batch)
    max_l = min(max_l, 64)
    ids = torch.stack([b["input_ids"][:max_l] for b in batch]).to(device)
    mask = torch.stack([b["attention_mask"][:max_l] for b in batch]).to(device)
    tags = torch.stack([b["tags"][:max_l] for b in batch]).to(device)
    tools = torch.stack([b["tool_labels"] for b in batch]).to(device)
    return ids, mask, tags, tools


def run_learning_curves():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 2d LEARNING CURVES SUITE on {device} ===", flush=True)

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)
    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
    gpu_data = precompute_gpu_data(dataset, raw_samples, frame_to_cluster, device)

    unique_clusters = sorted(list(set(cluster_ids)))
    n_total_clusters = len(unique_clusters)
    user_texts = [s["messages"][0]["content"].strip() for s in raw_samples]
    y_tools = gpu_data.tool_labels.cpu().numpy()

    # TF-IDF matrix
    tfidf = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    tfidf_X = tfidf.fit_transform(user_texts).toarray()

    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    n_test_splits = 3
    n_subsamples_per_frac = 5

    models = ["frozen_arn", "frozen_cls_crf", "unfrozen_cls_crf", "tfidf_knn"]
    results_by_frac = {f: {m: [] for m in models} for f in fractions}

    for test_idx in range(n_test_splits):
        rng = random.Random(1000 + test_idx * 137)
        shuffled_clusters = list(unique_clusters)
        rng.shuffle(shuffled_clusters)

        n_test = int(n_total_clusters * 0.20)
        test_clusters = set(shuffled_clusters[:n_test])
        pool_clusters = [c for c in shuffled_clusters[n_test:]]

        test_sample_idx = [i for i, cid in enumerate(cluster_ids) if cid in test_clusters]
        y_test = y_tools[test_sample_idx]
        test_idx_tensor = torch.tensor(test_sample_idx, device=device, dtype=torch.long)

        print(f"\n--- TEST SPLIT {test_idx+1}/{n_test_splits} (Test Clusters={len(test_clusters)}, Test Samples={len(test_sample_idx)}, Pool Clusters={len(pool_clusters)}) ---", flush=True)

        for frac in fractions:
            k_train = max(2, int(len(pool_clusters) * frac))

            for sub_i in range(n_subsamples_per_frac):
                sub_rng = random.Random(2000 + test_idx * 500 + int(frac * 100) * 10 + sub_i)
                sampled_clusters = set(sub_rng.sample(pool_clusters, k_train))
                train_sample_idx = [i for i, cid in enumerate(cluster_ids) if cid in sampled_clusters]
                y_train = y_tools[train_sample_idx]
                n_tr = len(train_sample_idx)

                # 1. TF-IDF kNN
                knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
                knn.fit(tfidf_X[train_sample_idx], y_train)
                knn_pred = knn.predict(tfidf_X[test_sample_idx])
                knn_acc = float(np.all(knn_pred == y_test, axis=1).mean())
                results_by_frac[frac]["tfidf_knn"].append(knn_acc)

                # 2. Frozen ARN
                set_seed(3000 + sub_i * 11)
                cfg = ARNConfig(freeze_trunk=True)
                arn_model = AetherRoutingNetwork(cfg, load_pretrained=False).to(device)
                arn_opt = torch.optim.AdamW(arn_model.far.parameters(), lr=3e-4, weight_decay=0.01)
                bce = nn.BCEWithLogitsLoss()

                tr_tensor = torch.tensor(train_sample_idx, device=device, dtype=torch.long)
                for _ in range(8):
                    arn_model.train()
                    perm = tr_tensor[torch.randperm(n_tr)]
                    for b in range(0, n_tr, 64):
                        b_idx = perm[b : min(b + 64, n_tr)]
                        m_b = gpu_data.mask[b_idx]
                        max_l = int(m_b.sum(dim=1).max().item())
                        h_b = gpu_data.h[b_idx, :max_l]
                        m_b = m_b[:, :max_l]
                        t_b = gpu_data.tool_labels[b_idx]

                        arn_opt.zero_grad()
                        logits, _ = arn_model.far(h_b, m_b)
                        loss = bce(logits, t_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(arn_model.far.parameters(), 1.0)
                        arn_opt.step()

                arn_model.eval()
                with torch.no_grad():
                    all_p = []
                    for b in range(0, len(test_sample_idx), 128):
                        v_idx = test_idx_tensor[b : min(b + 128, len(test_sample_idx))]
                        m_v = gpu_data.mask[v_idx]
                        max_l = int(m_v.sum(dim=1).max().item())
                        h_v = gpu_data.h[v_idx, :max_l]
                        m_v = m_v[:, :max_l]
                        logits, _ = arn_model.far(h_v, m_v)
                        all_p.append(torch.sigmoid(logits).cpu().numpy())
                    arn_probs = np.vstack(all_p)
                arn_pred = (arn_probs >= 0.50).astype(int)
                arn_acc = float(np.all(arn_pred == y_test, axis=1).mean())
                results_by_frac[frac]["frozen_arn"].append(arn_acc)

                # 3. Frozen CLS+CRF
                set_seed(3000 + sub_i * 11)
                cls_model = CLSCRFModel(cfg, load_pretrained=False).to(device)
                cls_opt = torch.optim.AdamW(cls_model.tool_classifier.parameters(), lr=3e-4, weight_decay=0.01)

                for _ in range(8):
                    cls_model.train()
                    perm = tr_tensor[torch.randperm(n_tr)]
                    for b in range(0, n_tr, 64):
                        b_idx = perm[b : min(b + 64, n_tr)]
                        m_b = gpu_data.mask[b_idx]
                        max_l = int(m_b.sum(dim=1).max().item())
                        h_b = gpu_data.h[b_idx, :max_l]
                        m_b = m_b[:, :max_l]
                        t_b = gpu_data.tool_labels[b_idx]

                        cls_opt.zero_grad()
                        logits = cls_model.tool_classifier(h_b[:, 0, :])
                        loss = bce(logits, t_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(cls_model.tool_classifier.parameters(), 1.0)
                        cls_opt.step()

                cls_model.eval()
                with torch.no_grad():
                    all_p = []
                    for b in range(0, len(test_sample_idx), 128):
                        v_idx = test_idx_tensor[b : min(b + 128, len(test_sample_idx))]
                        m_v = gpu_data.mask[v_idx]
                        max_l = int(m_v.sum(dim=1).max().item())
                        h_v = gpu_data.h[v_idx, :max_l]
                        m_v = m_v[:, :max_l]
                        logits = cls_model.tool_classifier(h_v[:, 0, :])
                        all_p.append(torch.sigmoid(logits).cpu().numpy())
                    cls_probs = np.vstack(all_p)
                cls_pred = (cls_probs >= 0.50).astype(int)
                cls_acc = float(np.all(cls_pred == y_test, axis=1).mean())
                results_by_frac[frac]["frozen_cls_crf"].append(cls_acc)

                # 4. Unfrozen CLS+CRF
                set_seed(3000 + sub_i * 11)
                unf_cls = UnfrozenBertCLS_CRF().to(device)
                unf_opt = torch.optim.AdamW([
                    {"params": list(unf_cls.encoder.parameters()), "lr": 3e-5, "weight_decay": 0.01},
                    {"params": list(unf_cls.tool_classifier.parameters()) + list(unf_cls.slot_tagger.parameters()), "lr": 3e-4, "weight_decay": 0.01},
                ])

                tr_batch_data = [dataset.encoded[i] for i in train_sample_idx]
                test_batch_data = [dataset.encoded[i] for i in test_sample_idx]

                for _ in range(6):
                    unf_cls.train()
                    perm = list(range(n_tr))
                    random.shuffle(perm)
                    for b in range(0, n_tr, 64):
                        b_idx = [perm[k] for k in range(b, min(b + 64, n_tr))]
                        batch = [tr_batch_data[k] for k in b_idx]
                        ids, mask, _, tools = collate_unfrozen(batch, device)

                        unf_opt.zero_grad()
                        tool_logits, _ = unf_cls(input_ids=ids, attention_mask=mask)
                        loss = bce(tool_logits, tools)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(unf_cls.parameters(), 1.0)
                        unf_opt.step()

                unf_cls.eval()
                with torch.no_grad():
                    all_p = []
                    for b in range(0, len(test_sample_idx), 128):
                        batch = test_batch_data[b : min(b + 128, len(test_sample_idx))]
                        ids, mask, _, _ = collate_unfrozen(batch, device)
                        tool_logits, _ = unf_cls(input_ids=ids, attention_mask=mask)
                        all_p.append(torch.sigmoid(tool_logits).cpu().numpy())
                    unf_probs = np.vstack(all_p)
                unf_pred = (unf_probs >= 0.50).astype(int)
                unf_acc = float(np.all(unf_pred == y_test, axis=1).mean())
                results_by_frac[frac]["unfrozen_cls_crf"].append(unf_acc)

            print(f"  Frac {int(frac*100):3d}% ({k_train:3d} cl) | kNN={np.mean(results_by_frac[frac]['tfidf_knn'])*100:5.2f}% | ARN={np.mean(results_by_frac[frac]['frozen_arn'])*100:5.2f}% | FrozenCLS={np.mean(results_by_frac[frac]['frozen_cls_crf'])*100:5.2f}% | UnfrozenCLS={np.mean(results_by_frac[frac]['unfrozen_cls_crf'])*100:5.2f}%", flush=True)

    # Synthesis & Paired CIs
    final_output = {}
    for frac in fractions:
        frac_summary = {}
        for m in models:
            vals = results_by_frac[frac][m]
            boot_means = [np.mean([vals[random.randint(0, len(vals)-1)] for _ in range(len(vals))]) for _ in range(1000)]
            frac_summary[m] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "ci_95": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
                "n_runs": len(vals),
            }

        # Paired deltas against frozen ARN
        arn_vals = results_by_frac[frac]["frozen_arn"]
        paired_deltas = {}
        for m in ["tfidf_knn", "frozen_cls_crf", "unfrozen_cls_crf"]:
            m_vals = results_by_frac[frac][m]
            deltas = [b - a for a, b in zip(arn_vals, m_vals)]
            boot_deltas = [np.mean([deltas[random.randint(0, len(deltas)-1)] for _ in range(len(deltas))]) for _ in range(1000)]
            paired_deltas[f"{m}_minus_arn"] = {
                "mean_delta": float(np.mean(deltas)),
                "ci_95": [float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))],
            }
        frac_summary["paired_deltas_vs_arn"] = paired_deltas
        final_output[f"fraction_{frac}"] = frac_summary

    out_file = Path("ARN/results/stage2d_learning_curves.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved complete learning curves results to {out_file}", flush=True)


if __name__ == "__main__":
    run_learning_curves()
