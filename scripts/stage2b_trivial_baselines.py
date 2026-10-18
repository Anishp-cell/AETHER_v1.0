"""
Stage 2b Item 4: Trivial Baselines on Identical Folds (3 Shuffles x 5 Folds = 15 Folds).

Evaluates:
1. Most-Frequent Tool Set (predict the single most common tool combination from train)
2. TF-IDF + Logistic Regression (One-vs-Rest)
3. TF-IDF kNN (k=5, cosine metric)
4. Fully Fine-Tuned BERT-tiny [CLS] Head (unfrozen trunk with differential LR)
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, BertTokenizerFast

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, TOOL_TO_IDX
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import partition_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.trivial_baselines")


def set_seed(seed: int):
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
        logits = self.classifier(cls_repr)
        return logits


def evaluate_preds(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    n_samples = len(targets)
    matches = np.all(preds == targets, axis=1)
    acc = float(np.mean(matches))
    acc_count = int(np.sum(matches))

    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))
    p = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    r = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0

    return {
        "tool_subset_acc": acc,
        "tool_subset_acc_count": acc_count,
        "n_samples": n_samples,
        "tool_micro_p": p,
        "tool_micro_r": r,
        "tool_micro_f1": f1,
    }


def run_trivial_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Trivial Baselines Suite on device: {device}", flush=True)

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

    # 15 Folds: 3 Shuffles x 5 Folds
    shuffles = [
        {"shuffle": 1, "seed": 42},
        {"shuffle": 2, "seed": 100},
        {"shuffle": 3, "seed": 2024},
    ]

    baselines = ["most_frequent", "tfidf_logistic", "tfidf_knn", "bert_tiny_full_finetune"]
    results = {b: {"folds": [], "per_shuffle": {}} for b in baselines}

    for sh in shuffles:
        sh_idx = sh["shuffle"]
        sh_seed = sh["seed"]
        print(f"\n{'='*75}\nSTARTING SHUFFLE {sh_idx}/3 (Seed {sh_seed})\n{'='*75}", flush=True)

        outer_folds = partition_clusters(cluster_ids, n_folds=5, seed=sh_seed)

        for f_idx in range(5):
            val_idx = outer_folds[f_idx]
            train_idx = [i for other_f in range(5) if other_f != f_idx for i in outer_folds[other_f]]

            y_tr = targets[train_idx]
            y_val = targets[val_idx]
            u_tr = [utterances[i] for i in train_idx]
            u_val = [utterances[i] for i in val_idx]

            # ---------------------------------------------------------
            # 1. Most-Frequent Tool Set Baseline
            # ---------------------------------------------------------
            tuples_tr = [tuple(y_tr[i]) for i in range(len(y_tr))]
            most_common_tuple = Counter(tuples_tr).most_common(1)[0][0]
            mf_preds = np.tile(np.array(most_common_tuple), (len(val_idx), 1))
            mf_eval = evaluate_preds(mf_preds, y_val)
            results["most_frequent"]["folds"].append(mf_eval)

            # ---------------------------------------------------------
            # 2. TF-IDF Vectorizer
            # ---------------------------------------------------------
            vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
            X_tr = vectorizer.fit_transform(u_tr)
            X_val = vectorizer.transform(u_val)

            # ---------------------------------------------------------
            # 2. TF-IDF + Logistic Regression (OvR)
            # ---------------------------------------------------------
            lr_preds = np.zeros_like(y_val)
            for k in range(18):
                if y_tr[:, k].sum() > 0:
                    clf = LogisticRegression(C=1.0, max_iter=200)
                    clf.fit(X_tr, y_tr[:, k])
                    probs = clf.predict_proba(X_val)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(val_idx))
                    lr_preds[:, k] = (probs >= 0.5).astype(int)
                else:
                    lr_preds[:, k] = 0
            lr_eval = evaluate_preds(lr_preds, y_val)
            results["tfidf_logistic"]["folds"].append(lr_eval)

            # ---------------------------------------------------------
            # 3. TF-IDF kNN (k=5)
            # ---------------------------------------------------------
            knn = NearestNeighbors(n_neighbors=5, metric="cosine")
            knn.fit(X_tr)
            dists, indices = knn.kneighbors(X_val)
            knn_probs = np.array([y_tr[indices[i]].mean(axis=0) for i in range(len(val_idx))])
            knn_preds = (knn_probs >= 0.5).astype(int)
            knn_eval = evaluate_preds(knn_preds, y_val)
            results["tfidf_knn"]["folds"].append(knn_eval)

            # ---------------------------------------------------------
            # 4. Fully Fine-Tuned BERT-tiny [CLS] Head
            # ---------------------------------------------------------
            set_seed(sh_seed * 10 + f_idx)
            model = BertTinyCLSHead(num_tools=18).to(device)
            # Differential learning rates: 2e-5 for trunk, 1e-3 for classifier head
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": 2e-5, "weight_decay": 0.01},
                {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01}
            ])

            # Prepare tensors
            tr_items = [dataset[i] for i in train_idx]
            val_items = [dataset[i] for i in val_idx]

            tr_ids = torch.stack([x["input_ids"] for x in tr_items]).to(device)
            tr_mask = torch.stack([x["attention_mask"] for x in tr_items]).to(device)
            tr_tools = torch.stack([x["tool_labels"] for x in tr_items]).to(device)

            val_ids = torch.stack([x["input_ids"] for x in val_items]).to(device)
            val_mask = torch.stack([x["attention_mask"] for x in val_items]).to(device)

            best_val_loss = float("inf")
            best_state = None
            patience = 3
            patience_cnt = 0
            bs = 32
            n_tr = len(tr_items)

            for epoch in range(1, 13):
                model.train()
                perm = torch.randperm(n_tr)
                for b in range(0, n_tr, bs):
                    b_idx = perm[b : min(b + bs, n_tr)]
                    optimizer.zero_grad()
                    logits = model(tr_ids[b_idx], tr_mask[b_idx])
                    loss = model.bce_loss(logits, tr_tools[b_idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Fast val loss
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in range(0, len(val_items), 64):
                        v_logits = model(val_ids[vb : vb + 64], val_mask[vb : vb + 64])
                        v_loss = model.bce_loss(v_logits, torch.tensor(y_val[vb : vb + 64], device=device, dtype=torch.float32))
                        val_losses.append(v_loss.item())
                cur_v_loss = float(np.mean(val_losses))
                if cur_v_loss < best_val_loss - 0.005:
                    best_val_loss = cur_v_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            all_bert_probs = []
            with torch.no_grad():
                for vb in range(0, len(val_items), 64):
                    v_logits = model(val_ids[vb : vb + 64], val_mask[vb : vb + 64])
                    all_bert_probs.append(torch.sigmoid(v_logits).cpu().numpy())
            bert_probs = np.vstack(all_bert_probs)
            bert_preds = (bert_probs >= 0.5).astype(int)
            bert_eval = evaluate_preds(bert_preds, y_val)
            results["bert_tiny_full_finetune"]["folds"].append(bert_eval)

            print(
                f"  Shuffle {sh_idx} Fold {f_idx+1}/5 | "
                f"Most-Freq: {mf_eval['tool_subset_acc']*100:.2f}% | "
                f"TF-IDF LR: {lr_eval['tool_subset_acc']*100:.2f}% | "
                f"TF-IDF kNN: {knn_eval['tool_subset_acc']*100:.2f}% | "
                f"BERT-tiny Full FT: {bert_eval['tool_subset_acc']*100:.2f}%",
                flush=True
            )

    # Summarize all baselines across 15 folds
    summary = {}
    print("\n" + "=" * 90)
    print("TRIVIAL & REFERENCE BASELINES: 15-FOLD COMPARATIVE SUMMARY")
    print("=" * 90)
    print(f"{'Baseline':30s} | {'Tool Acc Mean':15s} | {'Tool Acc SD':12s} | {'Micro F1 Mean':15s} | {'Micro F1 SD':12s}")
    print("-" * 90)

    for b in baselines:
        accs = [f["tool_subset_acc"] for f in results[b]["folds"]]
        f1s = [f["tool_micro_f1"] for f in results[b]["folds"]]
        mean_acc = float(np.mean(accs))
        sd_acc = float(np.std(accs, ddof=1))
        mean_f1 = float(np.mean(f1s))
        sd_f1 = float(np.std(f1s, ddof=1))

        summary[b] = {
            "mean_tool_subset_acc": mean_acc,
            "sd_tool_subset_acc": sd_acc,
            "mean_tool_micro_f1": mean_f1,
            "sd_tool_micro_f1": sd_f1,
            "raw_folds": results[b]["folds"]
        }
        print(f"{b:30s} | {mean_acc*100:13.2f}% | {sd_acc*100:10.2f}% | {mean_f1*100:13.2f}% | {sd_f1*100:10.2f}%")
    print("=" * 90)

    with open("ARN/results/stage2b_trivial_baselines.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved baseline results to ARN/results/stage2b_trivial_baselines.json", flush=True)


if __name__ == "__main__":
    run_trivial_baselines()
