"""
Stage 2d Master Unified Protocol (Strict Nested CV across 15 Folds).

Models evaluated:
  1. Frozen ARN
  2. Frozen CLS+CRF
  3. Unfrozen ARN
  4. Unfrozen CLS+CRF
  5. Unfrozen JointBERT
  6. TF-IDF kNN
  7. TF-IDF LogReg

Protocol:
  - 3 shuffles x 5 outer folds = 15 outer folds across K=209 semantic frame clusters (N=3,926).
  - Inside each outer fold, outer train clusters are split into Inner Train (80%) and Inner Val (20%).
  - Early stopping and threshold tuning (tau in [0.30, 0.70]) selected STRICTLY on Inner Val.
  - Outer Val evaluated ONCE with inner-selected checkpoint and tau*.
  - Out-of-fold predictions aggregated for:
      * Overall metrics (Mean, SD, Cluster Bootstrap 95% CIs)
      * Paired per-cluster deltas vs ARN with CIs
      * Single-Tool vs Multi-Tool breakdown out-of-fold
      * Accuracy per tool pair across all 7 pairs with held-out status
      * Raw per-cluster accuracy distribution (saving raw per-cluster file)
      * Intraclass correlation coefficient (rho) calculation via ANOVA
      * Held-out CRF vs Softmax decomposition across outer validation folds
"""

import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Set, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from arn.baselines import JointBERT, CLSCRFModel, FARNoCRFModel
from arn.crf import CRF
from arn.far import FactoredAttentionRouter
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import precompute_gpu_data, GPUData
from scripts.stage2c_head_independence import extract_spans_from_bio, partition_clusters
from scripts.stage2c_softmax_baselines import (
    build_bio_transition_mask,
    bio_constrained_viterbi_decode,
    extract_spans_strict,
    compute_span_metrics,
    compute_tag_level_f1
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.stage2d")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_assistant_tool_calls(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = []
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
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
    return calls


# Unfrozen Architectures
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


class UnfrozenBertARN(nn.Module):
    def __init__(self, num_tools: int = 18, num_tags: int = 26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.far = FactoredAttentionRouter(num_tools=num_tools, hidden_dim=128, dropout=0.1)
        self.slot_tagger = nn.Linear(128, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        tool_logits, _ = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)
        return tool_logits, emissions


class UnfrozenBertJoint(nn.Module):
    def __init__(self, num_tools: int = 18, num_tags: int = 26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.tool_classifier = nn.Linear(128, num_tools)
        self.slot_classifier = nn.Linear(128, num_tags)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls_repr = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_repr)
        emissions = self.slot_classifier(h)
        return tool_logits, emissions


class MasterProtocolRunner:
    def __init__(self, device: torch.device):
        self.device = device
        self.train_path = "ARN/data/splits/group_disjoint_train.jsonl"
        with open(self.train_path, "r", encoding="utf-8") as f:
            self.raw_samples = [json.loads(line) for line in f if line.strip()]

        raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
        self.frame_to_cluster = build_canonical_clusters(raw_path)
        self.dataset = ARNDataset(self.train_path, tokenizer=TOKENIZER)
        self.cluster_ids = [self.frame_to_cluster[extract_fine_semantic_frame(s)] for s in self.raw_samples]
        self.gpu_data = precompute_gpu_data(self.dataset, self.raw_samples, self.frame_to_cluster, device)

        self.user_texts = [s["messages"][0]["content"].strip() for s in self.raw_samples]
        self.tool_targets = self.gpu_data.tool_labels.cpu().numpy()

        self.candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

        # Precompute single vs multi and tool pairs
        self.sample_info = []
        for idx, s in enumerate(self.raw_samples):
            calls = get_assistant_tool_calls(s)
            n_tools = len(calls)
            pair_name = None
            if n_tools == 2:
                t_names = sorted([c.get("name") or c.get("function", {}).get("name") for c in calls])
                pair_name = f"{t_names[0]} + {t_names[1]}"
            self.sample_info.append({
                "n_tools": n_tools,
                "is_multi": n_tools > 1,
                "pair_name": pair_name,
                "cluster_id": self.cluster_ids[idx]
            })

        # Grammatical transition masks
        t_mask, s_mask, e_mask = build_bio_transition_mask(TAG_TO_IDX)
        self.t_mask = t_mask.to(device)
        self.s_mask = s_mask.to(device)
        self.e_mask = e_mask.to(device)

    def train_frozen_model(
        self,
        model_name: str,
        inner_train_idx: List[int],
        inner_val_idx: List[int],
        seed: int,
    ) -> Tuple[nn.Module, float]:
        """Train frozen trunk model, tune tau on inner val."""
        set_seed(seed)
        cfg = ARNConfig(freeze_trunk=True)
        if model_name == "frozen_arn":
            model = AetherRoutingNetwork(cfg, load_pretrained=False).to(self.device)
            t_params = list(model.far.parameters())
            s_params = list(model.slot_tagger.parameters()) + list(model.crf.parameters())
        elif model_name == "frozen_cls_crf":
            model = CLSCRFModel(cfg, load_pretrained=False).to(self.device)
            t_params = list(model.tool_classifier.parameters())
            s_params = list(model.slot_tagger.parameters()) + list(model.crf.parameters())

        t_opt = torch.optim.AdamW(t_params, lr=3e-4, weight_decay=0.01)
        s_opt = torch.optim.AdamW(s_params, lr=1e-3, weight_decay=0.01)
        bce = nn.BCEWithLogitsLoss()

        tr_idx = torch.tensor(inner_train_idx, device=self.device, dtype=torch.long)
        val_idx = torch.tensor(inner_val_idx, device=self.device, dtype=torch.long)
        n_tr = len(inner_train_idx)

        best_val_loss = float("inf")
        best_state = None
        patience = 6
        patience_cnt = 0

        for ep in range(1, 21):
            model.train()
            perm = tr_idx[torch.randperm(n_tr)]
            for b in range(0, n_tr, 64):
                b_idx = perm[b : min(b + 64, n_tr)]
                m_b = self.gpu_data.mask[b_idx]
                max_l = int(m_b.sum(dim=1).max().item())
                h_b = self.gpu_data.h[b_idx, :max_l]
                m_b = m_b[:, :max_l]
                t_b = self.gpu_data.tags[b_idx, :max_l]
                tools_b = self.gpu_data.tool_labels[b_idx]

                # Tool step
                t_opt.zero_grad()
                if hasattr(model, "far"):
                    t_logits, _ = model.far(h_b, m_b)
                else:
                    t_logits = model.tool_classifier(h_b[:, 0, :])
                t_loss = bce(t_logits, tools_b)
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(t_params, 1.0)
                t_opt.step()

                # Slot step
                s_opt.zero_grad()
                emissions = model.slot_tagger(h_b)
                nlls = model.crf.sentence_nll(emissions, t_b, m_b)
                active = m_b.sum(dim=1).float().clamp(min=1.0)
                s_loss = (nlls / active).mean()
                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(s_params, 1.0)
                s_opt.step()

            # Inner val evaluation
            model.eval()
            with torch.no_grad():
                val_losses = []
                for vb in range(0, len(inner_val_idx), 128):
                    v_idx = val_idx[vb : min(vb + 128, len(inner_val_idx))]
                    m_v = self.gpu_data.mask[v_idx]
                    max_l = int(m_v.sum(dim=1).max().item())
                    h_v = self.gpu_data.h[v_idx, :max_l]
                    m_v = m_v[:, :max_l]
                    tools_v = self.gpu_data.tool_labels[v_idx]
                    if hasattr(model, "far"):
                        t_logits, _ = model.far(h_v, m_v)
                    else:
                        t_logits = model.tool_classifier(h_v[:, 0, :])
                    val_losses.append(bce(t_logits, tools_v).item())

                mean_v = float(np.mean(val_losses))
                if mean_v < best_val_loss - 0.002:
                    best_val_loss = mean_v
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Select tau on Inner Val
        model.eval()
        with torch.no_grad():
            inner_probs = []
            for vb in range(0, len(inner_val_idx), 128):
                v_idx = val_idx[vb : min(vb + 128, len(inner_val_idx))]
                m_v = self.gpu_data.mask[v_idx]
                max_l = int(m_v.sum(dim=1).max().item())
                h_v = self.gpu_data.h[v_idx, :max_l]
                m_v = m_v[:, :max_l]
                if hasattr(model, "far"):
                    t_logits, _ = model.far(h_v, m_v)
                else:
                    t_logits = model.tool_classifier(h_v[:, 0, :])
                inner_probs.append(torch.sigmoid(t_logits).cpu().numpy())

        inner_probs = np.vstack(inner_probs)
        inner_targets = self.tool_targets[inner_val_idx]

        best_tau = 0.50
        best_acc = -1.0
        for tau in self.candidate_thresholds:
            p = (inner_probs >= tau).astype(int)
            acc = float(np.all(p == inner_targets, axis=1).mean())
            if acc > best_acc:
                best_acc = acc
                best_tau = tau

        return model, best_tau

    def train_unfrozen_model(
        self,
        model_name: str,
        inner_train_idx: List[int],
        inner_val_idx: List[int],
        seed: int,
    ) -> Tuple[nn.Module, float]:
        """Train unfrozen BERT-tiny model end-to-end."""
        set_seed(seed)
        if model_name == "unfrozen_arn":
            model = UnfrozenBertARN(num_tools=18, num_tags=26).to(self.device)
        elif model_name == "unfrozen_cls_crf":
            model = UnfrozenBertCLS_CRF(num_tools=18, num_tags=26).to(self.device)
        elif model_name == "unfrozen_joint_bert":
            model = UnfrozenBertJoint(num_tools=18, num_tags=26).to(self.device)

        trunk_params = list(model.encoder.parameters())
        head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        opt = torch.optim.AdamW([
            {"params": trunk_params, "lr": 3e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 3e-4, "weight_decay": 0.01},
        ])
        bce = nn.BCEWithLogitsLoss()

        tr_data = [self.dataset.encoded[i] for i in inner_train_idx]
        val_data = [self.dataset.encoded[i] for i in inner_val_idx]

        best_val_loss = float("inf")
        best_state = None
        patience = 5
        patience_cnt = 0

        def collate_fn(batch):
            max_l = max(b["attention_mask"].sum().item() for b in batch)
            max_l = min(max_l, 64)
            ids = torch.stack([b["input_ids"][:max_l] for b in batch]).to(self.device)
            mask = torch.stack([b["attention_mask"][:max_l] for b in batch]).to(self.device)
            tags = torch.stack([b["tags"][:max_l] for b in batch]).to(self.device)
            tools = torch.stack([b["tool_labels"] for b in batch]).to(self.device)
            return ids, mask, tags, tools

        n_tr = len(inner_train_idx)
        for ep in range(1, 13):
            model.train()
            perm = list(range(n_tr))
            random.shuffle(perm)
            for b in range(0, n_tr, 64):
                batch_idx = [perm[k] for k in range(b, min(b + 64, n_tr))]
                batch = [tr_data[k] for k in batch_idx]
                ids, mask, tags, tools = collate_fn(batch)

                opt.zero_grad()
                tool_logits, emissions = model(ids, mask)
                t_loss = bce(tool_logits, tools)

                if hasattr(model, "crf"):
                    sentence_nlls = model.crf.sentence_nll(emissions, tags, mask)
                    active = mask.sum(dim=1).float().clamp(min=1.0)
                    s_loss = (sentence_nlls / active).mean()
                else:
                    s_loss = model.ce(emissions.view(-1, 26), tags.view(-1))

                loss = t_loss + s_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Inner val
            model.eval()
            val_losses = []
            with torch.no_grad():
                for b in range(0, len(val_data), 128):
                    batch = val_data[b : min(b + 128, len(val_data))]
                    ids, mask, tags, tools = collate_fn(batch)
                    tool_logits, emissions = model(ids, mask)
                    t_loss = bce(tool_logits, tools)
                    if hasattr(model, "crf"):
                        nlls = model.crf.sentence_nll(emissions, tags, mask)
                        active = mask.sum(dim=1).float().clamp(min=1.0)
                        s_loss = (nlls / active).mean()
                    else:
                        s_loss = model.ce(emissions.view(-1, 26), tags.view(-1))
                    val_losses.append((t_loss + s_loss).item())

            mean_v = float(np.mean(val_losses))
            if mean_v < best_val_loss - 0.002:
                best_val_loss = mean_v
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Select tau on Inner Val
        model.eval()
        with torch.no_grad():
            inner_probs = []
            for b in range(0, len(val_data), 128):
                batch = val_data[b : min(b + 128, len(val_data))]
                ids, mask, _, _ = collate_fn(batch)
                tool_logits, _ = model(ids, mask)
                inner_probs.append(torch.sigmoid(tool_logits).cpu().numpy())

        inner_probs = np.vstack(inner_probs)
        inner_targets = self.tool_targets[inner_val_idx]

        best_tau = 0.50
        best_acc = -1.0
        for tau in self.candidate_thresholds:
            p = (inner_probs >= tau).astype(int)
            acc = float(np.all(p == inner_targets, axis=1).mean())
            if acc > best_acc:
                best_acc = acc
                best_tau = tau

        return model, best_tau

    def evaluate_outer_val(
        self,
        model_name: str,
        model: Any,
        best_tau: float,
        outer_val_idx: List[int],
    ) -> Dict[str, Any]:
        """Evaluate model on outer validation fold."""
        val_idx_tensor = torch.tensor(outer_val_idx, device=self.device, dtype=torch.long)
        gold_tools = self.tool_targets[outer_val_idx]
        gold_tags_raw = self.gpu_data.tags[val_idx_tensor].cpu().numpy()
        mask_raw = self.gpu_data.mask[val_idx_tensor].cpu().numpy()

        all_tool_probs = []
        all_crf_tags = []
        all_greedy_tags = []
        all_constrained_tags = []

        is_unfrozen = model_name.startswith("unfrozen")
        if is_unfrozen:
            val_batch_data = [self.dataset.encoded[i] for i in outer_val_idx]

        with torch.no_grad():
            for b in range(0, len(outer_val_idx), 128):
                b_slice = slice(b, min(b + 128, len(outer_val_idx)))
                v_sub = val_idx_tensor[b_slice]

                if not is_unfrozen:
                    m_b = self.gpu_data.mask[v_sub]
                    max_l = int(m_b.sum(dim=1).max().item())
                    h_b = self.gpu_data.h[v_sub, :max_l]
                    m_b = m_b[:, :max_l]

                    if hasattr(model, "far"):
                        t_logits, _ = model.far(h_b, m_b)
                    else:
                        t_logits = model.tool_classifier(h_b[:, 0, :])
                    probs = torch.sigmoid(t_logits).cpu().numpy()
                    all_tool_probs.append(probs)

                    # Slot emissions
                    if hasattr(model, "crf"):
                        emissions = model.slot_tagger(h_b)
                        crf_p = model.crf.viterbi_decode(emissions, m_b)
                    else:
                        emissions = model.slot_classifier(h_b)
                        crf_p = None

                else:
                    batch = val_batch_data[b_slice]
                    max_l = max(item["attention_mask"].sum().item() for item in batch)
                    max_l = min(max_l, 64)
                    ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(self.device)
                    m_b = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(self.device)

                    t_logits, emissions = model(ids, m_b)
                    probs = torch.sigmoid(t_logits).cpu().numpy()
                    all_tool_probs.append(probs)

                    if hasattr(model, "crf"):
                        crf_p = model.crf.viterbi_decode(emissions, m_b)
                    else:
                        crf_p = None

                # Greedy and Constrained Softmax
                greedy_preds = torch.argmax(emissions, dim=-1)
                constrained_preds = bio_constrained_viterbi_decode(emissions, m_b, self.t_mask, self.s_mask, self.e_mask)

                m_np = m_b.cpu().numpy()
                for i in range(len(v_sub)):
                    l_i = int(m_np[i].sum())
                    if crf_p is not None:
                        all_crf_tags.append([IDX_TO_TAG.get(idx, "O") for idx in crf_p[i][:l_i]])
                    else:
                        all_crf_tags.append([IDX_TO_TAG.get(idx, "O") for idx in greedy_preds[i, :l_i].tolist()])
                    all_greedy_tags.append([IDX_TO_TAG.get(idx, "O") for idx in greedy_preds[i, :l_i].tolist()])
                    all_constrained_tags.append([IDX_TO_TAG.get(idx, "O") for idx in constrained_preds[i][:l_i]])

        all_tool_probs = np.vstack(all_tool_probs)
        pred_tools = (all_tool_probs >= best_tau).astype(int)

        # Compute sample-level metrics
        gold_spans = []
        pred_spans_crf = []
        pred_spans_greedy = []
        pred_spans_constrained = []

        for i, idx in enumerate(outer_val_idx):
            l_i = int(mask_raw[i].sum())
            g_seq = [IDX_TO_TAG.get(x, "O") for x in gold_tags_raw[i, :l_i]]
            gold_spans.append(extract_spans_strict(g_seq))
            pred_spans_crf.append(extract_spans_strict(all_crf_tags[i]))
            pred_spans_greedy.append(extract_spans_strict(all_greedy_tags[i]))
            pred_spans_constrained.append(extract_spans_strict(all_constrained_tags[i]))

        tool_matches = np.all(pred_tools == gold_tools, axis=1)
        crf_slot_matches = [g == p for g, p in zip(gold_spans, pred_spans_crf)]
        joint_em_matches = [t and s for t, s in zip(tool_matches, crf_slot_matches)]

        tp = int(np.sum((pred_tools == 1) & (gold_tools == 1)))
        fp = int(np.sum((pred_tools == 1) & (gold_tools == 0)))
        fn = int(np.sum((pred_tools == 0) & (gold_tools == 1)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0

        crf_metrics = compute_span_metrics(gold_spans, pred_spans_crf)
        greedy_metrics = compute_span_metrics(gold_spans, pred_spans_greedy)
        constrained_metrics = compute_span_metrics(gold_spans, pred_spans_constrained)

        return {
            "best_tau": best_tau,
            "tool_subset_acc": float(np.mean(tool_matches)),
            "tool_subset_acc_count": int(np.sum(tool_matches)),
            "tool_micro_f1": f1,
            "span_f1": crf_metrics["strict_f1"],
            "span_correct": crf_metrics["strict_correct"],
            "span_gold_total": crf_metrics["total_gold"],
            "joint_em": float(np.mean(joint_em_matches)),
            "joint_em_count": int(np.sum(joint_em_matches)),
            "tool_matches": tool_matches.tolist(),
            "joint_matches": joint_em_matches,
            "crf_strict_f1": crf_metrics["strict_f1"],
            "greedy_strict_f1": greedy_metrics["strict_f1"],
            "constrained_strict_f1": constrained_metrics["strict_f1"],
        }


def run_stage2d_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 2d MASTER UNIFIED PROTOCOL RUNNER on {device} ===", flush=True)

    runner = MasterProtocolRunner(device)
    models_to_run = [
        "frozen_arn",
        "frozen_cls_crf",
        "unfrozen_arn",
        "unfrozen_cls_crf",
        "unfrozen_joint_bert",
        "tfidf_knn",
        "tfidf_logistic",
    ]

    shuffles = [42, 100, 2024]
    all_fold_results = {m: [] for m in models_to_run}
    out_of_fold_sample_preds = {m: {} for m in models_to_run}
    tool_pair_fold_records = []

    # TF-IDF Setup
    tfidf = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    tfidf_X = tfidf.fit_transform(runner.user_texts).toarray()
    y_tools = runner.tool_targets

    total_folds = len(shuffles) * 5
    current_fold_idx = 0

    for s_idx, seed in enumerate(shuffles, 1):
        outer_folds = partition_clusters(runner.cluster_ids, n_folds=5, seed=seed)

        for f_idx in range(5):
            current_fold_idx += 1
            val_idx = outer_folds[f_idx]
            train_idx = [i for of in range(5) if of != f_idx for i in outer_folds[of]]

            # Nested Inner Split: 80% train, 20% val by cluster
            train_clusters = list(set(runner.cluster_ids[i] for i in train_idx))
            rng = random.Random(seed + f_idx * 17)
            rng.shuffle(train_clusters)
            n_inner_val = max(1, int(len(train_clusters) * 0.20))
            inner_val_clusters = set(train_clusters[:n_inner_val])
            inner_tr_idx = [i for i in train_idx if runner.cluster_ids[i] not in inner_val_clusters]
            inner_vl_idx = [i for i in train_idx if runner.cluster_ids[i] in inner_val_clusters]

            # Audit held-out tool pairs for this fold
            train_pairs = Counter(runner.sample_info[i]["pair_name"] for i in train_idx if runner.sample_info[i]["is_multi"])
            val_pairs = Counter(runner.sample_info[i]["pair_name"] for i in val_idx if runner.sample_info[i]["is_multi"])

            fold_pair_status = {}
            for pair, count in val_pairs.items():
                if pair is not None:
                    is_held_out = (train_pairs[pair] == 0)
                    fold_pair_status[pair] = {
                        "count": count,
                        "is_held_out": is_held_out,
                        "train_count": train_pairs[pair]
                    }

            tool_pair_fold_records.append({
                "shuffle": s_idx,
                "fold": f_idx + 1,
                "pairs": fold_pair_status
            })

            print(f"\n>>> [Fold {current_fold_idx:2d}/{total_folds}] Shuffle {s_idx} Fold {f_idx+1} (Outer Val N={len(val_idx)}, Inner Tr N={len(inner_tr_idx)}, Inner Val N={len(inner_vl_idx)})", flush=True)

            # 1. TF-IDF kNN & Logistic
            best_k = 5
            best_knn_acc = -1.0
            for k in [3, 5, 7]:
                knn_clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
                knn_clf.fit(tfidf_X[inner_tr_idx], y_tools[inner_tr_idx])
                inner_pred = knn_clf.predict(tfidf_X[inner_vl_idx])
                acc = float(np.all(inner_pred == y_tools[inner_vl_idx], axis=1).mean())
                if acc > best_knn_acc:
                    best_knn_acc = acc
                    best_k = k

            knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine")
            knn.fit(tfidf_X[train_idx], y_tools[train_idx])
            knn_preds = knn.predict(tfidf_X[val_idx])
            knn_matches = np.all(knn_preds == y_tools[val_idx], axis=1)
            knn_res = {
                "tool_subset_acc": float(np.mean(knn_matches)),
                "tool_matches": knn_matches.tolist(),
                "best_tau": best_k,
                "span_f1": 0.0,
                "joint_em": 0.0,
            }
            all_fold_results["tfidf_knn"].append(knn_res)

            # LogReg
            lr_clf = OneVsRestClassifier(LogisticRegression(max_iter=500, C=1.0, random_state=seed))
            lr_clf.fit(tfidf_X[train_idx], y_tools[train_idx])
            lr_preds = lr_clf.predict(tfidf_X[val_idx])
            lr_matches = np.all(lr_preds == y_tools[val_idx], axis=1)
            lr_res = {
                "tool_subset_acc": float(np.mean(lr_matches)),
                "tool_matches": lr_matches.tolist(),
                "best_tau": 0.5,
                "span_f1": 0.0,
                "joint_em": 0.0,
            }
            all_fold_results["tfidf_logistic"].append(lr_res)

            # 2. Frozen Models
            for m_name in ["frozen_arn", "frozen_cls_crf"]:
                m, tau = runner.train_frozen_model(m_name, inner_tr_idx, inner_vl_idx, seed=seed * 10 + f_idx)
                res = runner.evaluate_outer_val(m_name, m, tau, val_idx)
                all_fold_results[m_name].append(res)
                print(f"  {m_name:18s} | Tool Acc={res['tool_subset_acc']*100:5.2f}% (tau={tau:.2f}) | Span F1={res['span_f1']*100:5.2f}% | Joint EM={res['joint_em']*100:5.2f}%", flush=True)

                for s_i, idx in enumerate(val_idx):
                    if idx not in out_of_fold_sample_preds[m_name]:
                        out_of_fold_sample_preds[m_name][idx] = []
                    out_of_fold_sample_preds[m_name][idx].append({
                        "tool_ok": bool(res["tool_matches"][s_i]),
                        "joint_ok": bool(res["joint_matches"][s_i]),
                    })

            # 3. Unfrozen Models
            for m_name in ["unfrozen_arn", "unfrozen_cls_crf", "unfrozen_joint_bert"]:
                m, tau = runner.train_unfrozen_model(m_name, inner_tr_idx, inner_vl_idx, seed=seed * 10 + f_idx)
                res = runner.evaluate_outer_val(m_name, m, tau, val_idx)
                all_fold_results[m_name].append(res)
                print(f"  {m_name:18s} | Tool Acc={res['tool_subset_acc']*100:5.2f}% (tau={tau:.2f}) | Span F1={res['span_f1']*100:5.2f}% | Joint EM={res['joint_em']*100:5.2f}%", flush=True)

                for s_i, idx in enumerate(val_idx):
                    if idx not in out_of_fold_sample_preds[m_name]:
                        out_of_fold_sample_preds[m_name][idx] = []
                    out_of_fold_sample_preds[m_name][idx].append({
                        "tool_ok": bool(res["tool_matches"][s_i]),
                        "joint_ok": bool(res["joint_matches"][s_i]),
                    })

            # Store for baselines
            for s_i, idx in enumerate(val_idx):
                for b_name, b_res in [("tfidf_knn", knn_res), ("tfidf_logistic", lr_res)]:
                    if idx not in out_of_fold_sample_preds[b_name]:
                        out_of_fold_sample_preds[b_name][idx] = []
                    out_of_fold_sample_preds[b_name][idx].append({
                        "tool_ok": bool(b_res["tool_matches"][s_i]),
                        "joint_ok": False,
                    })

    # =========================================================================
    # STATISTICAL SYNTHESIS & AUDIT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 90, flush=True)
    print("STAGE 2d COMPREHENSIVE STATISTICAL ANALYSIS", flush=True)
    print("=" * 90, flush=True)

    # 1. Compute Raw Per-Cluster Accuracy for ARN (and others)
    unique_clusters = sorted(list(set(runner.cluster_ids)))
    raw_cluster_acc = {m: {} for m in models_to_run}

    for cl in unique_clusters:
        cl_indices = [i for i, cid in enumerate(runner.cluster_ids) if cid == cl]
        for m in models_to_run:
            matches = [pred["tool_ok"] for idx in cl_indices for pred in out_of_fold_sample_preds[m].get(idx, [])]
            raw_cluster_acc[m][cl] = float(np.mean(matches)) if matches else 0.0

    # Save raw per cluster file
    with open("ARN/results/stage2d_raw_per_cluster.json", "w", encoding="utf-8") as f:
        json.dump(raw_cluster_acc, f, indent=2)
    print("Saved raw per-cluster accuracy distribution to ARN/results/stage2d_raw_per_cluster.json", flush=True)

    arn_cl_vals = list(raw_cluster_acc["frozen_arn"].values())
    print(f"Frozen ARN Raw Per-Cluster Tool Acc (K={len(arn_cl_vals)}): Min={min(arn_cl_vals)*100:.2f}%, Max={max(arn_cl_vals)*100:.2f}%, Mean={np.mean(arn_cl_vals)*100:.2f}%, Median={np.median(arn_cl_vals)*100:.2f}%", flush=True)

    # 2. Intraclass Correlation Coefficient (rho) Computation
    sample_accs = [np.mean([p["tool_ok"] for p in out_of_fold_sample_preds["frozen_arn"][i]]) for i in range(len(runner.raw_samples))]
    k_clusters = len(unique_clusters)
    n_total = len(sample_accs)
    grand_mean = np.mean(sample_accs)

    cluster_sample_map = defaultdict(list)
    for i, cid in enumerate(runner.cluster_ids):
        cluster_sample_map[cid].append(sample_accs[i])

    ss_between = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in cluster_sample_map.values())
    ss_within = sum(sum((x - np.mean(vals)) ** 2 for x in vals) for vals in cluster_sample_map.values())
    df_between = k_clusters - 1
    df_within = n_total - k_clusters
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    k0 = (n_total - sum(len(vals) ** 2 for vals in cluster_sample_map.values()) / n_total) / df_between
    icc_rho = max(0.0, (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within))
    print(f"Computed Intraclass Correlation Coefficient (ICC rho): {icc_rho:.4f} (MSB={ms_between:.4f}, MSW={ms_within:.4f}, k0={k0:.2f})", flush=True)

    # 3. Paired Cluster Bootstrap (B=1,000)
    B = 1000
    boot_deltas = {m: [] for m in models_to_run if m != "frozen_arn"}
    cl_keys = list(unique_clusters)
    n_cl = len(cl_keys)

    for _ in range(B):
        resampled_cl = [cl_keys[random.randint(0, n_cl - 1)] for _ in range(n_cl)]
        arn_mean = np.mean([raw_cluster_acc["frozen_arn"][c] for c in resampled_cl])
        for m in boot_deltas:
            m_mean = np.mean([raw_cluster_acc[m][c] for c in resampled_cl])
            boot_deltas[m].append(m_mean - arn_mean)

    paired_summary = {}
    for m, deltas in boot_deltas.items():
        paired_summary[m] = {
            "mean_delta": float(np.mean(deltas)),
            "ci_95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        }

    # 4. Out-of-Fold Single-Tool vs Multi-Tool Breakdown
    single_indices = [i for i, s in enumerate(runner.sample_info) if not s["is_multi"]]
    multi_indices = [i for i, s in enumerate(runner.sample_info) if s["is_multi"]]

    complexity_breakdown = {}
    for m in models_to_run:
        single_matches = [p["tool_ok"] for i in single_indices for p in out_of_fold_sample_preds[m].get(i, [])]
        multi_matches = [p["tool_ok"] for i in multi_indices for p in out_of_fold_sample_preds[m].get(i, [])]
        single_joint = [p["joint_ok"] for i in single_indices for p in out_of_fold_sample_preds[m].get(i, [])]
        multi_joint = [p["joint_ok"] for i in multi_indices for p in out_of_fold_sample_preds[m].get(i, [])]

        complexity_breakdown[m] = {
            "single_tool_acc": float(np.mean(single_matches)),
            "single_joint_em": float(np.mean(single_joint)),
            "multi_tool_acc": float(np.mean(multi_matches)),
            "multi_joint_em": float(np.mean(multi_joint)),
        }

    # 5. Out-of-Fold Accuracy Per Tool Pair (All 7 Pairs)
    all_7_pairs = [
        "send_whatsapp_message + set_timer",
        "search_web + write_and_run_script",
        "run_computer_command + search_web",
        "get_current_time + send_whatsapp_message",
        "handle_smart_home + media_control",
        "get_system_diagnostics + send_whatsapp_message",
        "search_web + send_whatsapp_message"
    ]

    tool_pair_accuracy = {p: {} for p in all_7_pairs}
    for p_name in all_7_pairs:
        p_indices = [i for i, s in enumerate(runner.sample_info) if s["pair_name"] == p_name]
        for m in models_to_run:
            matches = [pred["tool_ok"] for i in p_indices for pred in out_of_fold_sample_preds[m].get(i, [])]
            tool_pair_accuracy[p_name][m] = float(np.mean(matches)) if matches else 0.0

    # 6. Held-Out CRF vs Softmax Decomposition Across Folds
    held_out_decomp = {
        "greedy_strict_f1_mean": float(np.mean([f["greedy_strict_f1"] for f in all_fold_results["frozen_cls_crf"]])),
        "constrained_strict_f1_mean": float(np.mean([f["constrained_strict_f1"] for f in all_fold_results["frozen_cls_crf"]])),
        "crf_strict_f1_mean": float(np.mean([f["crf_strict_f1"] for f in all_fold_results["frozen_cls_crf"]])),
    }
    total_crf_delta = held_out_decomp["crf_strict_f1_mean"] - held_out_decomp["greedy_strict_f1_mean"]
    grammar_delta = held_out_decomp["constrained_strict_f1_mean"] - held_out_decomp["greedy_strict_f1_mean"]
    learned_delta = held_out_decomp["crf_strict_f1_mean"] - held_out_decomp["constrained_strict_f1_mean"]
    held_out_decomp["pct_from_grammar"] = (grammar_delta / max(total_crf_delta, 1e-8)) * 100
    held_out_decomp["pct_from_learned"] = (learned_delta / max(total_crf_delta, 1e-8)) * 100

    # Summary table across models
    overall_summary = {}
    for m in models_to_run:
        accs = [f["tool_subset_acc"] for f in all_fold_results[m]]
        spans = [f["span_f1"] for f in all_fold_results[m]]
        joints = [f["joint_em"] for f in all_fold_results[m]]

        # Cluster bootstrap CI
        cl_accs = list(raw_cluster_acc[m].values())
        boot_means = [np.mean([cl_accs[random.randint(0, len(cl_accs)-1)] for _ in range(len(cl_accs))]) for _ in range(1000)]

        overall_summary[m] = {
            "mean_tool_acc": float(np.mean(accs)),
            "sd_tool_acc": float(np.std(accs, ddof=1)),
            "ci_tool_acc": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
            "mean_span_f1": float(np.mean(spans)),
            "sd_span_f1": float(np.std(spans, ddof=1)),
            "mean_joint_em": float(np.mean(joints)),
            "sd_joint_em": float(np.std(joints, ddof=1)),
        }

    output_data = {
        "overall_summary": overall_summary,
        "paired_cluster_deltas_vs_arn": paired_summary,
        "single_vs_multitool_out_of_fold": complexity_breakdown,
        "tool_pair_accuracy_out_of_fold": tool_pair_accuracy,
        "tool_pair_fold_status": tool_pair_fold_records,
        "held_out_crf_decomposition": held_out_decomp,
        "intraclass_correlation_rho": icc_rho,
        "raw_cluster_distribution": {
            m: {
                "min": float(min(raw_cluster_acc[m].values())),
                "max": float(max(raw_cluster_acc[m].values())),
                "mean": float(np.mean(list(raw_cluster_acc[m].values()))),
                "median": float(np.median(list(raw_cluster_acc[m].values()))),
            } for m in models_to_run
        }
    }

    out_file = Path("ARN/results/stage2d_master_protocol.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved complete Stage 2d master protocol results to {out_file}", flush=True)


if __name__ == "__main__":
    run_stage2d_evaluation()
