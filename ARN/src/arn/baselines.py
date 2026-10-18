"""
Baseline Classifiers and Decoders for ARN (Phase 3).

Provides:
- MostFrequentBaseline (predicts most frequent label combination in train)
- FirstTokenPriorBaseline (predicts tool distribution conditioned on first token)
- TfidfKnnBaseline (fold-isolated TF-IDF + soft neighbor averaging + inner-tuned threshold)
- TfidfLogisticBaseline (fold-isolated TF-IDF + balanced OvR Logistic Regression + inner-tuned threshold)
- Slot decoding utilities:
    * decode_greedy (unconstrained token-wise argmax)
    * decode_masked_viterbi (BIO-constrained Viterbi with zero learned transitions)
    * decode_crf (full CRF Viterbi with learned transitions)
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

from .tokenizer import TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from .crf import CRF, build_bio_transition_mask
from .far import FactoredAttentionRouter
from transformers import AutoModel
import torch.nn as nn


class JointBERT(nn.Module):
    """
    Standard JointBERT baseline (Chen et al., 2019):
    - Tool routing: Linear projection on [CLS] token representation
    - Slot extraction: Linear projection to tag logits trained with token Cross-Entropy (Softmax, no CRF)
    """
    def __init__(
        self,
        num_tools: int = 18,
        num_tags: int = 26,
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
        model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    ):
        super().__init__()
        self.num_tools = num_tools
        self.num_tags = num_tags
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.tool_classifier = nn.Linear(hidden_dim, num_tools)
        self.slot_tagger = nn.Linear(hidden_dim, num_tags)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tool_labels: Optional[torch.Tensor] = None,
        tags: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls_rep = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_rep)
        emissions = self.slot_tagger(h)

        loss_tool: Optional[torch.Tensor] = None
        loss_slot: Optional[torch.Tensor] = None

        if tool_labels is not None:
            loss_tool = self.bce(tool_logits, tool_labels)

        if tags is not None:
            B, N, M = emissions.shape
            loss_slot = self.ce(emissions.view(-1, M), tags.view(-1))

        if tool_labels is not None or tags is not None:
            return loss_tool, loss_slot

        return tool_logits, emissions


class CLSCRFModel(nn.Module):
    """
    [CLS] Classifier + Linear-Chain CRF:
    - Tool routing: Linear projection on [CLS] token representation
    - Slot extraction: Linear projection to tag emissions decoded via hard BIO-masked CRF
    """
    def __init__(
        self,
        num_tools: int = 18,
        num_tags: int = 26,
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
        model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    ):
        super().__init__()
        self.num_tools = num_tools
        self.num_tags = num_tags
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.tool_classifier = nn.Linear(hidden_dim, num_tools)
        self.slot_tagger = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tool_labels: Optional[torch.Tensor] = None,
        tags: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls_rep = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_rep)
        emissions = self.slot_tagger(h)

        loss_tool: Optional[torch.Tensor] = None
        loss_slot: Optional[torch.Tensor] = None

        if tool_labels is not None:
            loss_tool = self.bce(tool_logits, tool_labels)

        if tags is not None:
            nlls = self.crf.sentence_nll(emissions, tags, attention_mask)
            seq_lens = attention_mask.sum(dim=1).float().clamp(min=1.0)
            loss_slot = (nlls / seq_lens).mean()

        if tool_labels is not None or tags is not None:
            return loss_tool, loss_slot

        return tool_logits, emissions


class FARNoCRFModel(nn.Module):
    """
    Factored Attention Router (FAR) + Softmax Slot Tagger (No CRF):
    - Tool routing: Factored Attention Router (FAR)
    - Slot extraction: Linear projection trained with Cross-Entropy Loss (Softmax)
    """
    def __init__(
        self,
        num_tools: int = 18,
        num_tags: int = 26,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    ):
        super().__init__()
        self.num_tools = num_tools
        self.num_tags = num_tags
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.far = FactoredAttentionRouter(num_tools=num_tools, hidden_dim=hidden_dim, dropout=dropout)
        self.slot_tagger = nn.Linear(hidden_dim, num_tags)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tool_labels: Optional[torch.Tensor] = None,
        tags: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        tool_logits, _ = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)

        loss_tool: Optional[torch.Tensor] = None
        loss_slot: Optional[torch.Tensor] = None

        if tool_labels is not None:
            loss_tool = self.bce(tool_logits, tool_labels)

        if tags is not None:
            B, N, M = emissions.shape
            loss_slot = self.ce(emissions.view(-1, M), tags.view(-1))

        if tool_labels is not None or tags is not None:
            return loss_tool, loss_slot

        return tool_logits, emissions


class MostFrequentBaseline:
    """Predicts the most frequent multi-label tool combination seen in the training split."""
    def __init__(self, num_tools: int = 18):
        self.num_tools = num_tools
        self.most_frequent_vector: Optional[np.ndarray] = None

    def fit(self, X_texts: List[str], y_tools: np.ndarray) -> "MostFrequentBaseline":
        tuples = [tuple(y_tools[i].astype(int)) for i in range(len(y_tools))]
        most_common = Counter(tuples).most_common(1)[0][0]
        self.most_frequent_vector = np.array(most_common, dtype=np.float32)
        return self

    def predict(self, X_texts: List[str]) -> np.ndarray:
        assert self.most_frequent_vector is not None, "Model must be fitted before predict"
        return np.tile(self.most_frequent_vector, (len(X_texts), 1))


class FirstTokenPriorBaseline:
    """
    Predicts tools based on the empirical tool prior conditioned on the utterance's first token.
    Falls back to the global tool prior if the first token was unseen in training.
    """
    def __init__(self, num_tools: int = 18):
        self.num_tools = num_tools
        self.token_priors: Dict[str, np.ndarray] = {}
        self.global_prior: np.ndarray = np.zeros(num_tools, dtype=np.float32)

    def fit(self, X_texts: List[str], y_tools: np.ndarray) -> "FirstTokenPriorBaseline":
        self.global_prior = y_tools.mean(axis=0)
        token_counts: Dict[str, int] = Counter()
        token_tool_sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.num_tools, dtype=np.float32))

        for text, y in zip(X_texts, y_tools):
            tokens = text.strip().lower().split()
            first_tok = tokens[0] if tokens else "<EMPTY>"
            token_counts[first_tok] += 1
            token_tool_sums[first_tok] += y

        for tok, count in token_counts.items():
            self.token_priors[tok] = token_tool_sums[tok] / count

        return self

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        probs = np.zeros((len(X_texts), self.num_tools), dtype=np.float32)
        for i, text in enumerate(X_texts):
            tokens = text.strip().lower().split()
            first_tok = tokens[0] if tokens else "<EMPTY>"
            probs[i] = self.token_priors.get(first_tok, self.global_prior)
        return probs

    def predict(self, X_texts: List[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_texts) >= threshold).astype(np.float32)


class TfidfKnnBaseline:
    """
    Fold-isolated TF-IDF + soft neighbor averaging kNN classifier.
    Fits the vectorizer strictly on training data (no leakage).
    Tunes threshold tau on inner validation.
    """
    def __init__(
        self,
        n_neighbors: int = 5,
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2
    ):
        self.n_neighbors = n_neighbors
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=None  # Do NOT use stop_words='english' for short imperative commands
        )
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.y_train: Optional[np.ndarray] = None
        self.best_tau: float = 0.50

    def fit(self, X_train_texts: List[str], y_train: np.ndarray) -> "TfidfKnnBaseline":
        X_tr = self.vectorizer.fit_transform(X_train_texts)
        self.nn.fit(X_tr)
        self.y_train = y_train.copy()
        return self

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        assert self.y_train is not None, "Must fit before predict_proba"
        X_feat = self.vectorizer.transform(X_texts)
        dists, indices = self.nn.kneighbors(X_feat)
        # Average tool vectors across the k nearest neighbors
        probs = np.array([self.y_train[idx].mean(axis=0) for idx in indices], dtype=np.float32)
        return probs

    def tune_threshold(
        self,
        X_val_texts: List[str],
        y_val: np.ndarray,
        candidate_taus: Optional[List[float]] = None
    ) -> float:
        if candidate_taus is None:
            candidate_taus = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        probs = self.predict_proba(X_val_texts)
        best_acc = -1.0
        best_t = 0.50
        for tau in candidate_taus:
            preds = (probs >= tau).astype(np.float32)
            acc = float(np.all(preds == y_val, axis=1).mean())
            if acc > best_acc:
                best_acc = acc
                best_t = tau
        self.best_tau = best_t
        return best_t

    def predict(self, X_texts: List[str], tau: Optional[float] = None) -> np.ndarray:
        t = self.best_tau if tau is None else tau
        return (self.predict_proba(X_texts) >= t).astype(np.float32)


class TfidfLogisticBaseline:
    """
    Fold-isolated TF-IDF + class-balanced Logistic Regression (One-vs-Rest).
    Fits vectorizer strictly on training data.
    Tunes threshold tau on inner validation.
    """
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 300,
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2
    ):
        self.C = C
        self.max_iter = max_iter
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=None
        )
        self.classifiers: Dict[int, LogisticRegression] = {}
        self.num_tools: int = 18
        self.best_tau: float = 0.50

    def fit(self, X_train_texts: List[str], y_train: np.ndarray) -> "TfidfLogisticBaseline":
        self.num_tools = y_train.shape[1]
        X_tr = self.vectorizer.fit_transform(X_train_texts)

        for k in range(self.num_tools):
            if y_train[:, k].sum() > 0:
                clf = LogisticRegression(
                    C=self.C,
                    max_iter=self.max_iter,
                    class_weight="balanced",
                    solver="lbfgs"
                )
                clf.fit(X_tr, y_train[:, k])
                self.classifiers[k] = clf

        return self

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        X_feat = self.vectorizer.transform(X_texts)
        probs = np.zeros((len(X_texts), self.num_tools), dtype=np.float32)

        for k, clf in self.classifiers.items():
            if len(clf.classes_) > 1:
                probs[:, k] = clf.predict_proba(X_feat)[:, 1]
            else:
                probs[:, k] = float(clf.classes_[0])

        return probs

    def tune_threshold(
        self,
        X_val_texts: List[str],
        y_val: np.ndarray,
        candidate_taus: Optional[List[float]] = None
    ) -> float:
        if candidate_taus is None:
            candidate_taus = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        probs = self.predict_proba(X_val_texts)
        best_acc = -1.0
        best_t = 0.50
        for tau in candidate_taus:
            preds = (probs >= tau).astype(np.float32)
            acc = float(np.all(preds == y_val, axis=1).mean())
            if acc > best_acc:
                best_acc = acc
                best_t = tau
        self.best_tau = best_t
        return best_t

    def predict(self, X_texts: List[str], tau: Optional[float] = None) -> np.ndarray:
        t = self.best_tau if tau is None else tau
        return (self.predict_proba(X_texts) >= t).astype(np.float32)


# ── Slot Tag Decoding Utilities ──────────────────────────────────────────────

def decode_greedy(emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
    """
    Unconstrained greedy token-wise argmax decoding.
    Does NOT enforce BIO transition constraints.
    """
    preds = emissions.argmax(dim=-1).cpu().tolist()
    pad_idx = TAG_TO_IDX["<PAD>"]
    out: List[List[int]] = []
    for b in range(len(preds)):
        seq_len = int(mask[b].sum().item())
        row = preds[b][:seq_len] + [pad_idx] * (len(preds[b]) - seq_len)
        out.append(row)
    return out


def decode_masked_viterbi(emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
    """
    Viterbi decoding with hard BIO mask and ZERO learned transitions (pure structural baseline).
    Enforces that forbidden BIO transitions receive -1e4 score.
    """
    device = emissions.device
    num_tags = emissions.shape[-1]
    trans_mask, start_mask, end_mask = build_bio_transition_mask(TAG_TO_IDX)
    trans_mask = trans_mask.to(device)
    start_mask = start_mask.to(device)
    end_mask = end_mask.to(device)

    # Effective transitions: 0 where valid, -1e4 where invalid
    trans = torch.where(trans_mask, torch.zeros_like(trans_mask, dtype=torch.float32), torch.full_like(trans_mask, -1e4, dtype=torch.float32))
    start_t = torch.where(start_mask, torch.zeros_like(start_mask, dtype=torch.float32), torch.full_like(start_mask, -1e4, dtype=torch.float32))
    end_t = torch.where(end_mask, torch.zeros_like(end_mask, dtype=torch.float32), torch.full_like(end_mask, -1e4, dtype=torch.float32))

    batch_size, seq_len, _ = emissions.shape
    viterbi_vars = start_t.unsqueeze(0) + emissions[:, 0, :]
    backpointers: List[torch.Tensor] = []

    for t in range(1, seq_len):
        active = mask[:, t].bool().unsqueeze(1)
        next_tag_var = viterbi_vars.unsqueeze(2) + trans.unsqueeze(0)
        max_vars, bptrs = torch.max(next_tag_var, dim=1)
        max_vars = max_vars + emissions[:, t, :]
        viterbi_vars = torch.where(active, max_vars, viterbi_vars)
        backpointers.append(bptrs)

    terminal_vars = viterbi_vars + end_t.unsqueeze(0)
    best_tag_ids = torch.argmax(terminal_vars, dim=1)

    pad_val = TAG_TO_IDX["<PAD>"]
    best_paths: List[List[int]] = []

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
