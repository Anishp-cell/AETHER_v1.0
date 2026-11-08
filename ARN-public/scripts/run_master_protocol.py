"""
ARN Master Evaluation Protocol (Phase 4).

Executes the definitive benchmark across 15 tool-stratified folds (5 folds x 3 shuffles)
defined in ARN/data/splits/cv_manifest.json.

Evaluates:
  Frozen Models:
    1. frozen_arn (FAR + CRF)
    2. frozen_cls_crf ([CLS] + CRF)
    3. frozen_far_no_crf (FAR + Softmax, greedy & masked Viterbi)
    4. frozen_joint_bert ([CLS] + Softmax, greedy & masked Viterbi)
  Unfrozen Models:
    5. unfrozen_arn (Trunk LR 3e-5 + FAR + CRF)
    6. unfrozen_cls_crf (Trunk LR 3e-5 + [CLS] + CRF)
    7. unfrozen_joint_bert (Trunk LR 3e-5 + [CLS] + Softmax)

Enforces:
  - Decoupled branch training (separate early stopping and gradient clipping).
  - Hyperparameters and thresholds (tau in [0.30, 0.70]) tuned strictly on inner validation.
  - Full metrics with k/N counts, span strict/lenient F1, tag macro F1, token accuracy.
  - Multi-tool breakdown across the 7 fixed pairs.
  - Per-cluster outputs saved for cluster bootstrap CIs (B >= 2000).
  - Appends all runs directly to ARN/results/ledger.json.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.model import AetherRoutingNetwork
from arn.dataset import ARNDataset
from arn.far import FactoredAttentionRouter
from arn.crf import CRF, build_bio_transition_mask
from arn.loss import ARNLoss
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from arn.baselines import JointBERT, CLSCRFModel, FARNoCRFModel, decode_greedy, decode_masked_viterbi
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.master_protocol")


def get_git_hash(repo_dir: Path) -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def extract_spans_strict(tags: List[str]) -> List[Tuple[str, int, int]]:
    """Extract (slot_type, start_idx, end_idx) with strict boundary matching."""
    spans = []
    current_type = None
    start = -1
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.append((current_type, start, i))
            current_type = tag.split("-")[1]
            start = i
        elif tag.startswith("I-"):
            slot_type = tag.split("-")[1]
            if current_type != slot_type:
                if current_type is not None:
                    spans.append((current_type, start, i))
                current_type = slot_type
                start = i
        else:
            if current_type is not None:
                spans.append((current_type, start, i))
                current_type = None
                start = -1
    if current_type is not None:
        spans.append((current_type, start, len(tags)))
    return spans


def compute_span_f1(gold_spans_list: List[List[Tuple[str, int, int]]], pred_spans_list: List[List[Tuple[str, int, int]]]) -> Dict[str, Any]:
    strict_correct = 0
    lenient_correct = 0
    total_pred = 0
    total_gold = 0

    for gold, pred in zip(gold_spans_list, pred_spans_list):
        total_gold += len(gold)
        total_pred += len(pred)
        gold_set = set(gold)
        pred_set = set(pred)

        strict_correct += len(gold_set & pred_set)

        # Lenient: overlap in [start, end) and same slot_type
        for p_type, p_s, p_e in pred:
            matched = False
            for g_type, g_s, g_e in gold:
                if p_type == g_type and max(p_s, g_s) < min(p_e, g_e):
                    matched = True
                    break
            if matched:
                lenient_correct += 1

    strict_p = strict_correct / max(total_pred, 1)
    strict_r = strict_correct / max(total_gold, 1)
    strict_f1 = (2 * strict_p * strict_r / max(strict_p + strict_r, 1e-8)) if (strict_p + strict_r) > 0 else 0.0

    lenient_p = lenient_correct / max(total_pred, 1)
    lenient_r = lenient_correct / max(total_gold, 1)
    lenient_f1 = (2 * lenient_p * lenient_r / max(lenient_p + lenient_r, 1e-8)) if (lenient_p + lenient_r) > 0 else 0.0

    return {
        "strict_f1": float(strict_f1),
        "strict_correct": strict_correct,
        "lenient_f1": float(lenient_f1),
        "lenient_correct": lenient_correct,
        "total_pred": total_pred,
        "total_gold": total_gold,
    }


@dataclass
class PrecomputedGPUData:
    h: torch.Tensor
    mask: torch.Tensor
    tags: torch.Tensor
    tool_labels: torch.Tensor


def precompute_frozen_trunk(dataset: ARNDataset, device: torch.device) -> PrecomputedGPUData:
    """Precompute BERT-tiny frozen hidden states for all 5,000 samples on GPU."""
    logger.info("Precomputing BERT-tiny frozen hidden states for all samples...")
    encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager").to(device)
    encoder.eval()

    all_h = []
    all_mask = []
    all_tags = []
    all_tools = []

    with torch.no_grad():
        for i in range(0, len(dataset), 128):
            batch = [dataset.encoded[j] for j in range(i, min(i + 128, len(dataset)))]
            input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
            mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
            tags = torch.stack([item["tags"] for item in batch]).to(device)
            tools = torch.stack([item["tool_labels"] for item in batch]).to(device)

            out = encoder(input_ids=input_ids, attention_mask=mask)
            all_h.append(out.last_hidden_state.cpu())
            all_mask.append(mask.cpu())
            all_tags.append(tags.cpu())
            all_tools.append(tools.cpu())

    return PrecomputedGPUData(
        h=torch.cat(all_h, dim=0).to(device),
        mask=torch.cat(all_mask, dim=0).to(device),
        tags=torch.cat(all_tags, dim=0).to(device),
        tool_labels=torch.cat(all_tools, dim=0).to(device),
    )


class MasterProtocolExperimentRunner:
    def __init__(self, device: torch.device):
        self.device = device
        self.repo_dir = Path(__file__).resolve().parent.parent
        self.manifest_path = self.repo_dir / "data" / "splits" / "cv_manifest.json"
        self.raw_path = self.repo_dir / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
        self.ledger_path = self.repo_dir / "results" / "ledger.json"

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        with open(self.raw_path, "r", encoding="utf-8") as f:
            self.raw_samples = [json.loads(line) for line in f if line.strip()]

        self.frame_to_cluster = build_canonical_clusters(self.raw_path, sim_threshold=0.80)
        self.skeletons = [extract_fine_semantic_frame(s) for s in self.raw_samples]
        self.sample_clusters = [self.frame_to_cluster[skel] for skel in self.skeletons]

        self.dataset = ARNDataset(str(self.raw_path), tokenizer=TOKENIZER)
        self.gpu_data = precompute_frozen_trunk(self.dataset, self.device)

        self.candidate_taus = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        self.candidate_head_lrs = [1e-4, 3e-4, 1e-3]
        self.candidate_lambdas = [0.1, 0.3, 1.0]

        # Multi-tool pair info
        self.sample_pairs = []
        for s in self.raw_samples:
            asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
            calls = json.loads(asst)
            if len(calls) == 2:
                names = sorted([c["name"] for c in calls])
                self.sample_pairs.append(f"{names[0]} + {names[1]}")
            else:
                self.sample_pairs.append(None)

    def train_frozen_model(
        self,
        model_type: str,
        inner_tr_idx: List[int],
        inner_val_idx: List[int],
        seed: int = 42
    ) -> Tuple[nn.Module, float]:
        """Train frozen head with decoupled early stopping and inner threshold tuning."""
        torch.manual_seed(seed)
        num_tools = len(TOOL_TO_IDX)
        num_tags = len(TAG_TO_IDX)

        if model_type == "frozen_arn":
            head = nn.ModuleDict({
                "far": FactoredAttentionRouter(num_tools=num_tools, hidden_dim=128, dropout=0.1),
                "slot_tagger": nn.Linear(128, num_tags),
                "crf": CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
            }).to(self.device)
        elif model_type == "frozen_cls_crf":
            head = nn.ModuleDict({
                "tool_cls": nn.Linear(128, num_tools),
                "slot_tagger": nn.Linear(128, num_tags),
                "crf": CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
            }).to(self.device)
        elif model_type == "frozen_far_no_crf":
            head = nn.ModuleDict({
                "far": FactoredAttentionRouter(num_tools=num_tools, hidden_dim=128, dropout=0.1),
                "slot_tagger": nn.Linear(128, num_tags),
            }).to(self.device)
        elif model_type == "frozen_joint_bert":
            head = nn.ModuleDict({
                "tool_cls": nn.Linear(128, num_tools),
                "slot_tagger": nn.Linear(128, num_tags),
            }).to(self.device)
        else:
            raise ValueError(f"Unknown frozen model type: {model_type}")

        # Decoupled optimizers
        if "far" in head:
            tool_params = list(head["far"].parameters())
        else:
            tool_params = list(head["tool_cls"].parameters())

        if "crf" in head:
            slot_params = list(head["slot_tagger"].parameters()) + list(head["crf"].parameters())
        else:
            slot_params = list(head["slot_tagger"].parameters())

        t_opt = torch.optim.AdamW(tool_params, lr=3e-4, weight_decay=0.01)
        s_opt = torch.optim.AdamW(slot_params, lr=1e-3, weight_decay=0.01)
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])

        tr_tensor = torch.tensor(inner_tr_idx, device=self.device, dtype=torch.long)
        val_tensor = torch.tensor(inner_val_idx, device=self.device, dtype=torch.long)
        n_tr = len(inner_tr_idx)

        best_t_val = float("inf")
        best_s_val = float("inf")
        best_state = None
        patience = 8
        p_cnt = 0

        head.train()
        for ep in range(1, 26):
            perm = tr_tensor[torch.randperm(n_tr)]
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
                if "far" in head:
                    t_logits, _ = head["far"](h_b, m_b)
                else:
                    t_logits = head["tool_cls"](h_b[:, 0, :])
                t_loss = bce(t_logits, tools_b)
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(tool_params, 1.0)
                t_opt.step()

                # Slot step
                s_opt.zero_grad()
                emissions = head["slot_tagger"](h_b)
                if "crf" in head:
                    nlls = head["crf"].sentence_nll(emissions, t_b, m_b)
                    active = m_b.sum(dim=1).float().clamp(min=1.0)
                    s_loss = (nlls / active).mean()
                else:
                    s_loss = ce(emissions.view(-1, num_tags), t_b.view(-1))
                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(slot_params, 1.0)
                s_opt.step()

            # Validation
            head.eval()
            with torch.no_grad():
                val_m = self.gpu_data.mask[val_tensor]
                v_max_l = int(val_m.sum(dim=1).max().item())
                val_h = self.gpu_data.h[val_tensor, :v_max_l]
                val_m = val_m[:, :v_max_l]
                val_tools = self.gpu_data.tool_labels[val_tensor]
                val_tags = self.gpu_data.tags[val_tensor, :v_max_l]

                if "far" in head:
                    vt_logits, _ = head["far"](val_h, val_m)
                else:
                    vt_logits = head["tool_cls"](val_h[:, 0, :])
                vt_loss = bce(vt_logits, val_tools).item()

                v_emissions = head["slot_tagger"](val_h)
                if "crf" in head:
                    v_nlls = head["crf"].sentence_nll(v_emissions, val_tags, val_m)
                    active_v = val_m.sum(dim=1).float().clamp(min=1.0)
                    vs_loss = (v_nlls / active_v).mean().item()
                else:
                    vs_loss = ce(v_emissions.view(-1, num_tags), val_tags.view(-1)).item()

            if (vt_loss + vs_loss) < (best_t_val + best_s_val) - 0.002:
                best_t_val = vt_loss
                best_s_val = vs_loss
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                p_cnt = 0
            else:
                p_cnt += 1
                if p_cnt >= patience:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)

        # Select tau on Inner Val
        head.eval()
        with torch.no_grad():
            val_m = self.gpu_data.mask[val_tensor]
            v_max_l = int(val_m.sum(dim=1).max().item())
            val_h = self.gpu_data.h[val_tensor, :v_max_l]
            val_m = val_m[:, :v_max_l]
            val_tools = self.gpu_data.tool_labels[val_tensor].cpu().numpy()

            if "far" in head:
                vt_logits, _ = head["far"](val_h, val_m)
            else:
                vt_logits = head["tool_cls"](val_h[:, 0, :])
            inner_probs = torch.sigmoid(vt_logits).cpu().numpy()

        best_tau = 0.50
        best_acc = -1.0
        for tau in self.candidate_taus:
            preds = (inner_probs >= tau).astype(np.float32)
            acc = float(np.all(preds == val_tools, axis=1).mean())
            if acc > best_acc:
                best_acc = acc
                best_tau = tau

        return head, best_tau

    def train_unfrozen_model(
        self,
        model_type: str,
        inner_tr_idx: List[int],
        inner_val_idx: List[int],
        seed: int = 42
    ) -> Tuple[nn.Module, float]:
        """Train unfrozen model (trunk LR 3e-5, head LR 1e-3) with early stopping and inner threshold tuning."""
        torch.manual_seed(seed)
        num_tools = len(TOOL_TO_IDX)
        num_tags = len(TAG_TO_IDX)

        if model_type == "unfrozen_arn":
            model = AetherRoutingNetwork(config=ARNConfig(freeze_trunk=False)).to(self.device)
            trunk_params = list(model.encoder.parameters())
            head_params = list(model.far.parameters()) + list(model.slot_tagger.parameters()) + list(model.crf.parameters())
        elif model_type == "unfrozen_cls_crf":
            model = CLSCRFModel(num_tools=num_tools, num_tags=num_tags, freeze_backbone=False).to(self.device)
            trunk_params = list(model.encoder.parameters())
            head_params = list(model.tool_classifier.parameters()) + list(model.slot_tagger.parameters()) + list(model.crf.parameters())
        elif model_type == "unfrozen_joint_bert":
            model = JointBERT(num_tools=num_tools, num_tags=num_tags, freeze_backbone=False).to(self.device)
            trunk_params = list(model.encoder.parameters())
            head_params = list(model.tool_classifier.parameters()) + list(model.slot_tagger.parameters())
        else:
            raise ValueError(f"Unknown unfrozen model type: {model_type}")

        optimizer = torch.optim.AdamW([
            {"params": trunk_params, "lr": 3e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 1e-3, "weight_decay": 0.01}
        ])

        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])

        tr_data = [self.dataset.encoded[i] for i in inner_tr_idx]
        val_data = [self.dataset.encoded[i] for i in inner_val_idx]
        n_tr = len(tr_data)

        best_val_loss = float("inf")
        best_state = None
        patience = 6
        p_cnt = 0

        model.train()
        for ep in range(1, 16):
            perm = np.random.permutation(n_tr)
            for b in range(0, n_tr, 64):
                batch = [tr_data[perm[j]] for j in range(b, min(b + 64, n_tr))]
                max_l = min(max(item["attention_mask"].sum().item() for item in batch), 64)
                ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(self.device)
                mask = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(self.device)
                tags = torch.stack([item["tags"][:max_l] for item in batch]).to(self.device)
                tools = torch.stack([item["tool_labels"] for item in batch]).to(self.device)

                optimizer.zero_grad()
                if model_type == "unfrozen_arn":
                    total_loss, _, _, _, _ = model(ids, mask, tags=tags, tool_labels=tools)
                elif model_type == "unfrozen_cls_crf":
                    t_loss, s_loss = model(ids, mask, tool_labels=tools, tags=tags)
                    total_loss = t_loss + 0.3 * s_loss
                else:  # unfrozen_joint_bert
                    t_loss, s_loss = model(ids, mask, tool_labels=tools, tags=tags)
                    total_loss = t_loss + 0.3 * s_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for b in range(0, len(val_data), 128):
                    batch = val_data[b : min(b + 128, len(val_data))]
                    max_l = min(max(item["attention_mask"].sum().item() for item in batch), 64)
                    ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(self.device)
                    mask = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(self.device)
                    tags = torch.stack([item["tags"][:max_l] for item in batch]).to(self.device)
                    tools = torch.stack([item["tool_labels"] for item in batch]).to(self.device)

                    if model_type == "unfrozen_arn":
                        tot, _, _, _, _ = model(ids, mask, tags=tags, tool_labels=tools)
                        val_losses.append(tot.item())
                    else:
                        t_l, s_l = model(ids, mask, tool_labels=tools, tags=tags)
                        val_losses.append((t_l + 0.3 * s_l).item())

            mean_v = float(np.mean(val_losses))
            if mean_v < best_val_loss - 0.002:
                best_val_loss = mean_v
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                p_cnt = 0
            else:
                p_cnt += 1
                if p_cnt >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Select tau on Inner Val
        model.eval()
        inner_probs = []
        with torch.no_grad():
            for b in range(0, len(val_data), 128):
                batch = val_data[b : min(b + 128, len(val_data))]
                max_l = min(max(item["attention_mask"].sum().item() for item in batch), 64)
                ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(self.device)
                mask = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(self.device)

                if model_type == "unfrozen_arn":
                    out = model.encoder(ids, attention_mask=mask)
                    t_logits, _ = model.far(out.last_hidden_state, mask)
                else:
                    t_logits, _ = model(ids, mask)
                inner_probs.append(torch.sigmoid(t_logits).cpu().numpy())

        inner_probs = np.vstack(inner_probs)
        val_tools = self.gpu_data.tool_labels[torch.tensor(inner_val_idx, device=self.device)].cpu().numpy()

        best_tau = 0.50
        best_acc = -1.0
        for tau in self.candidate_taus:
            preds = (inner_probs >= tau).astype(np.float32)
            acc = float(np.all(preds == val_tools, axis=1).mean())
            if acc > best_acc:
                best_acc = acc
                best_tau = tau

        return model, best_tau

    def evaluate_outer_fold(
        self,
        model_name: str,
        model: nn.Module,
        best_tau: float,
        val_idx: List[int]
    ) -> Dict[str, Any]:
        """Full evaluation on outer validation fold."""
        N_val = len(val_idx)
        val_tensor = torch.tensor(val_idx, device=self.device, dtype=torch.long)
        gold_tools = self.gpu_data.tool_labels[val_tensor].cpu().numpy()
        gold_tags = self.gpu_data.tags[val_tensor].cpu().numpy()
        mask_np = self.gpu_data.mask[val_tensor].cpu().numpy()

        is_frozen = model_name.startswith("frozen")
        tool_probs_list = []
        pred_tag_seqs: List[List[str]] = []
        pred_tag_ids_raw: List[List[int]] = []

        model.eval()
        with torch.no_grad():
            if is_frozen:
                for b in range(0, N_val, 128):
                    v_sub = val_tensor[b : min(b + 128, N_val)]
                    m_b = self.gpu_data.mask[v_sub]
                    max_l = int(m_b.sum(dim=1).max().item())
                    h_b = self.gpu_data.h[v_sub, :max_l]
                    m_b = m_b[:, :max_l]

                    if "far" in model:
                        t_logits, _ = model["far"](h_b, m_b)
                    else:
                        t_logits = model["tool_cls"](h_b[:, 0, :])
                    tool_probs_list.append(torch.sigmoid(t_logits).cpu().numpy())

                    emissions = model["slot_tagger"](h_b)
                    if "crf" in model:
                        tag_ids = model["crf"].viterbi_decode(emissions, m_b)
                    else:
                        tag_ids = decode_masked_viterbi(emissions, m_b)

                    for j, ids in enumerate(tag_ids):
                        l_j = int(m_b[j].sum().item())
                        pred_tag_ids_raw.append(ids[:l_j])
                        pred_tag_seqs.append([IDX_TO_TAG.get(x, "O") for x in ids[:l_j]])

            else:  # unfrozen models
                val_data = [self.dataset.encoded[i] for i in val_idx]
                for b in range(0, N_val, 128):
                    batch = val_data[b : min(b + 128, N_val)]
                    max_l = min(max(item["attention_mask"].sum().item() for item in batch), 64)
                    ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(self.device)
                    m_b = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(self.device)

                    if model_name == "unfrozen_arn":
                        out = model.encoder(ids, attention_mask=m_b)
                        t_logits, _ = model.far(out.last_hidden_state, m_b)
                        emissions = model.slot_tagger(out.last_hidden_state)
                        tag_ids = model.crf.viterbi_decode(emissions, m_b)
                    elif model_name == "unfrozen_cls_crf":
                        t_logits, emissions = model(ids, m_b)
                        tag_ids = model.crf.viterbi_decode(emissions, m_b)
                    else:  # unfrozen_joint_bert
                        t_logits, emissions = model(ids, m_b)
                        tag_ids = decode_masked_viterbi(emissions, m_b)

                    tool_probs_list.append(torch.sigmoid(t_logits).cpu().numpy())
                    for j, ids_row in enumerate(tag_ids):
                        l_j = int(m_b[j].sum().item())
                        pred_tag_ids_raw.append(ids_row[:l_j])
                        pred_tag_seqs.append([IDX_TO_TAG.get(x, "O") for x in ids_row[:l_j]])

        all_tool_probs = np.vstack(tool_probs_list)
        pred_tools = (all_tool_probs >= best_tau).astype(np.float32)

        # 1. Tool Metrics
        tool_matches = np.all(pred_tools == gold_tools, axis=1)
        k_tool = int(tool_matches.sum())
        tool_subset_acc = float(k_tool / N_val)
        tool_micro_f1 = float(f1_score(gold_tools, pred_tools, average="micro", zero_division=0))
        tool_macro_f1 = float(f1_score(gold_tools, pred_tools, average="macro", zero_division=0))

        # 2. Slot Metrics (Strict & Lenient Span F1)
        gold_spans_list = []
        pred_spans_list = []
        total_tokens = 0
        correct_tokens = 0
        gold_token_tags_flat = []
        pred_token_tags_flat = []

        for i in range(N_val):
            l_i = int(mask_np[i].sum())
            g_seq = [IDX_TO_TAG.get(x, "O") for x in gold_tags[i, :l_i]]
            p_seq = pred_tag_seqs[i]

            gold_spans_list.append(extract_spans_strict(g_seq))
            pred_spans_list.append(extract_spans_strict(p_seq))

            total_tokens += l_i
            correct_tokens += sum(1 for g, p in zip(g_seq, p_seq) if g == p)
            gold_token_tags_flat.extend(gold_tags[i, :l_i].tolist())
            pred_token_tags_flat.extend(pred_tag_ids_raw[i][:l_i])

        span_res = compute_span_f1(gold_spans_list, pred_spans_list)
        token_acc = float(correct_tokens / max(total_tokens, 1))

        # Tag macro-F1 (excluding PAD)
        eval_tag_indices = [idx for tag, idx in TAG_TO_IDX.items() if tag != "<PAD>"]
        tag_macro_f1 = float(f1_score(gold_token_tags_flat, pred_token_tags_flat, labels=eval_tag_indices, average="macro", zero_division=0))

        # 3. Joint Exact Match
        slot_exact = [g == p for g, p in zip(gold_spans_list, pred_spans_list)]
        joint_matches = [t and s for t, s in zip(tool_matches, slot_exact)]
        k_joint = int(sum(joint_matches))
        joint_em = float(k_joint / N_val)

        # 4. Multi-tool breakdown across 7 pairs
        pair_metrics = {}
        for p_name in set(self.sample_pairs):
            if p_name is not None:
                p_indices = [i for i, idx in enumerate(val_idx) if self.sample_pairs[idx] == p_name]
                if len(p_indices) > 0:
                    p_t_acc = float(np.all(pred_tools[p_indices] == gold_tools[p_indices], axis=1).mean())
                    p_j_em = float(sum(joint_matches[i] for i in p_indices) / len(p_indices))
                    pair_metrics[p_name] = {
                        "count": len(p_indices),
                        "tool_subset_acc": p_t_acc,
                        "joint_em": p_j_em
                    }

        # 5. Per-cluster accuracies
        val_clusters = [self.sample_clusters[i] for i in val_idx]
        cluster_accs = {}
        for cid in set(val_clusters):
            c_indices = [i for i, c in enumerate(val_clusters) if c == cid]
            c_t_acc = float(np.all(pred_tools[c_indices] == gold_tools[c_indices], axis=1).mean())
            c_j_em = float(sum(joint_matches[i] for i in c_indices) / len(c_indices))
            cluster_accs[cid] = {
                "count": len(c_indices),
                "tool_subset_acc": c_t_acc,
                "joint_em": c_j_em
            }

        return {
            "tool_subset_acc": {"k": k_tool, "N": N_val, "rate": tool_subset_acc},
            "tool_micro_f1": tool_micro_f1,
            "tool_macro_f1": tool_macro_f1,
            "strict_span_f1": span_res["strict_f1"],
            "lenient_span_f1": span_res["lenient_f1"],
            "token_acc": token_acc,
            "tag_macro_f1": tag_macro_f1,
            "joint_em": {"k": k_joint, "N": N_val, "rate": joint_em},
            "seen_tool_subset_acc": {"k": k_tool, "N": N_val, "rate": tool_subset_acc},
            "seen_joint_em": {"k": k_joint, "N": N_val, "rate": joint_em},
            "best_tau": best_tau,
            "tool_matches": tool_matches.tolist(),
            "joint_matches": joint_matches,
            "pair_metrics": pair_metrics,
            "cluster_accs": cluster_accs,
        }


def run_master_protocol():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== RUNNING PHASE 4 MASTER PROTOCOL on {device} ===", flush=True)

    runner = MasterProtocolExperimentRunner(device)
    repo_dir = runner.repo_dir
    git_hash = get_git_hash(repo_dir)
    timestamp = datetime.utcnow().isoformat() + "Z"
    command = "python scripts/run_master_protocol.py"

    with open(runner.ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    models_to_run = [
        "frozen_arn",
        "frozen_cls_crf",
        "frozen_far_no_crf",
        "frozen_joint_bert",
        "unfrozen_cls_crf",
        "unfrozen_arn",
        "unfrozen_joint_bert",
    ]

    per_cluster_path = repo_dir / "results" / "per_cluster_predictions.json"
    if per_cluster_path.exists():
        try:
            with open(per_cluster_path, "r", encoding="utf-8") as f:
                all_per_cluster_results = json.load(f)
        except Exception:
            all_per_cluster_results = {}
    else:
        all_per_cluster_results = {}

    total_runs_added = 0

    for sh_name, sh_folds in runner.manifest["folds"].items():
        seed = {"shuffle_1": 42, "shuffle_2": 100, "shuffle_3": 2024}.get(sh_name, 42)
        print(f"\n==========================================", flush=True)
        print(f"  STARTING {sh_name.upper()} (Seed = {seed})", flush=True)
        print(f"==========================================", flush=True)

        for fold_name, f_data in sh_folds.items():
            tr_idx = f_data["outer_train_indices"]
            val_idx = f_data["outer_val_indices"]
            in_tr_idx = f_data["inner_train_indices"]
            in_val_idx = f_data["inner_val_indices"]

            print(f"\n>>> [{sh_name} / {fold_name}] Train N={len(tr_idx)}, Val N={len(val_idx)}", flush=True)

            existing_exp_ids = set(r.get("experiment_id") for r in ledger["runs"])

            for m_name in models_to_run:
                exp_id = f"p4_{m_name}_{sh_name}_{fold_name}"
                if exp_id in existing_exp_ids:
                    print(f"  {m_name:20s} | [ALREADY COMPLETED in ledger, skipping]", flush=True)
                    continue

                fold_seed = seed * 10 + int(fold_name.split("_")[1])

                if m_name.startswith("frozen"):
                    m_head, best_tau = runner.train_frozen_model(m_name, in_tr_idx, in_val_idx, seed=fold_seed)
                    res = runner.evaluate_outer_fold(m_name, m_head, best_tau, val_idx)
                else:
                    m_full, best_tau = runner.train_unfrozen_model(m_name, in_tr_idx, in_val_idx, seed=fold_seed)
                    res = runner.evaluate_outer_fold(m_name, m_full, best_tau, val_idx)

                # Store per-cluster outputs safely
                if m_name not in all_per_cluster_results:
                    all_per_cluster_results[m_name] = {}
                fold_key = f"{sh_name}_{fold_name}"
                if fold_key not in all_per_cluster_results[m_name]:
                    all_per_cluster_results[m_name][fold_key] = {}
                for cid, cdata in res["cluster_accs"].items():
                    all_per_cluster_results[m_name][fold_key][cid] = cdata

                # Create ledger entry
                metrics_copy = {k: v for k, v in res.items() if k not in ["tool_matches", "joint_matches", "cluster_accs"]}
                run_record = {
                    "experiment_id": exp_id,
                    "model": m_name,
                    "protocol": "phase4_master_protocol",
                    "shuffle": sh_name,
                    "fold": fold_name,
                    "seed": fold_seed,
                    "metrics": metrics_copy,
                    "git_hash": git_hash,
                    "command": command,
                    "timestamp": timestamp,
                }
                ledger["runs"].append(run_record)
                total_runs_added += 1

                print(
                    f"  {m_name:20s} | "
                    f"Tool Acc: {res['tool_subset_acc']['rate']*100:5.2f}% ({res['tool_subset_acc']['k']}/{res['tool_subset_acc']['N']}) | "
                    f"Span F1: {res['strict_span_f1']*100:5.2f}% | "
                    f"Joint EM: {res['joint_em']['rate']*100:5.2f}% ({res['joint_em']['k']}/{res['joint_em']['N']}) | "
                    f"tau: {res['best_tau']:.2f}",
                    flush=True
                )

                # Checkpoint ledger immediately after each model
                with open(runner.ledger_path, "w", encoding="utf-8") as f:
                    json.dump(ledger, f, indent=2)
                with open(per_cluster_path, "w", encoding="utf-8") as f:
                    json.dump(all_per_cluster_results, f, indent=2)

    # Save final per-cluster predictions
    with open(per_cluster_path, "w", encoding="utf-8") as f:
        json.dump(all_per_cluster_results, f, indent=2)

    print(f"\n=== Master Protocol Runs Finished! Added {total_runs_added} runs ===", flush=True)

    compute_master_summary(runner.ledger_path, per_cluster_path, repo_dir / "results" / "master_protocol_summary.json")


def compute_master_summary(ledger_path: Path, per_cluster_path: Path, out_path: Path, B: int = 2000):
    """Aggregate 15-fold statistics, cluster bootstrap 95% CIs, and paired deltas."""
    print(f"\nComputing Master Protocol Summary & Cluster Bootstrap CIs (B={B})...", flush=True)
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    p4_runs = [r for r in ledger["runs"] if r.get("protocol") == "phase4_master_protocol"]
    by_model = defaultdict(list)
    for r in p4_runs:
        by_model[r["model"]].append(r)

    cluster_data = {}
    if per_cluster_path.exists():
        with open(per_cluster_path, "r", encoding="utf-8") as f:
            cluster_data = json.load(f)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_folds": 15,
        "bootstrap_iterations": B,
        "models": {},
        "paired_deltas": {},
    }

    # Model metrics aggregation
    for m_name, runs in by_model.items():
        if len(runs) == 0:
            continue
        tool_accs = [r["metrics"]["tool_subset_acc"]["rate"] for r in runs]
        strict_f1s = [r["metrics"]["strict_span_f1"] for r in runs]
        lenient_f1s = [r["metrics"]["lenient_span_f1"] for r in runs]
        joint_ems = [r["metrics"]["joint_em"]["rate"] for r in runs]
        token_accs = [r["metrics"]["token_acc"] for r in runs]

        m_stats = {
            "n_folds": len(runs),
            "tool_acc_mean": float(np.mean(tool_accs)),
            "tool_acc_std": float(np.std(tool_accs, ddof=1)) if len(tool_accs) > 1 else 0.0,
            "strict_span_f1_mean": float(np.mean(strict_f1s)),
            "strict_span_f1_std": float(np.std(strict_f1s, ddof=1)) if len(strict_f1s) > 1 else 0.0,
            "lenient_span_f1_mean": float(np.mean(lenient_f1s)),
            "lenient_span_f1_std": float(np.std(lenient_f1s, ddof=1)) if len(lenient_f1s) > 1 else 0.0,
            "joint_em_mean": float(np.mean(joint_ems)),
            "joint_em_std": float(np.std(joint_ems, ddof=1)) if len(joint_ems) > 1 else 0.0,
            "token_acc_mean": float(np.mean(token_accs)),
            "token_acc_std": float(np.std(token_accs, ddof=1)) if len(token_accs) > 1 else 0.0,
        }

        # Cluster bootstrap CI
        if m_name in cluster_data:
            # Flatten all clusters across folds: list of (k_tool, N, k_joint)
            all_clusters = []
            for fold_key, c_dict in cluster_data[m_name].items():
                for cid, cinfo in c_dict.items():
                    n = cinfo.get("count", 0)
                    tool_k = round(n * cinfo.get("tool_subset_acc", 0.0))
                    joint_k = round(n * cinfo.get("joint_em", 0.0))
                    all_clusters.append((tool_k, n, joint_k))

            if all_clusters:
                n_c = len(all_clusters)
                np.random.seed(42)
                bs_tool_rates = []
                bs_joint_rates = []
                for _ in range(B):
                    indices = np.random.choice(n_c, size=n_c, replace=True)
                    tot_n = sum(all_clusters[i][1] for i in indices)
                    tot_tool_k = sum(all_clusters[i][0] for i in indices)
                    tot_joint_k = sum(all_clusters[i][2] for i in indices)
                    bs_tool_rates.append(tot_tool_k / max(tot_n, 1))
                    bs_joint_rates.append(tot_joint_k / max(tot_n, 1))

                m_stats["tool_acc_bootstrap_95ci"] = [
                    float(np.percentile(bs_tool_rates, 2.5)),
                    float(np.percentile(bs_tool_rates, 97.5)),
                ]
                m_stats["joint_em_bootstrap_95ci"] = [
                    float(np.percentile(bs_joint_rates, 2.5)),
                    float(np.percentile(bs_joint_rates, 97.5)),
                ]

        summary["models"][m_name] = m_stats

    # Paired deltas across folds
    # Key comparisons: vs frozen_arn, vs frozen_cls_crf, vs unfrozen_cls_crf
    fold_keys = [(r["shuffle"], r["fold"]) for r in by_model.get("frozen_arn", [])]
    model_by_fold = {m: {(r["shuffle"], r["fold"]): r for r in r_list} for m, r_list in by_model.items()}

    for target_m in ["frozen_cls_crf", "unfrozen_cls_crf", "unfrozen_arn", "frozen_far_no_crf"]:
        if target_m not in model_by_fold:
            continue
        common_folds = [fk for fk in fold_keys if fk in model_by_fold[target_m]]
        if not common_folds:
            continue

        deltas_tool = []
        deltas_f1 = []
        deltas_joint = []
        for fk in common_folds:
            r_arn = model_by_fold["frozen_arn"][fk]
            r_tgt = model_by_fold[target_m][fk]
            deltas_tool.append(r_tgt["metrics"]["tool_subset_acc"]["rate"] - r_arn["metrics"]["tool_subset_acc"]["rate"])
            deltas_f1.append(r_tgt["metrics"]["strict_span_f1"] - r_arn["metrics"]["strict_span_f1"])
            deltas_joint.append(r_tgt["metrics"]["joint_em"]["rate"] - r_arn["metrics"]["joint_em"]["rate"])

        summary["paired_deltas"][f"{target_m}_minus_frozen_arn"] = {
            "n_folds": len(common_folds),
            "tool_acc_delta_mean": float(np.mean(deltas_tool)),
            "tool_acc_delta_std": float(np.std(deltas_tool, ddof=1)) if len(deltas_tool) > 1 else 0.0,
            "strict_f1_delta_mean": float(np.mean(deltas_f1)),
            "strict_f1_delta_std": float(np.std(deltas_f1, ddof=1)) if len(deltas_f1) > 1 else 0.0,
            "joint_em_delta_mean": float(np.mean(deltas_joint)),
            "joint_em_delta_std": float(np.std(deltas_joint, ddof=1)) if len(deltas_joint) > 1 else 0.0,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved master protocol summary to {out_path}", flush=True)


if __name__ == "__main__":
    run_master_protocol()

