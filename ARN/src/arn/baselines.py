"""
Baseline and Ablation Model Architectures for ARN.
Implements:
1. JointBERT (Chen et al. 2019): [CLS] tool classification + independent softmax token slot tagging.
2. CLS+CRF: [CLS] tool classification + Linear-Chain CRF slot tagging (ablating FAR).
3. FAR without CRF: Factored Attention Routing + independent softmax token slot tagging (ablating CRF).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel

from .config import ARNConfig
from .far import FactoredAttentionRouter
from .crf import CRF
from .tokenizer import TAG_TO_IDX


class JointBERT(nn.Module):
    """
    Standard JointBERT baseline:
    - [CLS] representation for multi-label tool routing via Linear projection.
    - Token representations for argument slot tagging via per-token Linear + Softmax.
    """

    def __init__(self, config: Optional[ARNConfig] = None, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config if config is not None else ARNConfig()

        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                attn_implementation="eager"
            )
            if self.config.freeze_trunk:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        else:
            self.encoder = None

        self.tool_classifier = nn.Linear(self.config.hidden_dim, self.config.num_tools)
        self.slot_classifier = nn.Linear(self.config.hidden_dim, self.config.num_tags)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_tag_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # (B, N, d)
        cls_repr = h[:, 0, :]          # (B, d)

        tool_logits = self.tool_classifier(cls_repr)
        slot_logits = self.slot_classifier(h)

        if tags is not None and tool_labels is not None:
            tool_loss = self.bce_loss(tool_logits, tool_labels)
            slot_loss = self.ce_loss(slot_logits.view(-1, self.config.num_tags), tags.view(-1))
            total_loss = tool_loss + self.config.lambda_slot * slot_loss
            return total_loss, tool_loss, slot_loss, tool_logits
        else:
            tool_probs = torch.sigmoid(tool_logits)
            tool_preds = (tool_probs >= self.config.activation_threshold)
            tag_preds_tensor = torch.argmax(slot_logits, dim=-1)  # (B, N)
            tag_preds = []
            for b in range(input_ids.size(0)):
                length = int(attention_mask[b].sum().item())
                tag_preds.append(tag_preds_tensor[b, :length].tolist())
            return tool_probs, tool_preds, tag_preds


class CLSCRFModel(nn.Module):
    """
    Ablation 1: CLS + CRF (removes FAR cross-attention query embeddings).
    - Multi-label tool classification via [CLS] projection.
    - Slot tagging via Linear emission + Linear-Chain CRF with hard BIO masking.
    """

    def __init__(self, config: Optional[ARNConfig] = None, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config if config is not None else ARNConfig()

        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                attn_implementation="eager"
            )
            if self.config.freeze_trunk:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        else:
            self.encoder = None

        self.tool_classifier = nn.Linear(self.config.hidden_dim, self.config.num_tools)
        self.slot_tagger = nn.Linear(self.config.hidden_dim, self.config.num_tags)
        self.crf = CRF(
            num_tags=self.config.num_tags,
            hard_bio_mask=self.config.hard_bio_mask,
            tag_to_idx=TAG_TO_IDX
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        cls_repr = h[:, 0, :]

        tool_logits = self.tool_classifier(cls_repr)
        emissions = self.slot_tagger(h)

        if tags is not None and tool_labels is not None:
            tool_loss = self.bce_loss(tool_logits, tool_labels)
            active_tokens = attention_mask.sum(dim=1).clamp(min=1)
            crf_loss = -self.crf(emissions, tags, attention_mask)
            slot_loss = (crf_loss / active_tokens).mean()
            total_loss = tool_loss + self.config.lambda_slot * slot_loss
            return total_loss, tool_loss, slot_loss, tool_logits
        else:
            tool_probs = torch.sigmoid(tool_logits)
            tool_preds = (tool_probs >= self.config.activation_threshold)
            tag_preds = self.crf.viterbi_decode(emissions, attention_mask)
            return tool_probs, tool_preds, tag_preds


class FARNoCRFModel(nn.Module):
    """
    Ablation 2: FAR without CRF (removes CRF transition constraints).
    - Multi-label tool classification via Factored Attention Routing.
    - Slot tagging via independent Linear + Softmax (per-token Cross-Entropy).
    """

    def __init__(self, config: Optional[ARNConfig] = None, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config if config is not None else ARNConfig()

        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                attn_implementation="eager"
            )
            if self.config.freeze_trunk:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        else:
            self.encoder = None

        self.far = FactoredAttentionRouter(
            num_tools=self.config.num_tools,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )
        self.slot_classifier = nn.Linear(self.config.hidden_dim, self.config.num_tags)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_tag_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state

        tool_logits, routing_weights = self.far(h, attention_mask)
        slot_logits = self.slot_classifier(h)

        if tags is not None and tool_labels is not None:
            tool_loss = self.bce_loss(tool_logits, tool_labels)
            slot_loss = self.ce_loss(slot_logits.view(-1, self.config.num_tags), tags.view(-1))
            total_loss = tool_loss + self.config.lambda_slot * slot_loss
            return total_loss, tool_loss, slot_loss, tool_logits, routing_weights
        else:
            tool_probs = torch.sigmoid(tool_logits)
            tool_preds = (tool_probs >= self.config.activation_threshold)
            tag_preds_tensor = torch.argmax(slot_logits, dim=-1)
            tag_preds = []
            for b in range(input_ids.size(0)):
                length = int(attention_mask[b].sum().item())
                tag_preds.append(tag_preds_tensor[b, :length].tolist())
            return tool_probs, tool_preds, tag_preds, routing_weights
