"""
ARN-v2 model: trunk + router (FAR-v2 or [CLS]) + tool-qualified emissions + CRF.

FAR-v2 keeps the original Factored Attention Router unchanged (learned tool
queries, single-head cross-attention, shared scalar projection) and adds two
terms the review identified as missing:
  * a per-tool bias b_k (tool base rates; the original had no bias anywhere)
  * a [CLS] residual (W_c h_cls)_k so routing never does worse than a CLS head
    logit_k = w^T ctx_k + b_k + (W_c h_cls)_k
FAR's attention weights are returned and used by the decoder to anchor calls
that have no argument spans (call ordering), see arn.v2.decode.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from ..crf import CRF
from ..far import FactoredAttentionRouter
from .schema import Registry


class FARv2(nn.Module):
    def __init__(self, num_tools: int, hidden: int, dropout: float = 0.1, cls_residual: bool = True) -> None:
        super().__init__()
        self.far = FactoredAttentionRouter(num_tools, hidden, dropout)
        self.bias = nn.Parameter(torch.zeros(num_tools))
        self.cls = nn.Linear(hidden, num_tools, bias=False) if cls_residual else None

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, weights = self.far(h, mask)
        logits = logits + self.bias
        if self.cls is not None:
            logits = logits + self.cls(h[:, 0])
        return logits, weights


class ARNv2(nn.Module):
    def __init__(self, trunk: str, reg: Registry, router: str = "far", dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(trunk, attn_implementation="eager")
        hidden = self.encoder.config.hidden_size
        self.router_kind = router
        k = len(reg.tools)
        if router == "far":
            self.router = FARv2(k, hidden, dropout)
        elif router == "far_plain":  # original FAR, for the ablation
            self.router = FARv2(k, hidden, dropout, cls_residual=False)
        elif router == "cls":
            self.router = nn.Linear(hidden, k)
        else:
            raise ValueError(router)
        self.drop = nn.Dropout(dropout)
        self.emit = nn.Linear(hidden, len(reg.tags))
        self.crf = CRF(len(reg.tags), hard_bio_mask=True, tag_to_idx=reg.tag_to_idx)
        with torch.no_grad():  # small CRF init instead of N(0,1)
            for p in (self.crf.transitions, self.crf.start_transitions, self.crf.end_transitions):
                p.uniform_(-0.1, 0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.router_kind == "cls":
            tool_logits, attn = self.router(self.drop(h[:, 0])), None
        else:
            tool_logits, attn = self.router(h, attention_mask)
        emissions = self.emit(self.drop(h))
        return tool_logits, emissions, attn

    def loss(self, tool_logits, emissions, attention_mask, tool_labels, tags, lambda_slot: float = 1.0):
        tool_loss = nn.functional.binary_cross_entropy_with_logits(tool_logits, tool_labels)
        nll = self.crf.sentence_nll(emissions, tags, attention_mask)
        slot_loss = (nll / attention_mask.sum(1).clamp(min=1)).mean()
        return tool_loss + lambda_slot * slot_loss, tool_loss, slot_loss
