"""
Aether Routing Network (ARN) Core Model Architecture.

Pairs a frozen 4.38M bidirectional Transformer trunk (BERT-tiny) with:
1. Factored Attention Routing (FAR) cross-attention head for multi-label tool selection.
2. Linear emission projection + Linear-Chain CRF for argument slot extraction.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel

from .config import ARNConfig
from .far import FactoredAttentionRouter
from .crf import CRF
from .loss import ARNLoss
from .tokenizer import TAG_TO_IDX, TOOL_TO_IDX


class AetherRoutingNetwork(nn.Module):
    """
    Ultra-compact non-generative joint model for real-time edge tool orchestration.

    Parameters
    ----------
    config : ARNConfig
        Configuration object containing architectural dimensions and hyperparams.
    load_pretrained : bool
        Whether to load pre-trained BERT-tiny weights for the encoder trunk.
    """

    def __init__(self, config: Optional[ARNConfig] = None, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config if config is not None else ARNConfig()

        # 1. Bidirectional Encoder Trunk (BERT-tiny)
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                attn_implementation="eager"
            )
            if self.config.freeze_trunk:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            self.encoder = None

        # 2. Factored Attention Routing (FAR) Cross-Attention Head
        self.far = FactoredAttentionRouter(
            num_tools=self.config.num_tools,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )

        # 3. Token-Level Slot Tagger Emission Projection
        self.slot_tagger = nn.Linear(self.config.hidden_dim, self.config.num_tags)

        # 4. Linear-Chain CRF Tagging Head
        self.crf = CRF(
            num_tags=self.config.num_tags,
            hard_bio_mask=self.config.hard_bio_mask,
            tag_to_idx=TAG_TO_IDX
        )

        # 5. Multi-Task Objective
        self.loss_fn = ARNLoss(lambda_slot=self.config.lambda_slot)

    def count_parameters(self) -> Dict[str, int]:
        """Compute exact parameter breakdown across modules."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        far = sum(p.numel() for p in self.far.parameters())
        slot = sum(p.numel() for p in self.slot_tagger.parameters())
        crf = sum(p.numel() for p in self.crf.parameters())
        trunk = sum(p.numel() for p in self.encoder.parameters()) if self.encoder else 0

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "far_parameters": far,
            "slot_emission_parameters": slot,
            "crf_parameters": crf,
            "encoder_trunk_parameters": trunk,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor]
    ]:
        """
        Forward pass of ARN.

        Parameters
        ----------
        input_ids : torch.Tensor
            Subtoken IDs of shape (B, N).
        attention_mask : torch.Tensor
            Binary attention mask of shape (B, N).
        tags : Optional[torch.Tensor]
            Target slot tags of shape (B, N). If provided, computes loss.
        tool_labels : Optional[torch.Tensor]
            Target binary tool labels of shape (B, K). If provided, computes loss.

        Returns
        -------
        If training (tags & tool_labels provided):
            Tuple[total_loss, tool_loss, slot_loss, tool_logits, routing_weights]
        If inference (eval):
            Tuple[tool_probs, tool_preds, tag_preds, routing_weights]
            - tool_probs: Sigmoid activation probabilities of shape (B, K).
            - tool_preds: Boolean activation tensor of shape (B, K).
            - tag_preds: List of Viterbi decoded tag IDs per sample.
            - routing_weights: Cross-attention weights of shape (B, K, N).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # (B, N, d_h)

        tool_logits, routing_weights = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)

        if tags is not None and tool_labels is not None:
            total_loss, tool_loss, slot_loss = self.loss_fn(
                tool_logits=tool_logits,
                tool_labels=tool_labels,
                crf=self.crf,
                emissions=emissions,
                tags=tags,
                attention_mask=attention_mask
            )
            return total_loss, tool_loss, slot_loss, tool_logits, routing_weights
        else:
            tool_probs = torch.sigmoid(tool_logits)
            tool_preds = (tool_probs >= self.config.activation_threshold)
            tag_preds = self.crf.viterbi_decode(emissions, attention_mask)
            return tool_probs, tool_preds, tag_preds, routing_weights
