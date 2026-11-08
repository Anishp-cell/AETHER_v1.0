"""
Multi-Task Loss Module for ARN.

Rigorously implements:
1. Tool Routing Loss: Multi-label Binary Cross-Entropy with Logits.
2. Slot Extraction Loss: Sequence-normalized Negative Log-Likelihood where each
   sentence's NLL is divided by its own active token count N_{active, b}, and then
   averaged across the batch.
3. Total Objective: L_{total} = L_{tool} + lambda_{slot} * L_{slot}.
"""

from typing import Tuple
import torch
import torch.nn as nn

from .crf import CRF


class ARNLoss(nn.Module):
    """
    Balanced Multi-Task Loss for ARN.

    Parameters
    ----------
    lambda_slot : float
        Scalar balance weight for the slot extraction loss.
    """

    def __init__(self, lambda_slot: float = 1.0) -> None:
        super().__init__()
        self.lambda_slot = lambda_slot
        self.tool_bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        tool_logits: torch.Tensor,
        tool_labels: torch.Tensor,
        crf: CRF,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute multi-task loss.

        Parameters
        ----------
        tool_logits : torch.Tensor
            Raw logits from FAR router of shape (B, K).
        tool_labels : torch.Tensor
            Binary target labels of shape (B, K).
        crf : CRF
            Linear-chain CRF instance.
        emissions : torch.Tensor
            Token slot emission logits of shape (B, N, M).
        tags : torch.Tensor
            Ground truth slot tag indices of shape (B, N).
        attention_mask : torch.Tensor
            Binary attention mask of shape (B, N).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - total_loss: L_tool + lambda_slot * L_slot
            - tool_loss: Multi-label BCE loss
            - slot_loss: Per-token normalized CRF NLL
        """
        # 1. Multi-label Tool Routing Loss (mean across batch and tools)
        tool_loss = self.tool_bce(tool_logits, tool_labels.float())

        # 2. Per-Sentence NLL: (B,)
        sentence_nlls = crf.sentence_nll(emissions, tags, attention_mask)

        # 3. Active tokens per sentence: (B,)
        active_counts = attention_mask.sum(dim=1).float().clamp(min=1.0)

        # 4. Per-token normalized sentence loss, averaged across batch
        per_token_nll = sentence_nlls / active_counts
        slot_loss = per_token_nll.mean()

        # 5. Combined objective
        total_loss = tool_loss + self.lambda_slot * slot_loss

        return total_loss, tool_loss, slot_loss
