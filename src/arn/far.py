"""
Factored Attention Router (FAR) for ARN.

Reformulates multi-label tool orchestration as cross-attention classification:
Learnable tool query vectors attend over the contextual encoder token representations,
producing tool-conditioned context states and well-calibrated independent tool logits.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class FactoredAttentionRouter(nn.Module):
    """
    Cross-attention router projecting learnable tool keys against query representations.

    Parameters
    ----------
    num_tools : int
        Size of the closed tool registry (K).
    hidden_dim : int
        Dimensionality of trunk representations (e.g., 128 for BERT-tiny).
    dropout : float
        Dropout probability applied to cross-attention weights.
    """

    def __init__(self, num_tools: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_tools = num_tools
        self.hidden_dim = hidden_dim

        # Static tool embedding registry: K x d_h
        self.tool_embeddings = nn.Embedding(num_embeddings=num_tools, embedding_dim=hidden_dim)

        # Cross-attention linear projections
        self.W_q = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.W_k = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.W_v = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)

        # Output projection mapping tool context vectors to scalar logits: d_h -> 1
        self.output_proj = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Factored Attention Routing.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Contextual embeddings from encoder trunk of shape (B, N, d_h).
        attention_mask : Optional[torch.Tensor]
            Binary mask of active tokens of shape (B, N) where 1=active, 0=padded.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - logits: Raw classification logits of shape (B, K).
            - weights: Cross-attention weights of shape (B, K, N).
        """
        batch_size, seq_len, _ = encoder_output.shape
        device = encoder_output.device

        # 1. Retrieve and project tool queries: (K, d_h)
        tool_indices = torch.arange(0, self.num_tools, device=device)
        tool_keys = self.tool_embeddings(tool_indices)  # (K, d_h)
        Q = self.W_q(tool_keys).unsqueeze(0)           # (1, K, d_h)

        # 2. Project text keys and values: (B, N, d_h)
        K_text = self.W_k(encoder_output)
        V_text = self.W_v(encoder_output)

        # 3. Scaled dot-product attention scores: (B, K, N)
        scores = torch.matmul(Q, K_text.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

        # 4. Mask padded tokens
        if attention_mask is not None:
            mask_expanded = (attention_mask.unsqueeze(1) == 0)  # (B, 1, N)
            scores = scores.masked_fill(mask_expanded, -1e9)

        weights = torch.softmax(scores, dim=-1)  # (B, K, N)
        weights_dropped = self.dropout(weights)

        # 5. Tool context aggregation: (B, K, d_h)
        context = torch.matmul(weights_dropped, V_text)

        # 6. Project to logits: (B, K)
        logits = self.output_proj(context).squeeze(-1)

        return logits, weights
