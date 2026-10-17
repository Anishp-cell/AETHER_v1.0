"""
Linear-Chain Conditional Random Field (CRF) with Hard BIO Transition Masking.

Provides:
- Exact sequence scoring with learnable start, end, and tag-to-tag transitions
- Forward algorithm for log-partition function Z(H)
- Viterbi decoding with hard-masked invalid BIO transitions
- Sentence-level negative log-likelihood computation
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .tokenizer import TAG_TO_IDX


def build_bio_transition_mask(
    tag_to_idx: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct transition validity masks enforcing strict BIO sequence grammar.

    Rules enforced:
    1. O cannot transition to I-X (must start with B-X).
    2. B-X cannot transition to I-Y where X != Y.
    3. I-X cannot transition to I-Y where X != Y.
    4. Start of sentence cannot be I-X or <PAD>.
    5. <PAD> cannot transition to active entity tags.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - trans_mask: (num_tags, num_tags) bool tensor (True = valid)
        - start_mask: (num_tags,) bool tensor (True = valid start)
        - end_mask:   (num_tags,) bool tensor (True = valid end)
    """
    num_tags = len(tag_to_idx)
    trans_mask = torch.ones((num_tags, num_tags), dtype=torch.bool)
    start_mask = torch.ones((num_tags,), dtype=torch.bool)
    end_mask = torch.ones((num_tags,), dtype=torch.bool)

    pad_idx = tag_to_idx.get("<PAD>", None)
    o_idx = tag_to_idx.get("O", 0)

    # 1. Start constraints: sentence cannot start with I-* or <PAD>
    for tag, idx in tag_to_idx.items():
        if tag.startswith("I-") or tag == "<PAD>":
            start_mask[idx] = False

    # 2. Tag-to-tag transition constraints
    for from_tag, from_idx in tag_to_idx.items():
        for to_tag, to_idx in tag_to_idx.items():
            # Rule 1: O -> I-X is invalid
            if from_tag == "O" and to_tag.startswith("I-"):
                trans_mask[from_idx, to_idx] = False

            # Rule 2: B-X -> I-Y (X != Y) is invalid
            elif from_tag.startswith("B-") and to_tag.startswith("I-"):
                from_slot = from_tag.split("-")[1]
                to_slot = to_tag.split("-")[1]
                if from_slot != to_slot:
                    trans_mask[from_idx, to_idx] = False

            # Rule 3: I-X -> I-Y (X != Y) is invalid
            elif from_tag.startswith("I-") and to_tag.startswith("I-"):
                from_slot = from_tag.split("-")[1]
                to_slot = to_tag.split("-")[1]
                if from_slot != to_slot:
                    trans_mask[from_idx, to_idx] = False

            # Rule 4: <PAD> cannot transition to any active tag
            elif from_tag == "<PAD>" and to_tag != "<PAD>":
                trans_mask[from_idx, to_idx] = False

    return trans_mask, start_mask, end_mask


class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field layer.

    Parameters
    ----------
    num_tags : int
        Number of BIO slot tags (including padding).
    hard_bio_mask : bool
        If True, enforces strict BIO transition constraints during Viterbi decoding
        and forward partition computation by masking impossible transitions.
    """

    def __init__(
        self,
        num_tags: int,
        hard_bio_mask: bool = True,
        tag_to_idx: Optional[Dict[str, int]] = None
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.hard_bio_mask = hard_bio_mask

        # Learnable transition parameters
        self.transitions = nn.Parameter(torch.randn((num_tags, num_tags)))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        # Register hard transition masks if enabled
        if hard_bio_mask:
            mapping = tag_to_idx if tag_to_idx is not None else TAG_TO_IDX
            t_mask, s_mask, e_mask = build_bio_transition_mask(mapping)
            self.register_buffer("valid_trans_mask", t_mask)
            self.register_buffer("valid_start_mask", s_mask)
            self.register_buffer("valid_end_mask", e_mask)
        else:
            self.valid_trans_mask = None
            self.valid_start_mask = None
            self.valid_end_mask = None

    def _get_effective_transitions(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply hard -10000.0 mask to invalid transitions if hard_bio_mask is active."""
        if not self.hard_bio_mask or self.valid_trans_mask is None:
            return self.transitions, self.start_transitions, self.end_transitions

        large_neg = -10000.0
        masked_trans = torch.where(
            self.valid_trans_mask, self.transitions, torch.tensor(large_neg, device=self.transitions.device)
        )
        masked_start = torch.where(
            self.valid_start_mask, self.start_transitions, torch.tensor(large_neg, device=self.start_transitions.device)
        )
        masked_end = torch.where(
            self.valid_end_mask, self.end_transitions, torch.tensor(large_neg, device=self.end_transitions.device)
        )
        return masked_trans, masked_start, masked_end

    def forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log-partition function Z(H) using the forward algorithm in O(N M^2) time.

        Parameters
        ----------
        emissions : torch.Tensor
            Emission scores of shape (B, N, M).
        mask : torch.Tensor
            Binary attention mask of shape (B, N).

        Returns
        -------
        torch.Tensor
            Log-sum-exp score of all paths for each sequence: shape (B,).
        """
        batch_size, seq_len, _ = emissions.shape
        trans, start_t, end_t = self._get_effective_transitions()

        # Initialize forward variable at t=0
        alpha = start_t.unsqueeze(0) + emissions[:, 0, :]  # (B, M)

        for t in range(1, seq_len):
            active = mask[:, t].bool().unsqueeze(1)         # (B, 1)
            alpha_expanded = alpha.unsqueeze(2)             # (B, M, 1)
            trans_expanded = trans.unsqueeze(0)             # (1, M, M)
            emit_expanded = emissions[:, t, :].unsqueeze(1) # (B, 1, M)

            scores = alpha_expanded + trans_expanded + emit_expanded  # (B, M, M)
            alpha_next = torch.logsumexp(scores, dim=1)               # (B, M)
            alpha = torch.where(active, alpha_next, alpha)

        terminal_scores = alpha + end_t.unsqueeze(0)
        return torch.logsumexp(terminal_scores, dim=1)

    def score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the unnormalized score S(H, y) of the ground-truth sequence y.

        Parameters
        ----------
        emissions : torch.Tensor
            Emission scores of shape (B, N, M).
        tags : torch.Tensor
            Gold tag indices of shape (B, N).
        mask : torch.Tensor
            Binary attention mask of shape (B, N).

        Returns
        -------
        torch.Tensor
            Gold sentence scores of shape (B,).
        """
        batch_size, seq_len, _ = emissions.shape
        trans, start_t, end_t = self._get_effective_transitions()

        # Start score
        score = start_t[tags[:, 0]] + emissions[torch.arange(batch_size), 0, tags[:, 0]] * mask[:, 0]

        for t in range(1, seq_len):
            active = mask[:, t].bool()
            from_tag = tags[:, t - 1]
            to_tag = tags[:, t]

            trans_step = trans[from_tag, to_tag]
            emit_step = emissions[torch.arange(batch_size), t, to_tag]
            step_score = trans_step + emit_step

            score = score + torch.where(active, step_score, torch.zeros_like(score))

        # Terminal end score at last active token
        last_indices = (mask.sum(dim=1).long() - 1).clamp(min=0)
        last_tags = tags[torch.arange(batch_size), last_indices]
        score = score + end_t[last_tags]

        return score

    def sentence_nll(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood per sequence: NLL_b = Z_b - S_b.

        Returns
        -------
        torch.Tensor
            Vector of per-sequence NLL scores of shape (B,).
        """
        forward_score = self.forward_algorithm(emissions, mask)
        gold_score = self.score_sentence(emissions, tags, mask)
        return forward_score - gold_score

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass computes sentence negative log-likelihood."""
        return self.sentence_nll(emissions, tags, mask)

    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Perform Viterbi decoding to identify the optimal tag sequence y* in O(N M^2) time.
        Strictly respects hard BIO transition constraints when hard_bio_mask is enabled.

        Parameters
        ----------
        emissions : torch.Tensor
            Emission logits of shape (B, N, M).
        mask : torch.Tensor
            Binary attention mask of shape (B, N).

        Returns
        -------
        List[List[int]]
            Decoded tag sequences of length seq_len for each sample in the batch.
        """
        batch_size, seq_len, _ = emissions.shape
        trans, start_t, end_t = self._get_effective_transitions()

        viterbi_vars = start_t.unsqueeze(0) + emissions[:, 0, :]  # (B, M)
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            active = mask[:, t].bool().unsqueeze(1)  # (B, 1)
            # Broadcasting: (B, M, 1) + (1, M, M) -> (B, M, M) where dim 1 is from, dim 2 is to
            next_tag_var = viterbi_vars.unsqueeze(2) + trans.unsqueeze(0)
            max_vars, bptrs = torch.max(next_tag_var, dim=1)  # (B, M)
            max_vars = max_vars + emissions[:, t, :]
            viterbi_vars = torch.where(active, max_vars, viterbi_vars)
            backpointers.append(bptrs)

        terminal_vars = viterbi_vars + end_t.unsqueeze(0)
        best_tag_ids = torch.argmax(terminal_vars, dim=1)  # (B,)

        pad_val = TAG_TO_IDX.get("<PAD>", 25)
        best_paths: List[List[int]] = []

        for b in range(batch_size):
            active_len = int(mask[b].sum().item())
            if active_len <= 0:
                best_paths.append([pad_val] * seq_len)
                continue

            curr_tag = best_tag_ids[b].item()
            path = [curr_tag]

            # Backtrack through backpointers
            for t in range(active_len - 2, -1, -1):
                curr_tag = backpointers[t][b, curr_tag].item()
                path.append(curr_tag)

            path.reverse()
            # Pad remaining sequence to seq_len
            path.extend([pad_val] * (seq_len - len(path)))
            best_paths.append(path)

        return best_paths
