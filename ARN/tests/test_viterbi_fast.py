"""
Tests for Vectorized NumPy Viterbi Decoder (Phase 8).

Verifies that viterbi_decode_numpy is 100% mathematically equivalent
to PyTorch CRF.viterbi_decode across 10,000 test sequences with various lengths,
emissions, and transition weights.
"""

import numpy as np
import pytest
import torch

from arn.crf import CRF
from arn.viterbi_fast import viterbi_decode_numpy


def test_viterbi_numpy_exact_equivalence_10000_sequences():
    """Verify exact equivalence across 10,000 sequences between NumPy and PyTorch implementations."""
    torch.manual_seed(42)
    np.random.seed(42)

    num_tags = 37
    crf = CRF(num_tags=num_tags, hard_bio_mask=False)

    trans_torch, start_torch, end_torch = crf._get_effective_transitions()
    trans_np = trans_torch.detach().cpu().numpy()
    start_np = start_torch.detach().cpu().numpy()
    end_np = end_torch.detach().cpu().numpy()

    # Generate 10 batches of 1,000 sequences = 10,000 sequences
    batch_size = 1000
    seq_len = 32
    num_batches = 10

    total_checked = 0
    total_matched = 0

    for b_idx in range(num_batches):
        emissions_torch = torch.randn(batch_size, seq_len, num_tags)
        # Random sequence lengths between 3 and seq_len
        lengths = torch.randint(3, seq_len + 1, (batch_size,))
        mask_torch = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i, l in enumerate(lengths):
            mask_torch[i, :l] = 1

        # PyTorch decoding
        paths_torch = crf.viterbi_decode(emissions_torch, mask_torch)

        # NumPy decoding
        emissions_np = emissions_torch.numpy()
        mask_np = mask_torch.numpy()
        paths_np = viterbi_decode_numpy(emissions_np, mask_np, trans_np, start_np, end_np)

        for i in range(batch_size):
            l = int(lengths[i].item())
            p_t = paths_torch[i][:l]
            p_n = paths_np[i][:l]
            assert p_t == p_n, f"Mismatch at sequence {total_checked + i}: PyTorch={p_t} vs NumPy={p_n}"
            total_matched += 1

        total_checked += batch_size

    assert total_matched == 10000, f"Expected 10,000 exact matches, got {total_matched}"


def test_viterbi_numpy_hard_bio_mask_equivalence():
    """Verify exact equivalence when hard BIO transition masking is active."""
    torch.manual_seed(123)
    np.random.seed(123)

    from arn.tokenizer import TAG_TO_IDX
    num_tags = len(TAG_TO_IDX)
    crf = CRF(num_tags=num_tags, hard_bio_mask=True)

    trans_torch, start_torch, end_torch = crf._get_effective_transitions()
    trans_np = trans_torch.detach().cpu().numpy()
    start_np = start_torch.detach().cpu().numpy()
    end_np = end_torch.detach().cpu().numpy()

    batch_size = 500
    seq_len = 24

    emissions_torch = torch.randn(batch_size, seq_len, num_tags)
    lengths = torch.randint(2, seq_len + 1, (batch_size,))
    mask_torch = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i, l in enumerate(lengths):
        mask_torch[i, :l] = 1

    paths_torch = crf.viterbi_decode(emissions_torch, mask_torch)
    paths_np = viterbi_decode_numpy(
        emissions_torch.numpy(),
        mask_torch.numpy(),
        trans_np,
        start_np,
        end_np
    )

    for i in range(batch_size):
        l = int(lengths[i].item())
        assert paths_torch[i][:l] == paths_np[i][:l]
