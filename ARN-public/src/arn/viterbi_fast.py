"""
Vectorized NumPy Viterbi Decoder for Linear-Chain CRF.

Provides zero-dependency, low-latency, fully vectorized NumPy Viterbi decoding
for on-device production deployment.
Mathematically identical to PyTorch CRF Viterbi decoding.
"""

from typing import List, Tuple, Optional
import numpy as np


def viterbi_decode_numpy(
    emissions: np.ndarray,
    mask: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    pad_val: int = 25
) -> List[List[int]]:
    """
    Vectorized NumPy Viterbi decoding for batch of sequences.

    Parameters
    ----------
    emissions : np.ndarray
        Emission logits of shape (B, N, M), dtype float32.
    mask : np.ndarray
        Attention mask of shape (B, N), dtype int or float or bool.
    transitions : np.ndarray
        Transition matrix of shape (M, M), where transitions[i, j] is score from tag i to tag j.
    start_transitions : np.ndarray
        Start transition vector of shape (M,).
    end_transitions : np.ndarray
        End transition vector of shape (M,).
    pad_val : int
        Padding token index for inactive sequence tail.

    Returns
    -------
    List[List[int]]
        Decoded optimal tag sequence IDs for each sample in batch.
    """
    batch_size, seq_len, num_tags = emissions.shape
    mask_bool = mask.astype(bool)

    # Initialize viterbi variables at t=0
    viterbi_vars = start_transitions[None, :] + emissions[:, 0, :]  # (B, M)
    backpointers = []

    # transitions shape: (M, M), from tag i (dim 0) to tag j (dim 1)
    # Broadcast with viterbi_vars: (B, M, 1) + (1, M, M) -> (B, M, M)
    trans_broadcast = transitions[None, :, :]  # (1, M, M)

    for t in range(1, seq_len):
        active = mask_bool[:, t, None]  # (B, 1)

        # Score of moving from all previous tags to each current tag
        next_tag_var = viterbi_vars[:, :, None] + trans_broadcast  # (B, M, M)
        bptrs = np.argmax(next_tag_var, axis=1)                    # (B, M)
        max_vars = np.max(next_tag_var, axis=1) + emissions[:, t, :]  # (B, M)

        viterbi_vars = np.where(active, max_vars, viterbi_vars)
        backpointers.append(bptrs)

    terminal_vars = viterbi_vars + end_transitions[None, :]  # (B, M)
    best_tag_ids = np.argmax(terminal_vars, axis=1)         # (B,)

    best_paths: List[List[int]] = []
    for b in range(batch_size):
        active_len = int(np.sum(mask_bool[b]))
        if active_len <= 0:
            best_paths.append([pad_val] * seq_len)
            continue

        curr_tag = int(best_tag_ids[b])
        path = [curr_tag]

        for t in range(active_len - 2, -1, -1):
            curr_tag = int(backpointers[t][b, curr_tag])
            path.append(curr_tag)

        path.reverse()
        if len(path) < seq_len:
            path.extend([pad_val] * (seq_len - len(path)))
        best_paths.append(path)

    return best_paths
