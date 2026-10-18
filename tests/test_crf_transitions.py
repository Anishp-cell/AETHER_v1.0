"""Test CRF hard BIO transition masking constraints."""

import torch
from arn.crf import CRF, build_bio_transition_mask
from arn.tokenizer import TAG_TO_IDX, IDX_TO_TAG


def test_transition_mask_rules():
    trans_mask, start_mask, end_mask = build_bio_transition_mask(TAG_TO_IDX)

    o_idx = TAG_TO_IDX["O"]
    b_contact = TAG_TO_IDX["B-contact"]
    i_contact = TAG_TO_IDX["I-contact"]
    b_device = TAG_TO_IDX["B-device"]
    i_device = TAG_TO_IDX["I-device"]
    pad_idx = TAG_TO_IDX["<PAD>"]

    # Rule 1: O -> I-X is forbidden
    assert trans_mask[o_idx, i_contact].item() is False
    assert trans_mask[o_idx, i_device].item() is False

    # Rule 2: O -> B-X is allowed
    assert trans_mask[o_idx, b_contact].item() is True
    assert trans_mask[o_idx, b_device].item() is True

    # Rule 3: B-contact -> I-contact is allowed, but B-contact -> I-device is forbidden
    assert trans_mask[b_contact, i_contact].item() is True
    assert trans_mask[b_contact, i_device].item() is False

    # Rule 4: I-contact -> I-device is forbidden
    assert trans_mask[i_contact, i_device].item() is False

    # Rule 5: Start cannot be I-* or <PAD>
    assert start_mask[i_contact].item() is False
    assert start_mask[i_device].item() is False
    assert start_mask[pad_idx].item() is False
    assert start_mask[o_idx].item() is True
    assert start_mask[b_contact].item() is True


def test_viterbi_never_decodes_invalid_bio():
    """Verify that decoded Viterbi sequences never contain invalid BIO transitions under random emissions."""
    torch.manual_seed(42)
    num_tags = len(TAG_TO_IDX)
    crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)

    batch_size = 20
    seq_len = 30
    mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    for b in range(batch_size):
        active = torch.randint(5, seq_len, (1,)).item()
        mask[b, active:] = 0

    # Test across 5 different random emission distributions
    for _ in range(5):
        emissions = torch.randn((batch_size, seq_len, num_tags)) * 5.0
        decoded_paths = crf.viterbi_decode(emissions, mask)

        for b, path in enumerate(decoded_paths):
            active_len = int(mask[b].sum().item())
            active_tags = [IDX_TO_TAG[idx] for idx in path[:active_len]]

            # Assert first tag is not I-*
            assert not active_tags[0].startswith("I-"), f"Sequence started with continuation: {active_tags[0]}"

            # Assert all transitions are valid
            for t in range(len(active_tags) - 1):
                t1 = active_tags[t]
                t2 = active_tags[t + 1]

                if t1 == "O":
                    assert not t2.startswith("I-"), f"Invalid transition O -> {t2} at pos {t}"
                elif t1.startswith("B-"):
                    slot1 = t1.split("-")[1]
                    if t2.startswith("I-"):
                        slot2 = t2.split("-")[1]
                        assert slot1 == slot2, f"Invalid transition {t1} -> {t2} at pos {t}"
                elif t1.startswith("I-"):
                    slot1 = t1.split("-")[1]
                    if t2.startswith("I-"):
                        slot2 = t2.split("-")[1]
                        assert slot1 == slot2, f"Invalid transition {t1} -> {t2} at pos {t}"
