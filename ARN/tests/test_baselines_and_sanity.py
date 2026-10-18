"""
Phase 3 Verification Tests: Baselines, Sanity, Monotonicity, and Gradient Flow.

Verifies:
1. TF-IDF kNN and LogReg overfit sanity (>=95% accuracy on 100 samples).
2. TF-IDF kNN and LogReg random-split sanity (>=65% accuracy on leaky random split).
3. TF-IDF monotonicity with data sanity (performance increases with data fraction).
4. Gradient flow independence in frozen models (tool loss gradients only flow to tool parameters;
   slot loss gradients only flow to slot parameters).
5. Softmax decoders: decode_masked_viterbi guarantees 100% BIO validity even on noisy logits,
   while decode_greedy can produce structural BIO violations.
"""

import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
import torch.nn as nn

from arn.baselines import (
    MostFrequentBaseline,
    FirstTokenPriorBaseline,
    TfidfKnnBaseline,
    TfidfLogisticBaseline,
    decode_greedy,
    decode_masked_viterbi,
)
from arn.tokenizer import TOOL_TO_IDX, TAG_TO_IDX, TOKENIZER, align_tags_to_tokens
from arn.config import ARNConfig
from arn.model import AetherRoutingNetwork
from arn.baselines import CLSCRFModel, JointBERT, FARNoCRFModel
from arn.crf import build_bio_transition_mask


@pytest.fixture(scope="module")
def sample_100_data():
    """Load first 100 samples from the raw dataset."""
    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    with open(raw_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for i, line in enumerate(f) if i < 100 and line.strip()]

    texts = [next(m["content"] for m in s["messages"] if m["role"] == "user") for s in samples]
    labels = np.zeros((len(samples), len(TOOL_TO_IDX)), dtype=np.float32)

    for i, s in enumerate(samples):
        asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
        for c in json.loads(asst):
            labels[i, TOOL_TO_IDX[c["name"]]] = 1.0

    return texts, labels


def test_tfidf_overfit_sanity(sample_100_data):
    """Assert TF-IDF kNN and LogReg fit and evaluate on 100 samples achieve >=95% accuracy."""
    texts, labels = sample_100_data

    # kNN overfit sanity (k=1)
    knn = TfidfKnnBaseline(n_neighbors=1, min_df=1)
    knn.fit(texts, labels)
    preds_knn = knn.predict(texts, tau=0.50)
    acc_knn = float(np.all(preds_knn == labels, axis=1).mean())
    assert acc_knn >= 0.95, f"kNN 100-sample overfit failed: acc={acc_knn*100:.2f}%"

    # LogReg overfit sanity
    lr = TfidfLogisticBaseline(C=10.0, max_iter=500, min_df=1)
    lr.fit(texts, labels)
    preds_lr = lr.predict(texts, tau=0.50)
    acc_lr = float(np.all(preds_lr == labels, axis=1).mean())
    assert acc_lr >= 0.95, f"LogReg 100-sample overfit failed: acc={acc_lr*100:.2f}%"


def test_tfidf_random_split_sanity():
    """Assert TF-IDF models achieve >=65% accuracy on an unconstrained random split."""
    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    with open(raw_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for i, line in enumerate(f) if i < 1000 and line.strip()]

    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    tr_idx = indices[:800]
    te_idx = indices[800:]

    texts = [next(m["content"] for m in s["messages"] if m["role"] == "user") for s in samples]
    labels = np.zeros((len(samples), len(TOOL_TO_IDX)), dtype=np.float32)

    for i, s in enumerate(samples):
        asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
        for c in json.loads(asst):
            labels[i, TOOL_TO_IDX[c["name"]]] = 1.0

    tr_texts = [texts[i] for i in tr_idx]
    te_texts = [texts[i] for i in te_idx]
    tr_labels = labels[tr_idx]
    te_labels = labels[te_idx]

    # kNN
    knn = TfidfKnnBaseline(n_neighbors=5, min_df=2)
    knn.fit(tr_texts, tr_labels)
    knn.tune_threshold(te_texts, te_labels)
    acc_knn = float(np.all(knn.predict(te_texts) == te_labels, axis=1).mean())
    assert acc_knn >= 0.65, f"kNN random split accuracy unexpectedly low: {acc_knn*100:.2f}%"

    # LogReg
    lr = TfidfLogisticBaseline(C=1.0, max_iter=300, min_df=2)
    lr.fit(tr_texts, tr_labels)
    lr.tune_threshold(te_texts, te_labels)
    acc_lr = float(np.all(lr.predict(te_texts) == te_labels, axis=1).mean())
    assert acc_lr >= 0.70, f"LogReg random split accuracy unexpectedly low: {acc_lr*100:.2f}%"


def test_tfidf_monotonic_with_data_sanity():
    """Assert TF-IDF models improve or remain non-decreasing as training data increases."""
    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    with open(raw_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for i, line in enumerate(f) if i < 1500 and line.strip()]

    rng = random.Random(123)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    pool_tr_idx = indices[:1000]
    te_idx = indices[1000:]

    texts = [next(m["content"] for m in s["messages"] if m["role"] == "user") for s in samples]
    labels = np.zeros((len(samples), len(TOOL_TO_IDX)), dtype=np.float32)

    for i, s in enumerate(samples):
        asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
        for c in json.loads(asst):
            labels[i, TOOL_TO_IDX[c["name"]]] = 1.0

    te_texts = [texts[i] for i in te_idx]
    te_labels = labels[te_idx]

    # Evaluate at 20% (200 samples) vs 100% (1000 samples)
    n_20 = 200
    sub_tr_20 = pool_tr_idx[:n_20]
    sub_tr_100 = pool_tr_idx

    lr_20 = TfidfLogisticBaseline(C=1.0, min_df=1).fit([texts[i] for i in sub_tr_20], labels[sub_tr_20])
    lr_100 = TfidfLogisticBaseline(C=1.0, min_df=1).fit([texts[i] for i in sub_tr_100], labels[sub_tr_100])

    acc_20 = float(np.all(lr_20.predict(te_texts, tau=0.5) == te_labels, axis=1).mean())
    acc_100 = float(np.all(lr_100.predict(te_texts, tau=0.5) == te_labels, axis=1).mean())

    assert acc_100 >= acc_20, f"Expected non-decreasing accuracy with more data: 20%={acc_20*100:.2f}%, 100%={acc_100*100:.2f}%"


def test_gradient_flow_independence():
    """
    Test decoupled gradient flow in frozen models:
    - Gradients from tool loss MUST NOT flow to slot parameters.
    - Gradients from slot loss MUST NOT flow to tool parameters.
    - Backbone parameters are frozen (requires_grad = False).
    """
    config = ARNConfig(freeze_trunk=True)
    model = AetherRoutingNetwork(config=config)
    model.train()

    # Check backbone is frozen
    for p in model.encoder.parameters():
        assert not p.requires_grad, "Backbone parameter was not frozen!"

    # Dummy inputs
    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    dummy_tools = torch.zeros((4, 18), dtype=torch.float32)
    dummy_tags = torch.zeros((4, 16), dtype=torch.long)

    # Forward pass returning separate losses
    total_loss, tool_loss, slot_loss, _, _ = model(
        input_ids, attention_mask, tags=dummy_tags, tool_labels=dummy_tools
    )

    # 1. Backprop ONLY tool loss
    model.zero_grad()
    tool_loss.backward(retain_graph=True)

    # Assert slot tagger and CRF transitions have ZERO gradient
    for name, p in model.slot_tagger.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), f"Slot tagger param {name} received grad from tool loss!"

    for name, p in model.crf.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), f"CRF param {name} received grad from tool loss!"

    # Assert tool parameters DID receive gradient
    tool_has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.far.parameters())
    assert tool_has_grad, "Tool parameters did not receive gradients from tool loss!"

    # 2. Backprop ONLY slot loss
    model.zero_grad()
    slot_loss.backward()

    # Assert tool router has ZERO gradient
    for name, p in model.far.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), f"FAR param {name} received grad from slot loss!"

    # Assert slot parameters DID receive gradient
    slot_has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.slot_tagger.parameters())
    assert slot_has_grad, "Slot tagger parameters did not receive gradients from slot loss!"


def test_softmax_decoders_and_bio_validity():
    """
    Test decode_greedy vs decode_masked_viterbi on adversarial emissions:
    - decode_greedy can output invalid transitions (e.g. O -> I-contact).
    - decode_masked_viterbi strictly eliminates invalid transitions.
    """
    num_tags = len(TAG_TO_IDX)
    B, N = 2, 5
    # Construct adversarial emissions where I-contact has higher score than O or B-contact at step 1
    emissions = torch.zeros((B, N, num_tags), dtype=torch.float32)
    mask = torch.ones((B, N), dtype=torch.long)

    o_idx = TAG_TO_IDX["O"]
    i_contact_idx = TAG_TO_IDX["I-contact"]
    b_contact_idx = TAG_TO_IDX["B-contact"]

    # Step 0: O is highest
    emissions[:, 0, o_idx] = 10.0
    # Step 1: Adversarially set I-contact to 15.0 (invalid transition O -> I-contact)
    emissions[:, 1, i_contact_idx] = 15.0
    emissions[:, 1, b_contact_idx] = 5.0
    emissions[:, 1, o_idx] = 2.0

    # 1. Greedy decoding outputs the forbidden transition
    greedy_paths = decode_greedy(emissions, mask)
    assert greedy_paths[0][0] == o_idx
    assert greedy_paths[0][1] == i_contact_idx, "Greedy failed to follow argmax (expected invalid transition)"

    # 2. Masked Viterbi decoding strictly forbids O -> I-contact and chooses valid path
    masked_paths = decode_masked_viterbi(emissions, mask)
    assert masked_paths[0][0] == o_idx
    assert masked_paths[0][1] != i_contact_idx, "Masked Viterbi produced illegal O -> I-contact transition!"
    # It should pick B-contact (or O), whichever is valid with maximum score
    assert masked_paths[0][1] in [b_contact_idx, o_idx], "Masked Viterbi did not pick a valid tag!"
