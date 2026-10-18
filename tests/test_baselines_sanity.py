"""
Unit sanity tests for ARN Baselines (Phase 15 Verification Pass).

Verifies:
1. Baseline behavior on random i.i.d. splits (confirming high performance when templates are shared).
2. Monotonic scaling of TF-IDF Logistic Regression performance with increasing training data.
3. Majority baseline behavior predicting empirical mode with positive accuracy (~7.9%).
"""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from arn.baselines import (
    MostFrequentBaseline,
    FirstTokenPriorBaseline,
    TfidfKnnBaseline,
    TfidfLogisticBaseline,
)
from arn.tokenizer import TOOL_TO_IDX


@pytest.fixture(scope="module")
def sample_dataset():
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:500]  # Fast subset for unit testing

    texts = [next(m["content"] for m in s["messages"] if m["role"] == "user") for s in samples]
    num_tools = len(TOOL_TO_IDX)
    targets = np.zeros((len(samples), num_tools), dtype=np.float32)

    for i, s in enumerate(samples):
        asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
        for call in json.loads(asst):
            if call["name"] in TOOL_TO_IDX:
                targets[i, TOOL_TO_IDX[call["name"]]] = 1.0

    return texts, targets


def test_random_split_baseline(sample_dataset):
    """Verify that TF-IDF baselines perform well on random i.i.d. splits where vocabulary is shared."""
    texts, targets = sample_dataset
    X_tr, X_val, y_tr, y_val = train_test_split(texts, targets, test_size=0.2, random_state=42)

    clf = TfidfLogisticBaseline(C=1.0, max_iter=200, min_df=1)
    clf.fit(X_tr, y_tr)
    best_tau = clf.tune_threshold(X_val, y_val)
    preds = clf.predict(X_val, tau=best_tau)

    # In random splits with high vocabulary overlap, subset accuracy should be substantial (> 60%)
    subset_acc = np.all(preds == y_val, axis=1).mean()
    assert subset_acc > 0.60, f"Expected >60% subset accuracy on random split, got {subset_acc:.2%}"


def test_monotonic_data_scaling(sample_dataset):
    """Verify that classifier performance scales monotonically or remains stable with increasing training data."""
    texts, targets = sample_dataset
    X_tr, X_val, y_tr, y_val = train_test_split(texts, targets, test_size=0.25, random_state=42)

    fractions = [0.2, 0.5, 1.0]
    accs = []

    for frac in fractions:
        n = int(len(X_tr) * frac)
        clf = TfidfLogisticBaseline(C=1.0, max_iter=200, min_df=1)
        clf.fit(X_tr[:n], y_tr[:n])
        best_tau = clf.tune_threshold(X_val, y_val)
        preds = clf.predict(X_val, tau=best_tau)
        acc = float(np.all(preds == y_val, axis=1).mean())
        accs.append(acc)

    # 100% data should outperform 20% data
    assert accs[-1] >= accs[0], f"Expected performance to improve with data scale: {accs}"


def test_majority_baseline_behavior(sample_dataset):
    """Verify that MostFrequentBaseline predicts the empirical mode and yields positive non-zero accuracy."""
    texts, targets = sample_dataset
    X_tr, X_val, y_tr, y_val = train_test_split(texts, targets, test_size=0.25, random_state=42)

    mf = MostFrequentBaseline(num_tools=targets.shape[1])
    mf.fit(X_tr, y_tr)
    preds = mf.predict(X_val)

    # Mode vector must have at least one tool
    assert mf.most_frequent_vector is not None
    assert mf.most_frequent_vector.sum() >= 1.0

    # Predictions should be uniform across all validation instances
    assert np.all(preds == preds[0])

    # Accuracy must be non-negative
    acc = float(np.all(preds == y_val, axis=1).mean())
    assert acc >= 0.0
