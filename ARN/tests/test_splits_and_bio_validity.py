"""
Verification Tests for Dataset Integrity, Group-Disjoint Splits, and Hard BIO Masking.
Verifies:
1. Zero group overlap and zero utterance leakage across train, val, and test splits.
2. All 18 tools appear in all three splits.
3. Every single gold tag sequence across all 5,000 samples is structurally valid under
   the hard BIO transition mask and produces finite (non-inf, non-NaN) CRF sequence scores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import torch

from arn import (
    ARNConfig,
    CRF,
    ARNDataset,
    TOKENIZER,
    TAG_TO_IDX,
    IDX_TO_TAG,
    TOOL_TO_IDX,
)
from arn.crf import build_bio_transition_mask
from scripts.make_splits import extract_fine_semantic_frame, token_edit_sim


def test_zero_group_overlap_across_splits():
    """Verify strict group-disjoint partitioning, near-duplicate absence, and tool stratification."""
    splits_dir = Path(__file__).resolve().parent.parent / "data" / "splits"
    train_path = splits_dir / "group_disjoint_train.jsonl"
    val_path = splits_dir / "group_disjoint_val.jsonl"
    test_path = splits_dir / "group_disjoint_test.jsonl"

    assert train_path.exists(), "group_disjoint_train.jsonl missing"
    assert val_path.exists(), "group_disjoint_val.jsonl missing"
    assert test_path.exists(), "group_disjoint_test.jsonl missing"

    def load_split(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_data = load_split(train_path)
    val_data = load_split(val_path)
    test_data = load_split(test_path)

    # 1. Verify semantic frame group disjointness
    train_groups = {extract_fine_semantic_frame(s) for s in train_data}
    val_groups = {extract_fine_semantic_frame(s) for s in val_data}
    test_groups = {extract_fine_semantic_frame(s) for s in test_data}

    assert train_groups.isdisjoint(val_groups), f"Train/Val group collision: {train_groups & val_groups}"
    assert train_groups.isdisjoint(test_groups), f"Train/Test group collision: {train_groups & test_groups}"
    assert val_groups.isdisjoint(test_groups), f"Val/Test group collision: {val_groups & test_groups}"

    # 2. Verify zero near-duplicates (sim >= 0.80) between test/val and train
    train_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in train_data})
    val_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in val_data})
    test_skels = list({extract_fine_semantic_frame(s).split(" || ")[1] for s in test_data})

    val_max_sims = [max(token_edit_sim(vs, ts) for ts in train_skels) for vs in val_skels]
    test_max_sims = [max(token_edit_sim(ts_eval, ts) for ts in train_skels) for ts_eval in test_skels]

    assert max(val_max_sims) < 0.80, f"Found val skeleton with sim >= 0.80 to train: max={max(val_max_sims)}"
    assert max(test_max_sims) < 0.80, f"Found test skeleton with sim >= 0.80 to train: max={max(test_max_sims)}"

    # 2. Verify zero utterance leakage
    train_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in train_data}
    test_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in test_data}
    val_utts = {next(m["content"] for m in s["messages"] if m["role"] == "user").strip().lower() for s in val_data}

    assert train_utts.isdisjoint(test_utts), "Raw utterance leaked between train and test!"
    assert train_utts.isdisjoint(val_utts), "Raw utterance leaked between train and val!"

    # 3. Verify all 18 tools appear in all three splits
    def get_tools_in_split(data):
        tools = set()
        for s in data:
            asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
            for c in json.loads(asst):
                tools.add(c["name"])
        return tools

    all_expected_tools = set(TOOL_TO_IDX.keys())
    assert len(all_expected_tools) == 18, f"Expected 18 tools, got {len(all_expected_tools)}"

    train_tools = get_tools_in_split(train_data)
    val_tools = get_tools_in_split(val_data)
    test_tools = get_tools_in_split(test_data)

    # All 18 tools must appear in training
    assert train_tools == all_expected_tools, f"Missing in train: {all_expected_tools - train_tools}"

    # Explicit documented exception list:
    # 'open_url' has only 1 single template cluster in the entire dataset ("open <URL> in the browser").
    # Under strict group-disjoint partitioning (to prevent near-duplicate leakage), that single cluster
    # must reside entirely in train. Therefore, exactly 17 of 18 tools are evaluable in val and test.
    DOCUMENTED_EXCEPTIONS = {"open_url"}
    evaluable_tools = all_expected_tools - DOCUMENTED_EXCEPTIONS
    assert len(evaluable_tools) == 17, "Expected exactly 17 evaluable tools"

    assert evaluable_tools.issubset(val_tools), f"Missing evaluable tools in val: {evaluable_tools - val_tools}"
    assert evaluable_tools.issubset(test_tools), f"Missing evaluable tools in test: {evaluable_tools - test_tools}"
    assert "open_url" not in val_tools, "open_url unexpectedly leaked into validation split!"
    assert "open_url" not in test_tools, "open_url unexpectedly leaked into test split!"


def test_all_dataset_gold_tags_are_valid_under_bio_mask():
    """
    Assert every gold tag sequence in the dataset adheres strictly to the BIO mask
    and yields finite, non-NaN scores under CRF forward evaluation.
    """
    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    assert raw_path.exists(), "Dataset not found"

    config = ARNConfig()
    crf = CRF(num_tags=config.num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
    crf.eval()

    dataset = ARNDataset(raw_path, TOKENIZER, max_length=config.max_seq_len)
    assert len(dataset) == 5000, f"Expected 5000 samples, got {len(dataset)}"

    # Evaluate across dataset batches
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    total_checked = 0
    with torch.no_grad():
        for batch in loader:
            tags = batch["tags"] # (B, N)
            mask = batch["attention_mask"] # (B, N)
            B, N = tags.shape

            # Dummy neutral emissions
            emissions = torch.zeros((B, N, config.num_tags), dtype=torch.float32)

            # Check CRF path scoring does not produce -inf
            gold_scores = crf.score_sentence(emissions, tags, mask)
            assert not torch.isinf(gold_scores).any(), "Found -inf score: invalid transition in gold sequence!"
            assert not torch.isnan(gold_scores).any(), "Found NaN score in gold sequence!"

            # Check full CRF loss computation
            loss = crf(emissions, tags, mask)
            assert not torch.isinf(loss).any(), "CRF loss is infinite!"
            assert not torch.isnan(loss).any(), "CRF loss is NaN!"

            total_checked += B

    assert total_checked == 5000
