"""
Unit tests for Human Evaluation Template and Evaluation Harness (Phase 9).
"""

import json
from pathlib import Path
import pytest
import numpy as np

from scripts.make_human_eval_template import TOOL_ARG_SCHEMA, ALL_TOOLS
from scripts.run_human_eval import compute_cohens_kappa, generate_mock_records, evaluate_records


def test_human_eval_template_schema_integrity():
    """Verify that human_eval_template.jsonl adheres strictly to schema specifications."""
    template_path = Path("ARN/data/human_eval_template.jsonl")
    if not template_path.exists():
        template_path = Path("data/human_eval_template.jsonl")

    assert template_path.exists(), f"Template not found at {template_path}"

    with open(template_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]

    assert len(records) == 250, f"Expected exactly 250 records, got {len(records)}"

    single_tool_records = [r for r in records if r["category"] == "single_tool"]
    multi_tool_records = [r for r in records if r["category"] == "multi_tool"]

    assert len(single_tool_records) == 180, f"Expected 180 single tool records, got {len(single_tool_records)}"
    assert len(multi_tool_records) == 70, f"Expected 70 multi tool records, got {len(multi_tool_records)}"

    # Check each single tool has exactly 10 records
    per_tool_counts = {t: 0 for t in ALL_TOOLS}
    for r in single_tool_records:
        tool = r["gold_tools"][0]
        per_tool_counts[tool] += 1
        assert tool in TOOL_ARG_SCHEMA, f"Unknown tool: {tool}"
        # Verify slot keys match schema
        for slot in TOOL_ARG_SCHEMA[tool]:
            assert slot in r["gold_slots"], f"Missing slot {slot} in template for {tool}"

    for t, count in per_tool_counts.items():
        assert count == 10, f"Tool {t} has {count} records, expected 10"


def test_cohens_kappa_calculation():
    """Test Cohen's kappa computation on known scenarios."""
    # Perfect agreement
    r1 = [1, 2, 3, 1, 2]
    r2 = [1, 2, 3, 1, 2]
    assert np.isclose(compute_cohens_kappa(r1, r2), 1.0)

    # Empty
    assert np.isclose(compute_cohens_kappa([], []), 1.0)

    # Single class
    assert np.isclose(compute_cohens_kappa([1, 1], [1, 1]), 1.0)

    # Known disagreement
    r1 = [1, 0, 1, 0]
    r2 = [0, 1, 0, 1]
    assert compute_cohens_kappa(r1, r2) < 0.0


def test_mock_records_validity():
    """Verify mock records satisfy schema requirements."""
    mock_records = generate_mock_records()
    assert len(mock_records) == 10
    for r in mock_records:
        assert "utterance" in r and len(r["utterance"]) > 0
        assert "gold_tools" in r and len(r["gold_tools"]) > 0
        assert "gold_slots" in r
        for t in r["gold_tools"]:
            assert t in TOOL_ARG_SCHEMA
