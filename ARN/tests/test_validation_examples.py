import json
import pytest
from arn.tokenizer import TOOL_REGISTRY, TOOL_TO_IDX

def test_validation_examples_provenance_and_registry():
    val_jsonl_path = "ARN/data/splits/group_disjoint_val.jsonl"
    examples_path = "ARN/results/stage1b_val_examples_and_spans.json"

    # Load raw validation samples
    val_samples = []
    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_samples.append(json.loads(line))

    # Load the 20 qualitative examples
    with open(examples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples_20 = data.get("examples_20", [])
    assert len(examples_20) == 20, f"Expected 20 examples, found {len(examples_20)}"

    for ex in examples_20:
        sample_idx = ex["sample_idx"]
        assert 0 <= sample_idx < len(val_samples), f"sample_idx {sample_idx} out of range"

        # Assert text matches the exact raw validation utterance
        raw_user_text = ""
        for msg in val_samples[sample_idx].get("messages", []):
            if msg.get("role") == "user":
                raw_user_text = msg.get("content", "").strip()

        assert ex["text"] == raw_user_text, (
            f"Sample {sample_idx} text mismatch:\n"
            f"Expected: {raw_user_text}\n"
            f"Found:    {ex['text']}"
        )

        # Assert all gold tools are in canonical TOOL_REGISTRY
        for tool in ex["gold_tools"]:
            assert tool in TOOL_REGISTRY, f"Gold tool '{tool}' not in TOOL_REGISTRY"
            assert tool in TOOL_TO_IDX, f"Gold tool '{tool}' not in TOOL_TO_IDX"

        # Assert all predicted tools are in canonical TOOL_REGISTRY
        for tool in ex["pred_tools"]:
            assert tool in TOOL_REGISTRY, f"Pred tool '{tool}' not in TOOL_REGISTRY"
            assert tool in TOOL_TO_IDX, f"Pred tool '{tool}' not in TOOL_TO_IDX"
