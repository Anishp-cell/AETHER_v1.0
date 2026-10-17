"""Test ARN package imports and component definitions."""

def test_imports():
    import arn
    assert hasattr(arn, "__version__")
    assert arn.__version__ == "0.2.0"


def test_core_classes_exported():
    from arn import (
        ARNConfig,
        AetherRoutingNetwork,
        FactoredAttentionRouter,
        CRF,
        ARNLoss,
        ARNDataset,
        predict,
        extract_slot_arguments,
        TOKENIZER,
        TOOL_TO_IDX,
        TAG_TO_IDX,
    )
    assert len(TOOL_TO_IDX) == 18
    assert len(TAG_TO_IDX) == 26
    assert TOKENIZER is not None
