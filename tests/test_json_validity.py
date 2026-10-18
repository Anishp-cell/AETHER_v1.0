"""Test JSON validity and crash-free execution of inference output."""

import json
from arn import ARNConfig, AetherRoutingNetwork, predict, TOKENIZER


def test_predict_returns_valid_json_structure():
    config = ARNConfig(hard_bio_mask=True)
    model = AetherRoutingNetwork(config=config, load_pretrained=True)
    model.eval()

    test_queries = [
        "turn on the living room lights",
        "send an email to Alex saying meeting is delayed",
        "what is the time right now",
        "set a timer for 10 minutes",
        "this is just casual chit chat with no tool command",
        "",
        "   ",
        "!!! ??? ...",
        "12345 67890",
        '{"malicious": "json"} <script>alert(1)</script>',
        "DROP TABLE users; rm -rf /",
    ]

    for q in test_queries:
        out = predict(q, model, TOKENIZER, config)

        # 1. Structural assertions
        assert isinstance(out, dict)
        assert "tools" in out
        assert "confidence" in out
        assert "tool_confidences" in out
        assert "arguments" in out
        assert "cloud_fallback" in out
        assert "rejected" in out

        assert isinstance(out["tools"], list)
        assert isinstance(out["confidence"], float)
        assert isinstance(out["tool_confidences"], dict)
        assert isinstance(out["arguments"], dict)
        assert isinstance(out["cloud_fallback"], bool)
        assert isinstance(out["rejected"], bool)

        # 2. Strict JSON serializability test
        serialized = json.dumps(out)
        deserialized = json.loads(serialized)
        assert deserialized == out
