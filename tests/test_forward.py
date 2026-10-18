"""Test forward pass and loss computation for ARN."""

import torch
from arn import ARNConfig, AetherRoutingNetwork


def test_tiny_eval_forward():
    config = ARNConfig(hard_bio_mask=True)
    model = AetherRoutingNetwork(config=config, load_pretrained=True)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    attention_mask[:, 12:] = 0

    with torch.no_grad():
        probs, preds, tags, weights = model(input_ids, attention_mask)

    assert probs.shape == (batch_size, config.num_tools)
    assert preds.shape == (batch_size, config.num_tools)
    assert len(tags) == batch_size
    assert len(tags[0]) == seq_len
    assert weights.shape == (batch_size, config.num_tools, seq_len)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_tiny_train_forward_and_backward():
    config = ARNConfig(hard_bio_mask=True)
    model = AetherRoutingNetwork(config=config, load_pretrained=True)
    model.train()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    attention_mask[:, 12:] = 0

    target_tags = torch.zeros((batch_size, seq_len), dtype=torch.long)
    target_tags[0, 1:4] = 1  # B-contact, I-contact
    target_tools = torch.zeros((batch_size, config.num_tools), dtype=torch.float32)
    target_tools[0, 0] = 1.0

    total_loss, tool_loss, slot_loss, logits, _ = model(
        input_ids,
        attention_mask,
        tags=target_tags,
        tool_labels=target_tools
    )

    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)
    assert total_loss.item() > 0

    # Test backward pass
    total_loss.backward()

    # Verify frozen trunk gradients are None
    for param in model.encoder.parameters():
        assert param.grad is None

    # Verify trainable heads received gradients
    assert model.far.W_q.weight.grad is not None
    assert model.slot_tagger.weight.grad is not None
    assert model.crf.transitions.grad is not None
