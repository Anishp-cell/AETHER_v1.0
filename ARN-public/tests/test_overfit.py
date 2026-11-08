"""
Test ARN Memorization / Overfit Capacity on 100 Samples.

Verifies that ARN can overfit and reach ~100% (>= 95%) on a 100-sample training subset.
"""

from pathlib import Path
import random
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from arn.tokenizer import TOKENIZER, IDX_TO_TAG


def extract_spans(tags):
    spans = set()
    current_type = None
    start_idx = -1
    for idx, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.add((current_type, start_idx, idx))
            current_type = tag[2:]
            start_idx = idx
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type != tag_type:
                if current_type is not None:
                    spans.add((current_type, start_idx, idx))
                current_type = tag_type
                start_idx = idx
        else:
            if current_type is not None:
                spans.add((current_type, start_idx, idx))
                current_type = None
                start_idx = -1
    if current_type is not None:
        spans.add((current_type, start_idx, len(tags)))
    return spans


@pytest.mark.slow
def test_100_sample_overfit():
    train_path = Path(__file__).resolve().parent.parent / "data" / "splits" / "group_disjoint_train.jsonl"
    assert train_path.exists(), f"Train file not found at {train_path}"

    train_ds = ARNDataset(str(train_path), tokenizer=TOKENIZER)
    rng = random.Random(42)
    indices = sorted(rng.sample(range(len(train_ds)), 100))
    sub_ds = Subset(train_ds, indices)
    loader = DataLoader(sub_ds, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    cfg = ARNConfig(freeze_trunk=False, lambda_slot=2.0)
    model = AetherRoutingNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Train for 90 epochs
    for epoch in range(1, 91):
        model.train()
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags"].to(device)
            tool_labels = batch["tool_labels"].to(device)

            opt.zero_grad()
            loss = model(input_ids, attention_mask, tags, tool_labels)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    # Evaluate
    model.eval()
    eval_loader = DataLoader(sub_ds, batch_size=32, shuffle=False)
    all_preds, all_targets = [], []
    gold_spans_all, pred_spans_all = [], []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            t_labels = batch["tool_labels"].cpu().numpy()
            tags_np = batch["tags"].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            preds = outputs[1].cpu().numpy()
            tag_preds = outputs[2]

            for b in range(input_ids.size(0)):
                length = int(attention_mask[b].sum().item())
                gold_tag_ids = tags_np[b][:length].tolist()
                pred_tag_ids = tag_preds[b][:length]
                g_tags = [IDX_TO_TAG.get(i, "O") for i in gold_tag_ids]
                p_tags = [IDX_TO_TAG.get(i, "O") for i in pred_tag_ids]
                gold_spans_all.append(extract_spans(g_tags))
                pred_spans_all.append(extract_spans(p_tags))

            all_preds.append(preds)
            all_targets.append(t_labels)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    tool_acc = float(np.all(all_preds == all_targets, axis=1).mean())

    tot_gold = sum(len(s) for s in gold_spans_all)
    tot_pred = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))
    span_p = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
    span_r = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
    span_f1 = (2 * span_p * span_r / max(span_p + span_r, 1e-8)) if (span_p + span_r) > 0 else 0.0

    joint_matches = sum(
        1 for i in range(100)
        if np.array_equal(all_preds[i], all_targets[i]) and (gold_spans_all[i] == pred_spans_all[i])
    )
    joint_em = joint_matches / 100.0

    print(f"100-Sample Overfit Result: Tool Acc={tool_acc*100:.2f}%, Span F1={span_f1*100:.2f}%, Joint EM={joint_em*100:.2f}%")
    assert tool_acc >= 0.95, f"Tool accuracy below 95%: {tool_acc}"
    assert span_f1 >= 0.95, f"Span F1 below 95%: {span_f1}"
    assert joint_em >= 0.95, f"Joint EM below 95%: {joint_em}"
