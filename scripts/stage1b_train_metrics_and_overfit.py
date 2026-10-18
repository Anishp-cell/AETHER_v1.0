"""
Stage 1b: Train-set Metrics and 100-sample Overfit Test.

1. Reports train-set metrics for the Stage 1 checkpoint on group_disjoint_train.jsonl:
   - Tool Subset Accuracy (k of N, %)
   - Argument-Span F1 (Precision, Recall, F1)
   - Joint Exact Match (k of N, %)
2. Runs a 100-sample overfit memorization test:
   - Verifies ARN can overfit and reach ~100% on a 100-sample subset.
"""

from pathlib import Path
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL


def extract_spans_from_bio(tags):
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


def evaluate_subset(model, dataset, indices, device, batch_size=32):
    model.eval()
    sub_ds = Subset(dataset, indices)
    loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False)

    all_tool_preds = []
    all_tool_targets = []
    gold_spans_all = []
    pred_spans_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tool_labels = batch["tool_labels"].cpu().numpy()
            tags = batch["tags"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            tool_preds = outputs[1].cpu().numpy()
            tag_preds = outputs[2]

            for b in range(input_ids.size(0)):
                length = int(attention_mask[b].sum().item())
                gold_tag_ids = tags[b][:length].tolist()
                pred_tag_ids = tag_preds[b][:length]

                gold_tags = [IDX_TO_TAG.get(i, "O") for i in gold_tag_ids]
                pred_tags = [IDX_TO_TAG.get(i, "O") for i in pred_tag_ids]

                gold_spans_all.append(extract_spans_from_bio(gold_tags))
                pred_spans_all.append(extract_spans_from_bio(pred_tags))

            all_tool_preds.append(tool_preds)
            all_tool_targets.append(tool_labels)

    all_tool_preds = np.vstack(all_tool_preds)
    all_tool_targets = np.vstack(all_tool_targets)
    total_samples = len(indices)

    # Tool subset accuracy
    tool_matches = np.all(all_tool_preds == all_tool_targets, axis=1)
    tool_em_count = int(np.sum(tool_matches))
    tool_em = tool_em_count / max(total_samples, 1)

    # Micro F1
    tp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 1)))
    fp = int(np.sum((all_tool_preds == 1) & (all_tool_targets == 0)))
    fn = int(np.sum((all_tool_preds == 0) & (all_tool_targets == 1)))
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = (2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)) if (micro_p + micro_r) > 0 else 0.0

    # Span F1
    tot_gold = sum(len(s) for s in gold_spans_all)
    tot_pred = sum(len(s) for s in pred_spans_all)
    correct_spans = sum(len(g & p) for g, p in zip(gold_spans_all, pred_spans_all))
    span_p = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
    span_r = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
    span_f1 = (2 * span_p * span_r / max(span_p + span_r, 1e-8)) if (span_p + span_r) > 0 else 0.0

    # Joint Exact Match
    joint_matches = 0
    for i in range(total_samples):
        if tool_matches[i] and (gold_spans_all[i] == pred_spans_all[i]):
            joint_matches += 1
    joint_em = joint_matches / max(total_samples, 1)

    return {
        "total_samples": total_samples,
        "tool_subset_acc_count": tool_em_count,
        "tool_subset_acc": tool_em,
        "tool_micro_p": micro_p,
        "tool_micro_r": micro_r,
        "tool_micro_f1": micro_f1,
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "span_correct": correct_spans,
        "span_gold_total": tot_gold,
        "span_pred_total": tot_pred,
        "joint_em_count": joint_matches,
        "joint_em": joint_em,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    train_ds = ARNDataset(train_path, tokenizer=TOKENIZER)
    n_train = len(train_ds)
    print(f"Loaded train dataset with {n_train} samples.")

    # ─────────────────────────────────────────────────────────────────────────────
    # PART 1: Train-set Metrics for Stage 1 Checkpoint
    # ─────────────────────────────────────────────────────────────────────────────
    ckpt_path = Path("ARN/results/checkpoints/arn/best_model.pt")
    if ckpt_path.exists():
        print("\n" + "=" * 70)
        print("PART 1: EVALUATING STAGE 1 CHECKPOINT ON FULL TRAIN SPLIT")
        print("=" * 70)
        config = ARNConfig(lambda_slot=1.0)
        model = AetherRoutingNetwork(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        train_indices = list(range(n_train))
        metrics = evaluate_subset(model, train_ds, train_indices, device, batch_size=64)

        print(f"Total Train Samples     : {metrics['total_samples']}")
        print(f"Tool Subset Accuracy    : {metrics['tool_subset_acc']*100:.2f}% ({metrics['tool_subset_acc_count']} of {metrics['total_samples']})")
        print(f"Tool Micro F1           : {metrics['tool_micro_f1']*100:.2f}% (P={metrics['tool_micro_p']*100:.2f}%, R={metrics['tool_micro_r']*100:.2f}%)")
        print(f"Argument-Span F1        : {metrics['span_f1']*100:.2f}% (P={metrics['span_precision']*100:.2f}%, R={metrics['span_recall']*100:.2f}%)")
        print(f"  Spans Matched         : {metrics['span_correct']} of {metrics['span_gold_total']} gold ({metrics['span_pred_total']} predicted)")
        print(f"Joint Exact Match       : {metrics['joint_em']*100:.2f}% ({metrics['joint_em_count']} of {metrics['total_samples']})")

        with open("ARN/results/stage1_train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────────
    # PART 2: 100-Sample Overfit Test
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: 100-SAMPLE OVERFIT MEMORIZATION TEST")
    print("=" * 70)

    # Select 100 reproducible samples from train set
    rng = random.Random(42)
    overfit_indices = sorted(rng.sample(range(n_train), 100))
    overfit_ds = Subset(train_ds, overfit_indices)
    loader = DataLoader(overfit_ds, batch_size=16, shuffle=True)

    torch.manual_seed(42)
    overfit_config = ARNConfig(lambda_slot=1.0)
    overfit_model = AetherRoutingNetwork(overfit_config).to(device)

    # Train for 50 epochs with lr=1e-3
    optimizer = torch.optim.AdamW(
        [p for p in overfit_model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.0
    )

    print(f"Training on 100 samples for up to 60 epochs (lr=1e-3, AdamW)...")
    history = []

    for epoch in range(1, 61):
        overfit_model.train()
        total_loss_sum = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags"].to(device)
            tool_labels = batch["tool_labels"].to(device)

            optimizer.zero_grad()
            loss_outputs = overfit_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tags=tags,
                tool_labels=tool_labels
            )
            loss = loss_outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(overfit_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            steps += 1

        avg_loss = total_loss_sum / max(steps, 1)

        if epoch % 5 == 0 or epoch == 1:
            eval_res = evaluate_subset(overfit_model, train_ds, overfit_indices, device, batch_size=32)
            print(f"Epoch {epoch:2d}/60 | Loss: {avg_loss:8.4f} | "
                  f"Tool Acc: {eval_res['tool_subset_acc']*100:6.2f}% ({eval_res['tool_subset_acc_count']}/100) | "
                  f"Span F1: {eval_res['span_f1']*100:6.2f}% | "
                  f"Joint EM: {eval_res['joint_em']*100:6.2f}% ({eval_res['joint_em_count']}/100)")
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                **eval_res
            })

            # Check if ~100% reached
            if eval_res["tool_subset_acc"] >= 0.99 and eval_res["span_f1"] >= 0.99 and eval_res["joint_em"] >= 0.99:
                print(f"\nSUCCESS: Overfit criteria reached 100% at Epoch {epoch}!")
                break

    final_eval = evaluate_subset(overfit_model, train_ds, overfit_indices, device, batch_size=32)
    print("\n" + "-" * 50)
    print("FINAL 100-SAMPLE OVERFIT EVALUATION:")
    print(f"  Tool Subset Accuracy : {final_eval['tool_subset_acc']*100:.2f}% ({final_eval['tool_subset_acc_count']} of 100)")
    print(f"  Argument-Span F1     : {final_eval['span_f1']*100:.2f}%")
    print(f"  Joint Exact Match    : {final_eval['joint_em']*100:.2f}% ({final_eval['joint_em_count']} of 100)")
    print("-" * 50)

    with open("ARN/results/stage1b_overfit_results.json", "w") as f:
        json.dump({"final": final_eval, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
