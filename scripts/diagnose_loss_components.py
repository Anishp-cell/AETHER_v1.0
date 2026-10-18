"""
Diagnose Loss Components for Stage 1 / Stage 1b.

Computes:
1. Tool multi-label BCE
2. Raw sentence-level slot NLL (CRF NLL before token normalization)
3. Length-normalized slot loss per token (raw NLL / active token count)
4. Weighted slot loss: lambda_slot * slot_loss_per_token
5. Total multi-task loss

Evaluated at:
- Start of training (initialization, seed 42)
- End of training (Stage 1 checkpoint)
On both Training split and Validation split.
"""

from pathlib import Path
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.model import AetherRoutingNetwork
from arn.tokenizer import TOKENIZER, TAG_TO_IDX


def evaluate_components(model, dataset, device, lambda_slot=1.0):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    total_tool_bce = 0.0
    total_raw_slot_nll = 0.0
    total_slot_loss_per_token = 0.0
    total_combined_loss = 0.0
    num_samples = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags"].to(device)
            tool_labels = batch["tool_labels"].to(device)

            batch_sz = input_ids.size(0)

            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state

            tool_logits, _ = model.far(h, attention_mask)
            emissions = model.slot_tagger(h)

            # 1. Tool BCE (mean over batch and tools)
            bce = model.loss_fn.tool_bce(tool_logits, tool_labels.float()).item()

            # 2. Raw sentence NLL
            sentence_nlls = model.crf.sentence_nll(emissions, tags, attention_mask)
            raw_nll_sum = sentence_nlls.sum().item()

            # 3. Active tokens per sentence
            active_counts = attention_mask.sum(dim=1).float().clamp(min=1.0)
            per_token_nll = (sentence_nlls / active_counts).sum().item()

            total_tool_bce += bce * batch_sz
            total_raw_slot_nll += raw_nll_sum
            total_slot_loss_per_token += per_token_nll
            num_samples += batch_sz
            total_tokens += active_counts.sum().item()

    avg_tool_bce = total_tool_bce / max(num_samples, 1)
    avg_raw_slot_nll = total_raw_slot_nll / max(num_samples, 1)
    avg_slot_loss_per_token = total_slot_loss_per_token / max(num_samples, 1)
    weighted_slot = lambda_slot * avg_slot_loss_per_token
    total_loss = avg_tool_bce + weighted_slot

    return {
        "num_samples": num_samples,
        "avg_active_tokens_per_sample": total_tokens / max(num_samples, 1),
        "tool_bce": avg_tool_bce,
        "raw_slot_nll": avg_raw_slot_nll,
        "slot_loss_per_token": avg_slot_loss_per_token,
        "lambda_slot": lambda_slot,
        "lambda_times_slot": weighted_slot,
        "total_loss": total_loss,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    val_path = "ARN/data/splits/group_disjoint_val.jsonl"

    train_ds = ARNDataset(train_path, tokenizer=TOKENIZER)
    val_ds = ARNDataset(val_path, tokenizer=TOKENIZER)
    print(f"Loaded train ({len(train_ds)} samples) and val ({len(val_ds)} samples).")

    # 1. At Initialization (Start of Training)
    torch.manual_seed(42)
    config = ARNConfig(lambda_slot=1.0)
    init_model = AetherRoutingNetwork(config).to(device)

    print("\n" + "=" * 70)
    print("START OF TRAINING (Random Init, Seed 42, Epoch 0)")
    print("=" * 70)
    init_train = evaluate_components(init_model, train_ds, device, lambda_slot=1.0)
    init_val = evaluate_components(init_model, val_ds, device, lambda_slot=1.0)
    print("\n--- Training Set (Start) ---")
    for k, v in init_train.items():
        print(f"  {k:30s}: {v:12.6f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    print("\n--- Validation Set (Start) ---")
    for k, v in init_val.items():
        print(f"  {k:30s}: {v:12.6f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    # 2. At End of Training (Stage 1 Checkpoint)
    ckpt_path = Path("ARN/results/checkpoints/arn/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        trained_model = AetherRoutingNetwork(config).to(device)
        trained_model.load_state_dict(ckpt["model_state_dict"])

        print("\n" + "=" * 70)
        print("END OF TRAINING (Stage 1 Checkpoint, Epoch 10)")
        print("=" * 70)
        end_train = evaluate_components(trained_model, train_ds, device, lambda_slot=1.0)
        end_val = evaluate_components(trained_model, val_ds, device, lambda_slot=1.0)
        print("\n--- Training Set (End) ---")
        for k, v in end_train.items():
            print(f"  {k:30s}: {v:12.6f}" if isinstance(v, float) else f"  {k:30s}: {v}")
        print("\n--- Validation Set (End) ---")
        for k, v in end_val.items():
            print(f"  {k:30s}: {v:12.6f}" if isinstance(v, float) else f"  {k:30s}: {v}")

        # Save to json
        out_results = {
            "start_train": init_train,
            "start_val": init_val,
            "end_train": end_train,
            "end_val": end_val,
        }
        out_file = Path("ARN/results/stage1b_loss_components.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(out_results, f, indent=2)
        print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
