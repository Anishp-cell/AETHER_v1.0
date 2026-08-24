import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout/stderr on Windows to avoid CP1252 encoding errors
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from torch.utils.data import DataLoader, random_split
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from mobile_actions_config import (
    ARNMobileConfig, MOBILE_TOOL_TO_IDX, MOBILE_TAG_TO_IDX,
    IDX_TO_MOBILE_TOOL, IDX_TO_MOBILE_TAG
)
from mobile_actions_dataset import MobileActionsDataset
from mobile_actions_model import ARNMobileModel

def train_arn_mobile(epochs: int = 25, batch_size: int = 32, lr: float = 5e-4):
    config = ARNMobileConfig(EPOCHS=epochs, BATCH_SIZE=batch_size, LR=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Train] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load Dataset
    print(f"  [Train] Loading Mobile-Actions training split...")
    full_train_dataset = MobileActionsDataset("train")
    
    # 80/20 train/val split of the 8,693 training rows
    n_total = len(full_train_dataset)
    n_train = int(0.80 * n_total)
    n_val = n_total - n_train
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train_dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"  [Train] Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # 2. Initialize Model
    model = ARNMobileModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Train] Total parameters:     {total_params:,}")
    print(f"  [Train] Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LR,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-5)

    checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoints_dir, "arn_mobile_best.pt")

    best_val_score = 0.0

    print(f"\n{'='*78}")
    print(f"  STARTING ARN-MOBILE TRAINING ({config.EPOCHS} Epochs)")
    print(f"{'='*78}")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_tool_loss = 0.0
        train_slot_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tool_labels = batch["tool_labels"].to(device)
            tags = batch["tags"].to(device)

            optimizer.zero_grad()
            total_loss, tool_loss, slot_loss, _, _ = model(input_ids, attention_mask, tags, tool_labels)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_tool_loss += tool_loss.item()
            train_slot_loss += slot_loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_tool_correct = 0
        val_total = 0
        val_slot_correct = 0
        val_slot_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tool_labels = batch["tool_labels"].to(device)
                tags = batch["tags"].to(device)

                tool_preds, tag_preds, tool_confs, _ = model(input_ids, attention_mask)

                # Tool subset accuracy
                matched = (tool_preds == tool_labels.bool()).all(dim=1)
                val_tool_correct += matched.sum().item()
                val_total += input_ids.size(0)

                # Active slot accuracy
                for b_idx in range(len(tag_preds)):
                    active_mask = attention_mask[b_idx].bool()
                    pred_seq = torch.tensor(tag_preds[b_idx], device=device)[active_mask]
                    true_seq = tags[b_idx][active_mask]
                    val_slot_correct += (pred_seq == true_seq).sum().item()
                    val_slot_total += active_mask.sum().item()

        tool_acc = (val_tool_correct / val_total) * 100
        slot_acc = (val_slot_correct / val_slot_total) * 100 if val_slot_total > 0 else 0
        composite_score = tool_acc * 0.6 + slot_acc * 0.4

        is_best = composite_score > best_val_score
        if is_best:
            best_val_score = composite_score
            torch.save(model.state_dict(), best_ckpt_path)

        star = " *" if is_best else ""
        print(f"  Epoch [{epoch:02d}/{config.EPOCHS:02d}] "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Tool Acc: {tool_acc:5.2f}% | "
              f"Val Slot Acc: {slot_acc:5.2f}% | "
              f"Time: {epoch_time:.1f}s{star}", flush=True)

    print(f"\n{'='*78}", flush=True)
    print(f"  TRAINING COMPLETE — Best model saved to: {best_ckpt_path}", flush=True)
    print(f"{'='*78}", flush=True)
    return best_ckpt_path

if __name__ == "__main__":
    train_arn_mobile(epochs=25, batch_size=32, lr=5e-4)
