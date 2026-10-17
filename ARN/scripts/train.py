#!/usr/bin/env python3
"""
ARN Training Script for Main Model and Baselines.
Supports training:
- arn: Full Aether Routing Network (BERT-tiny + FAR + CRF with hard BIO masking)
- jointbert: JointBERT baseline ([CLS] tool classification + independent softmax token slot tagging)
- cls_crf: Ablation 1 ([CLS] tool classification + Linear-Chain CRF)
- far_no_crf: Ablation 2 (FAR + independent softmax token slot tagging)
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from arn import (
    ARNConfig,
    AetherRoutingNetwork,
    JointBERT,
    CLSCRFModel,
    FARNoCRFModel,
    ARNDataset,
    TOKENIZER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_type: str, config: ARNConfig):
    if model_type == "arn":
        return AetherRoutingNetwork(config)
    elif model_type == "jointbert":
        return JointBERT(config)
    elif model_type == "cls_crf":
        return CLSCRFModel(config)
    elif model_type == "far_no_crf":
        return FARNoCRFModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_epoch(model, dataloader, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss_accum = 0.0
    tool_loss_accum = 0.0
    slot_loss_accum = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tags = batch["tags"].to(device)
        tool_labels = batch["tool_labels"].to(device)

        optimizer.zero_grad()
        loss_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tags=tags,
            tool_labels=tool_labels
        )
        total_loss, tool_loss, slot_loss = loss_outputs[0], loss_outputs[1], loss_outputs[2]

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_accum += total_loss.item()
        tool_loss_accum += tool_loss.item()
        slot_loss_accum += slot_loss.item()
        steps += 1

    return {
        "total_loss": total_loss_accum / max(steps, 1),
        "tool_loss": tool_loss_accum / max(steps, 1),
        "slot_loss": slot_loss_accum / max(steps, 1),
    }


@torch.no_grad()
def evaluate_val(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    total_loss_accum = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tags = batch["tags"].to(device)
        tool_labels = batch["tool_labels"].to(device)

        loss_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tags=tags,
            tool_labels=tool_labels
        )
        total_loss = loss_outputs[0]
        total_loss_accum += total_loss.item()
        steps += 1

    return {"val_loss": total_loss_accum / max(steps, 1)}


def main():
    parser = argparse.ArgumentParser(description="ARN Model Training Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model-type", type=str, default="arn",
                        choices=["arn", "jointbert", "cls_crf", "far_no_crf"],
                        help="Model architecture variant")
    parser.add_argument("--train-data", type=str, default="data/splits/group_disjoint_train.jsonl",
                        help="Path to training data")
    parser.add_argument("--val-data", type=str, default="data/splits/group_disjoint_val.jsonl",
                        help="Path to validation data")
    parser.add_argument("--output-dir", type=str, default="results/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--slot-loss-weight", type=float, default=None, help="Slot loss weight override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu)")
    args = parser.parse_args()

    set_seed(args.seed)

    config_path = Path(args.config)
    if config_path.exists():
        config = ARNConfig.from_yaml(config_path)
    else:
        config = ARNConfig()

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.slot_loss_weight is not None:
        config.lambda_slot = args.slot_loss_weight

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Initializing %s training on %s (epochs=%d, bs=%d, lr=%.2e, lambda_slot=%.2f)",
                args.model_type, device, config.num_epochs, config.batch_size, config.learning_rate, config.lambda_slot)

    model = get_model(args.model_type, config).to(device)

    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    if not train_path.exists() or not val_path.exists():
        logger.error("Dataset not found: train=%s, val=%s. Please generate splits first.", train_path, val_path)
        return

    train_dataset = ARNDataset(train_path, TOKENIZER, max_length=config.max_seq_len)
    val_dataset = ARNDataset(val_path, TOKENIZER, max_length=config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    out_dir = Path(args.output_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "best_model.pt"

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_val(model, val_loader, device)

        logger.info("Epoch %2d/%2d | Train Loss: %.4f (Tool: %.4f, Slot: %.4f) | Val Loss: %.4f",
                    epoch, config.num_epochs,
                    train_metrics["total_loss"], train_metrics["tool_loss"], train_metrics["slot_loss"],
                    val_metrics["val_loss"])

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch,
                "model_type": args.model_type,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": config.to_dict(),
            }, best_ckpt_path)
            logger.info("  [*] Saved new best checkpoint to %s (Val Loss: %.4f)", best_ckpt_path, best_val_loss)

    logger.info("Training complete for %s. Best Val Loss: %.4f", args.model_type, best_val_loss)


if __name__ == "__main__":
    main()
