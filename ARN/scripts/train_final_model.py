"""
ARN Phase 7 Final Model Training & Artifact Packaging.

Trains the pre-registered winning model (`unfrozen_cls_crf`) on all 5,000 samples
using median inner-validation hyperparameters:
  - Trunk LR: 3e-5
  - Head LR: 1e-3
  - Weight Decay: 0.01
  - Batch Size: 64
  - Epochs: 10
  - Optimal Activation Threshold (tau): 0.30 (median across 15 outer folds)

Outputs:
  - checkpoints/arn_final_model/model.pt (PyTorch state dict)
  - checkpoints/arn_final_model/config.json (Model architecture & hyperparams)
  - checkpoints/arn_final_model/tools.json & slots.json (Vocabulary registries)
  - checkpoints/arn_final_model/tokenizer/ (Tokenizer files)
  - checkpoints/arn_final_model/manifest.json (SHA-256 checksums and provenance)
"""

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from arn.baselines import CLSCRFModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.train_final_model")


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_hash(repo_dir: Path) -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def train_final_model():
    repo_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Final Model on {device} using all 5,000 samples...")

    torch.manual_seed(42)
    np.random.seed(42)

    data_path = repo_dir / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    dataset = ARNDataset(str(data_path), max_seq_len=64)
    N = len(dataset)
    logger.info(f"Loaded {N} samples from {data_path}")

    num_tools = len(TOOL_TO_IDX)
    num_tags = len(TAG_TO_IDX)

    model = CLSCRFModel(num_tools=num_tools, num_tags=num_tags, freeze_backbone=False).to(device)

    trunk_params = list(model.encoder.parameters())
    head_params = list(model.tool_classifier.parameters()) + list(model.slot_tagger.parameters()) + list(model.crf.parameters())

    optimizer = torch.optim.AdamW([
        {"params": trunk_params, "lr": 3e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-3, "weight_decay": 0.01}
    ])

    batch_size = 64
    epochs = 10

    model.train()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(N)
        epoch_losses = []
        for b in range(0, N, batch_size):
            batch_indices = perm[b : min(b + batch_size, N)]
            batch = [dataset.encoded[i] for i in batch_indices]
            max_l = min(max(item["attention_mask"].sum().item() for item in batch), 64)

            ids = torch.stack([item["input_ids"][:max_l] for item in batch]).to(device)
            mask = torch.stack([item["attention_mask"][:max_l] for item in batch]).to(device)
            tags = torch.stack([item["tags"][:max_l] for item in batch]).to(device)
            tools = torch.stack([item["tool_labels"] for item in batch]).to(device)

            optimizer.zero_grad()
            t_loss, s_loss = model(ids, mask, tool_labels=tools, tags=tags)
            total_loss = t_loss + 0.3 * s_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())

        logger.info(f"Epoch {ep:2d}/{epochs:2d} | Mean Loss: {np.mean(epoch_losses):.4f}")

    # Create destination checkpoint directory
    out_dir = repo_dir / "checkpoints" / "arn_final_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save PyTorch state dict
    weights_path = out_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Saved model weights to {weights_path}")

    # 2. Save config
    config = {
        "architecture": "CLSCRFModel",
        "backbone": "google/bert_uncased_L-2_H-128_A-2",
        "num_tools": num_tools,
        "num_tags": num_tags,
        "hidden_size": 128,
        "max_seq_len": 64,
        "optimal_threshold": 0.30,
        "decision_rule": "pre_registered_p2_joint_em",
        "training_samples": N,
        "training_epochs": epochs,
        "trunk_lr": 3e-5,
        "head_lr": 1e-3,
        "loss_weight_slot": 0.3,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_hash": get_git_hash(repo_dir),
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 3. Save vocabularies
    tools_path = out_dir / "tools.json"
    with open(tools_path, "w", encoding="utf-8") as f:
        json.dump({"tool_to_idx": TOOL_TO_IDX, "idx_to_tool": IDX_TO_TOOL}, f, indent=2)

    slots_path = out_dir / "slots.json"
    with open(slots_path, "w", encoding="utf-8") as f:
        json.dump({"tag_to_idx": TAG_TO_IDX, "idx_to_tag": IDX_TO_TAG}, f, indent=2)

    # 4. Save tokenizer
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    TOKENIZER.save_pretrained(str(tok_dir))

    # 5. Compute SHA-256 manifest
    manifest_files = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel_name = str(p.relative_to(out_dir)).replace("\\", "/")
            manifest_files[rel_name] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size
            }

    manifest = {
        "model_name": "arn_final_model",
        "version": "1.0.0",
        "architecture": "CLSCRFModel (BERT-tiny backbone + joint multi-label tool classifier + linear-chain CRF slot tagger)",
        "selection_outcome": "Outcome A (Fine-tuning superiority supported)",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files": manifest_files
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved manifest to {manifest_path}")

    # Append to ledger.json
    ledger_path = repo_dir / "results" / "ledger.json"
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        ledger["runs"].append({
            "experiment_id": "p7_final_model_training",
            "protocol": "phase7_final_model",
            "model": "unfrozen_cls_crf",
            "hyperparameters": config,
            "artifact_dir": str(out_dir),
            "manifest": manifest,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

    logger.info("Phase 7 Final Model Training & Packaging Complete!")


if __name__ == "__main__":
    train_final_model()
