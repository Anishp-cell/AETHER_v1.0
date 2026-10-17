"""
Multi-Seed Baseline & Ablation Benchmark for ARN (AAAI 2027 Paper Revision)
Evaluates:
1. Full ARN (FAR Cross-Attention + CRF Tagging + Tool-Prefixed BIO)
2. JointBERT Baseline ([CLS] Pooled Linear Routing + Softmax Slot Tagger)
3. ARN No-CRF (FAR Cross-Attention + Independent Softmax Slot Tagger)

Reports Mean ± Std across 3 Random Seeds (42, 100, 2026) + 95% Bootstrap Confidence Intervals.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel, logging
logging.set_verbosity_error()

from config import ARNconfig
from tokenizer_utils import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG
from dataset import ARNDataset
from far import FactoredAttentionRouter
from crf import CRF

class JointBERT_Baseline(nn.Module):
    """
    Standard JointBERT baseline:
    - Shared BERT-tiny encoder trunk (frozen)
    - [CLS] pooled linear classifier for multi-label tool routing
    - Token-level linear classifier with Softmax for slot filling
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.tool_classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_TOOLS)
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state # (B, L, H)
        cls_rep = h[:, 0, :] # [CLS] token representation (B, H)
        
        tool_logits = self.tool_classifier(cls_rep) # (B, NUM_TOOLS)
        emissions = self.slot_tagger(h) # (B, L, NUM_TAGS)
        
        tool_preds = (torch.sigmoid(tool_logits) > 0.5)
        slot_preds = torch.argmax(emissions, dim=-1).cpu().numpy().tolist()
        dummy_weights = torch.zeros((input_ids.size(0), self.config.NUM_TOOLS, input_ids.size(1)))
        return tool_preds, slot_preds, dummy_weights


class AetherRoutingNetwork_NoCRF(nn.Module):
    """Variant of ARN that uses Softmax argmax instead of CRF for slot tagging."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.far = FactoredAttentionRouter(
            num_tools=config.NUM_TOOLS,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        )
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        tool_logits, routing_weights = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)
        tool_preds = (torch.sigmoid(tool_logits) > 0.5)
        slot_preds = torch.argmax(emissions, dim=-1).cpu().numpy().tolist()
        return tool_preds, slot_preds, routing_weights


def compute_bootstrap_ci(data, num_bootstraps=1000, ci=95):
    """Calculates 95% bootstrap confidence interval."""
    if len(data) == 0:
        return (0.0, 0.0)
    bootstraps = []
    rng = np.random.RandomState(42)
    for _ in range(num_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstraps.append(np.mean(sample))
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    return (lower, upper)


def evaluate_model_on_loader(model, loader):
    """Evaluates a model's tool accuracy, active slot accuracy, and joint match accuracy."""
    model.eval()
    tool_correct, tool_total = 0, 0
    y_true_active, y_pred_active = [], []
    joint_correct = 0
    
    pad_idx = TAG_TO_IDX.get("<PAD>", 21)
    o_idx = TAG_TO_IDX.get("O", 0)
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            tag_ids = batch["tags_ids"]
            tool_labels = batch["tool_labels"]
            
            tool_preds, tag_preds, _ = model(input_ids, mask)
            tool_preds_np = tool_preds.cpu().numpy()
            tool_labels_np = tool_labels.cpu().numpy().astype(bool)
            
            for b in range(input_ids.size(0)):
                match_t = np.array_equal(tool_preds_np[b], tool_labels_np[b])
                if match_t:
                    tool_correct += 1
                tool_total += 1
                
                match_s = True
                for t_idx in range(input_ids.size(1)):
                    gt = tag_ids[b, t_idx].item()
                    if gt == pad_idx:
                        continue
                    pr = tag_preds[b][t_idx]
                    if gt != o_idx:
                        y_true_active.append(gt)
                        y_pred_active.append(pr)
                    if pr != gt:
                        match_s = False
                if match_t and match_s:
                    joint_correct += 1
                    
    tool_acc = (tool_correct / tool_total) * 100 if tool_total > 0 else 0.0
    active_acc = (np.mean(np.array(y_true_active) == np.array(y_pred_active))) * 100 if y_true_active else 0.0
    joint_acc = (joint_correct / tool_total) * 100 if tool_total > 0 else 0.0
    return tool_acc, active_acc, joint_acc


def run_benchmark():
    print("=" * 80)
    print("RUNNING DYNAMIC MULTI-SEED BASELINE BENCHMARK FOR AAAI 2027 REVISION")
    print("=" * 80)
    
    config = ARNconfig()
    dataset_path = "ARN/aether_orchestrator_dataset_augmented.jsonl"
    if not os.path.exists(dataset_path):
        dataset_path = "aether_orchestrator_dataset_augmented.jsonl"
        
    full_dataset = ARNDataset(jsonl_path=dataset_path, tokenizer=TOKENIZER, max_seq_len=config.MAX_SEQ_LEN)
    
    # Load trained ARN model
    from model import AetherRoutingNetwork
    arn_model = AetherRoutingNetwork(config=config)
    ckpt_path = "ARN/checkpoints/arn_best_model.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/arn_best_model.pt"
    if os.path.exists(ckpt_path):
        arn_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    arn_model.eval()
    
    # Instantiate JointBERT and No-CRF models
    joint_bert = JointBERT_Baseline(config=config)
    joint_bert.eval()
    
    arn_nocrf = AetherRoutingNetwork_NoCRF(config=config)
    # Share trained FAR and slot tagger weights with No-CRF for direct ablation comparison
    arn_nocrf.far.load_state_dict(arn_model.far.state_dict())
    arn_nocrf.slot_tagger.load_state_dict(arn_model.slot_tagger.state_dict())
    arn_nocrf.eval()
    
    seeds = [42, 100, 2026]
    results = {
        "Full_ARN": {"tool_acc": [], "active_slot_acc": [], "joint_acc": []},
        "JointBERT": {"tool_acc": [], "active_slot_acc": [], "joint_acc": []},
        "ARN_NoCRF": {"tool_acc": [], "active_slot_acc": [], "joint_acc": []}
    }
    
    for seed in seeds:
        print(f"Evaluating Seed {seed}...")
        generator = torch.Generator().manual_seed(seed)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        _, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Evaluate Full ARN
        t_acc, s_acc, j_acc = evaluate_model_on_loader(arn_model, val_loader)
        results["Full_ARN"]["tool_acc"].append(t_acc)
        results["Full_ARN"]["active_slot_acc"].append(s_acc)
        results["Full_ARN"]["joint_acc"].append(j_acc)
        
        # Evaluate JointBERT
        t_acc_b, s_acc_b, j_acc_b = evaluate_model_on_loader(joint_bert, val_loader)
        results["JointBERT"]["tool_acc"].append(t_acc_b)
        results["JointBERT"]["active_slot_acc"].append(s_acc_b)
        results["JointBERT"]["joint_acc"].append(j_acc_b)
        
        # Evaluate ARN No-CRF
        t_acc_nc, s_acc_nc, j_acc_nc = evaluate_model_on_loader(arn_nocrf, val_loader)
        results["ARN_NoCRF"]["tool_acc"].append(t_acc_nc)
        results["ARN_NoCRF"]["active_slot_acc"].append(s_acc_nc)
        results["ARN_NoCRF"]["joint_acc"].append(j_acc_nc)

    print("\nSUMMARY TABLE FOR PAPER REVISION (Mean ± Std & 95% CIs across 3 random splits):")
    print("-" * 90)
    print(f"{'Model Architecture':<30} | {'Tool Routing Acc':<22} | {'Active Slot Acc':<22} | {'Joint Match Acc'}")
    print("-" * 90)
    
    for model_name, metrics in results.items():
        t_m, t_s = np.mean(metrics["tool_acc"]), np.std(metrics["tool_acc"])
        t_ci = compute_bootstrap_ci(metrics["tool_acc"])
        
        s_m, s_s = np.mean(metrics["active_slot_acc"]), np.std(metrics["active_slot_acc"])
        
        j_m, j_s = np.mean(metrics["joint_acc"]), np.std(metrics["joint_acc"])
        
        print(f"{model_name:<30} | {t_m:.2f} ± {t_s:.2f}% ({t_ci[0]:.1f}-{t_ci[1]:.1f}%) | {s_m:.2f} ± {s_s:.2f}% | {j_m:.2f} ± {j_s:.2f}%")
        
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
