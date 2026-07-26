import torch
from torch import nn
from config import ARNconfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from tokenizer_utils import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL
from dataset import ARNDataset
from model import AetherRoutingNetwork
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, classification_report

def calculate_metrics(tool_preds, tool_labels, tag_preds, target_tag_ids):
    # Convert PyTorch tensors to NumPy arrays/lists
    tool_preds_np = tool_preds.cpu().numpy()
    tool_labels_np = tool_labels.cpu().numpy().astype(bool)
    
    # ── 1. Tool Routing Metrics ──
    subset_acc = np.all(tool_preds_np == tool_labels_np, axis=1).mean()
    h_loss = hamming_loss(tool_labels_np, tool_preds_np)
    tool_f1_macro = f1_score(tool_labels_np, tool_preds_np, average='macro', zero_division=0)
    
    # ── 2. Slot Tagging (Token-level & Entity-level) Metrics ──
    pad_idx = TAG_TO_IDX.get("<PAD>", 21)
    o_idx = TAG_TO_IDX.get("O", 0)
    
    y_true_all = []
    y_pred_all = []
    
    y_true_active = []
    y_pred_active = []
    
    batch_size = target_tag_ids.size(0)
    seq_len = target_tag_ids.size(1)
    
    joint_correct = 0
    
    for b in range(batch_size):
        tools_match = np.array_equal(tool_preds_np[b], tool_labels_np[b])
        slots_match = True
        
        for t in range(seq_len):
            true_tag = target_tag_ids[b, t].item()
            if true_tag == pad_idx:
                continue
                
            pred_tag = tag_preds[b][t]
            
            y_true_all.append(true_tag)
            y_pred_all.append(pred_tag)
            
            # Track active entity slots (excluding background "O" tags)
            if true_tag != o_idx:
                y_true_active.append(true_tag)
                y_pred_active.append(pred_tag)
                
            if pred_tag != true_tag:
                slots_match = False
                
        if tools_match and slots_match:
            joint_correct += 1
            
    # Token accuracy (includes 'O')
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    token_acc = (y_true_all == y_pred_all).mean() if len(y_true_all) > 0 else 0.0
    
    # Active entity slot F1 (excludes 'O')
    active_f1_macro = f1_score(y_true_active, y_pred_active, average='macro', zero_division=0) if len(y_true_active) > 0 else 1.0
    
    # ── 3. Combined Orchestrator Metric ──
    joint_acc = joint_correct / batch_size
    
    return {
        "subset_acc": subset_acc,
        "hamming_loss": h_loss,
        "tool_f1_macro": tool_f1_macro,
        "token_acc": token_acc,
        "active_slot_f1": active_f1_macro,
        "joint_acc": joint_acc
    }

def main():
    config = ARNconfig()
    device = torch.device("cpu")
    
    dataset = ARNDataset(r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset_augmented.jsonl", TOKENIZER, config.MAX_SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = AetherRoutingNetwork(config).to(device)
    
    # ── Diagnostic: Log trainable vs frozen params ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETER BREAKDOWN")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen params:    {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  LAMBDA_SLOT:      {config.LAMBDA_SLOT}")
    print(f"{'='*60}\n")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR, weight_decay=0.01)
    schedular = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        running_tool_loss = 0.0
        running_slot_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tag_ids = batch["tags_ids"].to(device)
            tool_labels = batch["tool_labels"].to(device)
            
            optimizer.zero_grad()
            loss, tool_loss, slot_loss, tool_logits, _ = model(input_ids, attention_mask, tags=tag_ids, tool_labels=tool_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * input_ids.size(0)
            running_tool_loss += tool_loss.item() * input_ids.size(0)
            running_slot_loss += slot_loss.item() * input_ids.size(0)
            
        schedular.step()
        epoch_train_loss = running_loss / train_size
        epoch_train_tool_loss = running_tool_loss / train_size
        epoch_train_slot_loss = running_slot_loss / train_size
        
        model.eval()
        val_loss_accum = 0.0
        val_tool_loss_accum = 0.0
        val_slot_loss_accum = 0.0
        
        # Accumulate metrics across validation batches
        metric_sums = {
            "subset_acc": 0.0, "hamming_loss": 0.0, "tool_f1_macro": 0.0,
            "token_acc": 0.0, "active_slot_f1": 0.0, "joint_acc": 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tag_ids = batch["tags_ids"].to(device)
                tool_labels = batch["tool_labels"].to(device)
                
                loss, tool_loss, slot_loss, _, _ = model(input_ids, attention_mask, tags=tag_ids, tool_labels=tool_labels)
                val_loss_accum += loss.item() * input_ids.size(0)
                val_tool_loss_accum += tool_loss.item() * input_ids.size(0)
                val_slot_loss_accum += slot_loss.item() * input_ids.size(0)
                
                tool_preds, tag_preds, _ = model(input_ids, attention_mask)
                batch_metrics = calculate_metrics(tool_preds, tool_labels, tag_preds, tag_ids)
                
                for k in metric_sums:
                    metric_sums[k] += batch_metrics[k]
                    
        epoch_val_loss = val_loss_accum / val_size
        epoch_val_tool_loss = val_tool_loss_accum / val_size
        epoch_val_slot_loss = val_slot_loss_accum / val_size
        
        # Calculate averages
        num_batches = len(val_loader)
        avg_metrics = {k: v / num_batches for k, v in metric_sums.items()}
        
        print(
            f"EPOCH:{epoch+1:02d}/{config.EPOCHS} | "
            f"TRAIN LOSS:{epoch_train_loss:.4f} (Tool:{epoch_train_tool_loss:.4f}, Slot:{epoch_train_slot_loss:.4f}, "
            f"Ratio Tool:Slot = 1:{epoch_train_slot_loss/(epoch_train_tool_loss+1e-8):.1f}) | "
            f"VAL LOSS:{epoch_val_loss:.4f} (Tool:{epoch_val_tool_loss:.4f}, Slot:{epoch_val_slot_loss:.4f})\n"
            f"  [Tools] Subset Acc:{avg_metrics['subset_acc']:.4f} | Hamming Loss:{avg_metrics['hamming_loss']:.4f} | F1 Macro:{avg_metrics['tool_f1_macro']:.4f}\n"
            f"  [Slots] Token Acc:{avg_metrics['token_acc']:.4f} | Active Slot F1:{avg_metrics['active_slot_f1']:.4f}\n"
            f"  [Combined] Joint Acc:{avg_metrics['joint_acc']:.4f}"
        )
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/arn_best_model.pt")
            print("  --> Saved new best checkpoint")
            
    # ── Final Per-Class Evaluation on Best Checkpoint ──
    print("\n" + "="*70)
    print("GENERATING FINAL PER-CLASS PERFORMANCE REPORT (BEST CHECKPOINT)")
    print("="*70)
    
    best_model = AetherRoutingNetwork(config).to(device)
    best_model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location=device))
    best_model.eval()
    
    all_tool_labels = []
    all_tool_preds = []
    
    all_slot_labels = []
    all_slot_preds = []
    
    pad_idx = TAG_TO_IDX.get("<PAD>", 21)
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tag_ids = batch["tags_ids"].to(device)
            tool_labels = batch["tool_labels"].to(device)
            
            tool_preds, tag_preds, _ = best_model(input_ids, attention_mask)
            
            all_tool_labels.append(tool_labels.cpu().numpy())
            all_tool_preds.append(tool_preds.cpu().numpy())
            
            batch_size = tag_ids.size(0)
            seq_len = tag_ids.size(1)
            for b in range(batch_size):
                for t in range(seq_len):
                    true_tag = tag_ids[b, t].item()
                    if true_tag == pad_idx:
                        continue
                    all_slot_labels.append(true_tag)
                    all_slot_preds.append(tag_preds[b][t])
                    
    all_tool_labels = np.concatenate(all_tool_labels, axis=0)
    all_tool_preds = np.concatenate(all_tool_preds, axis=0)
    
    # 1. Print Tool Classification Report
    tool_names = [name for name, idx in sorted(TOOL_TO_IDX.items(), key=lambda x: x[1])]
    print("\n--- TOOL ROUTING REPORT ---")
    print(classification_report(all_tool_labels, all_tool_preds, target_names=tool_names, zero_division=0))
    
    # 2. Print Slot Tagging Report (Filter out PAD)
    unique_tags = sorted(list(set(all_slot_labels + all_slot_preds)))
    unique_tag_names = [IDX_TO_TAG[idx] for idx in unique_tags]
    print("\n--- SLOT TAGGING (NER) REPORT ---")
    print(classification_report(all_slot_labels, all_slot_preds, labels=unique_tags, target_names=unique_tag_names, zero_division=0))

if __name__ == '__main__':
    main()