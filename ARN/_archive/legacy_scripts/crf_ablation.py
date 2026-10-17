"""
CRF vs Softmax Ablation Study
Compares slot tagging performance with CRF layer vs simple softmax argmax.
Both models use frozen BERT-tiny trunk + FAR for tool routing.
"""
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from config import ARNconfig
from tokenizer_utils import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG
from dataset import ARNDataset
from model import AetherRoutingNetwork
import numpy as np

def evaluate_slot_accuracy(model, loader, use_crf=True):
    """Evaluate slot tagging accuracy (token-level, excluding PAD and O)."""
    model.eval()
    pad_idx = TAG_TO_IDX.get("<PAD>", 21)
    o_idx = TAG_TO_IDX.get("O", 0)
    
    y_true_all = []
    y_pred_all = []
    y_true_active = []
    y_pred_active = []
    tool_correct = 0
    tool_total = 0
    joint_correct = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            tag_ids = batch["tags_ids"]
            tool_labels = batch["tool_labels"]
            
            # Get predictions
            tool_preds, tag_preds, _ = model(input_ids, attention_mask)
            
            tool_preds_np = tool_preds.cpu().numpy()
            tool_labels_np = tool_labels.cpu().numpy().astype(bool)
            
            batch_size = tag_ids.size(0)
            seq_len = tag_ids.size(1)
            
            for b in range(batch_size):
                tools_match = np.array_equal(tool_preds_np[b], tool_labels_np[b])
                if tools_match:
                    tool_correct += 1
                tool_total += 1
                
                slots_match = True
                for t in range(seq_len):
                    true_tag = tag_ids[b, t].item()
                    if true_tag == pad_idx:
                        continue
                    pred_tag = tag_preds[b][t]
                    
                    y_true_all.append(true_tag)
                    y_pred_all.append(pred_tag)
                    
                    if true_tag != o_idx:
                        y_true_active.append(true_tag)
                        y_pred_active.append(pred_tag)
                    
                    if pred_tag != true_tag:
                        slots_match = False
                
                if tools_match and slots_match:
                    joint_correct += 1
    
    token_acc = np.mean(np.array(y_true_all) == np.array(y_pred_all)) if y_true_all else 0
    active_acc = np.mean(np.array(y_true_active) == np.array(y_pred_active)) if y_true_active else 0
    tool_acc = tool_correct / tool_total if tool_total > 0 else 0
    joint_acc = joint_correct / tool_total if tool_total > 0 else 0
    
    return {
        "token_acc": token_acc,
        "active_slot_acc": active_acc,
        "tool_acc": tool_acc,
        "joint_acc": joint_acc
    }


class AetherRoutingNetwork_NoCRF(nn.Module):
    """Variant of ARN that uses softmax argmax instead of CRF for slot tagging."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        from transformers import AutoModel
        from far import FactoredAttentionRouter
        
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.far = FactoredAttentionRouter(
            num_tools=config.NUM_TOOLS,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        )
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))
    
    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        
        tool_logits, routing_weights = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)
        
        if tags is not None:
            tool_loss_fn = nn.BCEWithLogitsLoss()
            tool_loss = tool_loss_fn(tool_logits, tool_labels)
            # Simple cross-entropy for slots instead of CRF
            slot_loss_fn = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])
            slot_loss = slot_loss_fn(emissions.view(-1, len(TAG_TO_IDX)), tags.view(-1))
            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return (total_loss, tool_loss, slot_loss, tool_logits, routing_weights)
        else:
            tool_preds = (torch.sigmoid(tool_logits) > 0.5)
            # Simple argmax instead of Viterbi decode
            tag_preds_tensor = torch.argmax(emissions, dim=-1)  # (B, L)
            tag_preds = tag_preds_tensor.tolist()
            return (tool_preds, tag_preds, routing_weights)


def train_model(model, train_loader, val_loader, config, model_name, epochs=15):
    """Quick training loop for ablation."""
    from torch.optim import AdamW
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR, weight_decay=0.01)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_tool = 0.0
        running_slot = 0.0
        n_samples = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            tag_ids = batch["tags_ids"]
            tool_labels = batch["tool_labels"]
            
            optimizer.zero_grad()
            loss, tool_loss, slot_loss, _, _ = model(input_ids, attention_mask, tags=tag_ids, tool_labels=tool_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            bs = input_ids.size(0)
            running_loss += loss.item() * bs
            running_tool += tool_loss.item() * bs
            running_slot += slot_loss.item() * bs
            n_samples += bs
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics = evaluate_slot_accuracy(model, val_loader)
            print(f"  [{model_name}] Epoch {epoch+1:02d} | "
                  f"Loss: {running_loss/n_samples:.4f} (T:{running_tool/n_samples:.4f} S:{running_slot/n_samples:.4f}) | "
                  f"Tool: {metrics['tool_acc']:.3f} | Slot: {metrics['active_slot_acc']:.3f} | Joint: {metrics['joint_acc']:.3f}")
    
    return evaluate_slot_accuracy(model, val_loader)


def main():
    config = ARNconfig()
    
    print("Loading dataset...")
    dataset = ARNDataset(r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset.jsonl", TOKENIZER, config.MAX_SEQ_LEN)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Use same split for both models
    gen = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    ablation_epochs = 15  # Shorter training for ablation
    
    # ── Model A: With CRF (frozen trunk) ──
    print(f"\n{'='*60}")
    print("Training Model A: Frozen BERT-tiny + CRF")
    print(f"{'='*60}")
    model_crf = AetherRoutingNetwork(config)
    crf_results = train_model(model_crf, train_loader, val_loader, config, "CRF", ablation_epochs)
    
    # ── Model B: Without CRF (softmax argmax, frozen trunk) ──
    print(f"\n{'='*60}")
    print("Training Model B: Frozen BERT-tiny + Softmax (No CRF)")
    print(f"{'='*60}")
    model_softmax = AetherRoutingNetwork_NoCRF(config)
    softmax_results = train_model(model_softmax, train_loader, val_loader, config, "Softmax", ablation_epochs)
    
    # ── Comparison ──
    print(f"\n{'='*60}")
    print("CRF vs SOFTMAX ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<25} | {'CRF':>10} | {'Softmax':>10} | {'Delta':>10}")
    print("-" * 62)
    for key in ["token_acc", "active_slot_acc", "tool_acc", "joint_acc"]:
        c = crf_results[key]
        s = softmax_results[key]
        delta = c - s
        indicator = "✅ CRF" if delta > 0.01 else ("❌ Softmax" if delta < -0.01 else "≈ Tie")
        print(f"{key:<25} | {c:>10.4f} | {s:>10.4f} | {delta:>+10.4f}  {indicator}")
    
    crf_params = sum(p.numel() for p in model_crf.crf.parameters())
    print(f"\nCRF adds {crf_params:,} parameters ({crf_params} transition weights)")
    print("If CRF ≈ Softmax or Softmax wins, consider dropping CRF to reduce complexity.")


if __name__ == "__main__":
    main()
