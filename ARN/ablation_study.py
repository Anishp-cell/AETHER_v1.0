import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import numpy as np

from config import ARNconfig
from tokenizer_utils import TOKENIZER, TAG_TO_IDX
from dataset import ARNDataset
from embedding import FactorizedEmbedding
from encoder import TransformerEncoder
from far import FactoredAttentionRouter
from crf import CRF

# ── 1. Define Ablated Model Architectures ────────────────────────────────

class AblatedARN(nn.Module):
    def __init__(self, config: ARNconfig, ablate_far=False, ablate_crf=False, ablate_factorization=False):
        super().__init__()
        self.config = config
        self.ablate_far = ablate_far
        self.ablate_crf = ablate_crf
        self.ablate_factorization = ablate_factorization
        
        # Embedding Ablation
        if ablate_factorization:
            # Standard embedding (direct projection, no factorized dimension reduction)
            self.embedding = nn.Sequential(
                nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM),
                nn.Dropout(config.DROPOUT),
                nn.LayerNorm(config.HIDDEN_DIM)
            )
        else:
            self.embedding = FactorizedEmbedding(
                vocab_size=config.VOCAB_SIZE,
                embed_dim_small=config.EMBED_DIM_SMALL,
                hidden_dim=config.HIDDEN_DIM,
                max_seq_len=config.MAX_SEQ_LEN,
                dropout=config.DROPOUT
            )
            
        self.encoder = TransformerEncoder(
            num_layers=config.NUM_LAYERS,
            hidden_dim=config.HIDDEN_DIM,
            num_heads=config.NUM_HEADS,
            ffn_dim=config.FFN_DIM,
            dropout=config.DROPOUT
        )
        
        # FAR Router Ablation
        if ablate_far:
            # Replaced with standard CLS-pooled linear classifier
            self.tool_classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_TOOLS)
        else:
            self.far = FactoredAttentionRouter(
                num_tools=config.NUM_TOOLS,
                hidden_dim=config.HIDDEN_DIM,
                dropout=config.DROPOUT
            )
            
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))
        
        # CRF Tagging Ablation
        if ablate_crf:
            # Plain Softmax sequence tagger using Cross Entropy
            self.crf_loss_fn = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX.get("<PAD>", 21))
        else:
            self.crf = CRF(num_tags=len(TAG_TO_IDX))

    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        # 1. Embeddings
        if self.ablate_factorization:
            x = self.embedding(input_ids)
            # Add positional encodings manually for direct embedding ablation
            seq_len = input_ids.shape[1]
            pe = torch.zeros(seq_len, self.config.HIDDEN_DIM, device=input_ids.device)
            # register pe slice in forward
            x = x + pe.unsqueeze(0)[:, :seq_len, :]
        else:
            x = self.embedding(input_ids)
            
        # 2. Encoder
        h = self.encoder(x, attention_mask)
        
        # 3. Tool Routing
        if self.ablate_far:
            # CLS pooling (take first hidden state representing [CLS] token)
            cls_repr = h[:, 0, :]
            tool_logits = self.tool_classifier(cls_repr)
            routing_weights = None
        else:
            tool_logits, routing_weights = self.far(h, attention_mask)
            
        # 4. Slot Tagging
        emissions = self.slot_tagger(h)
        
        if tags is not None:
            # Train Mode
            tool_loss_fn = nn.BCEWithLogitsLoss()
            tool_loss = tool_loss_fn(tool_logits, tool_labels)
            
            if self.ablate_crf:
                # Token-level Cross Entropy Loss
                # Reshape emissions to (B * L, C) and tags to (B * L)
                slot_loss = self.crf_loss_fn(emissions.view(-1, len(TAG_TO_IDX)), tags.view(-1))
            else:
                slot_loss = self.crf.loss(emissions, tags, attention_mask)
                
            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return total_loss
        else:
            # Inference Mode
            tool_preds = (torch.sigmoid(tool_logits) > 0.5)
            if self.ablate_crf:
                tag_preds = torch.argmax(emissions, dim=-1).cpu().numpy().tolist()
            else:
                tag_preds = self.crf.viterbi_decode(emissions, attention_mask)
            return tool_preds, tag_preds

# ── 2. Evaluation Helper ──────────────────────────────────────────────────

def evaluate_model(model, val_loader, device):
    model.eval()
    correct_tools = 0
    correct_slots = 0
    joint_correct = 0
    total_samples = 0
    
    pad_idx = TAG_TO_IDX.get("<PAD>", 21)
    
    # Measure Latency (warm up + average loop)
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tag_ids = batch["tags_ids"].to(device)
            tool_labels = batch["tool_labels"].to(device)
            
            tool_preds, tag_preds = model(input_ids, attention_mask)
            
            tool_preds_np = tool_preds.cpu().numpy()
            tool_labels_np = tool_labels.cpu().numpy().astype(bool)
            
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            total_samples += batch_size
            
            for b in range(batch_size):
                tools_match = np.array_equal(tool_preds_np[b], tool_labels_np[b])
                slots_match = True
                
                for t in range(seq_len):
                    true_tag = tag_ids[b, t].item()
                    if true_tag == pad_idx:
                        continue
                    if tag_preds[b][t] != true_tag:
                        slots_match = False
                        break
                        
                if tools_match:
                    correct_tools += 1
                if slots_match:
                    correct_slots += 1
                if tools_match and slots_match:
                    joint_correct += 1
                    
    end_time = time.perf_counter()
    latency_ms = ((end_time - start_time) / total_samples) * 1000
    
    return {
        "tool_acc": (correct_tools / total_samples) * 100,
        "slot_acc": (correct_slots / total_samples) * 100,
        "joint_acc": (joint_correct / total_samples) * 100,
        "latency_ms": latency_ms
    }

# ── 3. Run Training Loop for an Ablation Variant ──────────────────────────

def train_variant(config, ablate_far=False, ablate_crf=False, ablate_factorization=False):
    device = torch.device("cpu")
    dataset = ARNDataset("aether_orchestrator_dataset.jsonl", TOKENIZER, config.MAX_SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = AblatedARN(
        config, 
        ablate_far=ablate_far, 
        ablate_crf=ablate_crf, 
        ablate_factorization=ablate_factorization
    ).to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)
    
    for epoch in range(15):  # Train for 15 epochs to get a fast comparative reading
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tag_ids = batch["tags_ids"].to(device)
            tool_labels = batch["tool_labels"].to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, tags=tag_ids, tool_labels=tool_labels)
            loss.backward()
            optimizer.step()
            
    metrics = evaluate_model(model, val_loader, device)
    metrics["parameters"] = total_params
    return metrics

def main():
    config = ARNconfig()
    
    print("="*70)
    print("RUNNING ARN ABLATION EXPERIMENTS (15 EPOCHS)")
    print("="*70)
    
    results = {}
    
    # ── 1. Full Model (Base) ──
    print("\n[+] Training Full Model (Base)...")
    results["Full Model (Base)"] = train_variant(config)
    
    # ── 2. Ablation: No-FAR (Linear CLS Router) ──
    print("\n[+] Training Ablated Variant: No-FAR (CLS Classifier)...")
    results["No-FAR (CLS Classify)"] = train_variant(config, ablate_far=True)
    
    # ── 3. Ablation: No-CRF (Softmax Tagging) ──
    print("\n[+] Training Ablated Variant: No-CRF (Softmax Tagging)...")
    results["No-CRF (Softmax Tag)"] = train_variant(config, ablate_crf=True)
    
    # ── 4. Ablation: No-Factorization (Standard Embeddings) ──
    print("\n[+] Training Ablated Variant: No-Factorization (Std Embed)...")
    results["No-Factorization (Std Embed)"] = train_variant(config, ablate_factorization=True)
    
    # ── 4. Print Comparative Table ──
    print("\n" + "="*85)
    print(f"{'Ablation Variant':<28} | {'Params':<9} | {'Tool Acc (%)':<12} | {'Slot Acc (%)':<12} | {'Joint Acc (%)':<13} | {'Latency (ms)':<10}")
    print("="*85)
    for variant, metrics in results.items():
        print(
            f"{variant:<28} | "
            f"{metrics['parameters']/1e6:<7.2f}M | "
            f"{metrics['tool_acc']:<12.1f} | "
            f"{metrics['slot_acc']:<12.1f} | "
            f"{metrics['joint_acc']:<13.1f} | "
            f"{metrics['latency_ms']:<10.2f}"
        )
    print("="*85)

if __name__ == "__main__":
    main()
