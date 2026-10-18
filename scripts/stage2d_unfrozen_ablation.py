"""
Stage 2d Item 6: Unfrozen Supervision Ablation.

Compares:
  1. Tool-Only Fine-Tuning (L = L_tool)
  2. Slot-Only Fine-Tuning (L = L_slot)
  3. Joint Tool+Slot Fine-Tuning (L = L_tool + L_slot)

Under identical trunk initialization (bert_uncased_L-2_H-128_A-2),
identical learning rate budget (3e-5 trunk, 3e-4 heads, AdamW, wd=0.01),
and identical 5-fold grouped CV on Shuffle 1.
"""

import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.crf import CRF
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2c_head_independence import partition_clusters, extract_spans_from_bio
from scripts.stage2c_softmax_baselines import extract_spans_strict, compute_span_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.stage2d_ablation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnfrozenBertCLS_CRF(nn.Module):
    def __init__(self, num_tools: int = 18, num_tags: int = 26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.tool_classifier = nn.Linear(128, num_tools)
        self.slot_tagger = nn.Linear(128, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls_repr = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_repr)
        emissions = self.slot_tagger(h)
        return tool_logits, emissions


def collate_fn(batch, device):
    max_l = max(b["attention_mask"].sum().item() for b in batch)
    max_l = min(max_l, 64)
    ids = torch.stack([b["input_ids"][:max_l] for b in batch]).to(device)
    mask = torch.stack([b["attention_mask"][:max_l] for b in batch]).to(device)
    tags = torch.stack([b["tags"][:max_l] for b in batch]).to(device)
    tools = torch.stack([b["tool_labels"] for b in batch]).to(device)
    return ids, mask, tags, tools


def run_supervision_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 2d UNFROZEN SUPERVISION ABLATION on {device} ===", flush=True)

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)
    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]

    outer_folds = partition_clusters(cluster_ids, n_folds=5, seed=42)

    modes = ["tool_only", "slot_only", "joint_tool_slot"]
    results = {m: [] for m in modes}

    for fold_idx in range(5):
        val_idx = outer_folds[fold_idx]
        train_idx = [i for other_f in range(5) if other_f != fold_idx for i in outer_folds[other_f]]
        print(f"\n--- FOLD {fold_idx + 1}/5 (Train N={len(train_idx)}, Val N={len(val_idx)}) ---", flush=True)

        tr_data = [dataset.encoded[i] for i in train_idx]
        vl_data = [dataset.encoded[i] for i in val_idx]
        n_tr = len(train_idx)

        for mode in modes:
            set_seed(42 + fold_idx * 10)
            model = UnfrozenBertCLS_CRF().to(device)

            trunk_params = list(model.encoder.parameters())
            head_params = list(model.tool_classifier.parameters()) + list(model.slot_tagger.parameters()) + list(model.crf.parameters())
            opt = torch.optim.AdamW([
                {"params": trunk_params, "lr": 3e-5, "weight_decay": 0.01},
                {"params": head_params, "lr": 3e-4, "weight_decay": 0.01},
            ])

            # Train 10 epochs
            for ep in range(1, 11):
                model.train()
                perm = list(range(n_tr))
                random.shuffle(perm)
                for b in range(0, n_tr, 64):
                    batch_idx = [perm[k] for k in range(b, min(b + 64, n_tr))]
                    batch = [tr_data[k] for k in batch_idx]
                    ids, mask, tags, tools = collate_fn(batch, device)

                    opt.zero_grad()
                    tool_logits, emissions = model(ids, mask)

                    if mode == "tool_only":
                        loss = model.bce(tool_logits, tools)
                    elif mode == "slot_only":
                        active_tokens = mask.sum(dim=1).clamp(min=1)
                        crf_loss = model.crf.sentence_nll(emissions, tags, mask)
                        loss = (crf_loss / active_tokens).mean()
                    else:  # joint_tool_slot
                        active_tokens = mask.sum(dim=1).clamp(min=1)
                        tool_loss = model.bce(tool_logits, tools)
                        crf_loss = model.crf.sentence_nll(emissions, tags, mask)
                        slot_loss = (crf_loss / active_tokens).mean()
                        loss = tool_loss + slot_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            # Evaluate on held-out fold
            model.eval()
            all_tool_probs = []
            all_crf_tags = []
            gold_spans = []
            gold_tools_list = []

            with torch.no_grad():
                for b in range(0, len(vl_data), 128):
                    batch = vl_data[b : min(b + 128, len(vl_data))]
                    ids, mask, tags, tools = collate_fn(batch, device)
                    tool_logits, emissions = model(ids, mask)

                    probs = torch.sigmoid(tool_logits).cpu().numpy()
                    all_tool_probs.append(probs)
                    gold_tools_list.append(tools.cpu().numpy())

                    if mode != "tool_only":
                        crf_p = model.crf.viterbi_decode(emissions, mask)
                        m_np = mask.cpu().numpy()
                        tags_np = tags.cpu().numpy()
                        for i in range(len(batch)):
                            l_i = int(m_np[i].sum())
                            all_crf_tags.append([IDX_TO_TAG.get(idx, "O") for idx in crf_p[i][:l_i]])
                            g_seq = [IDX_TO_TAG.get(x, "O") for x in tags_np[i, :l_i]]
                            gold_spans.append(extract_spans_strict(g_seq))

            all_tool_probs = np.vstack(all_tool_probs)
            gold_tools = np.vstack(gold_tools_list)
            pred_tools = (all_tool_probs >= 0.50).astype(int)

            t_matches = np.all(pred_tools == gold_tools, axis=1)
            t_acc = float(np.mean(t_matches))

            tp = int(np.sum((pred_tools == 1) & (gold_tools == 1)))
            fp = int(np.sum((pred_tools == 1) & (gold_tools == 0)))
            fn = int(np.sum((pred_tools == 0) & (gold_tools == 1)))
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            t_f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0

            if mode != "tool_only":
                pred_spans = [extract_spans_strict(tags) for tags in all_crf_tags]
                span_metrics = compute_span_metrics(gold_spans, pred_spans)
                s_f1 = span_metrics["strict_f1"]
                slot_matches = [g == p for g, p in zip(gold_spans, pred_spans)]
                joint_em = float(np.mean([t and s for t, s in zip(t_matches, slot_matches)]))
            else:
                s_f1 = 0.0
                joint_em = 0.0

            print(f"  [{mode:15s}] Tool Acc={t_acc*100:5.2f}% | Tool F1={t_f1*100:5.2f}% | Span F1={s_f1*100:5.2f}% | Joint EM={joint_em*100:5.2f}%", flush=True)

            results[mode].append({
                "fold": fold_idx + 1,
                "tool_acc": t_acc,
                "tool_f1": t_f1,
                "span_f1": s_f1,
                "joint_em": joint_em,
            })

    # Summary
    summary = {}
    for m in modes:
        accs = [r["tool_acc"] for r in results[m]]
        f1s = [r["tool_f1"] for r in results[m]]
        sf1s = [r["span_f1"] for r in results[m]]
        jems = [r["joint_em"] for r in results[m]]

        summary[m] = {
            "mean_tool_acc": float(np.mean(accs)),
            "std_tool_acc": float(np.std(accs)),
            "mean_tool_f1": float(np.mean(f1s)),
            "mean_span_f1": float(np.mean(sf1s)),
            "std_span_f1": float(np.std(sf1s)),
            "mean_joint_em": float(np.mean(jems)),
            "std_joint_em": float(np.std(jems)),
        }

    output = {
        "summary": summary,
        "per_fold": results,
    }

    out_file = Path("ARN/results/stage2d_unfrozen_ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== FINAL SUPERVISION ABLATION SUMMARY ===", flush=True)
    for m in modes:
        s = summary[m]
        print(f"  {m:15s}: Tool Acc={s['mean_tool_acc']*100:.2f}% (+/- {s['std_tool_acc']*100:.2f}%) | "
              f"Tool F1={s['mean_tool_f1']*100:.2f}% | Span F1={s['mean_span_f1']*100:.2f}% | Joint EM={s['mean_joint_em']*100:.2f}%", flush=True)


if __name__ == "__main__":
    run_supervision_ablation()
