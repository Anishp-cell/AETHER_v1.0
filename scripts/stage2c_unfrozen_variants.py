"""
Stage 2c Item 7: Unfrozen Variants (Fine-Tuned BERT-tiny + CLS+CRF & + ARN).

Evaluates:
1. End-to-end fine-tuning with differential LR: 2e-5 for BERT-tiny trunk, 1e-3 for tool and slot heads.
2. Accuracy: Tool Subset Acc, Micro F1, Span F1, Joint EM.
3. Model Size: FP32 vs. Dynamic INT8 Quantized (MB).
4. Latency Benchmark: Batch size 1, seq_len 32, 100 warmup, 500 benchmark iterations on CPU and GPU (p50, p95, p99 ms).
"""

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG
from arn.far import FactoredAttentionRouter
from arn.crf import CRF
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2c_unfrozen")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


class UnfrozenBertARN(nn.Module):
    def __init__(self, num_tools: int = 18, num_tags: int = 26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.far = FactoredAttentionRouter(num_tools=num_tools, hidden_dim=128, dropout=0.1)
        self.slot_tagger = nn.Linear(128, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        tool_logits, _ = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)
        return tool_logits, emissions


def extract_spans_strict(tags: List[str]):
    spans = set()
    current_label = None
    start = -1
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_label is not None:
                spans.add((current_label, start, i - 1))
            current_label = tag[2:]
            start = i
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label is None or label != current_label:
                if current_label is not None:
                    spans.add((current_label, start, i - 1))
                current_label = label
                start = i
        else:
            if current_label is not None:
                spans.add((current_label, start, i - 1))
                current_label = None
                start = -1
    if current_label is not None:
        spans.add((current_label, start, len(tags) - 1))
    return spans


def measure_model_size(model: nn.Module) -> Tuple[float, float]:
    """Measure FP32 and INT8 dynamically quantized model sizes in MB."""
    # FP32 size
    buffer_fp32 = io.BytesIO()
    torch.save(model.state_dict(), buffer_fp32)
    fp32_mb = len(buffer_fp32.getvalue()) / (1024 * 1024)

    # Dynamic INT8 quantization on linear layers (CPU)
    model_cpu = model.to("cpu")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    buffer_int8 = io.BytesIO()
    torch.save(quantized_model.state_dict(), buffer_int8)
    int8_mb = len(buffer_int8.getvalue()) / (1024 * 1024)

    return fp32_mb, int8_mb


def benchmark_latency(model: nn.Module, device: torch.device, n_warmup: int = 100, n_iter: int = 500) -> Dict[str, float]:
    """Benchmark batch size 1 inference latency in milliseconds."""
    model.eval()
    model.to(device)

    dummy_ids = torch.randint(0, 1000, (1, 32), device=device)
    dummy_mask = torch.ones((1, 32), dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            t_logits, emissions = model(dummy_ids, dummy_mask)
            _ = model.crf.viterbi_decode(emissions, dummy_mask)

    latencies = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(n_iter):
                starter.record()
                t_logits, emissions = model(dummy_ids, dummy_mask)
                _ = model.crf.viterbi_decode(emissions, dummy_mask)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))  # milliseconds
    else:
        with torch.no_grad():
            for _ in range(n_iter):
                t0 = time.perf_counter()
                t_logits, emissions = model(dummy_ids, dummy_mask)
                _ = model.crf.viterbi_decode(emissions, dummy_mask)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
    }


def run_unfrozen_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Unfrozen Variants Evaluation on device: {device}", flush=True)

    # 1. Load Data
    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)

    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]

    # Use Fold 1 split from Shuffle 1 as clean out-of-domain evaluation
    from scripts.stage2_master_suite import partition_clusters
    outer_folds = partition_clusters(cluster_ids, n_folds=5, seed=42)
    val_idx = outer_folds[0]
    train_idx = [i for other_f in range(1, 5) for i in outer_folds[other_f]]

    print(f"Data split: {len(train_idx)} train samples, {len(val_idx)} validation samples", flush=True)

    tr_items = [dataset[i] for i in train_idx]
    val_items = [dataset[i] for i in val_idx]

    tr_ids = torch.stack([x["input_ids"] for x in tr_items]).to(device)
    tr_mask = torch.stack([x["attention_mask"] for x in tr_items]).to(device)
    tr_tags = torch.stack([x["tags"] for x in tr_items]).to(device)
    tr_tools = torch.stack([x["tool_labels"] for x in tr_items]).to(device)

    val_ids = torch.stack([x["input_ids"] for x in val_items]).to(device)
    val_mask = torch.stack([x["attention_mask"] for x in val_items]).to(device)
    val_tags = torch.stack([x["tags"] for x in val_items]).to(device)
    val_tools = torch.stack([x["tool_labels"] for x in val_items]).to(device)

    models_to_test = {
        "unfrozen_bert_cls_crf": UnfrozenBertCLS_CRF(),
        "unfrozen_bert_arn": UnfrozenBertARN(),
    }

    results = {}

    for name, model in models_to_test.items():
        print(f"\n{'='*75}\nTRAINING & BENCHMARKING: {name.upper()}\n{'='*75}", flush=True)
        set_seed(42)
        model.to(device)

        # Differential Learning Rate Optimizer: 2e-5 for BERT trunk, 1e-3 for heads
        head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
        trunk_params = list(model.encoder.parameters())

        optimizer = torch.optim.AdamW([
            {"params": trunk_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
        ])

        n_tr = len(train_idx)
        best_val_loss = float("inf")
        best_state = None
        patience = 4
        patience_cnt = 0

        # Train for 12 epochs with early stopping
        for epoch in range(1, 13):
            model.train()
            perm = torch.randperm(n_tr)
            for b in range(0, n_tr, 32):
                b_idx = perm[b : min(b + 32, n_tr)]
                m_b = tr_mask[b_idx]
                max_l = int(m_b.sum(dim=1).max().item())

                ids_b = tr_ids[b_idx, :max_l]
                mask_b = m_b[:, :max_l]
                tags_b = tr_tags[b_idx, :max_l]
                tools_b = tr_tools[b_idx]

                optimizer.zero_grad()
                tool_logits, emissions = model(ids_b, mask_b)
                tool_loss = model.bce(tool_logits, tools_b)
                sentence_nlls = model.crf.sentence_nll(emissions, tags_b, mask_b)
                active = mask_b.sum(dim=1).float().clamp(min=1.0)
                slot_loss = (sentence_nlls / active).mean()
                total_loss = tool_loss + slot_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Fast validation loss
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in range(0, len(val_idx), 64):
                    m_v = val_mask[vb : vb + 64]
                    max_l = int(m_v.sum(dim=1).max().item())
                    ids_v = val_ids[vb : vb + 64, :max_l]
                    mask_v = m_v[:, :max_l]
                    tags_v = val_tags[vb : vb + 64, :max_l]
                    tools_v = val_tools[vb : vb + 64]

                    tool_logits, emissions = model(ids_v, mask_v)
                    tool_loss = model.bce(tool_logits, tools_v)
                    sentence_nlls = model.crf.sentence_nll(emissions, tags_v, mask_v)
                    active = mask_v.sum(dim=1).float().clamp(min=1.0)
                    slot_loss = (sentence_nlls / active).mean()
                    val_losses.append((tool_loss + slot_loss).item())

            cur_v_loss = float(np.mean(val_losses))
            print(f"  Epoch {epoch:2d}/12 - Val Loss: {cur_v_loss:.4f}", flush=True)

            if cur_v_loss < best_val_loss - 0.005:
                best_val_loss = cur_v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  Early stopping triggered at epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 2. Evaluate Full Metrics on Out-of-Domain Val
        model.eval()
        all_tool_probs = []
        all_pred_spans = []
        all_gold_spans = []

        with torch.no_grad():
            for vb in range(0, len(val_idx), 64):
                m_v = val_mask[vb : vb + 64]
                max_l = int(m_v.sum(dim=1).max().item())
                ids_v = val_ids[vb : vb + 64, :max_l]
                mask_v = m_v[:, :max_l]
                tags_v = val_tags[vb : vb + 64, :max_l]

                tool_logits, emissions = model(ids_v, mask_v)
                probs = torch.sigmoid(tool_logits).cpu().numpy()
                all_tool_probs.append(probs)

                tag_preds = model.crf.viterbi_decode(emissions, mask_v)
                tags_np = tags_v.cpu().numpy()
                mask_np = mask_v.cpu().numpy()

                for i in range(len(ids_v)):
                    l_i = int(mask_np[i].sum())
                    g_seq = [IDX_TO_TAG.get(idx, "O") for idx in tags_np[i][:l_i].tolist()]
                    p_seq = [IDX_TO_TAG.get(idx, "O") for idx in tag_preds[i][:l_i]]
                    all_gold_spans.append(extract_spans_strict(g_seq))
                    all_pred_spans.append(extract_spans_strict(p_seq))

        probs = np.vstack(all_tool_probs)
        targets = val_tools.cpu().numpy()
        preds = (probs >= 0.40).astype(int)

        tool_matches = np.all(preds == targets, axis=1)
        tool_acc = float(np.mean(tool_matches))

        tp = int(np.sum((preds == 1) & (targets == 1)))
        fp = int(np.sum((preds == 1) & (targets == 0)))
        fn = int(np.sum((preds == 0) & (targets == 1)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        tool_f1 = (2 * p * r / max(p + r, 1e-8)) if (p + r) > 0 else 0.0

        tot_gold = sum(len(s) for s in all_gold_spans)
        tot_pred = sum(len(s) for s in all_pred_spans)
        correct_spans = sum(len(g & p) for g, p in zip(all_gold_spans, all_pred_spans))
        span_p = correct_spans / max(tot_pred, 1) if tot_pred > 0 else 0.0
        span_r = correct_spans / max(tot_gold, 1) if tot_gold > 0 else 0.0
        span_f1 = (2 * span_p * span_r / max(span_p + span_r, 1e-8)) if (span_p + span_r) > 0 else 0.0

        joint_matches = sum(
            1 for i in range(len(val_idx))
            if tool_matches[i] and (all_gold_spans[i] == all_pred_spans[i])
        )
        joint_em = joint_matches / max(len(val_idx), 1)

        # 3. Model Size (FP32 vs INT8)
        print("Measuring model footprint...", flush=True)
        fp32_size, int8_size = measure_model_size(model)

        # 4. Latency Benchmarking (CPU & GPU)
        print("Benchmarking latency on CPU and GPU...", flush=True)
        cpu_latency = benchmark_latency(model, torch.device("cpu"), n_warmup=50, n_iter=200)
        gpu_latency = benchmark_latency(model, device, n_warmup=100, n_iter=500) if device.type == "cuda" else cpu_latency

        results[name] = {
            "tool_subset_acc": tool_acc,
            "tool_micro_f1": tool_f1,
            "span_f1": span_f1,
            "span_correct": correct_spans,
            "span_gold_total": tot_gold,
            "joint_em": joint_em,
            "fp32_size_mb": fp32_size,
            "int8_size_mb": int8_size,
            "compression_ratio": fp32_size / int8_size,
            "cpu_latency_ms": cpu_latency,
            "gpu_latency_ms": gpu_latency,
        }

        print(
            f"  Metrics: Tool Acc={tool_acc*100:.2f}%, Micro F1={tool_f1*100:.2f}%, "
            f"Span F1={span_f1*100:.2f}%, Joint EM={joint_em*100:.2f}%",
            flush=True
        )
        print(f"  Footprint: FP32 = {fp32_size:.2f} MB | INT8 Dynamic = {int8_size:.2f} MB ({fp32_size/int8_size:.1f}x compression)", flush=True)
        print(f"  CPU Latency (bs=1): p50={cpu_latency['p50']:.2f}ms | p95={cpu_latency['p95']:.2f}ms | p99={cpu_latency['p99']:.2f}ms", flush=True)
        if device.type == "cuda":
            print(f"  GPU Latency (bs=1): p50={gpu_latency['p50']:.2f}ms | p95={gpu_latency['p95']:.2f}ms | p99={gpu_latency['p99']:.2f}ms", flush=True)

    # Comparison summary table
    print("\n" + "=" * 115, flush=True)
    print("UNFROZEN VARIANTS COMPARATIVE SUMMARY (END-TO-END FINE-TUNED)")
    print("=" * 115, flush=True)
    header = f"{'Architecture':28s} | {'Tool Acc':9s} | {'Span F1':9s} | {'Joint EM':9s} | {'FP32 MB':8s} | {'INT8 MB':8s} | {'CPU p50':8s} | {'GPU p50':8s}"
    print(header, flush=True)
    print("-" * 115, flush=True)

    for name, r in results.items():
        print(
            f"{name:28s} | {r['tool_subset_acc']*100:7.2f}% | {r['span_f1']*100:7.2f}% | {r['joint_em']*100:7.2f}% | "
            f"{r['fp32_size_mb']:6.2f}MB | {r['int8_size_mb']:6.2f}MB | {r['cpu_latency_ms']['p50']:6.2f}ms | {r['gpu_latency_ms']['p50']:6.2f}ms",
            flush=True
        )
    print("=" * 115, flush=True)

    out_file = Path("ARN/results/stage2c_unfrozen_variants.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved unfrozen variants results to {out_file}", flush=True)


if __name__ == "__main__":
    run_unfrozen_evaluation()
