"""
Stage 2d Item 7: ONNX Deployment Protocol.

Exports and benchmarks:
  1. Unfrozen CLS+CRF (FP32 & INT8)
  2. Unfrozen ARN (FP32 & INT8)

Measures:
  - Exact file sizes in MB (10^6 bytes) and MiB (2^20 bytes).
  - INT8 validation accuracy (quantization loss audit).
  - Latency: ONNX Runtime 1.23.2, 1 thread vs multi-thread (4 threads).
  - Breakdown: Model Forward Only vs End-to-End (Tokenization + Forward + CRF Viterbi).
  - RSS memory: Baseline, Peak, Incremental.
  - Explains the 15.60 MB (PyTorch eager) vs 4.54 MB (ONNX INT8) disparity.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import psutil
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.baselines import CLSCRFModel
from arn.model import AetherRoutingNetwork
from arn.tokenizer import TOKENIZER, IDX_TO_TAG
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2c_head_independence import partition_clusters, extract_spans_from_bio
from scripts.stage2c_softmax_baselines import extract_spans_strict, compute_span_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.stage2d_onnx")


class ONNXExportWrapper(nn.Module):
    """Clean wrapper exposing forward(input_ids, attention_mask) returning (tool_logits, slot_emissions)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = model.encoder if hasattr(model, "encoder") else model.bert
        self.tool_classifier = getattr(model, "tool_classifier", None)
        self.far = getattr(model, "far", None)
        self.slot_tagger = getattr(model, "slot_tagger", None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder.embeddings(input_ids=input_ids)
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        enc_out = self.encoder.encoder(emb, attention_mask=ext_mask)
        h = enc_out[0]
        if self.tool_classifier is not None:
            tool_logits = self.tool_classifier(h[:, 0, :])
        else:
            tool_logits, _ = self.far(h, attention_mask)
        slot_emissions = self.slot_tagger(h)
        return tool_logits, slot_emissions


def measure_memory_and_latency(
    ort_session: ort.InferenceSession,
    crf_module: Any,
    test_texts: List[str],
    n_warmup: int = 100,
    n_runs: int = 500,
) -> Dict[str, Any]:
    """Measures latency (forward only vs end-to-end) and RSS memory."""
    process = psutil.Process()
    baseline_rss = process.memory_info().rss

    # Prepare sample inputs
    sample_text = test_texts[0]
    enc = TOKENIZER(sample_text, max_length=64, padding="max_length", truncation=True, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # 1. Warmup
    for _ in range(n_warmup):
        _ = ort_session.run(None, ort_inputs)

    # 2. Benchmark Forward Only
    fwd_times = []
    peak_rss = baseline_rss
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        t1 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000.0)

        current_rss = process.memory_info().rss
        if current_rss > peak_rss:
            peak_rss = current_rss

    # 3. Benchmark End-to-End (Tokenization + Forward + CRF Viterbi)
    e2e_times = []
    for i in range(n_runs):
        text = test_texts[i % len(test_texts)]
        t0 = time.perf_counter()

        # Tokenization
        tokens = TOKENIZER(text, max_length=64, padding="max_length", truncation=True, return_tensors="np")
        in_ids = tokens["input_ids"].astype(np.int64)
        att_m = tokens["attention_mask"].astype(np.int64)

        # ONNX Run
        tool_logits, slot_emissions = ort_session.run(None, {"input_ids": in_ids, "attention_mask": att_m})

        # CRF Decoding
        if crf_module is not None:
            em_tensor = torch.from_numpy(slot_emissions)
            m_tensor = torch.from_numpy(att_m)
            _ = crf_module.viterbi_decode(em_tensor, m_tensor)
        else:
            _ = np.argmax(slot_emissions, axis=-1)

        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000.0)

        current_rss = process.memory_info().rss
        if current_rss > peak_rss:
            peak_rss = current_rss

    return {
        "forward_only_ms": {
            "p50": float(np.percentile(fwd_times, 50)),
            "p95": float(np.percentile(fwd_times, 95)),
            "p99": float(np.percentile(fwd_times, 99)),
            "mean": float(np.mean(fwd_times)),
        },
        "end_to_end_ms": {
            "p50": float(np.percentile(e2e_times, 50)),
            "p95": float(np.percentile(e2e_times, 95)),
            "p99": float(np.percentile(e2e_times, 99)),
            "mean": float(np.mean(e2e_times)),
        },
        "rss_memory_mb": {
            "baseline": baseline_rss / (1024 * 1024),
            "peak": peak_rss / (1024 * 1024),
            "incremental": (peak_rss - baseline_rss) / (1024 * 1024),
        }
    }


def evaluate_onnx_accuracy(
    ort_session: ort.InferenceSession,
    crf_module: Any,
    dataset: ARNDataset,
    val_indices: List[int],
    gold_tools: np.ndarray,
    tau: float = 0.50,
) -> Dict[str, float]:
    """Evaluate INT8 ONNX model on validation samples."""
    val_data = [dataset.encoded[i] for i in val_indices]
    all_tool_probs = []
    all_pred_spans = []
    gold_spans = []

    for item in val_data:
        in_ids = item["input_ids"].unsqueeze(0).numpy().astype(np.int64)
        att_m = item["attention_mask"].unsqueeze(0).numpy().astype(np.int64)
        t_logits, s_emissions = ort_session.run(None, {"input_ids": in_ids, "attention_mask": att_m})

        probs = 1.0 / (1.0 + np.exp(-t_logits[0]))
        all_tool_probs.append(probs)

        # CRF decoding
        l_i = int(att_m[0].sum())
        em_tensor = torch.from_numpy(s_emissions)
        m_tensor = torch.from_numpy(att_m)
        pred_tags = crf_module.viterbi_decode(em_tensor, m_tensor)[0]
        p_seq = [IDX_TO_TAG.get(idx, "O") for idx in pred_tags[:l_i]]
        all_pred_spans.append(extract_spans_strict(p_seq))

        g_seq = [IDX_TO_TAG.get(idx, "O") for idx in item["tags"][:l_i].tolist()]
        gold_spans.append(extract_spans_strict(g_seq))

    all_tool_probs = np.vstack(all_tool_probs)
    val_gold = gold_tools[val_indices]
    pred_tools = (all_tool_probs >= tau).astype(int)

    tool_matches = np.all(pred_tools == val_gold, axis=1)
    tool_acc = float(np.mean(tool_matches))

    span_metrics = compute_span_metrics(gold_spans, all_pred_spans)
    slot_matches = [g == p for g, p in zip(gold_spans, all_pred_spans)]
    joint_em = float(np.mean([t and s for t, s in zip(tool_matches, slot_matches)]))

    return {
        "tool_subset_acc": tool_acc,
        "span_f1": span_metrics["strict_f1"],
        "joint_em": joint_em,
    }


def run_stage2d_onnx():
    device = torch.device("cpu")
    print("=== STAGE 2d ONNX DEPLOYMENT & BENCHMARKING SUITE ===", flush=True)

    export_dir = Path("ARN/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)
    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
    test_texts = [s["messages"][0]["content"].strip() for s in raw_samples]
    gold_tools = torch.stack([b["tool_labels"] for b in dataset.encoded]).numpy()

    # Fold 1 for validation comparison
    folds = partition_clusters(cluster_ids, n_folds=5, seed=42)
    val_idx = folds[0]

    models_to_export = [
        ("unfrozen_cls_crf", CLSCRFModel(ARNConfig(freeze_trunk=False), load_pretrained=True)),
        ("unfrozen_arn", AetherRoutingNetwork(ARNConfig(freeze_trunk=False), load_pretrained=True)),
    ]

    dummy_input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 64), dtype=torch.long)

    deployment_results = {}

    for model_name, raw_model in models_to_export:
        print(f"\n=======================================================", flush=True)
        print(f"PROCESSING MODEL: {model_name.upper()}", flush=True)
        print(f"=======================================================", flush=True)

        raw_model.eval()
        wrapper = ONNXExportWrapper(raw_model)

        fp32_path = export_dir / f"{model_name}_fp32.onnx"
        int8_path = export_dir / f"{model_name}_int8.onnx"

        # 1. Export FP32 ONNX
        print(f"Exporting FP32 ONNX to {fp32_path}...", flush=True)
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["tool_logits", "slot_emissions"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "tool_logits": {0: "batch_size"},
                "slot_emissions": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )

        fp32_bytes = os.path.getsize(fp32_path)
        fp32_mb = fp32_bytes / 1e6
        fp32_mib = fp32_bytes / (1024 * 1024)
        print(f"  FP32 Size: {fp32_mb:.2f} MB ({fp32_mib:.2f} MiB)", flush=True)

        # 2. Dynamic Quantization to INT8 using ONNX Runtime
        print(f"Quantizing to INT8 via onnxruntime.quantization.quantize_dynamic...", flush=True)
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        int8_bytes = os.path.getsize(int8_path)
        int8_mb = int8_bytes / 1e6
        int8_mib = int8_bytes / (1024 * 1024)
        compression = fp32_bytes / int8_bytes
        print(f"  INT8 Size: {int8_mb:.2f} MB ({int8_mib:.2f} MiB) | Compression: {compression:.2f}x", flush=True)

        # 3. Create ORT Sessions (1 thread & 4 threads)
        opts_1thread = ort.SessionOptions()
        opts_1thread.intra_op_num_threads = 1
        opts_1thread.inter_op_num_threads = 1
        opts_1thread.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_1thread = ort.InferenceSession(str(int8_path), sess_options=opts_1thread, providers=["CPUExecutionProvider"])

        opts_4thread = ort.SessionOptions()
        opts_4thread.intra_op_num_threads = 4
        opts_4thread.inter_op_num_threads = 1
        opts_4thread.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_4thread = ort.InferenceSession(str(int8_path), sess_options=opts_4thread, providers=["CPUExecutionProvider"])

        crf_module = raw_model.crf if hasattr(raw_model, "crf") else None

        # 4. Measure Accuracy of INT8 ONNX
        print("Evaluating INT8 ONNX Accuracy on validation fold...", flush=True)
        int8_acc = evaluate_onnx_accuracy(session_1thread, crf_module, dataset, val_idx, gold_tools, tau=0.50)
        print(f"  INT8 Accuracy: Tool Acc={int8_acc['tool_subset_acc']*100:.2f}%, Span F1={int8_acc['span_f1']*100:.2f}%, Joint EM={int8_acc['joint_em']*100:.2f}%", flush=True)

        # 5. Latency & Memory (1 Thread)
        print("Benchmarking Latency & Memory (1 Thread)...", flush=True)
        bench_1thread = measure_memory_and_latency(session_1thread, crf_module, test_texts)
        print(f"  1 Thread: Forward p50={bench_1thread['forward_only_ms']['p50']:.2f}ms (p95={bench_1thread['forward_only_ms']['p95']:.2f}ms) | End-to-End p50={bench_1thread['end_to_end_ms']['p50']:.2f}ms", flush=True)

        # 6. Latency (4 Threads)
        print("Benchmarking Latency (4 Threads)...", flush=True)
        bench_4thread = measure_memory_and_latency(session_4thread, crf_module, test_texts)
        print(f"  4 Threads: Forward p50={bench_4thread['forward_only_ms']['p50']:.2f}ms (p95={bench_4thread['forward_only_ms']['p95']:.2f}ms) | End-to-End p50={bench_4thread['end_to_end_ms']['p50']:.2f}ms", flush=True)

        deployment_results[model_name] = {
            "onnx_fp32": {
                "size_mb": fp32_mb,
                "size_mib": fp32_mib,
                "size_bytes": fp32_bytes,
            },
            "onnx_int8": {
                "size_mb": int8_mb,
                "size_mib": int8_mib,
                "size_bytes": int8_bytes,
                "compression_ratio": compression,
            },
            "int8_validation_accuracy": int8_acc,
            "latency_1thread": bench_1thread,
            "latency_4threads": bench_4thread,
        }

    # Technical Explanation of 15.60 MB vs 4.54 MB
    explanation = (
        "In Stage 2c, PyTorch eager dynamic quantization (torch.ao.quantization.quantize_dynamic) was used targeting "
        "only {nn.Linear} modules. In BERT-tiny, the token embeddings table (BertEmbeddings: 30,522 vocab x 128 hidden) "
        "contains 3,906,816 parameters (15.62 MB in FP32), which PyTorch eager quantization leaves in float32. "
        "In contrast, ONNX Runtime dynamic quantization (onnxruntime.quantization.quantize_dynamic) quantizes both "
        "linear projection weights and embedded tensors down to 8-bit integers, compressing the entire model to ~4.54 MB."
    )

    output = {
        "onnx_runtime_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "size_disparity_explanation": explanation,
        "models": deployment_results,
    }

    out_file = Path("ARN/results/stage2d_onnx_deployment.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved complete ONNX deployment results to {out_file}", flush=True)


if __name__ == "__main__":
    run_stage2d_onnx()
