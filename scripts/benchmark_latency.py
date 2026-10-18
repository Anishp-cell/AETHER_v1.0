#!/usr/bin/env python3
"""
Standardized CPU Latency Benchmark for ARN.
Measures high-precision latency (p50, p95, p99, mean, std) under a strictly controlled
hardware protocol on laptop-class x86 CPU (Intel i7-13700HX, Windows 11).
Supports ONNX Runtime (FP32 & INT8) and PyTorch CPU with fixed thread counts.
"""

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.benchmark_latency")


def get_hardware_info() -> Dict[str, Any]:
    """Retrieve verified host metadata."""
    return {
        "device_category": "laptop-class x86 CPU",
        "cpu_processor": platform.processor(),
        "cpu_name": "Intel Core i7-13700HX",
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }


def benchmark_onnx(
    model_path: Path,
    num_runs: int = 500,
    warmup: int = 50,
    threads: int = 1,
    seq_len: int = 32,
) -> Dict[str, Any]:
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])

    input_ids = np.ones((1, seq_len), dtype=np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Warmup
    for _ in range(warmup):
        session.run(None, inputs)

    # Timed runs
    durations_ms: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    durations = np.array(durations_ms)
    return {
        "backend": "onnxruntime",
        "ort_version": ort.__version__,
        "threads": threads,
        "runs": num_runs,
        "warmup": warmup,
        "seq_len": seq_len,
        "mean_ms": float(np.mean(durations)),
        "std_ms": float(np.std(durations)),
        "p50_ms": float(np.percentile(durations, 50)),
        "p90_ms": float(np.percentile(durations, 90)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
        "min_ms": float(np.min(durations)),
        "max_ms": float(np.max(durations)),
        "throughput_qps": float(1000.0 / np.mean(durations)),
    }


def benchmark_pytorch(
    model_path: Path,
    num_runs: int = 500,
    warmup: int = 50,
    threads: int = 1,
    seq_len: int = 32,
) -> Dict[str, Any]:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch
    from arn import AetherRoutingNetwork, ARNConfig

    torch.set_num_threads(threads)
    ckpt = torch.load(model_path, map_location="cpu")
    config_dict = ckpt.get("config", {})
    config = ARNConfig(**{k: v for k, v in config_dict.items() if hasattr(ARNConfig, k)}) if config_dict else ARNConfig()

    model = AetherRoutingNetwork(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_ids = torch.ones((1, seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids, attention_mask)

        # Timed runs
        durations_ms: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model(input_ids, attention_mask)
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)

    durations = np.array(durations_ms)
    return {
        "backend": "pytorch",
        "pytorch_version": torch.__version__,
        "threads": threads,
        "runs": num_runs,
        "warmup": warmup,
        "seq_len": seq_len,
        "mean_ms": float(np.mean(durations)),
        "std_ms": float(np.std(durations)),
        "p50_ms": float(np.percentile(durations, 50)),
        "p90_ms": float(np.percentile(durations, 90)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
        "min_ms": float(np.min(durations)),
        "max_ms": float(np.max(durations)),
        "throughput_qps": float(1000.0 / np.mean(durations)),
    }


def main():
    parser = argparse.ArgumentParser(description="ARN Standardized CPU Latency Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx or .pt checkpoint")
    parser.add_argument("--runs", type=int, default=500, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--threads", type=int, default=1, help="Fixed intra-op CPU threads")
    parser.add_argument("--seq-len", type=int, default=32, help="Input sequence length")
    parser.add_argument("--output-json", type=str, default=None, help="Output path for JSON report")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return

    hw_info = get_hardware_info()
    logger.info("Host Target: %s (%s, %s)", hw_info["device_category"], hw_info["cpu_name"], hw_info["os"])
    logger.info("Benchmarking %s with %d threads, seq_len=%d...", model_path.name, args.threads, args.seq_len)

    if model_path.suffix.lower() == ".onnx":
        bench_metrics = benchmark_onnx(model_path, args.runs, args.warmup, args.threads, args.seq_len)
    elif model_path.suffix.lower() == ".pt":
        bench_metrics = benchmark_pytorch(model_path, args.runs, args.warmup, args.threads, args.seq_len)
    else:
        logger.error("Unsupported file extension: %s", model_path.suffix)
        return

    result = {
        "hardware": hw_info,
        "benchmark": bench_metrics,
        "model_file": str(model_path),
        "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
    }

    logger.info("=== Latency Benchmark Results ===")
    logger.info("Model:        %s (%.2f MB)", model_path.name, result["file_size_mb"])
    logger.info("Threads:      %d", bench_metrics["threads"])
    logger.info("Mean Latency: %.2f ms (+/- %.2f)", bench_metrics["mean_ms"], bench_metrics["std_ms"])
    logger.info("Median (p50): %.2f ms", bench_metrics["p50_ms"])
    logger.info("95th pct p95: %.2f ms", bench_metrics["p95_ms"])
    logger.info("99th pct p99: %.2f ms", bench_metrics["p99_ms"])
    logger.info("Throughput:   %.1f qps", bench_metrics["throughput_qps"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved report to %s", out_path)


if __name__ == "__main__":
    main()
