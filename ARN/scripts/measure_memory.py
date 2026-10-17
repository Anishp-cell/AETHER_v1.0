#!/usr/bin/env python3
"""
Standardized Memory Footprint Benchmark for ARN via psutil.
Measures:
1. Baseline Process RSS (clean Python process with runtime loaded)
2. Peak Process RSS (after loading model and executing inference)
3. Incremental RSS (Peak RSS - Baseline RSS)
Directly addresses peer review feedback on ungrounded memory claims.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.measure_memory")


WORKER_CODE = """
import sys
import os
import json
import psutil
import numpy as np

backend = sys.argv[1]
model_path = sys.argv[2]
seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 32

proc = psutil.Process(os.getpid())

# 1. Measure baseline RSS after importing runtime libraries
if backend == "onnx":
    import onnxruntime as ort
elif backend == "torch":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch
    from arn import AetherRoutingNetwork, ARNConfig

baseline_rss = proc.memory_info().rss

# 2. Load model
if backend == "onnx":
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    inputs = {
        "input_ids": np.ones((1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64)
    }
    # Warmup and active inference
    for _ in range(10):
        session.run(None, inputs)
elif backend == "torch":
    ckpt = torch.load(model_path, map_location="cpu")
    config = ARNConfig()
    model = AetherRoutingNetwork(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    inp = torch.ones((1, seq_len), dtype=torch.long)
    mask = torch.ones((1, seq_len), dtype=torch.long)
    with torch.no_grad():
        for _ in range(10):
            model(inp, mask)

peak_rss = proc.memory_info().rss
incremental_rss = peak_rss - baseline_rss

result = {
    "backend": backend,
    "model_path": model_path,
    "baseline_rss_bytes": baseline_rss,
    "peak_rss_bytes": peak_rss,
    "incremental_rss_bytes": incremental_rss,
    "baseline_rss_mb": round(baseline_rss / (1024 * 1024), 2),
    "peak_rss_mb": round(peak_rss / (1024 * 1024), 2),
    "incremental_rss_mb": round(incremental_rss / (1024 * 1024), 2),
}

print("__RESULT_START__" + json.dumps(result) + "__RESULT_END__")
"""


def measure_isolated_memory(backend: str, model_path: Path, seq_len: int = 32) -> Dict[str, Any]:
    """Execute memory measurement in a clean isolated subprocess."""
    cmd = [
        sys.executable,
        "-c",
        WORKER_CODE,
        backend,
        str(model_path.resolve()),
        str(seq_len),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    stdout = proc.stdout
    start_tag = "__RESULT_START__"
    end_tag = "__RESULT_END__"

    if start_tag not in stdout or end_tag not in stdout:
        raise RuntimeError(f"Subprocess failed to output memory metrics. Output:\n{stdout}\nErrors:\n{proc.stderr}")

    raw_json = stdout.split(start_tag)[1].split(end_tag)[0]
    return json.loads(raw_json)


def main():
    parser = argparse.ArgumentParser(description="ARN Standardized Memory Measurement via psutil")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx or .pt model")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for inference")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return

    backend = "onnx" if model_path.suffix.lower() == ".onnx" else "torch"
    logger.info("Measuring memory for %s using isolated subprocess (%s backend)...", model_path.name, backend)

    metrics = measure_isolated_memory(backend, model_path, args.seq_len)
    metrics["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)

    logger.info("=== Memory Footprint Results ===")
    logger.info("Model File:      %s (%.2f MB on disk)", model_path.name, metrics["file_size_mb"])
    logger.info("Baseline RSS:    %.2f MB (clean runtime environment)", metrics["baseline_rss_mb"])
    logger.info("Peak Process RSS:%.2f MB (peak physical memory)", metrics["peak_rss_mb"])
    logger.info("Incremental RSS: %.2f MB (model & inference overhead)", metrics["incremental_rss_mb"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved report to %s", out_path)


if __name__ == "__main__":
    main()
