"""
ARN Phase 8 Deployment Benchmark Runner.

Standardized Hardware: laptop-class x86 CPU (13th Gen Intel(R) Core(TM) i7-13700HX).

Benchmarks:
  1. Latency (100 warmup runs, 1000 benchmark iterations):
     - PyTorch FP32 CPU (1 thread & 4 threads)
     - ONNX Runtime FP32 (1 thread & 4 threads)
     - ONNX Runtime Dynamic INT8 (1 thread & 4 threads)
     - Full End-to-End Pipeline: Tokenizer + ONNX INT8 + Fast NumPy Viterbi
  2. Subprocess-isolated Peak RSS Memory (MB) using psutil.

Outputs:
  - results/deployment_benchmark.json
  - Appends benchmark run to results/ledger.json
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Any

import numpy as np
import onnxruntime as ort
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arn.baselines import CLSCRFModel
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, TOOL_TO_IDX
from arn.viterbi_fast import viterbi_decode_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.deployment_benchmark")


def benchmark_latency(inference_fn, warmup: int = 100, iterations: int = 1000) -> Dict[str, float]:
    """Benchmark inference latency over repeated iterations."""
    for _ in range(warmup):
        inference_fn()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        inference_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def measure_memory_subprocess(script_code: str) -> float:
    """Measure peak RSS memory (MB) in an isolated subprocess."""
    cmd = [sys.executable, "-c", script_code]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = psutil.Process(p.pid)
    peak_rss = 0
    try:
        while p.poll() is None:
            rss = proc.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
            time.sleep(0.005)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    p.communicate()
    return float(peak_rss / (1024 * 1024))


def run_deployment_benchmarks():
    repo_dir = Path(__file__).resolve().parent.parent
    checkpoint_dir = repo_dir / "checkpoints" / "arn_final_model"

    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.pt"
    onnx_fp32_path = checkpoint_dir / "model.onnx"
    onnx_int8_path = checkpoint_dir / "model_int8.onnx"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    test_utterance = "Set a timer for 15 minutes to take out the laundry"
    encoded = TOKENIZER(test_utterance, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
    input_ids_pt = encoded["input_ids"]
    mask_pt = encoded["attention_mask"]
    input_ids_np = input_ids_pt.numpy()
    mask_np = mask_pt.numpy()

    # Load CRF parameters for numpy viterbi
    logger.info("Loading PyTorch model for CRF transition weights...")
    pt_model = CLSCRFModel(num_tools=config["num_tools"], num_tags=config["num_tags"])
    pt_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    pt_model.eval()

    trans_torch, start_torch, end_torch = pt_model.crf._get_effective_transitions()
    trans_np = trans_torch.detach().cpu().numpy()
    start_np = start_torch.detach().cpu().numpy()
    end_np = end_torch.detach().cpu().numpy()

    results: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": "laptop-class x86 CPU (13th Gen Intel(R) Core(TM) i7-13700HX)",
        "warmup_runs": 100,
        "benchmark_iterations": 1000,
        "input_seq_len": 64,
        "latency": {},
        "memory_peak_rss_mb": {},
        "model_sizes_mb": {
            "pytorch_fp32_weights_mb": float(weights_path.stat().st_size / (1024 * 1024)),
            "onnx_fp32_mb": float(onnx_fp32_path.stat().st_size / (1024 * 1024)),
            "onnx_int8_mb": float(onnx_int8_path.stat().st_size / (1024 * 1024)),
        }
    }

    for threads in [1, 4]:
        logger.info(f"\n--- Benchmarking Latency with {threads} Thread(s) ---")
        torch.set_num_threads(threads)

        # PyTorch CPU inference (encoder + heads + crf)
        def run_pt():
            with torch.no_grad():
                t_logits, emissions = pt_model(input_ids_pt, mask_pt)
                pt_model.crf.viterbi_decode(emissions, mask_pt)

        pt_lat = benchmark_latency(run_pt)
        logger.info(f"PyTorch CPU ({threads}T): mean={pt_lat['mean_ms']:.2f}ms, p50={pt_lat['p50_ms']:.2f}ms, p95={pt_lat['p95_ms']:.2f}ms")
        results["latency"][f"pytorch_cpu_{threads}t"] = pt_lat

        # ONNX Runtime FP32
        so_fp32 = ort.SessionOptions()
        so_fp32.intra_op_num_threads = threads
        session_fp32 = ort.InferenceSession(str(onnx_fp32_path), sess_options=so_fp32, providers=["CPUExecutionProvider"])
        ort_inputs = {"input_ids": input_ids_np, "attention_mask": mask_np}

        def run_onnx_fp32():
            session_fp32.run(None, ort_inputs)

        onnx_fp32_lat = benchmark_latency(run_onnx_fp32)
        logger.info(f"ONNX FP32 ({threads}T): mean={onnx_fp32_lat['mean_ms']:.2f}ms, p50={onnx_fp32_lat['p50_ms']:.2f}ms, p95={onnx_fp32_lat['p95_ms']:.2f}ms")
        results["latency"][f"onnx_fp32_{threads}t"] = onnx_fp32_lat

        # ONNX Runtime INT8
        so_int8 = ort.SessionOptions()
        so_int8.intra_op_num_threads = threads
        session_int8 = ort.InferenceSession(str(onnx_int8_path), sess_options=so_int8, providers=["CPUExecutionProvider"])

        def run_onnx_int8():
            session_int8.run(None, ort_inputs)

        onnx_int8_lat = benchmark_latency(run_onnx_int8)
        logger.info(f"ONNX INT8 ({threads}T): mean={onnx_int8_lat['mean_ms']:.2f}ms, p50={onnx_int8_lat['p50_ms']:.2f}ms, p95={onnx_int8_lat['p95_ms']:.2f}ms")
        results["latency"][f"onnx_int8_{threads}t"] = onnx_int8_lat

        # End-to-End Pipeline: Tokenizer + ONNX INT8 + Vectorized NumPy Viterbi
        def run_e2e():
            enc = TOKENIZER(test_utterance, max_length=64, padding="max_length", truncation=True)
            inp = {"input_ids": np.array([enc["input_ids"]], dtype=np.int64), "attention_mask": np.array([enc["attention_mask"]], dtype=np.int64)}
            t_logits, em = session_int8.run(None, inp)
            viterbi_decode_numpy(em, inp["attention_mask"], trans_np, start_np, end_np)

        e2e_lat = benchmark_latency(run_e2e)
        logger.info(f"End-to-End Pipeline ({threads}T): mean={e2e_lat['mean_ms']:.2f}ms, p50={e2e_lat['p50_ms']:.2f}ms, p95={e2e_lat['p95_ms']:.2f}ms")
        results["latency"][f"end_to_end_pipeline_{threads}t"] = e2e_lat

    # 2. Subprocess-isolated Memory Benchmarking
    logger.info("\n--- Measuring Subprocess-Isolated Memory RSS ---")
    mem_script_pt = f"""
import torch, time
from pathlib import Path
from arn.baselines import CLSCRFModel
m = CLSCRFModel(num_tools=18, num_tags=37)
m.load_state_dict(torch.load(r'{weights_path}', map_location='cpu'))
m.eval()
ids = torch.randint(10, 100, (1, 64))
mask = torch.ones((1, 64), dtype=torch.long)
for _ in range(50):
    t, em = m(ids, mask)
    m.crf.viterbi_decode(em, mask)
"""
    rss_pt = measure_memory_subprocess(mem_script_pt)
    results["memory_peak_rss_mb"]["pytorch_cpu_process_mb"] = rss_pt
    logger.info(f"PyTorch CPU Peak RSS: {rss_pt:.2f} MB")

    mem_script_onnx = f"""
import onnxruntime as ort, numpy as np, time
from arn.viterbi_fast import viterbi_decode_numpy
s = ort.InferenceSession(r'{onnx_int8_path}', providers=['CPUExecutionProvider'])
inp = {{'input_ids': np.ones((1, 64), dtype=np.int64), 'attention_mask': np.ones((1, 64), dtype=np.int64)}}
for _ in range(50):
    t, em = s.run(None, inp)
"""
    rss_onnx = measure_memory_subprocess(mem_script_onnx)
    results["memory_peak_rss_mb"]["onnx_int8_process_mb"] = rss_onnx
    logger.info(f"ONNX INT8 Peak RSS: {rss_onnx:.2f} MB")

    # Save results
    out_path = repo_dir / "results" / "deployment_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved deployment benchmark results to {out_path}")

    # Append to ledger.json
    ledger_path = repo_dir / "results" / "ledger.json"
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        ledger["runs"].append({
            "experiment_id": "p8_deployment_benchmark",
            "protocol": "phase8_deployment",
            "metrics": results,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

    logger.info("Phase 8 Deployment Benchmark Complete!")


if __name__ == "__main__":
    run_deployment_benchmarks()
