# ARN: Aether Routing Network

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23.2-brightgreen)](https://onnxruntime.ai/)

**ARN (Aether Routing Network)** is an ultra-compact, non-generative neural tool orchestrator designed for deterministic, real-time, on-device edge routing. Rather than deploying multi-billion parameter autoregressive LLMs for tool selection and argument parsing, ARN uses a frozen 4.38M bidirectional Transformer trunk (BERT-tiny) paired with:

1. **Factored Attention Routing (FAR):** Cross-attention queries for closed $K$-tool selection ($K=18$).
2. **Constrained Linear-Chain CRF:** Token emission projections coupled with hard BIO transition masking for guaranteed valid argument span extraction.
3. **Single Forward Pass:** Executes multi-label tool routing and argument parsing simultaneously in a single forward pass without autoregressive generation.

---

## Architecture Overview

```
                      User Prompt x (Tokens x_1 ... x_T)
                                     │
                        ┌────────────▼────────────┐
                        │   Frozen BERT-tiny      │ (4.38M parameters, d = 128)
                        └────────────┬────────────┘
                                     │ Hidden states H in R^{T x d}
                   ┌─────────────────┴─────────────────┐
                   │                                   │
        ┌──────────▼──────────┐             ┌──────────▼──────────┐
        │  FAR Cross-Attention│             │  Linear Slot Tagger │
        │  Query Embeddings Q │             │  Emissions E in     │
        │  in R^{18 x 128}    │             │  R^{T x 26}         │
        └──────────┬──────────┘             └──────────┬──────────┘
                   │ Context vectors                   │
        ┌──────────▼──────────┐             ┌──────────▼──────────┐
        │  Sigmoid Classifier │             │  Linear-Chain CRF   │
        │  (Multi-Label BCEL) │             │  w/ Hard BIO Masking│
        └──────────┬──────────┘             └──────────┬──────────┘
                   │                                   │
      Predicted Active Tools                Viterbi Decoded BIO Spans
   (e.g., send_whatsapp_message)        (contact_name, message_body)
```

For mathematical formulations, dynamic programming recurrences, and loss normalization, see [`docs/MATH.md`](docs/MATH.md).

---

## Installation

```bash
git clone https://github.com/Anishp-cell/ARN.git
cd ARN
pip install -e .
```

Verify installation:
```bash
pytest -v tests/
```

---

## Quickstart & CLI

```bash
# Run real-time tool orchestration via CLI
arn-cli "send a whatsapp message to John saying let's meet at 5pm"

# Inspect active schema and tool registry
arn-cli --list-tools
```

Sample output:
```json
{
  "active_tools": ["send_whatsapp_message"],
  "arguments": {
    "send_whatsapp_message": {
      "contact_name": "John",
      "message": "let's meet at 5pm"
    }
  },
  "tool_confidences": {
    "send_whatsapp_message": "TODO(verify: to be populated by retrained checkpoint)"
  },
  "latency_ms": "TODO(verify: measured on host CPU under benchmark protocol)",
  "fallback_to_llm": false
}
```

---

## Experimental Reproducibility Pipeline

All empirical results in the revision are 100% script-generated:

### 1. Group-Disjoint Dataset Splits
```bash
python scripts/make_splits.py --input data/raw/aether_orchestrator_dataset.jsonl --output-dir data/splits
```

### 2. Model Training (Ours & Baselines)
```bash
# Train ARN
python scripts/train.py --model-type arn --output-dir results/checkpoints

# Train Baselines
python scripts/train.py --model-type jointbert --output-dir results/checkpoints
python scripts/train.py --model-type cls_crf --output-dir results/checkpoints
python scripts/train.py --model-type far_no_crf --output-dir results/checkpoints
```

### 3. Model Evaluation
```bash
python scripts/evaluate.py --checkpoint results/checkpoints/arn/best_model.pt --data data/splits/group_disjoint_test.jsonl --output-json results/arn_eval.json
```

### 4. ONNX Export & INT8 Quantization
```bash
python scripts/export_onnx.py --checkpoint results/checkpoints/arn/best_model.pt --output-dir results/checkpoints
```

### 5. Hardware Latency & Memory Footprint Protocol
Benchmarking is conducted under a strictly controlled hardware protocol on host **Intel Core i7-13700HX (Windows 11, ONNX Runtime 1.23.2, 1 thread)**:
```bash
# CPU Latency (p50, p95, mean)
python scripts/benchmark_latency.py --model results/checkpoints/arn_model.onnx --threads 1 --runs 500 --output-json results/latency_fp32.json
python scripts/benchmark_latency.py --model results/checkpoints/arn_model_int8.onnx --threads 1 --runs 500 --output-json results/latency_int8.json

# Memory RSS Footprint (Peak & Incremental RSS via psutil)
python scripts/measure_memory.py --model results/checkpoints/arn_model.onnx --output-json results/memory_fp32.json
python scripts/measure_memory.py --model results/checkpoints/arn_model_int8.onnx --output-json results/memory_int8.json
```

### 6. Automated LaTeX Table Generation
```bash
python scripts/make_tables.py --results-dir results --output-dir docs/paper/tables
```

---

## Peer Review Audit & Historical Artifacts

In response to peer review feedback:
- The complete forensic audit is documented in [`AUDIT.md`](AUDIT.md).
- Untraceable claims (e.g. historical 5.55 MB RAM and 14.2 ms Raspberry Pi figures, unverified LoRA assertions, and synthetic percentages) are withdrawn.
- All historical checkpoints and pre-refactor scripts are preserved in [`_archive/`](_archive/).
- Third-party licenses and attribution are detailed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

## Standalone Repository Extraction

To publish ARN as a clean, independent standalone repository without historical parent repo commits:

```bash
# 1. From the repository root, create an orphan split branch of ARN/
git subtree split --prefix=ARN -b arn-standalone-clean

# 2. Clone into a clean new directory
git clone --branch arn-standalone-clean . ../ARN-standalone

# 3. Add fresh remote origin and push
cd ../ARN-standalone
git remote remove origin
git remote add origin https://github.com/Anishp-cell/ARN.git
git push -u origin main
```

---

## License & Citation

Licensed under the [MIT License](LICENSE).

```bibtex
@article{pattnaik2026arn,
  title={ARN: A Sub-18MB Non-Generative Joint Encoder-CRF Model for Real-Time On-Device Tool Orchestration},
  author={Pattnaik, Anish},
  journal={arXiv preprint},
  year={2026}
}
```
