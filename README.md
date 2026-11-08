# ARN: Aether Routing Network (v1.0.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23.2-brightgreen)](https://onnxruntime.ai/)

**ARN (Aether Routing Network)** is an ultra-compact, non-generative neural tool orchestrator and argument extractor designed for real-time edge dispatch. Rather than querying multi-billion parameter autoregressive LLMs for standard on-device commands, ARN formulates tool selection and argument parsing as a joint multi-label classification and linear-chain conditional random field (CRF) sequence tagging problem over a 4.39M parameter bidirectional Transformer trunk (`google/bert_uncased_L-2_H-128_A-2`).

In rigorous 15-fold cluster-stratified cross-validation, the fine-tuned joint encoder-CRF architecture achieves **16.91% Joint Exact Match** and **67.88% Lenient Slot F1**, with an ONNX INT8 runtime latency of **1.39 ms** (1 thread) / **0.75 ms** (4 threads) and a peak process memory of **58.69 MB RSS** (12.37 MB incremental over the bare runtime) on host CPU.

---

## Aether Routing Network v2 (branch `arn-v2-confirmatory`, Nov 2026)

v2 is a rebuilt extractive router: tool-qualified BIO tags (`B-send_email.subject`), registry-constrained joint decoding over (tool set, spans), a train-fitted rule-based datetime canonicalizer, and FAR-v2 (FAR plus per-tool bias and a [CLS] residual). All numbers below come from `results/xmodel/*.json` prediction files scored by one shared scorer (`src/arn/eval/mobile_actions_scorer.py`); see `docs/RESULTS_v2.md` for the full tables, confidence intervals and disclosures.

| Benchmark (test split) | Aether Routing Network v2, 4.4M params | Comparison (same harness unless noted) |
|---|---|---|
| Google Mobile Actions, strict exact match (961) | **89.32 ± 0.47** (3 seeds) | FunctionGemma-270M fine-tuned on the same split 85.12; LFM2.5-230M fine-tuned + tuned 92.51; FunctionGemma base 60.15; LFM2.5 base 68.05; Qwen3-0.6B base 65.04 |
| MixATIS overall accuracy (828), pre-registered | 40.94 ± 2.62 | AGIF 40.8, GL-GIN 43.5 (published; ~0.6-0.9M param BiLSTMs) |
| MixSNIPS overall accuracy (2,199), pre-registered | 70.29 ± 0.46 | AGIF 74.2, GL-GIN 75.4 (published) |

Important limitations: ~33 points of the Mobile Actions score depend on the rule-based datetime canonicalizer; the Mobile Actions eval split was inspected during decoder development (disclosed); the model only supports a closed, pre-trained tool registry and is not applicable to BFCL. The v1 "routing accuracy" leaderboard numbers in the legacy repository (93.0 / 87.3 / 98.3 / 94.2) have no primary source and are **withdrawn**.

---

## Key Characteristics & Verified Specifications

All metrics reported below trace directly to [`results/ledger.json`](results/ledger.json), [`results/master_protocol_summary.json`](results/master_protocol_summary.json), and [`results/deployment_benchmark.json`](results/deployment_benchmark.json):

| Metric / Dimension | Empirical Value (Master Protocol) | Hardware & Protocol Context |
| :--- | :---: | :--- |
| **Joint Exact Match** | **16.91% ± 6.56%** (95% CI: [13.51%, 20.59%]) | 15-fold tool-stratified cluster CV ($N=5,000$) |
| **Tool Subset Accuracy** | **39.91% ± 12.19%** (95% CI: [34.78%, 45.45%]) | Exact match across all $K=18$ candidate tools |
| **Strict Span F1** | **43.61% ± 10.43%** (95% CI: [39.02%, 48.06%]) | Exact span boundary and slot category match |
| **Lenient Span F1** | **67.88% ± 9.86%** (95% CI: [63.66%, 71.93%]) | Partial character overlap and slot category match |
| **ONNX INT8 Latency (1T)** | **1.39 ms** mean (p50: 1.25 ms, p95: 1.83 ms) | Intel Core i7-13700HX, ONNX Runtime 1.23.2 |
| **ONNX INT8 Latency (4T)** | **0.75 ms** mean (p50: 0.69 ms, p95: 1.09 ms) | Intel Core i7-13700HX, 4 execution threads |
| **End-to-End Pipeline Latency** | **3.10 ms** (1 thread) / **2.34 ms** (4 threads) | Tokenize + INT8 ONNX + NumPy Viterbi + Extract |
| **Memory Footprint** | **58.69 MB peak RSS** (12.37 MB incremental) | `results/deployment_benchmark.json`; the earlier "4.21 MB" figure was a measurement artefact and is withdrawn |
| **Model Size (PyTorch FP32)** | **17.58 MB** (`model.pt`) | 4,386,474 total parameters |
| **Model Size (ONNX FP32)** | **16.75 MB** (`model.onnx`) | Maximum absolute tensor difference $< 5 \times 10^{-6}$ |
| **Model Size (ONNX INT8)** | **4.28 MB** (`model_int8.onnx`) | Dynamic 8-bit quantization (3.92x compression) |

---

## Architectural Design

```
                       User Utterance (Tokens x_1 ... x_T)
                                       │
                         ┌─────────────▼─────────────┐
                         │   Fine-Tuned BERT-tiny    │ (4.39M parameters, d = 128)
                         └─────────────┬─────────────┘
                                       │ Hidden states H in R^{T x 128}
                    ┌──────────────────┴──────────────────┐
                    │                                     │
         ┌──────────▼──────────┐               ┌──────────▼──────────┐
         │  [CLS] Pooling Head │               │  Linear Slot Tagger │
         │  h_cls in R^{128}   │               │  Emissions in       │
         └──────────┬──────────┘               │  R^{T x 26}         │
                    │ Linear projection                └──────────┬──────────┘
         ┌──────────▼──────────┐                          │
         │  Sigmoid Classifier │               ┌──────────▼──────────┐
         │  Multi-Label Tools  │               │  Linear-Chain CRF   │
         │  (K = 18 APIs)      │               │  w/ Hard BIO Masking│
         └──────────┬──────────┘               └──────────┬──────────┘
                    │                                     │
         Active Tool Set                       Viterbi-Decoded Entity Spans
    (e.g., handle_smart_home)               (action="turn off", device="lights")
```

### Key Components:
1. **Backbone:** 2-layer, 128-hidden dimension BERT-tiny encoder fine-tuned jointly on tool classification and slot tagging.
2. **Tool Routing Head:** Sigmoid classification layer over pooled `[CLS]` token representation supporting multi-tool dispatch.
3. **Slot Extraction Head:** Linear emission layer projected to 26 BIO slot tags, globally decoded via a linear-chain Conditional Random Field with hard BIO transition constraints.
4. **Fast Inference:** Accelerated deployment via Dynamic INT8 ONNX Runtime graph execution and vectorized NumPy Viterbi decoding (`src/arn/viterbi_fast.py`).

---

## Installation

```bash
git clone https://github.com/Anishp-cell/ARN.git
cd ARN
pip install -e .
```

Verify the environment and test suite:
```bash
pytest -v tests/
```

---

## CLI & Quickstart

ARN provides a command-line tool `arn-cli` for real-time inference:

```bash
# Natural command routing
arn-cli "turn off the living room lights"

# Formatted JSON output
arn-cli "set a timer for 10 minutes labeled pasta" --json

# Inspect the canonical 18-tool schema and 26 BIO tags
arn-cli --list-tools
```

Sample JSON output:
```json
{
  "tools": [
    "handle_smart_home"
  ],
  "confidence": 0.9871,
  "tool_confidences": {
    "handle_smart_home": 0.9871
  },
  "arguments": {
    "action": "turn off",
    "device": "living room lights"
  },
  "cloud_fallback": false,
  "rejected": false
}
```

---

## Reproducibility & Master Protocol

All empirical claims are generated end-to-end via reproducible command scripts:

### 1. 15-Fold Master Cross-Validation Protocol
```bash
python scripts/run_master_protocol.py
```
Executes 105 total model runs across 15 tool-stratified cluster folds, storing predictions in `results/per_cluster_predictions.json` and computing 2,000-iteration cluster bootstrap confidence intervals in `results/master_protocol_summary.json`.

### 2. Ablations & Learning Curves
```bash
python scripts/run_phase5_ablations.py
```
Performs CRF layer decomposition (CRF vs Masked-Viterbi vs Greedy), supervision ablation (Joint vs Tool-only vs Slot-only), sample efficiency scaling (20% to 100%), and multi-tool pairing counts.

### 3. Error Analysis
```bash
python scripts/run_phase6_error_analysis.py
```
Generates 8,506-span error taxonomy breakdowns (exact matches, boundary shifts, label mismatches, false negatives, false positives) and cluster hardness rankings in `results/error_analysis.json`.

### 4. Deployment & Latency Benchmarks
```bash
python scripts/run_deployment_benchmark.py
```
Exports ONNX FP32 and Dynamic INT8 models, validates mathematical equivalence of fast NumPy Viterbi against PyTorch CRF, and logs 500-iteration latency distributions and isolated subprocess RSS to `results/deployment_benchmark.json`.

---

## Repository Documentation Index

- [`MODEL_CARD.md`](MODEL_CARD.md): Detailed model card, operational envelope, ethical considerations, and limitations.
- [`DATA_STATEMENT.md`](DATA_STATEMENT.md): Dataset curation, representability audit, and split stratification details.
- [`CHANGELOG.md`](CHANGELOG.md): Project release history and audit milestones.
- [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md): Pre-registered evaluation protocols and decision rules.
- [`docs/RECONCILIATION.md`](docs/RECONCILIATION.md): Complete forensic audit and resolution of historical discrepancies.
- [`docs/HUMAN_EVAL_PENDING.md`](docs/HUMAN_EVAL_PENDING.md): Status, guide, and harness for the 250-utterance human evaluation benchmark.
- [`docs/PYPI_NOTE.md`](docs/PYPI_NOTE.md): Deprecation notice for legacy unverified checkpoints.

---

## Citation & Attribution

```bibtex
@article{pattnaik2026arn,
  title={ARN: A Sub-18MB Non-Generative Joint Encoder-CRF Model for Real-Time On-Device Tool Orchestration},
  author={Pattnaik, Anish},
  year={2026},
  journal={arXiv preprint}
}
```

Licensed under the [MIT License](LICENSE).
