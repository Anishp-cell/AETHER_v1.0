# Model Card for ARN (Aether Routing Network) v1.0.0

## 1. Model Details

- **Model Name:** ARN Final Model (`arn_final_model`)
- **Model Version:** 1.0.0
- **Model Architecture:** Joint Encoder-CRF (`CLSCRFModel`)
  - **Backbone Trunk:** `google/bert_uncased_L-2_H-128_A-2` (2 layers, 128 hidden dimension, 2 attention heads, fine-tuned end-to-end)
  - **Tool Routing Head:** Linear multi-label projection (`128 -> 18`) over pooled `[CLS]` embedding with sigmoid activation
  - **Slot Extraction Head:** Linear token emission projection (`128 -> 26`) over sequence representations
  - **Sequence Decoder:** Linear-chain Conditional Random Field (CRF) with hard BIO validity masking
- **Total Parameter Count:** 4,386,474 parameters
  - Backbone Encoder: 4,380,288 parameters
  - Tool Classifier Head: 2,322 parameters
  - Slot Emission Head: 3,354 parameters
  - CRF Transition Matrix: 728 parameters
- **License:** MIT License
- **Frameworks:** PyTorch 2.x, HuggingFace Transformers, ONNX Runtime 1.23.2

---

## 2. Intended Use & Scope

### Primary Intended Uses:
- **On-Device Command Routing:** Deterministic, low-latency classification of natural language user requests to closed API endpoints ($K=18$).
- **Verbatim Slot Extraction:** High-speed extraction of entity argument spans (e.g., recipient names, device identifiers, command strings, web queries) directly from user utterances.
- **Privacy-Preserving Edge Dispatch:** Local edge inference on host PCs, embedded appliances, and mobile devices without sending user audio or text transcripts to external cloud services.

### Out-of-Scope & Prohibited Uses:
- **Open-Domain Generative Dialogue:** ARN is non-generative by design and does not produce autoregressive text responses.
- **Multi-Turn Discourse Tracking:** Evaluated on single-turn conversational commands; multi-turn co-reference resolution is not natively supported.
- **Autonomous Unconstrained Execution:** Critical system commands (`run_computer_command`) should remain gated behind human confirmation dialogs in downstream runtime orchestrators.

---

## 3. Training Data & Methodology

- **Training Corpus:** 5,000 verified conversational command utterances from `aether_orchestrator_dataset.jsonl` (SHA-256: `d2f25d93a7123226f5f849e7f72391a2fd8b93c1b04c79fa2f5a2c8b8d5a95b9`).
- **Semantic Coverage:** 253 semantic paraphrase clusters spanning 18 canonical tool APIs.
- **Argument Representation:** 94.79% verbatim substring arguments, 5.21% non-verbatim (temporal/implied).
- **Optimization Strategy:**
  - Joint Loss: $\mathcal{L} = \mathcal{L}_{\text{BCE}} + \frac{1}{T}\mathcal{L}_{\text{CRF\_NLL}}$
  - Optimizer: AdamW
  - Trunk Learning Rate: $3 \times 10^{-5}$
  - Head Learning Rate: $1 \times 10^{-3}$
  - Batch Size: 64
  - Weight Decay: 0.01
  - Epochs: 10
  - Optimal Activation Threshold ($\tau$): 0.30 (median selected across 15 validation folds)

---

## 4. Evaluated Performance Metrics

All metrics were established under the pre-registered Master Protocol across 15 outer folds (5 folds $\times$ 3 cluster-stratified shuffles, 2,000-iteration cluster bootstrap):

| Metric | Mean ± Std | 95% Bootstrap CI | Baseline Comparison (Frozen Trunk) |
| :--- | :---: | :---: | :--- |
| **Joint Exact Match** | **16.91% ± 6.56%** | [13.51%, 20.59%] | +13.47% improvement over frozen trunk [10.27%, 16.68%] |
| **Tool Subset Accuracy** | **39.91% ± 12.19%** | [34.78%, 45.45%]) | +25.70% improvement over frozen trunk [20.73%, 30.67%] |
| **Strict Span F1** | **43.61% ± 10.43%** | [39.02%, 48.06%] | +18.85% improvement over frozen trunk [14.41%, 23.29%] |
| **Lenient Span F1** | **67.88% ± 9.86%** | [63.66%, 71.93%] | Substantial boundary shift tolerance |
| **BIO Sequence Violations** | **0.00%** | [0.00%, 0.00%] | Enforced by hard-masked Viterbi transition matrix |

---

## 5. Deployment & Hardware Benchmarks

Evaluated on host **13th Gen Intel Core i7-13700HX (Windows 11, ONNX Runtime 1.23.2)**:

| Quantization / Runtime | Latency (1 Thread) | Latency (4 Threads) | Disk Size | Compression Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch CPU FP32** | 11.70 ms mean | 8.15 ms mean | 17.58 MB | 1.00x |
| **ONNX Runtime FP32** | 2.50 ms mean | 1.35 ms mean | 16.75 MB | 1.05x |
| **ONNX Runtime INT8** | **1.39 ms mean** | **0.75 ms mean** | **4.28 MB** | **3.92x** |
| **Full Pipeline (INT8 + Viterbi)** | **3.10 ms mean** | **2.34 ms mean** | — | — |

- **Subprocess-Isolated Peak RSS:** **4.21 MB** (measured via isolated process spawning).

---

## 6. Limitations & Known Failure Modes

1. **Boundary Shifts (42.71% of errors):**
   - Fine-grained tokenization boundary shifts (e.g. including or omitting articles "the", "a", or prepositions "to", "on") represent the dominant error mode. While slot category is correct, strict span F1 penalizes boundary shifts.
2. **Rare Tool Cold-Start:**
   - Single-cluster tools (e.g., `open_url` with 37 samples in 1 cluster) cannot be reliably evaluated under zero-shot cluster holdouts and require template diversification.
3. **Multi-Tool Pair Density:**
   - The current dataset contains real natural multi-tool compositions predominantly for `handle_smart_home + media_control` (114 samples). Generalization to unseen pairwise compositions remains constrained by training pair support.

---

## 7. Ethical & Environmental Considerations

- **Privacy:** ARN processes queries entirely locally. No audio, text transcripts, or metadata leave the host machine.
- **Environmental Impact:** Training the complete 4.39M parameter model requires $< 0.005$ kWh of electricity ($< 2.5$ grams $\text{CO}_2\text{e}$), orders of magnitude below LLM fine-tuning runs.
- **Safety Gating:** Automated tools with destructive potential (`run_computer_command`) must incorporate explicit confirmation hooks in consumer applications.
