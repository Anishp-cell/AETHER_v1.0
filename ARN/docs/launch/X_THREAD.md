# X (Twitter) Launch Thread Draft

**Thread Title:** Introducing ARN v1.0.0: Sub-18MB Non-Generative Joint Tool Orchestration

---

### Tweet 1 (Hook)
Stop using multi-billion parameter autoregressive LLMs just to parse local user commands on edge devices.

Today, we're open-sourcing **ARN (Aether Routing Network) v1.0.0**: an ultra-compact, non-generative neural tool orchestrator designed for real-time edge execution.

1.39 ms latency. 4.21 MB RAM. 100% deterministic JSON. 🧵👇

---

### Tweet 2 (The Problem)
Why is edge tool calling broken?

When a user says "turn off living room lights", deploying a 1B–3B generative model means:
- 100s of MBs of RAM reserved for intent parsing
- 50–200 ms autoregressive token decoding
- Syntax crashes from unescaped quotes or malformed JSON

Free-form text generation is the wrong abstraction for API dispatch.

---

### Tweet 3 (The Solution: Non-Generative Joint Formulation)
ARN formulates tool orchestration as simultaneous multi-label classification and token sequence labeling:

1. **Encoder Trunk:** 4.39M param BERT-tiny (`google/bert_uncased_L-2_H-128_A-2`)
2. **Tool Routing:** Multi-label sigmoid classifier over 18 APIs
3. **Slot Extraction:** Linear-chain CRF with hard BIO transition constraints

Single forward pass. Zero syntax errors by design.

---

### Tweet 4 (The Pre-Registered Protocol)
No cherry-picked benchmarks.

We executed a pre-registered Master Protocol across 15 tool-stratified cluster folds:
- 5,000 conversational utterances
- 253 semantic paraphrase clusters (bounded token edit distance < 0.80)
- 105 outer training runs
- 2,000-iteration cluster bootstrap confidence intervals

---

### Tweet 5 (Outcome A Confirmed)
The results? End-to-end fine-tuning statistically outperforms frozen trunk representations:

- **Joint Exact Match:** 16.91% [95% CI: 13.51%, 20.59%]
- **Tool Subset Accuracy:** 39.91% [95% CI: 34.78%, 45.45%]
- **Strict Span F1:** 43.61% | **Lenient F1:** 67.88%
- **Paired Delta:** +13.47% Joint EM over frozen trunk [95% CI: 10.27%, 16.68%]

Task-specific adaptation of edge encoders is essential.

---

### Tweet 6 (Why the CRF Layer is Essential)
We ablated the sequence decoding layer:
- **Greedy Argmax:** 1.45% Strict F1 with 58.7% illegal BIO transition violations
- **Masked Viterbi:** 8.89% Strict F1 (0.0% violations)
- **Full Linear-Chain CRF:** 36.58% Strict F1 (0.0% violations)

Global transition modeling provides sequence grammar guarantees that local classifiers cannot match.

---

### Tweet 7 (Real-Time Deployment Benchmarks)
Profiled on host CPU (Intel Core i7-13700HX, ONNX Runtime 1.23.2):

⚡ **ONNX INT8 Latency:**
- 1 Thread: 1.39 ms mean (p50: 1.25 ms, p95: 1.83 ms)
- 4 Threads: 0.75 ms mean (p50: 0.69 ms, p95: 1.09 ms)

🚀 **Full Pipeline Latency:** 3.10 ms (1T) / 2.34 ms (4T)  
💾 **Peak RAM Footprint:** 4.21 MB RSS (isolated subprocess)  
📦 **Disk Footprint:** 4.28 MB (3.92x compression over FP32)

---

### Tweet 8 (100% Open Source & Verified)
Every number in our paper traces strictly to `results/ledger.json`. No marketing inflation.

- GitHub: https://github.com/Anishp-cell/ARN
- Pretrained weights (`model.pt`, `model_int8.onnx`)
- 26 unit tests (`pytest tests/`)
- Model Card & Data Statement

Try the CLI:
`arn-cli "set a timer for 10 minutes labeled pasta" --json`

Star the repo and share your feedback! ⭐️
