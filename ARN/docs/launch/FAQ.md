# Frequently Asked Questions (FAQ)

### Q1: What is ARN and what problem does it solve?
**A:** ARN (Aether Routing Network) is an ultra-compact, non-generative neural network designed for real-time tool orchestration on edge devices. Instead of relying on a multi-billion parameter autoregressive language model to decide which local API to invoke and what arguments to extract, ARN formulates tool calling as joint multi-label classification and structured token sequence tagging over a 4.39M parameter BERT-tiny encoder trunk. This enables deterministic tool dispatch with sub-2 ms latency, an isolated peak RAM footprint of 4.21 MB, and guaranteed valid JSON syntax by construction.

---

### Q2: Why use a non-generative model instead of small LLMs (e.g., Llama-3-8B, Gemma-2B, Phi-3)?
**A:** Small LLMs remain too resource-intensive and fragile for low-power edge daemons:
1. **Memory:** LLMs require hundreds of megabytes to multiple gigabytes of memory to hold weights and KV caches, whereas ARN has an isolated peak memory footprint of only 4.21 MB RSS.
2. **Latency:** Sequential autoregressive decoding introduces 50–300 ms token latency, while ARN performs joint classification and argument extraction in a single forward pass (1.39 ms ONNX INT8 engine latency on 1 thread, 0.75 ms on 4 threads).
3. **Syntax Reliability:** Generative models can hallucinate invalid parameters, drop quotation marks, or generate malformed JSON. ARN extracts argument spans directly from input token offsets, guaranteeing zero syntax errors and zero hallucinated text.

---

### Q3: Why was Outcome A (fine-tuned encoder) selected instead of a frozen trunk?
**A:** Under our pre-registered Decision Rule P-2, model selection was determined strictly by empirical performance under 15-fold cluster-stratified cross-validation. The fine-tuned architecture achieved **16.91% Joint Exact Match** [95% CI: 13.51%, 20.59%], compared to only **3.44% Joint Exact Match** for the frozen trunk baseline — an improvement of **+13.47% Joint Exact Match** [95% CI: 10.27%, 16.68%]. Fine-tuning the compact encoder is essential for aligning general language representations with domain-specific tool semantics.

---

### Q4: Why is the Linear-Chain CRF necessary, and what happened in the ablations?
**A:** Standard classification heads predict token tags independently, which frequently results in illegal BIO sequence transitions (such as predicting `I-ARG` without an immediately preceding `B-ARG` of the same type). In our ablation experiments:
- **Greedy Argmax Decoding:** Achieved only 1.45% Strict Span F1 and produced **58.70% illegal transition violations**.
- **Masked Viterbi Decoding:** Enforced valid transitions post-hoc, reaching 8.89% Strict Span F1 (0.00% violations).
- **Full Linear-Chain CRF:** Jointly learned global tag transition dynamics during training, achieving **36.58% Strict Span F1** with **0.00% transition violations**.
The CRF ensures that extracted argument spans strictly obey BIO sequence grammar rules.

---

### Q5: Why is Joint Exact Match 16.91% while Lenient Span F1 is 67.88%?
**A:** Joint Exact Match is an all-or-nothing metric requiring *every* activated tool and *every* argument span (with exact character boundaries) to be 100% correct simultaneously. Our error taxonomy analysis across 8,506 test spans reveals:
- **Exact Span Matches:** 38.10% (3,241 spans)
- **Boundary Shifts:** 42.71% (3,633 spans)
- **Over-Extraction:** 11.23% (955 spans)
- **Under-Extraction:** 6.44% (548 spans)
- **Label Mismatches:** Only 1.52% (129 spans)

The dominant source of error is minor boundary variation (e.g., including or omitting leading determiners like "the" or trailing prepositions). Semantically, the model identifies the correct entity type 98.48% of the time, explaining the high Lenient Span F1 (67.88%).

---

### Q6: Can ARN extract open-vocabulary entities or unseen words?
**A:** Yes. Because ARN operates as an extractive sequence tagger over WordPiece tokens, it extracts argument spans directly by slicing character offsets from the original user prompt. It does not generate text from a fixed output vocabulary. As verified in our Data Statement, 94.79% of argument spans in the benchmark are verbatim substrings of the input prompt.

---

### Q7: What hardware was used for deployment benchmarks?
**A:** Benchmarks were conducted on host x86 CPU hardware:
- **Processor:** Intel Core i7-13700HX (Windows 11)
- **Runtime:** ONNX Runtime 1.23.2
- **ONNX INT8 Latency (1 thread):** 1.39 ms mean (p50: 1.25 ms, p95: 1.83 ms)
- **ONNX INT8 Latency (4 threads):** 0.75 ms mean (p50: 0.69 ms, p95: 1.09 ms)
- **End-to-End Pipeline Latency:** 3.10 ms (1T) / 2.34 ms (4T) (including tokenization, ONNX forward pass, vectorized NumPy Viterbi decoding, and JSON slot reconstruction)
- **Peak RAM Footprint:** 4.21 MB (isolated subprocess RSS)
- **Model Storage:** 4.28 MB (Dynamic INT8 quantized ONNX, 3.92x compression over 16.75 MB FP32)

---

### Q8: What is the status of the human evaluation?
**A:** We pre-registered and constructed a 250-utterance sealed holdout test set (`data/human_eval_template.jsonl`) spanning all 18 tools and balanced prompt pairs. In accordance with strict scientific integrity standards (documented in `docs/HUMAN_EVAL_PENDING.md`), we do not report synthetic LLM-as-a-judge ratings as human evaluation. The evaluation harness (`scripts/run_human_eval.py`) is fully implemented with Cohen's kappa inter-annotator agreement and is ready to score independent external human ratings once completed.

---

### Q9: How are the numbers in the paper, documentation, and release verified?
**A:** Every single empirical metric in the paper, README, and documentation is generated directly from `results/ledger.json` via automated scripts (`scripts/generate_paper_numbers.py` and `scripts/make_tables.py`). Automated CI tests (`tests/test_paper_integrity.py`) audit the manuscript to guarantee zero unverified marketing claims and zero fabricated numbers.

---

### Q10: How do I install and run ARN locally?
**A:** You can install and run ARN locally using pip:
```bash
git clone https://github.com/Anishp-cell/ARN.git
cd ARN
pip install -e .
arn-cli "set an alarm for 7:30 AM tomorrow labeled morning" --json
```
For low-latency production deployments, an ONNX INT8 pipeline script is provided in `scripts/benchmark_deployment.py`.
