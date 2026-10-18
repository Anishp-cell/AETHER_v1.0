# Changelog

All notable changes to the Aether Routing Network (ARN) project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-18 (Master Protocol & Audited Release)

### Added
- **15-Fold Master Cross-Validation Protocol:** Fully automated 105-run evaluation across 15 tool-stratified cluster folds over all 5,000 samples (`scripts/run_master_protocol.py`).
- **Cluster Bootstrap Statistical Analysis:** 2,000-iteration cluster bootstrap confidence intervals reported in `results/master_protocol_summary.json`.
- **Pre-Registered Winning Model Selection:** Fine-tuned `CLSCRFModel` selected via pre-registered decision rules, confirming Outcome A (+13.47% Joint EM over frozen trunk).
- **Final Model Artifacts:** Packaged `checkpoints/arn_final_model/` with `model.pt` (17.58 MB), `config.json`, `tools.json`, `slots.json`, `tokenizer/`, and `manifest.json` with SHA-256 verification.
- **Fast Vectorized NumPy Viterbi:** Implemented in `src/arn/viterbi_fast.py`, verified with 100% exact mathematical equivalence across 10,000 randomized test sequences (`tests/test_viterbi_fast.py`).
- **ONNX FP32 & Dynamic INT8 Deployment:** Exported `model.onnx` (16.75 MB) and `model_int8.onnx` (4.28 MB, 3.92x compression ratio).
- **Hardware & Latency Benchmarks:** Measured 1.39 ms (1T) and 0.75 ms (4T) ONNX INT8 latency, 3.10 ms (1T) end-to-end pipeline latency, and 4.21 MB isolated peak RSS (`results/deployment_benchmark.json`).
- **Error Taxonomy Analysis:** Detailed categorization of 8,506 test spans across boundary shifts (42.71%), exact matches (38.10%), false negatives (12.93%), false positives (4.74%), and label mismatches (1.52%).
- **CRF Decomposition & Ablations:** Quantified CRF contribution (36.58% strict F1) vs Masked-Viterbi (8.89%) vs Greedy (1.45%).
- **Human Evaluation Protocol:** Sealed 250-utterance benchmark template (`data/human_eval_template.jsonl`), automated verification script (`scripts/make_human_eval_template.py`), evaluation harness (`scripts/run_human_eval.py`), and documentation (`docs/HUMAN_EVAL_PENDING.md`).
- **Documentation Suite:** Added `MODEL_CARD.md`, `DATA_STATEMENT.md`, `docs/PREREGISTRATION.md`, `docs/RECONCILIATION.md`, and `docs/PYPI_NOTE.md`.

### Changed
- **CLI Upgraded (`arn-cli`):** Updated `src/arn/cli.py` to prioritize `checkpoints/arn_final_model/model.pt` and dynamically support both `CLSCRFModel` and `AetherRoutingNetwork`.
- **Inference Pipeline:** Updated `src/arn/inference.py` to support dual forward signatures.
- **Split Redesign:** Replaced greedy descending binning with tool-stratified cluster grouping, guaranteeing tool representation across all folds and bounding skeleton edit distances $< 0.80$.

### Removed
- **Unverified Claims & Figures:** Permanently removed untraceable numbers ("14.2 ms", "5.55 MB", "810x", "38x", "216x", "180x", "88.0", "92.3%", "96.0%", "62.4") and unverified LoRA assertions.

---

## [0.2.0] - 2026-10-18 (Forensic Audit & Baseline Resolution)

### Added
- **Single Source of Truth:** Initialized `results/ledger.json` governing all downstream tables, macros, and text numbers.
- **Reconciliation Audit:** Documented 8 historical discrepancies in `docs/RECONCILIATION.md` ($MS_W = 0.01183, \rho = 0.8778$, smart-home+media count = 114).
- **Baseline Fixes:** Diagnosed and corrected TF-IDF kNN (~64.5%) and LogReg (~75.4%) by eliminating English stop-word filtering and introducing inner-validation threshold tuning.

---

## [0.1.0] - 2026-07-20 (Initial Exploration & Prototypes)

### Added
- Initial exploratory research codebase on Aether Routing Network architecture.
- Prototype Factored Attention Routing (FAR) and Linear-Chain CRF modules.
- Initial synthetic dataset collection (5,000 samples).
