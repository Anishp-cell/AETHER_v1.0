# ARN Architecture & Protocol Decisions Log

This document records all non-trivial decisions made during the ARN project revision and final execution, including alternatives considered, empirical rationale, and constraints enforced.

---

## Decision Index

### D-001: Branch Strategy and Clean State
- **Date:** 2026-10-18
- **Context:** User requested end-to-end execution of Phases 0 through 14 without stopping for approval, with commits to branch `arn-final`.
- **Decision:** Created branch `arn-final` from the clean working state on `arn-revision`. All changes confined strictly to `ARN/` (and staged `../ARN-public` in Phase 12).
- **Alternative Rejected:** Working directly on master or force-pushing (violates Hard Rule 2).

### D-002: Automatic Hardware Environment Capture
- **Date:** 2026-10-18
- **Context:** Phase 0 requires recording exact hardware specs into `results/environment.json` without prompting the user.
- **Decision:** Executed `record_environment.py` gathering psutil, torch, and platform attributes. Identified exact CPU as `13th Gen Intel(R) Core(TM) i7-13700HX` (16 physical cores, 24 logical processors) with 15.71 GB RAM and RTX 5050 Laptop GPU under Windows 10/11 64-bit.
- **Alternative Rejected:** Hand-typing or approximating CPU model (violates Hard Rule 3 and Item 10).

### D-003: Pre-registration of Decision Rules (Phase 2 Specifications)
- **Date:** 2026-10-18
- **Context:** Hard pre-registration rules must be fixed before Phase 4 experiments to eliminate post-hoc p-hacking and cherry-picking.
- **Decision:** Established `docs/PREREGISTRATION.md` specifying:
  1. Unit of independence = semantic cluster ($B \ge 2000$ cluster bootstrap).
  2. Primary metrics: Tool Subset Accuracy (P1), Joint Exact Match (P2), Strict Span F1 (P3).
  3. Minimum delta for claiming a difference: 95% CI excludes 0 AND $|\Delta| \ge 2.0\%$.
  4. Model selection: Highest mean Joint EM; simpler model chosen if not reliably worse.
  5. FAR described as a contribution ONLY if matching CLS-head is beaten by the rule; otherwise reported as evaluated alternative.
  6. Outcome A (fine-tuning supported) vs. Outcome B (otherwise) determining paper narrative.
- **Alternative Rejected:** Selecting model by tool accuracy alone or evaluating significance at the utterance level.

### D-004: Tool-Stratified Cluster Allocation and Coverage Confound Elimination (Phase 1)
- **Date:** 2026-10-18
- **Context:** An audit of the Stage 2d cross-validation folds revealed that greedy descending binning caused rare tools to be absent from training in several folds (e.g., Old Fold 5 missing 3 tools affecting 33.25% of its validation set; Old Fold 2 missing `read_specific_url` affecting 7.9% of its validation set).
- **Decision:** Built a tool-stratified grouped K-fold allocation (5 folds $\times$ 3 shuffles = 15 outer folds) across all 5,000 samples. Tools with $\le 5$ clusters are distributed across folds first before general clusters are allocated. `open_url` (1 cluster, 37 samples) is designated train-only across all 15 folds. Every fold guarantees $\ge 1$ cluster per tool in outer training and inner training. Maximum skeleton token-edit similarity is verified $< 0.80$ (range 0.71–0.79). Frozen in `ARN/data/splits/cv_manifest.json`.
- **Alternative Rejected:** Random cluster assignment (causes severe tool starvation on rare classes); standard stratified k-fold without grouping (causes data leakage across paraphrased templates).

### D-005: Results Ledger and Resolution of Historical Inconsistencies (Phase 2)
- **Date:** 2026-10-18
- **Context:** Previous exploration stages (Stage 1 to 2d) had discrepancies across tables, CI coverage issues, tool-pair count differences (114 vs 45), and differing frozen ARN vs CLS numbers.
- **Decision:**
  1. Initialized `ARN/results/ledger.json` as the exclusive single source of truth for all downstream scripts, tables, figures, and macros.
  2. Moved all legacy stage outputs to `ARN/results/_superseded/` with an explanatory README.
  3. Formalized the 8 reconciliation verdicts in `ARN/docs/RECONCILIATION.md`, including exact ICC recomputation ($MS_W = 0.01183, \rho = 0.8778$), tool-pair count verification (114 confirmed), and CI cluster-bootstrap methodology.
- **Alternative Rejected:** Silently overwriting or deleting old files without reconciliation (violates Hard Rule 3).

### D-006: Root Cause Diagnosis and Resolution for TF-IDF Baselines (Phase 3)
- **Date:** 2026-10-18
- **Context:** In early Stage 2b, TF-IDF kNN achieved 53.0% and LogReg achieved 34.5%, but in Stage 2d they collapsed to 2.55% and 0.00%. kNN also failed to improve with data size.
- **Root Cause Identified:**
  1. `stop_words="english"`: English stop words stripped essential imperative trigger words ("what", "is", "it", "on", "off", "to", "and", "me", "a", "the") from command utterances like "tell me the time" or "turn on lights", leaving empty or single-word vectors.
  2. Fixed 0.50 threshold on uncalibrated/unbalanced multi-label probabilities: rare tools (0.5% support) produced probabilities below 0.50, causing LogReg to predict all-zero vectors.
  3. Hard multi-label predict in kNN: required $>50\%$ votes simultaneously for multiple tools, which collapsed on multi-tool samples.
  4. Global vectorizer fit: vocabulary and IDF weights leaked across fold splits.
- **Decision:**
  1. Enforced strictly fold-isolated vectorizer fitting (`fit_transform` on `train_idx` only; `transform` on `val_idx`).
  2. Removed `stop_words="english"`.
  3. Introduced soft probability averaging for kNN and class-balanced OvR for LogReg, tuning decision threshold $\tau \in [0.20, 0.70]$ strictly on inner validation.
  4. Added unit tests verifying 100-sample overfit ($\ge 95\%$), random split sanity ($\ge 65\%$), and monotonic scaling with training data.
- **Alternative Rejected:** Reporting artificially collapsed 0.0% numbers or using global vectorizer fits.

### D-007: Human Evaluation Benchmark Integrity & Deferred Protocol (Phase 9)
- **Date:** 2026-10-18
- **Context:** The repository contains a 250-utterance benchmark template (`ARN/data/human_eval_template.jsonl`) designed across 18 tools and compositions, but no external filled human ratings currently exist.
- **Decision:**
  1. Adhered strictly to Hard Rule 1 (Zero Fabrication) and pre-registration guidelines: synthetic LLM auto-filling of the human evaluation test set is strictly prohibited to avoid LLM stylistic leakage.
  2. Sealed the 250-sample template at `data/human_eval_template.jsonl`.
  3. Authored `docs/HUMAN_EVAL_PENDING.md` documenting the benchmark structure, annotation instructions, and deferred evaluation protocol.
  4. Built and unit-tested `scripts/run_human_eval.py` equipped with exact scoring metrics, Cohen's kappa calculation for inter-annotator agreement, and a `--mock` validation mode.
  5. Recorded the status as PENDING in `results/ledger.json` and deferred empirical claims to future multi-rater user studies.
- **Alternative Rejected:** Populating "human" evaluation entries using LLM synthetic generations or reporting fictitious user ratings.

### D-008: Repository Finalization and Architecture Harmonization (Phase 10)
- **Date:** 2026-10-18
- **Context:** To finalize the repository for public release, CLI routing, documentation, model card, data statement, and changelog needed to be aligned with the Outcome A winning model (`CLSCRFModel`) while preserving backward compatibility.
- **Decision:**
  1. Updated `src/arn/cli.py` to prioritize `checkpoints/arn_final_model/model.pt` and dynamically detect whether a checkpoint belongs to `CLSCRFModel` or `AetherRoutingNetwork`.
  2. Updated `src/arn/inference.py` to accept both dual-output models (`tool_logits, emissions`) and 4-tuple models (`tool_probs, tool_preds, tag_preds, loss`).
  3. Rewrote `README.md` strictly using verified ledger numbers and deployment benchmark outputs (16.91% Joint EM, 1.39 ms INT8 latency, 4.21 MB peak RSS).
  4. Authored standard documentation suite: `MODEL_CARD.md`, `DATA_STATEMENT.md`, `CHANGELOG.md`, and validated `LICENSE`.
- **Alternative Rejected:** Forcing legacy FAR-only checkpoint loading in the CLI or publishing unverified marketing statistics in `README.md`.

### D-009: Paper Compilation, LaTeX Macro Traceability, and Zero Banned Claims (Phase 11)
- **Date:** 2026-10-18
- **Context:** The legacy paper draft contained superseded claims and unverified metrics from pre-audit iterations.
- **Decision:**
  1. Built `scripts/generate_paper_numbers.py` to compile all empirical metrics directly from `results/ledger.json` and `results/master_protocol_summary.json` into `docs/paper/numbers.tex`.
  2. Upgraded `scripts/make_tables.py` to auto-generate Tables 1 through 4 (Master Protocol 15-fold cross-validation with cluster bootstrap CIs, hardware latency/RSS benchmarks, CRF layer decomposition, and 8,506-span error taxonomy).
  3. Completely rewrote `AnonymousSubmission2027.tex`, `CameraReady2027.tex`, and `ReproducibilityChecklist.tex`, framing the primary narrative around the verified winning model (Outcome A: Fine-Tuned CLS+CRF, 16.91% Joint EM, +13.47% over frozen trunk).
  4. Implemented `scripts/audit_paper_numbers.py` and unit test `tests/test_paper_integrity.py` to enforce zero banned claims, macro consistency, and balanced LaTeX syntax in CI.
- **Alternative Rejected:** Manually editing LaTeX numbers or retaining superseded narrative claims.






