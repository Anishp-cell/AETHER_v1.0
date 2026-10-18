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


