# Work Package Prompt: WP-13 (WP-M) — Baseline and Ledger Integrity Audit

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Large (1 session, ~90–120 min)  
**Compute Needed:** CPU only (statistical audits, scikit-learn fitting, inference timing)  
**Prerequisites:** WP-01 (WP-A) completed  
**Blocks:** WP-03 (WP-C), WP-04 (WP-D), WP-09 (WP-I), WP-10 (WP-J), WP-11 (WP-K)  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-M and Section 4: Authoritative Sources)
3. `ARN/docs/plan/STATE_RECONSTRUCTION.md`
4. `ARN/docs/plan/RISKS_AND_DECISIONS.md` (Decision on TF-IDF LogReg vs Primary Model)
5. `ARN/results/ledger.json`
6. `ARN/results/master_protocol_summary.json`
7. `ARN/FINAL_REPORT.md`
8. `ARN/scripts/run_baselines.py` (or baseline training scripts)
9. `ARN/data/splits/cv_manifest.json`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- No fabrication: every baseline score must be derived from `cv_manifest.json` folds via rigorous Python scripts.
- Vectorizers, vocabulary builders, and scaling pipelines MUST be fitted strictly inside training folds; zero leakage across fold boundaries.
- Decision Tree resolution is mandatory: WP-I (paper), WP-J (README), and WP-K (launch) CANNOT proceed until WP-M has validated the TF-IDF LogReg router and established the framing.

---

### Objective & Scope

Establish absolute statistical integrity across all 10 models in the master table. Audit the surprising `74.24%` tool subset accuracy of TF-IDF Logistic Regression (which significantly exceeds the 39.91% tool accuracy of the primary fine-tuned CLS+CRF model). Construct a hybrid baseline (TF-IDF LogReg + BERT-tiny CRF), evaluate size and latency across all baselines, and resolve the architectural value proposition.

---

### Exact Execution Steps

1. **Three-Column Master Table Diff:**
   Build a comprehensive 3-column verification table for all 10 models across Joint Exact Match, Tool Subset Accuracy, Strict Span F1, and Lenient Span F1:
   - **Column 1:** Value in last pre-context-loss `FINAL_REPORT.md` (e.g., 16.91% joint EM, 39.91% tool acc).
   - **Column 2:** Value in `results/master_protocol_summary.json`.
   - **Column 3:** Value calculated directly from the 15 outer fold runs in `results/ledger.json`.
   - Document and explain every numerical divergence. Designate `results/master_protocol_summary.json` (as compiled from `ledger.json`) as the sole authoritative file.

2. **Exhaustive Forensic Audit of TF-IDF Logistic Regression (74.24% Tool Accuracy):**
   Execute a dedicated audit script `scripts/audit_tfidf_logreg.py`:
   - **Fold Alignment:** Verify that TF-IDF LogReg was evaluated on the identical 15 cluster-disjoint folds specified in `data/splits/cv_manifest.json`.
   - **Data Leakage Check:** Verify `TfidfVectorizer` was initialized and fit strictly on `train_indices` within each fold; verify no global vocabulary or IDF weights leaked from validation or test folds.
   - **Threshold Calibration:** Verify that multi-label prediction thresholds $\tau_k$ were calibrated strictly on inner validation folds, not test folds.
   - **Metric Calculation:** Confirm exact subset accuracy calculation matches the neural models: $\mathbb{I}(\hat{Y} = Y_{\text{gold}})$.
   - **Sample Counts:** Provide exact integer $k/N$ counts per fold ($N \approx 333$ per fold, total $N=5,000$).
   - **Label-Shuffle Sanity Test:** Train TF-IDF LogReg on permuted/shuffled tool labels within the training fold. Evaluate on the unpermuted test fold. Accuracy MUST collapse to chance / majority class baseline (~7.90%). If accuracy remains high, a fatal data leak exists.
   - **Random vs Cluster-Disjoint Sanity Test:** Evaluate TF-IDF LogReg on random splits vs cluster-disjoint splits to measure the generalization drop.
   - **Per-Tool & Per-Cluster Breakdown:** Identify which tools and semantic clusters drive the 74.24% accuracy.
   - **Historical Reconciliation:** Compare against earlier Stage 2b reports (kNN 53%, LogReg 34.5%) and document the exact reason for the increase (sublinear TF scaling, n-gram range, calibrated thresholds per D-006).

3. **Evaluate TF-IDF kNN & Hybrid Router-Tagger Baseline:**
   - **TF-IDF kNN:** Evaluate kNN ($k=5$) over identical 15 folds with exact subset scoring.
   - **Hybrid Baseline (TF-IDF LogReg Router + Frozen BERT-tiny CRF Tagger):**
     - Step 1: Route tools using the TF-IDF LogReg model (74.24% tool acc).
     - Step 2: Extract argument spans using the frozen BERT-tiny linear-chain CRF tagger from Phase 4.
     - Measure Joint Exact Match, Strict Span F1, and Lenient Span F1 across the 15 folds.
     - Compare Joint EM of Hybrid vs Fine-Tuned CLS+CRF (16.91%).

4. **Baseline Memory & Latency Profiling:**
   - Measure on-disk footprint (serialized pickle / joblib size in MB) for:
     - Majority Class baseline
     - First-Token Prior baseline
     - TF-IDF kNN baseline
     - TF-IDF Logistic Regression baseline
     - Hybrid baseline (TF-IDF + BERT-tiny CRF)
   - Measure single-thread and 4-thread CPU inference latency per query (in milliseconds) on the reference i7-13700HX CPU.

5. **Decision Tree Execution (Resolving the Value Proposition):**
   - **Branch A — If LogReg 74.24% is Legitimate (No Leaks, Shuffles Collapse):**
     * Conclude that lexical n-gram features on 5,000 template-generated utterances provide a strong routing signal for closed $K=18$ tools.
     * **Paper / Launch Shift:** Remove claims that the tiny BERT-tiny encoder is superior at tool routing. Restate the core value proposition: *ARN is a unified, single-pass joint model that achieves argument span extraction (43.61% strict F1) and tool routing in a single 4.28 MB INT8 binary in 0.75 ms CPU latency, eliminating the multi-stage latency, serialization overhead, and separate memory footprint of pipeline architectures (such as TF-IDF + Tagger hybrids).*
   - **Branch B — If a Leak or Bug is Discovered:**
     * Document the flaw in `docs/RECONCILIATION.md` and `DECISIONS.md`.
     * Fix the implementation, rerun over the 15 folds with strictly inner-validation tuning, and append corrected runs to `results/ledger.json`.

---

### Acceptance Criteria (Pass / Fail)

- [ ] 3-column diff completed for all 10 models; all differences explained.
- [ ] Label-shuffle sanity test passes (shuffled accuracy collapses to $\le 10\%$).
- [ ] Fold-isolation verified (TfidfVectorizer fit strictly on training fold).
- [ ] Hybrid baseline (LogReg + CRF) evaluated across all 15 folds with Joint EM recorded.
- [ ] Size (MB) and CPU latency (ms) measured for all baselines under the standard deployment protocol.
- [ ] Decision Tree branch resolved and documented in `DECISIONS.md`.
- [ ] Git commit created with exact message: `feat: complete baseline and ledger integrity audit (WP-M)`.
