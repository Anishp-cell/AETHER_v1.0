# Work Package Prompt: WP-03 (WP-C) — Results Reconciliation & Discrepancy Audits

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Medium-Large (1 session, ~90 min)  
**Compute Needed:** CPU only (Python script execution and document editing)  
**Prerequisites:** WP-01 (WP-A) and WP-13 (WP-M) completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-C and Section 4: Authoritative Sources)
3. `ARN/docs/plan/STATE_RECONSTRUCTION.md` (Section 3: Briefing vs Repo Discrepancies)
4. `ARN/docs/plan/WP_13_baseline_ledger_integrity.md` (WP-M findings)
5. `ARN/results/master_protocol_summary.json`
6. `ARN/results/ablations_summary.json`
7. `ARN/results/ledger.json`
8. `ARN/docs/paper/tables/table1_main_results.tex`
9. `ARN/docs/paper/numbers.tex`
10. `ARN/docs/RECONCILIATION.md`
11. `ARN/FINAL_REPORT.md`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- No fabrication: every metric must match `master_protocol_summary.json` and `ablations_summary.json`.
- Integer $k/N$ counts must accompany every rate or percentage.
- One single source of truth per statistic.

---

### Objective & Scope

Eliminate all numeric contradictions across the paper, reports, and reconciliation docs. Provide explicit explanations for apparent anomalies (decoder ablation gap, multi-tool counts, TF-IDF baseline progression).

---

### Exact Execution Steps

1. **Reconcile Decoder Ablation Discrepancy:**
   - In `ablations_summary.json`, the CRF decomposition reports:
     - Greedy: 1.45% strict span F1 (58.70% BIO constraint violations).
     - Masked Viterbi: 8.89% strict span F1 (0% BIO violations).
     - Linear-Chain CRF: 36.58% strict span F1 (0% BIO violations).
   - In `Table 1`, fine-tuned CLS+CRF reports 43.61% strict span F1, and frozen FAR+CRF reports 34.34%.
   - **Root Cause & Reconciliation:** The decoder ablation evaluates the *frozen ARN trunk* under isolated decoder modifications, whereas the headline 43.61% is achieved by the *fine-tuned CLS+CRF* architecture. Document this explicitly in `docs/RECONCILIATION.md` and `FINAL_REPORT.md`.
2. **Script-Based Verification of Multi-Tool Utterance Counts:**
   - Implement and execute `scripts/verify_multitool_counts.py`:
     - Load canonical dataset (`data/raw/synthetic_5000.jsonl` or canonical split file).
     - Compute and verify total sample count $N=5,000$.
     - Compute multi-tool count (`len(gold_tools) > 1`) and verify exact count is 1,132.
     - Compute per-pair breakdown across all tool combinations.
     - Verify exactly 114 samples for `handle_smart_home` + `media_control`.
     - Verify remaining 6 template pairs have 0 instantiated training samples.
     - Verify 142 total spans with label collisions / overlaps (114 action collisions + 28 overlap boundaries).
     - Save summary output to `results/multitool_audit.json`.
   - Update `docs/RECONCILIATION.md`, `FINAL_REPORT.md`, and `docs/paper/numbers.tex` so that 1,132 is the universal multi-tool count, and 114 is specifically qualified as the smart home + media control collision subset.
3. **Document TF-IDF Baseline Evolution:**
   - Incorporate the audit findings from WP-M into `docs/RECONCILIATION.md`:
     - Prototype Phase: Flawed feature extraction yielded artifactual 53%/34%.
     - Initial Cluster Audit: Severe OOV leakage collapse (2.6%/0%).
     - Bug Fix (Phase 3, D-006): Corrected tokenization, sub-linear TF scaling, and multi-label threshold calibration yielded verified 63.52% (kNN) and 74.24% (Logistic Regression).
   - Ensure Table 1 uses `--` for span and joint metrics on tool-only baselines, with a note that span tagging is non-applicable.
4. **Audit and Synchronize Table 1 & numbers.tex:**
   - Run `python scripts/generate_paper_numbers.py` and `python scripts/make_tables.py`.
   - Run `python scripts/audit_paper_numbers.py` to confirm zero discrepancies between LaTeX tables and underlying JSON summaries.

---

### Acceptance Criteria (Pass / Fail)

- [ ] `docs/RECONCILIATION.md` contains dedicated sections for Decoder Ablation, Multi-Tool Counts, and TF-IDF Evolution.
- [ ] Table 1 in `docs/paper/tables/table1_main_results.tex` perfectly matches `master_protocol_summary.json`.
- [ ] `python scripts/audit_paper_numbers.py` executes with exit code 0.
- [ ] Git commit created with exact message: `fix: reconcile all numbers, explain decoder ablation gap, unify multi-tool counts`.
