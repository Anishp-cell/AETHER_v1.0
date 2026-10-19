# Work Package Prompt: WP-12 (WP-L) — Final Forensic Audit & Definitive Final Report

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Medium (1 session, ~60 min)  
**Compute Needed:** CPU + lightweight GPU validation (reproducibility slice)  
**Prerequisites:** All previous Work Packages (WP-01 through WP-11) completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-L and Section 4: Global Verification Suite)
3. `ARN/FINAL_REPORT.md`
4. `ARN/results/ledger.json`
5. `ARN/scripts/audit_paper_numbers.py`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- Zero tolerance for banned words in `FINAL_REPORT.md`: "proves", "decisively", "pristine", "immaculate", "complete scientific rigor".
- Every number cited in `FINAL_REPORT.md` must be cross-referenced against `results/ledger.json`.
- The audit MUST run all global verification steps end-to-end.

---

### Objective & Scope

Execute the definitive end-to-end forensic verification suite across the entire ARN repository. Ensure all unit tests pass, numbers reconcile, secrets and personal identifiers are removed, and `FINAL_REPORT.md` provides a sober, accurate scientific summary.

---

### Exact Execution Steps

1. **Execute Complete Test Suite:**
   - Run `pytest tests/ -v`.
   - Verify all unit and integration tests (tagger, model architecture, CRF viterbi, calibration, evaluation harness) pass cleanly.
2. **Execute Full Repository Banned-Claims Grep:**
   - Run search for banned phrases:
     `git grep -i "14.2 ms\|5.55 MB\|4.21 MB\|810x\|38x\|216x\|180x\|88.0\|92.3%\|96.0%\|62.4\|tool-prefixed BIO\|dynamic registry\|beats Jev\|beats Laya\|before TypeSafe\|peer-reviewed\|published at AAAI\|SOTA"`
   - Confirm 0 matches across all tracked code, markdown, and LaTeX files.
3. **Execute Paper Numbers Audit:**
   - Run `python scripts/audit_paper_numbers.py`.
   - Confirm zero discrepancies between LaTeX tables and underlying JSON files.
4. **Execute Secrets & Personal Data Grep:**
   - Search for personal emails (`anishpathak778`), API tokens, internal development paths.
5. **Run Reproducibility Slice:**
   - Execute a single test fold evaluation (e.g. `python scripts/evaluate_fold_slice.py --fold 0`) and verify output metrics match the corresponding record in `results/ledger.json` within floating-point tolerance ($10^{-5}$).
6. **Rewrite and Finalize FINAL_REPORT.md:**
   - Restructure `FINAL_REPORT.md` into factual, defensible sections:
     - Executive Summary: Scope, architectural reality, honest baseline metrics.
     - Forensic Audit Findings: Historical errors and remediation.
     - Verified Performance: Table 1 (cluster-disjoint) and Table 2 (deployment).
     - Known Limitations & Scope Boundaries: Synthetic templates, closed registry, x86 CPU scope.
     - Public Release Artifacts: Checkpoints, ONNX models, preprint paper, launch texts.
   - Update `STATE.md` to reflect full project completion and final verified commit hash.

---

### Acceptance Criteria (Pass / Fail)

- [ ] All pytest test suites exit with code 0.
- [ ] Banned claims scan returns 0 hits across all tracked files.
- [ ] Paper numbers audit passes with 0 mismatches.
- [ ] Secrets and email scan returns 0 hits in tracked files.
- [ ] Reproducibility slice validates ledger reproducibility.
- [ ] `FINAL_REPORT.md` and `STATE.md` completely updated without promotional vocabulary.
- [ ] Git commit created with exact message: `docs: final self-audit passed, update FINAL_REPORT and STATE`.
