# Work Package Prompt: WP-01 (WP-A) — State Verification & Interrupted Phase 16 Salvage Decision

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Small (1 session, ~30 min)  
**Compute Needed:** Minimal CPU (read-only verification)  
**Prerequisites:** None  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-A)
3. `ARN/docs/plan/STATE_RECONSTRUCTION.md` (Sections 1, 2, 5)
4. `ARN/DECISIONS.md`
5. `ARN/results/ledger.json` (metadata and structure)
6. `ARN/scripts/run_far_fairness_study.py` (check split loading and seed)
7. `ARN/results/far_fairness_study.json` (check fold structure and completeness)

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/` (never touch the rest of `AETHER_V1.0`).
- No fabrication: every metric must originate from `results/ledger.json` via scripts.
- No modifying core results or running unapproved commands.
- Banned claims list must remain strictly respected.

---

### Objective & Scope

Determine with proof whether the interrupted Phase 16 artifacts can be safely salvaged or if any part must be re-run from scratch. Specifically:
1. Verify the findings in `STATE_RECONSTRUCTION.md` regarding `results/far_fairness_study.json` and `scripts/run_far_fairness_study.py`.
2. Verify whether `run_far_fairness_study.py` adheres to the locked protocol:
   - Does it load splits strictly from `data/splits/cv_manifest.json`?
   - Does it perform hyperparameter (head LR and threshold $\tau$) tuning strictly on inner validation folds without test-set leakage?
   - Does it handle seeds deterministically?
3. Document Decision D-014 in `DECISIONS.md` on whether `results/far_fairness_study.json` is accepted for integration in WP-D or marked for complete rerun.
4. Record the interrupted Phase 16 status in `STATE.md`.

---

### Exact Execution Steps

1. **Verify Git & Working Tree:**
   Run read-only git checks (`git status`, `git log -n 5`) to confirm HEAD is `bd7570d` and untracked files are limited to the expected Phase 16 artifacts.
2. **Inspect FAR Retraining Script:**
   Inspect `scripts/run_far_fairness_study.py` for:
   - Data loading: Ensure `cv_manifest.json` is used and cluster-disjoint boundaries are preserved.
   - Evaluation: Ensure threshold selection $\tau$ is chosen on validation folds, not test folds.
   - Model definition: Verify architecture matches `unfrozen_arn` (BERT-tiny + FAR + CRF).
3. **Inspect FAR Retraining JSON:**
   Inspect `results/far_fairness_study.json` to verify:
   - 15 outer folds are present for both `unfrozen_far_crf_tuned` and `unfrozen_far_no_crf_tuned`.
   - All expected metrics (`joint_exact_match`, `tool_subset_acc`, `strict_span_f1`, `lenient_span_f1`) are populated.
4. **Formulate Salvage Decision (D-014):**
   - If protocol matches perfectly: Recommend SALVAGE with explicit requirements for WP-D (append-only ledger diff, separate "v2" labeling).
   - If leakage or mismatch is found: Recommend REDO with clear documentation of the flaw.
5. **Update Project Tracking:**
   - Append Decision `D-014` to `DECISIONS.md`.
   - Update `STATE.md` to reflect Phase 15/16 status and next steps.

---

### Acceptance Criteria (Pass / Fail)

- [ ] `DECISIONS.md` contains D-014 with full rationale and comparison table.
- [ ] `STATE.md` updated with accurate Phase 16 audit status.
- [ ] No tracked model code or existing ledger entries modified.
- [ ] Git commit created with exact message: `docs: verify Phase 16 state and document salvage decision`.
