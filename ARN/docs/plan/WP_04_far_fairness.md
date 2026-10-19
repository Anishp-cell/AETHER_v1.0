# Work Package Prompt: WP-04 (WP-D) — FAR Fairness Retraining & Diagnostic Integration

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Large (1 session, ~1–3 hours depending on salvage vs rerun)  
**Compute Needed:** GPU required if re-running (RTX 5050); CPU only if salvaging existing study  
**Prerequisites:** WP-01 (WP-A) and WP-13 (WP-M) completed (with D-014 decision)  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-D)
3. `ARN/DECISIONS.md` (Check decision D-014 from WP-A and baseline audit from WP-M)
4. `ARN/scripts/run_far_fairness_study.py`
5. `ARN/results/far_fairness_study.json`
6. `ARN/data/splits/cv_manifest.json`
7. `ARN/results/ledger.json`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- STRICT SALVAGE RULE: The untracked Phase 16 FAR results (`results/far_fairness_study.json`) are usable ONLY IF every single one of the 15 folds, seeds, and variants completed under the identical protocol (same `cv_manifest.json` cluster-disjoint splits, identical inner-validation-only hyperparameter tuning, identical loss weighting). If any fold, seed, or variant is missing, truncated, or drifted in protocol, discard and re-run from scratch.
- NEVER MIX RESULTS: Never mix partial, salvage, or new FAR results with original v1 results in any single table row or aggregate summary. Always keep original baseline labeled "Fine-Tuned FAR+CRF (v1, pre-fix)" and new results labeled "Fine-Tuned FAR+CRF (v2, fairness-tuned)".
- Never silently overwrite `results/ledger.json`; ledger updates must be strictly append-only with documented diff.
- Tuning MUST be restricted to inner validation folds; never select thresholds $\tau$ or learning rates on test folds.
- Bootstrap confidence intervals must be cluster-level ($B=2000$).

---

### Objective & Scope

Fairly evaluate the Factored Attention Routing (FAR) head under an expanded learning rate grid $\{3\times 10^{-4}, 1\times 10^{-3}, 3\times 10^{-3}\}$ with inner-validation threshold tuning. Integrate the resulting metrics, training loss curves, and gradient diagnostics into the ledger and paper.

---

### Exact Execution Steps

1. **Verify Strict Protocol Identity (Salvage Gate):**
   - Check `results/far_fairness_study.json`:
     - Confirm all 15 outer folds are complete for both `unfrozen_far_crf_tuned` and `unfrozen_far_no_crf_tuned`.
     - Verify `run_far_fairness_study.py` loads splits strictly from `data/splits/cv_manifest.json`.
     - Verify inner-validation tuning: confirm `selected_head_lr` and `best_tau` were chosen solely on inner-val performance.
     - Confirm random seeds and model definitions match Phase 4 `unfrozen_arn`.
     - If ANY requirement fails: reject salvage and trigger Step 2.
2. **Execute Full Retraining (If Salvage Gate Fails):**
   - Execute `python scripts/run_far_fairness_study.py --manifest data/splits/cv_manifest.json --seed 42` across all 15 folds.
3. **Generate Diagnostics & Visualizations:**
   - Execute or adapt `scripts/diagnose_far_fairness.py` to produce:
     - `docs/figures/far_loss_curves.png`: Training vs validation loss over 10 epochs.
     - `docs/figures/far_gradient_norms.png`: Gradient norm tracking across `tool_emb`, `wq`, and encoder layers.
4. **Append to Ledger (Strictly Append-Only):**
   - Ingest the 15 fold runs for each variant into `results/ledger.json` under experiment IDs:
     - `p16_far_crf_tuned_fold_<k>`
     - `p16_far_no_crf_tuned_fold_<k>`
   - Document git commit hash and timestamp in the run metadata.
5. **Update Aggregates & Paper Tables:**
   - Update `results/master_protocol_summary.json` to include the tuned FAR variants.
   - Update `docs/paper/tables/table1_main_results.tex` to display both:
     - Fine-Tuned FAR+CRF (v1, pre-fix)
     - Fine-Tuned FAR+CRF (v2, fairness-tuned)
   - Regenerate LaTeX numbers macros: `python scripts/generate_paper_numbers.py`.

---

### Acceptance Criteria (Pass / Fail)

- [ ] All 15 folds recorded in `results/ledger.json` without modifying any existing `p1_` through `p8_` records.
- [ ] `docs/figures/far_loss_curves.png` and `docs/figures/far_gradient_norms.png` generated and verified.
- [ ] Table 1 clearly distinguishes between v1 and v2 FAR runs.
- [ ] Paired cluster bootstrap confidence interval computed for v2 FAR vs CLS+CRF.
- [ ] Git commit created with exact message: `feat: integrate FAR fairness retraining results (v2) with documented ledger diff`.
