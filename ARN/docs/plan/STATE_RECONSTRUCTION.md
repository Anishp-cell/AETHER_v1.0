# ARN State Reconstruction

**Date:** 2026-10-19T10:20:00+05:30
**Author:** Reconstruction Agent (read-only forensic audit)
**Working Branch:** `arn-final`
**Last Committed Hash:** `bd7570d` (2026-10-18 23:13:28 +0530)

---

## 1. Git Branch & Commit Summary

### Branches
| Branch | Status |
| :--- | :--- |
| `arn-final` | **Active** (HEAD). All Phase 0–14 work committed here. |
| `arn-revision` | Exists. Predecessor branch. |
| `main` | Exists. Original branch. |
| `wip-phase16-interrupted` | **DOES NOT EXIST** |

### Commit History (Phase 0–14, newest first)
| Hash | Timestamp | Phase | Message |
| :--- | :--- | :---: | :--- |
| `bd7570d` | 2026-10-18 23:13:28 | Post-14 | Record post-audit commit hash in STATE.md |
| `6bf185f` | 2026-10-18 23:13:06 | Post-14 | Harmonize manuscript tables, dual memory reporting, neutral FAR framing |
| `e583a94` | 2026-10-18 22:44:54 | 14 | Finalize STATE.md with Phase 14 commit hash |
| `67fba8f` | 2026-10-18 22:44:41 | 14 | Complete self-audit, banned claims grep, archive legacy, FINAL_REPORT |
| `492a71a` | 2026-10-18 22:40:14 | 13 | Launch materials |
| `40b812c` | 2026-10-18 22:35:31 | 12 | Release staging (../ARN-public) |
| `2bf9607` | 2026-10-18 22:33:10 | 11 | Paper compilation |
| `af4ceba` | 2026-10-18 22:25:28 | 10 | Repo finalization |
| `591b3d5` | 2026-10-18 22:19:52 | 9 | Human evaluation protocol |
| `02a2693` | 2026-10-18 22:10:54 | 8 | Deployment benchmark |
| `c534b28` | 2026-10-18 22:00:21 | 7 | Final model training |
| `1f41def` | 2026-10-18 21:56:48 | 6 | Error analysis |
| `2fa50f7` | 2026-10-18 21:52:08 | 5 | Ablations |
| `8a94a69` | 2026-10-18 21:14:29 | 4 | Master protocol (105 runs) |
| `d327d74` | 2026-10-18 18:07:48 | 3 | Baseline debugging (60 ledger records) |
| `2d1bf25` | 2026-10-18 17:58:48 | 2 | Ledger & reconciliation |
| `0204f95` | 2026-10-18 17:55:50 | 1 | Split redesign |
| `3b46e1b` | 2026-10-18 17:41:14 | 0 | State tracker, decisions, preregistration |

**Observation:** Phases 0–14 all committed on 2026-10-18 within ~5.5 hours (17:41–23:13). PREREGISTRATION.md was added in Phase 0 (`3b46e1b` at 17:41) and the master protocol ran in Phase 4 (`8a94a69` at 21:14) — git timestamps confirm protocol was locked ~3.5 hours before execution, but all within a single autonomous session. This is NOT a pre-registration in the traditional sense (days/weeks before).

### Working Tree Status
```
?? results/far_fairness_study.json          (UNTRACKED, 14928 bytes)
?? scripts/diagnose_far_fairness.py         (UNTRACKED, 2776 bytes)
?? scripts/run_far_fairness_study.py        (UNTRACKED, 21176 bytes)
?? scripts/run_oov_and_adversarial_suite.py (UNTRACKED, 24197 bytes)
```

**No staged or modified tracked files.** The working tree is clean except for 4 untracked files — all Phase 16 artifacts.

---

## 2. Interrupted Phase 16 Forensics

### 2.1 Branch Status
- `wip-phase16-interrupted` branch: **DOES NOT EXIST**
- No Phase 16 commit was made on any branch
- Phase 16 work exists ONLY as 4 untracked files in the working tree

### 2.2 Untracked Phase 16 Files

| File | Size | Status | Content |
| :--- | :--- | :--- | :--- |
| `results/far_fairness_study.json` | 14,928 B | Valid JSON | FAR retraining results: fold1 training history (10 epochs) + 15-fold CV results for 2 variants |
| `scripts/diagnose_far_fairness.py` | 2,776 B | Complete script | Diagnostic script for FAR fairness analysis |
| `scripts/run_far_fairness_study.py` | 21,176 B | Complete script | FAR fairness retraining runner with LR grid |
| `scripts/run_oov_and_adversarial_suite.py` | 24,197 B | Complete script | OOV and adversarial evaluation (unrelated to FAR fairness) |

### 2.3 FAR Fairness Study Details

The `results/far_fairness_study.json` contains:
- **`fold1_history`**: 10-epoch training history with per-epoch losses and gradient norms for `tool_emb`, `wq`, `out`, and `encoder` components. Training loss decreased from 1.54 to 0.078 but validation loss increased from 0.76 to 0.84 (clear overfitting by epoch 3).
- **`fold1_stopped_epoch`**: 10 (ran to completion)
- **`cv_results`**: Two variants, each with **15 fold results** (COMPLETE):
  - `unfrozen_far_crf_tuned`: 15 fold results with `selected_head_lr` in {0.0003, 0.001, 0.003} and `best_tau` in {0.35 to 0.50}
  - `unfrozen_far_no_crf_tuned`: 15 fold results with `selected_head_lr` in {0.0003, 0.001, 0.003} and `best_tau = 0.35` (most)

**Key Finding:** The FAR fairness retraining appears NUMERICALLY COMPLETE (all 15 folds for both variants). However:
1. Results were never committed to git
2. Results were never integrated into ledger.json
3. Results were never reflected in any tables or documents
4. No checkpoints from this run are preserved
5. Protocol: LR grid covered {3e-4, 1e-3, 3e-3} — inner-validation-only selection confirmed by the per-fold `selected_head_lr` field

### 2.4 Results JSON Validation (ALL files)

| File | Size | Valid JSON | Notes |
| :--- | :--- | :--- | :--- |
| `ablations_summary.json` | 4,308 B | VALID | |
| `argument_label_audit.json` | 4,562 B | VALID | |
| `argument_representability_audit.json` | 4,832 B | VALID | |
| `coverage_audit_new_folds.json` | 49,459 B | VALID | |
| `coverage_audit_old_folds.json` | 60,696 B | VALID | |
| `deployment_benchmark.json` | 3,341 B | VALID | |
| `environment.json` | 763 B | VALID | |
| `error_analysis.json` | 21,779 B | VALID | |
| `far_fairness_study.json` | 14,928 B | VALID | (untracked) |
| `ledger.json` | 244,535 B | VALID | |
| `master_protocol_summary.json` | 6,786 B | VALID | |
| `per_cluster_predictions.json` | 917,519 B | VALID | |

**Zero truncated, zero-byte, or invalid JSON files.**

### 2.5 Ledger Summary

- **Structure:** Dict with keys `schema_version`, `single_source_of_truth`, `description`, `metadata`, `runs`, `human_evaluation`
- **Total runs:** 170
- **Experiment ID breakdown:**
  - `p3_*`: 60 runs (15 folds x 4 baselines: first_token_prior, most_frequent, tfidf_knn, tfidf_logistic)
  - `p4_*`: 105 runs (15 folds x 7 models: frozen_arn, frozen_cls_crf, frozen_far_no_crf, frozen_joint_bert, unfrozen_arn, unfrozen_cls_crf, unfrozen_joint_bert)
  - `p5_*`: 3 runs (crf_decomposition, learning_curves, supervision_ablation)
  - `p7_*`: 1 run (final_model_training)
  - `p8_*`: 1 run (deployment_benchmark)

**No entries after Phase 8 (deployment benchmark).** The FAR fairness study results are NOT in the ledger.

### 2.6 Phase 16 Item-by-Item Status

| # | Item | Status | Evidence |
| :--- | :--- | :--- | :--- |
| 1 | FAR fairness retraining | **PARTIAL** | `results/far_fairness_study.json` exists (untracked, complete 15-fold data for 2 variants), scripts exist. NOT in ledger, NOT in any table, NOT committed. |
| 2 | Rename to Factored Attention Routing | **DONE** | grep for "Feature Attention" returns 0 hits. "Factored Attention Routing" appears in PREREGISTRATION, MATH.md, CHANGELOG, AUDIT.md, CITATION.cff. |
| 3 | Memory protocol fix (delete 4.21 MB) | **NOT DONE** | `deployment_benchmark.json` has baseline/peak/incremental RSS. Table 2 reports both Peak and Incr. RSS. **BUT "4.21 MB" still appears in 20+ locations** (STATE.md, README, MODEL_CARD, FINAL_REPORT, LINKEDIN, X_THREAD, MEDIUM_OUTLINE, FAQ, DECISIONS, CHANGELOG). |
| 4 | Trace or delete ">85% on random splits" | **DONE** | grep for ">85%", "above 85", "over 85" returns 0 hits. |
| 5 | Restore TF-IDF kNN, mark tool-only baselines N/A, baseline history | **PARTIAL** | Table 1 includes TF-IDF kNN (63.52%) and LogReg (74.24%) with span columns marked `--`. Baseline history explained in RECONCILIATION.md D-006. But no end-to-end narrative, no explicit paired tool-acc deltas. |
| 6 | One definition and source for all SDs | **NOT DONE** | `master_protocol_summary.json` reports `tool_acc_std = 12.19%`. FINAL_REPORT and MEDIUM_OUTLINE use `9.42%`. No source for 9.42%. |
| 7 | Reconcile decoder ablation with master table | **NOT DONE** | Ablation CRF strict F1 = 36.58% vs master table CLS+CRF = 43.61%. No document explains this gap. |
| 8 | Reconcile multi-tool counts | **PARTIAL** | 114 confirmed for one pair. numbers.tex says 1,132 total multi-tool. Briefing says 1,107. Need to verify 1,132. |
| 9 | Prove pre-registration by git timestamps or reword | **NOT DONE** | Same-day, same-session. Documents still say "pre-registered" without caveat. |
| 10 | Release hygiene | **NOT DONE** | Paper filenames AAAI-named, personal email in git, ARN-public inside Aether tree, author name unresolved, Mobile-Actions archived. |
| 11 | Update FINAL_REPORT, NEEDS_USER, README, MODEL_CARD | **NOT DONE** | 4.21 MB everywhere, inconsistent SDs, promotional language. |

---

## 3. Briefing vs. Repo Discrepancies

| Briefing Claim | Repo Reality | Verdict |
| :--- | :--- | :--- |
| TF-IDF LogReg tool acc 18.06% | Ledger: 74.24% mean (15 folds) | **BRIEFING WRONG** (stale pre-fix number) |
| Frozen FAR+CRF tool subset acc 16.27% | master_protocol_summary: 14.21% | **BRIEFING WRONG** |
| Fine-tuned FAR+CRF tool subset acc 15.65% | master_protocol_summary: 21.28% (unfrozen_arn) | **BRIEFING WRONG** |
| Frozen CLS+CRF tool acc 11.23% | master_protocol_summary: frozen_cls_crf = 21.56% | **BRIEFING WRONG** (naming confusion) |
| Frozen CLS+CRF joint EM 3.44% | This is frozen_arn, not frozen_cls. frozen_cls = 6.03% | **BRIEFING NAMING ERROR** |
| Fine-tuned CLS+CRF joint EM 16.91% | master_protocol_summary: 16.91% | MATCHES |
| Fine-tuned CLS+CRF tool acc 39.91% | master_protocol_summary: 39.91% | MATCHES |
| SD values 12.19/9.42/10.97 | 12.19% correct. 9.42% in FINAL_REPORT (wrong). 10.97% not found. | INCONSISTENCY CONFIRMED |
| 4.21 MB RSS not credible | Correct: peak 58.69 MB, incremental 12.37 MB, INT8 disk 4.28 MB | **4.21 MB IS WRONG** |
| Paired delta frozen FAR vs frozen CLS: +2.59 | master_protocol_summary delta = 2.59% | MATCHES |
| 114 multi-tool vs 1,107 | 114 = one pair. Total = 1,132 per numbers.tex | **1,107 is wrong; 1,132 per repo** |
| Multi-tool pair count 45 | RECONCILIATION explains: 45 was early prototype subset of 114 | RECONCILED |

---

## 4. Documents Containing Phase 16 Fix Targets

### 4.1 Files containing "4.21 MB" (must be deleted/corrected)
- `STATE.md` line 19
- `README.md` lines 9, 26
- `MODEL_CARD.md` line 78
- `FINAL_REPORT.md` lines 42, 136
- `DECISIONS.md` line 80
- `CHANGELOG.md` line 17
- `docs/launch/LINKEDIN.md` line 36
- `docs/launch/X_THREAD.md` lines 12, 80
- `docs/launch/MEDIUM_OUTLINE.md` lines 18, 90
- `docs/launch/FAQ.md` lines 4, 10, 54

### 4.2 Files with wrong SD (9.42% instead of 12.19%)
- `FINAL_REPORT.md` lines 71, 94
- `docs/launch/MEDIUM_OUTLINE.md` line 55

### 4.3 AAAI filename residue
- `docs/paper/AnonymousSubmission2027.tex`
- `docs/paper/CameraReady2027.tex`
- `docs/paper/aaai2027.bib`

### 4.4 Promotional language in FINAL_REPORT
- Line 199: "complete scientific rigor and engineering maturity"
- Line 199: "proves that non-generative..."

### 4.5 Personal email in git config
- `anishpathak778@gmail.com` — must be changed to noreply before public push

---

## 5. FAR Fairness Rerun — Salvage Assessment

### Data Completeness
Complete 15-fold results for both variants. LR grid covered {3e-4, 1e-3, 3e-3}. Inner-validation-only tuning confirmed.

### Salvage Verdict
**CONDITIONALLY SALVAGEABLE.** Requires WP-D to:
1. Verify `run_far_fairness_study.py` uses same `cv_manifest.json`
2. Verify identical seed and training protocol
3. Append to ledger.json with documented diff
4. Label as "v2 (fairness retraining)" alongside original "v1" results
5. Never mix partial and old results in one table

### Preliminary Results Summary
`unfrozen_far_crf_tuned` 15-fold averages (approximate):
- Tool subset acc: ~19.5% (vs original unfrozen_arn 21.3%)
- Strict span F1: ~34.2% (vs original 29.6%)
- Joint EM: ~7.1% (vs original 5.9%)
- Very high variance (some folds at 0% joint EM)

---

## 6. STATE.md Accuracy
STATE.md reflects committed work accurately (Phases 0-14 DONE) but:
1. Contains erroneous "4.21MB RSS"
2. Does not acknowledge unresolved integrity issues
3. `Status: DONE` is premature given known defects

---

## 7. Baseline & Ledger Integrity Audit (Forensics for WP-M)

### 7.1 The TF-IDF Logistic Regression Discrepancy & Routing Anomaly
- **Briefing claim:** TF-IDF LogReg tool accuracy was claimed to be `18.06%`.
- **Repo ground truth:** `results/ledger.json` records a 15-fold average of **`74.24%`** ($\pm 9.42\%$ in ledger, $12.19\%$ in `master_protocol_summary.json`).
- **Core Scientific Issue:** If TF-IDF LogReg legitimately achieves 74.24% tool subset accuracy on cluster-disjoint splits while fine-tuned CLS+CRF achieves 39.91%, a bag-of-words linear model is outperforming the 4.39M parameter neural encoder on tool routing by +34.33 percentage points.
- **Audit Mandate for WP-M:**
  1. Verify strict fold isolation: Ensure `TfidfVectorizer` vocabulary and IDF weights were learned strictly on training indices without test fold leakage.
  2. Verify threshold calibration: Multi-label thresholds $\tau_k$ must be chosen on inner validation folds, not test folds.
  3. Execute label-shuffle test: Randomly permuting training labels must cause test accuracy to collapse to majority chance (~7.90%).
  4. Compare with Stage 2b reports: Stage 2b reported kNN 53% and LogReg 34.5%. Reconcile the jump to 74.24% via Phase 3 (D-006) sublinear TF and n-gram tuning.
  5. Evaluate Hybrid Router-Tagger: Measure Joint Exact Match of TF-IDF LogReg router (74.24%) + frozen BERT-tiny CRF tagger (34.34%).
  6. Measure size (MB) and CPU latency (ms) for all baselines.

### 7.2 Three-Column Master Verification Gate
Before finalizing any results, every metric across all 10 models must be verified across three files:
1. Pre-context-loss `FINAL_REPORT.md`
2. `results/master_protocol_summary.json`
3. Direct computation from `results/ledger.json` runs

`results/master_protocol_summary.json` (compiled deterministically from `results/ledger.json`) is the sole authoritative source of truth. All downstream tables and macros must derive strictly from it.

