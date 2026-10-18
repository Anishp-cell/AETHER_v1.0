# Stage 1 Test Set Protocol & Status Documentation

**Date**: October 18, 2026  
**Status**: BURNED (COMPROMISED VIA POOLED CROSS-VALIDATION) — PERMANENTLY RETIRED  
**Target File**: `ARN/data/splits/group_disjoint_test.jsonl`  
**File SHA-256**: `1997d91cbcfdbbfa406cb32cf3d5a230553bb182f5b66d8b9e6e492aeafcbb7c`  

---

## 1. Formal Designation & Compromise Reconciliation

### Test Set Formally Declared Burned
During the Stage 1b diagnostics suite, the cross-validation runner (`stage1b_run_all.py`) loaded the entire 5,000-sample raw dataset (`aether_orchestrator_dataset.jsonl`), partitioning it into 5 folds of 1,000 samples each. As a consequence:
1. **Exposure**: All 573 test samples across all 22 test clusters were absorbed into the cross-validation training and validation splits.
2. **Loss of Holdout Status**: Because models evaluated during Stage 1b were trained on or validated against subsets containing these samples, `group_disjoint_test.jsonl` is **no longer a clean holdout**.
3. **Formal Retirement**: `group_disjoint_test.jsonl` is permanently retired and declared **BURNED**. It may not serve as the final unbiased holdout benchmark for Stage 3 or publication claims.
4. **Replacement Benchmark**: Final out-of-distribution evaluation will be conducted against the hand-authored, sealed Human Evaluation Benchmark (`ARN/data/human_eval_template.jsonl`, 250 curated non-synthetic utterances), authored with zero LLM generation.

---

## 2. Cluster Count Correction: 209 Training Clusters vs 253 Total Corpus Clusters

Earlier documentation referred interchangeably to "253 clusters" across cross-validation. This is formally reconciled and corrected:
- **Total Corpus Clusters**: Exactly **253 canonical semantic clusters** exist across the full 5,000-sample dataset.
- **Holdout Allocation**:
  - Validation Split: 22 clusters (501 samples)
  - Retired Test Split: 22 clusters (573 samples)
- **Active Training Split (`group_disjoint_train.jsonl`)**: Exactly **209 clusters** (3,926 samples).
- **Stage 2 Protocol Reconciliation**:
  - The Stage 2 comparative evaluation was conducted strictly across the **209 clusters** of `group_disjoint_train.jsonl` using 15 folds (3 shuffles $\times$ 5 folds, 785–786 samples per fold).
  - All cluster-level bootstrap resampling procedures resample from these **209 training clusters** on pooled out-of-fold predictions.

---

## 3. Strict Guardrails for Remaining Diagnostics & Stage 3

1. **No Access to Burned Splits**: No script or runner in Stage 2b or Stage 3 may load, evaluate, or optimize against `group_disjoint_test.jsonl`.
2. **Pristine Holdout Isolation**: The newly authored human evaluation benchmark stays sealed and completely isolated until final post-Stage 3 testing.
3. **Empirical Grounding**: All reporting must be strictly empirical with exact integer rates ($k$ of $N$) and cluster bootstrap confidence intervals. Speculative narratives regarding convergence or zero-shot routing are strictly excised.
