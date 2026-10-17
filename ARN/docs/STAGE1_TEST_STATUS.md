# Stage 1 Test Set Protocol & Status Documentation

**Date**: October 17, 2026  
**Status**: EXPLORATORY (SEEN ONCE) — STRICTLY FROZEN  
**Target File**: `ARN/data/splits/group_disjoint_test.jsonl`  
**File SHA-256**: `1997d91cbcfdbbfa406cb32cf3d5a230553bb182f5b66d8b9e6e492aeafcbb7c`  

---

## 1. Formal Designation

The test split `group_disjoint_test.jsonl` (containing 573 samples across 22 clusters / 30 skeleton frames) was evaluated exactly once at the completion of Stage 1 to establish initial performance under group-disjoint semantic clustering.

Following Stage 1 evaluation, the test split is formally classified as:
> **"Exploratory, seen once"**

## 2. Protocol and Execution Guardrails

To prevent test-set leakage, data dredging, and circular tuning, the following constraints are permanently enforced:

1. **No Decisions Based on Test Results**: No architectural modifications, loss weight adjustments, hyperparameter selections, or decision threshold tunings may utilize test set feedback.
2. **Access Prohibition**: No script, module, or test runner executed during Stage 1b (or any subsequent diagnostic phase) is permitted to load, parse, or evaluate `group_disjoint_test.jsonl`.
3. **Validation and CV Isolation**: All Stage 1b diagnostics, loss verifications, learning rate sweeps, decision threshold calibrations, span error analyses, and generalization estimates must be conducted exclusively on `group_disjoint_val.jsonl` or via repeated grouped cross-validation across the remaining clusters.
4. **Re-Evaluation Condition**: The test split shall remain frozen and unread until formal instructions are received to conduct the final comparative evaluation against baselines.
