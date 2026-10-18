# Pre-Registration of Experimental Protocol & Decision Rules

**Registration Timestamp:** 2026-10-18T17:40:00+05:30  
**Status:** LOCKED (Must not be altered during or after Phase 4 execution)

---

## 1. Unit of Statistical Independence
- The **semantic cluster** is the sole unit of independence.
- Samples belonging to the same semantic cluster share lexical frames, slot value templates, and structural token n-grams, violating sample-level i.i.d. assumptions.
- **Confidence Intervals (CIs):** All 95% confidence intervals are computed via **non-parametric cluster bootstrap** ($B \ge 2000$ resamples) resampled with replacement over cluster IDs.
- Outer-fold standard deviations ($SD$) across the 15 outer folds are reported alongside bootstrap CIs.
- Sample-level or shuffle-repeat level independence is strictly forbidden.

---

## 2. Evaluation Metrics & Hierarchy
Primary evaluation metrics are evaluated separately on:
1. **Seen-tool samples:** Samples whose constituent tool(s) were present in the training partition of that fold.
2. **All samples:** The complete out-of-fold outer validation set.

### Primary Metrics
- **P1: Tool Subset Accuracy (Exact Match)** — $\mathbb{I}(\hat{\mathbf{y}}_{\text{tool}} = \mathbf{y}_{\text{tool}})$. Binary vector match across all 18 tools.
- **P2: Joint Exact Match (EM)** — $\mathbb{I}(\hat{\mathbf{y}}_{\text{tool}} = \mathbf{y}_{\text{tool}} \land \hat{\mathbf{s}} = \mathbf{s})$. Exact tool vector match AND exact token-level strict span match.
- **P3: Strict Span F1** — Exact match of entity boundary $[start, end)$ and slot label, computed micro-averaged over all true and predicted spans.

### Secondary Metrics
- Tool Micro F1 and Macro F1.
- Lenient Span F1 (boundary overlap $\ge 1$ token with correct slot label).
- Token-level classification accuracy and tag macro F1.

---

## 3. Threshold for Claiming Statistical Differences
A performance difference between two architectures $M_A$ and $M_B$ is claimed if and only if:
1. The **95% cluster-bootstrap confidence interval** for the paired per-cluster difference ($\Delta = M_A - M_B$) strictly **excludes 0.0%**; AND
2. The absolute mean difference is at least **2.0 percentage points** ($|\bar{\Delta}| \ge 2.0\%$).

If a CI spans zero or $|\bar{\Delta}| < 2.0\%$, the report and paper must explicitly state: **"no reliable difference"**.

---

## 4. Pre-Registered Final Model Selection Rule
Implemented deterministically in `scripts/select_model.py`:
- **Candidate Set:** `[frozen_arn, frozen_cls_crf, unfrozen_cls_crf, unfrozen_arn, unfrozen_jointbert]`
- **Selection Criterion:**
  1. Identify candidate with the highest mean **Joint Exact Match (P2)** on all validation samples.
  2. If a simpler model (defined by fewer trainable parameters or significantly lower deployment latency, e.g., frozen trunk vs. unfrozen trunk, or linear head vs. attention routing) is **not reliably worse** than the top model by the 2.0-point / CI-zero exclusion rule above, select the **simpler model**.
  3. Otherwise, select the top-performing candidate.

---

## 5. Pre-Registered Framing for FAR (Factored Attention Routing)
- FAR is described as a **novel architectural contribution** in the abstract and paper title **ONLY IF** ARN outperforms the matching CLS-head model (under the identical trunk freeze state) on **P1 (Tool Accuracy)** or **P2 (Joint EM)** by the statistical claim rule above.
- If ARN does not reliably beat CLS-head (or is matched/exceeded by CLS-head), FAR must be documented neutrally as an **evaluated architectural alternative**, reporting its exact empirical measurements without exaggerated superiority claims.

---

## 6. Pre-Registered Outcome Branching
- **Outcome A (Fine-Tuning Superiority):**
  - Condition: Best unfrozen model reliably outperforms Frozen ARN on P1 and P2 by the 2.0% + CI rule.
  - Paper & Launch Draft Focus: Joint fine-tuning of compact 4.4M-parameter encoders delivers significant generalization gains on complex composition frames compared to frozen representations.
- **Outcome B (Frozen Efficiency / Minimal Fine-Tuning Gain):**
  - Condition: Unfrozen models do not reliably beat Frozen ARN by the 2.0% + CI rule.
  - Paper & Launch Draft Focus: Frozen-trunk routing offers extreme parameter efficiency and zero trunk fine-tuning overhead with competitive accuracy.

Both manuscript variants are prepared, but **only the outcome directly supported by the ledger** will be included in the final release package.

---

## 7. Multiple Comparisons Policy
- All pairwise model comparisons evaluated will be enumerated in full.
- No post-hoc pruning or selective reporting of winning splits or subsets is permitted.
