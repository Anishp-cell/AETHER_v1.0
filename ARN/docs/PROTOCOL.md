# ARN Master Evaluation Protocol (Phase 1)

## 1. Executive Summary & Rationale: The Coverage Confound

In Stage 2d, an audit of the initial 5-fold cross-validation splits revealed a severe structural flaw termed the **Coverage Confound**:
- The legacy cluster-assignment algorithm sorted canonical semantic clusters by sample size in descending order and greedily assigned each cluster to the fold with the minimum total sample count.
- Because rare tools (e.g., `teach_new_skill`, `open_app_and_type`, `read_specific_url`, `open_url`) only possess between 1 and 4 semantic clusters across the dataset, this greedy binning accidentally concentrated all clusters of specific tools into single folds.
- Consequently, training sets were completely starved of training examples for those tools:
  - **Old Fold 5**: Missing `open_app_and_type`, `open_url`, and `teach_new_skill` from training. Validation samples containing these tools represented **33.25%** of the validation set (133 of 400 samples). The model was evaluated on tools it had never seen during training.
  - **Old Fold 2 & Fold 3**: Missing `read_specific_url` from training (**7.90%** of validation samples, 64 of 810 samples).
  - Out-of-fold metrics conflated semantic generalization on known tools with zero-shot routing failures on completely unobserved tools.

To eliminate this confound and establish an unimpeachable empirical benchmark, Phase 1 redesigns the evaluation protocol from first principles.

---

## 2. Side-by-Side Comparison: Old vs. New Fold Design

| Property | Legacy Stage 2d Folds | Phase 1 Tool-Stratified Folds (Frozen) |
| :--- | :--- | :--- |
| **Data Universe** | 4,000 samples (1,000 burned test split excluded) | **All 5,000 samples** (burned test split re-merged into pool) |
| **Number of Folds** | 5 outer folds (1 shuffle) | **15 outer folds** (5 folds $\times$ 3 shuffles: seeds 42, 100, 2024) |
| **Cluster Grouping** | Canonical semantic frame ($\ge 0.80$ token sim) | Canonical semantic frame ($\ge 0.80$ token sim) |
| **Cluster Allocation** | Greedy sample-count binning | **Tool-stratified cluster allocation** (rare tools allocated first) |
| **Training Tool Coverage** | Incomplete: Fold 5 missing 3 tools; Fold 2/3 missing 1 tool | **100% complete**: All 18 tools have $\ge 1$ cluster in outer & inner train |
| **Unseen Tools in Val** | Up to **33.25%** of validation samples | **0.00%** (zero validation samples contain unseen tools) |
| **Special Case: `open_url`** | Leaked into val in Fold 5 (where it was unseen in train) | **Train-only across all 15 folds** (1 cluster, 37 samples; never in val) |
| **Near-Duplicate Check** | Partial | **Max skeleton edit sim $< 0.80$** verified across all 15 folds (range 0.71–0.79) |
| **Inner Validation** | Heuristic / unstructured | **Nested 80/20 cluster-disjoint inner split** per outer fold |
| **Ledger & Manifest** | Fragmented across stage JSONs | Frozen `ARN/data/splits/cv_manifest.json` with SHA-256 hashes |

---

## 3. Structural Constraints & Guarantees

1. **Group-Disjoint Independence**:
   - Semantic clusters are never split across train and validation sets, nor across inner train and inner validation sets.
   - For any outer fold $k$, $\text{Clusters}(\text{Train}_k) \cap \text{Clusters}(\text{Val}_k) = \emptyset$.
   - For inner tuning splits, $\text{Clusters}(\text{InnerTrain}_k) \cap \text{Clusters}(\text{InnerVal}_k) = \emptyset$.

2. **Near-Duplicate Absence**:
   - The token-edit similarity between every validation skeleton and every training skeleton must satisfy $\text{sim}(s_{\text{val}}, s_{\text{train}}) < 0.80$.
   - In the frozen manifest, maximum similarity across all 15 folds is bounded between 0.7143 and 0.7857.

3. **Guaranteed Tool Stratification**:
   - For every tool $t \in \mathcal{T}$ (where $|\mathcal{T}| = 18$), $\text{Count}_{\text{clusters}}(t, \text{Train}_k) \ge 1$ and $\text{Count}_{\text{clusters}}(t, \text{InnerTrain}_k) \ge 1$.
   - `open_url` has exactly 1 cluster (37 samples). To prevent leakage under group-disjoint partitioning, it is assigned to training in all 15 folds and excluded from validation. Exactly 17 tools are evaluated out-of-fold.

4. **Nested Inner Validation Tuning**:
   - All hyperparameter selection (learning rate, loss weights $\lambda$, decision threshold $\tau$, early stopping) is conducted strictly on the inner validation split.
   - Outer validation sets are evaluated exactly once per model using the inner-selected configuration.

---

## 4. Manifest Provenance & Audit Artifacts

- **Dataset File**: `ARN/data/raw/aether_orchestrator_dataset.jsonl`
  - SHA-256: `d2f25d93a7123226f5f849e7f72391a2fd8b93c1b04c79fa2f5a2c8b8d5a95b9`
  - Total samples: 5,000 | Total canonical clusters: 253
- **CV Manifest**: `ARN/data/splits/cv_manifest.json`
  - Contains indices, sample counts, and SHA-256 hashes for all 15 outer and inner folds.
- **Coverage Audit**:
  - Legacy folds audit: `ARN/results/coverage_audit_old_folds.json`
  - New folds audit: `ARN/results/coverage_audit_new_folds.json`
- **Schema Argument Label Audit**:
  - Total arguments: 8,748 across 5,000 samples.
  - Verbatim arguments in utterance: 8,292 (94.79%).
  - Non-verbatim arguments: 456 (5.21%) (discrete enum actions such as `action_type="shell"`, `action="set_volume"`).
  - Unmapped schema argument keys: 0.
