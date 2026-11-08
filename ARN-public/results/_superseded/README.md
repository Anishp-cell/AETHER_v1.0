# Superseded Evaluation and Intermediate Results Archive

This directory retains all raw outputs from early exploration, diagnostic runs, and previous stage evaluations (Stage 1 through Stage 2d) prior to the Phase 1 Master Protocol freeze.

## Why These Results Are Superseded

1. **The Coverage Confound**:
   Earlier 5-fold cross-validation splits used greedy descending binning, which inadvertently caused rare tools (e.g. `teach_new_skill`, `open_app_and_type`, `read_specific_url`, `open_url`) to be missing from the training sets of certain folds (e.g. Old Fold 5 missed 3 tools affecting 33.25% of its validation set). This conflated router generalization with evaluating on tools completely omitted from training.

2. **Burned External Split & Old 4k Folds**:
   Earlier runs excluded 1,000 samples (`group_disjoint_test.jsonl`). Under the Phase 1 protocol, that burned split is merged back into the 5,000-sample pool for balanced 15-fold cross-validation.

3. **Single Source of Truth**:
   All active paper tables, LaTeX macros, figures, README metrics, and model selection rules now read strictly from `ARN/results/ledger.json`.

Nothing has been deleted; all files are retained here for full auditability, reproducibility, and historical tracing.
