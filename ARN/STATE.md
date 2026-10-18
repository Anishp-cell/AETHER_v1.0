# ARN Project State Tracking

Last Updated: 2026-10-18T17:40:00+05:30
Current Branch: `arn-final`
Status: IN_PROGRESS

## Phase Progress Tracker

| Phase | Description | Status | Commit Hash | Next Action |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Orient & Environment Recording | DONE | `3b46e1b` | Recorded environment.json; created tracking docs |
| **1** | Split Redesign (Coverage Fix) | DONE | `0204f95` | Generated cv_manifest.json, audits, tests, PROTOCOL.md |
| **2** | Results Ledger & Reconciliation | DONE | `2d1bf25` | Initialized ledger.json; reconciled 8 historical discrepancies |
| **3** | Baseline Debugging | DONE | `d327d74` | Debugged TF-IDF kNN/LogReg, tested decoders & grad flow, recorded 60 ledger runs |
| **4** | Master Protocol (Main Experiment) | DONE | `8a94a69` | 105 runs across 15 folds completed; unfrozen_cls_crf leads with 16.91% Joint EM (Outcome A) |
| **5** | Ablations & Curves | DONE | `2fa50f7` | CRF decomp, supervision ablation, learning curves, multi-tool 114 count, argument audit |
| **6** | Error Analysis | DONE | `1f41def` | Per-tool F1, 8,506-span error taxonomy (42.7% boundary shifts, 1.5% label errors), hardest clusters |
| **7** | Final Model Training | DONE | `c534b28` | Trained unfrozen_cls_crf on all 5k samples (17.58MB model.pt, SHA-256 verified in manifest.json); docs/PYPI_NOTE.md |
| **8** | Deployment Protocol | DONE | `02a2693` | Fast NumPy Viterbi (10k test verified), ONNX FP32 (16.75MB), Dynamic INT8 (4.28MB), 1.39ms (1T) / 0.75ms (4T) latency, 4.21MB RSS |
| **9** | Human Evaluation | DONE | `591b3d5` | Sealed holdout at data/human_eval_template.jsonl; docs/HUMAN_EVAL_PENDING.md; scripts/run_human_eval.py verified |
| **10** | Repo Finalization | DONE | `af4ceba` | Upgraded CLI, truthful README with verified numbers, MODEL_CARD.md, DATA_STATEMENT.md, CHANGELOG.md |
| **11** | Paper Compilation | DONE | `2bf9607` | Compiled LaTeX tables 1-4, generated numbers.tex, rewritten manuscripts, audit passed |
| **12** | Release Staging | DONE | Pending commit | Staged ../ARN-public clean git repo on main (7b28cc8); verified RELEASE_CHECKLIST.md |
| **13** | Launch Drafts | IN_PROGRESS | - | Write verified-numbers launch drafts |
| **14** | Self-Audit & Final Report | PENDING | - | Full test sweep, banned claims grep, FINAL_REPORT.md |
