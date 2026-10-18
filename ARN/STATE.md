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
| **6** | Error Analysis | DONE | Pending commit | Per-tool F1, 8,506-span error taxonomy (42.7% boundary shifts, 1.5% label errors), hardest clusters |
| **7** | Final Model Training | IN_PROGRESS | - | Train unfrozen_cls_crf on all 5k samples; package artifact & hashes |
| **8** | Deployment Protocol | PENDING | - | Vectorized Viterbi, Fast tokenizer, ONNX INT8, latency/RAM |
| **9** | Human Evaluation | PENDING | - | Check human_eval_filled.jsonl (run or park) |
| **10** | Repo Finalization | PENDING | - | Clean layout, CLI verification, README, model card |
| **11** | Paper Compilation | PENDING | - | Generate LaTeX tables & numbers; compile preprint PDF |
| **12** | Release Staging | PENDING | - | Stage ../ARN-public; prepare release checklist |
| **13** | Launch Drafts | PENDING | - | Write verified-numbers launch drafts |
| **14** | Self-Audit & Final Report | PENDING | - | Full test sweep, banned claims grep, FINAL_REPORT.md |
