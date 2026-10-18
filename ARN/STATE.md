# ARN Project State Tracking

Last Updated: 2026-10-18T17:40:00+05:30
Current Branch: `arn-final`
Status: IN_PROGRESS

## Phase Progress Tracker

| Phase | Description | Status | Commit Hash | Next Action |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Orient & Environment Recording | IN_PROGRESS | pending | Finalize tracking docs & PREREGISTRATION.md |
| **1** | Split Redesign (Coverage Fix) | PENDING | - | Audit coverage; build tool-stratified grouped K-folds |
| **2** | Results Ledger & Reconciliation | PENDING | - | Initialize ledger.json; reconcile discrepancies |
| **3** | Baseline Debugging | PENDING | - | Fix TF-IDF kNN/LogReg; verify Softmax + masked Viterbi |
| **4** | Master Protocol (Main Experiment) | PENDING | - | Run 15 folds nested CV across models |
| **5** | Ablations & Curves | PENDING | - | CRF decomposition, supervision ablation, learning curves |
| **6** | Error Analysis | PENDING | - | Per-tool/cluster failures, slot error breakdown |
| **7** | Final Model Training | PENDING | - | Train on all 5k samples; package artifact & hashes |
| **8** | Deployment Protocol | PENDING | - | Vectorized Viterbi, Fast tokenizer, ONNX INT8, latency/RAM |
| **9** | Human Evaluation | PENDING | - | Check human_eval_filled.jsonl (run or park) |
| **10** | Repo Finalization | PENDING | - | Clean layout, CLI verification, README, model card |
| **11** | Paper Compilation | PENDING | - | Generate LaTeX tables & numbers; compile preprint PDF |
| **12** | Release Staging | PENDING | - | Stage ../ARN-public; prepare release checklist |
| **13** | Launch Drafts | PENDING | - | Write verified-numbers launch drafts |
| **14** | Self-Audit & Final Report | PENDING | - | Full test sweep, banned claims grep, FINAL_REPORT.md |
