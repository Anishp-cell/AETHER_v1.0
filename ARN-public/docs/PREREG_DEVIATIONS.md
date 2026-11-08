# Deviations from the pre-registration (`docs/PREREGISTRATION_v2_confirmatory.md`, commit 1fa7a50)

| # | Registered | What happened | Effect on reported numbers |
|---|---|---|---|
| 1 | Train only seeds 42, 1, 2 | One 1-epoch smoke run (seed 7) on MixATIS was executed to test the script; it read train and dev only (`--no-test`), wrote nothing to the ledger. Dev overall 40.58. | None: test not read, run not reported as a result. |
| 2 | MASSIVE en-US "planned" as a second confirmatory benchmark | Not run. | Reported as not done; no MASSIVE claim is made. |
| 3 | `--no-test` flag | Added to `scripts/v2_train_slu.py` after registration, before any registered run. | None (smoke-run guard only). |
| 4 | — | Dev and test accuracy differ strongly on MixATIS (dev overall ≈ 74–76, test ≈ 39–44). Reported as observed; no change was made after seeing test numbers. | Disclosed. |

All registered runs (2 datasets × 3 seeds) were executed with the registered hyperparameters, selection rule
(best dev overall accuracy) and metrics, and every seed is reported.
