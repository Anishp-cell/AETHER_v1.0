# Action Items Requiring User Input (NEEDS_USER.md)

This file contains items that only the user can provide or confirm. The autonomous runner does not block on these items; it logs them here, applies reasonable, safe fallbacks, and continues.

---

## Pending Action Items

1. **Human Evaluation Test Set (`data/human_eval/human_eval_filled.jsonl`)**:
   - *Status:* Pending user annotation.
   - *Details:* The template is sealed at `data/human_eval_template.jsonl`. If `human_eval_filled.jsonl` is not present, Phase 9 will write `docs/HUMAN_EVAL_PENDING.md`, validate the evaluation harness on dummy/schema checks, and mark human evaluation as pending in the paper and final report.

2. **Public Release Credentials & Identity**:
   - *Status:* Pending public launch.
   - *Details:* Provide GitHub username, noreply email, and Hugging Face namespace for staging the final release in `../ARN-public`.

3. **Legacy PyPI Package Management**:
   - *Status:* Action required before promotion.
   - *Details:* The old `arn-orchestrator` release on PyPI contains legacy checkpoints trained prior to the `[CLS]/[SEP]` tokenizer fix. See `docs/PYPI_NOTE.md` for instructions on yanking or replacing the release.

4. **Biographical Narrative Confirmation**:
   - *Status:* To be reviewed prior to social posting.
   - *Details:* Verify dates (started January 2026, submitted July 2026) and estimated effort (approx. 5 months, 14–15 hours/day) in `docs/launch/`.
