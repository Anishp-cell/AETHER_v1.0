# ARN Architecture & Protocol Decisions Log

This document records all non-trivial decisions made during the ARN project revision and final execution, including alternatives considered, empirical rationale, and constraints enforced.

---

## Decision Index

### D-001: Branch Strategy and Clean State
- **Date:** 2026-10-18
- **Context:** User requested end-to-end execution of Phases 0 through 14 without stopping for approval, with commits to branch `arn-final`.
- **Decision:** Created branch `arn-final` from the clean working state on `arn-revision`. All changes confined strictly to `ARN/` (and staged `../ARN-public` in Phase 12).
- **Alternative Rejected:** Working directly on master or force-pushing (violates Hard Rule 2).

### D-002: Automatic Hardware Environment Capture
- **Date:** 2026-10-18
- **Context:** Phase 0 requires recording exact hardware specs into `results/environment.json` without prompting the user.
- **Decision:** Executed `record_environment.py` gathering psutil, torch, and platform attributes. Identified exact CPU as `13th Gen Intel(R) Core(TM) i7-13700HX` (16 physical cores, 24 logical processors) with 15.71 GB RAM and RTX 5050 Laptop GPU under Windows 10/11 64-bit.
- **Alternative Rejected:** Hand-typing or approximating CPU model (violates Hard Rule 3 and Item 10).

### D-003: Pre-registration of Decision Rules (Phase 2 Specifications)
- **Date:** 2026-10-18
- **Context:** Hard pre-registration rules must be fixed before Phase 4 experiments to eliminate post-hoc p-hacking and cherry-picking.
- **Decision:** Established `docs/PREREGISTRATION.md` specifying:
  1. Unit of independence = semantic cluster ($B \ge 2000$ cluster bootstrap).
  2. Primary metrics: Tool Subset Accuracy (P1), Joint Exact Match (P2), Strict Span F1 (P3).
  3. Minimum delta for claiming a difference: 95% CI excludes 0 AND $|\Delta| \ge 2.0\%$.
  4. Model selection: Highest mean Joint EM; simpler model chosen if not reliably worse.
  5. FAR described as a contribution ONLY if matching CLS-head is beaten by the rule; otherwise reported as evaluated alternative.
  6. Outcome A (fine-tuning supported) vs. Outcome B (otherwise) determining paper narrative.
- **Alternative Rejected:** Selecting model by tool accuracy alone or evaluating significance at the utterance level.
