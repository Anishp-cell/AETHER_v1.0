# Work Package Prompt: WP-02 (WP-B) — Documentation Integrity Fixes

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Medium (1 session, ~60 min)  
**Compute Needed:** CPU only (text search and replacement)  
**Prerequisites:** WP-01 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-B)
3. `ARN/docs/plan/STATE_RECONSTRUCTION.md` (Section 4: Inventory of Fix Targets)
4. Target documents:
   - `ARN/STATE.md`
   - `ARN/README.md`
   - `ARN/MODEL_CARD.md`
   - `ARN/FINAL_REPORT.md`
   - `ARN/DECISIONS.md`
   - `ARN/CHANGELOG.md`
   - `ARN/CITATION.cff`
   - `ARN/docs/launch/LINKEDIN.md`
   - `ARN/docs/launch/X_THREAD.md`
   - `ARN/docs/launch/MEDIUM_OUTLINE.md`
   - `ARN/docs/launch/FAQ.md`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- No fabrication: every metric must match `master_protocol_summary.json` and `deployment_benchmark.json`.
- Zero banned claims (specifically: delete "4.21 MB" RSS everywhere; report peak 58.69 MB and incremental 12.37 MB INT8).
- One definition and source for all standard deviations: `tool_acc_std = 12.19%`.

---

### Objective & Scope

Resolve all outstanding documentation integrity issues across markdown, LaTeX, YAML, and launch texts without requiring recomputation.

---

### Exact Execution Steps

1. **Eradicate "4.21 MB" Across All Files:**
   Replace the invalid "4.21 MB RSS" metric with accurate measurements from `results/deployment_benchmark.json`:
   - ONNX INT8 Peak RSS: 58.69 MB (Incremental RSS: 12.37 MB).
   - ONNX FP32 Peak RSS: 70.75 MB (Incremental RSS: 24.21 MB).
   - PyTorch FP32 Peak RSS: 760.69 MB (Incremental RSS: 81.49 MB).
   - Disk footprint: 4.28 MB (INT8 ONNX model weight file).
   - Apply fixes to: `STATE.md`, `README.md`, `MODEL_CARD.md`, `FINAL_REPORT.md`, `DECISIONS.md`, `CHANGELOG.md`, `LINKEDIN.md`, `X_THREAD.md`, `MEDIUM_OUTLINE.md`, `FAQ.md`.
2. **Harmonize Standard Deviations:**
   - Replace any occurrence of `9.42%` or `10.97%` for tool accuracy SD with `12.19%` (source: `results/master_protocol_summary.json`).
   - Audit `FINAL_REPORT.md` and `docs/launch/MEDIUM_OUTLINE.md`.
3. **Correct Pre-Registration Language:**
   - Replace unvarnished "pre-registered" assertions with honest phrasing: "protocol-locked in the same automated session (~3.5 hours before master execution)" or "protocol-locked via automated Git commit hash `3b46e1b`".
4. **Remove Promotional Language from FINAL_REPORT:**
   - Scan for and eliminate terms: "proves", "decisively", "pristine", "immaculate", "complete scientific rigor".
   - Reframe claims neutrally using phrases like "demonstrates", "the evidence indicates", "under the evaluated distribution".
5. **Update CITATION.cff:**
   - Update model description: reflect that fine-tuned CLS+CRF is the primary model (not frozen).
   - Update repository URL to canonical path.
   - Author name must match the chosen public persona (pending user decision in `USER_CHECKLIST.md`).
6. **Double-Check FAR Naming:**
   - Confirm zero occurrences of "Feature Attention Routing"; verify "Factored Attention Routing" is universal.

---

### Acceptance Criteria (Pass / Fail)

- [ ] `git grep -i "4.21 mb"` returns zero matches across the repository.
- [ ] `git grep "9\.42\|10\.97"` returns zero matches for tool accuracy SD.
- [ ] `git grep -i "proves\|decisively\|pristine\|immaculate" FINAL_REPORT.md` returns zero matches.
- [ ] `CITATION.cff` updated and valid YAML.
- [ ] Git commit created with exact message: `fix: remove 4.21 MB claims, unify SDs, fix pre-registration language, remove promotional text`.
