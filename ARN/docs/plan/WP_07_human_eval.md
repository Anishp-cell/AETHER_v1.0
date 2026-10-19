# Work Package Prompt: WP-07 (WP-G) — Human-Written Evaluation Tooling & Protocol Staging

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Small-Medium (1 session, ~45 min)  
**Compute Needed:** CPU only  
**Prerequisites:** WP-01 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-G)
3. `ARN/data/human_eval_template.jsonl`
4. `ARN/scripts/run_human_eval.py`
5. `ARN/tests/test_human_eval_harness.py`
6. `ARN/docs/paper/` (Human evaluation text)

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- CRITICAL INTEGRITY RULE: The agent MUST NOT write synthetic utterances and pass them off as human data. `data/human_eval_template.jsonl` is currently an unpopulated template / synthetic mock; it is NOT empirical human evidence.
- The paper MUST state that human evaluation is pending human data collection by the author.
- Overlap with training templates must be quantified via $n$-gram jaccard similarity ($n=1,2,3$).

---

### Objective & Scope

Build and verify the standalone evaluation harness that allows the human author to provide 200–300 naturally written utterances, validate their JSONL schema, check for training data leakage, and execute a single-pass inference audit. Ensure the paper accurately describes human evaluation as staged and pending.

---

### Exact Execution Steps

1. **Implement Human Eval Validator (`scripts/validate_human_eval.py`):**
   Create a script that verifies any candidate `data/human_eval_user.jsonl`:
   - Valid JSONL syntax.
   - Required keys: `id`, `utterance`, `gold_tools` (list of strings matching the closed 18-tool registry), `gold_spans` (list of `[start, end, label]` matching the 26 BIO slot types).
   - Character alignment: `utterance[start:end]` must match the verbatim argument text.
2. **Implement $N$-gram Overlap Audit:**
   Add a check in the validator to compute token 1-gram, 2-gram, and 3-gram overlap against `data/raw/synthetic_5000.jsonl` to ensure user-written utterances are not mere copies of the synthetic templates.
3. **Add Write-Protection / Tamper Seal:**
   Add a hashing check: when `scripts/run_human_eval.py` is invoked on user data, compute SHA-256 of the input file, log it to `results/human_eval_seal.json`, and set read-only file permissions on Windows/POSIX.
4. **Update Paper Language:**
   Inspect LaTeX files (`docs/paper/`) and ensure the Human Evaluation section states:
   *"To address the ecological validity limitation of synthetic templates, an independent human-written evaluation protocol has been pre-specified with schema validation and n-gram overlap safeguards. Empirical collection is currently staged."*

---

### Acceptance Criteria (Pass / Fail)

- [ ] `scripts/validate_human_eval.py` implemented and tested on valid and invalid mock inputs.
- [ ] `pytest tests/test_human_eval_harness.py` passes.
- [ ] No claim of completed human evaluation appears anywhere in the paper, README, or launch materials.
- [ ] Git commit created with exact message: `feat: improve human eval tooling, add validation, update paper language`.
