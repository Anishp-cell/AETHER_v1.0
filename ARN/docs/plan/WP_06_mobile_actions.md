# Work Package Prompt: WP-06 (WP-F) — Mobile-Actions External Benchmark Assessment

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Large (1 session, ~3–4 hours if training; ~45 min if documenting as pending)  
**Compute Needed:** GPU required if training 7-tool model; CPU only if documenting as pending  
**Prerequisites:** WP-01 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-F)
3. `ARN/docs/plan/RISKS_AND_DECISIONS.md` (Check decision recommendation on Mobile-Actions)
4. `ARN/_archive/` (Inspect any archived Mobile-Actions code or references)
5. `ARN/docs/paper/` (Check whether Mobile-Actions is cited or claimed)

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- If evaluating: Must use the official Mobile-Actions train/test split; zero threshold or parameter tuning on the test set.
- If deferring: Explicitly state that external evaluation on Mobile-Actions is pending and unverified.
- Never use unverified historical numbers from earlier drafts.

---

### Objective & Scope

Determine the final status of the Google Mobile-Actions external benchmark. Either execute a rigorous, leak-free evaluation on the official 7-tool split or cleanly archive all references and state in the paper that external benchmark evaluation is deferred to future work.

---

### Exact Execution Steps

1. **Evaluate Feasibility & Dataset Availability:**
   - Check if the official Google Mobile-Actions dataset is present locally in `data/` or if external download/licensing is required.
   - If missing or licenses are restrictive, execute the **Deferral Path**.
2. **Path A — Execution Path (If Dataset Present & Approved):**
   - Implement `scripts/run_mobile_actions_benchmark.py`:
     - Load official 7-tool subset and official train/test splits.
     - Train ARN (BERT-tiny + CRF) on the training split.
     - Evaluate zero-shot / fine-tuned accuracy on test split without hyperparameter tuning on test.
   - Record runs in `results/ledger.json` under `p16_mobile_actions_*`.
   - Add Table 4 to the paper comparing ARN against published Mobile-Actions baselines.
3. **Path B — Deferral Path (Recommended Default):**
   - Confirm that no unverified Mobile-Actions numbers appear in `docs/paper/`, `FINAL_REPORT.md`, or launch drafts.
   - Add a clear subsection in `docs/paper/` under Future Work / Limitations:
     *"While ARN is architected for extensible tool schemas, empirical evaluation on public external suites such as Google Mobile-Actions remains future work pending license and multi-turn schema alignment."*
   - Ensure all legacy Mobile-Actions artifacts remain quarantined in `_archive/`.

---

### Acceptance Criteria (Pass / Fail)

- [ ] Either: Verified, reproducible evaluation runs recorded in `results/ledger.json`; OR explicit statement in paper and README that Mobile-Actions is deferred.
- [ ] Zero unverified Mobile-Actions metrics present in any active paper table or text.
- [ ] Git commit created with exact message: `feat: add Mobile-Actions benchmark results` OR `docs: document Mobile-Actions as pending`.
