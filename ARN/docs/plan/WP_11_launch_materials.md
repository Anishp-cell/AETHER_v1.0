# Work Package Prompt: WP-11 (WP-K) — Launch Communications & Verified Narrative Drafting

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Medium (1 session, ~60 min)  
**Compute Needed:** CPU only  
**Prerequisites:** WP-03, WP-05, and WP-09 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-K)
3. `ARN/docs/launch/LINKEDIN.md`
4. `ARN/docs/launch/X_THREAD.md`
5. `ARN/docs/launch/MEDIUM_OUTLINE.md`
6. `ARN/docs/launch/FAQ.md`
7. `ARN/docs/paper/numbers.tex`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- Every number must be script-filled from `docs/paper/numbers.tex` or verified against `results/ledger.json`.
- Zero banned claims (no "4.21 MB", no "810x", no "beats Jev", no "peer-reviewed", no "SOTA").
- Story narrative must be humble and honest: A 3rd-year engineering student in Pune building over 5 months, receiving severe peer-review rejection at AAAI, conducting an exhaustive forensic self-audit, and releasing a reproducible, honest preprint.
- Commercial positioning (Jev, Laya): mention neutrally as validation that small, fast, non-autoregressive decision models represent an important emerging paradigm.

---

### Objective & Scope

Synchronize all public launch materials (LinkedIn post, X thread, Medium outline, and technical FAQ) with verified numbers. Ensure the narrative focuses on rigorous engineering, audit lessons, and realistic edge AI performance without sensationalism.

---

### Exact Execution Steps

1. **Synchronize All Numbers:**
   - Audit and update all metric references across:
     - `docs/launch/LINKEDIN.md`
     - `docs/launch/X_THREAD.md`
     - `docs/launch/MEDIUM_OUTLINE.md`
     - `docs/launch/FAQ.md`
   - Ensure tool subset accuracy is reported as 39.91% ($\pm 12.19\%$), joint exact match as 16.91%, INT8 engine latency as 1.39 ms / 0.75 ms, and peak memory as 58.69 MB.
2. **Draft Two LinkedIn Formats:**
   - **Version 1 (Long Form / Story-Led):** Detailed narrative of the 5-month journey, the AAAI rejection due to evaluation leakage and loss masking bugs, the decision to self-audit rather than bury the failure, and key lessons in benchmarking joint SLU models.
   - **Version 2 (Short Form / Technical Summary):** Direct, metric-focused release announcement emphasizing small footprint (4.28 MB INT8), sub-millisecond multi-threaded latency, and open-source availability.
3. **Refine Technical FAQ (`docs/launch/FAQ.md`):**
   - Provide direct, honest answers to tough technical inquiries:
     - *Why did AAAI reject the paper?* (Explain random split leakage, synthetic bias, and transition mask bugs).
     - *Why is tool accuracy 39.91% instead of 90%+?* (Explain cluster-disjoint template evaluation).
     - *How does ARN compare to Jev or Laya?* (Explain architectural distinctions neutrally).
     - *Why is memory reported as 58.69 MB instead of 4.21 MB?* (Distinguish between on-disk model weight size and active process RSS).
4. **Banned Claims Automated Grep:**
   - Run grep across `docs/launch/` for all banned phrases and unverified metrics.

---

### Acceptance Criteria (Pass / Fail)

- [ ] All metrics in launch materials match `results/master_protocol_summary.json` and `results/deployment_benchmark.json`.
- [ ] Banned claims check returns 0 matches across `docs/launch/`.
- [ ] LinkedIn draft includes both long-form and short-form options.
- [ ] Technical FAQ addresses AAAI rejection, memory definitions, and template limitations directly.
- [ ] Git commit created with exact message: `fix: update launch materials with final verified numbers`.
