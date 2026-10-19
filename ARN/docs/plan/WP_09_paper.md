# Work Package Prompt: WP-09 (WP-I) — Paper Revision, Neutral Template, & Scientific Integrity

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Large (1–2 sessions, ~3–4 hours)  
**Compute Needed:** CPU only (LaTeX compilation, BibTeX, script checks)  
**Prerequisites:** WP-02, WP-03, WP-04, and WP-05 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-I)
3. `ARN/docs/paper/` (all LaTeX sources)
4. `ARN/docs/paper/numbers.tex`
5. `ARN/docs/paper/tables/table1_main_results.tex`
6. `ARN/docs/paper/tables/table2_hardware.tex`
7. Reviewer critiques: Reviewers Nq1K, k32Y, and automated review comments

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- No fabrication: All tables must compile directly from generated macros and files.
- Paper must be formatted as an independent PREPRINT (e.g., standard `article` class or neutral arXiv template); completely remove all AAAI submission and camera-ready residue.
- Mathematical rigor: Linear-chain CRF formulation must reflect the exact implementation (including start/end transitions, hard BIO mask, and $O(T |\mathcal{Y}|^2)$ decoding complexity).
- Zero banned claims (no 14.2 ms, no 5.55 MB, no 4.21 MB, no 810x/38x/216x, no 88.0 slot F1, no 92.3% Table 1, no 96.0% headline, no LoRA 62.4, no dynamic registry).

---

### Objective & Scope

Transform the rejected AAAI manuscript into a transparent, scientifically honest, and methodologically sound preprint. Fix all mathematical inaccuracies, integrate verified non-autoregressive SLU prior art, answer reviewer critiques point-by-point in `REVISION_NOTES.md`, and compile the preprint PDF.

---

### Exact Execution Steps

1. **Clean Paper Architecture & Rename Files:**
   - Rename `docs/paper/AnonymousSubmission2027.tex` → `docs/paper/arn_preprint.tex`.
   - Rename `docs/paper/CameraReady2027.tex` → `docs/paper/arn_preprint_named.tex`.
   - Rename `docs/paper/aaai2027.bib` → `docs/paper/references.bib`.
   - Remove AAAI submission styling (`aaai2027.sty`, `aaai2027.bst`) and adopt clean standard LaTeX formatting.
2. **Correct CRF Mathematical Formulation:**
   - Explicitly formulate the linear-chain CRF transition matrix $\mathbf{A} \in \mathbb{R}^{(|\mathcal{Y}|+2) \times (|\mathcal{Y}|+2)}$ including explicit $\langle\text{START}\rangle$ and $\langle\text{STOP}\rangle$ states.
   - Formalize the hard BIO transition mask: disallowing illegal transitions (e.g., $\langle\text{START}\rangle \to \text{I-X}$, $\text{O} \to \text{I-X}$, $\text{B-X} \to \text{I-Y}$ where $X \ne Y$).
   - Correct the Viterbi decoding time complexity to $O(T |\mathcal{Y}|^2)$.
   - Accurately document the joint objective function $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tool}} + \lambda \mathcal{L}_{\text{CRF}}$, verifying slot loss normalization.
3. **Revise Related Work & Prior Art:**
   - Cite and contextualize foundational non-autoregressive SLU systems:
     - GL-GIN (Qin et al., 2021).
     - Stack-Propagation (Qin et al., 2019) — clarify this is separate from GL-GIN.
     - MISCA (Pham et al., 2023).
     - SLIM (Cai et al., 2022).
     - DIET (Bunk et al., 2020).
     - JointBERT (Chen et al., 2019).
   - Contextualize small on-device models: Octopus v2 (Chen et al., 2024), TinyAgent (Zhang et al., 2024), FunctionGemma.
   - Neutral positioning for commercial systems (Jev by TypeSafe AI, Laya): mention neutrally as concurrent zero-shot routing systems without claiming priority or performance superiority.
4. **Draft Honest "Audit & Failure Modes" Section:**
   - Include a frank discussion of what went wrong in v1: random-split data leakage, CRF loss masking bugs, and the gap between synthetic template performance and realistic open-world generalization.
5. **Draft Comprehensive Limitations Section:**
   - Closed registry constraint ($K=18$ tools).
   - Template synthetics limitation (39.91% tool accuracy on unseen templates).
   - Inability of flat BIO tagging to represent nested/overlapping spans (~142 instances).
   - Evaluation restricted to laptop-class x86 CPU (no ARM/mobile measurements).
6. **Produce Reviewer Response Document (`docs/paper/REVISION_NOTES.md`):**
   - Detailed point-by-point rebuttal and acknowledgment addressing Reviewer Nq1K, Reviewer k32Y, and the automated critique.
7. **Compile Preprint & Run Numbers Audit:**
   - Run `pdflatex` to generate `arn_preprint.pdf`.
   - Run `python scripts/audit_paper_numbers.py` to ensure complete consistency.

---

### Acceptance Criteria (Pass / Fail)

- [ ] All AAAI-specific filenames eliminated; clean preprint `.tex` files compile without fatal errors.
- [ ] CRF math includes hard BIO transition mask, explicit boundary states, and $O(T|\mathcal{Y}|^2)$ complexity.
- [ ] Related work includes verified citations for GL-GIN, Stack-Propagation, MISCA, SLIM, DIET, and JointBERT.
- [ ] `docs/paper/REVISION_NOTES.md` thoroughly answers all reviewer objections.
- [ ] `python scripts/audit_paper_numbers.py` executes with 0 errors.
- [ ] Git commit created with exact message: `feat: rename paper to neutral template, add revision notes, compile preprint`.
