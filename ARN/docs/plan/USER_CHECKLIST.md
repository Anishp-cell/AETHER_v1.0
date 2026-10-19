# ARN Human-Author Action Checklist

**Date:** 2026-10-19  
**Status:** Mandatory Manual Action Items  
**Audience:** Author (Anish)  

---

## Overview

This checklist contains all tasks that **only the human author** can execute. Automated agents cannot make legal determinations, create external accounts, post to public platforms, or fabricate authentic human natural language data.

Complete these action items in the specified order before launching or publishing.

---

## 1. Pre-Release Identity & Attribution

- [ ] **1.1 Resolve Author Name Discrepancy:**
  - *Context:* `CITATION.cff` lists `Pattnaik, Anish`, while the AAAI review correspondence was addressed to `Anish Sohan Pathak`.
  - *Action:* Decide on the exact official name for paper header, citation metadata, and copyright notices (e.g., `Anish Pathak` or `Anish Sohan Pathak` or `Anish Pattnaik`).
  - *Files to update once decided:* `CITATION.cff`, `docs/paper/arn_preprint_named.tex`, `setup.py`.

- [ ] **1.2 Configure GitHub & Git Identity:**
  - *Context:* The local git config contains personal email `anishpathak778@gmail.com`.
  - *Action:* Configure a public GitHub noreply address:
    ```bash
    git config --global user.name "Anish Pathak"  # Or chosen name
    git config --global user.email "Anishp-cell@users.noreply.github.com"
    ```
  - *Verification:* Ensure no personal email is committed to the public repository tree.

- [ ] **1.3 Create Public GitHub Repository:**
  - *Action:* Create a clean public repository (e.g. `github.com/Anishp-cell/arn-orchestrator` or `github.com/Anishp-cell/ARN`).
  - *Rule:* Push only the staged export `ARN-public/` (located outside the Aether root). Never push the entire Aether workspace.

---

## 2. Licensing & Legal Verification

- [ ] **2.1 Verify BERT-tiny Base License:**
  - *Context:* Backbone weights originate from `google/bert_uncased_L-2_H-128_A-2`.
  - *Action:* Confirm Apache-2.0 compliance; ensure Google Apache-2.0 license notice is preserved in `checkpoints/` and `LICENSE`.

- [ ] **2.2 Verify Dataset IP & Terms:**
  - *Context:* 5,000 synthetic utterances were generated via local algorithmic combinatorial templates.
  - *Action:* Confirm no third-party terms of service (OpenAI, Anthropic, Google Gemini) were violated during synthetic template generation.

- [ ] **2.3 External Benchmark Terms (If Running Mobile-Actions):**
  - *Context:* Google Mobile-Actions is subject to Google dataset terms.
  - *Action:* Review redistribution and benchmark citation permissions before including any Mobile-Actions files.

---

## 3. Package Management & Deprecation

- [ ] **3.1 Yank or Deprecate Legacy PyPI Package:**
  - *Context:* Legacy package `arn-orchestrator` v0.1.0 was published to PyPI with unmasked [CLS]/[SEP] CRF loss and broken argument tagger bugs.
  - *Action:* Log in to PyPI, mark v0.1.0 as **Yanked** with the reason: *"Defective loss masking in legacy release; see GitHub for corrected v0.2.0 preprint models."*
  - *Alternative:* Release `arn-orchestrator` v0.2.0 packaging the verified ONNX models and CLS+CRF pipeline.

---

## 4. Empirical Validation (Human Utterances)

- [ ] **4.1 Author 200–300 Natural Human Utterances:**
  - *Context:* Synthetic template evaluations are prone to distributional blind spots. A human evaluation set is needed to measure real-world performance.
  - *Action:* Hand-write 200–300 authentic utterances reflecting real smartphone voice assistant commands across the 18 registered tools.
  - *Strict Rules:*
    - Write them manually without using ChatGPT, Claude, Gemini, or any LLM paraphrase tools.
    - Annotate with exact verbatim span boundaries `[start, end, slot_label]` and gold tool list.
    - Save to `ARN/data/human_eval_user.jsonl`.
  - *Validation:* Run `python scripts/validate_human_eval.py data/human_eval_user.jsonl`.
  - *Evaluation:* Run `python scripts/run_human_eval.py data/human_eval_user.jsonl`.

---

## 5. Public Narrative & Release Execution

- [ ] **5.1 Confirm Personal Narrative Numbers:**
  - *Context:* The draft launch text states: *"3rd-year student in Pune, started January 2026, submitted July 2026, ~5 months at 14–15 hours/day mostly during summer break."*
  - *Action:* Review and confirm these details are accurate and comfortable to share publicly.

- [ ] **5.2 Preprint Hosting (arXiv / Zenodo):**
  - *Action:*
    - If arXiv endorsement in `cs.CL` or `cs.AI` is available: upload `docs/paper/arn_preprint_named.pdf` and LaTeX source.
    - If endorsement is pending: upload to Zenodo / OpenReview to obtain an immutable DOI and timestamp.

- [ ] **5.3 Review & Publish Launch Materials:**
  - *Action:* Read `docs/launch/LINKEDIN.md`, `docs/launch/X_THREAD.md`, and `docs/launch/FAQ.md`. Approve phrasing and post on respective personal profiles once the GitHub repo and preprint links are live.
