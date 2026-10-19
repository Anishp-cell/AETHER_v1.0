# ARN Strategic Risks, Open Scientific Questions, & Author Decisions

**Date:** 2026-10-19  
**Status:** Strategic Architecture & Decision Guide  
**Audience:** Human Author (Anish)  

---

## 1. Executive Summary

This document outlines the major scientific and strategic risks facing the ARN project post-audit, formulates the open empirical questions, and presents key architectural and publication decisions requiring the author's input. For each decision, a **Recommended Default** is provided based on research integrity, technical defensibility, and public narrative strength.

---

## 2. Key Author Decisions & Recommended Defaults

### Decision 1: Model Naming & Public Headline
* **Context:** In original drafts and the rejected AAAI submission, Factored Attention Routing (FAR) was marketed as the primary architectural innovation. However, after rigorous cluster-disjoint cross-validation, the fine-tuned CLS+CRF model substantially outperformed fine-tuned FAR+CRF (joint EM: 16.91% vs 5.90%; tool subset accuracy: 39.91% vs 15.65%). Frozen FAR achieved modest gains over frozen CLS (+2.59 joint EM), but under full fine-tuning, standard CLS pooling is clearly superior.
* **Options:**
  1. Continue marketing FAR as the core headline and downplay CLS+CRF.
  2. Frame ARN primarily as a compact joint Encoder-CRF architecture, reporting CLS+CRF as the primary winning model and presenting FAR transparently as an architectural ablation/study with detailed training dynamics.
* **Recommended Default:** **Option 2.** Position ARN as a sub-5MB joint encoder-CRF for on-device tool orchestration. Present CLS+CRF as the primary working model and treat FAR as a studied attention-routing mechanism whose inductive bias helps frozen representations but suffers from optimization challenges during joint end-to-end fine-tuning.

---

### Decision 2: Disclosure of AAAI-27 Submission & Rejection
* **Context:** The original manuscript was submitted to AAAI-27 and rejected in Phase 1 due to data leakage on random splits, loss masking bugs, and unsupported efficiency claims.
* **Options:**
  1. Bury the AAAI submission entirely and publish the preprint as an unblemished first attempt.
  2. Acknowledge the rejection transparently in the public narrative (LinkedIn, README, revision notes), framing the project around rigorous self-audit, learning from peer review, and rectifying evaluation leakage.
* **Recommended Default:** **Option 2.** In the academic and open-source ML community, an undergraduate student openly admitting to benchmark leakage, taking reviewer feedback seriously, rebuilding an entire evaluation protocol from scratch, and publishing realistic numbers is far more impressive and credible than another paper claiming 99% accuracy on synthetic benchmarks.

---

### Decision 3: Scope of Google Mobile-Actions External Benchmark
* **Context:** Mobile-Actions is a public 7-tool benchmark. Evaluating ARN on Mobile-Actions would provide external validation beyond synthetic templates, but requires mapping schemas, verifying dataset licensing, and spending 3–4 hours of GPU compute.
* **Options:**
  1. Train and evaluate a 7-tool ARN variant on the official Mobile-Actions split before release.
  2. Explicitly document Mobile-Actions evaluation as planned future work and release the preprint focused strictly on the synthetic cluster-disjoint benchmark.
* **Recommended Default:** **Option 2 (Defer).** Given that the primary goal is publishing an unassailable preprint and releasing verified artifacts, adding a new dataset introduces risks of schema mismatch or training instability. Defer Mobile-Actions to a v1.1 follow-up study unless the author has already verified the dataset locally.

---

### Decision 4: Target Publication Venue & Preprint Strategy
* **Context:** The paper needs a credible academic venue while establishing priority and open-source availability.
* **Options:**
  1. Upload directly to arXiv and submit to an on-device/edge AI workshop (e.g., MLSys on-device workshop, ACL Workshop on Tool Learning, or EMNLP demo track).
  2. Target a major conference (AAAI-28, ACL 2027) without uploading to arXiv.
  3. Upload to Zenodo / GitHub with no formal venue submission.
* **Recommended Default:** **Option 1.** Post the preprint to arXiv (or Zenodo if arXiv endorsement is unavailable) under a CC-BY license and submit to an upcoming edge ML / system workshop. Workshops welcome candid evaluation audits and practical deployment benchmarks.

---

## 3. Scientific Risks & Mitigations

| Risk | Impact | Probability | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Headline Accuracy Gap** (Tool acc 39.91% / Joint EM 16.91% vs old 90%+) | Public audience expects generative LLM-level accuracy | High | Emphasize latency (0.75 ms engine), tiny disk size (4.28 MB), and deterministic guarantees (0% JSON syntax errors). Clearly frame this as a 4.39M parameter model running on CPU. |
| **Synthetic Template Bias** (Evaluation limited to 5,000 synthetic queries) | Reviewers question real-world conversational transfer | High | Acknowledge directly in Limitations. Stage the human-written evaluation protocol so the community or author can collect out-of-distribution real queries. |
| **Overlapping BIO Spans** (142 utterances with colliding slots) | Flat linear-chain CRF mathematically cannot tag overlapping spans | Medium | Document that single-layer flat BIO models cannot represent overlapping slot boundaries by design; measure the exact impact (2.8% of test set) and propose multi-head or transition-based decoders in Future Work. |
| **Legacy PyPI Artifact Confusion** (Defective v0.1.0 package on PyPI) | Users install old package and find severe tagging bugs | High | Author must yank v0.1.0 on PyPI or upload v0.2.0 with a prominent deprecation notice explaining the CRF loss masking correction. |

---

## 4. Open Scientific Questions

1. **Why does fine-tuning impair FAR?**  
   *Observation:* Under frozen encoders, FAR outperforms CLS (+2.59 joint EM). Under fine-tuning, CLS achieves 39.91% tool accuracy while FAR struggles around 16–21%.  
   *Hypothesis:* Factored attention weights create ill-conditioned gradients that compete with BERT-tiny intermediate representations, leading to early over-fitting or gradient saturation in the query projection matrix $\mathbf{W}_q$.
2. **What is the true ceiling of non-autoregressive joint SLU on edge CPUs?**  
   *Observation:* At 4.28 MB, ARN executes in 2.34 ms end-to-end. Can a 15M–30M parameter MobileBERT or TinyBERT backbone push tool accuracy above 75% while keeping latency under 10 ms? This is the central research question ARN establishes for future work.
