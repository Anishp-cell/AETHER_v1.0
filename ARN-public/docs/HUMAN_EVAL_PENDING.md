# Human Evaluation Benchmark: Status & Protocol

**Benchmark Identifier:** `ARN-HumanEval-Pending`  
**Template Specification:** `ARN/data/human_eval_template.jsonl`  
**Status:** **HUMAN-WRITTEN EVALUATION PENDING** (Awaiting User/Author Test Set)  
**Date Updated:** 2026-10-18  

---

## 1. Executive Summary & Provenance Disclosure

In accordance with strict scientific integrity rules:
1. **Agent-Generated Notice:** The 250 prompt structures generated in `ARN/data/human_eval_template.jsonl` were synthesized by an automated AI agent. They were **NOT written by the human author/user** and **must not be used as empirical evidence of human performance or generalization.**
2. **Pending Human Benchmark:** The human evaluation test suite will be independently authored directly by the human user/author, consisting of **200–300 original utterances** with verbatim gold tools and slot argument boundaries.
3. **Evaluation Protocol:** Evaluation will consist strictly of transparent classification accuracy against those human-authored gold labels:
   - Tool Subset Exact Match
   - Argument Slot Strict Span F1 & Lenient F1
   - Joint Exact Match (both tools and arguments correct)
4. **No Cohen's Kappa:** Since evaluation is measured directly against authoritative ground-truth gold labels written by the user rather than between subjective multi-annotator crowd workers, Cohen's kappa inter-rater agreement is unnecessary and removed.
5. **Manuscript Statement:** All paper drafts, documentation, and model cards explicitly report:
   > *"Human-written evaluation pending."*

---

## 2. Benchmark Specification for Human Authoring

When authoring the 200–300 test utterances, the following rules apply:
1. **Natural Vocabulary & Phrasing:** The author should write colloquial, natural, and complex instructions reflecting real-world usage (e.g. multi-intent requests, typos, conversational phrasing).
2. **Verbatim Gold Arguments:** All gold argument values specified for a tool must appear verbatim as exact character substrings in the utterance. If an argument is implicit or non-verbatim, it should not be assigned to slot span extraction.
3. **Gold Tool Ground Truth:** Annotate the exact set of active tools required to fulfill the user's intent from the 18 canonical agent tools.

---

## 3. Evaluation Execution

When the user-authored test set `ARN/data/human_eval_gold.jsonl` is ready:
```bash
.venv/Scripts/python ARN/scripts/run_human_eval.py \
    --input ARN/data/human_eval_gold.jsonl \
    --model-dir ARN/checkpoints/arn_final_model/ \
    --output ARN/results/human_eval_results.json
```
Metrics will be appended to `ARN/results/ledger.json`.

