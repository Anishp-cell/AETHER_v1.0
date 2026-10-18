# Human Evaluation Benchmark: Status & Deferred Execution

**Benchmark Identifier:** `ARN-HumanEval-250`  
**Template Specification:** `ARN/data/human_eval_template.jsonl`  
**Status:** **SEALED & PENDING EXTERNAL RATING** (Unannotated)  
**Date Logged:** 2026-10-18  

---

## 1. Executive Summary

In accordance with Hard Rule 1 (Zero Fabrication), Hard Rule 3 (Full Discrepancy Reconciliation), and the pre-registered protocol in `docs/PREREGISTRATION.md`, **no synthetic or LLM-generated human evaluation results are included in this repository or the research paper.**

The benchmark template consisting of 250 balanced conversational test cases remains sealed and unannotated at `ARN/data/human_eval_template.jsonl`. It is strictly reserved for independent human evaluators.

---

## 2. Benchmark Design & Composition

The `ARN-HumanEval-250` benchmark was structured using `scripts/make_human_eval_template.py` to test out-of-distribution generalization across all 18 canonical agent tools:

1. **Single-Tool Requests (180 instances):**
   - Exactly 10 distinct scenario prompts per tool across all 18 canonical tools.
   - Spans diverse functional domains: operating system execution (`run_computer_command`, `open_app_and_type`), web navigation (`browser_task`, `open_url`, `read_specific_url`, `search_web`, `search_and_read_web`), communication (`send_whatsapp_message`), home automation (`handle_smart_home`), media playback (`media_control`), system utilities (`get_current_time`, `get_system_diagnostics`, `set_timer`), computer vision (`analyze_screen_with_llava`), complex reasoning (`route_to_deepseek`), and automation authoring (`teach_new_skill`, `write_and_run_script`).

2. **Multi-Tool Compositional Requests (70 instances):**
   - Balanced pairs representing natural chained user workflows (e.g., `handle_smart_home` + `media_control`, `search_web` + `send_whatsapp_message`, `get_current_time` + `set_timer`).
   - Requires simultaneous multi-label tool activation and disjoint argument extraction.

---

## 3. Strict Integrity Rules (Why Synthetic Annotations Are Prohibited)

1. **No LLM Auto-Filling:**
   - Populating "human" evaluation test sets using frontier LLMs (e.g. GPT-4, Claude, DeepSeek) introduces severe stylistic leakage. Synthetic generation mirrors synthetic training biases, masking real out-of-distribution failure modes such as colloquial speech, typos, fragmented grammar, and implicit phrasing.
   - Per `docs/HUMAN_EVAL_GUIDE.md`, all utterances and slot values must be hand-authored by human subjects.

2. **Verbatim Substring Constraint:**
   - Every extracted argument value assigned to `gold_slots` must appear verbatim in the utterance text. Non-verbatim or inferred arguments are disallowed or left blank.

3. **Sealed Holdout:**
   - Neither the model weights, tokenizer, nor hyperparameter selection may access or optimize against this benchmark prior to final evaluation.

---

## 4. Evaluation Harness & Execution Instructions

The evaluation harness has been implemented and unit-tested in `scripts/run_human_eval.py`.

### When external annotator ratings become available:
1. Save the filled and verified file to `ARN/data/human_eval_filled.jsonl`.
2. Run integrity validation:
   ```bash
   .venv/Scripts/python ARN/scripts/make_human_eval_template.py --verify ARN/data/human_eval_filled.jsonl
   ```
3. Execute the automated evaluation harness against the final trained model:
   ```bash
   .venv/Scripts/python ARN/scripts/run_human_eval.py \
       --input ARN/data/human_eval_filled.jsonl \
       --model-dir ARN/checkpoints/arn_final_model/ \
       --output ARN/results/human_eval_results.json
   ```
4. Update `ARN/results/ledger.json` with the empirical scores and publish the final report addendum.

---

## 5. Paper & Public Disclosure Statement

In all paper manuscripts, model cards, and documentation:
- Cross-validation results across the 15 tool-stratified cluster folds (105 runs) represent the primary empirical evaluation.
- The Human Evaluation section is explicitly reported as:
  > *"The 250-utterance human evaluation benchmark is sealed and deferred to an independent multi-annotator user study to prevent synthetic artifact leakage. The complete evaluation protocol, template, and harness are published open-source to enable third-party reproduction."*
