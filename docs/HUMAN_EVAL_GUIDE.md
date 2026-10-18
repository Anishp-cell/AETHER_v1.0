# Human Evaluation Benchmark Guide & Protocol

**Status:** Protocol Active & Sealed Holdout Specification  
**Benchmark Target:** 200–300 Hand-Crafted Conversational Utterances  
**Template Location:** `ARN/data/human_eval_template.jsonl`  
**Validator Script:** `ARN/scripts/make_human_eval_template.py --verify <path>`

---

## 1. Strict Evaluation Integrity Rules

1. **Zero LLM Generation:**
   - **No language model (GPT-4, Claude, Gemini, DeepSeek, LLaMA, etc.) may generate or rephrase any utterance.**
   - All utterances must be hand-authored directly by the human evaluator. Synthetic LLM-generated benchmarks replicate LLM distribution artifacts and reward models that overfit synthetic style. Human authoring ensures authentic linguistic variance, typos, colloquialisms, and structural diversity.

2. **Sealed Holdout Protocol:**
   - Once filled, this dataset is strictly **sealed**.
   - **No model, training script, hyperparameter search, or threshold tuning script may touch, load, or observe this file.**
   - It will be evaluated exactly **once** as the final pristine out-of-distribution benchmark.

3. **Verbatim Argument Substring Requirement:**
   - Every value assigned to a key in `gold_slots` **must appear as an exact, case-insensitive substring inside the `utterance` text.**
   - The automated verifier (`--verify`) will parse each utterance and assert that `val.lower() in utterance.lower()`.
   - If an argument is non-verbatim or inferred (e.g. categorical command types not uttered by the user), leave that slot key empty (`""`) or omit it.

---

## 2. Schema Reference (Canonical 18 Tools)

Every utterance must map to one or more of the 18 canonical tools below:

| Tool Name | Expected Slot Arguments | Notes & Requirements |
|---|---|---|
| `open_url` | `url` | Clean URL or web address mentioned in utterance |
| `open_app_and_type` | `app_name`, `text_to_type` | Both app name and typed text must appear verbatim |
| `route_to_deepseek` | `query` | Complex reasoning, coding, or math prompts |
| `read_specific_url` | `url` | Reading/summarizing a specific webpage |
| `search_and_read_web` | `query` | Information lookup requiring deep web reading |
| `send_whatsapp_message` | `recipient`, `message` | Contact name and message body verbatim |
| `set_timer` | `duration`, `label` | e.g. duration: "10 minutes", label: "pizza" |
| `browser_open_and_interact` | `url`, `action_description` | URL and specific browser interaction |
| `teach_new_skill` | `skill_name`, `instruction` | Explicit skill authoring |
| `write_and_run_script` | `language`, `code_description` | Python, bash, or script execution |
| `search_web` | `query` | Quick web search or fact lookup |
| `run_computer_command` | `command` | Terminal or shell command string |
| `analyze_screen_with_llava` | `instruction` | Multimodal visual inspection of current desktop |
| `get_current_time` | *(None)* | Zero arguments |
| `handle_smart_home` | `device`, `action` | Device name and state toggle ("turn on", "dim") |
| `media_control` | `action`, `media_item` | Playback command and media/track title |
| `get_system_diagnostics` | *(None)* | System stats, CPU/RAM status |
| `browser_task` | `task_description` | Multi-step autonomous browser goal |

---

## 3. Authoring Workflow

1. **Generate the Clean Template:**
   ```bash
   .venv\Scripts\python ARN/scripts/make_human_eval_template.py --output ARN/data/human_eval_template.jsonl --count 250
   ```
   This generates 250 JSONL records:
   - 180 Single-Tool records (10 per tool across all 18 tools)
   - 70 Multi-Tool Composition records (balanced pairs such as `handle_smart_home` + `media_control`, `search_web` + `send_whatsapp_message`)

2. **Fill Utterances & Slots:**
   Open `ARN/data/human_eval_template.jsonl` in your editor. For each record:
   - Write a natural user utterance into `"utterance"`.
   - Fill the extracted values into `"gold_slots"` (ensuring they match text verbatim).
   - Set `"completed": true`.

3. **Verify Integrity:**
   Run the automated checker at any time during authoring:
   ```bash
   .venv\Scripts\python ARN/scripts/make_human_eval_template.py --verify ARN/data/human_eval_template.jsonl
   ```
   The checker will highlight any missing strings, schema typos, or non-verbatim argument values.

4. **Seal File:**
   Once verified, copy or commit the final benchmark to `ARN/data/splits/human_eval_sealed.jsonl`.
   Do not allow any script to read this file until the final post-Stage 3 benchmark evaluation.
