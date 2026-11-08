# Data Statement for ARN Benchmark Corpus

Following the data statement framework outlined by Bender & Friedman (2018), this document describes the curation, composition, and validation of the benchmark dataset used to train and evaluate the Aether Routing Network (ARN).

---

## 1. Curation Rationale

The ARN dataset was developed to evaluate non-generative, real-time neural tool routing and argument slot extraction in edge environments. Traditional assistant benchmarks often focus on multi-turn dialogue state tracking (e.g., MultiWOZ) or open-domain web question answering (e.g., ToolBench). In contrast, the ARN corpus targets single-turn, on-device assistant commands mapped to a closed registry of 18 operating system, home automation, and web APIs.

---

## 2. Dataset Composition & Metadata

- **Corpus File:** `data/raw/aether_orchestrator_dataset.jsonl`
- **Cryptographic Hash (SHA-256):** `d2f25d93a7123226f5f849e7f72391a2fd8b93c1b04c79fa2f5a2c8b8d5a95b9`
- **Total Instance Count:** 5,000 conversational command utterances
- **Semantic Cluster Count:** 253 distinct paraphrase clusters
- **Candidate Tool Inventory ($K=18$):**
  - OS & Terminal: `run_computer_command`, `open_app_and_type`, `get_system_diagnostics`
  - Web & Browser: `open_url`, `read_specific_url`, `search_web`, `search_and_read_web`, `browser_task`, `browser_open_and_interact`
  - Communication & Media: `send_whatsapp_message`, `media_control`
  - Home Automation: `handle_smart_home`
  - System Utilities: `get_current_time`, `set_timer`
  - Reasoning & Vision: `route_to_deepseek`, `analyze_screen_with_llava`
  - Automation Authoring: `teach_new_skill`, `write_and_run_script`
- **Slot Tag Inventory ($|Y|=26$ BIO Tags = 12 slot types × B/I + O + PAD):**
  - Slot types (from `src/arn/tokenizer.py` `TAG_REGISTRY`): `contact`, `url`, `app`, `device`, `query`, `message`, `action`, `target`, `value`, `reminder`, `site`, `browser_task`. Tool argument names (e.g. `text_to_type`, `instruction`, `skill_name`) are mapped onto these 12 types by `ARG_TO_SLOT_BASE`; an earlier version of this statement listed 17 argument names as if they were tag types.

---

## 3. Language & Demographic Characteristics

- **Primary Language:** English (en-US / international English conversational assistant style).
- **Stylistic Distribution:**
  - Short imperative commands (e.g., "dim the lights in the kitchen")
  - Question phrasing (e.g., "could you please tell me what time it is right now")
  - Indirect or colloquial requests (e.g., "play some jazz on spotify")
  - Multi-intent compound instructions (e.g., "turn off bedroom ac and play smooth jazz")

---

## 4. Argument Representability Audit

An audit conducted in Phase 5 evaluated whether slot argument strings can be extracted verbatim from the input text:
- **Verbatim Substrings:** **94.79%** of all slot values appear as exact, case-insensitive contiguous substrings in the utterance text.
- **Non-Verbatim / Implied Values:** **5.21%** (456 of 8,748) of slot values are non-verbatim. Per `results/argument_representability_audit.json`, they are categorical action labels implied by the utterance (e.g. `action=set_volume` for "volume 70 asap", `action_type=open_app`, `action_type=shell`). An earlier version of this statement described them as temporal offsets and boolean flags; that was not supported by the audit.

This confirms that sequence tagging (BIO tagging) represents a sound, highly representative formulation for 95% of target arguments.

---

## 5. Multi-Tool Compositionality

A complete audit of multi-label combinations across all 5,000 samples revealed:
*Corrected 2026-11-08 by recounting `data/raw/aether_orchestrator_dataset.jsonl`; the previous figures (114 multi-tool, 2.28%, one pair) were wrong.*
- **Single-Tool Utterances:** 3,868 samples (77.36%)
- **Multi-Tool Utterances:** 1,132 samples (22.64%) across 7 tool pairs:
  `get_system_diagnostics+send_whatsapp_message` 209, `search_web+write_and_run_script` 193, `run_computer_command+search_web` 180, `search_web+send_whatsapp_message` 156, `send_whatsapp_message+set_timer` 156, `get_current_time+send_whatsapp_message` 124, `handle_smart_home+media_control` 114.

## 5b. Provenance of the augmented file

`data/raw/aether_orchestrator_dataset_augmented.jsonl` (13,798 rows, 4,095 multi-tool) is a legacy file. Earlier paper drafts described it as produced with LLM paraphrase augmentation (Groq-hosted Llama-3 70B), and the legacy codebase contains such a script (`ARN/groq_paraphrase_augment.py`). Its exact generation procedure is not documented, and **it is not used for any reported result**; all v1 results use the 5,000-row template-generated file above.

---

## 6. Split Methodology & Leakage Prevention

To ensure rigorous out-of-distribution evaluation:
1. **Cluster-Grouped Stratification:** Utterances are grouped by semantic paraphrase cluster ($N=253$). Under no circumstances is a cluster split across train and validation folds.
2. **Skeleton Token-Edit Distance:** All cross-fold template skeleton edit distances are bounded $< 0.80$ (range 0.71–0.79), preventing lexical leakage.
3. **Tool Stratification:** Tools with $\le 5$ clusters are distributed across folds first before general clusters are allocated, ensuring outer training and inner validation folds have representation for every evaluable tool.
4. **Reproducible Manifest:** The 15 outer folds (5 folds $\times$ 3 random shuffles) are frozen with individual SHA-256 hashes in `data/splits/cv_manifest.json`.
