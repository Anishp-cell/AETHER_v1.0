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
- **Slot Tag Inventory ($|Y|=26$ BIO Tags):**
  - Entity types: `action`, `device`, `query`, `url`, `app_name`, `text_to_type`, `recipient`, `message`, `duration`, `label`, `media_item`, `command`, `instruction`, `task_description`, `skill_name`, `language`, `code_description`.

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
- **Non-Verbatim / Implied Values:** **5.21%** of slot values are non-verbatim:
  - Relative temporal offsets (e.g., "tomorrow morning" mapped to ISO dates)
  - Boolean or state flags (e.g., implied "toggle=true")
  - Categorical action normalizations.

This confirms that sequence tagging (BIO tagging) represents a sound, highly representative formulation for 95% of target arguments.

---

## 5. Multi-Tool Compositionality

A complete audit of multi-label combinations across all 5,000 samples revealed:
- **Single-Tool Utterances:** 4,886 samples (97.72%)
- **Multi-Tool Utterances:** 114 samples (2.28%)
- **Active Pair:** All 114 multi-tool utterances correspond to compound commands combining `handle_smart_home` + `media_control` (e.g., "turn on living room lamp and play relaxing acoustic guitar"). All other hypothetical pairs have 0 occurrences in the dataset.

---

## 6. Split Methodology & Leakage Prevention

To ensure rigorous out-of-distribution evaluation:
1. **Cluster-Grouped Stratification:** Utterances are grouped by semantic paraphrase cluster ($N=253$). Under no circumstances is a cluster split across train and validation folds.
2. **Skeleton Token-Edit Distance:** All cross-fold template skeleton edit distances are bounded $< 0.80$ (range 0.71–0.79), preventing lexical leakage.
3. **Tool Stratification:** Tools with $\le 5$ clusters are distributed across folds first before general clusters are allocated, ensuring outer training and inner validation folds have representation for every evaluable tool.
4. **Reproducible Manifest:** The 15 outer folds (5 folds $\times$ 3 random shuffles) are frozen with individual SHA-256 hashes in `data/splits/cv_manifest.json`.
