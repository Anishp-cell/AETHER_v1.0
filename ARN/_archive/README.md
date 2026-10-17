# ARN Archive Inventory

This directory archives historical code, previous model checkpoints, legacy scripts, and Phase 1 review results. In accordance with strict reproducibility and git safety policies, **no historical artifacts are ever permanently deleted**.

---

## 1. Archived Model Checkpoints (`_archive/checkpoints/`)

| Checkpoint Name | Size | Architecture / Origin | Reason for Archiving |
| :--- | :--- | :--- | :--- |
| `arn_best_model.pt` | 17.79 MB | 18 tools, 26 slot tags | Historical PyTorch checkpoint overwritten in commit `9248a09` during RTX 5050 fine-tuning. Stored here as reference. Retraining from scratch in Phase 3. |
| `arn_expanded_18tool.pt` | 17.79 MB | 18 tools, 26 slot tags | Intermediate surgical weight transfer checkpoint produced by `expand_checkpoint.py`. |
| `arn_mobile_best.pt` | 17.78 MB | 7 tools, 22 slot tags | Checkpoint trained on the Google Mobile-Actions external benchmark dataset. |
| `arn_model.onnx` | 17.76 MB | 18 tools, 26 slot tags | FP32 ONNX graph export of `arn_best_model.pt`. |
| `arn_model_int8.onnx` | 4.54 MB | 18 tools, 26 slot tags | Dynamic INT8 quantized ONNX graph (~4.33 MiB) of `arn_best_model.pt`. |

---

## 2. Archived Benchmark Results (`_archive/benchmark_results/`)

| File | Content | Reason for Archiving |
| :--- | :--- | :--- |
| `ieee_benchmark_results.json` | 5-fold cross-validation, hardware latency, tracemalloc RAM (0.25 MB), and 200-prompt OOD rejection results. | Generated on leaked random split (non-group-disjoint). Retraining with group-disjoint splits in Phase 3. |
| `ieee_paper_tables.tex` | Auto-generated LaTeX tables from `ieee_benchmark_suite.py`. | Legacy table source containing outdated leaked-split metrics. |
| `mobile_actions_benchmark_results.json` | Google Mobile-Actions evaluation metrics (strict exact match, relaxed, name-only). | Historical evaluation run. |
| `mobile_actions_paper_table.tex` | Mobile-Actions comparison table with hardcoded generative model baselines. | Legacy table source. |

---

## 3. Archived Legacy & Dead Scripts (`_archive/legacy_scripts/`)

| Script Name | Purpose / Description | Issue / Why Archived |
| :--- | :--- | :--- |
| `ablation_study.py` | Standalone ablation trainer. | Dead code: trained custom from-scratch encoder rather than BERT-tiny trunk. |
| `adversarial_eval.py` | 10-prompt OOD evaluation. | Ad-hoc handcrafted script; superseded by unified `scripts/evaluate.py`. |
| `attention.py` | Custom multi-head attention module. | Dead code: unreferenced by active BERT-tiny model. |
| `baselines_ablation.py` | Multi-seed baseline benchmark (JointBERT, ARN No-CRF). | **Flawed methodology:** Evaluated untrained JointBERT with random heads and No-CRF without independent training. |
| `benchmark_arn.py` | Early latency and OOD rejection benchmark. | Superseded by standardized `scripts/benchmark_latency.py`. |
| `crf_ablation.py` | CRF vs Softmax comparison script. | Outdated paths, stale PAD index 21 instead of 25. |
| `diagnose_eval.py` | Error diagnostics for Mobile-Actions. | Scratch diagnostic script. |
| `embedding.py` | Custom factorized embedding class. | Dead code: unreferenced by active BERT-tiny model. |
| `encoder.py` | Custom Transformer encoder class. | Dead code: unreferenced by active BERT-tiny model. |
| `eval_mobile_actions.py` | Mobile-Actions evaluation runner. | Relocating to modular evaluation pipeline. |
| `expand_checkpoint.py` | Surgical weight expansion utility. | One-off migration utility from 16 to 18 tools. |
| `flat_tagging_ablation.py` | Multi-tool slot collision test. | **Emulation script:** Did not test a trained model; hardcoded 4 samples with Python if-statements. |
| `groq_paraphrase_augment.py`| Groq Llama-3.3-70B paraphrase generator. | External API script requiring keys. |
| `ieee_benchmark_suite.py` | Monolithic benchmark runner. | Replaced by modular, traceable scripts in `scripts/`. |
| `local_paraphrase_augment.py`| Rule-based prefix/suffix augmentation. | Archived historical data pipeline. |
| `mobile_actions_config.py` | Mobile-Actions config. | Integrated into `src/arn/config.py`. |
| `mobile_actions_dataset.py` | Mobile-Actions dataset loader. | Integrated into `src/arn/dataset.py`. |
| `mobile_actions_model.py` | Mobile-Actions model variant. | Integrated into `src/arn/model.py`. |
| `ood_per_tool_analysis.py` | Per-tool degradation analysis. | **Flawed metric:** Evaluated token attention weights instead of sigmoid tool probabilities. |
| `paraphrase_augment.py` | Gemini paraphrase generator. | External API script. |
| `template_generator.py` | Core synthetic template generator. | Cleaned and modernized in `scripts/make_data.py`. |
| `test_clean_eval.py` | Scratch character offset test. | Scratch exploration script. |
| `test_dataset_spans.py` | Scratch substring containment test. | Scratch exploration script. |
| `test_datetime_resolver.py` | Scratch datetime resolution test. | Scratch exploration script. |
| `train_mobile_actions.py` | Mobile-Actions training script. | Replaced by unified `scripts/train.py`. |
| `zero_shot_lexical_eval.py` | Lexical shift benchmark. | Produced impossible Table 3 percentages due to template duplication to reach N=50. Replaced by `scripts/evaluate.py`. |

---

## 4. Archived Legacy Root Core Files (`_archive/legacy_root/`)

These 10 root Python modules represented the pre-refactor implementation before modularization into `src/arn/`:

| Module | Description | Issues Addressed in Refactor |
| :--- | :--- | :--- |
| `config.py` | Hardcoded dictionaries for tools and slot mappings. | Refactored into typed `ARNConfig` dataclass and YAML in `src/arn/config.py`. |
| `crf.py` | PyTorch Linear-chain CRF. | Lacked invalid transition masking; had sequence-length normalization imbalance. Refactored into `src/arn/crf.py` with hard BIO masking. |
| `dataset.py` | PyTorch dataset loader using substring matching for span alignment. | Character offset misalignment on repeated subwords. Refactored into token-boundary aligned `src/arn/dataset.py`. |
| `dataset_generator.py` | Standalone template generation script. | Relocated to `scripts/make_data.py`. |
| `export_onnx.py` | Ad-hoc ONNX export script. | Refactored into standard CLI script `scripts/export_onnx.py`. |
| `far.py` | Factored Attention Routing layer. | Unconstrained attention softmax mislabeled as tool confidence. Refactored in `src/arn/far.py` with sigmoid output calibration. |
| `inference.py` | Inference pipeline. | Did not enforce hard BIO masking or handle JSON-safe schema validation. Refactored into `src/arn/inference.py`. |
| `model.py` | Monolithic joint model definition. | Unclean module interfaces and hardcoded paths. Modularized in `src/arn/model.py`. |
| `tokenizer_utils.py` | Legacy subword alignment helper. | Torchvision conflict handling was unstable. Modernized in `src/arn/tokenizer.py`. |
| `train.py` | Legacy training loop. | Untrained baselines, non-disjoint data splitting, uncalibrated losses. Modernized in `src/arn/loss.py` and `scripts/train.py`. |

