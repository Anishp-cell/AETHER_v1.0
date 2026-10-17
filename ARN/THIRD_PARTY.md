# Third-Party Notices and Dataset Licenses

This repository uses and references third-party models, datasets, and API-generated augmentations. This document outlines their respective licenses and terms of use.

---

## 1. Pre-Trained Foundation Trunk

### BERT-tiny (`google/bert_uncased_L-2_H-128_A-2`)
- **Origin:** Google Research (Turc et al., 2019, *"Well-Read Students Learn Better: On the Importance of Pre-training Compact Models"*)
- **Repository:** [HuggingFace Hub: google/bert_uncased_L-2_H-128_A-2](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)
- **License:** **Apache License 2.0**
- **Usage:** ARN uses the frozen trunk weights (~4.38M parameters) as a static bidirectional contextual encoder. The weights are not fine-tuned during standard ARN training.

---

## 2. Evaluation Benchmarks

### Google Mobile-Actions (`google/mobile-actions`)
- **Origin:** Google LLC (2025)
- **Repository:** [HuggingFace Hub: google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
- **License:** **Apache License 2.0**
- **Usage:** Standardized public benchmark dataset ($N=961$ eval split) used for external out-of-domain tool routing and argument extraction evaluation.

---

## 3. Data Augmentation & LLM Generation

### Synthetic Seed Templates
- **Origin:** Internal algorithmic generation (`scripts/make_data.py`).
- **License:** Released under the repository's **MIT License**.

### LLM Paraphrase Augmentation (Historical Runs)
- **Services Used:** Groq API (Meta Llama-3.3-70B / Llama-3.1-8B) and Google Gemini API.
- **Terms & Compliance Notices:**
  - **Meta Llama 3 Community License:** Subject to Meta's Acceptable Use Policy.
  - **Google Generative AI Terms of Service:** Section on commercial/derivative model training restrictions. *Note for maintainers:* Google TOS prohibits using model outputs to train models that compete with Google services. Because ARN is a closed-registry discriminative sequence tagger / router rather than a general-purpose generative LLM, compliance is standard, but commercial distributors should verify downstream compatibility.

---

## 4. Software Dependencies

- **PyTorch:** BSD-3-Clause License (Meta Platforms, Inc.)
- **Transformers:** Apache License 2.0 (Hugging Face, Inc.)
- **ONNX / ONNX Runtime:** MIT License (Microsoft Corporation)
- **Scikit-Learn:** BSD-3-Clause License
- **NumPy:** BSD-3-Clause License
- **PyYAML:** MIT License
- **psutil:** BSD-3-Clause License
