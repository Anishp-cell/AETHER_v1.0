# PyPI / Release Deprecation & Artifact Notice

**Document Version:** 1.0.0  
**Date:** October 18, 2026  
**Status:** Authoritative Release Notice

---

## 1. Deprecation Notice for Legacy Checkpoints

All prior unverified or provisional checkpoints (including `checkpoint_step_1899.pt`, `arn_best.pt`, and any checkpoints generated under legacy Stage 1–2d protocols) are formally **DEPRECATED and RETIRED**.

### Reasons for Deprecation:
1. **Coverage Confound in Prior Splits:** Legacy Stage 2d splits suffered from a severe coverage confound where entire tools (up to 33.25% of validation samples) were absent from training splits, distorting validation metrics.
2. **Data Leakage in Baseline Evaluators:** Early baseline scripts suffered from stopword removal issues, leakage in TF-IDF fit transforms, and uncalibrated thresholds.
3. **Unverified Claims:** Previous manuscript drafts cited unverified hardware claims (e.g. latency, memory, compression multiples) not measured under standardized runtime conditions.

---

## 2. Authoritative Release Artifact: `arn_final_model`

The authoritative model artifact for release is located at:
`checkpoints/arn_final_model/`

### Model Specifications:
- **Architecture:** `CLSCRFModel`
  - **Trunk:** Pre-trained `google/bert_uncased_L-2_H-128_A-2` (BERT-tiny, 2 layers, 128 hidden dim, 2 attention heads).
  - **Tool Head:** Dense multi-label binary classifier over `[CLS]` token representation ($128 \to 18$).
  - **Slot Head:** Linear-chain Conditional Random Field (CRF) token tagger ($128 \to 37$ BIO emission states).
- **Selection Decision:** Pre-registered Decision Rule P-2 (Joint Exact Match). Under 15-fold tool-stratified grouped cross-validation, `unfrozen_cls_crf` achieved **16.91% ± 6.56% Joint EM** (vs `frozen_arn` 3.44% ± 3.37%), representing a verified paired gain of **+13.47% [95% CI: 10.27%, 16.68%]**. This conclusively establishes **Outcome A** (fine-tuning superiority).
- **Training Data:** Full canonical pool of 5,000 verified command-and-control utterances (`data/clean/canonical_train_5000.jsonl`).
- **Optimal Multi-Label Activation Threshold:** $\tau = 0.30$ (median outer fold tuning).

---

## 3. Artifact Manifest & Verification

The release artifact directory contains:
- `model.pt`: Full PyTorch model state dictionary.
- `config.json`: Architecture, hyperparameters, and provenance metadata.
- `tools.json`: Binary tool index mapping (18 tools).
- `slots.json`: BIO tag index mapping (37 tags).
- `tokenizer/`: HuggingFace BertTokenizerFast vocabulary and configuration files.
- `manifest.json`: Cryptographic SHA-256 hashes of all artifact components.

To verify artifact integrity:
```bash
python -c "import hashlib, json; m = json.load(open('checkpoints/arn_final_model/manifest.json')); [print(f, hashlib.sha256(open('checkpoints/arn_final_model/' + f, 'rb').read()).hexdigest() == v['sha256']) for f, v in m['files'].items()]"
```

---

## 4. Quickstart Inference Example

```python
import json
import torch
from arn.baselines import CLSCRFModel
from arn.tokenizer import TOKENIZER, IDX_TO_TOOL, IDX_TO_TAG

# 1. Load config and model
with open("checkpoints/arn_final_model/config.json") as f:
    config = json.load(f)

model = CLSCRFModel(num_tools=config["num_tools"], num_tags=config["num_tags"])
model.load_state_dict(torch.load("checkpoints/arn_final_model/model.pt", map_location="cpu"))
model.eval()

# 2. Tokenize user command
text = "Set a timer for 15 minutes to take out the laundry"
encoded = TOKENIZER(text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
input_ids = encoded["input_ids"]
mask = encoded["attention_mask"]

# 3. Predict tools and slot spans
with torch.no_grad():
    t_logits, emissions = model(input_ids, mask)
    probs = torch.sigmoid(t_logits)[0]
    active_tools = [IDX_TO_TOOL[i] for i, p in enumerate(probs) if p >= config["optimal_threshold"]]
    tag_ids = model.crf.viterbi_decode(emissions, mask)[0]
    tokens = TOKENIZER.convert_ids_to_tokens(input_ids[0])
    tags = [IDX_TO_TAG.get(idx, "O") for idx in tag_ids[:int(mask[0].sum())]]

print("Active Tools:", active_tools)
print("Predicted Tokens & Tags:", list(zip(tokens[:len(tags)], tags)))
```
