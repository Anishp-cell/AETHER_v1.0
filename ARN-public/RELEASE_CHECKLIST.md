# ARN Public Release Checklist

This document details the final steps to publish the staged `ARN-public` repository to GitHub, Hugging Face, and PyPI.

---

## 1. Local Verification (Pre-Flight)

- [x] All 26 unit tests passing (`pytest tests/`)
- [x] Zero unverified or banned claims verified via `scripts/audit_paper_numbers.py`
- [x] 15-fold cross-validation results verified against `results/ledger.json`
- [x] Model checkpoints and SHA-256 manifest verified in `checkpoints/arn_final_model/manifest.json`
- [x] ONNX FP32 and Dynamic INT8 engines benchmarked and verified
- [x] Sealed human evaluation benchmark template preserved at `data/human_eval_template.jsonl`

---

## 2. GitHub Public Launch

When ready to push to public GitHub:

1. Create a new public repository on GitHub:
   - Repository name: `ARN` (or `aether-routing-network`)
   - Description: `Ultra-compact, sub-18MB non-generative joint neural tool orchestrator and slot extractor for real-time edge dispatch.`
   - Public visibility

2. Add remote and push:
   ```bash
   cd ../ARN-public
   git remote add origin https://github.com/Anishp-cell/ARN.git
   git branch -M main
   git push -u origin main
   ```

3. Create GitHub Release `v1.0.0`:
   - Tag: `v1.0.0`
   - Release title: `ARN v1.0.0: Master Protocol Audited Release`
   - Upload pre-built binaries:
     - `checkpoints/arn_final_model/model.pt`
     - `checkpoints/arn_final_model/model.onnx`
     - `checkpoints/arn_final_model/model_int8.onnx`

---

## 3. Hugging Face Model Hub

To publish the model card and weights to Hugging Face Hub:

```bash
huggingface-cli login
huggingface-cli upload Anishp-cell/arn-tiny checkpoints/arn_final_model/ . --repo-type model
```

Include `MODEL_CARD.md` as the `README.md` on Hugging Face.

---

## 4. PyPI Package Publishing

Before publishing to PyPI:
1. Ensure legacy pre-refactor release is yanked or upgraded per `docs/PYPI_NOTE.md`.
2. Build clean wheels:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   ```
3. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

---

## 5. Social & Community Announcements

Draft launch announcements verified against `results/ledger.json` are prepared in `docs/launch/`:
- LinkedIn technical announcement
- X (Twitter) thread with architecture diagram
- GitHub README quickstart and demo GIF
