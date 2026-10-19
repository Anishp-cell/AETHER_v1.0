# Work Package Prompt: WP-08 (WP-H) — Final Model Packaging & Release Artifact Integrity

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Medium (1 session, ~60 min)  
**Compute Needed:** CPU only (ONNX loading, hash verification)  
**Prerequisites:** WP-02, WP-04, and WP-05 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-H)
3. `ARN/checkpoints/arn_final_model/manifest.json`
4. `ARN/exports/`
5. `ARN/MODEL_CARD.md`
6. `ARN/docs/PYPI_NOTE.md`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- No fabrication: Checkpoint hashes must match exact file contents.
- Winning model architecture is Fine-Tuned CLS+CRF; model card must explicitly state this rather than marketing FAR.
- Legacy defective PyPI checkpoints must be explicitly documented with a deprecation notice.

---

### Objective & Scope

Verify the cryptographic integrity, runtime loadability, and architectural documentation of the final release model (`arn_final_model`), its ONNX exports (FP32 and INT8), and model card documentation.

---

### Exact Execution Steps

1. **Verify Checkpoint Cryptographic Hashes:**
   - Compute SHA-256 for `checkpoints/arn_final_model/model.pt` and `checkpoints/arn_final_model/config.json`.
   - Verify hashes match `checkpoints/arn_final_model/manifest.json`.
2. **Verify ONNX Export Loadability:**
   - Write and run a lightweight verification snippet loading `exports/arn_int8.onnx` and `exports/arn_fp32.onnx` using `onnxruntime.InferenceSession`.
   - Confirm dummy input tensor `input_ids = np.ones((1, 32), dtype=np.int64)` executes successfully without graph errors.
3. **Synchronize MODEL_CARD.md:**
   - Confirm Model Architecture describes: BERT-tiny trunk (fine-tuned) + CLS tool head + Linear-Chain CRF span decoder.
   - Confirm Parameter Count: 4.39M total.
   - Remove any legacy "4.21 MB" memory claims and insert peak/incremental RSS numbers.
   - Clarify intended use, known limitations (synthetic data only, 39.91% tool accuracy on unseen templates), and ethical considerations.
4. **Prepare PyPI Deprecation Notice (`docs/PYPI_NOTE.md`):**
   - Provide explicit markdown and text guidance explaining that the legacy `arn-orchestrator` v0.1.0 release on PyPI contained tagger masking bugs ([CLS]/[SEP] unmasked in CRF loss) and must be superseded or yanked.

---

### Acceptance Criteria (Pass / Fail)

- [ ] All checkpoint and export SHA-256 hashes verified against manifests.
- [ ] ONNX INT8 and FP32 runtime inference verified via automated script.
- [ ] `MODEL_CARD.md` accurately describes the winning model (CLS+CRF) and verified memory numbers.
- [ ] Git commit created with exact message: `fix: verify final model packaging and update model card`.
