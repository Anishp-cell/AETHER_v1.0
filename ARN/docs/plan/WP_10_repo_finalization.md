# Work Package Prompt: WP-10 (WP-J) — Repository Finalization & Public Release Staging

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Medium (1 session, ~60 min)  
**Compute Needed:** CPU only (file exports, tests, environment validation)  
**Prerequisites:** WP-08 and WP-09 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-J)
3. `ARN/README.md`
4. `ARN/setup.py` / `ARN/pyproject.toml`
5. `ARN/requirements.txt`
6. `ARN/.gitignore`
7. `ARN/RELEASE_CHECKLIST.md` (if existing)

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/` and the external export path.
- CRITICAL RELEASE RULE: `ARN-public/` MUST be staged OUTSIDE the `AETHER_V1.0` workspace (e.g., at `d:\python\ARN-public` or `..\ARN-public`), completely decoupled from parent repo history.
- Never publish or push directly to GitHub or PyPI from this session. Staging only.
- Git identity in public export must use a clean GitHub noreply email (no personal email exposure).

---

### Objective & Scope

Prepare the ARN repository and public staging export for public release. Finalize the README with prominent limitations, pin exact dependency versions, verify clean-room installation, export the public repository outside the Aether root, and generate a step-by-step human release checklist.

---

### Exact Execution Steps

1. **Rewrite README.md for Public Truthfulness:**
   - Prominent banner: *"Preprint / Research Artifact — Not Peer-Reviewed"*.
   - Prominent Limitations section right after introduction:
     - Synthetic training distribution (5,000 utterances).
     - 39.91% tool subset accuracy on unseen template structures.
     - Laptop x86 deployment figures (not embedded/mobile ARM).
   - Verbatim CLI session output (generated from real inference calls, never mocked).
   - Accurate memory and latency tables matching `deployment_benchmark.json`.
2. **Pin Dependencies & Verify Packaging:**
   - Review `requirements.txt` and `setup.py`: pin exact compatible versions (e.g., `torch`, `transformers`, `onnxruntime`, `numpy`).
   - Ensure `arn-cli` entry point is correctly registered.
3. **Stage Clean Public Repository (`ARN-public`):**
   - Verify that any existing `d:\python\AETHER_V1.0\ARN-public` is removed from the Aether workspace.
   - Stage the export at `d:\python\ARN-public\` containing only:
     - Cleaned source code (`arn/`)
     - Pre-trained checkpoints (`checkpoints/arn_final_model/`)
     - ONNX exports (`exports/`)
     - Core test suite (`tests/`)
     - Public documentation (`README.md`, `MODEL_CARD.md`, `LICENSE`, `CITATION.cff`)
     - Clean git history with noreply author identity.
4. **Secrets & PII Scan:**
   - Run grep for personal emails (`anishpathak778@gmail.com`), internal paths (`d:\python\AETHER_V1.0`), and API keys across the staged repository.
5. **Create RELEASE_CHECKLIST.md:**
   - Document the exact manual commands the human author must execute to push to GitHub, register Zenodo DOI, and handle legacy PyPI package yanking.

---

### Acceptance Criteria (Pass / Fail)

- [ ] `README.md` features prominent "Preprint" and "Limitations" headers.
- [ ] Staged `ARN-public` directory exists strictly OUTSIDE `AETHER_V1.0`.
- [ ] No personal emails or confidential workspace paths exist in the staged public repository.
- [ ] `RELEASE_CHECKLIST.md` contains step-by-step instructions for the author.
- [ ] Git commit created with exact message: `feat: finalize repo for public release`.
