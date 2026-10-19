# Work Package Prompt: WP-05 (WP-E) — Memory & Latency Protocol Finalization

**Session Type:** Fresh standalone agent session  
**Estimated Session Size:** Small-Medium (1 session, ~45 min)  
**Compute Needed:** CPU only (benchmark verification and documentation)  
**Prerequisites:** WP-02 and WP-03 completed  

---

### Instructions for Agent

You are an ML engineer and research-integrity auditor. You are operating in a fresh session with NO prior chat history. Do NOT assume any prior state from conversation memory.

**First Step — Read These Files:**
1. `ARN/STATE.md`
2. `ARN/docs/plan/IMPLEMENTATION_PLAN.md` (Section 3: WP-E)
3. `ARN/results/deployment_benchmark.json`
4. `ARN/scripts/measure_memory.py`
5. `ARN/docs/paper/tables/table2_hardware.tex`
6. `ARN/docs/paper/numbers.tex`
7. `ARN/MODEL_CARD.md`

**Non-Negotiable Project Rules:**
- Scope is strictly limited to `ARN/`.
- No fabrication: every memory and latency figure must match `results/deployment_benchmark.json`.
- Zero banned claims (no "4.21 MB RAM", no "5.55 MB RAM", no "14.2 ms ARM").
- Hardware setup MUST be explicitly qualified: "Intel Core i7-13700HX, Windows 11, single-socket x86 laptop CPU".
- ARM, phone, and embedded hardware MUST be explicitly stated as UNMEASURED.

---

### Objective & Scope

Standardize memory and latency reporting across the entire project. Ensure the memory protocol is mathematically and empirically transparent (distinguishing between baseline, peak, and incremental process RSS), and that device constraints are accurately disclaimed.

---

### Exact Execution Steps

1. **Verify Benchmark Metrics:**
   Verify the contents of `results/deployment_benchmark.json`:
   - ONNX Runtime INT8:
     - Engine latency: 1.39 ms (1 thread), 0.75 ms (4 threads).
     - End-to-end latency: 3.10 ms (1 thread), 2.34 ms (4 threads).
     - Process Peak RSS: 58.69 MB.
     - Process Incremental RSS: 12.37 MB.
     - Model File Size: 4.28 MB.
   - ONNX Runtime FP32:
     - Peak RSS: 70.75 MB, Incremental RSS: 24.21 MB.
     - Model File Size: 16.75 MB.
   - PyTorch FP32:
     - Peak RSS: 760.69 MB, Incremental RSS: 81.49 MB.
2. **Document Measurement Protocol in Paper & Model Card:**
   - Define Baseline RSS: Python process RSS after importing runtime libraries (`onnxruntime`, `numpy`) prior to model load.
   - Define Peak RSS: Maximum resident set size observed during model loading and warm inference iterations ($N=100$).
   - Define Incremental RSS: $\text{Incremental} = \text{Peak} - \text{Baseline}$.
   - Add explicit disclaimer: *"All measurements were collected on a laptop workstation equipped with an Intel Core i7-13700HX CPU running Windows 11. No mobile ARM or edge hardware (Raspberry Pi, Cortex-M, Android NPU) was benchmarked."*
3. **Document Quantization Impact:**
   - Confirm INT8 vs FP32 accuracy delta is documented (zero drop in tool accuracy, $<0.15$ F1 change on argument spans).
4. **Synchronize Table 2 & numbers.tex:**
   - Inspect `docs/paper/tables/table2_hardware.tex` to ensure it reports both Peak and Incremental RSS columns clearly labeled.
   - Run `python scripts/audit_paper_numbers.py` to confirm zero LaTeX macro mismatches.

---

### Acceptance Criteria (Pass / Fail)

- [ ] Table 2 correctly presents both Peak RSS and Incremental RSS for all three runtimes.
- [ ] No mention of "4.21 MB" or "5.55 MB" exists in any hardware table or text.
- [ ] Hardware environment (i7-13700HX x86) and unmeasured ARM disclaimer are present in paper text.
- [ ] `scripts/audit_paper_numbers.py` passes without warnings.
- [ ] Git commit created with exact message: `fix: finalize memory/latency reporting with correct methodology and caveats`.
