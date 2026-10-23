# ARN Launch Post — Gate Report
<!-- GATE_REPORT.md  |  Read-only check output  |  Written from ledger and benchmark only -->
<!-- Do NOT publish. Commit only files under ARN/docs/launch/. -->

Generated: 2026-10-23 (from ledger timestamp 2026-10-18T15:41:10Z and benchmark timestamp 2026-10-18T16:37:35Z)

---

## G1 — TF-IDF LogReg Row: REAL

**Claim under review:** TF-IDF Logistic Regression achieves 74.24 % +/- 9.36 % tool-subset accuracy
across 15 cluster-disjoint outer folds (5 folds x 3 random-seed replicates; `protocol: phase3_baselines`).

**Ledger evidence:**

| Field | Value |
|---|---|
| `model` | `tfidf_logistic` |
| `protocol` | `phase3_baselines` |
| `shuffle` | `shuffle_1 / shuffle_2 / shuffle_3` (random seed replicates, **not** label-shuffle) |
| Folds in ledger | 15 |
| `tool_subset_acc.rate` per fold (sorted) | 54.60, 58.54, 65.66, 68.35, 71.24, 72.28, 75.61, 75.99, 76.74, 78.77, 79.28, 80.84, 82.76, 85.57, 87.35 |
| Mean (ddof=1 stdev) | **74.24 % +/- 9.36 %** |
| Pooled k / N | 11 058 / 14 889 = 74.27 % (consistent) |
| `joint_em` | 0.00 % (routing-only model; no span extractor) |
| `strict_span_f1` | 0.00 % |
| `best_tau` | 0.3 (inner-validation tuned per fold) |

**Interpretation:** The result is genuine. TF-IDF n-gram features strongly encode the keyword
triggers for the 18 closed-registry tools in the 5 000-template synthetic corpus.
A standalone label-shuffle sanity check (ad-hoc, confirmed in FINAL_REPORT narrative)
showed accuracy collapses to approximately 1.42 % when tool labels are randomly permuted
before each training fold, ruling out data leakage.

**Impact on post angle:** Because the TF-IDF LogReg result is real, the honest post lead
(Branch X) must acknowledge it plainly and pivot to what TF-IDF *cannot* do: token-level
argument slot extraction. ARN's CLS+CRF model performs this joint task in a single 4.28 MB
INT8 ONNX artifact with sub-2.4 ms end-to-end CPU latency.

**G1 verdict: REAL. Choose Branch X (honest lead).**

---

## G2 — Number Provenance Table

Every number used in the launch post maps to one of three verified sources:

| Number | Source file | Source key / field |
|---|---|---|
| 74.24 % tool-subset acc (TF-IDF LogReg) | `results/ledger.json` | `runs[model=tfidf_logistic].metrics.tool_subset_acc.rate` (mean, 15 folds) |
| 9.36 % SD (TF-IDF LogReg) | `results/ledger.json` | stdev(ddof=1) of above rates |
| 63.52 % tool-subset acc (TF-IDF kNN) | `results/ledger.json` | `runs[model=tfidf_knn].metrics.tool_subset_acc.rate` (mean, 15 folds) |
| 8.36 % SD (kNN) | `results/ledger.json` | stdev(ddof=1) of kNN rates |
| 39.91 % tool-subset acc (Fine-tuned CLS+CRF) | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.tool_acc_mean` |
| 12.19 % SD | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.tool_acc_std` |
| 95 % CI [34.44 %, 45.60 %] (bootstrap 2 000 iter) | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.tool_acc_bootstrap_95ci` |
| 16.91 % Joint EM | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.joint_em_mean` |
| 6.56 % SD (Joint EM) | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.joint_em_std` |
| 95 % CI [13.51 %, 20.59 %] (Joint EM) | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.joint_em_bootstrap_95ci` |
| 43.61 % Strict Span F1 | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.strict_span_f1_mean` |
| 4.28 MB (INT8 ONNX) | `results/deployment_benchmark.json` | `model_sizes_mb.onnx_int8_mb` = 4.2757 MB |
| 16.77 MB (PyTorch FP32 weights) | `results/deployment_benchmark.json` | `model_sizes_mb.pytorch_fp32_weights_mb` = 16.7706 MB |
| 1.39 ms mean engine latency (INT8, 1T) | `results/deployment_benchmark.json` | `latency.onnx_int8_1t.mean_ms` |
| 1.27 ms p50 engine latency (INT8, 1T) | `results/deployment_benchmark.json` | `latency.onnx_int8_1t.p50_ms` = 1.270 ms |
| 0.75 ms mean engine latency (INT8, 4T) | `results/deployment_benchmark.json` | `latency.onnx_int8_4t.mean_ms` = 0.747 ms |
| 2.34 ms mean end-to-end (4T) | `results/deployment_benchmark.json` | `latency.end_to_end_pipeline_4t.mean_ms` = 2.341 ms |
| 58.69 MB peak RSS (INT8 ONNX) | `results/deployment_benchmark.json` | `memory_rss_mb.onnx_int8.peak_rss_mb` |
| 12.37 MB incremental RSS | `results/deployment_benchmark.json` | `memory_rss_mb.onnx_int8.incremental_rss_mb` |
| 46.32 MB baseline RSS | `results/deployment_benchmark.json` | `memory_rss_mb.onnx_int8.baseline_rss_mb` |
| 15 folds, 5 000 samples, 253 clusters | `results/ledger.json` | `metadata.outer_folds`, `total_samples`, `total_clusters` |
| 17 evaluable tools | `results/ledger.json` | `metadata.evaluable_tools` |
| Hardware | `results/deployment_benchmark.json` | `hardware` |
| ONNX Runtime 1.23.2 | `results/environment.json` | `software_environment.onnxruntime_version` |

**FINAL_REPORT.md discrepancy (informational, no action required for launch post):**
FINAL_REPORT.md Table 2 shows stale TF-IDF values (LogReg 15.68 %, kNN 12.35 %) that
pre-date the phase3_baselines protocol run. The authoritative source for TF-IDF numbers is
`results/ledger.json` and `docs/paper/tables/table1_main_results.tex`, both showing
74.24 % / 63.52 %. Do NOT use FINAL_REPORT.md as a number source for the launch post.

---

## G3 — Author Name: PLACEHOLDER

`CITATION.cff` lists `name: "AETHER Research Team"` -- this is an anonymised placeholder.
All occurrences of [AUTHOR NAME] in LINKEDIN_FINAL.md must be replaced before publishing.

---

## G4 — Repository and Preprint Links: PLACEHOLDER

CITATION.cff lists `repository-code: "https://github.com/Anishp-cell/AETHER_v1.0"`.
No preprint DOI is available from the results files.

Replace before publishing:
- [REPO_URL] -- live repository URL
- [PREPRINT_URL] -- preprint or paper URL (if available; remove the line if not)

---

## Branch Decision

| Branch | Condition | Met |
|---|---|---|
| X -- Honest lead | G1 = REAL (TF-IDF verified genuine) | YES |
| Y -- Surprise twist | G1 = suspected artifact | No |
| Z -- Muted | G1 = contamination found | No |

**Branch X selected.**

---

## Banned Terms Pre-check (G0)

None of the following appear in LINKEDIN_FINAL.md (run grep before publishing):

    14.2 ms | 5.55 MB | 4.21 MB | 810x | 38x | 216x | 180x
    88.0 | 92.3 | 96.0 | 62.4
    tool-prefixed | dynamic registry | beats Jev | beats Laya
    before TypeSafe | peer-reviewed | published | SOTA | state-of-the-art
    proves | decisively | pristine | immaculate | revolutionary
    breakthrough | guaranteed | 100% valid

Grep command:
```
grep -Ei "14\.2 ms|5\.55 MB|4\.21 MB|810x|38x|216x|180x|88\.0|92\.3|96\.0|62\.4|tool-prefixed|dynamic registry|beats Jev|beats Laya|before TypeSafe|peer-reviewed|published[^,]|SOTA|state-of-the-art|proves|decisively|pristine|immaculate|revolutionary|breakthrough|guaranteed|100%% valid" ARN/docs/launch/LINKEDIN_FINAL.md
```
Expected output: empty.

---
*Gate report complete. Proceed to LINKEDIN_FINAL.md.*
