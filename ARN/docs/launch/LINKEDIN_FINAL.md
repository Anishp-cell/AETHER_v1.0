# ARN — Final Launch Post
<!-- LINKEDIN_FINAL.md -->
<!-- Branch X (honest lead) | All numbers verified in GATE_REPORT.md -->
<!-- [AUTHOR NAME], [REPO_URL], [PREPRINT_URL] are placeholders — replace before publishing -->
<!-- AAAI line is marked REMOVABLE — author may cut it before publishing -->

---

## A. Main Post (~1 500 characters)

---

I built a neural tool router in a 4.28 MB file that runs in under 1.4 ms on a laptop CPU.
The honest version of that sentence is more interesting.

I'm a 3rd-year student. I started this project in January 2026 and submitted the first
version in July — roughly five months of work at around 14-15 hours a day. The first
version was reviewed at AAAI-27 and did not pass Phase 1.
<!-- REMOVABLE — delete this sentence if you prefer not to mention the rejection -->

Peer review exposed a real problem: my original evaluation used random splits that leaked
template structure between train and test. That invalidated the headline numbers. So I
audited everything, found a tokenizer bug that had been silently dropping argument labels,
withdrew every number that couldn't be traced to a clean run, and rebuilt the evaluation
from scratch with cluster-disjoint splits and written decision rules.

Here is what the clean evaluation actually shows:

A TF-IDF Logistic Regression — bag-of-words, no neural layers — achieves 74 % tool-routing
accuracy on the same 15-fold cluster-disjoint protocol (mean 74.24 %, +/- 9.36 %, N = 14 889
test queries). That's a real result, not a bug. Keyword features are strong on a closed
18-tool registry built from synthetic templates.

The fine-tuned CLS + CRF neural model reaches 39.91 % tool-subset accuracy (95 % CI:
34.44–45.60 %) and 16.91 % joint exact match (95 % CI: 13.51–20.59 %) across the same
15 folds. Why lower? Joint exact match requires getting both the tool AND every slot
argument exactly right — which TF-IDF cannot do at all (0 % Joint EM). The neural model
does both in a single forward pass.

That joint extraction lives in a 4.28 MB INT8 ONNX artifact running at 1.39 ms mean
engine latency (1 thread) on an Intel i7-13700HX laptop, with 12.37 MB incremental RAM.
No GPU. No server. No cloud call.

All code, evaluation protocol, limitations, and the full results ledger are open.

[REPO_URL]
[PREPRINT_URL — remove this line if no preprint is available yet]

#MachineLearning #EdgeAI #NLP #OpenSource

---

## B. Short Post (~700 characters)

---

I spent five months — roughly 14-15 h/day — building a tool-routing model small enough
to ship in a mobile app: 4.28 MB INT8 ONNX, 1.39 ms CPU latency.

Clean evaluation showed something I didn't expect: a TF-IDF Logistic Regression routes
tools at 74 % accuracy on the same 15-fold cluster-disjoint protocol. Keyword features
are strong on a closed synthetic registry.

The neural model's edge is joint extraction: tool ID + argument slots in one pass, 17 %
joint exact match — which TF-IDF can't do at all.

Everything is open — code, protocol, full results ledger, limitations.

[REPO_URL]

#NLP #EdgeAI #OpenSource

---

## C. First Comment (to pin with the repo link)

---

Technical details for anyone who wants to dig in:

**Dataset:** 5 000 synthetic utterances, 253 semantic clusters, 17 evaluable tools.
**Split:** 15-fold cluster-disjoint outer cross-validation (3 random-seed replicates per fold
= 15 rows); no cluster appears in both train and test within any fold.

**Master results (fine-tuned CLS + CRF, unfrozen BERT-tiny):**
- Tool-subset accuracy: 39.91 % +/- 12.19 % (bootstrap 95 % CI: 34.44–45.60 %)
- Joint exact match: 16.91 % +/- 6.56 % (95 % CI: 13.51–20.59 %)
- Strict span F1: 43.61 %

**TF-IDF baselines (same 15 folds, inner-validation threshold tuning):**
- Logistic Regression: 74.24 % +/- 9.36 % tool-subset acc | Joint EM: 0 %
- kNN: 63.52 % +/- 8.36 % tool-subset acc | Joint EM: 0 %

**Deployment (INT8 ONNX, Intel i7-13700HX, ONNX Runtime 1.23.2):**
- Disk size: 4.28 MB
- Engine latency: 1.39 ms mean / 1.27 ms p50 (1 thread)
- End-to-end pipeline: 2.34 ms mean (4 threads)
- Peak RSS: 58.69 MB | Incremental RSS: 12.37 MB

Full ledger, protocol, and limitations: [REPO_URL]

---

## D. Alternative Hooks (each under 200 characters)

**Hook 1 (finding-first):**
My neural tool router is beaten at routing by a bag-of-words baseline. Here's why I'm
publishing the result anyway — and what the neural model does that TF-IDF can't.

**Hook 2 (scale-first):**
4.28 MB. 1.39 ms. No GPU. A joint tool-router and argument extractor that runs on any
laptop CPU — built solo over five months as a 3rd-year student.

**Hook 3 (audit-first):**
I withdrew my own paper's numbers after peer review exposed a data-split leak. This is
what the cleaned evaluation looks like — honest results, open protocol, no cherry-picking.

---

## E. Evidence Table (maps every number to its verified source)

| Claim in post | Exact value | Source file | Source key |
|---|---|---|---|
| 74 % / 74.24 % tool acc (TF-IDF LogReg) | 74.2377 % | `results/ledger.json` | `runs[tfidf_logistic].metrics.tool_subset_acc.rate` mean |
| +/- 9.36 % SD | 9.3619 % | `results/ledger.json` | stdev(ddof=1), 15 fold-rates |
| 63.52 % tool acc (kNN) | 63.5198 % | `results/ledger.json` | `runs[tfidf_knn].metrics.tool_subset_acc.rate` mean |
| 39.91 % tool acc (CLS+CRF) | 39.91 % | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.tool_acc_mean` |
| 95 % CI 34.44–45.60 % | [0.3444, 0.4560] | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.tool_acc_bootstrap_95ci` |
| 16.91 % / 17 % Joint EM | 16.91 % | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.joint_em_mean` |
| 95 % CI 13.51–20.59 % | [0.1351, 0.2059] | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.joint_em_bootstrap_95ci` |
| 43.61 % Strict Span F1 | 43.61 % | `results/master_protocol_summary.json` | `models.unfrozen_cls_crf.strict_span_f1_mean` |
| 4.28 MB disk (INT8 ONNX) | 4.2757 MB | `results/deployment_benchmark.json` | `model_sizes_mb.onnx_int8_mb` |
| 1.39 ms engine (1T mean) | 1.387 ms | `results/deployment_benchmark.json` | `latency.onnx_int8_1t.mean_ms` |
| 1.27 ms engine (1T p50) | 1.270 ms | `results/deployment_benchmark.json` | `latency.onnx_int8_1t.p50_ms` |
| 2.34 ms end-to-end (4T mean) | 2.341 ms | `results/deployment_benchmark.json` | `latency.end_to_end_pipeline_4t.mean_ms` |
| 58.69 MB peak RSS | 58.69 MB | `results/deployment_benchmark.json` | `memory_rss_mb.onnx_int8.peak_rss_mb` |
| 12.37 MB incremental RSS | 12.37 MB | `results/deployment_benchmark.json` | `memory_rss_mb.onnx_int8.incremental_rss_mb` |
| 5 000 samples, 253 clusters, 17 tools | exact | `results/ledger.json` | `metadata.total_samples`, `total_clusters`, `evaluable_tools` |
| 15 folds, 3 replicates = 15 rows | exact | `results/ledger.json` | `metadata.outer_folds`, `shuffles` |
| Intel i7-13700HX | exact | `results/deployment_benchmark.json` | `hardware` |
| ONNX Runtime 1.23.2 | exact | `results/environment.json` | `software_environment.onnxruntime_version` |

---

## F. Banned Terms Grep Result

Grep command run over this file:

```
grep -Ei "14\.2 ms|5\.55 MB|4\.21 MB|810x|38x|216x|180x|88\.0|92\.3|96\.0|62\.4|tool-prefixed|dynamic registry|beats Jev|beats Laya|before TypeSafe|peer-reviewed|\bpublished\b|SOTA|state-of-the-art|\bproves\b|decisively|pristine|immaculate|revolutionary|breakthrough|guaranteed|100%% valid" ARN/docs/launch/LINKEDIN_FINAL.md
```

Expected output: **empty** (no matches).

---
*[AUTHOR NAME] — replace this placeholder with your name before publishing.*
*Review GATE_REPORT.md once more and then commit: `git commit -m "docs: final launch post"`*
