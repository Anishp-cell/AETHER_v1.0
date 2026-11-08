# Aether Routing Network v2: results and disclosures

Single source for every v2 number. Each number is recomputable from the prediction files in
`results/xmodel/` with `src/arn/eval/mobile_actions_scorer.py` (Mobile Actions) or from
`results/slu/*.json` (MixATIS/MixSNIPS). Environment: `results/environment_v2.json`
(RTX 5050 Laptop GPU sm_120, PyTorch 2.12.0.dev20260408+cu128 nightly, transformers 5.5.3).

## 1. Google Mobile Actions (eval split, 961 rows, strict exact match)

Strict exact match = ordered call list, identical names, identical non-null argument key sets,
equal values after lower-casing and stripping quotes/dots; datetime compared as full ISO string.

| System | Params | Trained on the train split | Strict EM | 95% CI | Source |
|---|---|---|---|---|---|
| LFM2.5-230M, fine-tuned, best of 5-config sweep (selected on val) | 230M | yes | 92.51 | — | ours |
| LFM2.5-230M, fine-tuned (default recipe) | 230M | yes | 91.57 | [89.80, 93.34] | ours |
| **Aether Routing Network v2, BERT-Tiny + FAR-v2** | **4.44M** | yes | **89.32 ± 0.47** (seeds 42/1/2: 89.28, 89.80, 88.87) | s42 [87.41, 91.26] | ours |
| ARN-v2, BERT-Mini trunk (3 seeds) | 11.38M | yes | 89.91 ± 0.46 (89.59, 89.70, 90.43) | s42 [87.62, 91.47] | ours |
| ARN-v2, [CLS] router (3 seeds) | 4.39M | yes | 89.07 ± 0.38 (88.97, 89.49, 88.76) | s42 [86.99, 91.05] | ours |
| FunctionGemma-270M, fine-tuned (same split, same recipe as LFM2.5 default) | 268M | yes | 85.12 | — | ours (Google reports 85) |
| LFM2.5-230M, base | 230M | no | 68.05 | [65.14, 71.18] | ours (69.09 under the lenient subset rule; Cactus reports 69.1) |
| Qwen3-0.6B, base, thinking off | 596M | no | 65.04 | [61.81, 68.05] | ours |
| FunctionGemma-270M, base | 268M | no | 60.15 | — | ours (Google reports 58) |
| ARN v1 (legacy) | 4.4M | yes | 49.74 | — | legacy repo |

Vendor-reported, other harnesses (not paired with ours): DeepSeek V4 Flash 88.4, Needle 3 (121M) 86.0,
LFM2.5-1.2B 82.4, Qwen3.5-0.8B 76.0, LFM2.5-350M 72.8, Needle 2 (45M) 63.5, Apple FM (~3B) 57.6 (Cactus charts).

**Paired bootstrap (2,000 resamples over eval rows):**
- ARN-v2 − FunctionGemma-FT: +4.16 [+1.77, +6.66] (s42), +4.68 [+2.19, +7.08] (s1), +3.75 [+1.25, +6.24] (s2).
- ARN-v2 (s42) − LFM2.5-FT (default): −2.29 [−4.27, −0.31].

**LFM2.5 equal-budget sweep (val strict EM, 869 rows):** lr 5e-5/2 ep 90.56; 2e-5/2 ep 90.91; 1e-4/2 ep 90.45;
**5e-5/3 ep 91.02 (selected)**; 1e-4/3 ep 89.64. ARN-v2 val strict EM: 91.02 / 89.64 / 91.02 (s42/s1/s2).

**Robustness checks:**
- Train/eval overlap: 10 eval utterances (1.0%) duplicate a train utterance, 36 (3.7%) have a train neighbour with word-trigram Jaccard ≥ 0.8. On the 902 rows with nearest-neighbour Jaccard < 0.7: ARN-v2 s42 88.69, LFM2.5-FT 91.02.
- Same rules for baselines (ARN's canonicalizer applied to their datetime values): LFM2.5 base 68.05 → 71.90; Qwen3 base 65.04 → 72.53.
- Scorer sanity: gold-as-prediction 100.00.

**Ablations (three final checkpoints, `results/key_results_v2.json`, `scripts/aggregate_key_results.py`):**

| Variant | Seeds 42 / 1 / 2 | Mean ± SD | Seed-matched paired Δ vs full (95% CI) |
|---|---|---|---|
| Full | 89.28 / 89.80 / 88.87 | 89.32 ± 0.47 | — |
| No canonicalizer | 55.25 / 55.57 / 54.84 | 55.22 ± 0.37 | −34.03 [−37.04, −31.11]; −34.24 [−37.25, −31.32]; −34.03 [−37.04, −31.22] |
| FAR-attention call ordering | 78.77 / 89.80 / 88.87 | 85.81 ± 6.12 | −10.51 [−12.49, −8.64]; 0.00; 0.00 |
| No multi-span join | 88.24 / 88.55 / 87.72 | 88.17 ± 0.42 | −1.04 [−1.87, −0.31]; −1.25 [−2.08, −0.42]; −1.14 [−1.98, −0.42] |
| No registry mask in Viterbi | 89.18 / 89.70 / 88.76 | 89.21 ± 0.47 | −0.10 [−0.31, 0.00] each |

FAR-attention ordering on seeds 1 and 2 changed no prediction (verified: 0 of 961 rows differ, although the
attention anchors differ from keyword anchors; the induced call order is the same), on seed 42 it reordered 106 rows.
It is seed-unstable, which is why keyword ordering is used.

**Variants (seed-matched paired Δ):** FAR-v2 − [CLS]: +0.31 [−0.83, +1.46], +0.31 [−0.83, +1.56], +0.10 [−1.04, +1.25]
(no significant difference). BERT-Mini − BERT-Tiny: +0.31 [−0.83, +1.56], −0.10 [−1.04, +0.94], +1.56 [+0.31, +2.81].

## 2. Pre-registered confirmatory benchmarks (commit `1fa7a50`, test read once per seed)

| Benchmark | Metric | ARN-v2 (BERT-Tiny, 4.48M) | Seeds 42 / 1 / 2 | Intent acc | Slot F1 |
|---|---|---|---|---|---|
| MixATIS (828) | overall acc | **40.94 ± 2.62** | 43.96 / 39.61 / 39.25 | 68.88 | 86.68 |
| MixSNIPS (2,199) | overall acc | **70.29 ± 0.46** | 69.76 / 70.58 / 70.53 | 92.41 | 93.70 |

Registered forecast: MixATIS median 43 (35–50): **inside**. MixSNIPS median 77 (70–82): **at the lower edge**.

Published comparators (overall acc, MixATIS / MixSNIPS): AGIF 40.8 / 74.2; GL-GIN 43.5 / 75.4;
Co-guiding 51.3 / 77.5; MISCA 53.0 / 77.9; MISCA + RoBERTa-base 59.1 / 86.2; Uni-MIS 52.5 / 83.4;
ECLM (Llama-3.1-8B FT) 56.2 / 86.5; Llama-3.1-8B plain FT 47.7 / 65.3; ChatGPT 5-shot 14.6 / 12.7.
AGIF and GL-GIN are BiLSTM models of roughly 0.6–0.9M parameters (`results/baseline_param_estimates.json`),
i.e. **smaller** than ARN-v2; these benchmarks test generality without rules, not size efficiency.

## 3. Latency (`results/latency_v2.json`)

Single-stream, batch 1, FP32 PyTorch, CPU (Intel, see environment file), warmup discarded, p50 / p90,
1 and 4 threads. ARN-v2 is timed end to end, including tokenization through the current (unoptimized)
preprocessing path; model + decoding alone was ~3.8 ms p50 (1 thread) in the training logs.
No optimized LM runtime (llama.cpp / int8) was used; LM latencies are upper bounds for optimized deployment.

| System (1 thread) | p50 | p90 |
|---|---|---|
| ARN-v2, end to end | 11.40 ms | 18.22 ms |
| LFM2.5-230M | 4,315.97 ms | 9,249.54 ms |
| other rows | see `results/latency_v2.json` | |

## 4. Disclosures

1. **Eval inspection during development.** Decoder changes (call ordering, trailing-punctuation stripping,
   multi-span de-duplication) were first motivated by eval-split errors of a 1-epoch checkpoint; a few eval
   datetime failures of the legacy resolver were viewed before the canonicalizer was written. Each change was
   confirmed on the validation split and/or train split (ordering rule 99.4% on train; canonicalizer 97.36% train
   vs 97.26% eval). Mobile Actions has no separate dev/test split; its eval split therefore served as a
   development test set. The pre-registered MixATIS/MixSNIPS evaluation is the confirmatory evidence.
2. **Rule-based components.** The datetime canonicalizer (conventions fitted on train only) and keyword-based
   call ordering contribute ~34 points (3-seed ablation) and up to ~10 points (seed-dependent) respectively. The system is neuro-symbolic.
3. **External code review** (Antigravity CLI, Gemini 3.8 Flash) found a lenient scorer rule (extra arguments
   accepted); fixed, all files re-scored, no ARN-v2 number changed.
4. **Closed registry.** ARN-v2 needs training data for a fixed tool set, cannot call the same tool twice in one
   utterance (0 eval rows require it), and is not applicable to BFCL (no train split, per-item tools).
5. **Seeds:** all ARN-v2 Mobile Actions variants are reported over 3 seeds (42, 1, 2); baselines are single runs.
6. **Baseline tuning:** LFM2.5 received a 5-config sweep selected on validation; FunctionGemma and Qwen3 did not.

## 5. Planned before running (pre-specified 2026-11-08, committed before any run)

**Key result 1: decoder ablations on the three final checkpoints** (`arn_v2_tiny_far_s42_newdec`, `..._far_s1`,
`..._far_s2`), re-decoded with `scripts/v2_redecode.py`: full, `--no-canon`, `--anchor far`,
`--multi-span-min-rate 2`, `--no-constrained`. Report mean ± SD and a seed-matched paired bootstrap vs full.
A "generic (non-tool-qualified) tags" ablation is **not run**: every Mobile Actions argument name belongs to exactly one
tool, so generic and tool-qualified tag sets are identical on this dataset (stated as a limitation of the evidence for
contribution 1).

**Key result 2: three seeds for single-seed variants**: BERT-Tiny + [CLS] router seeds 1, 2; BERT-Mini + FAR-v2 seeds 1, 2;
same recipe (20 epochs, val selection, eval decoded once). Report mean ± SD and seed-matched paired differences
(FAR-v2 vs [CLS], Tiny vs Mini). Reading rule: if the 95% CI of a paired difference includes 0, report "no significant
difference".
