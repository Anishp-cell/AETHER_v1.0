# Pre-registration: ARN-v2 confirmatory evaluation on multi-intent SLU benchmarks

Written and committed **before** any model is trained on these datasets and **before** any
test file is read. The commit timestamp of this file is the registration time. Anything not
fixed here is decided on the dev split only, and every deviation will be reported.

## 1. Why

ARN-v2 was developed on Google Mobile Actions, whose eval split was looked at during
development (decoder fixes were informed by eval errors of an early checkpoint; disclosed).
This evaluation tests whether the architecture generalises to benchmarks whose test sets
nobody on this project has seen, with no rule-based components.

## 2. Data (official files, unmodified)

Source: https://github.com/LooperXX/AGIF/tree/master/data (cleaned MixATIS / MixSNIPS of Qin et al., 2020).

| File | Sentences | SHA-256 |
|---|---|---|
| MixATIS_clean/train.txt | 13,162 | bc54ad5e842975dce9a62e06303ea6d9e89754f74c3e56c8ae9b6e5257d99012 |
| MixATIS_clean/dev.txt | 759 | c07df0e5687d1445ccebf7e1f23bf8f63ac0d9b0fce7a51592354981dcf60d3f |
| MixATIS_clean/test.txt | 828 | d68fdccdc2487e740c80f76164956604d594a4e86f8d492b434b64b407658f89 |
| MixSNIPS_clean/train.txt | 39,776 | bd05f8c6013a5df853f821124acafd0cf5eda272ddbb904ed075111efe51a4bc |
| MixSNIPS_clean/dev.txt | 2,198 | cadd26a93d5c19dde8e35aca9d37c68a343f628ffbe0515a8e6a47f34bc6c7a9 |
| MixSNIPS_clean/test.txt | 2,199 | dcc6eaff01d939ac749745b0b49066e675159e571ebe7296d70a615d89d7ea0f |

Only sentence counts were computed on dev/test files before this registration; their contents
were not displayed or analysed.

MASSIVE en-US (FitzGerald et al., 2022) is planned as a second confirmatory benchmark under the
same protocol; its file hashes will be appended in a separate commit before training on it.

## 3. System (fixed)

- ARN-v2 as implemented in `src/arn/v2/` (commit hash recorded per run in the ledger), adapted
  to SLU: intents play the role of tools (multi-label sigmoid router, FAR-v2); the dataset's own
  BIO slot tags are the CRF label set (hard BIO transition mask); decoding = intents with
  p >= 0.5 (the single most probable intent if none), slots = Viterbi path.
- **No canonicalizer, no keyword rules, no post-processing** beyond BIO decoding.
- Primary trunk: `google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny, ~4.4M params incl. heads).
  Secondary (reported, not headline): `google/bert_uncased_L-4_H-256_A-4` (~11M).
- Word-level evaluation: each word's prediction is its first word-piece's tag.

## 4. Training and selection (fixed)

- Hyperparameters identical to the Mobile Actions runs: AdamW, trunk LR 2e-4, head LR 2e-3,
  weight decay 0.01, batch 32, 20 epochs, cosine schedule with 6% warmup, max length 128
  word-pieces, loss = BCE(intents) + per-token-normalised CRF NLL (lambda = 1).
- Checkpoint selection: epoch with the best **dev** overall accuracy.
- Seeds: 42, 1, 2. The test split is decoded **once per seed**, after selection.

## 5. Metrics (as defined by Qin et al., 2020 and follow-up work)

- **Overall accuracy** (primary): the predicted intent set equals the gold set AND every slot tag
  is correct.
- Intent accuracy: exact intent-set match.
- Slot F1: span-level micro F1 (conlleval convention).
- Reported as mean ± SD over the 3 seeds, plus params and CPU latency (1 thread).

## 6. Comparators (published numbers, cited, not re-run)

| Model | MixATIS overall | MixSNIPS overall |
|---|---|---|
| AGIF | 40.8 | 74.2 |
| GL-GIN | 43.5 | 75.4 |
| Co-guiding | 51.3 | 77.5 |
| MISCA | 53.0 | 77.9 |
| MISCA + RoBERTa-base (~125M) | 59.1 | 86.2 |
| Uni-MIS | 52.5 | 83.4 |
| ECLM (Llama-3.1-8B, fine-tuned) | 56.2 | 86.5 |
| Llama-3.1-8B, plain fine-tune | 47.7 | 65.3 |
| ChatGPT, 5-shot | 14.6 | 12.7 |

## 7. Registered forecast (made before any run)

- MixSNIPS overall accuracy: median 77, 80% interval 70–82.
- MixATIS overall accuracy: median 43, 80% interval 35–50.

## 8. What will be claimed for each outcome

- Whatever the result, all three seeds and all metrics are reported. No seed is dropped.
- If ARN-v2 (BERT-Tiny) is within ~10 points of MISCA + RoBERTa on MixSNIPS: "a 4.4M model
  reaches the range of ~125M specialised models".
- If it falls below AGIF/GL-GIN: report it as a limitation of the size class on multi-intent SLU.
- No architecture or hyperparameter change will be made after the test numbers are seen. Any
  later variant is labelled post-hoc.
