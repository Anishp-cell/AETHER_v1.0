"""Aggregate the two pre-specified key results (docs/RESULTS_v2.md section 5).

Stdlib only. Reads prediction files in results/xmodel/, scores them with the shared scorer and
writes results/key_results_v2.json: mean +/- SD over seeds and seed-matched paired bootstrap
(2,000 resamples over eval rows) for ablations vs full, [CLS] vs FAR-v2 and Mini vs Tiny.

Usage: python scripts/aggregate_key_results.py <path/to/mobile-actions/dataset.jsonl>
"""
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sc", ROOT / "src/arn/eval/mobile_actions_scorer.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
X = ROOT / "results/xmodel"

TINY = {"42": "arn_v2_tiny_far_s42_newdec", "1": "arn_v2_bert_uncased_L-2_H-128_A-2_far_s1",
        "2": "arn_v2_bert_uncased_L-2_H-128_A-2_far_s2"}
CLS = {s: f"arn_v2_bert_uncased_L-2_H-128_A-2_cls_s{s}" for s in ("42", "1", "2")}
MINI = {s: f"arn_v2_bert_uncased_L-4_H-256_A-4_far_s{s}" for s in ("42", "1", "2")}
ABLS = ["full", "nocanon", "faranchor", "nomultispan", "unconstrained"]


def correct(rows, name):
    preds = json.loads((X / f"{name}.json").read_text(encoding="utf-8"))["predictions"]
    return [int(sc.row_em(preds.get(str(i)) or [], sc.gold_calls(r))) for i, r in enumerate(rows)]


def em(v):
    return round(100 * sum(v) / len(v), 2)


def paired(a, b, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    boots = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"delta": round(100 * sum(d) / n, 2), "ci95": [round(boots[int(.025 * n_boot)], 2),
                                                        round(boots[int(.975 * n_boot) - 1], 2)]}


def summary(per_seed):
    v = list(per_seed.values())
    return {"per_seed": per_seed, "mean": round(statistics.mean(v), 2), "sd": round(statistics.stdev(v), 2)}


def main():
    rows = sc.load_eval_rows(sys.argv[1])
    out = {"n_eval": len(rows), "ablations": {}, "variants": {}}
    tiny = {s: correct(rows, n) for s, n in TINY.items()}
    for ab in ABLS:
        c = {s: correct(rows, f"{n}_abl_{ab}") for s, n in TINY.items()}
        out["ablations"][ab] = summary({s: em(v) for s, v in c.items()})
        out["ablations"][ab]["paired_vs_full"] = {s: paired(c[s], tiny[s]) for s in TINY}
    for label, names in (("far_tiny", TINY), ("cls_tiny", CLS), ("far_mini", MINI)):
        out["variants"][label] = summary({s: em(correct(rows, n)) for s, n in names.items()})
    cls = {s: correct(rows, n) for s, n in CLS.items()}
    mini = {s: correct(rows, n) for s, n in MINI.items()}
    out["variants"]["paired_far_minus_cls"] = {s: paired(tiny[s], cls[s]) for s in TINY}
    out["variants"]["paired_mini_minus_tiny"] = {s: paired(mini[s], tiny[s]) for s in TINY}
    (ROOT / "results/key_results_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
