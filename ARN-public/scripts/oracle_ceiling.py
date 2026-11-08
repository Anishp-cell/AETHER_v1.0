"""
Extractive-ceiling analysis for Mobile-Actions (plan Phase 1 sanity + Phase 2/3 targets).

Builds oracle prediction files and scores them with the shared scorer:
  gold                 - gold calls verbatim (scorer sanity check; must be 100%)
  extractive_legacy_dt - gold tool names; non-datetime args only if verbatim in the
                         utterance (copied from the utterance text); datetime via the
                         legacy regex+dateutil resolver is NOT reimplemented here, so
                         datetime args are left empty (lower bound)
  extractive_canon     - same, datetime via the train-fitted temporal canonicalizer
  extractive_canon_ms  - same, plus multi-span joins (gold value = two verbatim
                         pieces of the utterance joined by a space)

These are ceilings for a perfect span tagger + perfect router; real models land below.

Usage:
    python scripts/oracle_ceiling.py --data dataset.jsonl --policy results/temporal_policy.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("arn_canon_temporal", "src/arn/canon/temporal.py")
S = _load("arn_eval_scorer", "src/arn/eval/mobile_actions_scorer.py")


def verbatim(text: str, value: str):
    i = text.lower().find(value.lower())
    return text[i:i + len(value)] if i >= 0 else None


def two_span(text: str, value: str):
    words = value.split()
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        ia = text.lower().find(a.lower())
        ib = text.lower().find(b.lower(), ia + len(a) if ia >= 0 else 0)
        if ia >= 0 and ib >= 0:
            return text[ia:ia + len(a)] + " " + text[ib:ib + len(b)]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--policy", default="results/temporal_policy.json")
    ap.add_argument("--outdir", default="results/xmodel")
    args = ap.parse_args()

    pol_d = json.load(open(args.policy))["policy"]
    pol = T.TemporalPolicy(pol_d["next_weekday"], pol_d["bare_weekday_allows_today"],
                           pol_d["roll_year_forward"], tuple(pol_d["default_time"]))
    rows = S.load_eval_rows(args.data)
    variants = {k: {} for k in ("gold", "extractive_no_dt", "extractive_canon", "extractive_canon_ms")}
    for i, r in enumerate(rows):
        msgs = r["messages"]
        user = next(m["content"] for m in msgs if m.get("role") == "user")
        ref = T.parse_reference(msgs[0].get("content", ""))
        gold = S.gold_calls(r)
        variants["gold"][str(i)] = gold
        for v in ("extractive_no_dt", "extractive_canon", "extractive_canon_ms"):
            calls = []
            for g in gold:
                a = {}
                for k, val in g["arguments"].items():
                    if k == "datetime":
                        a[k] = None if v == "extractive_no_dt" else T.canonicalize(user, ref, pol)
                    else:
                        x = verbatim(user, str(val))
                        if x is None and v == "extractive_canon_ms":
                            x = two_span(user, str(val))
                        a[k] = x
                calls.append({"name": g["name"], "arguments": a})
            variants[v][str(i)] = calls

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, preds in variants.items():
        (out / f"oracle_{name}.json").write_text(json.dumps({"model": f"oracle_{name}", "params": None, "predictions": preds}))
        res = S.score(rows, preds)
        summary[name] = {k: res[k] for k in ("strict_em", "single_call_em", "multi_call_em", "datetime_arg_acc")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
