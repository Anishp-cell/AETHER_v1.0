"""
Model-agnostic scorer for google/mobile-actions (plan Phase 1).

Every system (ARN, FunctionGemma, LFM2.5, Qwen3, EmbeddingGemma router, ...)
writes predictions in one format and is scored by this single function, so
leaderboard numbers are comparable by construction.

Prediction file format (JSON):
    {"model": "<name>", "params": <int or null>,
     "predictions": {"<eval_row_index>": [{"name": str, "arguments": {..}}, ...]}}
`eval_row_index` is the 0-based position of the row among rows whose
metadata == "eval", in dataset file order.

Strict EM matches ARN/eval_mobile_actions.py (ordered calls, same names, every
non-null gold argument present and equal after lower/strip of quotes and dots;
datetime compared as the full ISO string). Relaxed EM differs from the legacy
script on purpose: legacy relaxed silently passed every string datetime; here
relaxed compares the date part only. An order-insensitive EM is also reported.

Usage (stdlib-only; run the file directly):
    python src/arn/eval/mobile_actions_scorer.py --data dataset.jsonl preds.json [...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional


def load_eval_rows(dataset_path: str) -> List[dict]:
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("metadata") == "eval":
                    rows.append(r)
    return rows


def gold_calls(row: dict) -> List[dict]:
    out = []
    for tc in row["messages"][-1].get("tool_calls") or []:
        fn = tc["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        out.append({"name": fn["name"], "arguments": {k: v for k, v in args.items() if v is not None}})
    return out


def _clean(v: Any) -> str:
    return str(v).lower().strip().strip("'\".")


def _iso(v: Any) -> str:
    return v.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(v, "strftime") else str(v).strip()


def call_matches(pred: dict, gold: dict, relaxed: bool = False, allow_extra: bool = False) -> bool:
    """Exact call match. Argument KEY SETS must be equal (no extra or missing non-null
    arguments), as in Google's FunctionGemma evaluation. `allow_extra=True` reproduces the
    earlier, more lenient rule (gold args must be a subset); it is reported only as
    `strict_em_legacy_subset` for comparability with numbers computed before this fix."""
    if pred.get("name") != gold["name"]:
        return False
    p_args = {k: v for k, v in (pred.get("arguments") or {}).items() if v is not None}
    if not allow_extra and set(p_args) != set(gold["arguments"]):
        return False
    for k, g in gold["arguments"].items():
        p = p_args.get(k)
        if p is None:
            return False
        if k == "datetime":
            ps, gs = _iso(p), _iso(g)
            if (ps[:10] != gs[:10]) if relaxed else (ps[:19] != gs[:19]):
                return False
        elif _clean(p) != _clean(g):
            return False
    return True


def row_em(pred_calls: List[dict], gold: List[dict], relaxed: bool = False, ordered: bool = True,
           allow_extra: bool = False) -> bool:
    if len(pred_calls) != len(gold):
        return False
    if ordered:
        return all(call_matches(p, g, relaxed, allow_extra) for p, g in zip(pred_calls, gold))
    remaining = list(pred_calls)
    for g in gold:
        hit = next((i for i, p in enumerate(remaining) if call_matches(p, g, relaxed, allow_extra)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def score(rows: List[dict], predictions: Dict[str, List[dict]]) -> Dict[str, Any]:
    n = len(rows)
    c: Counter = Counter()
    missing = 0
    for i, row in enumerate(rows):
        g = gold_calls(row)
        p = predictions.get(str(i))
        if p is None:
            missing += 1
            p = []
        strict = row_em(p, g)
        c["strict"] += strict
        c["strict_legacy"] += row_em(p, g, allow_extra=True)
        c["relaxed"] += row_em(p, g, relaxed=True)
        c["unordered"] += row_em(p, g, ordered=False)
        c["names"] += [x.get("name") for x in p] == [x["name"] for x in g]
        c["name_sets"] += sorted(str(x.get("name")) for x in p) == sorted(x["name"] for x in g)
        bucket = "multi" if len(g) > 1 else "single"
        c[f"{bucket}_n"] += 1
        c[f"{bucket}_strict"] += strict
        remaining = list(p)  # consume matched calls so repeated tools pair up positionally
        for gc in g:
            j = next((k for k, x in enumerate(remaining) if x.get("name") == gc["name"]), None)
            pc = remaining.pop(j) if j is not None else None
            if "datetime" in gc["arguments"]:
                c["dt_n"] += 1
                pv = ((pc or {}).get("arguments") or {}).get("datetime")
                c["dt_ok"] += pv is not None and _iso(pv)[:19] == _iso(gc["arguments"]["datetime"])[:19]
    pct = lambda a, b: round(100.0 * a / b, 2) if b else None  # noqa: E731
    return {
        "n_rows": n,
        "missing_predictions": missing,
        "strict_em": pct(c["strict"], n),
        "strict_em_legacy_subset": pct(c["strict_legacy"], n),
        "relaxed_em_date_only": pct(c["relaxed"], n),
        "unordered_em": pct(c["unordered"], n),
        "tool_sequence_acc": pct(c["names"], n),
        "tool_set_acc": pct(c["name_sets"], n),
        "single_call_em": pct(c["single_strict"], c["single_n"]),
        "multi_call_em": pct(c["multi_strict"], c["multi_n"]),
        "datetime_arg_acc": pct(c["dt_ok"], c["dt_n"]),
        "counts": dict(c),
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Score Mobile-Actions predictions (model-agnostic).")
    ap.add_argument("--data", required=True, help="mobile-actions dataset.jsonl")
    ap.add_argument("pred_files", nargs="+")
    args = ap.parse_args(argv)
    rows = load_eval_rows(args.data)
    for pf in args.pred_files:
        obj = json.load(open(pf, encoding="utf-8"))
        res = score(rows, obj["predictions"])
        print(json.dumps({"model": obj.get("model", pf), "params": obj.get("params"), **res}, indent=2))


if __name__ == "__main__":
    main()
