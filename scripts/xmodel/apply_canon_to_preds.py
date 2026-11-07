"""
Fairness control: give a generative baseline the SAME temporal canonicalizer ARN-v2 uses.

For each predicted call whose tool schema has a `datetime` argument, the model's own
datetime value is replaced by arn.canon.temporal.canonicalize(user_text, reference).
Everything else in the prediction is untouched. Writes <name>+canon.json and prints
strict EM before/after, so the paper can report "ARN vs LM" and "ARN vs LM + same rules".

Usage:
  python scripts/xmodel/apply_canon_to_preds.py --data <dataset.jsonl> results/xmodel/<name>.json [...]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("arn_canon_temporal", "src/arn/canon/temporal.py")
S = _load("arn_eval_scorer", "src/arn/eval/mobile_actions_scorer.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--policy", default=str(ROOT / "results" / "temporal_policy.json"))
    ap.add_argument("pred_files", nargs="+")
    args = ap.parse_args()
    pd = json.load(open(args.policy))["policy"]
    pol = T.TemporalPolicy(pd["next_weekday"], pd["bare_weekday_allows_today"], pd["roll_year_forward"],
                           tuple(pd["default_time"]))
    rows = S.load_eval_rows(args.data)
    dt_tools = {t["function"]["name"] for r in rows for t in r["tools"]
                if "datetime" in ((t["function"].get("parameters") or {}).get("properties") or {})}
    for pf in args.pred_files:
        obj = json.load(open(pf, encoding="utf-8"))
        new = {}
        for i, r in enumerate(rows):
            user = next(m["content"] for m in r["messages"] if m.get("role") == "user")
            ref = T.parse_reference(r["messages"][0].get("content", ""))
            calls = []
            for c in obj["predictions"].get(str(i), []):
                c = {"name": c.get("name"), "arguments": dict(c.get("arguments") or {})}
                if c["name"] in dt_tools:
                    v = T.canonicalize(user, ref, pol)
                    if v is not None:  # never replace a model's value with nothing
                        c["arguments"]["datetime"] = v
                calls.append(c)
            new[str(i)] = calls
        before, after = S.score(rows, obj["predictions"]), S.score(rows, new)
        out = Path(pf).with_name(Path(pf).stem + "+canon.json")
        out.write_text(json.dumps({**{k: v for k, v in obj.items() if k != "predictions"},
                                   "model": obj.get("model", "") + "+canon", "predictions": new}))
        print(f"{obj.get('model')}: strict {before['strict_em']} -> {after['strict_em']} "
              f"(datetime {before['datetime_arg_acc']} -> {after['datetime_arg_acc']})")


if __name__ == "__main__":
    main()
