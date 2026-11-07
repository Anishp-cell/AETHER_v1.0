"""
Re-decode a saved ARN-v2 checkpoint with the current decoder (no retraining).

Rebuilds the registry exactly as training does (same seeded fit/val split, train-only
multi-span statistics), loads the weights, decodes val and eval, scores both, and
measures CPU latency. Decoder changes must be justified on VAL; eval is reported.

Usage:
  python scripts/v2_redecode.py --data <dataset.jsonl> --ckpt checkpoints/v2/<tag>.pt [--anchor keyword]
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from arn.canon.temporal import TemporalPolicy  # noqa: E402
from arn.eval.mobile_actions_scorer import score  # noqa: E402
from arn.v2.model import ARNv2  # noqa: E402
from arn.v2.schema import Registry  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_train", ROOT / "scripts" / "v2_train_mobile.py")
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchor", default="keyword", choices=["keyword", "far"])
    ap.add_argument("--no-constrained", action="store_true")
    ap.add_argument("--no-canon", action="store_true")
    ap.add_argument("--multi-span-min-rate", type=float, default=0.05)
    ap.add_argument("--suffix", default="redecode")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ck["args"]
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    train_all = [r for r in rows if r.get("metadata") == "train"]
    eval_rows = [r for r in rows if r.get("metadata") == "eval"]
    rng = random.Random(targs["seed"])
    order = list(range(len(train_all)))
    rng.shuffle(order)
    n_val = len(order) // 10
    val_rows = [train_all[i] for i in order[:n_val]]
    fit_rows = [train_all[i] for i in order[n_val:]]

    pd = json.load(open(ROOT / "results" / "temporal_policy.json"))["policy"]
    pol = TemporalPolicy(pd["next_weekday"], pd["bare_weekday_allows_today"], pd["roll_year_forward"],
                         tuple(pd["default_time"]))
    reg = Registry.from_rows(train_all)  # train rows only (schema is identical across splits; verified)
    tok = AutoTokenizer.from_pretrained(targs["trunk"])
    V2.encode(fit_rows, reg, tok, targs["max_len"], record_multi_span=True)
    reg.finalize_multi_span(args.multi_span_min_rate)
    val_f, _, _ = V2.encode(val_rows, reg, tok, targs["max_len"], record_multi_span=False)
    ev_f, _, _ = V2.encode(eval_rows, reg, tok, targs["max_len"], record_multi_span=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARNv2(targs["trunk"], reg, router=targs["router"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    kw = dict(constrained=not args.no_constrained, use_canon=not args.no_canon, anchor=args.anchor)
    vs = score(val_rows, V2.predict(model, val_f, val_rows, reg, pol, dev, **kw))
    ep = V2.predict(model, ev_f, eval_rows, reg, pol, dev, **kw)
    es = score(eval_rows, ep)
    lat = V2.cpu_latency(model, ev_f, eval_rows, reg, pol)
    keys = ("strict_em", "unordered_em", "tool_set_acc", "single_call_em", "multi_call_em", "datetime_arg_acc")
    print(json.dumps({"multi_span_args": sorted(reg.multi_span_args), "val": {k: vs[k] for k in keys},
                      "eval": {k: es[k] for k in keys}, "cpu_latency": lat, **kw}, indent=1))
    tag = Path(args.ckpt).stem + f"_{args.suffix}"
    n_params = sum(p.numel() for p in model.parameters())
    (ROOT / "results" / "xmodel" / f"{tag}.json").write_text(
        json.dumps({"model": tag, "params": n_params, "predictions": ep}))


if __name__ == "__main__":
    main()
