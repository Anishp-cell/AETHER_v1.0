"""
Train + evaluate ARN-v2 on google/mobile-actions (plan Phases 2-4).

Protocol (no eval leakage):
  * train rows (metadata == "train") are split 90/10 with a fixed seed into
    fit / validation; model selection uses validation strict EM only.
  * the eval split (961 rows) is decoded once, after the best checkpoint is frozen.
  * the temporal policy comes from results/temporal_policy.json (fit on train).

Outputs:
  results/xmodel/<tag>.json   predictions in the shared-scorer format
  results/v2/<tag>.json       metrics (val + eval) and run metadata
  results/ledger_v2.jsonl     one appended line per run

Example:
  python scripts/v2_train_mobile.py --data <dataset.jsonl> --trunk google/bert_uncased_L-2_H-128_A-2 \
      --router far --epochs 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

from arn.canon.temporal import TemporalPolicy, canonicalize, parse_reference  # noqa: E402
from arn.eval.mobile_actions_scorer import score  # noqa: E402
from arn.v2.decode import joint_decode  # noqa: E402
from arn.v2.model import ARNv2  # noqa: E402
from arn.v2.schema import Registry, label_row, token_tags  # noqa: E402


def encode(rows, reg, tok, max_len, record_multi_span):
    feats, trunc, dropped = [], 0, 0
    for r in rows:
        lr = label_row(r, reg, record_multi_span=record_multi_span)
        dropped += lr.dropped
        enc = tok(lr.text, max_length=max_len, truncation=True, padding="max_length",
                  return_offsets_mapping=True)
        if len(tok(lr.text)["input_ids"]) > max_len:
            trunc += 1
        tags = token_tags(enc["offset_mapping"], enc["attention_mask"], lr.char_spans, reg)
        feats.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                      "tags": tags, "tools": lr.tool_vec, "offsets": enc["offset_mapping"], "text": lr.text})
    return feats, trunc, dropped


def batches(feats, bs, shuffle, rng):
    idx = list(range(len(feats)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [feats[j] for j in idx[i:i + bs]]


def to_t(batch, key, dtype, device):
    return torch.tensor([f[key] for f in batch], dtype=dtype, device=device)


def predict(model, feats, rows, reg, pol, device, bs=64, constrained=True, use_canon=True, anchor="keyword"):
    model.eval()
    preds = {}
    for start in range(0, len(feats), bs):
        b = feats[start:start + bs]
        rs = rows[start:start + bs]
        fns = []
        for r, f in zip(rs, b):
            ref = parse_reference(r["messages"][0].get("content", ""))
            text = f["text"]
            fns.append((lambda a, text=text, ref=ref: canonicalize(text, ref, pol)) if use_canon
                       else (lambda a: None))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = joint_decode(model, to_t(b, "input_ids", torch.long, device),
                               to_t(b, "attention_mask", torch.long, device),
                               [f["text"] for f in b], [f["offsets"] for f in b], reg, fns,
                               constrained=constrained, anchor=anchor)
        for k, calls in enumerate(out):
            preds[str(start + k)] = calls
    return preds


def cpu_latency(model, feats, rows, reg, pol, n=200):
    m = model.to("cpu").eval()
    torch.set_num_threads(1)
    ts = []
    for i in range(min(n, len(feats))):
        t0 = time.perf_counter()
        predict(m, feats[i:i + 1], rows[i:i + 1], reg, pol, torch.device("cpu"), bs=1)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 2), "p95_ms": round(ts[int(len(ts) * 0.95)], 2),
            "threads": 1, "n": len(ts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--trunk", default="google/bert_uncased_L-2_H-128_A-2")
    ap.add_argument("--router", default="far", choices=["far", "far_plain", "cls"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4, help="trunk LR")
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", default=str(ROOT / "results" / "temporal_policy.json"))
    ap.add_argument("--no-constrained", action="store_true", help="ablation: unconstrained Viterbi")
    ap.add_argument("--no-canon", action="store_true", help="ablation: no temporal canonicalizer")
    ap.add_argument("--anchor", default="keyword", choices=["keyword", "far"])
    ap.add_argument("--multi-span-min-rate", type=float, default=0.05)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or (f"arn_v2_{args.trunk.split('/')[-1]}_{args.router}_s{args.seed}"
                       + ("_unconstrained" if args.no_constrained else "") + ("_nocanon" if args.no_canon else ""))

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    train_all = [r for r in rows if r.get("metadata") == "train"]
    eval_rows = [r for r in rows if r.get("metadata") == "eval"]
    rng = random.Random(args.seed)
    order = list(range(len(train_all)))
    rng.shuffle(order)
    n_val = len(order) // 10
    val_rows = [train_all[i] for i in order[:n_val]]
    fit_rows = [train_all[i] for i in order[n_val:]]

    pd = json.load(open(args.policy))["policy"]
    pol = TemporalPolicy(pd["next_weekday"], pd["bare_weekday_allows_today"], pd["roll_year_forward"],
                         tuple(pd["default_time"]))

    reg = Registry.from_rows(train_all)  # train rows only (schema is identical across splits; verified)
    tok = AutoTokenizer.from_pretrained(args.trunk)
    fit_f, tr_trunc, tr_drop = encode(fit_rows, reg, tok, args.max_len, record_multi_span=True)
    reg.finalize_multi_span(args.multi_span_min_rate)
    val_f, _, _ = encode(val_rows, reg, tok, args.max_len, record_multi_span=False)
    ev_f, ev_trunc, _ = encode(eval_rows, reg, tok, args.max_len, record_multi_span=False)
    print(f"[data] fit={len(fit_f)} val={len(val_f)} eval={len(ev_f)} tags={len(reg.tags)} "
          f"multi_span_args={sorted(reg.multi_span_args)} truncated(fit/eval)={tr_trunc}/{ev_trunc} "
          f"dropped_args(fit)={tr_drop}", flush=True)

    model = ARNv2(args.trunk, reg, router=args.router).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} trunk={args.trunk} router={args.router} device={device}", flush=True)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * ((len(fit_f) + args.bs - 1) // args.bs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best = (-1.0, None, -1)
    brng = random.Random(args.seed + 1)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for b in batches(fit_f, args.bs, True, brng):
            ids, am = to_t(b, "input_ids", torch.long, device), to_t(b, "attention_mask", torch.long, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, em, _ = model(ids, am)
            loss, _, _ = model.loss(logits.float(), em.float(), am, to_t(b, "tools", torch.float, device),
                                    to_t(b, "tags", torch.long, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        vp = predict(model, val_f, val_rows, reg, pol, device,
                     constrained=not args.no_constrained, use_canon=not args.no_canon, anchor=args.anchor)
        vs = score(val_rows, vp)
        print(f"[ep {ep:02d}] loss={tot / len(fit_f):.4f} val_strict={vs['strict_em']} "
              f"val_multi={vs['multi_call_em']} val_toolset={vs['tool_set_acc']} ({time.time() - t0:.0f}s)", flush=True)
        if vs["strict_em"] > best[0]:
            best = (vs["strict_em"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)

    model.load_state_dict(best[1])
    ep_pred = predict(model, ev_f, eval_rows, reg, pol, device,
                      constrained=not args.no_constrained, use_canon=not args.no_canon, anchor=args.anchor)
    es = score(eval_rows, ep_pred)
    print(f"[eval] best_ep={best[2]} val_strict={best[0]} EVAL strict={es['strict_em']} "
          f"single={es['single_call_em']} multi={es['multi_call_em']} toolset={es['tool_set_acc']} "
          f"datetime={es['datetime_arg_acc']}", flush=True)

    lat = cpu_latency(model, ev_f, eval_rows, reg, pol)
    print(f"[latency] {lat}", flush=True)

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        git = None
    (ROOT / "results" / "xmodel").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "v2").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints" / "v2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "xmodel" / f"{tag}.json").write_text(
        json.dumps({"model": tag, "params": n_params, "predictions": ep_pred}))
    torch.save({"state_dict": best[1], "registry": reg.to_dict(), "args": vars(args)},
               ROOT / "checkpoints" / "v2" / f"{tag}.pt")
    rec = {"experiment_id": tag, "protocol": "v2_mobile_actions", "model": tag, "params": n_params,
           "seed": args.seed, "args": vars(args), "best_epoch": best[2], "val_strict_em": best[0],
           "eval": es, "cpu_latency": lat, "git_hash": git,
           "command": " ".join(sys.argv), "timestamp": datetime.now(timezone.utc).isoformat()}
    (ROOT / "results" / "v2" / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    with open(ROOT / "results" / "ledger_v2.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
