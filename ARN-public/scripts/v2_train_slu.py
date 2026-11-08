"""
ARN-v2 on multi-intent SLU (MixATIS / MixSNIPS), exactly as pre-registered in
docs/PREREGISTRATION_v2_confirmatory.md (commit 1fa7a50).

  * intents = tools (multi-label FAR-v2 router), dataset BIO tags = CRF labels (hard BIO mask)
  * no canonicalizer, no keyword rules, no post-processing beyond BIO decoding
  * selection: best dev overall accuracy; test decoded once, after selection
  * metrics: overall acc (intent set AND all word tags exact), intent acc, slot span F1 (conlleval)

Usage:
  python scripts/v2_train_slu.py --dataset MixATIS_clean --seed 42
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

from arn.v2.model import ARNv2  # noqa: E402
from arn.v2.schema import Registry  # noqa: E402
from arn.viterbi_fast import viterbi_decode_numpy  # noqa: E402


def read_split(path: Path):
    """AGIF format: 'word TAG' lines, then one intent line ('a#b'), blank line between sentences."""
    out, words, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1:  # intent line closes the sentence
            out.append((words, tags, sorted(set(parts[0].split("#")))))
            words, tags = [], []
        else:
            words.append(parts[0])
            tags.append(parts[1])
    return out


def spans(tags):
    """conlleval-style chunks: (type, start, end); I-X after a different chunk starts a new chunk."""
    res, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                res.append((cur[0], cur[1], i))
            cur = (t[2:], i) if t != "O" else None
    return set(res)


def build_registry(train):
    intents = sorted({i for _, _, ints in train for i in ints})
    types = sorted({t[2:] for _, ts, _ in train for t in ts if t != "O"})
    reg = Registry(tools=intents, args={}, required={})
    reg.tags = ["O"] + [f"{p}-{x}" for x in types for p in ("B", "I")] + ["<PAD>"]
    reg.tag_to_idx = {t: i for i, t in enumerate(reg.tags)}
    return reg


def encode(data, reg, tok, max_len):
    feats = []
    o = reg.tag_to_idx["O"]
    for words, tags, ints in data:
        enc = tok(words, is_split_into_words=True, max_length=max_len, truncation=True, padding="max_length")
        wid = enc.word_ids()
        lab, first, prev = [], [], None
        for pos, w in enumerate(wid):
            if w is None:
                lab.append(o if enc["attention_mask"][pos] else reg.pad_idx)
            else:
                t = tags[w]
                if w == prev and t.startswith("B-"):
                    t = "I-" + t[2:]
                lab.append(reg.tag_to_idx.get(t, o))
                if w != prev:
                    first.append(pos)
            prev = w
        vec = [1.0 if x in ints else 0.0 for x in reg.tools]
        feats.append({"ids": enc["input_ids"], "am": enc["attention_mask"], "tags": lab, "tools": vec,
                      "first": first, "n_words": len(words), "gold_tags": tags, "gold_ints": ints})
    return feats


@torch.no_grad()
def predict(model, feats, reg, dev, bs=128):
    model.eval()
    trans, st, en = (x.detach().float().cpu().numpy() for x in model.crf._get_effective_transitions())
    preds = []
    for i in range(0, len(feats), bs):
        b = feats[i:i + bs]
        ids = torch.tensor([f["ids"] for f in b], device=dev)
        am = torch.tensor([f["am"] for f in b], device=dev)
        L = int(am.sum(1).max())
        ids, am = ids[:, :L], am[:, :L]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, em, _ = model(ids, am)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        paths = viterbi_decode_numpy(em.float().cpu().numpy(), am.cpu().numpy(), trans, st, en, pad_val=reg.pad_idx)
        for f, p, path in zip(b, probs, paths):
            ints = sorted(reg.tools[k] for k in np.where(p >= 0.5)[0]) or [reg.tools[int(p.argmax())]]
            wt = [reg.tags[path[q]] if q < len(path) else "O" for q in f["first"]]
            wt = [t if t != "<PAD>" else "O" for t in wt] + ["O"] * (f["n_words"] - len(wt))
            preds.append((ints, wt))
    return preds


def metrics(feats, preds):
    n = len(feats)
    overall = intent = 0
    tp = fp = fn = 0
    for f, (ints, wt) in zip(feats, preds):
        ok_i = ints == f["gold_ints"]
        ok_s = wt == f["gold_tags"]
        intent += ok_i
        overall += ok_i and ok_s
        g, p = spans(f["gold_tags"]), spans(wt)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"overall_acc": round(100 * overall / n, 2), "intent_acc": round(100 * intent / n, 2),
            "slot_f1": round(100 * f1, 2), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MixATIS_clean", "MixSNIPS_clean"])
    ap.add_argument("--trunk", default="google/bert_uncased_L-2_H-128_A-2")
    ap.add_argument("--router", default="far")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-test", action="store_true",
                    help="smoke/debug run: never read the test split, write nothing to the ledger")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = ROOT / "data" / "external" / args.dataset
    train, val = read_split(d / "train.txt"), read_split(d / "dev.txt")
    reg = build_registry(train)
    tok = AutoTokenizer.from_pretrained(args.trunk)
    tr_f, va_f = encode(train, reg, tok, args.max_len), encode(val, reg, tok, args.max_len)
    model = ARNv2(args.trunk, reg, router=args.router).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"slu_{args.dataset}_{args.trunk.split('/')[-1]}_{args.router}_s{args.seed}"
    print(f"[data] train={len(tr_f)} dev={len(va_f)} intents={len(reg.tools)} tags={len(reg.tags)} params={n_params:,}",
          flush=True)

    enc_ids = {id(p) for p in model.encoder.parameters()}
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": args.lr},
                             {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.head_lr}],
                            weight_decay=0.01)
    steps = args.epochs * ((len(tr_f) + args.bs - 1) // args.bs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = random.Random(args.seed + 1)
    best = (-1.0, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = list(range(len(tr_f)))
        rng.shuffle(idx)
        t0 = time.time()
        for i in range(0, len(idx), args.bs):
            b = [tr_f[j] for j in idx[i:i + args.bs]]
            ids = torch.tensor([f["ids"] for f in b], device=dev)
            am = torch.tensor([f["am"] for f in b], device=dev)
            L = int(am.sum(1).max())
            ids, am = ids[:, :L], am[:, :L]
            tg = torch.tensor([f["tags"][:L] for f in b], device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, em, _ = model(ids, am)
            loss, _, _ = model.loss(logits.float(), em.float(), am,
                                    torch.tensor([f["tools"] for f in b], device=dev), tg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vm = metrics(va_f, predict(model, va_f, reg, dev))
        print(f"[ep {ep:02d}] dev overall={vm['overall_acc']} intent={vm['intent_acc']} slotF1={vm['slot_f1']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if vm["overall_acc"] > best[0]:
            best = (vm["overall_acc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)

    if args.no_test:
        print(f"[smoke] best_ep={best[2]} dev_overall={best[0]} (test split not read)", flush=True)
        return

    # ── selection frozen; test is read for the first time here ──
    model.load_state_dict(best[1])
    test = read_split(d / "test.txt")
    te_f = encode(test, reg, tok, args.max_len)
    tm = metrics(te_f, predict(model, te_f, reg, dev))
    print(f"[test] best_ep={best[2]} dev_overall={best[0]} TEST overall={tm['overall_acc']} "
          f"intent={tm['intent_acc']} slotF1={tm['slot_f1']} n={tm['n']}", flush=True)

    m = model.to("cpu")
    torch.set_num_threads(1)
    ts = []
    for f in te_f[:200]:
        t0 = time.perf_counter()
        predict(m, [f], reg, torch.device("cpu"), bs=1)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    lat = {"p50_ms": round(ts[len(ts) // 2], 2), "threads": 1, "n": len(ts)}
    print(f"[latency] {lat}", flush=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec = {"experiment_id": tag, "protocol": "v2_slu_preregistered_1fa7a50", "params": n_params, "seed": args.seed,
           "args": vars(args), "best_epoch": best[2], "dev_overall": best[0], "test": tm, "cpu_latency": lat,
           "git_hash": git, "command": " ".join(sys.argv), "timestamp": datetime.now(timezone.utc).isoformat()}
    (ROOT / "results" / "slu").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "slu" / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    with open(ROOT / "results" / "ledger_v2.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
