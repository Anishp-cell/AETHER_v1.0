"""
Single-stream CPU latency benchmark (MLPerf-style reporting) for ARN-v2 and the LM baselines.

Protocol, identical for every system:
  * batch size 1, one query at a time, back to back (single-stream)
  * the first `--warmup` queries are discarded, then `--n` timed queries from the eval split
  * end-to-end time per query: tokenization + model + decoding/generation (+ canonicalizer for ARN)
  * reported: p50, p90 (MLPerf single-stream metric), mean; at 1 and 4 CPU threads
  * precision: FP32 PyTorch for every model (no quantization for anyone)
Limitations (stated in the paper): no optimized LM runtime (llama.cpp / ONNX int8) was used;
LM latency here is an upper bound relative to optimized deployments.

Usage:
  python scripts/benchmark_latency_v2.py --data <dataset.jsonl> --out results/latency_v2.json
"""

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from arn.canon.temporal import TemporalPolicy  # noqa: E402
from arn.v2.model import ARNv2  # noqa: E402
from arn.v2.schema import Registry  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load("v2_train", "scripts/v2_train_mobile.py")
H = _load("run_hf_fc", "scripts/xmodel/run_hf_fc.py")


def stats(ts):
    ts = sorted(ts)
    return {"p50_ms": round(ts[len(ts) // 2], 2), "p90_ms": round(ts[int(len(ts) * 0.9)], 2),
            "mean_ms": round(statistics.mean(ts), 2), "n": len(ts)}


def bench_arn(ckpt, rows, train_all, warmup, n, threads):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    reg = Registry.from_rows(train_all)
    tok = AutoTokenizer.from_pretrained(a["trunk"])
    order = list(range(len(train_all)))
    random.Random(a["seed"]).shuffle(order)
    V2.encode([train_all[i] for i in order[len(order) // 10:]], reg, tok, a["max_len"], record_multi_span=True)
    reg.finalize_multi_span(0.05)
    pd = json.load(open(ROOT / "results" / "temporal_policy.json"))["policy"]
    pol = TemporalPolicy(pd["next_weekday"], pd["bare_weekday_allows_today"], pd["roll_year_forward"],
                         tuple(pd["default_time"]))
    model = ARNv2(a["trunk"], reg, router=a["router"]).eval()
    model.load_state_dict(ck["state_dict"])
    params = sum(p.numel() for p in model.parameters())
    torch.set_num_threads(threads)
    ts = []
    for i, r in enumerate(rows[: warmup + n]):
        t0 = time.perf_counter()
        feats, _, _ = V2.encode([r], reg, tok, a["max_len"], record_multi_span=False)  # tokenization counted
        V2.predict(model, feats, [r], reg, pol, torch.device("cpu"), bs=1)
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1000)
    return params, stats(ts)


def bench_lm(model_id, rows, warmup, n, threads, no_think):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    params = sum(p.numel() for p in model.parameters())
    torch.set_num_threads(threads)
    kw = {"enable_thinking": False} if no_think else {}
    ts = []
    for i, r in enumerate(rows[: warmup + n]):
        m = r["messages"]
        msgs = [{"role": "system", "content": m[0]["content"]},
                {"role": "user", "content": next(x["content"] for x in m if x["role"] == "user")}]
        t0 = time.perf_counter()
        p = tok.apply_chat_template(msgs, tools=H.to_json_schema(r["tools"]), tokenize=False,
                                    add_generation_prompt=True, **kw)
        enc = tok(p, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=320, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        H.parse_calls(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False))
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1000)
    return params, stats(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arn-ckpt", default=str(ROOT / "checkpoints" / "v2" / "arn_v2_tiny_far_s42_newdec.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "latency_v2.json"))
    ap.add_argument("--arn-n", type=int, default=500)
    ap.add_argument("--lm-n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    train_all = [r for r in rows if r["metadata"] == "train"]
    ev = [r for r in rows if r["metadata"] == "eval"]
    res = {"protocol": "single-stream, batch 1, FP32, CPU, warmup discarded", "results": []}
    for th in (1, 4):
        p, s = bench_arn(args.arn_ckpt, ev, train_all, args.warmup, args.arn_n, th)
        res["results"].append({"system": "Aether Routing Network v2 (BERT-Tiny)", "params": p, "threads": th, **s})
        print(res["results"][-1], flush=True)
        for mid, nt in (("LiquidAI/LFM2.5-230M", False), ("google/functiongemma-270m-it", False),
                        ("Qwen/Qwen3-0.6B", True)):
            p, s = bench_lm(mid, ev, 3, args.lm_n, th, nt)
            res["results"].append({"system": mid, "params": p, "threads": th, **s})
            print(res["results"][-1], flush=True)
    res["cpu"] = json.load(open(ROOT / "results" / "environment_v2.json"))["cpu"]
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
