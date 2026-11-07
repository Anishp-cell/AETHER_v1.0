"""
Run a Hugging Face function-calling LM on the Mobile-Actions eval split and write
predictions in the shared-scorer format (plan Phase 1, "same harness for everyone").

Each model gets: the row's developer/system message (it carries the reference
date), the user turn, and the row's own tool schemas via the model's chat
template (`tools=`). Greedy decoding. Output calls are parsed from any of:
  * JSON:        {"name": ..., "arguments": {...}}   (Qwen <tool_call>, generic)
  * Pythonic:    [fn(a="x", b="y")]                  (LFM2 / LFM2.5)
  * FunctionGemma: call:fn{a:<escape>x<escape>,...}

Usage:
  python scripts/xmodel/run_hf_fc.py --data <dataset.jsonl> --model Qwen/Qwen3-0.6B [--limit 50]
Then score with:
  python src/arn/eval/mobile_actions_scorer.py --data <dataset.jsonl> results/xmodel/<name>.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def to_json_schema(tools):
    """Dataset schemas use Gemini-style upper-case types; most templates expect JSON Schema."""
    def fix(o):
        if isinstance(o, dict):
            return {k: (v.lower() if k == "type" and isinstance(v, str) else fix(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(x) for x in o]
        return o
    return [{"type": "function", "function": fix(t["function"])} for t in tools]


_JSON_OBJ = re.compile(r"\{.*\}", re.S)
_FG_CALL = re.compile(r"call:([A-Za-z_]\w*)\{(.*?)\}(?=<end_function_call>|call:|$)", re.S)
_FG_ARG = re.compile(r"([A-Za-z_]\w*):(?:<escape>(.*?)<escape>|([^,}]*))", re.S)


def _json_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            o = json.loads(m.group(1).strip())
            calls.append({"name": o.get("name"), "arguments": o.get("arguments") or o.get("parameters") or {}})
        except Exception:
            pass
    if calls:
        return calls
    m = _JSON_OBJ.search(text)
    if m:
        try:
            o = json.loads(m.group(0))
            items = o if isinstance(o, list) else [o]
            for it in items:
                if isinstance(it, dict) and "name" in it:
                    calls.append({"name": it["name"], "arguments": it.get("arguments") or it.get("parameters") or {}})
        except Exception:
            pass
    return calls


def _pythonic_calls(text):
    m = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, re.S)
    src = m.group(1) if m else text.strip()
    if not src.startswith("["):
        src = "[" + src + "]"
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        return []
    calls = []
    for node in getattr(tree.body, "elts", []):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = {}
            for kw in node.keywords:
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    pass
            calls.append({"name": node.func.id, "arguments": args})
    return calls


def _fg_calls(text):
    calls = []
    for m in _FG_CALL.finditer(text):
        args = {}
        for a in _FG_ARG.finditer(m.group(2)):
            args[a.group(1)] = (a.group(2) if a.group(2) is not None else a.group(3)).strip()
        calls.append({"name": m.group(1), "arguments": args})
    return calls


def parse_calls(text: str):
    for fn in (_fg_calls, _json_calls, _pythonic_calls):
        c = fn(text)
        if c:
            return c
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--no-think", action="store_true", help="pass enable_thinking=False (Qwen3)")
    ap.add_argument("--split", default="eval", choices=["eval", "val"])
    ap.add_argument("--seed", type=int, default=42, help="seed of the train/val split (val only)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.split == "val":  # identical seeded 10% slice of train used for ARN-v2 selection
        import random
        train_all = [r for r in rows if r.get("metadata") == "train"]
        order = list(range(len(train_all)))
        random.Random(args.seed).shuffle(order)
        rows = [train_all[i] for i in order[: len(order) // 10]][: args.limit]
    else:
        rows = [r for r in rows if r.get("metadata") == "eval"][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())

    prompts = []
    for r in rows:
        m = r["messages"]
        msgs = [{"role": "system" if m[0]["role"] in ("developer", "system") else m[0]["role"],
                 "content": m[0]["content"]},
                {"role": "user", "content": next(x["content"] for x in m if x["role"] == "user")}]
        kw = {"enable_thinking": False} if args.no_think else {}
        prompts.append(tok.apply_chat_template(msgs, tools=to_json_schema(r["tools"]), tokenize=False,
                                               add_generation_prompt=True, **kw))

    preds, raw, t0 = {}, {}, time.time()
    for i in range(0, len(prompts), args.batch):
        enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k, seq in enumerate(out):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=False)
            preds[str(i + k)] = parse_calls(text)
            raw[str(i + k)] = text
        print(f"{min(i + args.batch, len(prompts))}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)

    name = args.name or args.model.split("/")[-1]
    out_dir = ROOT / "results" / "xmodel"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({"model": name, "hf_id": args.model, "params": n_params,
                                                      "predictions": preds}))
    (out_dir / f"{name}.raw.json").write_text(json.dumps(raw))
    print(f"wrote {out_dir / (name + '.json')}  params={n_params:,}")


if __name__ == "__main__":
    main()
