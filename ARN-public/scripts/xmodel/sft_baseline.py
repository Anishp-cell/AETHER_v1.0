"""
Fine-tune a generative function-calling baseline on the SAME Mobile-Actions fit split
ARN-v2 uses (fairness control: ARN is fine-tuned, so the comparison must be too).

  * fit/val split: identical seeded 90/10 split of metadata=="train" rows (seed 42)
  * prompt: system (developer) message + user turn + the row's tool schemas, rendered by
    the model's own chat template, exactly as scripts/xmodel/run_hf_fc.py renders them
  * target: the gold calls in the model's native call syntax; loss on target tokens only
  * output: a local model directory; evaluate it with run_hf_fc.py --model <dir>

Usage:
  python scripts/xmodel/sft_baseline.py --data <dataset.jsonl> --model LiquidAI/LFM2.5-230M \
      --out checkpoints/sft/lfm2.5_230m_mobile --epochs 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_hf_fc", ROOT / "scripts" / "xmodel" / "run_hf_fc.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def gold_calls(row):
    out = []
    for tc in row["messages"][-1].get("tool_calls") or []:
        a = tc["function"].get("arguments") or {}
        out.append((tc["function"]["name"], {k: v for k, v in a.items() if v is not None}))
    return out


def render_target(calls, style: str, eos: str) -> str:
    if style == "fg":  # FunctionGemma native syntax (as emitted by the base model)
        return "".join("<start_function_call>call:" + n + "{"
                       + ",".join(f"{k}:<escape>{v}<escape>" for k, v in a.items())
                       + "}<end_function_call>" for n, a in calls) + eos
    if style == "pythonic":  # LFM2 / LFM2.5
        body = ", ".join(f"{n}(" + ", ".join(f"{k}={json.dumps(str(v))}" for k, v in a.items()) + ")"
                         for n, a in calls)
        return f"<|tool_call_start|>[{body}]<|tool_call_end|>{eos}"
    # JSON <tool_call> blocks (Qwen-style)
    return "".join(f"<tool_call>\n{json.dumps({'name': n, 'arguments': a})}\n</tool_call>" for n, a in calls) + eos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="pythonic", choices=["pythonic", "json", "fg"])
    ap.add_argument("--eos", default="<|im_end|>", help="end-of-turn token of the model's template")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-think", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    train_all = [r for r in rows if r.get("metadata") == "train"]
    order = list(range(len(train_all)))
    random.Random(args.seed).shuffle(order)          # identical to v2_train_mobile.py
    fit_rows = [train_all[i] for i in order[len(order) // 10:]]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = args.eos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    examples = []
    for r in fit_rows:
        m = r["messages"]
        msgs = [{"role": "system", "content": m[0]["content"]},
                {"role": "user", "content": next(x["content"] for x in m if x["role"] == "user")}]
        kw = {"enable_thinking": False} if args.no_think else {}
        prompt = tok.apply_chat_template(msgs, tools=H.to_json_schema(r["tools"]), tokenize=False,
                                         add_generation_prompt=True, **kw)
        target = render_target(gold_calls(r), args.style, eos)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        examples.append((p_ids, t_ids))
    # sanity: the parser used at eval time must recover the gold calls from the target
    probe = render_target(gold_calls(fit_rows[0]), args.style, eos)
    assert H.parse_calls(probe), f"target not parseable: {probe[:200]}"
    print(f"[sft] fit={len(examples)} model={args.model} params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(examples) // (args.bs * args.accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), max(steps, 1))
    model.train()
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        random.Random(args.seed + ep).shuffle(examples)
        for i in range(0, len(examples), args.bs):
            batch = examples[i:i + args.bs]
            L = max(len(p) + len(t) for p, t in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id)
            lab = torch.full((len(batch), L), -100)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (p, t) in enumerate(batch):
                seq = p + t
                ids[j, :len(seq)] = torch.tensor(seq)
                lab[j, len(p):len(seq)] = torch.tensor(t)
                att[j, :len(seq)] = 1
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
            (out.loss / args.accum).backward()
            if (i // args.bs + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"[sft] ep={ep + 1} step={step}/{steps} loss={out.loss.item():.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[sft] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
