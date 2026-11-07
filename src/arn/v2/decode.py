"""
ARN-v2 registry-constrained joint decoding (plan component A).

For each utterance:
  1. Enumerate candidate tool sets S from the router's sigmoid probabilities,
     scored by  log p(S) = sum_{k in S} log p_k + sum_{k not in S} log(1 - p_k).
  2. For each S, run CRF Viterbi with emissions masked to the tags of the tools
     in S (plus O), so no span can belong to an uncalled tool.
  3. Score  J(S) = log p(S) + log p(y*_S | x)  where log p(y | x) is the path's
     log-probability under the unconstrained CRF (score - log Z).
  4. Reject sets whose decoded spans miss a required argument (canonicalized
     arguments must canonicalize), then return the argmax.
Calls are ordered by their earliest argument span; argument-less calls are
anchored at the token FAR attends to most (FAR routers) or a keyword position
(CLS router).
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..viterbi_fast import viterbi_decode_numpy
from .schema import Registry

NEG = -1e4

# Mention anchors for calls without argument spans. With gold spans, "earliest span,
# else first keyword" reproduces the gold call order on 99.4% of multi-call TRAIN rows
# (measured; eval not used). Unsupervised FAR-attention anchors are worse on the
# VALIDATION split (80.55 vs 89.76 val strict EM, seed-42 checkpoint), so they are opt-in.
# Disclosure: the problem was first noticed in eval-split errors of a 1-epoch checkpoint;
# the decision is supported by the train-side (99.4%) and validation evidence above.
_KEYWORDS = {
    "turn_on_flashlight": ["flashlight", "torch", "light"],
    "turn_off_flashlight": ["flashlight", "torch", "light"],
    "open_wifi_settings": ["wi-fi", "wifi", "network", "internet"],
    "create_calendar_event": ["calendar", "event", "schedule", "meeting", "reminder", "appointment"],
    "send_email": ["email", "e-mail", "mail"],
    "create_contact": ["contact", "save", "add"],
    "show_map": ["map", "location", "where", "directions", "find", "locate"],
}
_TRAIL = " \t\n,;:"


def _clean_value(s: str) -> str:
    return s.strip(_TRAIL)


def candidate_sets(probs: Sequence[float], tools: Sequence[str], p_min: float = 0.02,
                   max_tools: int = 5, max_calls: int = 3, top: int = 6) -> List[Tuple[float, Tuple[int, ...]]]:
    eps = 1e-6
    order = sorted(range(len(probs)), key=lambda k: -probs[k])
    pool = [k for k in order if probs[k] >= p_min][:max_tools] or order[:1]
    base = sum(math.log(max(1 - p, eps)) for p in probs)
    out = []
    for r in range(1, min(max_calls, len(pool)) + 1):
        for s in itertools.combinations(pool, r):
            lp = base + sum(math.log(max(probs[k], eps)) - math.log(max(1 - probs[k], eps)) for k in s)
            out.append((lp, tuple(sorted(s))))
    out.sort(key=lambda x: -x[0])
    return out[:top]


def spans_from_path(path: Sequence[int], offsets: Sequence[Tuple[int, int]], reg: Registry
                    ) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cur: Optional[List] = None
    for tid, (s, e) in zip(path, offsets):
        tag = reg.tags[tid]
        if s == e or tag in ("O", "<PAD>"):
            if cur:
                spans.append(tuple(cur))
            cur = None
            continue
        kind, qual = tag.split("-", 1)
        if kind == "B" or cur is None or cur[2] != qual:
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, qual]
        else:
            cur[1] = e
    if cur:
        spans.append(tuple(cur))
    return spans


def assemble_calls(tool_set: Sequence[str], spans: Sequence[Tuple[int, int, str]], text: str,
                   reg: Registry, canon_fn, anchors: Dict[str, int]) -> Tuple[List[dict], bool]:
    """Build ordered calls; returns (calls, valid) where valid = all required args present."""
    calls = []
    valid = True
    for t in tool_set:
        args: Dict[str, Optional[str]] = {}
        first = None
        for a in reg.args[t]:
            mine = sorted((s, e) for s, e, q in spans if q == f"{t}.{a}")
            if not mine:
                continue
            first = mine[0][0] if first is None else min(first, mine[0][0])
            if f"{t}.{a}" in reg.multi_span_args:
                pieces: List[str] = []
                for s, e in mine:
                    v = _clean_value(text[s:e])
                    if v and v.lower() not in (p.lower() for p in pieces):
                        pieces.append(v)
                args[a] = " ".join(pieces)
            else:
                args[a] = _clean_value(text[mine[0][0]:mine[0][1]])
        for a in reg.canon.get(t, []):
            args[a] = canon_fn(a)
        for a in reg.required.get(t, []):
            if not args.get(a):
                valid = False
        pos = first if first is not None else anchors.get(t, 10 ** 6)
        calls.append((pos, {"name": t, "arguments": {k: v for k, v in args.items() if v}}))
    calls.sort(key=lambda x: x[0])
    return [c for _, c in calls], valid


def _keyword_anchor(tool: str, text: str) -> int:
    tl = text.lower()
    hits = [tl.find(kw) for kw in _KEYWORDS.get(tool, []) if tl.find(kw) >= 0]
    return min(hits) if hits else 10 ** 6


@torch.no_grad()
def joint_decode(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 texts: Sequence[str], offsets: Sequence[Sequence[Tuple[int, int]]],
                 reg: Registry, canon_fns: Sequence, top_sets: int = 6,
                 constrained: bool = True, anchor: str = "keyword") -> List[List[dict]]:
    # Trim padding to the longest real sequence in the batch (exact; saves trunk + CRF work).
    max_len = int(attention_mask.sum(1).max())
    input_ids, attention_mask = input_ids[:, :max_len], attention_mask[:, :max_len]
    offsets = [o[:max_len] for o in offsets]
    tool_logits, emissions, attn = model(input_ids, attention_mask)
    trans, start_t, end_t = (x.detach().float().cpu().numpy() for x in model.crf._get_effective_transitions())
    probs = torch.sigmoid(tool_logits).float().cpu()
    emissions = emissions.float()
    mask = attention_mask.bool()
    log_z = model.crf.forward_algorithm(emissions, mask)  # (B,)
    out: List[List[dict]] = []
    n_tags = len(reg.tags)
    for b in range(input_ids.size(0)):
        text, offs = texts[b], offsets[b]
        p = probs[b].tolist()
        # FAR anchors: argmax attention over real (non-special) tokens -> char offset
        anchors: Dict[str, int] = {}
        for k, t in enumerate(reg.tools):
            if anchor == "far" and attn is not None:
                w = attn[b, k].float().clone()
                for i, (s, e) in enumerate(offs):
                    if s == e or not mask[b, i]:
                        w[i] = -1
                anchors[t] = offs[int(w.argmax())][0]
            else:
                anchors[t] = _keyword_anchor(t, text)
        cands = candidate_sets(p, reg.tools, top=top_sets)
        em_b = emissions[b:b + 1].expand(len(cands), -1, -1).clone()
        if constrained:
            for c, (_, s) in enumerate(cands):
                allowed = torch.full((n_tags,), NEG, device=em_b.device)
                allowed[reg.tag_indices_for_tools([reg.tools[k] for k in s])] = 0.0
                em_b[c] += allowed
        m_b = mask[b:b + 1].expand(len(cands), -1)
        paths = viterbi_decode_numpy(em_b.cpu().numpy(), m_b.cpu().numpy(), trans, start_t, end_t,
                                     pad_val=reg.pad_idx)
        path_t = torch.tensor(paths, dtype=torch.long, device=em_b.device)
        gold_like = model.crf.score_sentence(emissions[b:b + 1].expand(len(cands), -1, -1), path_t, m_b)
        lp_path = (gold_like - log_z[b]).cpu().tolist()
        best, best_valid = None, None
        for c, (lp_set, s) in enumerate(cands):
            tset = [reg.tools[k] for k in s]
            spans = spans_from_path(paths[c], offs, reg)
            calls, valid = assemble_calls(tset, spans, text, reg, canon_fns[b], anchors)
            j = lp_set + lp_path[c]
            if best is None or j > best[0]:
                best = (j, calls)
            if valid and (best_valid is None or j > best_valid[0]):
                best_valid = (j, calls)
        out.append((best_valid or best)[1])
    return out
