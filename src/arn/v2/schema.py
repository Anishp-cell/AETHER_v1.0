"""
ARN-v2 registry schema and span labelling (plan component A + C).

Tags are tool-qualified ("B-send_email.subject"), so a span already says which
call it belongs to. Datetime arguments are not tagged: they are produced by the
temporal canonicalizer (arn.canon.temporal) from the full utterance.

Multi-span arguments (component C): when a gold value is two verbatim pieces of
the utterance joined by a space, both pieces are tagged with the same qualified
tag; the decoder joins spans of such arguments in text order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

CANON_ARGS = {"datetime"}  # produced by canonicalizers, never tagged


@dataclass
class Registry:
    tools: List[str]
    args: Dict[str, List[str]]          # tool -> taggable argument names (schema order)
    required: Dict[str, List[str]]      # tool -> required argument names (incl. canon args)
    tags: List[str] = field(default_factory=list)
    tag_to_idx: Dict[str, int] = field(default_factory=dict)
    multi_span_args: set = field(default_factory=set)  # qualified "tool.arg" seen as multi-span in train
    canon: Dict[str, List[str]] = field(default_factory=dict)  # tool -> canonicalized args (e.g. datetime)
    ms_counts: Dict[str, int] = field(default_factory=dict)    # train-time: multi-span occurrences per tool.arg
    arg_counts: Dict[str, int] = field(default_factory=dict)   # train-time: located occurrences per tool.arg

    def finalize_multi_span(self, min_rate: float = 0.05) -> None:
        """Keep join semantics only for args that are multi-span in >= min_rate of TRAIN cases."""
        self.multi_span_args = {q for q, n in self.ms_counts.items()
                                if n / max(self.arg_counts.get(q, 1), 1) >= min_rate}

    @classmethod
    def from_rows(cls, rows: Sequence[dict]) -> "Registry":
        tools: List[str] = []
        args: Dict[str, List[str]] = {}
        required: Dict[str, List[str]] = {}
        canon: Dict[str, List[str]] = {}
        for r in rows:
            for t in r.get("tools") or []:
                fn = t["function"]
                name = fn["name"]
                if name in args:
                    continue
                tools.append(name)
                props = (fn.get("parameters") or {}).get("properties") or {}
                args[name] = [a for a in props if a not in CANON_ARGS]
                canon[name] = [a for a in props if a in CANON_ARGS]
                required[name] = list((fn.get("parameters") or {}).get("required") or [])
        tools.sort()
        reg = cls(tools=tools, args=args, required=required, canon=canon)
        reg._build_tags()
        return reg

    def _build_tags(self) -> None:
        tags = ["O"]
        for t in self.tools:
            for a in self.args[t]:
                tags += [f"B-{t}.{a}", f"I-{t}.{a}"]
        tags.append("<PAD>")
        self.tags = tags
        self.tag_to_idx = {t: i for i, t in enumerate(tags)}

    @property
    def pad_idx(self) -> int:
        return self.tag_to_idx["<PAD>"]

    def tag_indices_for_tools(self, tool_set: Sequence[str]) -> List[int]:
        """Allowed tag ids when exactly `tool_set` is called (O and PAD always allowed)."""
        keep = {self.tag_to_idx["O"], self.pad_idx}
        for t in tool_set:
            for a in self.args[t]:
                keep.add(self.tag_to_idx[f"B-{t}.{a}"])
                keep.add(self.tag_to_idx[f"I-{t}.{a}"])
        return sorted(keep)

    def to_dict(self) -> dict:
        return {"tools": self.tools, "args": self.args, "required": self.required, "canon": self.canon,
                "tags": self.tags, "multi_span_args": sorted(self.multi_span_args)}

    @classmethod
    def from_dict(cls, d: dict) -> "Registry":
        reg = cls(tools=d["tools"], args=d["args"], required=d["required"], canon=d.get("canon", {}),
                  multi_span_args=set(d.get("multi_span_args", [])))
        reg._build_tags()
        return reg


# ── span finding ────────────────────────────────────────────────────────────
def _find_free(text_l: str, value_l: str, taken: List[Tuple[int, int]], start: int = 0) -> int:
    i = text_l.find(value_l, start)
    while i >= 0:
        if all(i >= e or i + len(value_l) <= s for s, e in taken):
            return i
        i = text_l.find(value_l, i + 1)
    return -1


def locate_value(text: str, value: str, taken: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """Char spans whose text (joined by a space) equals `value` case-insensitively."""
    tl, vl = text.lower(), value.strip().lower()
    if not vl:
        return None
    i = _find_free(tl, vl, taken)
    if i >= 0:
        return [(i, i + len(vl))]
    words = vl.split()
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        ia = _find_free(tl, a, taken)
        if ia < 0:
            continue
        ib = _find_free(tl, b, taken + [(ia, ia + len(a))], ia + len(a))
        if ib >= 0:
            return [(ia, ia + len(a)), (ib, ib + len(b))]
    return None


@dataclass
class LabelledRow:
    text: str
    tool_vec: List[float]
    char_spans: List[Tuple[int, int, str]]   # (start, end, "tool.arg")
    dropped: int                             # gold args that could not be located


def label_row(row: dict, reg: Registry, record_multi_span: bool = True) -> LabelledRow:
    msgs = row["messages"]
    text = next(m["content"] for m in msgs if m.get("role") == "user")
    calls = msgs[-1].get("tool_calls") or []
    vec = [0.0] * len(reg.tools)
    spans: List[Tuple[int, int, str]] = []
    taken: List[Tuple[int, int]] = []
    dropped = 0
    for c in calls:
        fn = c["function"]
        name = fn["name"]
        vec[reg.tools.index(name)] = 1.0
        a = fn.get("arguments") or {}
        for arg in reg.args[name]:
            v = a.get(arg)
            if v is None or not str(v).strip():
                continue
            loc = locate_value(text, str(v), taken)
            if loc is None:
                dropped += 1
                continue
            if record_multi_span:
                q = f"{name}.{arg}"
                reg.arg_counts[q] = reg.arg_counts.get(q, 0) + 1
                if len(loc) > 1:
                    reg.ms_counts[q] = reg.ms_counts.get(q, 0) + 1
            for s, e in loc:
                spans.append((s, e, f"{name}.{arg}"))
                taken.append((s, e))
    return LabelledRow(text, vec, spans, dropped)


def token_tags(offsets: Sequence[Tuple[int, int]], attn: Sequence[int],
               spans: Sequence[Tuple[int, int, str]], reg: Registry) -> List[int]:
    """Map char spans to BIO tag ids over WordPiece tokens (special tokens -> O, padding -> PAD)."""
    out: List[int] = []
    prev_span = None
    o = reg.tag_to_idx["O"]
    for (s, e), m in zip(offsets, attn):
        if not m:
            out.append(reg.pad_idx)
            prev_span = None
            continue
        if s == e:  # [CLS]/[SEP]
            out.append(o)
            prev_span = None
            continue
        hit = next((sp for sp in spans if s >= sp[0] and e <= sp[1] + 1 and s < sp[1]), None)
        if hit is None:
            out.append(o)
            prev_span = None
        else:
            prefix = "I" if prev_span == hit else "B"
            out.append(reg.tag_to_idx[f"{prefix}-{hit[2]}"])
            prev_span = hit
    return out
