"""
Deterministic temporal canonicalizer for ARN-v2 (plan component B).

ARN is extractive: it copies argument spans out of the user's text. Datetime
arguments are the one Mobile-Actions slot that is almost never verbatim (gold
values are ISO-8601 strings), so a typed canonicalizer maps the user's phrasing
to ISO against the reference time given in the developer/system message.

Integrity rules:
- Every convention below (weekday semantics, missing-year roll-forward,
  default time) is a *policy* whose value is chosen by `fit_policy()` on the
  TRAIN split only. Eval rows are never used to pick a policy.
- Results must be reported both with and without this layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_WD_RE = "|".join(sorted(WEEKDAYS, key=len, reverse=True))

_REF_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class TemporalPolicy:
    """Conventions that differ between datasets. Fit on train data only."""
    # "next <weekday>": "upcoming" = first such day after today; "following" = +7 on top
    next_weekday: str = "upcoming"
    # bare "<weekday>" / "this <weekday>": allow today (0 days ahead) or not
    bare_weekday_allows_today: bool = False
    # date given without a year that falls before the reference date -> next year
    roll_year_forward: bool = True
    # time used when the utterance gives a date but no time (HH, MM)
    default_time: Tuple[int, int] = (9, 0)


def parse_reference(system_text: str) -> Optional[datetime]:
    m = _REF_RE.search(system_text or "")
    return datetime.fromisoformat(m.group(1)) if m else None


# ── time of day ─────────────────────────────────────────────────────────────
_TIME_PATTERNS = [
    # 2:30 PM, 2 pm, 2:30p.m.
    (re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", re.I), "ampm"),
    # 14:30 / 09:00 (24h)
    (re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:a\.?m|p\.?m))", re.I), "24h"),
    (re.compile(r"\bnoon\b|\bmidday\b", re.I), "noon"),
    (re.compile(r"\bmidnight\b", re.I), "midnight"),
]


def _find_time(text: str) -> Optional[Tuple[int, int]]:
    for pat, kind in _TIME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if kind == "noon":
            return 12, 0
        if kind == "midnight":
            return 0, 0
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if kind == "ampm":
            pm = m.group(3).lower().startswith("p")
            if h == 12:
                h = 12 if pm else 0
            elif pm:
                h += 12
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return h, mi
    return None


# ── calendar date ───────────────────────────────────────────────────────────
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\b\.?(?:,?\s+(\d{{4}}))?", re.I)
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_REL_DAY = re.compile(r"\b(day after tomorrow|tomorrow|today|tonight|this (?:morning|afternoon|evening))\b", re.I)
_IN_N = re.compile(r"\bin\s+(\d+|a|one|two|three|four|five|six|seven)\s+(day|week)s?\b", re.I)
_WEEKDAY = re.compile(rf"\b(?:(this coming|this|next|coming|upcoming)\s+)?({_WD_RE})\b", re.I)
_NEXT_WEEK = re.compile(r"\bnext week\b", re.I)
_WORDNUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def _safe_date(y: int, m: int, d: int) -> Optional[datetime]:
    try:
        return datetime(y, m, d)
    except ValueError:
        return None


def _explicit_date(text: str, ref: datetime, pol: TemporalPolicy) -> Optional[datetime]:
    m = _ISO_DATE.search(text)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    cand = None
    m = _MDY.search(text)
    if m:
        cand = (MONTHS[m.group(1).lower()], int(m.group(2)), m.group(3))
    else:
        m = _DMY.search(text)
        if m:
            cand = (MONTHS[m.group(2).lower()], int(m.group(1)), m.group(3))
        else:
            m = _SLASH.search(text)
            if m:
                yr = m.group(3)
                if yr and len(yr) == 2:
                    yr = "20" + yr
                cand = (int(m.group(1)), int(m.group(2)), yr)
    if cand is None:
        return None
    month, day, year = cand
    if year:
        return _safe_date(int(year), month, day)
    d = _safe_date(ref.year, month, day)
    if d and pol.roll_year_forward and d.date() < ref.date():
        d = _safe_date(ref.year + 1, month, day)
    return d


def _relative_date(text: str, ref: datetime, pol: TemporalPolicy) -> Optional[datetime]:
    base = ref.replace(hour=0, minute=0, second=0, microsecond=0)
    m = _REL_DAY.search(text)
    if m:
        w = m.group(1).lower()
        if w == "day after tomorrow":
            return base + timedelta(days=2)
        if w == "tomorrow":
            return base + timedelta(days=1)
        return base
    m = _IN_N.search(text)
    if m:
        n = m.group(1).lower()
        n = int(n) if n.isdigit() else _WORDNUM[n]
        return base + timedelta(days=n * (7 if m.group(2).lower() == "week" else 1))
    m = _WEEKDAY.search(text)
    if m:
        mod = (m.group(1) or "").lower()
        target = WEEKDAYS[m.group(2).lower()]
        ahead = (target - base.weekday()) % 7
        same_day = ahead == 0
        if same_day and not (mod == "" and pol.bare_weekday_allows_today):
            ahead = 7
        # "following": "next X" means the X of next week, i.e. one week beyond the upcoming X
        # (fixes the same-weekday case flagged by the external review, where +7 was returned).
        if mod == "next" and pol.next_weekday == "following":
            ahead = (7 if same_day else ahead) + 7
        return base + timedelta(days=ahead)
    if _NEXT_WEEK.search(text):
        return base + timedelta(days=7)
    return None


def canonicalize(text: str, ref: Optional[datetime], pol: TemporalPolicy = TemporalPolicy()) -> Optional[str]:
    """Map an utterance (or an extracted span) to 'YYYY-MM-DDTHH:MM:SS', or None."""
    if ref is None:
        return None
    day = _explicit_date(text, ref, pol) or _relative_date(text, ref, pol)
    tod = _find_time(text)
    if day is None and tod is None:
        return None
    if day is None:
        day = ref.replace(hour=0, minute=0, second=0, microsecond=0)
        if tod and (tod[0], tod[1]) <= (ref.hour, ref.minute):
            day += timedelta(days=1)
    h, mi = tod if tod else pol.default_time
    return day.replace(hour=h, minute=mi, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")


# ── policy fitting (TRAIN ONLY) ─────────────────────────────────────────────
def iter_datetime_examples(rows: Iterable[dict]) -> Iterable[Tuple[str, datetime, str]]:
    """Yield (user_text, reference_dt, gold_iso) for every datetime argument."""
    for r in rows:
        msgs = r["messages"]
        ref = parse_reference(msgs[0].get("content", ""))
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        for c in msgs[-1].get("tool_calls") or []:
            g = (c.get("function", {}).get("arguments") or {}).get("datetime")
            if g and ref:
                yield user, ref, str(g)


def accuracy(examples: List[Tuple[str, datetime, str]], pol: TemporalPolicy) -> float:
    if not examples:
        return 0.0
    ok = sum(canonicalize(t, ref, pol) == g[:19] for t, ref, g in examples)
    return ok / len(examples)


def fit_policy(train_examples: List[Tuple[str, datetime, str]]) -> Tuple[TemporalPolicy, float]:
    """Grid-search the conventions on TRAIN examples; returns (best policy, train acc)."""
    best, best_acc = TemporalPolicy(), -1.0
    for nw in ("upcoming", "following"):
        for today in (False, True):
            for roll in (True, False):
                for dt in ((9, 0), (10, 0), (12, 0), (0, 0)):
                    p = TemporalPolicy(nw, today, roll, dt)
                    a = accuracy(train_examples, p)
                    if a > best_acc:
                        best, best_acc = p, a
    return best, best_acc
