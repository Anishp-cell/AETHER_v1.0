"""Unit tests for the temporal canonicalizer (synthetic cases, no dataset rows)."""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "src" / "arn" / "canon" / "temporal.py"
_spec = importlib.util.spec_from_file_location("arn_canon_temporal_test", _P)
T = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = T
_spec.loader.exec_module(T)

REF = datetime(2026, 11, 12, 6, 25, 24)  # a Thursday


@pytest.mark.parametrize("text,expected", [
    ("for 2024-01-23 at 10:00 AM", "2024-01-23T10:00:00"),
    ("on March 20, 2026, at 11:00 AM", "2026-03-20T11:00:00"),
    ("on the 5th of December at 7:30 pm", "2026-12-05T19:30:00"),
    ("tomorrow at 4:00 PM", "2026-11-13T16:00:00"),
    ("day after tomorrow at noon", "2026-11-14T12:00:00"),
    ("next Tuesday at 2:30 PM", "2026-11-17T14:30:00"),
    ("this Thursday at 9am", "2026-11-19T09:00:00"),   # same weekday -> a week ahead
    ("at 14:30", "2026-11-12T14:30:00"),
    ("at 5:00 AM", "2026-11-13T05:00:00"),            # time already passed today -> tomorrow
    ("on January 3rd at 12:00 AM", "2027-01-03T00:00:00"),  # past date without year rolls forward
])
def test_canonicalize_cases(text, expected):
    assert T.canonicalize(text, REF) == expected


def test_no_temporal_expression_returns_none():
    assert T.canonicalize("turn on the flashlight", REF) is None


def test_missing_reference_returns_none():
    assert T.canonicalize("tomorrow at 4 PM", None) is None


def test_parse_reference_from_system_message():
    sys_msg = "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2025-06-04T15:29:23\n"
    assert T.parse_reference(sys_msg) == datetime(2025, 6, 4, 15, 29, 23)


def test_fit_policy_picks_following_when_data_says_so():
    # Synthetic data where "next <weekday>" always means the week after.
    ex = [("next Tuesday at 2:30 PM", REF, "2026-11-24T14:30:00"),
          ("next Friday at 10:00 AM", REF, "2026-11-20T10:00:00")]
    pol, acc = T.fit_policy(ex)
    assert pol.next_weekday == "following" and acc == 1.0
