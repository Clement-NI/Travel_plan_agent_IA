"""Tests for the regex fast-path in MCP/skills.py.

These cover the pure parsing layer (parse_date, parse_query, _detect_mode,
_iata, _split_trailing_date). No MCP subprocesses are spawned, no network
is hit — every function under test is deterministic given a fixed `today`.
"""
from __future__ import annotations

from datetime import date

import pytest

from MCP.skills import (
    _detect_mode,
    _iata,
    _split_trailing_date,
    parse_date,
    parse_query,
)


TODAY = date(2026, 5, 14)  # a Thursday


# ---------- parse_date ----------

class TestParseDate:
    def test_iso(self):
        assert parse_date("2026-05-20", today=TODAY) == date(2026, 5, 20)

    def test_today_tomorrow(self):
        assert parse_date("today", today=TODAY) == TODAY
        assert parse_date("tomorrow", today=TODAY) == date(2026, 5, 15)
        assert parse_date("day after tomorrow", today=TODAY) == date(2026, 5, 16)

    def test_empty_returns_today(self):
        assert parse_date("", today=TODAY) == TODAY
        assert parse_date("   ", today=TODAY) == TODAY

    def test_next_weekday(self):
        # TODAY is Thursday (2026-05-14). "next monday" → 2026-05-18.
        assert parse_date("next monday", today=TODAY) == date(2026, 5, 18)
        # "this friday" → tomorrow's tomorrow (2026-05-15 is Fri).
        assert parse_date("this friday", today=TODAY) == date(2026, 5, 15)

    def test_bare_weekday_picks_next_occurrence(self):
        # "friday" from Thursday → next day, not 7 days later.
        assert parse_date("friday", today=TODAY) == date(2026, 5, 15)
        # "thursday" from Thursday → 7 days later (not today).
        assert parse_date("thursday", today=TODAY) == date(2026, 5, 21)

    def test_in_n_days(self):
        assert parse_date("in 3 days", today=TODAY) == date(2026, 5, 17)
        assert parse_date("in 2 weeks", today=TODAY) == date(2026, 5, 28)
        assert parse_date("in 1 day", today=TODAY) == date(2026, 5, 15)

    def test_month_day_variants(self):
        assert parse_date("may 20", today=TODAY) == date(2026, 5, 20)
        assert parse_date("20 may", today=TODAY) == date(2026, 5, 20)
        assert parse_date("May 20 2027", today=TODAY) == date(2027, 5, 20)
        assert parse_date("20 may 2027", today=TODAY) == date(2027, 5, 20)
        # abbreviations
        assert parse_date("jun 5", today=TODAY) == date(2026, 6, 5)

    def test_past_month_day_rolls_to_next_year(self):
        # April already passed in May → rolls to 2027.
        assert parse_date("april 10", today=TODAY) == date(2027, 4, 10)

    def test_invalid_returns_none(self):
        assert parse_date("nonsense", today=TODAY) is None
        assert parse_date("2026-13-40", today=TODAY) is None
        assert parse_date("february 30", today=TODAY) is None


# ---------- _iata ----------

class TestIata:
    def test_known_cities(self):
        assert _iata("Paris") == "CDG"
        assert _iata("paris") == "CDG"
        assert _iata("  LYON  ") == "LYS"
        assert _iata("Barcelona") == "BCN"

    def test_unknown_city(self):
        assert _iata("Atlantis") is None
        assert _iata("") is None


# ---------- _detect_mode ----------

class TestDetectMode:
    def test_single_mode_wins(self):
        assert _detect_mode("flights from Paris to Lyon") == "flight"
        assert _detect_mode("TGV from Paris to Bordeaux") == "train"
        assert _detect_mode("bus to Berlin") == "bus"

    def test_multiple_modes_become_plan(self):
        assert _detect_mode("flights or trains to Madrid") == "plan"

    def test_bare_route_defaults_to_plan(self):
        assert _detect_mode("Paris to Lyon") == "plan"

    def test_explicit_plan_keywords(self):
        assert _detect_mode("plan a trip from Paris to Lyon") == "plan"
        assert _detect_mode("best way from Paris to Lyon") == "plan"


# ---------- _split_trailing_date ----------

class TestSplitTrailingDate:
    def test_no_trailing_date(self):
        assert _split_trailing_date("Lille Europe", TODAY) == ("Lille Europe", "")

    def test_iso_suffix(self):
        assert _split_trailing_date("Lyon 2026-05-20", TODAY) == ("Lyon", "2026-05-20")

    def test_word_suffix(self):
        assert _split_trailing_date("Lyon tomorrow", TODAY) == ("Lyon", "tomorrow")
        assert _split_trailing_date("Lyon next monday", TODAY) == ("Lyon", "next monday")

    def test_multiword_city_preserved(self):
        city, date_str = _split_trailing_date("Lille Europe next tuesday", TODAY)
        assert city == "Lille Europe"
        assert date_str == "next tuesday"

    def test_typo_recovery(self):
        # "tomorrw" → corrected to "tomorrow" and peeled off.
        city, date_str = _split_trailing_date("Lyon tomorrw", TODAY)
        assert city == "Lyon"
        assert date_str == "tomorrow"


# ---------- parse_query (the headline function) ----------

class TestParseQuery:
    def test_basic_from_to_on_date(self, monkeypatch):
        # parse_query uses date.today() internally; freeze it.
        _freeze_today(monkeypatch)
        out = parse_query("flights from Paris to Lyon on 2026-05-20")
        assert out == {
            "mode": "flight",
            "from": "Paris",
            "to": "Lyon",
            "date": date(2026, 5, 20),
        }

    def test_bare_route_defaults_to_plan_and_today(self, monkeypatch):
        _freeze_today(monkeypatch)
        out = parse_query("Paris to Lyon")
        assert out["mode"] == "plan"
        assert out["date"] == TODAY

    def test_trailing_date_without_on(self, monkeypatch):
        _freeze_today(monkeypatch)
        out = parse_query("trains from Paris to Bordeaux tomorrow")
        assert out["mode"] == "train"
        assert out["to"] == "Bordeaux"
        assert out["date"] == date(2026, 5, 15)

    def test_date_before_route(self, monkeypatch):
        # The date-anywhere pre-pass only kicks in cleanly when the route
        # is at the tail. Trailing modifiers ("by flight") would get glued
        # onto to_city — that's a known fast-path limitation.
        _freeze_today(monkeypatch)
        out = parse_query("on May 20 from Paris to Lyon")
        assert out is not None
        assert out["from"] == "Paris"
        assert out["to"] == "Lyon"
        assert out["date"] == date(2026, 5, 20)

    def test_punctuation_stripped_from_destination(self, monkeypatch):
        _freeze_today(monkeypatch)
        out = parse_query("Paris to Lyon?")
        assert out["to"] == "Lyon"

    def test_noise_words_in_destination_punt_to_llm(self, monkeypatch):
        _freeze_today(monkeypatch)
        # "Lyon what is the best route" should NOT parse — "what" is a noise token.
        assert parse_query("Paris to Lyon what is the best route") is None

    def test_past_date_punts_to_llm(self, monkeypatch):
        _freeze_today(monkeypatch)
        # 2026-01-01 is before TODAY (2026-05-14).
        assert parse_query("flights from Paris to Lyon on 2026-01-01") is None

    def test_unrecognized_date_punts_to_llm(self, monkeypatch):
        _freeze_today(monkeypatch)
        assert parse_query("flights from Paris to Lyon on the third weekend of June") is None

    def test_empty_input(self):
        assert parse_query("") is None
        assert parse_query("   ") is None

    def test_no_route_returns_none(self, monkeypatch):
        _freeze_today(monkeypatch)
        assert parse_query("what's the weather like in Paris") is None

    def test_multiword_city(self, monkeypatch):
        _freeze_today(monkeypatch)
        out = parse_query("trains from Paris to Aix en Provence tomorrow")
        assert out is not None
        assert out["to"] == "Aix en Provence"


# ---------- helpers ----------

def _freeze_today(monkeypatch):
    """Pin `date.today()` inside MCP.skills so parse_query is deterministic."""
    import MCP.skills as skills_mod

    class _FrozenDate(skills_mod._date):
        @classmethod
        def today(cls):
            return TODAY

    monkeypatch.setattr(skills_mod, "_date", _FrozenDate)
