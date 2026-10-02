"""Jackpot outlook and jackpot history (outlook.py)."""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd
import pytest

from huatbot import constants as C
from huatbot.models import NextToto, PrizeRules
from huatbot.outlook import (
    FALLBACK_BOARDS,
    chance_won_by_cascade,
    estimate_next_jackpot,
    jackpot_history,
    jackpot_outlook,
    next_regular_draw_days,
)
from huatbot.store import normalise_toto
from huatbot.synth import SG, synth_next_toto, synth_toto


def history(rows: list[tuple[int, str, float, int, str]]) -> pd.DataFrame:
    """Rows of (draw, ISO date, jackpot, Group 1 winners, draw type) with clean other groups."""
    out = []
    for n, d, jackpot, winners, kind in rows:
        out.append({
            "draw_number": n, "draw_date": d, "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6,
            "additional": 7, "jackpot": jackpot, "draw_type": kind,
            "g1_share": jackpot / winners if winners else float("nan"), "g1_winners": winners,
            "g2_share": 90_000.0, "g2_winners": 2, "g3_share": 2_000.0, "g3_winners": 120,
            "g4_share": 300.0, "g4_winners": 450, "g5_share": 50.0, "g5_winners": 9000,
            "g6_share": 25.0, "g6_winners": 12000, "g7_share": 10.0, "g7_winners": 160_000,
        })
    return normalise_toto(pd.DataFrame(out))


def next_page(day: date, jackpot: float | None, kind: str = "normal") -> NextToto:
    return NextToto(draw_datetime=datetime.combine(day, C.DRAW_TIME, tzinfo=SG), jackpot_estimate=jackpot,
                    draw_type=kind, draw_type_hint=None if kind == "normal" else kind)


def test_next_regular_draw_days_are_mondays_and_thursdays():
    thu = date(2026, 10, 1)
    assert next_regular_draw_days(thu, 4) == [date(2026, 10, 5), date(2026, 10, 8),
                                              date(2026, 10, 12), date(2026, 10, 15)]
    assert next_regular_draw_days(thu, 0) == []


def test_projection_runs_to_the_cascade(rules):
    df = synth_toto(n_draws=600)
    nt = synth_next_toto(df)
    today = df["draw_date"].max().date()
    out = jackpot_outlook(df, rules, nt, today=today)
    assert out.jackpot == nt.jackpot_estimate
    assert out.draws_to_cascade == C.TOTO_SNOWBALL_LIMIT - out.snowball_draws
    assert len(out.steps) == out.draws_to_cascade
    assert [s.index for s in out.steps] == list(range(1, len(out.steps) + 1))
    assert out.steps[0].draw_date == nt.draw_datetime.date()
    assert out.steps[0].chance_reached == 1.0
    assert [s.cascade for s in out.steps] == [False] * (len(out.steps) - 1) + [True]
    for a, b in zip(out.steps, out.steps[1:]):
        assert b.jackpot > a.jackpot  # it snowballs
        assert b.chance_reached == pytest.approx(a.chance_reached * (1 - a.chance_won))
        assert b.draw_date > a.draw_date and b.draw_date.weekday() in C.TOTO_WEEKDAYS
    for s in out.steps:
        assert 0 < s.chance_won < 1
        assert s.chance_won == pytest.approx(1 - math.exp(-s.boards / C.TOTO_COMBOS))
        assert s.ev_per_dollar is not None and s.ev_per_dollar > 0.2
    assert out.biggest is out.steps[-1]
    expected = 1 - out.steps[-1].chance_reached * (1 - out.steps[-1].chance_won)
    assert chance_won_by_cascade(out) == pytest.approx(expected)


def test_growth_adds_38_percent_of_the_next_pool(rules):
    df = synth_toto(n_draws=600)
    out = jackpot_outlook(df, rules, synth_next_toto(df), today=df["draw_date"].max().date())
    if len(out.steps) < 2:
        pytest.skip("synthetic history ends one draw before a cascade")
    a, b = out.steps[0], out.steps[1]
    added = b.jackpot - a.jackpot
    assert added == pytest.approx(0.38 * 0.54 * b.boards, rel=0.05)


def test_cascade_draw_is_a_single_step(rules):
    df = history([(100, "2026-09-17", 2e6, 0, "normal"), (101, "2026-09-21", 3e6, 0, "normal"),
                  (102, "2026-09-24", 4e6, 0, "normal"), (103, "2026-09-28", 5e6, 0, "normal")])
    df = df[df["draw_number"] > 100].reset_index(drop=True)  # three rollovers in a row
    out = jackpot_outlook(df, rules, next_page(date(2026, 10, 1), 6e6, "cascade"), today=date(2026, 9, 29))
    assert out.draw_type == "cascade" and out.draws_to_cascade == 1
    assert len(out.steps) == 1 and out.steps[0].cascade
    # Without the page the cascade is predicted from the three rollovers.
    out = jackpot_outlook(df, rules, None, today=date(2026, 9, 29))
    assert out.draw_type == "cascade" and len(out.steps) == 1


def test_hongbao_draw_has_its_own_jackpot(rules):
    df = history([(200, "2027-01-28", 1e6, 1, "normal")])
    page = next_page(date(2027, 2, 5), 12e6, "hongbao")
    out = jackpot_outlook(df, rules, page, today=date(2027, 1, 30))
    assert out.draws_to_cascade is None and len(out.steps) == 1
    assert out.steps[0].jackpot == 12e6
    assert (date(2027, 2, 5), "hongbao") in out.special_draws
    assert any("Hongbao" in n for n in out.notes)


def test_page_about_a_draw_already_held_is_ignored(rules):
    df = history([(300, "2026-09-28", 1e6, 1, "normal"), (301, "2026-10-01", 1.8e6, 0, "normal")])
    stale = next_page(date(2026, 10, 1), 1.8e6)  # still shows draw 301
    out = jackpot_outlook(df, rules, stale, today=date(2026, 10, 1))
    assert out.jackpot != 1.8e6
    assert out.jackpot == pytest.approx(estimate_next_jackpot(df, rules))
    assert out.jackpot > 1.8e6  # the unwon jackpot rolls over and grows
    assert out.steps[0].draw_date == date(2026, 10, 5)
    assert any("worked out from the stored results" in n for n in out.notes)


def test_after_a_win_the_estimate_restarts_at_the_minimum(rules):
    df = history([(400, "2026-09-28", 3e6, 1, "normal")])
    assert estimate_next_jackpot(df, rules) >= rules.min_group1
    assert estimate_next_jackpot(df, rules) < 3e6


def test_empty_history(rules):
    out = jackpot_outlook(history([]), rules, None, today=date(2026, 10, 2))
    assert out.jackpot is None and out.steps == [] and out.snowball_draws == 0
    assert chance_won_by_cascade(out) is None
    out = jackpot_outlook(history([]), rules, next_page(date(2026, 10, 5), 1e6), today=date(2026, 10, 2))
    assert out.steps and out.steps[0].boards == FALLBACK_BOARDS
    assert any("rough" in n for n in out.notes)


def test_announced_special_draws_are_listed_when_still_ahead(rules):
    df = history([(500, "2026-10-01", 1e6, 1, "normal")])
    announced = [(date(2026, 9, 25), "special"), (date(2026, 10, 9), "special"), (date(2026, 10, 5), "normal")]
    out = jackpot_outlook(df, rules, None, today=date(2026, 10, 2), announced=announced)
    assert out.special_draws == [(date(2026, 10, 9), "special")]


def test_history_counts():
    df = history([
        (1, "2026-01-01", 1e6, 0, "normal"),
        (2, "2026-01-05", 2e6, 2, "normal"),      # won after 2 draws
        (3, "2026-01-08", 1e6, 0, "normal"),
        (4, "2026-01-12", 2e6, 0, "normal"),
        (5, "2026-01-15", 3e6, 0, "normal"),
        (6, "2026-01-19", 4e6, 0, "cascade"),     # cascaded after 4 draws
        (7, "2026-01-22", 12e6, 1, "hongbao"),    # won at once
        (8, "2026-01-26", 1e6, 0, "normal"),      # still running
    ])
    h = jackpot_history(df)
    assert h.draws == 8 and h.won_draws == 2 and h.cascades == 1
    assert h.won_share == pytest.approx(2 / 8)
    assert h.average_run == pytest.approx((2 + 4 + 1) / 3)
    assert h.typical_won == pytest.approx((2e6 + 12e6) / 2)
    assert [b["draw_number"] for b in h.biggest] == [7, 6, 5, 4, 2]
    assert h.last_won["draw_number"] == 7 and h.last_won["winners"] == 1


def test_history_of_nothing():
    h = jackpot_history(history([]))
    assert h.draws == 0 and h.won_share is None and h.biggest == [] and h.last_won is None


@pytest.fixture
def rules():
    return PrizeRules()
