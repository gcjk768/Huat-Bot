"""Tests for huatbot.buysignal: expected value formulas, sales estimate and labels."""
from __future__ import annotations

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from huatbot import buysignal as B
from huatbot import constants as C
from huatbot.analysis_toto import crowd_table
from huatbot.models import NextToto, Settings
from huatbot.store import no_winner_streak, normalise_toto

DASHES = re.compile("[-–—]")
SG = ZoneInfo(C.SG_TZ_NAME)
BIG_C = 13_983_816


def nxt(jackpot, draw_type="normal") -> NextToto:
    return NextToto(draw_datetime=datetime(2026, 10, 5, 18, 30, tzinfo=SG), jackpot_estimate=jackpot,
                    draw_type=draw_type, draw_type_hint=None)


# Expected value


def test_ev_reference_point(rules):
    ev = B.toto_ev_per_dollar(1_000_000, 2_600_000, rules)
    assert set(ev) == {"g1", "g2", "g3", "g4", "fixed", "cascade", "total"}
    assert 0.38 <= ev["total"] <= 0.40
    assert ev["fixed"] == pytest.approx(0.2412, abs=5e-5)
    assert ev["fixed"] == pytest.approx((50 * 12_915 + 25 * 17_220 + 10 * 229_600) / BIG_C)
    assert ev["cascade"] == 0.0
    parts = ev["g1"] + ev["g2"] + ev["g3"] + ev["g4"] + ev["fixed"] + ev["cascade"]
    assert ev["total"] == pytest.approx(parts)


def test_ev_matches_formulas(rules):
    jackpot, boards = 4_000_000.0, 5_000_000.0
    ev = B.toto_ev_per_dollar(jackpot, boards, rules)
    lam1 = boards / BIG_C
    assert ev["g1"] == pytest.approx(jackpot * (1 - math.exp(-lam1)) / lam1 / BIG_C)
    for g in (2, 3, 4):
        lam = boards * C.TOTO_GROUP_COMBOS[g] / BIG_C
        assert ev[f"g{g}"] == pytest.approx(rules.group_pool_pct[g] * 0.54 * (1 - math.exp(-lam)))


def test_ev_rises_with_jackpot(rules):
    totals = [B.toto_ev_per_dollar(j, 4_000_000, rules)["total"] for j in (1e6, 2e6, 5e6, 1e7, 3e7)]
    assert all(b > a for a, b in zip(totals, totals[1:]))


def test_g1_share_factor_tends_to_one_with_few_boards(rules):
    jackpot = 2_000_000.0
    alone = jackpot / BIG_C
    assert B.toto_ev_per_dollar(jackpot, 0, rules)["g1"] == pytest.approx(alone, rel=1e-12)
    assert B.toto_ev_per_dollar(jackpot, 1, rules)["g1"] == pytest.approx(alone, rel=1e-6)
    factors = [B.toto_ev_per_dollar(jackpot, b, rules)["g1"] / alone for b in (1e7, 1e6, 1e4, 1e2)]
    assert all(b > a for a, b in zip(factors, factors[1:]))
    assert factors[-1] == pytest.approx(1.0, abs=1e-5)
    assert factors[0] < 0.75  # 10 million boards: real chance of sharing the jackpot
    # No sales at all: Groups 2 to 4 pay nothing because nobody else funds them.
    zero = B.toto_ev_per_dollar(jackpot, 0, rules)
    assert zero["g2"] == zero["g3"] == zero["g4"] == 0.0


def test_cascade_term_only_for_cascade_and_hongbao(rules):
    jackpot, boards = 8_000_000.0, 6_000_000.0
    base = B.toto_ev_per_dollar(jackpot, boards, rules, "normal")
    for draw_type in ("normal", "special"):
        assert B.toto_ev_per_dollar(jackpot, boards, rules, draw_type)["cascade"] == 0.0
    lam1, lam2 = boards / BIG_C, boards * 6 / BIG_C
    expected = math.exp(-lam1) * jackpot * (6 / BIG_C) * (1 - math.exp(-lam2)) / lam2
    for draw_type in ("cascade", "hongbao"):
        ev = B.toto_ev_per_dollar(jackpot, boards, rules, draw_type)
        assert ev["cascade"] == pytest.approx(expected)
        assert ev["total"] == pytest.approx(base["total"] + expected)


# Sales estimate


def test_estimate_boards_on_synthetic_history(toto_df, rules):
    boards, method = B.estimate_boards(toto_df, 1_000_000, rules)
    # synth sells about 2.3 million + 0.55 x (2 million prospective jackpot) at the minimum.
    assert 2_500_000 < boards < 5_000_000
    assert re.fullmatch(r"median of \d+ past draws with a jackpot near \$1,000,000", method)
    assert not DASHES.search(method)

    table = crowd_table(toto_df, rules)
    same, _ = B.estimate_boards(toto_df, 1_000_000, rules, table=table)
    assert same == boards

    far, far_method = B.estimate_boards(toto_df, 40_000_000, rules, table=table)
    assert far is not None and far > boards
    assert far_method.startswith("trend of sales against jackpot over")
    assert "no past draw had a jackpot near $40,000,000" in far_method
    assert not DASHES.search(far_method)


def _jackpot_frame(jackpots: list[float]) -> pd.DataFrame:
    rows = [{"draw_number": 100 + i, "draw_date": pd.Timestamp("2025-01-02") + pd.Timedelta(days=3 * i),
             "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "additional": 7, "jackpot": j}
            for i, j in enumerate(jackpots)]
    return normalise_toto(pd.DataFrame(rows))


def _boards_table(df: pd.DataFrame, boards) -> pd.DataFrame:
    return pd.DataFrame({"boards": np.asarray(boards, dtype=float)},
                        index=pd.Index(df["draw_number"].to_numpy(), name="draw_number"))


def test_estimate_boards_median_of_near_draws(rules):
    jackpots = [1_000_000.0] * 4 + [1_200_000.0] * 4 + [3_000_000.0] * 3
    df = _jackpot_frame(jackpots)
    boards = [3.0e6, 3.1e6, 3.2e6, 3.3e6, 3.4e6, 3.5e6, 3.6e6, np.nan, 9e6, 9e6, 9e6]
    est, method = B.estimate_boards(df, 1_100_000, rules, table=_boards_table(df, boards))
    # 7 usable near draws only (one has no clean estimate): falls through to "not enough history".
    assert (est, method) == (None, "not enough history")

    boards[7] = 3.7e6
    est, method = B.estimate_boards(df, 1_100_000, rules, table=_boards_table(df, boards))
    assert est == pytest.approx(np.median([3.0e6, 3.1e6, 3.2e6, 3.3e6, 3.4e6, 3.5e6, 3.6e6, 3.7e6]))
    assert method == "median of 8 past draws with a jackpot near $1,100,000"


def test_estimate_boards_log_log_fit(rules):
    jackpots = np.geomspace(1e6, 2e7, 25)
    df = _jackpot_frame(list(jackpots))
    boards = 1000.0 * jackpots ** 0.5  # exact power law: the log log fit recovers it
    est, method = B.estimate_boards(df, 5e7, rules, table=_boards_table(df, boards))
    assert est == pytest.approx(1000.0 * (5e7) ** 0.5, rel=1e-9)
    assert method == ("trend of sales against jackpot over 25 past draws, "
                      "as no past draw had a jackpot near $50,000,000")
    est, method = B.estimate_boards(df, 1.5e7, rules, table=_boards_table(df, boards))
    n_near = int(((jackpots >= 1.5e7 / 1.25) & (jackpots <= 1.5e7 * 1.25)).sum())
    assert 1 < n_near < 8
    assert f"as only {n_near} past draws had a jackpot near $15,000,000" in method
    assert est == pytest.approx(1000.0 * (1.5e7) ** 0.5, rel=1e-9)


def test_estimate_boards_not_enough_history(rules):
    df = _jackpot_frame([1e6, 5e6, 9e6])
    assert B.estimate_boards(df, 2e7, rules, table=_boards_table(df, [3e6, 5e6, 7e6])) == (
        None, "not enough history")
    assert B.estimate_boards(df, float("nan"), rules)[0] is None


# Labels


@pytest.mark.parametrize("jackpot,label", [
    (5_000_000.0, "HIGH"), (3_000_000.0, "HIGH"), (2_999_999.0, "MEDIUM"), (2_000_000.0, "MEDIUM"),
    (1_002_000.0, "MEDIUM"), (1_000_500.0, "LOW"), (1_000_000.0, "LOW"),
])
def test_labels_for_normal_draws(toto_df, settings, rules, jackpot, label):
    sig = B.buy_signal(nxt(jackpot), toto_df, settings, rules)
    assert sig.label == label
    assert sig.jackpot == jackpot and sig.draw_type == "normal"
    assert sig.ev_per_dollar == pytest.approx(sig.ev_breakdown["total"])
    assert 0.3 < sig.ev_per_dollar < 1.0
    assert sig.boards_estimate is not None and sig.boards_method
    assert not DASHES.search(sig.reason) and not DASHES.search(sig.boards_method)
    assert sig.reason.endswith(".")


@pytest.mark.parametrize("draw_type", ["cascade", "hongbao", "special"])
def test_special_draws_follow_alert_setting(toto_df, rules, draw_type):
    on, off = Settings(alert_on_special_draws=True), Settings(alert_on_special_draws=False)
    assert B.buy_signal(nxt(2_000_000.0, draw_type), toto_df, on, rules).label == "HIGH"
    assert B.buy_signal(nxt(2_000_000.0, draw_type), toto_df, off, rules).label == "MEDIUM"
    # HIGH beats LOW when the user asked for special draw alerts.
    assert B.buy_signal(nxt(1_000_000.0, draw_type), toto_df, on, rules).label == "HIGH"
    assert B.buy_signal(nxt(1_000_000.0, draw_type), toto_df, off, rules).label == "LOW"
    sig = B.buy_signal(nxt(2_000_000.0, draw_type), toto_df, on, rules)
    assert sig.draw_type == draw_type and not DASHES.search(sig.reason)
    assert (sig.ev_breakdown["cascade"] > 0) == (draw_type in ("cascade", "hongbao"))


def test_custom_alert_threshold(toto_df, rules):
    sig = B.buy_signal(nxt(2_000_000.0), toto_df, Settings(jackpot_alert=1_500_000.0), rules)
    assert sig.label == "HIGH" and "$1,500,000" in sig.reason


def test_missing_jackpot_is_medium(toto_df, settings, rules):
    for next_toto in (None, nxt(None)):
        sig = B.buy_signal(next_toto, toto_df, settings, rules)
        assert sig.label == "MEDIUM"
        assert sig.jackpot is None and sig.ev_per_dollar is None and sig.ev_breakdown == {}
        assert "jackpot estimate not available" in sig.reason
        assert sig.no_winner_streak == no_winner_streak(toto_df)
        assert not DASHES.search(sig.reason)
    # A known cascade draw still raises the alert without a jackpot figure.
    sig = B.buy_signal(nxt(None, "cascade"), toto_df, settings, rules)
    assert sig.label == "HIGH" and "jackpot estimate not available" in sig.reason
    # Empty history and no next draw information: still a usable signal.
    sig = B.buy_signal(None, normalise_toto(pd.DataFrame()), settings, rules)
    assert sig.label == "MEDIUM" and sig.no_winner_streak == 0


def test_no_winner_streak_and_predicted_cascade(settings, rules):
    rows = [{"draw_number": 200 + i, "draw_date": pd.Timestamp("2026-09-01") + pd.Timedelta(days=3 * i),
             "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "additional": 7,
             "jackpot": 1e6, "g1_winners": w} for i, w in enumerate([0, 1, 0, 0, 0])]
    df = normalise_toto(pd.DataFrame(rows))
    sig = B.buy_signal(None, df, settings, rules)
    assert sig.no_winner_streak == 3
    # Three unwon draws in a row: the next one is the cascade draw.
    assert sig.draw_type == "cascade" and sig.label == "HIGH"
    # The page's draw type wins when available.
    assert B.buy_signal(nxt(4e6, "normal"), df, settings, rules).draw_type == "normal"


def test_signal_with_synthetic_next_draw(toto_df, next_toto, settings, rules):
    table = crowd_table(toto_df, rules)
    sig = B.buy_signal(next_toto, toto_df, settings, rules, table=table)
    boards, method = B.estimate_boards(toto_df, next_toto.jackpot_estimate, rules, table=table)
    assert sig.boards_estimate == boards and sig.boards_method == method
    expected = B.toto_ev_per_dollar(next_toto.jackpot_estimate, boards, rules, next_toto.draw_type)
    assert sig.ev_breakdown == expected
    assert sig.label in ("HIGH", "MEDIUM", "LOW")


def test_an_average_above_one_dollar_carries_a_caveat(toto_df, settings, rules):
    # A cascade draw near $3.9M averages more than $1 back per $1, almost all of it from a prize
    # nearly nobody wins: the reason must say so, in the same single sentence.
    sig = B.buy_signal(nxt(3_900_000.0, "cascade"), toto_df, settings, rules)
    assert sig.ev_per_dollar > 1
    assert "almost every ticket still loses" in sig.reason
    assert f"7 boards in {C.TOTO_COMBOS:,} can win" in sig.reason  # Group 1 plus the 6 Group 2 boards
    assert "above your budget" in sig.reason
    assert sig.reason.endswith(".") and not re.search(r"\.\s", sig.reason)  # one sentence
    assert not DASHES.search(sig.reason)
    # A normal draw at $2.1M averages well under $1: no caveat.
    normal = B.buy_signal(nxt(2_100_000.0), toto_df, settings, rules)
    assert normal.ev_per_dollar < 1
    assert "almost every ticket still loses" not in normal.reason
