"""Tests for huatbot.analysis_fourd: long form, digit frequencies, repeats, digit sets,
chi square verdicts and bet type value."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from huatbot import analysis_fourd as A
from huatbot import constants as C
from huatbot.models import FOURD_NUMBER_COLUMNS, PrizeRules
from huatbot.store import normalise_fourd

DASHES = ("-", "–", "—")


def no_dashes(text: str) -> bool:
    return not any(d in text for d in DASHES)


def make_df(draws: list[tuple[int, str, list[str]]]) -> pd.DataFrame:
    """fourd.csv style frame from (draw number, ISO date, up to 23 numbers in slot order)."""
    rows = []
    for draw_number, day, numbers in draws:
        row = {"draw_number": draw_number, "draw_date": day, "fetched_at": "test"}
        padded = list(numbers) + [""] * (len(FOURD_NUMBER_COLUMNS) - len(numbers))
        row.update(dict(zip(FOURD_NUMBER_COLUMNS, padded)))
        rows.append(row)
    return normalise_fourd(pd.DataFrame(rows))


def filler(start: int, n: int = 23) -> list[str]:
    """``n`` distinct numbers that never collide with the hand picked ones below (all >= 9000)."""
    return [f"{9000 + start + i:04d}" for i in range(n)]


# all_numbers


def test_all_numbers_long_form(fourd_df):
    long = A.all_numbers(fourd_df)
    assert list(long.columns) == ["draw_number", "draw_date", "tier", "number"]
    assert len(long) == 23 * len(fourd_df)
    per_draw = long.groupby(["draw_number", "tier"]).size().unstack()
    assert all((per_draw[t] == n).all() for t, n in C.FOURD_TIER_COUNTS.items())
    first = long[long["draw_number"] == fourd_df["draw_number"].iloc[0]]
    assert list(first["tier"]) == ["first", "second", "third"] + ["starter"] * 10 + ["consolation"] * 10
    assert list(first["number"]) == [fourd_df.iloc[0][c] for c in FOURD_NUMBER_COLUMNS]
    assert long["number"].map(len).eq(4).all()
    assert long["draw_number"].is_monotonic_increasing


def test_all_numbers_skips_blanks_and_keeps_leading_zeros():
    df = make_df([(1, "2026-09-30", ["0042", "", "0007"] + filler(0, 10) + ["", "0100"])])
    long = A.all_numbers(df)
    assert list(long["number"][:2]) == ["0042", "0007"]
    assert list(long["tier"][:2]) == ["first", "third"]
    assert len(long) == 2 + 10 + 1
    assert long.iloc[-1]["tier"] == "consolation" and long.iloc[-1]["number"] == "0100"


def test_all_numbers_empty():
    empty = normalise_fourd(pd.DataFrame(columns=["draw_number"]))
    long = A.all_numbers(empty)
    assert long.empty and list(long.columns) == ["draw_number", "draw_date", "tier", "number"]
    assert A.position_digit_freq(empty).to_numpy().sum() == 0
    assert A.repeat_winners(empty) == []
    assert A.digit_set_freq(empty) == []


# Digit frequencies


def test_position_digit_freq_exact():
    df = make_df([(1, "2026-09-30", ["1234", "1299", "0004"]), (2, "2026-10-03", ["1230"])])
    freq = A.position_digit_freq(df)
    assert list(freq.columns) == list(C.FOURD_POSITIONS)
    assert list(freq.index) == list(range(10))
    assert freq["thousands"].tolist() == [1, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    assert freq.loc[2, "hundreds"] == 3 and freq.loc[0, "hundreds"] == 1
    assert freq.loc[3, "tens"] == 2 and freq.loc[9, "tens"] == 1 and freq.loc[0, "tens"] == 1
    assert freq.loc[4, "units"] == 2 and freq.loc[9, "units"] == 1 and freq.loc[0, "units"] == 1
    assert (freq.sum() == 4).all()


def test_position_digit_freq_matches_slow_count(fourd_df):
    freq = A.position_digit_freq(fourd_df)
    slow = {pos: Counter() for pos in C.FOURD_POSITIONS}
    for _, row in fourd_df.iterrows():
        for col in FOURD_NUMBER_COLUMNS:
            for pos, digit in zip(C.FOURD_POSITIONS, row[col]):
                slow[pos][int(digit)] += 1
    for pos in C.FOURD_POSITIONS:
        assert freq[pos].tolist() == [slow[pos][d] for d in range(10)]
    assert (freq.sum() == 23 * len(fourd_df)).all()
    assert freq.dtypes.map(lambda t: np.issubdtype(t, np.integer)).all()


# Repeat winners and digit sets


def repeat_df() -> pd.DataFrame:
    return make_df([
        (1, "2026-09-01", ["1234", "0042"] + filler(0, 21)),
        (2, "2026-09-05", ["5678", "1234"] + filler(30, 21)),
        (3, "2026-09-08", ["0042"] + filler(60, 22)),
        (4, "2026-09-12", ["1234", "5678"] + filler(90, 21)),
    ])


def test_repeat_winners_order_and_dates():
    out = A.repeat_winners(repeat_df())
    assert out == [
        ("1234", 3, pd.Timestamp("2026-09-12")),
        ("5678", 2, pd.Timestamp("2026-09-12")),
        ("0042", 2, pd.Timestamp("2026-09-08")),
    ]
    assert A.repeat_winners(repeat_df(), min_count=3) == [("1234", 3, pd.Timestamp("2026-09-12"))]
    assert A.repeat_winners(repeat_df(), k=1) == out[:1]
    assert isinstance(out[0][2], pd.Timestamp)


def test_repeat_winners_on_synthetic(fourd_df):
    out = A.repeat_winners(fourd_df, k=15)
    assert 0 < len(out) <= 15
    counts = Counter(A.all_numbers(fourd_df)["number"])
    for number, times, last in out:
        assert counts[number] == times >= 2
    assert [t for _, t, _ in out] == sorted((t for _, t, _ in out), reverse=True)
    assert out[0][1] == max(counts.values())


def test_digit_set_freq():
    df = make_df([
        (1, "2026-09-01", ["1234", "4321", "0042"]),
        (2, "2026-09-05", ["2143", "4200", "7777"]),
        (3, "2026-09-08", ["7777", "0420"]),
    ])
    assert A.digit_set_freq(df) == [("0024", 3), ("1234", 3), ("7777", 2)]
    assert A.digit_set_freq(df, k=1) == [("0024", 3)]


# Chi square


def test_chi_square_on_fair_data(fourd_df):
    chi = A.chi_square_digits(fourd_df)
    assert set(chi["per_position"]) == set(C.FOURD_POSITIONS)
    assert chi["overall"]["dof"] == 36
    freq = A.position_digit_freq(fourd_df)
    total = 0.0
    for pos, res in chi["per_position"].items():
        observed = freq[pos].to_numpy(dtype=float)
        expected = observed.sum() / 10
        manual = float(((observed - expected) ** 2 / expected).sum())
        assert res["stat"] == pytest.approx(manual)
        assert res["p_value"] == pytest.approx(stats.chi2.sf(manual, 9))
        assert res["dof"] == 9
        total += manual
    assert chi["overall"]["stat"] == pytest.approx(total)
    assert chi["overall"]["p_value"] == pytest.approx(stats.chi2.sf(total, 36))
    assert chi["n_numbers"] == 23 * len(fourd_df)
    # The synthetic draws are uniform, so the verdict is "noise" here.
    assert chi["overall"]["p_value"] > 0.05
    assert "consistent with pure chance" in chi["verdict"]
    assert no_dashes(chi["verdict"])


def test_chi_square_on_biased_data():
    rng = np.random.default_rng(0)
    draws = []
    for i in range(60):
        nums = [f"7{int(x):03d}" for x in rng.choice(1000, 23, replace=False)]  # thousands always 7
        draws.append((i + 1, f"2026-01-{(i % 28) + 1:02d}", nums))
    chi = A.chi_square_digits(make_df(draws))
    assert chi["overall"]["p_value"] < 0.0001
    assert chi["per_position"]["thousands"]["p_value"] < 0.0001
    assert "more uneven than chance" in chi["verdict"]
    assert "thousands position" in chi["verdict"]
    assert "p below 0.0001" in chi["verdict"]
    assert no_dashes(chi["verdict"]) and "e" not in A.format_p(1e-12).replace("below", "")


def test_chi_square_on_a_perfectly_even_spread_says_p_above():
    # Every position holds every digit equally often, so p is 1: "p above 0.99", not "p = above 0.99".
    draws, j = [], 0
    for i in range(10):
        nums = []
        for _ in range(23):
            d = j % 10
            nums.append(f"{d}{(d + 3) % 10}{(d + 7) % 10}{(d + 1) % 10}")
            j += 1
        draws.append((i + 1, f"2026-01-{i + 1:02d}", nums))
    chi = A.chi_square_digits(make_df(draws))
    assert chi["overall"]["p_value"] > 0.99
    assert "(p above 0.99)" in chi["verdict"]
    assert "p = above" not in chi["verdict"]


def test_chi_square_empty_and_tiny():
    empty = normalise_fourd(pd.DataFrame(columns=["draw_number"]))
    chi = A.chi_square_digits(empty)
    assert np.isnan(chi["overall"]["stat"]) and chi["overall"]["dof"] == 36
    assert "no 4D draws" in chi["verdict"]
    one = make_df([(1, "2026-09-30", filler(0))])
    chi = A.chi_square_digits(one)
    assert np.isfinite(chi["overall"]["p_value"])
    assert "too few" in chi["verdict"] and no_dashes(chi["verdict"])


@pytest.mark.parametrize("p,text", [(0.5, "0.50"), (0.012, "0.012"), (0.995, "above 0.99"),
                                    (0.00001, "below 0.0001"), (float("nan"), "n/a")])
def test_format_p(p, text):
    assert A.format_p(p) == text


# Bet type value


def test_bet_type_value_defaults_exact():
    v = A.bet_type_value(PrizeRules())
    assert v["Big"] == pytest.approx(0.659)
    assert v["Small"] == pytest.approx(0.58)
    big = {24: 0.6336, 12: 0.6468, 6: 0.654, 4: 0.6568}
    small = {24: 0.5784, 12: 0.5784, 6: 0.5796, 4: 0.58}
    assert set(v["iBet Big"]) == set(big) and set(v["iBet Small"]) == set(small)
    for perms in big:
        assert v["iBet Big"][perms] == pytest.approx(big[perms])
        assert v["iBet Small"][perms] == pytest.approx(small[perms])
    assert v["best"] == "Big"
    assert "Big gives the most back" in v["explanation"]
    assert "$0.66" in v["explanation"] and "$0.63 to $0.66" in v["explanation"]
    assert "less than the $1 it costs" in v["explanation"]
    assert "rounded down to whole dollars" in v["ibet_note"]
    assert "$83" in v["ibet_note"]
    assert no_dashes(v["explanation"]) and no_dashes(v["ibet_note"])


def test_bet_type_value_follows_rules():
    rules = PrizeRules()
    rules.fourd_prizes["small"]["first"] = 5000.0  # a made up table where Small wins
    v = A.bet_type_value(rules)
    assert v["Small"] == pytest.approx(0.78)
    assert v["best"] == "Small"
    assert v["iBet Small"][24] == pytest.approx(24 * (208 + 83 + 33) / 10_000)


def test_bet_type_value_ibet_published_tables():
    full = {bet: {p: {t: 1.0 for t in C.FOURD_PRIZES[bet]} for p in (4, 6, 12, 24)}
            for bet in ("big", "small")}
    full["big"][24] = {"first": 100.0, "second": 50.0, "third": 25.0, "starter": 12.0, "consolation": 3.0}
    v = A.bet_type_value(PrizeRules(ibet_prizes=full))
    assert v["ibet_note"] == "iBet prizes are taken from the official iBet prize table."
    assert v["iBet Big"][24] == pytest.approx(24 * (100 + 50 + 25 + 120 + 30) / 10_000)
    assert v["best"] == "iBet Big"  # 0.78 beats Big's 0.659 in this made up table
    assert "iBet Big gives the most back" in v["explanation"]

    partial = PrizeRules(ibet_prizes={"big": {24: {"first": 83.0}}, "small": {"24": {"first": 125.0}}})
    note = A.bet_type_value(partial)["ibet_note"]
    assert note.startswith("The iBet 24 Big and iBet 24 Small prizes are taken from the official")
    assert "rounded down" in note and no_dashes(note)
