"""Tests for huatbot.strategies: TOTO and 4D picks, pattern checks and the System 7 builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from huatbot import analysis_fourd as AF
from huatbot import analysis_toto as AT
from huatbot import strategies as S
from huatbot.models import FOURD_COLUMNS, TOTO_COLUMNS, TotoPick
from huatbot.store import normalise_fourd, normalise_toto
from huatbot.textfmt import contains_dash


@pytest.fixture(scope="module")
def scores(toto_df):
    from huatbot.models import PrizeRules

    s, diag = AT.crowd_scores(toto_df, PrizeRules())
    assert diag["n_draws"] >= 30
    return s


def _toto_frame(sets: list[list[int]], start: int = 4000) -> pd.DataFrame:
    """A small toto.csv frame with the given winning sets (oldest first)."""
    rows = []
    for i, nums in enumerate(sets):
        nums = sorted(nums)
        row = {c: np.nan for c in TOTO_COLUMNS}
        row.update({"draw_number": start + i, "draw_date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=3 * i),
                    "additional": next(n for n in range(1, 50) if n not in nums), "jackpot": 1e6,
                    "draw_type": "normal", "fetched_at": "test"})
        row.update({f"n{k + 1}": v for k, v in enumerate(nums)})
        for g in range(1, 8):
            row[f"g{g}_winners"] = 0
        rows.append(row)
    return normalise_toto(pd.DataFrame(rows))


def _fourd_frame(draws: list[list[str]], start: int = 5000) -> pd.DataFrame:
    """A small fourd.csv frame; each draw lists its numbers in column order (first, second, ...)."""
    cols = [c for c in FOURD_COLUMNS if c not in ("draw_number", "draw_date", "fetched_at")]
    rows = []
    for i, nums in enumerate(draws):
        row = {"draw_number": start + i, "draw_date": pd.Timestamp("2026-01-07") + pd.Timedelta(days=7 * i),
               "fetched_at": "test"}
        row.update({c: (nums[k] if k < len(nums) else "") for k, c in enumerate(cols)})
        rows.append(row)
    return normalise_fourd(pd.DataFrame(rows))


def _assert_pick(pick, name):
    assert pick.name == name
    assert len(pick.numbers) == 6 and len(set(pick.numbers)) == 6
    assert pick.numbers == sorted(pick.numbers)
    assert all(1 <= n <= 49 and isinstance(n, int) for n in pick.numbers)
    assert pick.reason and "\n" not in pick.reason and not contains_dash(pick.reason)


# Hot and Overdue


def test_hot_set_matches_last50_ranking(toto_df):
    pick = S.hot_set(toto_df)
    _assert_pick(pick, "Hot")
    table = AT.frequency_table(toto_df)
    ranked = sorted(table.index, key=lambda n: (-table.loc[n, "last50"], -table.loc[n, "last100"],
                                                -table.loc[n, "all"], n))
    assert pick.numbers == sorted(ranked[:6])
    counts = table.loc[pick.numbers, "last50"]
    assert f"Drawn {counts.min()} to {counts.max()} times each in the last 50 draws" in pick.reason
    assert f"average of {table['last50'].mean():.1f}" in pick.reason


def test_hot_set_tie_breaks():
    # 1 to 5 appear twice; 6, 7, 10, 20, ... once each in every window, so the sixth number is
    # decided by the last tie break: the lower number.
    sets = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [10, 20, 30, 40, 41, 42]]
    assert S.hot_set(_toto_frame(sets)).numbers == [1, 2, 3, 4, 5, 6]
    # 60 draws. In the last 50, numbers 1 to 5 come up every draw and 6 to 11 twice each, so
    # the sixth number is a tie at 2; 11 also came up in the 10 earlier draws, so it wins the
    # tie on the last 100 count instead of the lower number 6.
    early = [[11, 20, 21, 22, 23, 24]] * 10
    last50 = [[1, 2, 3, 4, 5, x] for x in list(range(6, 50)) + list(range(6, 12))]
    assert S.hot_set(_toto_frame(early + last50)).numbers == [1, 2, 3, 4, 5, 11]


def test_overdue_set_matches_gaps(toto_df):
    pick = S.overdue_set(toto_df)
    _assert_pick(pick, "Overdue")
    gaps, freq = AT.overdue(toto_df), AT.frequency(toto_df)
    ranked = sorted(gaps.index, key=lambda n: (-gaps[n], freq[n], n))
    assert pick.numbers == sorted(ranked[:6])
    g = gaps.loc[pick.numbers]
    assert f"Not drawn for {g.min()} to {g.max()} draws each" in pick.reason
    assert "8.2 draws" in pick.reason


def test_overdue_tie_break_lower_frequency_then_lower_number():
    # 12 to 15 were last seen in the first draw (longest gap). 10, 11 and 20 to 23 share the
    # next gap, but 10 and 11 were drawn twice, so 20 and 21 (drawn once) fill the last slots.
    rest = list(range(1, 10)) + list(range(16, 20)) + list(range(24, 50))
    rest += rest[: (-len(rest)) % 6]
    sets = [[10, 11, 12, 13, 14, 15], [10, 11, 20, 21, 22, 23]]
    sets += [rest[i:i + 6] for i in range(0, len(rest), 6)]
    hist = _toto_frame(sets)
    assert AT.overdue(hist).min() == 0 and (AT.frequency(hist) > 0).all()
    assert S.overdue_set(hist).numbers == [12, 13, 14, 15, 20, 21]


def test_strategies_use_only_the_given_history(toto_df):
    early = toto_df.iloc[:300]
    before = early.copy()
    assert S.hot_set(early).numbers == S.hot_set(early.copy()).numbers
    table = AT.frequency_table(early)
    ranked = sorted(table.index, key=lambda n: (-table.loc[n, "last50"], -table.loc[n, "last100"],
                                                -table.loc[n, "all"], n))
    assert S.hot_set(early).numbers == sorted(ranked[:6])
    pd.testing.assert_frame_equal(early, before)  # not mutated


def test_unsorted_history_is_sorted_first(toto_df):
    shuffled = toto_df.sample(frac=1.0, random_state=3)
    assert S.hot_set(shuffled).numbers == S.hot_set(toto_df).numbers
    assert S.overdue_set(shuffled).numbers == S.overdue_set(toto_df).numbers
    assert S.balanced_set(shuffled, 9).numbers == S.balanced_set(toto_df, 9).numbers


# Balanced


@pytest.mark.parametrize("seed", [0, 1, 4124, 99999])
def test_balanced_set_meets_shape(toto_df, seed):
    pick = S.balanced_set(toto_df, seed)
    _assert_pick(pick, "Balanced")
    shape = AT.shape_stats(toto_df)
    nums = pick.numbers
    assert sum(n % 2 for n in nums) == shape["typical_odd"]
    assert sum(n <= 24 for n in nums) == shape["typical_low"]
    assert shape["sum_q25"] <= sum(nums) <= shape["sum_q75"]
    assert "Matches the usual shape" in pick.reason and "relaxed" not in pick.reason
    assert f"sum {sum(nums)} inside the middle half range" in pick.reason


def test_balanced_set_deterministic(toto_df):
    a, b = S.balanced_set(toto_df, 7), S.balanced_set(toto_df, 7)
    assert a == b
    others = {tuple(S.balanced_set(toto_df, s).numbers) for s in range(8)}
    assert len(others) > 1  # the seed matters


def test_balanced_set_relaxes_when_shape_impossible():
    # One past draw, 1 2 3 4 5 7: the only set with its shape (4 odd, 6 low) and sum (22) is
    # that very set, which 20,000 random tries almost never meet, so the sum range is dropped.
    hist = _toto_frame([[1, 2, 3, 4, 5, 7]])
    pick = S.balanced_set(hist, 1)
    _assert_pick(pick, "Balanced")
    assert "Close to the usual shape" in pick.reason
    assert pick.reason.endswith("rules relaxed: sum range dropped")
    assert sum(n % 2 for n in pick.numbers) == 4 and all(n <= 24 for n in pick.numbers)


# Low Crowd


def test_low_crowd_set_rules(toto_df, scores):
    pick = S.low_crowd_set(toto_df, scores, 4124)
    _assert_pick(pick, "Low Crowd")
    nums = pick.numbers
    candidates = set(scores.sort_values(kind="stable").index[:24])
    assert set(nums) <= candidates
    last = [int(toto_df.iloc[-1][f"n{i}"]) for i in range(1, 7)]
    assert S.pattern_problems(nums, last) == []
    assert 2 <= sum(n % 2 for n in nums) <= 4
    assert 2 <= sum(n <= 24 for n in nums) <= 4
    shape = AT.shape_stats(toto_df)
    assert shape["sum_q25"] <= sum(nums) <= shape["sum_q75"]
    # The planted favourites (synth.DEFAULT_POPULAR) are avoided.
    assert not {7, 8, 9, 18, 28} & set(nums)
    total = float(scores.loc[nums].sum())
    assert total < 0
    assert "crowd score minus" in pick.reason and "fewer people" in pick.reason
    assert "relaxed" not in pick.reason


def test_low_crowd_set_is_least_crowded_of_valid_samples(toto_df, scores):
    pick = S.low_crowd_set(toto_df, scores, 11)
    total = float(scores.loc[pick.numbers].sum())
    # Any other valid set from the same candidates should rarely beat it; check a sample.
    rng = np.random.default_rng(0)
    candidates = np.array(sorted(scores.sort_values(kind="stable").index[:24]))
    last = [int(toto_df.iloc[-1][f"n{i}"]) for i in range(1, 7)]
    shape = AT.shape_stats(toto_df)
    beaten = 0
    for _ in range(300):
        nums = sorted(rng.choice(candidates, 6, replace=False).tolist())
        valid = (not S.pattern_problems(nums, last) and 2 <= sum(n % 2 for n in nums) <= 4
                 and 2 <= sum(n <= 24 for n in nums) <= 4
                 and shape["sum_q25"] <= sum(nums) <= shape["sum_q75"])
        if valid and float(scores.loc[nums].sum()) < total - 1e-12:
            beaten += 1
    assert beaten <= 3


@pytest.mark.parametrize("missing", [None, "zeros", "empty"])
def test_low_crowd_set_without_scores(toto_df, missing):
    crowd = {None: None, "zeros": pd.Series(0.0, index=range(1, 50)), "empty": pd.Series(dtype=float)}[missing]
    pick = S.low_crowd_set(toto_df, crowd, 5)
    _assert_pick(pick, "Low Crowd")
    assert "crowd scores not available, so this is a pattern free balanced set" in pick.reason.lower()
    last = [int(toto_df.iloc[-1][f"n{i}"]) for i in range(1, 7)]
    assert S.pattern_problems(pick.numbers, last) == []
    assert 2 <= sum(n % 2 for n in pick.numbers) <= 4


def test_low_crowd_set_differs_from_balanced_without_scores(toto_df):
    same = sum(S.low_crowd_set(toto_df, None, s).numbers == S.balanced_set(toto_df, s).numbers
               for s in range(10))
    assert same == 0


def test_low_crowd_set_deterministic_and_accepts_json_keys(toto_df, scores):
    a = S.low_crowd_set(toto_df, scores, 3)
    b = S.low_crowd_set(toto_df, {str(k): float(v) for k, v in scores.items()}, 3)
    assert a == b


def test_low_crowd_set_relaxes_when_candidates_cannot_qualify():
    # Only high numbers have negative scores, so the 24 candidates hold too few low numbers.
    scores = pd.Series(0.0, index=range(1, 50))
    scores.loc[26:49] = -np.linspace(0.01, 0.3, 24)
    hist = _toto_frame([[1, 9, 17, 25, 33, 41], [2, 10, 18, 26, 34, 42]] * 10)
    pick = S.low_crowd_set(hist, scores, 2)
    _assert_pick(pick, "Low Crowd")
    assert "rules relaxed: picked from all 49 numbers" in pick.reason


# pattern_problems and the vectorised rules


def test_pattern_problems_each_rule():
    assert S.pattern_problems([3, 12, 25, 33, 41, 46]) == []
    assert any("3 numbers in a row (7 8 9)" in p for p in S.pattern_problems([7, 8, 9, 20, 33, 44]))
    assert any("birthday" in p for p in S.pattern_problems([1, 5, 12, 19, 26, 31]))
    assert any("only 1 number above 31" in p for p in S.pattern_problems([1, 5, 12, 19, 26, 40]))
    probs = S.pattern_problems([3, 12, 25, 33, 41, 46], last_draw=[3, 12, 20, 21, 22, 23])
    assert probs == ["repeats 2 numbers from the last draw (3 12)"]
    assert S.pattern_problems([3, 12, 25, 33, 41, 46], last_draw=[3, 13, 20, 21, 22, 23]) == []
    assert any("evenly spaced numbers (every 8)" in p for p in S.pattern_problems([5, 13, 21, 29, 37, 45]))
    assert any("4 numbers end in 7" in p for p in S.pattern_problems([7, 17, 27, 37, 40, 44]))
    for p in S.pattern_problems([1, 2, 3, 4, 5, 6], last_draw=[1, 2, 3, 4, 5, 6]):
        assert not contains_dash(p)


def test_pattern_problems_accepts_a_row(toto_df):
    row = toto_df.iloc[-1]
    nums = [int(row[f"n{i}"]) for i in range(1, 7)]
    assert any("repeats 6 numbers" in p for p in S.pattern_problems(nums, row))


def test_vectorised_rules_agree_with_pattern_problems():
    rng = np.random.default_rng(42)
    last = [5, 6, 7, 30, 40, 45]
    sets = np.sort(np.array([rng.choice(np.arange(1, 50), 6, replace=False) for _ in range(3000)]), axis=1)
    # Add some deliberately patterned sets.
    extra = np.array([[7, 8, 9, 20, 33, 44], [5, 13, 21, 29, 37, 45], [7, 17, 27, 37, 40, 44],
                      [1, 5, 12, 19, 26, 31], [5, 6, 12, 33, 40, 48]])
    sets = np.vstack([sets, extra])
    ok = S._passes(sets, S._PATTERN_RULES, last, 0, 999, (0, 6), (0, 6))
    expected = np.array([not S.pattern_problems(row, last) for row in sets.tolist()])
    assert (ok == expected).all()
    assert not ok[-5:].any()


# System 7


def test_system7_from_adds_least_crowded_safe_number(toto_df, scores):
    pick = S.low_crowd_set(toto_df, scores, 4124)
    last = [int(toto_df.iloc[-1][f"n{i}"]) for i in range(1, 7)]
    seven = S.system7_from(pick, scores, last)
    assert len(seven) == 7 and seven == sorted(seven) and set(pick.numbers) <= set(seven)
    extra = (set(seven) - set(pick.numbers)).pop()
    assert S.pattern_problems(seven, last) == []
    for n in range(1, 50):
        if n in pick.numbers or n == extra:
            continue
        if not S.pattern_problems(pick.numbers + [n], last):
            assert (scores[n], n) > (scores[extra], extra)


def test_system7_from_without_scores_and_with_patterned_pick():
    pick = TotoPick("Hot", [1, 2, 3, 4, 5, 6], "test")
    seven = S.system7_from(pick, None)
    assert len(seven) == 7 and set(pick.numbers) <= set(seven)  # best effort, never fails
    clean = TotoPick("Low Crowd", [3, 12, 25, 33, 41, 46], "test")
    seven = S.system7_from(clean, None)
    assert S.pattern_problems(seven) == []
    assert seven == sorted([3, 12, 25, 33, 41, 46, 1])  # lowest number that stays pattern free
    with pytest.raises(ValueError):
        S.system7_from(TotoPick("x", [1, 2, 3], "bad"), None)


def test_toto_picks_order_and_reasons(toto_df, scores):
    picks = S.toto_picks(toto_df, scores, 4124)
    assert [p.name for p in picks] == ["Hot", "Overdue", "Balanced", "Low Crowd"]
    for p in picks:
        _assert_pick(p, p.name)
    assert picks == S.toto_picks(toto_df, scores, 4124)


def test_toto_picks_with_empty_or_tiny_history(toto_df):
    for hist in (toto_df.iloc[:0], toto_df.iloc[:1], toto_df.iloc[:5]):
        for p in S.toto_picks(hist, None, 1):
            _assert_pick(p, p.name)


# 4D


def test_fourd_strategy_numbers_match_analysis(fourd_df):
    out = S.fourd_strategy_numbers(fourd_df, 77)
    assert list(out) == list(S.FOURD_PICK_ORDER)
    for number, bet in out.values():
        assert len(number) == 4 and number.isdigit()
        assert bet in ("Big", "Small", "iBet Big", "iBet Small")

    freq = AF.position_digit_freq(fourd_df.tail(100))
    hot = "".join(str(int(freq[pos].idxmax())) for pos in freq.columns)
    cold = "".join(str(int(freq[pos].idxmin())) for pos in freq.columns)
    assert out["Hot Digits"] == (hot, "Big")
    assert out["Cold Digits"] == (cold, "Big")
    assert out["Repeat Winner"] == (AF.repeat_winners(fourd_df, 2, 1)[0][0], "Big")
    digit_set = AF.digit_set_freq(fourd_df, 1)[0][0]
    assert out["Digit Set"] == (digit_set, "iBet Big")
    assert out["Random"][1] == "Big"


def test_fourd_random_is_seeded(fourd_df):
    a = S.fourd_strategy_numbers(fourd_df, 5)["Random"]
    assert a == S.fourd_strategy_numbers(fourd_df, 5)["Random"]
    assert a[0] == f"{int(np.random.default_rng(5).integers(0, 10000)):04d}"
    assert len({S.fourd_strategy_numbers(fourd_df, s)["Random"] for s in range(10)}) > 1


def test_fourd_uses_only_given_history(fourd_df):
    early = fourd_df.iloc[:200]
    out = S.fourd_strategy_numbers(early, 1)
    assert out["Repeat Winner"][0] == AF.repeat_winners(early, 2, 1)[0][0]
    assert out["Digit Set"][0] == AF.digit_set_freq(early, 1)[0][0]


def test_fourd_picks_reasons(fourd_df):
    picks = S.fourd_picks(fourd_df, 3)
    assert [p.name for p in picks] == list(S.FOURD_PICK_ORDER)
    for p in picks:
        assert p.reason and "\n" not in p.reason and not contains_dash(p.reason)
    by = {p.name: p for p in picks}
    assert "over the last 100 draws" in by["Hot Digits"].reason
    assert "average of 230" in by["Hot Digits"].reason
    times = AF.repeat_winners(fourd_df, 2, 1)[0][1]
    assert f"Won {times} times in the last 500 draws" in by["Repeat Winner"].reason
    assert "iBet Big covers all 24 orders" in by["Digit Set"].reason
    assert "23 chances in 10,000" in by["Random"].reason


def test_fourd_digit_set_with_one_order_is_big_and_repeat_fallback():
    # 7777 wins in every draw (as different tiers), nothing else repeats.
    draws = [["7777", f"{1000 + i:04d}"] for i in range(3)]
    out = S.fourd_strategy_numbers(_fourd_frame(draws), 1)
    assert out["Digit Set"] == ("7777", "Big")
    assert out["Repeat Winner"] == ("7777", "Big")
    # No repeats at all: fall back to the latest 1st Prize.
    draws = [["0001", "0002"], ["0003", "0004"]]
    picks = {p.name: p for p in S.fourd_picks(_fourd_frame(draws), 1)}
    assert picks["Repeat Winner"].number == "0003"
    assert "latest 1st Prize" in picks["Repeat Winner"].reason


def test_fourd_repeat_winner_ties_go_to_most_recent():
    draws = [["1111", "2222"], ["2222", "1111"], ["3333", "1111"], ["2222", "4444"]]
    out = S.fourd_strategy_numbers(_fourd_frame(draws), 1)
    # 1111 and 2222 both won 3 times; 2222 won most recently.
    assert out["Repeat Winner"][0] == "2222"
    assert out["Repeat Winner"][0] == AF.repeat_winners(_fourd_frame(draws), 2, 1)[0][0]


def test_fourd_handles_blanks_and_raw_values(fourd_df):
    hist = fourd_df.iloc[:50].copy()
    hist.loc[hist.index[0], "first"] = ""
    hist["second"] = hist["second"].astype(object)
    hist.loc[hist.index[1], "second"] = 42  # int, should read as 0042
    vals = S._fourd_values(hist)
    assert vals[0, 0] == -1 and vals[1, 1] == 42
    assert S.fourd_picks(hist, 1)


def test_fourd_empty_history(fourd_df):
    picks = S.fourd_picks(fourd_df.iloc[:0], 1)
    assert [p.name for p in picks] == list(S.FOURD_PICK_ORDER)
    for p in picks:
        assert len(p.number) == 4 and not contains_dash(p.reason)

