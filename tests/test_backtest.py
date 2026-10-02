"""Tests for huatbot.backtest: no look ahead, real prize scoring, random baseline, verdicts, JSON.

Real strategy runs are kept tiny (a handful of draws). Tests that need many draws swap the
strategies for instant stand ins, because the random players do not depend on them.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from huatbot import analysis_toto as AT
from huatbot import backtest as BT
from huatbot import prizes
from huatbot import strategies as S
from huatbot.models import BacktestResult, StrategyScore, TotoPick
from huatbot.textfmt import contains_dash

TOTO_NAMES = ["Hot", "Overdue", "Balanced", "Low Crowd"]
FOURD_NAMES = ["Hot Digits", "Repeat Winner", "Digit Set", "Cold Digits"]


# Helpers


def _winning(df, draw_no):
    row = df.loc[df["draw_number"] == draw_no].iloc[0]
    return row, prizes.toto_winning(row)


def _losing_set(winning, additional):
    """Six numbers that share nothing with the draw, so the set wins nothing."""
    taken = set(winning) | {additional}
    return [n for n in range(1, 50) if n not in taken][:6]


def _patch_toto(monkeypatch, df, hot=None, overdue=None, calls=None):
    """Replace the four TOTO strategies with instant stand ins.

    ``hot`` / ``overdue`` map the target draw's row to a set; the rest return fixed sets. Every
    call records (strategy, newest draw number seen, seed) in ``calls``.
    """
    calls = calls if calls is not None else []
    numbers = df["draw_number"].to_numpy()

    def target(hist):
        newest = int(hist["draw_number"].max())
        nxt = numbers[numbers > newest].min()  # the draw being picked for (synthetic draws are consecutive)
        return newest, df.loc[df["draw_number"] == nxt].iloc[0]

    def make(name, chooser, fixed):
        def fn(hist, *args):
            newest, row = target(hist)
            calls.append((name, newest, args[-1] if args else None, len(hist)))
            nums = chooser(row) if chooser else fixed
            return TotoPick(name=name, numbers=sorted(nums), reason="test")
        return fn

    monkeypatch.setattr(S, "hot_set", make("Hot", hot, [1, 2, 3, 4, 5, 6]))
    monkeypatch.setattr(S, "overdue_set", make("Overdue", overdue, [7, 8, 9, 10, 11, 12]))
    monkeypatch.setattr(S, "balanced_set", make("Balanced", None, [3, 14, 25, 31, 40, 47]))
    monkeypatch.setattr(S, "low_crowd_set", make("Low Crowd", None, [5, 16, 23, 34, 41, 48]))
    return calls


@pytest.fixture(scope="module")
def real_toto(toto_df):
    """A tiny backtest with the real strategies (kept small: about 45 ms per draw)."""
    from huatbot.models import PrizeRules

    return BT.backtest_toto(toto_df, PrizeRules(), n_draws=6, n_random=200, seed=3)


@pytest.fixture(scope="module")
def real_fourd(fourd_df):
    from huatbot.models import PrizeRules

    return BT.backtest_fourd(fourd_df, PrizeRules(), n_draws=12, n_random=300, seed=3)


# verdict_for and the percentile


@pytest.mark.parametrize("p, expected", [
    (50.0, "No better than random (percentile rank 50 among random players)"),
    (5.0, "No better than random (percentile rank 5 among random players)"),
    (95.0, "No better than random (percentile rank 95 among random players)"),
    (43.4, "No better than random (percentile rank 43 among random players)"),
    (97.3, "Beat random in this sample (top 3%), not expected to last"),
    (99.99, "Beat random in this sample (top 1%), not expected to last"),
    (100.0, "Beat random in this sample (top 1%), not expected to last"),
    (4.9, "Worse than random in this sample"),
    (0.0, "Worse than random in this sample"),
])
def test_verdict_for_bands(p, expected):
    assert BT.verdict_for(p) == expected
    assert not contains_dash(BT.verdict_for(p))
    # The odds note says once that every draw is independent; verdicts never repeat it.
    assert "independent" not in BT.verdict_for(p)


@pytest.mark.parametrize("p, beat, tied, expected", [
    (43.4, 43.4, 0.0, "No better than random (beat 43% of random players)"),
    (40.0, 30.0, 20.0, "No better than random (beat 30% of random players, tied with 20%)"),
    (27.0, 0.0, 54.0, "No better than random (beat 0% of random players, tied with 54%)"),
    (99.0, 98.0, 2.0, "Beat random in this sample (top 1%), not expected to last"),
])
def test_verdict_for_counts_ties_separately(p, beat, tied, expected):
    assert BT.verdict_for(p, beat, tied) == expected


def test_zero_winnings_strategy_does_not_claim_to_beat_tied_players():
    """About half of random 4D players win nothing; a $0 strategy beat none of them."""
    totals = [0.0] * 54 + [5.0] * 20 + [100.0] * 26
    score = BT._strategy_score("Repeat Winner", np.zeros(300), np.array(totals))
    assert score.percentile_vs_random == 27.0  # mid rank still sets the band
    assert score.verdict == "No better than random (beat 0% of random players, tied with 54%)"
    assert BT.shares_vs_random(5.0, totals) == pytest.approx((54.0, 20.0))
    assert BT.shares_vs_random(5.0, []) is None


@pytest.mark.parametrize("p", [None, float("nan"), "junk"])
def test_verdict_for_missing(p):
    assert BT.verdict_for(p) == "Not compared with random players"


def test_percentile_is_mid_rank():
    totals = [0, 0, 10, 10, 20, 30]
    assert BT.percentile_vs_random(10, totals) == pytest.approx(100 * (2 + 0.5 * 2) / 6)
    assert BT.percentile_vs_random(0, totals) == pytest.approx(100 * (0.5 * 2) / 6)
    assert BT.percentile_vs_random(100, totals) == 100.0
    assert BT.percentile_vs_random(-1, totals) == 0.0
    assert BT.percentile_vs_random(5, [5.0] * 10) == 50.0  # ties with everyone sit in the middle
    assert BT.percentile_vs_random(5, []) is None
    # Float sums that differ only by rounding noise still count as ties.
    assert BT.percentile_vs_random(0.1 + 0.2, [0.3]) == 50.0


# TOTO: no look ahead


def test_toto_strategies_and_crowd_scores_see_only_earlier_draws(monkeypatch, toto_df, rules):
    calls = _patch_toto(monkeypatch, toto_df)
    crowd_calls = []
    real_crowd_scores = AT.crowd_scores

    def spy_crowd(df, rules_, alpha=None, table=None):
        crowd_calls.append((int(df["draw_number"].max()), int(table.index.max()), len(df)))
        return real_crowd_scores(df, rules_, alpha=alpha, table=table)

    monkeypatch.setattr(AT, "crowd_scores", spy_crowd)
    n = 15
    result = BT.backtest_toto(toto_df, rules, n_draws=n, n_random=50)
    tested = toto_df["draw_number"].to_numpy()[-n:]
    assert result.first_draw == tested[0] and result.last_draw == tested[-1]

    for name in TOTO_NAMES:
        mine = [c for c in calls if c[0] == name]
        assert len(mine) == n
        for (_, newest, seed, size), draw_no in zip(mine, tested):
            assert newest < draw_no  # nothing from the tested draw or later
            assert newest == draw_no - 1  # but everything before it
            assert size == len(toto_df) - n + list(tested).index(draw_no)
            if name in ("Balanced", "Low Crowd"):
                assert seed == draw_no  # seeded like the live bot (seed = draw being picked for)

    assert len(crowd_calls) == n
    for (newest_df, newest_table, _), draw_no in zip(crowd_calls, tested):
        assert newest_df < draw_no and newest_table < draw_no


def test_toto_refit_every(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    count = []
    real_crowd_scores = AT.crowd_scores
    monkeypatch.setattr(AT, "crowd_scores", lambda *a, **k: count.append(1) or real_crowd_scores(*a, **k))
    result = BT.backtest_toto(toto_df, rules, n_draws=10, n_random=10, refit_every=4)
    assert len(count) == 3  # test draws 0, 4 and 8
    assert any("refitted every 4 draws" in note for note in result.notes)


def test_toto_matches_strategies_run_on_history_alone(real_toto, toto_df, rules):
    """Rebuild each pick from df.iloc[:i] alone (own crowd table) and score it by hand."""
    n = real_toto.draws_tested
    expected = {name: [] for name in TOTO_NAMES}
    for i in range(len(toto_df) - n, len(toto_df)):
        hist, row = toto_df.iloc[:i], toto_df.iloc[i]
        scores, _ = AT.crowd_scores(hist, rules)  # crowd table built from the history only
        picks = S.toto_picks(hist, scores, seed=int(row["draw_number"]))
        won = prizes.toto_prize_matrix(np.array([p.numbers for p in picks]), row, rules)
        for pick, amount in zip(picks, won):
            expected[pick.name].append(amount)
    by_name = {s.name: s for s in real_toto.scores}
    for name in TOTO_NAMES:
        assert by_name[name].winnings == pytest.approx(sum(expected[name]), abs=0.01)
        assert by_name[name].wins == sum(a > 0 for a in expected[name])
        assert by_name[name].best_prize == pytest.approx(max(expected[name]), abs=0.01)


# TOTO: scoring, baseline, notes


def test_toto_result_shape(real_toto, toto_df):
    r = real_toto
    assert isinstance(r, BacktestResult) and r.game == "toto"
    assert r.draws_tested == 6 and r.random_sets_per_draw == 200
    assert r.last_draw == int(toto_df["draw_number"].iloc[-1])
    assert r.first_draw == int(toto_df["draw_number"].iloc[-6])
    assert [s.name for s in r.scores] == TOTO_NAMES + ["Random"]
    for s in r.scores:
        assert s.draws == 6 and s.cost == 6.0
        assert s.winnings >= 0 and 0 <= s.wins <= 6
        assert not contains_dash(s.verdict)
    for s in r.scores[:-1]:
        assert 0 <= s.percentile_vs_random <= 100
        band = BT.verdict_for(s.percentile_vs_random).split(" (")[0]
        assert s.verdict.split(" (")[0] == band  # the mid rank picks the band
        if band == "No better than random":
            assert s.verdict.startswith("No better than random (beat ")  # strict share, ties apart
    rnd = r.scores[-1]
    assert rnd.percentile_vs_random is None and rnd.verdict == "Baseline"
    assert r.notes and not any(contains_dash(n) for n in r.notes)
    assert f"draw {r.first_draw} to {r.last_draw}" in r.notes[0]
    assert any("200 random players" in n for n in r.notes)


def test_toto_cheating_strategy_beats_random_and_loser_is_worse(monkeypatch, toto_df, rules):
    """A stand in that knows the result must top the random players; one that never matches
    anything must sit below almost all of them. Also checks the real share amounts are used."""
    hot = lambda row: prizes.toto_winning(row)[0]  # noqa: E731, look ahead on purpose
    overdue = lambda row: _losing_set(*prizes.toto_winning(row))  # noqa: E731
    _patch_toto(monkeypatch, toto_df, hot=hot, overdue=overdue)
    n = 200
    r = BT.backtest_toto(toto_df, rules, n_draws=n, n_random=300, seed=1)
    by_name = {s.name: s for s in r.scores}

    expected_hot = sum(prizes.toto_share_if_won(1, toto_df.iloc[i], rules)
                       for i in range(len(toto_df) - n, len(toto_df)))
    assert by_name["Hot"].winnings == pytest.approx(expected_hot, abs=0.01)
    assert by_name["Hot"].wins == n
    assert by_name["Hot"].percentile_vs_random == 100.0
    assert by_name["Hot"].verdict.startswith("Beat random in this sample (top 1%)")

    assert by_name["Overdue"].winnings == 0 and by_name["Overdue"].wins == 0
    assert by_name["Overdue"].percentile_vs_random < 5
    assert by_name["Overdue"].verdict == "Worse than random in this sample"


def test_toto_random_baseline_is_average_player(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    n, k = 40, 120
    r = BT.backtest_toto(toto_df, rules, n_draws=n, n_random=k, seed=5)

    # Rebuild the random players: the backtest draws its random sets from default_rng(seed)
    # in draw order, and player j plays set j of every draw.
    rng = np.random.default_rng(5)
    won = np.zeros((n, k))
    for t, i in enumerate(range(len(toto_df) - n, len(toto_df))):
        won[t] = prizes.toto_prize_matrix(BT._random_toto_sets(rng, k), toto_df.iloc[i], rules)
    totals = won.sum(axis=0)
    rnd = r.scores[-1]
    assert rnd.name == "Random" and rnd.cost == n
    assert rnd.winnings == pytest.approx(totals.mean(), abs=0.01)
    assert rnd.wins == round(float((won > 0).sum(axis=0).mean()))
    # One statistic for the whole row: the best prize is the mean of each player's best too.
    assert rnd.best_prize == pytest.approx(won.max(axis=0).mean(), abs=0.01)
    for s in r.scores[:-1]:
        assert s.percentile_vs_random == pytest.approx(
            round(BT.percentile_vs_random(s.winnings, totals), 1), abs=0.051)


def test_toto_random_return_is_plausible(monkeypatch, toto_df, rules):
    """Sanity: the average random player gets back roughly 20 to 60 cents per $1.

    Groups 5 to 7 alone are worth about 24 cents per board and Groups 2 to 4 add a few cents.
    The bound is loose but not unconditional: one Group 1 or Group 2 hit among the random players
    lifts the average far above it (that is what the skew note explains), so this deterministic
    sample (30,000 boards, seed 0) is one without such a hit.
    """
    _patch_toto(monkeypatch, toto_df)
    r = BT.backtest_toto(toto_df, rules, n_draws=100, n_random=300, seed=0)
    rnd = r.scores[-1]
    assert 0.2 <= rnd.return_per_dollar <= 0.6


def test_toto_random_sets_are_valid_and_uniform():
    rng = np.random.default_rng(0)
    sets = BT._random_toto_sets(rng, 20_000)
    assert sets.shape == (20_000, 6)
    assert sets.min() == 1 and sets.max() == 49
    assert all(len(set(row)) == 6 for row in sets[:500])
    counts = np.bincount(sets.ravel(), minlength=50)[1:]
    expected = 20_000 * 6 / 49
    assert np.abs(counts - expected).max() < 0.1 * expected
    assert BT._random_toto_sets(rng, 0).shape == (0, 6)


def test_toto_deterministic_for_seed(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    a = BT.backtest_toto(toto_df, rules, n_draws=30, n_random=100, seed=9)
    b = BT.backtest_toto(toto_df, rules, n_draws=30, n_random=100, seed=9)
    c = BT.backtest_toto(toto_df, rules, n_draws=30, n_random=100, seed=10)
    assert a == b
    assert a != c  # the seed drives the random players


def test_toto_unsorted_input_same_result(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    shuffled = toto_df.sample(frac=1.0, random_state=1)
    a = BT.backtest_toto(toto_df, rules, n_draws=20, n_random=50, seed=2)
    b = BT.backtest_toto(shuffled, rules, n_draws=20, n_random=50, seed=2)
    assert a == b


# Short history and edge cases


def test_short_history_reduces_window(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    df = toto_df.iloc[:130]
    r = BT.backtest_toto(df, rules, n_draws=300, n_random=20, min_history=100)
    assert r.draws_tested == 30
    assert all(s.draws == 30 for s in r.scores)
    assert any("instead of 300" in n and "30 draws" in n for n in r.notes)
    assert not any(contains_dash(n) for n in r.notes)


def test_too_little_history_returns_empty_result(toto_df, fourd_df, rules):
    r = BT.backtest_toto(toto_df.iloc[:50], rules, n_draws=300, n_random=20, min_history=100)
    assert r.draws_tested == 0 and r.scores == [] and r.first_draw is None
    assert r.notes and "Not enough history" in r.notes[0] and not contains_dash(r.notes[0])
    f = BT.backtest_fourd(fourd_df.iloc[:10], rules, n_draws=300, n_random=20, min_history=100)
    assert f.game == "4d" and f.draws_tested == 0 and f.scores == []


def test_no_random_players(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    r = BT.backtest_toto(toto_df, rules, n_draws=10, n_random=0)
    assert [s.name for s in r.scores] == TOTO_NAMES  # no Random row without random players
    assert all(s.percentile_vs_random is None for s in r.scores)
    assert all(s.verdict == "Not compared with random players" for s in r.scores)
    assert any("No random players" in n for n in r.notes)


def test_crowd_scores_thin_history_note(monkeypatch, toto_df, rules):
    _patch_toto(monkeypatch, toto_df)
    r = BT.backtest_toto(toto_df.iloc[:30], rules, n_draws=10, n_random=10, min_history=5)
    assert r.draws_tested == 10
    assert any("too little usable history" in n for n in r.notes)


# 4D


def _patch_fourd(monkeypatch, df, chooser, calls=None):
    calls = calls if calls is not None else []
    numbers = df["draw_number"].to_numpy()

    def fake(hist, seed):
        newest = int(hist["draw_number"].max())
        calls.append((newest, seed, len(hist)))
        row = df.loc[df["draw_number"] == numbers[numbers > newest].min()].iloc[0]
        return chooser(row)

    monkeypatch.setattr(S, "fourd_strategy_numbers", fake)
    return calls


def test_fourd_strategies_see_only_earlier_draws(monkeypatch, fourd_df, rules):
    fixed = {"Hot Digits": ("1234", "Big"), "Repeat Winner": ("0042", "Big"),
             "Digit Set": ("1347", "iBet Big"), "Cold Digits": ("9876", "Big"), "Random": ("5555", "Big")}
    calls = _patch_fourd(monkeypatch, fourd_df, lambda row: fixed)
    n = 20
    r = BT.backtest_fourd(fourd_df, rules, n_draws=n, n_random=10)
    tested = fourd_df["draw_number"].to_numpy()[-n:]
    assert len(calls) == n
    for (newest, seed, size), draw_no in zip(calls, tested):
        assert newest < draw_no and newest == draw_no - 1
        assert seed == draw_no
        assert size == len(fourd_df) - n + list(tested).index(draw_no)
    assert (r.first_draw, r.last_draw) == (tested[0], tested[-1])


def test_fourd_scoring_uses_real_prizes(monkeypatch, fourd_df, rules):
    """Big, iBet Big and Small picks are scored exactly like real tickets of that draw."""
    def chooser(row):
        first = row["first"]
        scrambled = first[::-1]  # a permutation of the 1st Prize (iBet pays for any order)
        return {"Hot Digits": (first, "Big"), "Repeat Winner": (row["starter_1"], "Big"),
                "Digit Set": (scrambled, "iBet Big"), "Cold Digits": (row["second"], "Small"),
                "Random": ("0000", "Big")}

    _patch_fourd(monkeypatch, fourd_df, chooser)
    n = 25
    r = BT.backtest_fourd(fourd_df, rules, n_draws=n, n_random=200, seed=4)
    by_name = {s.name: s for s in r.scores}
    assert [s.name for s in r.scores] == FOURD_NAMES + ["Random"]

    expected = {name: 0.0 for name in FOURD_NAMES}
    for i in range(len(fourd_df) - n, len(fourd_df)):
        row = fourd_df.iloc[i]
        for name, (number, bet) in chooser(row).items():
            if name in expected:
                expected[name] += prizes.fourd_ticket_prize(number, bet, 1.0, row, rules).amount
    for name in FOURD_NAMES:
        assert by_name[name].winnings == pytest.approx(expected[name], abs=0.01)
        assert by_name[name].cost == n
    assert by_name["Hot Digits"].winnings >= 2000 * n  # 1st Prize Big every draw
    assert by_name["Cold Digits"].winnings >= 2000 * n  # 2nd Prize Small every draw
    assert by_name["Hot Digits"].verdict.startswith("Beat random in this sample")
    assert any("Digit Set iBet Big" in note for note in r.notes)


def test_fourd_random_big_return_near_0659(monkeypatch, fourd_df, rules):
    """Sanity: random Big $1 numbers return about $0.659 per $1 on average (noisy, loose bounds).

    400 draws x 1,000 players = 400,000 bets; the standard error is about 0.04.
    """
    fixed = {"Hot Digits": ("1234", "Big"), "Repeat Winner": ("0042", "Big"),
             "Digit Set": ("1347", "iBet Big"), "Cold Digits": ("9876", "Big"), "Random": ("5555", "Big")}
    _patch_fourd(monkeypatch, fourd_df, lambda row: fixed)
    r = BT.backtest_fourd(fourd_df, rules, n_draws=400, n_random=1000, seed=0)
    rnd = r.scores[-1]
    assert rnd.name == "Random" and rnd.verdict == "Baseline" and rnd.percentile_vs_random is None
    assert 0.5 <= rnd.return_per_dollar <= 0.82
    # The row is one average player: a prize draw comes with a best prize above $0.
    assert rnd.wins >= 1 and rnd.best_prize > 0
    assert any("mean over those players" in n and "tied with" in n for n in r.notes)


def test_fourd_real_run_shape(real_fourd, fourd_df):
    r = real_fourd
    assert r.game == "4d" and r.draws_tested == 12 and r.random_sets_per_draw == 300
    assert [s.name for s in r.scores] == FOURD_NAMES + ["Random"]
    assert r.last_draw == int(fourd_df["draw_number"].iloc[-1])
    for s in r.scores:
        assert s.cost == 12.0 and s.winnings >= 0 and not contains_dash(s.verdict)
    assert all(s.percentile_vs_random is not None for s in r.scores[:-1])
    assert r.notes and not any(contains_dash(n) for n in r.notes)


def test_fourd_matches_strategies_run_on_history_alone(real_fourd, fourd_df, rules):
    n = real_fourd.draws_tested
    expected = {name: 0.0 for name in FOURD_NAMES}
    for i in range(len(fourd_df) - n, len(fourd_df)):
        hist, row = fourd_df.iloc[:i], fourd_df.iloc[i]
        picks = S.fourd_strategy_numbers(hist, seed=int(row["draw_number"]))
        for name in FOURD_NAMES:
            number, bet = picks[name]
            expected[name] += prizes.fourd_ticket_prize(number, bet, 1.0, row, rules).amount
    by_name = {s.name: s for s in real_fourd.scores}
    for name in FOURD_NAMES:
        assert by_name[name].winnings == pytest.approx(expected[name], abs=0.01)


# JSON cache


def test_round_trip_through_json(real_toto, real_fourd):
    for r in (real_toto, real_fourd):
        d = BT.result_to_dict(r)
        text = json.dumps(d, allow_nan=False)
        back = BT.result_from_dict(json.loads(text))
        assert back == r
        assert [s.return_per_dollar for s in back.scores] == [s.return_per_dollar for s in r.scores]


def test_round_trip_handles_numpy_and_missing_values():
    r = BacktestResult(
        game="toto", draws_tested=np.int64(3), first_draw=np.int64(10), last_draw=None,
        random_sets_per_draw=5,
        scores=[StrategyScore("Hot", np.int64(3), np.float64(3.0), np.float64(20.0), np.int64(2),
                              np.float64(10.0), np.float64(55.5), "No better than random"),
                StrategyScore("Random", 3, 3.0, 1.5, 0, 0.0, float("nan"), "Baseline")],
        notes=["a note"],
    )
    d = json.loads(json.dumps(BT.result_to_dict(r), allow_nan=False))
    assert d["scores"][1]["percentile_vs_random"] is None
    assert d["scores"][0]["return_per_dollar"] == pytest.approx(20.0 / 3)
    back = BT.result_from_dict(d)
    assert back.first_draw == 10 and back.last_draw is None
    assert back.scores[0].percentile_vs_random == 55.5
    assert back.scores[1].percentile_vs_random is None
    assert isinstance(back.scores[0].draws, int) and isinstance(back.scores[0].cost, float)


def test_from_dict_rejects_other_format():
    with pytest.raises(ValueError):
        BT.result_from_dict({"format": 999, "game": "toto"})
    empty = BT.result_from_dict({"game": "4d"})
    assert empty.game == "4d" and empty.scores == [] and empty.draws_tested == 0


def test_strategy_names_follow_pick_order():
    assert list(BT.TOTO_STRATEGIES) == list(S.TOTO_PICK_ORDER)
    assert list(BT.FOURD_STRATEGIES) == [n for n in S.FOURD_PICK_ORDER if n != "Random"]
    assert math.isclose(BT.STAKE, 1.0)


def test_skew_note_explains_lopsided_average():
    won = np.zeros((10, 100))
    won[0, :] = 10.0  # every player wins $10 once
    assert BT._skew_note(won) is None  # mean equals median
    won[3, 7] = 1_000_000.0  # one player hits Group 1
    note = BT._skew_note(won)
    assert note is not None and "$1,000,000" in note
    assert "middle random player won $10, $1.00 per $1" in note
    assert not contains_dash(note)
    assert BT._skew_note(np.zeros((5, 0))) is None
