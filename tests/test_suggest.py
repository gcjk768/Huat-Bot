"""Tests for huatbot.suggest: TOTO and 4D purchase plans always stay within the budget."""
from __future__ import annotations

import json

import pytest

from huatbot import analysis_fourd as AF
from huatbot import analysis_toto as AT
from huatbot import prizes
from huatbot import strategies as S
from huatbot import suggest as G
from huatbot.models import FourDPick, PrizeRules, TotoPick
from huatbot.textfmt import contains_dash, per_dollar

BUDGETS = [x / 2 for x in range(0, 101)]  # 0, 0.5, 1, ..., 50


@pytest.fixture(scope="module")
def scores(toto_df):
    return AT.crowd_scores(toto_df, PrizeRules())[0]


@pytest.fixture(scope="module")
def last_draw(toto_df):
    return [int(toto_df.iloc[-1][f"n{i}"]) for i in range(1, 7)]


@pytest.fixture(scope="module")
def picks(toto_df, scores):
    return S.toto_picks(toto_df, scores, 4124)


@pytest.fixture(scope="module")
def dpicks(fourd_df):
    return S.fourd_picks(fourd_df, 5001)


@pytest.fixture(scope="module")
def bet_values():
    return AF.bet_type_value(PrizeRules())


def _texts(plan):
    out = list(plan.notes) + [line.reason for line in plan.lines] + [line.numbers for line in plan.lines]
    if plan.alternative is not None:
        out += _texts(plan.alternative)
    return out


# TOTO


@pytest.mark.parametrize("offer", [True, False])
def test_toto_plan_never_exceeds_budget(picks, scores, last_draw, offer):
    for budget in BUDGETS:
        plan = G.toto_plan(picks, budget, scores, offer, last_draw)
        assert plan.game == "TOTO" and plan.budget == budget
        assert plan.total <= budget
        if plan.alternative is not None:
            assert plan.alternative.total <= budget
            assert plan.alternative.total > plan.total
        assert plan.notes
        for text in _texts(plan):
            assert not contains_dash(text), text
        assert len([ln for ln in plan.lines if ln.label != "System 7"]) <= 4
        if not offer:
            assert plan.alternative is None
            assert all(ln.label != "System 7" for ln in plan.lines)


def test_toto_plan_ten_dollars(picks, scores, last_draw):
    plan = G.toto_plan(picks, 10, scores, True, last_draw)
    assert [ln.label for ln in plan.lines] == list(G.TOTO_PRIORITY)
    assert all(ln.cost == 1.0 and ln.bet_type == "Ordinary" for ln in plan.lines)
    assert plan.total == 4.0
    alt = plan.alternative
    assert alt is not None and alt.total == 10.0
    assert alt.lines[0].label == "System 7" and alt.lines[0].cost == 7.0
    assert alt.lines[0].bet_type == "System 7"
    # The System 7 already holds the Low Crowd board, so the 3 sets are the other picks.
    assert [ln.label for ln in alt.lines[1:]] == ["Balanced", "Hot", "Overdue"]
    low = next(p for p in picks if p.name == "Low Crowd")
    seven = S.system7_from(low, scores, last_draw)
    assert alt.lines[0].numbers == " ".join(map(str, seven))
    assert any("alternative" in n and "$10" in n for n in plan.notes)
    assert any("$6 of the budget is left unspent" in n for n in plan.notes)
    assert any("13,983,816" in n for n in plan.notes)


def test_toto_plan_eleven_dollars_includes_system7(picks, scores, last_draw):
    plan = G.toto_plan(picks, 11, scores, True, last_draw)
    labels = [ln.label for ln in plan.lines]
    assert labels == list(G.TOTO_PRIORITY) + ["System 7"]
    assert plan.total == 11.0 and plan.alternative is None
    assert any("played twice" in n for n in plan.notes)
    big = G.toto_plan(picks, 30, scores, True, last_draw)
    assert big.total == 11.0
    assert any("$19 of the budget is left unspent" in n for n in big.notes)


@pytest.mark.parametrize("budget", [0, 0.5, 0.99])
def test_toto_plan_below_one_dollar(picks, scores, budget):
    plan = G.toto_plan(picks, budget, scores)
    assert plan.lines == [] and plan.total == 0 and plan.alternative is None
    assert plan.notes == ["Budget below $1, nothing to buy."]


def test_toto_plan_small_budgets(picks, scores):
    plan = G.toto_plan(picks, 3, scores)
    assert [ln.label for ln in plan.lines] == ["Low Crowd", "Balanced", "Hot"]
    assert plan.alternative is None
    assert any("A System 7 costs $7, more than the $3 budget" in n for n in plan.notes)
    half = G.toto_plan(picks, 3.5, scores)
    assert half.total == 3.0
    assert any("$0.50 of the budget is left unspent" in n for n in half.notes)
    seven = G.toto_plan(picks, 7, scores)
    assert seven.total == 4.0 and seven.alternative.total == 7.0
    assert [ln.label for ln in seven.alternative.lines] == ["System 7"]


def test_toto_plan_priority_order_and_unknown_names(picks):
    shuffled = list(reversed(picks)) + [TotoPick("Extra", [1, 12, 23, 34, 45, 46], "x")]
    plan = G.toto_plan(shuffled, 5, None, offer_system7=False)
    assert [ln.label for ln in plan.lines] == list(G.TOTO_PRIORITY)


def test_toto_plan_without_low_crowd_and_without_scores(picks):
    others = [p for p in picks if p.name != "Low Crowd"]
    plan = G.toto_plan(others, 10, None, True)
    assert [ln.label for ln in plan.lines] == ["Balanced", "Hot", "Overdue", "System 7"]
    assert plan.total == 10.0
    assert "crowd scores not available" in plan.lines[-1].reason
    assert plan.lines[-1].reason.startswith("The Balanced set plus")


def test_toto_plan_bad_inputs(picks):
    for bad in (None, float("nan"), -5, "lots", float("inf")):
        plan = G.toto_plan(picks, bad)
        assert plan.total == 0 and plan.budget == 0
    empty = G.toto_plan([], 10)
    assert empty.lines == [] and "nothing to buy" in empty.notes[0]


# 4D


def test_fourd_plan_never_exceeds_budget(dpicks, bet_values):
    for budget in BUDGETS:
        for values in (bet_values, None):
            plan = G.fourd_plan(dpicks, budget, values)
            assert plan.game == "4D" and plan.total <= budget
            assert all(float(ln.cost).is_integer() and ln.cost >= 1 for ln in plan.lines)
            assert len({ln.cost for ln in plan.lines}) <= 1  # same stake for every pick
            for text in _texts(plan):
                assert not contains_dash(text), text


def test_fourd_plan_five_dollars(dpicks, bet_values):
    plan = G.fourd_plan(dpicks, 5, bet_values)
    assert [ln.label for ln in plan.lines] == [p.name for p in dpicks]
    assert [ln.cost for ln in plan.lines] == [1.0] * 5
    assert plan.total == 5.0
    assert [ln.bet_type for ln in plan.lines] == [p.bet_type for p in dpicks]
    assert dict((ln.label, ln.bet_type) for ln in plan.lines)["Digit Set"] == "iBet Big"
    assert [ln.numbers for ln in plan.lines] == [p.number for p in dpicks]
    assert not any("unspent" in n for n in plan.notes)


def test_fourd_plan_twelve_dollars(dpicks, bet_values):
    plan = G.fourd_plan(dpicks, 12, bet_values)
    assert [ln.cost for ln in plan.lines] == [2.0] * 5
    assert plan.total == 10.0
    assert any("$2 is unspent" in n for n in plan.notes)


def test_fourd_plan_three_dollars(dpicks, bet_values):
    plan = G.fourd_plan(dpicks, 3, bet_values)
    assert [ln.label for ln in plan.lines] == [p.name for p in dpicks[:3]]
    assert [ln.cost for ln in plan.lines] == [1.0] * 3
    assert any("first 3 of the 5 picks" in n for n in plan.notes)


@pytest.mark.parametrize("budget", [0, 0.5])
def test_fourd_plan_below_one_dollar(dpicks, budget):
    plan = G.fourd_plan(dpicks, budget)
    assert plan.lines == [] and plan.notes == ["Budget below $1, nothing to buy."]


def test_fourd_plan_bet_notes_use_bet_values(dpicks, bet_values):
    plan = G.fourd_plan(dpicks, 5, bet_values)
    big_note = next(n for n in plan.notes if n.startswith("Big returns the most per $1"))
    assert per_dollar(bet_values["Big"]) in big_note and per_dollar(bet_values["Small"]) in big_note
    digit = next(p for p in dpicks if p.name == "Digit Set")
    perms = prizes.permutations_count(digit.number)
    ibet_note = next(n for n in plan.notes if "iBet Big bet" in n)
    assert f"all {perms} orders" in ibet_note
    assert per_dollar(bet_values["iBet Big"][perms]) in ibet_note

    # Custom values flow through; JSON round trip (string keys) works too.
    custom = json.loads(json.dumps({**bet_values, "Big": 0.7, "iBet Big": {str(k): 0.5 for k in (4, 6, 12, 24)}}))
    notes = G.fourd_plan(dpicks, 5, custom).notes
    assert any("$0.70 on average" in n for n in notes)
    assert any("returning $0.50 per $1" in n for n in notes)

    # Without bet_values the built in prize table is used and the note says so.
    notes = G.fourd_plan(dpicks, 5).notes
    assert any("built in prize table" in n for n in notes)


def test_fourd_plan_other_best_and_small_picks(bet_values):
    picks = [FourDPick("A", "1234", "Small", "r"), FourDPick("B", "5678", "Big", "r")]
    plan = G.fourd_plan(picks, 4, {**bet_values, "best": "iBet Big"})
    assert [ln.cost for ln in plan.lines] == [2.0, 2.0]
    assert [ln.bet_type for ln in plan.lines] == ["Small", "Big"]
    assert any(n.startswith("iBet Big returns the most per $1") for n in plan.notes)
    plan = G.fourd_plan(picks, 2, bet_values)
    assert not any("so the straight picks are Big bets" in n for n in plan.notes)


def test_fourd_plan_no_picks():
    plan = G.fourd_plan([], 5)
    assert plan.lines == [] and "nothing to buy" in plan.notes[0]
