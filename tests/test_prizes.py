"""Tests for huatbot.prizes: TOTO groups, system bets, the pool estimate and ticket prizes."""
from __future__ import annotations

import re
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import prizes as P
from huatbot.models import PrizeRules

WIN = [3, 11, 19, 27, 38, 45]
ADD = 8
NON_WIN = [n for n in range(1, 50) if n not in WIN and n != ADD]  # the 42 other numbers
DASHES = ("-", "–", "—")


def toto_row(**overrides) -> dict:
    """A plausible toto.csv row with every group won, overridable per field."""
    row = {
        "draw_number": 4123,
        "draw_date": pd.Timestamp("2026-10-01"),
        **{f"n{i + 1}": n for i, n in enumerate(WIN)},
        "additional": ADD,
        "jackpot": 2_400_000.0,
        "g1_share": 1_200_000.0, "g1_winners": 2,
        "g2_share": 90_000.0, "g2_winners": 3,
        "g3_share": 2_000.0, "g3_winners": 110,
        "g4_share": 400.0, "g4_winners": 300,
        "g5_share": 50.0, "g5_winners": 6_000,
        "g6_share": 25.0, "g6_winners": 8_000,
        "g7_share": 10.0, "g7_winners": 120_000,
        "draw_type": "normal",
        "fetched_at": "test",
    }
    row.update(overrides)
    return row


def board_with(matched: int, with_additional: bool) -> list[int]:
    """A board holding ``matched`` winning numbers, the additional number or not, rest filler."""
    board = WIN[:matched] + ([ADD] if with_additional else [])
    return board + NON_WIN[: 6 - len(board)]


# toto_group


EXPECTED_GROUP = {(6, False): 1, (5, True): 2, (5, False): 3, (4, True): 4, (4, False): 5,
                  (3, True): 6, (3, False): 7}


# Every (matched, additional) pattern a 6 number board can have (6 + additional cannot fit).
PATTERNS = [(m, a) for m in range(0, 7) for a in (False, True) if m + a <= 6]


@pytest.mark.parametrize("matched,with_additional", PATTERNS)
def test_toto_group_every_match_pattern(matched, with_additional):
    board = board_with(matched, with_additional)
    assert len(set(board)) == 6
    assert P.toto_group(board, WIN, ADD) == EXPECTED_GROUP.get((matched, with_additional))


def test_toto_group_exhaustive_over_all_intersections():
    """Every subset of the 7 drawn numbers that fits on a board, in every board order."""
    special = WIN + [ADD]
    rng = np.random.default_rng(1)
    for size in range(0, 7):
        for subset in combinations(special, size):
            board = list(subset) + NON_WIN[: 6 - size]
            rng.shuffle(board)
            m = len(set(subset) & set(WIN))
            a = ADD in subset
            expected = 1 if m == 6 else EXPECTED_GROUP.get((m, a))
            assert P.toto_group(board, WIN, ADD) == expected
            assert P.toto_group(tuple(board), tuple(reversed(WIN)), ADD) == expected


def test_toto_group_rejects_bad_boards():
    with pytest.raises(ValueError):
        P.toto_group([1, 2, 3, 4, 5], WIN, ADD)
    with pytest.raises(ValueError):
        P.toto_group([1, 1, 2, 3, 4, 5], WIN, ADD)


def test_group_distribution_matches_constants_by_counting():
    """All 13,983,816 boards, counted by their overlap with the 7 drawn numbers.

    A board's group depends only on which of the 6 winning numbers and the additional number
    it holds, so the boards split into 2**7 overlap patterns, each filled from the other 42
    numbers in C(42, 6 - size) ways. ``toto_group`` classifies a real board of every pattern.
    """
    special = WIN + [ADD]
    per_group = {g: 0 for g in range(1, 8)}
    no_prize = 0
    for size in range(0, 7):
        for subset in combinations(special, size):
            ways = comb(len(NON_WIN), 6 - size)
            board = list(subset) + NON_WIN[: 6 - size]
            g = P.toto_group(board, WIN, ADD)
            if g is None:
                no_prize += ways
            else:
                per_group[g] += ways
    assert per_group == C.TOTO_GROUP_COMBOS
    assert sum(per_group.values()) == C.TOTO_ANY_PRIZE_COMBOS == 260_624
    assert sum(per_group.values()) + no_prize == C.TOTO_COMBOS == 13_983_816


# Boards and system bets


def test_toto_boards_counts_for_every_bet_size():
    assert P.toto_boards([45, 3, 11, 19, 27, 38]) == [tuple(sorted(WIN))]
    for n in range(7, 13):
        boards = P.toto_boards(list(range(1, n + 1)))
        assert len(boards) == C.TOTO_SYSTEM_BOARDS[n] == comb(n, 6)
        assert len(set(boards)) == len(boards)
        assert all(list(b) == sorted(b) for b in boards)


def test_toto_boards_accepts_string_and_validates():
    assert P.toto_boards("3 11 19 27 38 45") == [tuple(WIN)]
    assert P.toto_boards("3, 11, 19, 27, 38, 45, 49") == P.toto_boards(WIN + [49])
    for bad in ([1, 2, 3, 4, 5], list(range(1, 14)), [1, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 50]):
        with pytest.raises(ValueError):
            P.toto_boards(bad)


def test_system7_with_all_winning_and_additional():
    counts = P.toto_group_counts(WIN + [ADD], WIN, ADD)
    assert counts == {1: 1, 2: 6}


def test_system7_with_all_winning_and_a_losing_number():
    counts = P.toto_group_counts(WIN + [NON_WIN[0]], WIN, ADD)
    assert counts == {1: 1, 3: 6}


def test_system_counts_match_brute_force_for_random_systems():
    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(7, 13))
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 50), n, replace=False))
        # Force some overlap so prizes actually happen.
        k = int(rng.integers(0, 7))
        nums = sorted(set(nums[: n - k]) | set(WIN[:k]))
        while len(nums) < n:
            extra = int(rng.integers(1, 50))
            if extra not in nums:
                nums = sorted(nums + [extra])
        counts = P.toto_group_counts(nums, WIN, ADD)
        brute: dict[int, int] = {}
        for board in combinations(nums, 6):
            g = P.toto_group(board, WIN, ADD)
            if g:
                brute[g] = brute.get(g, 0) + 1
        assert counts == brute
        assert sum(counts.values()) <= comb(n, 6)


def test_system8_closed_form():
    # 4 winning numbers + additional + 3 others: the counts follow from choosing 6 of 8.
    nums = WIN[:4] + [ADD] + NON_WIN[:3]
    counts = P.toto_group_counts(nums, WIN, ADD)
    # 4 matched + additional + 1 other: C(3, 1) = 3 boards -> Group 4
    # 4 matched + 2 others: C(3, 2) = 3 -> Group 5
    # 3 matched + additional + 2 others: C(4, 3) * C(3, 2) = 12 -> Group 6
    # 3 matched + 3 others: C(4, 3) * 1 = 4 -> Group 7
    assert counts == {4: 3, 5: 3, 6: 12, 7: 4}


# Pool estimate and share if won


def test_pool_estimate_uses_group3():
    rules = PrizeRules()
    row = toto_row()
    assert P.toto_pool_estimate(row, rules) == pytest.approx(2_000.0 * 110 / 0.055)
    assert P.toto_pool_estimate(pd.Series(row), rules) == pytest.approx(2_000.0 * 110 / 0.055)


def test_pool_estimate_falls_back_to_group4_when_group3_unwon():
    rules = PrizeRules()
    row = toto_row(g3_share=np.nan, g3_winners=0)
    assert P.toto_pool_estimate(row, rules) == pytest.approx(400.0 * 300 / 0.03)


def test_pool_estimate_group3_snowball_needs_previous_draw():
    rules = PrizeRules()
    row = toto_row()
    prev = toto_row(draw_number=4122, g3_winners=0, g3_share=np.nan)
    assert P.toto_pool_estimate(row, rules, prev=prev) == pytest.approx(400.0 * 300 / 0.03)
    # A previous row that is not the draw just before is ignored.
    older = dict(prev, draw_number=4100)
    assert P.toto_pool_estimate(row, rules, prev=older) == pytest.approx(2_000.0 * 110 / 0.055)
    # Both groups snowballed: no clean estimate.
    both = dict(prev, g4_winners=0, g4_share=np.nan)
    assert P.toto_pool_estimate(row, rules, prev=both) is None


def test_pool_estimate_cascade_landing():
    rules = PrizeRules()
    # Cascade draw, no Group 1 or 2 winner: the jackpot landed in Group 3, so use Group 4.
    row = toto_row(draw_type="cascade", g1_winners=0, g1_share=np.nan, g2_winners=0, g2_share=np.nan)
    assert P.toto_pool_estimate(row, rules) == pytest.approx(400.0 * 300 / 0.03)
    # Hongbao with Group 3 unwon too: jackpot landed in Group 4, nothing clean.
    row = dict(row, draw_type="hongbao", g3_winners=0, g3_share=np.nan)
    assert P.toto_pool_estimate(row, rules) is None
    # A cascade draw where Group 2 was won: jackpot went to Group 2, Group 3 is clean.
    row = toto_row(draw_type="cascade", g1_winners=0, g1_share=np.nan)
    assert P.toto_pool_estimate(row, rules) == pytest.approx(2_000.0 * 110 / 0.055)
    # Normal draw with no Group 1 or 2 winners: no cascade, Group 3 is clean.
    row = toto_row(g1_winners=0, g1_share=np.nan, g2_winners=0, g2_share=np.nan)
    assert P.toto_pool_estimate(row, rules) == pytest.approx(2_000.0 * 110 / 0.055)


def test_pool_estimate_none_when_nothing_usable():
    rules = PrizeRules()
    row = toto_row(g3_share=np.nan, g3_winners=0, g4_share=np.nan, g4_winners=0)
    assert P.toto_pool_estimate(row, rules) is None
    row = toto_row(g3_share=np.nan)  # winners but no amount
    assert P.toto_pool_estimate(row, rules) == pytest.approx(400.0 * 300 / 0.03)


def test_share_if_won_rules():
    rules = PrizeRules()
    row = toto_row()
    assert P.toto_share_if_won(None, row, rules) == 0.0
    assert P.toto_share_if_won(1, row, rules) == pytest.approx(1_200_000 * 2 / 3)
    assert P.toto_share_if_won(2, row, rules) == pytest.approx(90_000 * 3 / 4)
    assert P.toto_share_if_won(3, row, rules) == pytest.approx(2_000 * 110 / 111)
    assert P.toto_share_if_won(4, row, rules) == pytest.approx(400 * 300 / 301)
    assert P.toto_share_if_won(5, row, rules) == 50.0
    assert P.toto_share_if_won(6, row, rules) == 25.0
    assert P.toto_share_if_won(7, row, rules) == 10.0


def test_share_if_won_unwon_groups():
    rules = PrizeRules()
    row = toto_row(g1_winners=0, g1_share=np.nan, g2_winners=0, g2_share=np.nan,
                   g3_winners=0, g3_share=np.nan)
    pool = 400.0 * 300 / 0.03  # from Group 4
    assert P.toto_share_if_won(1, row, rules) == 2_400_000.0  # the jackpot
    assert P.toto_share_if_won(2, row, rules) == pytest.approx(0.08 * pool)
    assert P.toto_share_if_won(3, row, rules) == pytest.approx(0.055 * pool)
    # No pool estimate at all: unwon pool groups are worth 0 (unknown), fixed prizes stay.
    row = dict(row, g4_winners=0, g4_share=np.nan)
    assert P.toto_share_if_won(2, row, rules) == 0.0
    assert P.toto_share_if_won(7, row, rules) == 10.0


def test_share_if_won_with_winners_but_no_share_amount_is_split():
    """A missing share cell with winners > 0 must not pay the whole group pool."""
    rules = PrizeRules()
    pool = 2_000.0 * 110 / 0.055  # from Group 3
    row = toto_row(g2_share=np.nan)  # 3 winners, amount not captured
    assert P.toto_share_if_won(2, row, rules) == pytest.approx(0.08 * pool / 4)
    row = toto_row(g4_share=np.nan)
    assert P.toto_share_if_won(4, row, rules) == pytest.approx(0.03 * pool / 301)
    # Unknown pool: unknown amount, so 0.
    row = toto_row(g2_share=np.nan, g3_share=np.nan, g3_winners=0, g4_share=np.nan, g4_winners=0)
    assert P.toto_share_if_won(2, row, rules) == 0.0
    # toto_group_amounts gives the same amounts.
    row = toto_row(g2_share=np.nan)
    assert P.toto_group_amounts(row, rules)[2] == pytest.approx(0.08 * pool / 4)


def test_ticket_prize_with_winners_but_no_share_amount_is_one_of_the_winners():
    rules = PrizeRules()
    row = toto_row(g1_share=np.nan)  # 2 winners, the jackpot is $2,400,000
    res = P.toto_ticket_prize(WIN, "Ordinary", 1.0, row, rules)
    assert res.amount == pytest.approx(1_200_000.0)  # split 2 ways, not 3
    pool = 2_000.0 * 110 / 0.055
    row = toto_row(g2_share=np.nan)  # 3 winners
    res = P.toto_ticket_prize(board_with(5, True), "Ordinary", 1.0, row, rules)
    assert res.groups == {2: 1}
    assert res.amount == pytest.approx(round(0.08 * pool / 3, 2))


def test_share_if_won_uses_rule_overrides_and_string_keys():
    rules = PrizeRules(fixed_prizes={"5": 55.0, "6": 26.0, "7": 11.0})
    row = toto_row()
    assert P.toto_share_if_won(7, row, rules) == 11.0
    assert P.toto_group_amounts(row, rules)[5] == 55.0


def test_group_amounts_match_share_if_won():
    rules = PrizeRules()
    for row in (toto_row(), toto_row(g2_winners=0, g2_share=np.nan),
                toto_row(g1_winners=0, g1_share=np.nan)):
        amounts = P.toto_group_amounts(row, rules)
        assert amounts == {g: pytest.approx(P.toto_share_if_won(g, row, rules)) for g in range(1, 8)}


# Ticket prize


def test_ticket_prize_ordinary_uses_published_share():
    rules = PrizeRules()
    row = toto_row()
    board = board_with(4, True)
    res = P.toto_ticket_prize(board, "Ordinary", 1.0, row, rules)
    assert res.amount == 400.0
    assert res.groups == {4: 1}
    assert res.detail == "Group 4 x1"
    res = P.toto_ticket_prize(NON_WIN[:6], "Ordinary", 1.0, row, rules)
    assert res.amount == 0.0 and res.groups == {} and res.detail == "No prize"


def test_ticket_prize_system7_and_units():
    rules = PrizeRules()
    row = toto_row()
    nums = WIN + [ADD]
    res = P.toto_ticket_prize(nums, "System 7", 1.0, row, rules)
    assert res.groups == {1: 1, 2: 6}
    assert res.amount == pytest.approx(1_200_000 + 6 * 90_000)
    assert res.detail == "Group 2 x6, Group 1 x1"
    res2 = P.toto_ticket_prize(" ".join(map(str, nums)), "system 7", 2.0, row, rules)
    assert res2.amount == pytest.approx(2 * res.amount)


def test_ticket_prize_detail_lists_lower_groups_first():
    rules = PrizeRules()
    row = toto_row()
    nums = WIN[:4] + [ADD] + NON_WIN[:3]  # System 8, see test_system8_closed_form
    res = P.toto_ticket_prize(nums, "System 8", 1.0, row, rules)
    assert res.detail == "Group 7 x4, Group 6 x12, Group 5 x3, Group 4 x3"
    assert res.amount == pytest.approx(4 * 10 + 12 * 25 + 3 * 50 + 3 * 400)


def test_ticket_prize_falls_back_when_page_shows_no_winner():
    rules = PrizeRules()
    row = toto_row(g1_winners=0, g1_share=np.nan)  # the page lost the Group 1 row
    res = P.toto_ticket_prize(WIN, "Ordinary", 1.0, row, rules)
    assert res.amount == 2_400_000.0
    row = toto_row(g7_share=np.nan, g7_winners=0)
    res = P.toto_ticket_prize(board_with(3, False), "Ordinary", 1.0, row, rules)
    assert res.amount == 10.0


def test_ticket_prize_rejects_inconsistent_bet():
    rules = PrizeRules()
    with pytest.raises(ValueError):
        P.toto_ticket_prize(WIN, "System 7", 1.0, toto_row(), rules)
    with pytest.raises(ValueError):
        P.toto_ticket_prize(WIN + [1], "Ordinary", 1.0, toto_row(), rules)
    with pytest.raises(ValueError):
        P.toto_ticket_prize(WIN, "Lucky Dip", 1.0, toto_row(), rules)


def test_ticket_prize_detail_has_no_dashes():
    rules = PrizeRules()
    for nums, bet in ((WIN + [ADD], "System 7"), (NON_WIN[:6], "Ordinary")):
        detail = P.toto_ticket_prize(nums, bet, 1.0, toto_row(), rules).detail
        assert not any(d in detail for d in DASHES)


@pytest.mark.parametrize("n", sorted(C.TOTO_SYSTEM_BOARDS))
def test_ticket_prize_every_system_size(n):
    rules = PrizeRules()
    row = toto_row()
    nums = WIN[:3] + [ADD] + NON_WIN[: n - 4]
    res = P.toto_ticket_prize(nums, f"System {n}", 1.0, row, rules)
    assert res.groups == P.toto_group_counts(nums, WIN, ADD)
    assert sum(res.groups.values()) <= C.TOTO_SYSTEM_BOARDS[n]
    assert res.amount == pytest.approx(sum(row[f"g{g}_share"] * k for g, k in res.groups.items()))


def test_ticket_prize_on_synthetic_history(toto_df, rules):
    """The winning set of every stored draw wins Group 1: the published share when the page
    shows winners, else the whole jackpot. A set with none of the drawn numbers wins nothing."""
    for _, row in toto_df.tail(40).iterrows():
        winning, additional = P.toto_winning(row)
        res = P.toto_ticket_prize(winning, "Ordinary", 1.0, row, rules)
        assert res.groups == {1: 1} and res.detail == "Group 1 x1"
        if int(row["g1_winners"]) > 0:
            assert res.amount == pytest.approx(round(float(row["g1_share"]), 2))
        else:
            assert res.amount == pytest.approx(float(row["jackpot"]))
        losing = [n for n in range(1, 50) if n not in winning and n != additional][:6]
        res = P.toto_ticket_prize(losing, "Ordinary", 1.0, row, rules)
        assert (res.amount, res.groups, res.detail) == (0.0, {}, "No prize")


# Bet types


@pytest.mark.parametrize("bet, size", [
    ("Ordinary", 6), ("ord", 6), ("", 6), (None, 6), (" ORDINARY ", 6),
    ("System 7", 7), ("system7", 7), ("SYSTEM  12", 12),
])
def test_toto_bet_size(bet, size):
    assert P.toto_bet_size(bet) == size


@pytest.mark.parametrize("bet", ["System 6", "System 13", "Big", "iBet Big", "Lucky Dip"])
def test_toto_bet_size_rejects_other_bets(bet):
    with pytest.raises(ValueError):
        P.toto_bet_size(bet)
    with pytest.raises(ValueError):
        P.toto_ticket_prize(WIN, bet, 1.0, toto_row(), PrizeRules())


def test_module_is_toto_only():
    names = [n for n in dir(P) if not n.startswith("__")]
    assert not [n for n in names if re.search(r"fourd|4d|ibet|permutation|straight|matrix", n, re.I)]
