"""Tests for huatbot.prizes: TOTO groups, system bets, pool estimate, vectorised scoring and 4D."""
from __future__ import annotations

import time
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


def test_group_distribution_vectorised_sample_matches_odds():
    """A large random sample through the vectorised path lands in groups at the right rates."""
    rng = np.random.default_rng(42)
    sets = np.argpartition(rng.random((100_000, 49)), 6, axis=1)[:, :6] + 1
    groups = P.toto_group_matrix(sets, WIN, ADD)
    any_prize = (groups > 0).mean()
    assert any_prize == pytest.approx(C.TOTO_ANY_PRIZE_COMBOS / C.TOTO_COMBOS, rel=0.07)
    g7 = (groups == 7).mean()
    assert g7 == pytest.approx(C.TOTO_GROUP_COMBOS[7] / C.TOTO_COMBOS, rel=0.07)


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
    # The backtest matrix uses the same amounts.
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


# Vectorised TOTO


def _random_sets(rng, k: int) -> np.ndarray:
    return np.sort(np.argpartition(rng.random((k, 49)), 6, axis=1)[:, :6] + 1, axis=1)


def _planted_sets(rng, k: int) -> np.ndarray:
    """Sets that overlap the winning numbers heavily, so every group shows up."""
    out = []
    special = WIN + [ADD]
    for _ in range(k):
        size = int(rng.integers(2, 7))
        chosen = list(rng.choice(special, size, replace=False))
        rest = [n for n in range(1, 50) if n not in chosen]
        chosen += list(rng.choice(rest, 6 - size, replace=False))
        out.append(sorted(int(x) for x in chosen))
    return np.array(out)


@pytest.mark.parametrize("row_kwargs", [
    {},
    {"g1_winners": 0, "g1_share": np.nan},
    {"g2_winners": 0, "g2_share": np.nan, "g3_winners": 0, "g3_share": np.nan},
    {"draw_type": "cascade", "g1_winners": 0, "g1_share": np.nan, "g2_winners": 0, "g2_share": np.nan},
])
def test_prize_matrix_agrees_with_scalar(row_kwargs):
    rules = PrizeRules()
    row = pd.Series(toto_row(**row_kwargs))
    rng = np.random.default_rng(3)
    sets = np.vstack([_random_sets(rng, 3000), _planted_sets(rng, 2000)])
    got = P.toto_prize_matrix(sets, row, rules)
    assert got.shape == (len(sets),)
    winning, additional = P.toto_winning(row)
    expected = np.array([P.toto_share_if_won(P.toto_group(s, winning, additional), row, rules)
                         for s in sets])
    np.testing.assert_allclose(got, expected)
    groups = P.toto_group_matrix(sets, winning, additional)
    assert set(np.unique(groups)) == {0, 1, 2, 3, 4, 5, 6, 7}


def test_prize_matrix_on_synthetic_history(toto_df, rules):
    rng = np.random.default_rng(5)
    for _, row in toto_df.tail(20).iterrows():
        sets = np.vstack([_random_sets(rng, 200), np.array([P.toto_winning(row)[0]])])
        got = P.toto_prize_matrix(sets, row, rules)
        winning, additional = P.toto_winning(row)
        expected = [P.toto_share_if_won(P.toto_group(s, winning, additional), row, rules) for s in sets]
        np.testing.assert_allclose(got, expected)
        assert got[-1] > 0  # the winning set itself wins Group 1


def test_prize_matrix_is_fast():
    rules = PrizeRules()
    row = pd.Series(toto_row())
    sets = _random_sets(np.random.default_rng(9), 1000)
    P.toto_prize_matrix(sets, row, rules)  # warm up
    best = float("inf")
    for _ in range(7):
        start = time.perf_counter()
        P.toto_prize_matrix(sets, row, rules)
        best = min(best, time.perf_counter() - start)
    assert best < 0.010, f"1000 sets took {best * 1000:.2f} ms"


def test_prize_matrix_shapes_and_validation():
    rules = PrizeRules()
    row = toto_row()
    assert P.toto_prize_matrix(np.zeros((0, 6), dtype=int), row, rules).shape == (0,)
    single = P.toto_prize_matrix(np.array(WIN), row, rules)
    assert single.shape == (1,) and single[0] == pytest.approx(800_000)
    with pytest.raises(ValueError):
        P.toto_prize_matrix(np.array([[1, 2, 3, 4, 5]]), row, rules)
    with pytest.raises(ValueError):
        P.toto_prize_matrix(np.array([[1, 1, 2, 3, 4, 5]]), row, rules)
    with pytest.raises(ValueError):
        P.toto_prize_matrix(np.array([[0, 1, 2, 3, 4, 5]]), row, rules)


def test_popcount_fallback_matches():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2**62, size=5000, dtype=np.uint64) | np.uint64(1 << 63)
    expected = np.array([bin(int(v)).count("1") for v in x])
    np.testing.assert_array_equal(P._popcount_swar(x), expected)
    np.testing.assert_array_equal(P._popcount(x), expected)


# 4D


def fourd_row(**overrides) -> dict:
    row = {"draw_number": 5432, "draw_date": pd.Timestamp("2026-09-30"),
           "first": "1234", "second": "5678", "third": "0042", "fetched_at": "test"}
    starters = ["1111", "2222", "3333", "4321", "0001", "9090", "1243", "", "7777", "8888"]
    consols = ["0000", "9999", "5555", "6666", "1234", "2143", "3412", "0420", "", "1324"]
    for i in range(10):
        row[f"starter_{i + 1}"] = starters[i]
        row[f"consolation_{i + 1}"] = consols[i]
    row.update(overrides)
    return row


@pytest.mark.parametrize("number,count", [("1234", 24), ("1123", 12), ("1122", 6), ("1112", 4),
                                          ("1111", 1), ("0042", 12), ("0004", 4)])
def test_permutations(number, count):
    perms = P.permutations(number)
    assert len(perms) == count == P.permutations_count(number)
    assert perms == sorted(perms) and len(set(perms)) == count
    assert all(sorted(p) == sorted(number) for p in perms)
    assert number in perms


def test_permutations_reject_bad_numbers():
    for bad in ("123", "12345", "12a4", "", "-123"):
        with pytest.raises(ValueError):
            P.permutations_count(bad)
    assert P.permutations(42) == P.permutations("0042")


def test_ibet_table_default_floor():
    rules = PrizeRules()
    assert P.ibet_table(rules, "big", 24) == {"first": 83, "second": 41, "third": 20,
                                             "starter": 10, "consolation": 2}
    assert P.ibet_table(rules, "small", 24) == {"first": 125, "second": 83, "third": 33}
    assert P.ibet_table(rules, "Big", 4) == {"first": 500, "second": 250, "third": 122,
                                            "starter": 62, "consolation": 15}
    assert P.ibet_table(rules, "big", 1) == C.FOURD_PRIZES["big"]
    with pytest.raises(ValueError):
        P.ibet_table(rules, "big", 0)
    with pytest.raises(ValueError):
        P.ibet_table(rules, "medium", 24)


def test_ibet_table_uses_published_values():
    rules = PrizeRules(ibet_prizes={"big": {24: {"first": 84.0, "second": 42.0}},
                                    "small": {"12": {"first": 251.0}}})
    table = P.ibet_table(rules, "big", 24)
    assert table["first"] == 84 and table["second"] == 42 and table["third"] == 20
    assert P.ibet_table(rules, "small", 12)["first"] == 251  # JSON style string key
    assert P.ibet_table(rules, "big", 12)["first"] == 166
    assert P.ibet_is_published(rules, "big", 24) and not P.ibet_is_published(rules, "big", 12)


def test_fourd_hits():
    row = fourd_row()
    assert P.fourd_hits("1234", row) == ["first", "consolation"]
    assert P.fourd_hits("4321", row) == ["starter"]
    assert P.fourd_hits("0042", row) == ["third"]
    assert P.fourd_hits("4242", row) == []
    assert P.fourd_hits(42, row) == ["third"]


def test_fourd_ticket_prize_big_and_small():
    rules = PrizeRules()
    row = fourd_row()
    res = P.fourd_ticket_prize("1234", "Big", 1, row, rules)
    assert res.amount == 2000 + 60 and res.groups == {"first": 1, "consolation": 1}
    assert res.detail == "1st Prize x1, Consolation x1"
    assert P.fourd_ticket_prize("1234", "Big", 3, row, rules).amount == 3 * 2060
    res = P.fourd_ticket_prize("1234", "Small", 2, row, rules)
    assert res.amount == 6000 and res.groups == {"first": 1}
    # Small pays nothing for Starter or Consolation.
    res = P.fourd_ticket_prize("4321", "Small", 5, row, rules)
    assert res.amount == 0 and res.groups == {} and res.detail == "No prize"
    assert P.fourd_ticket_prize("4321", "big", 5, row, rules).amount == 5 * 250
    assert P.fourd_ticket_prize("0042", "Small", 1, row, rules).amount == 800
    assert P.fourd_ticket_prize("5678", "Small", 1, row, rules).amount == 2000


def test_fourd_ticket_prize_ibet():
    rules = PrizeRules()
    row = fourd_row()
    # 1234 covers 24 arrangements; winners among them: first 1234, starters 4321 and 1243,
    # consolations 1234, 2143, 3412, 1324.
    res = P.fourd_ticket_prize("4123", "iBet Big", 1, row, rules)
    assert res.groups == {"first": 1, "starter": 2, "consolation": 4}
    assert res.amount == 83 + 2 * 10 + 4 * 2
    res = P.fourd_ticket_prize("4123", "iBet Big", 2, row, rules)
    assert res.amount == 2 * (83 + 2 * 10 + 4 * 2)
    res = P.fourd_ticket_prize("4123", "iBet Small", 1, row, rules)
    assert res.groups == {"first": 1} and res.amount == 125
    # 0042 has 12 arrangements: third prize 0042 and consolation 0420 are both covered.
    res = P.fourd_ticket_prize("4200", "iBet Big", 1, row, rules)
    assert res.groups == {"third": 1, "consolation": 1}
    assert res.amount == 490 // 12 + 60 // 12
    # A number with 1 arrangement pays the straight table.
    assert P.fourd_ticket_prize("1111", "iBet Big", 1, row, rules).amount == 250


def test_fourd_ticket_prize_validation():
    rules = PrizeRules()
    row = fourd_row()
    with pytest.raises(ValueError):
        P.fourd_ticket_prize("1234", "Huge", 1, row, rules)
    with pytest.raises(ValueError):
        P.fourd_ticket_prize("123", "Big", 1, row, rules)
    with pytest.raises(ValueError):
        P.fourd_ticket_prize("1234", "Big", -1, row, rules)


def test_fourd_row_numbers_skip_blanks():
    row = fourd_row(first="", starter_1=np.nan, consolation_1=None)
    pairs = P.fourd_row_numbers(pd.Series(row))
    assert len(pairs) == 23 - 2 - 3  # two blanks in the fixture plus three cleared
    assert all(len(n) == 4 for _, n in pairs)
    assert P.fourd_hits("1234", row) == ["consolation"]


@pytest.mark.parametrize("bet", ["big", "small"])
def test_fourd_prize_vector_agrees_with_ticket_prize(bet, fourd_df):
    rules = PrizeRules()
    rng = np.random.default_rng(4)
    for row in (pd.Series(fourd_row()), fourd_df.iloc[-1]):
        winners = [int(n) for _, n in P.fourd_row_numbers(row)]
        sample = np.unique(np.concatenate([winners, rng.integers(0, 10_000, 1500)]))
        vec = P.fourd_prize_vector(sample, row, rules, bet=bet)
        scalar = [P.fourd_ticket_prize(f"{n:04d}", bet.title(), 1, row, rules).amount for n in sample]
        np.testing.assert_allclose(vec, scalar)


@pytest.mark.parametrize("bet", ["Big", "Small"])
def test_mean_prize_over_every_number_is_expected_return(bet, fourd_df):
    """A draw has 23 distinct winning numbers, so the average $1 prize over all 10,000
    numbers is exactly the expected return per $1."""
    rules = PrizeRules()
    for _, row in fourd_df.tail(5).iterrows():
        vec = P.fourd_prize_vector(np.arange(10_000), row, rules, bet=bet)
        assert vec.mean() == pytest.approx(P.fourd_expected_return(rules, bet))


def test_fourd_prize_vector_shapes_and_validation():
    rules = PrizeRules()
    row = fourd_row()
    out = P.fourd_prize_vector(np.array([[1234, 42], [4321, 7]]), row, rules)
    assert out.shape == (2, 2)
    assert out.tolist() == [[2060.0, 490.0], [250.0, 0.0]]
    with pytest.raises(ValueError):
        P.fourd_prize_vector(np.array([10_000]), row, rules)
    with pytest.raises(ValueError):
        P.fourd_prize_vector(np.array([1]), row, rules, bet="iBet Big")


EXPECTED_IBET = {
    "iBet Big": {24: 0.6336, 12: 0.6468, 6: 0.654, 4: 0.6568},
    "iBet Small": {24: 0.5784, 12: 0.5784, 6: 0.5796, 4: 0.58},
}


def test_expected_returns_exact():
    rules = PrizeRules()
    assert P.fourd_expected_return(rules, "Big") == pytest.approx(0.659)
    assert P.fourd_expected_return(rules, "Small") == pytest.approx(0.58)
    for bet, by_perm in EXPECTED_IBET.items():
        for perms, value in by_perm.items():
            assert P.fourd_expected_return(rules, bet, perms) == pytest.approx(value)
    with pytest.raises(ValueError):
        P.fourd_expected_return(rules, "iBet Big")


def test_ibet_payouts_cover_each_winner_perms_times(fourd_df):
    """Every winning number is covered by exactly ``perms`` iBet tickets, each paid the iBet
    amount, so summed over all tickets of a shape the payout is perms x iBet prize per
    winner of that shape. This is what makes the iBet expected return perms x table / 10,000."""
    rules = PrizeRules()
    row = fourd_df.iloc[-1]
    pairs = P.fourd_row_numbers(row)
    covered = sorted({p for _, n in pairs for p in P.permutations(n)})
    totals: dict[int, float] = {}
    for n in covered:
        perms = P.permutations_count(n)
        totals[perms] = totals.get(perms, 0.0) + P.fourd_ticket_prize(n, "iBet Big", 1, row, rules).amount
    for perms in set(totals) | {4, 6, 12, 24}:
        table = P.ibet_table(rules, "big", perms)
        expected = sum(perms * table[t] for t, n in pairs if P.permutations_count(n) == perms)
        assert totals.get(perms, 0.0) == pytest.approx(expected)
    # Tickets that cover no winning number win nothing.
    rng = np.random.default_rng(8)
    covered_set = set(covered)
    for n in rng.integers(0, 10_000, 300):
        number = f"{n:04d}"
        if number not in covered_set:
            assert P.fourd_ticket_prize(number, "iBet Big", 1, row, rules).amount == 0
