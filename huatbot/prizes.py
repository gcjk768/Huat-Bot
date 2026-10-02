"""Prize rules for TOTO and 4D: what a board, ticket or number won in a given draw.

Two kinds of callers use this module:

* ``tickets.settle_ledger`` checks a real ticket against a real result with
  ``toto_ticket_prize`` / ``fourd_ticket_prize`` (published share amounts).
* the backtests score thousands of candidate sets per draw with the vectorised
  ``toto_prize_matrix`` / ``fourd_prize_vector`` (what ONE extra winning board
  would have received, see ``toto_share_if_won``).

Rows are ``pd.Series`` rows of the toto.csv / fourd.csv frames (``models.TOTO_COLUMNS``,
``models.FOURD_COLUMNS``) or plain dicts with the same keys.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import combinations
from itertools import permutations as _iter_permutations
from typing import Any

import numpy as np
import pandas as pd

from . import constants as C
from .models import PrizeResult, PrizeRules

log = logging.getLogger(__name__)

# TOTO

# (numbers matched, additional number matched) -> prize group.
_GROUP_BY_MATCH: dict[tuple[int, bool], int] = {
    (6, False): 1,
    (5, True): 2,
    (5, False): 3,
    (4, True): 4,
    (4, False): 5,
    (3, True): 6,
    (3, False): 7,
}

# Same table as a numpy lookup for the vectorised path: GROUP_TABLE[matches, has_additional],
# 0 means no prize. Row 6 column 1 cannot happen (a 6 number board that matches all six
# winning numbers has no room for the additional number) and stays Group 1 for safety.
GROUP_TABLE = np.zeros((C.TOTO_PICK + 1, 2), dtype=np.int64)
for (_m, _a), _g in _GROUP_BY_MATCH.items():
    GROUP_TABLE[_m, int(_a)] = _g
GROUP_TABLE[6, 1] = 1

TOTO_GROUPS = tuple(range(1, 8))
_POOL_GROUPS = (2, 3, 4)  # groups paid as a percentage of the prize pool
_CASCADE_TYPES = ("cascade", "hongbao")  # draw types whose unwon jackpot cascades down

# Bit k of a board mask is set when number k is on the board (bit 0 is never used).
_BIT = np.left_shift(np.uint64(1), np.arange(C.TOTO_MAX_NUMBER + 1, dtype=np.uint64))


def _popcount_swar(x: np.ndarray) -> np.ndarray:
    """Set bits per uint64 element (classic SWAR popcount), for numpy without bitwise_count."""
    x = np.asarray(x, dtype=np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    with np.errstate(over="ignore"):
        x = x * h01  # wraps modulo 2**64 by design
    return (x >> np.uint64(56)).astype(np.int64)


if hasattr(np, "bitwise_count"):  # numpy 2

    def _popcount(x: np.ndarray) -> np.ndarray:
        return np.bitwise_count(x).astype(np.int64)

else:  # pragma: no cover, numpy 1.26
    _popcount = _popcount_swar


# Row access helpers (rows may be pd.Series or dicts, values may be NaN)


def _get(row: Any, key: str, default: Any = None) -> Any:
    """``row[key]`` with missing keys, None and NaN / NaT mapped to ``default``."""
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):  # list like values: pd.isna returns an array
        pass
    return value


def _num(row: Any, key: str) -> float | None:
    """A finite float from the row, else None."""
    value = _get(row, key)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(row: Any, key: str, default: int = 0) -> int:
    f = _num(row, key)
    return int(f) if f is not None else default


def _rule(mapping: Mapping, key: Any, default: Any = None) -> Any:
    """Look up ``key`` in a rules mapping that may have been through JSON (string keys)."""
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def _parse_toto_numbers(numbers: Any) -> list[int]:
    """Numbers as a list of ints from a list / array / "3 11 19 27 38 45" style string."""
    if isinstance(numbers, str):
        parts = [p for p in re.split(r"[\s,]+", numbers.strip()) if p]
        return [int(p) for p in parts]
    return [int(x) for x in numbers]


def toto_winning(row: Any) -> tuple[list[int], int]:
    """(six winning numbers ascending, additional number) of a toto.csv row."""
    winning = sorted(int(_get(row, f"n{i}", 0)) for i in range(1, C.TOTO_PICK + 1))
    return winning, int(_get(row, "additional", 0))


# TOTO scalar rules


def toto_group(board: Iterable[int], winning: Iterable[int], additional: int) -> int | None:
    """Prize group (1 to 7) of one 6 number board, or None when it wins nothing.

    6 matched -> 1; 5 + additional -> 2; 5 -> 3; 4 + additional -> 4; 4 -> 5;
    3 + additional -> 6; 3 -> 7.
    """
    b = {int(x) for x in board}
    if len(b) != C.TOTO_PICK:
        raise ValueError(f"a TOTO board has {C.TOTO_PICK} distinct numbers, got {sorted(b)}")
    w = {int(x) for x in winning}
    matched = len(b & w)
    if matched == C.TOTO_PICK:
        return 1
    has_additional = int(additional) in b and int(additional) not in w
    return _GROUP_BY_MATCH.get((matched, has_additional))


def toto_boards(numbers: Any) -> list[tuple[int, ...]]:
    """Every 6 number board a bet covers: 1 for an Ordinary bet, C(n, 6) for System n."""
    nums = sorted(_parse_toto_numbers(numbers))
    if len(set(nums)) != len(nums):
        raise ValueError(f"TOTO numbers must be distinct, got {nums}")
    if not C.TOTO_PICK <= len(nums) <= 12:
        raise ValueError(f"a TOTO bet has 6 to 12 numbers, got {len(nums)}")
    if nums[0] < 1 or nums[-1] > C.TOTO_MAX_NUMBER:
        raise ValueError(f"TOTO numbers run from 1 to {C.TOTO_MAX_NUMBER}, got {nums}")
    return list(combinations(nums, C.TOTO_PICK))


def toto_group_counts(numbers: Any, winning: Iterable[int], additional: int) -> dict[int, int]:
    """{group: boards} over every board of the bet; groups with no board are left out."""
    winning = list(winning)
    counts: Counter[int] = Counter()
    for board in toto_boards(numbers):
        g = toto_group(board, winning, additional)
        if g is not None:
            counts[g] += 1
    return dict(sorted(counts.items()))


def _cascade_landed(row: Any, group: int) -> bool:
    """True when an unwon cascade / Hongbao jackpot was added to ``group`` in this draw.

    The jackpot goes to the highest group below Group 1 that has winners.
    """
    if str(_get(row, "draw_type", "normal")).lower() not in _CASCADE_TYPES:
        return False
    if _int(row, "g1_winners") > 0:
        return False
    return all(_int(row, f"g{g}_winners") == 0 for g in range(2, group))


def _clean_group_total(row: Any, prev: Any, group: int) -> float | None:
    """share x winners for ``group`` when that amount is pure pool money, else None."""
    winners = _int(row, f"g{group}_winners")
    share = _num(row, f"g{group}_share")
    if winners <= 0 or share is None or share <= 0:
        return None
    if prev is not None and _int(prev, f"g{group}_winners", default=-1) == 0:
        return None  # the group carried a snowball from the previous draw
    if _cascade_landed(row, group):
        return None
    return share * winners


def toto_pool_estimate(row: Any, rules: PrizeRules, prev: Any = None) -> float | None:
    """Prize pool of the draw, estimated the same way as ``analysis_toto.crowd_table``.

    pool = Group 3 share x Group 3 winners / Group 3 pool percentage, or Group 4 with its
    percentage when Group 3 is not clean: no winner or no amount, a snowball from the
    previous draw, or a cascaded jackpot that landed in Group 3. None when neither group
    is clean.

    ``prev`` (optional, an addition to the SPEC signature) is the row of the previous draw.
    It is only used when its draw number is exactly one less; without it no snowball can be
    seen and none is assumed, as ``crowd_table`` does for a draw missing from the frame.
    """
    if prev is not None:
        this_no, prev_no = _num(row, "draw_number"), _num(prev, "draw_number")
        if this_no is not None and prev_no is not None and prev_no != this_no - 1:
            prev = None
    for group in (3, 4):
        pct = _rule(rules.group_pool_pct, group)
        if not pct or pct <= 0:
            continue
        total = _clean_group_total(row, prev, group)
        if total is None:
            continue
        pool = total / float(pct)
        if math.isfinite(pool) and pool > 0:
            return float(pool)
    return None


def _share_if_won(group: int, row: Any, rules: PrizeRules, pool: float | None) -> float:
    """``toto_share_if_won`` with the pool estimate supplied (so callers compute it once)."""
    if group == 1:
        winners = _int(row, "g1_winners")
        if winners > 0:
            share = _num(row, "g1_share")
            if share is not None:
                return share * winners / (winners + 1)
            # Share amount not captured: split the Group 1 prize shown on the page.
            total = _num(row, "jackpot") or float(rules.min_group1) * winners
            return total / (winners + 1)
        jackpot = _num(row, "jackpot")
        return jackpot if jackpot is not None else float(rules.min_group1)
    if group in _POOL_GROUPS:
        winners = _int(row, f"g{group}_winners")
        share = _num(row, f"g{group}_share")
        if winners > 0 and share is not None:
            return share * winners / (winners + 1)
        pct = _rule(rules.group_pool_pct, group, 0.0) or 0.0
        return float(pct) * pool if pool else 0.0
    if group in (5, 6, 7):
        return float(_rule(rules.fixed_prizes, group, C.TOTO_FIXED_PRIZES[group]))
    raise ValueError(f"TOTO prize groups are 1 to 7, got {group}")


def toto_share_if_won(group: int | None, row: Any, rules: PrizeRules) -> float:
    """What ONE extra winning board in ``group`` would have received in this draw.

    Group 1: winners > 0 -> g1_share x w / (w + 1), else the row's jackpot.
    Groups 2 to 4: winners > 0 -> share x w / (w + 1), else group percentage x pool estimate
    (0 when the pool cannot be estimated). Groups 5 to 7: the fixed prize. None -> 0.
    Snowballs and a cascaded jackpot that an unwon group would also have collected are not
    added (rare, and the per draw data does not show them reliably).
    """
    if group is None:
        return 0.0
    group = int(group)
    pool = toto_pool_estimate(row, rules) if group in _POOL_GROUPS else None
    return float(_share_if_won(group, row, rules, pool))


def toto_group_amounts(row: Any, rules: PrizeRules) -> dict[int, float]:
    """{group: toto_share_if_won(group)} for groups 1 to 7, computing the pool estimate once."""
    pool = toto_pool_estimate(row, rules)
    return {g: float(_share_if_won(g, row, rules, pool)) for g in TOTO_GROUPS}


def _published_share(group: int, row: Any) -> float | None:
    """The share amount printed on the result page, when the page shows winners for it."""
    share = _num(row, f"g{group}_share")
    if share is None or share <= 0 or _int(row, f"g{group}_winners") <= 0:
        return None
    return share


def _check_toto_bet(bet_type: str, count: int) -> None:
    """Raise ValueError if ``bet_type`` ("Ordinary", "System 7".."System 12") does not fit."""
    s = str(bet_type or "").strip().lower()
    if s in ("", "ordinary", "ord"):
        expected = C.TOTO_PICK
    else:
        m = re.fullmatch(r"system\s*(\d+)", s)
        if not m:
            raise ValueError(f"unknown TOTO bet type {bet_type!r}")
        expected = int(m.group(1))
    if count != expected:
        raise ValueError(f"{bet_type} needs {expected} numbers, got {count}")


def toto_detail(groups: Mapping[int, int]) -> str:
    """Plain summary such as "Group 7 x3, Group 6 x1" (lower prizes first), or "No prize"."""
    parts = [f"Group {g} x{n}" for g, n in sorted(groups.items(), reverse=True) if n > 0]
    return ", ".join(parts) if parts else "No prize"


def toto_ticket_prize(numbers: Any, bet_type: str, units: float, row: Any,
                      rules: PrizeRules) -> PrizeResult:
    """What a real ticket that played this draw won.

    Each winning board earns the share amount published for its group. If the page shows no
    winner for that group (missing data), ``toto_share_if_won`` is used instead. The total is
    multiplied by ``units`` (the stake per board, cost / boards).
    """
    nums = _parse_toto_numbers(numbers)
    _check_toto_bet(bet_type, len(nums))
    winning, additional = toto_winning(row)
    groups = toto_group_counts(nums, winning, additional)
    amount = 0.0
    for group, boards in groups.items():
        per_board = _published_share(group, row)
        if per_board is None:
            per_board = toto_share_if_won(group, row, rules)
            log.debug("draw %s: no published Group %d share, using %.2f",
                      _get(row, "draw_number"), group, per_board)
        amount += per_board * boards
    amount = round(amount * float(units), 2)
    return PrizeResult(amount=amount, groups=dict(groups), detail=toto_detail(groups))


# TOTO vectorised (backtests)


def _board_masks(sets: Any) -> np.ndarray:
    """(k, 6) numbers -> (k,) uint64 bit masks, validating range and distinctness."""
    arr = np.asarray(sets)
    if arr.ndim == 1 and arr.size == C.TOTO_PICK:
        arr = arr.reshape(1, C.TOTO_PICK)
    if arr.ndim != 2 or (arr.shape[0] and arr.shape[1] != C.TOTO_PICK):
        raise ValueError(f"sets must have shape (k, {C.TOTO_PICK}), got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(0, dtype=np.uint64)
    arr = arr.astype(np.int64, copy=False)
    if arr.min() < 1 or arr.max() > C.TOTO_MAX_NUMBER:
        raise ValueError(f"TOTO numbers run from 1 to {C.TOTO_MAX_NUMBER}")
    masks = np.bitwise_or.reduce(_BIT[arr], axis=1)
    if (_popcount(masks) != C.TOTO_PICK).any():
        raise ValueError("every set needs 6 distinct numbers")
    return masks


def toto_group_matrix(sets: Any, winning: Iterable[int], additional: int) -> np.ndarray:
    """Vectorised ``toto_group``: (k, 6) sets -> (k,) int groups, 0 meaning no prize."""
    masks = _board_masks(sets)
    win_mask = np.bitwise_or.reduce(_BIT[np.asarray(list(winning), dtype=np.int64)])
    add_bit = _BIT[int(additional)]
    matches = _popcount(masks & win_mask)
    has_add = ((masks & add_bit) != 0).astype(np.intp)
    return GROUP_TABLE[matches, has_add]


def toto_prize_matrix(sets: Any, row: Any, rules: PrizeRules) -> np.ndarray:
    """Prize per set (k,) for (k, 6) candidate sets, using ``toto_share_if_won`` amounts.

    Each set is scored on its own as one extra board in this draw, which is what a strategy
    that bought that set would have received.
    """
    winning, additional = toto_winning(row)
    groups = toto_group_matrix(sets, winning, additional)
    if groups.size == 0:
        return np.zeros(0, dtype=np.float64)
    amounts = np.zeros(len(TOTO_GROUPS) + 1, dtype=np.float64)  # index 0 = no prize
    for g, amount in toto_group_amounts(row, rules).items():
        amounts[g] = amount
    return amounts[groups]


# 4D

FOURD_BET_TYPES = ("Big", "Small", "iBet Big", "iBet Small")
SMALL_TIERS = ("first", "second", "third")  # Small pays only the top three prizes
FOURD_TIER_LABELS = {
    "first": "1st Prize",
    "second": "2nd Prize",
    "third": "3rd Prize",
    "starter": "Starter",
    "consolation": "Consolation",
}
IBET_PERMUTATIONS = (4, 6, 12, 24)

# Row column -> tier, in the order of models.FOURD_NUMBER_COLUMNS.
FOURD_SLOT_TIER: dict[str, str] = {"first": "first", "second": "second", "third": "third"}
FOURD_SLOT_TIER.update({f"starter_{i}": "starter" for i in range(1, 11)})
FOURD_SLOT_TIER.update({f"consolation_{i}": "consolation" for i in range(1, 11)})

_FOUR_DIGITS = re.compile(r"[0-9]{4}")


def clean_fourd_number(value: Any) -> str:
    """A 4 digit string from a stored cell ("0042", 42, "42.0"), or "" if blank or invalid."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):04d}" if 0 <= int(value) < C.FOURD_SPACE else ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value) or value != int(value):
            return ""
        return clean_fourd_number(int(value))
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isascii() and s.isdigit() and len(s) < 4:
        s = s.zfill(4)
    return s if _FOUR_DIGITS.fullmatch(s) else ""


def normalise_fourd_number(number: Any) -> str:
    """Ticket number as a 4 digit string ("0042"); ValueError if it is not one."""
    if isinstance(number, str):
        s = number.strip()
        if not _FOUR_DIGITS.fullmatch(s):
            raise ValueError(f"a 4D number has 4 digits, got {number!r}")
        return s
    s = clean_fourd_number(number)
    if not s:
        raise ValueError(f"a 4D number runs from 0000 to 9999, got {number!r}")
    return s


def fourd_row_numbers(row: Any) -> list[tuple[str, str]]:
    """(tier, number) for every winning number of a fourd.csv row, blanks skipped, in order."""
    out = []
    for col, tier in FOURD_SLOT_TIER.items():
        number = clean_fourd_number(_get(row, col, ""))
        if number:
            out.append((tier, number))
    return out


def permutations(number: str) -> list[str]:
    """Distinct digit arrangements of a 4D number, sorted ("1123" gives 12 of them)."""
    num = normalise_fourd_number(number)
    return sorted({"".join(p) for p in _iter_permutations(num)})


def permutations_count(number: str) -> int:
    """How many distinct arrangements a 4D number has: 1, 4, 6, 12 or 24."""
    num = normalise_fourd_number(number)
    count = math.factorial(len(num))
    for repeats in Counter(num).values():
        count //= math.factorial(repeats)
    return count


def parse_fourd_bet(bet_type: str) -> tuple[bool, str]:
    """"Big" -> (False, "big"), "iBet Small" -> (True, "small"). Case and spacing do not matter."""
    s = re.sub(r"\s+", " ", str(bet_type or "").strip().lower())
    ibet = False
    for prefix in ("ibet ", "i bet "):
        if s.startswith(prefix):
            ibet, s = True, s[len(prefix):]
            break
    if s not in ("big", "small"):
        raise ValueError(f"unknown 4D bet type {bet_type!r}, expected one of {FOURD_BET_TYPES}")
    return ibet, s


def _bet_name(bet: str) -> str:
    """"big" / "Big" / "iBet Big" -> "big" (iBet prefix allowed for convenience)."""
    return parse_fourd_bet(bet)[1]


def straight_table(rules: PrizeRules, bet: str) -> dict[str, float]:
    """Prize per $1 for a straight Big or Small bet, tier -> amount (Small: top three only)."""
    bet = _bet_name(bet)
    table = rules.fourd_prizes.get(bet) or C.FOURD_PRIZES[bet]
    tiers = SMALL_TIERS if bet == "small" else C.FOURD_TIERS
    return {t: float(table[t]) for t in tiers if t in table}


def _published_ibet(rules: PrizeRules, bet: str, perms: int) -> Mapping | None:
    """The iBet table read from the official page for (bet, perms), if there is one."""
    by_bet = next((v for k, v in (rules.ibet_prizes or {}).items() if str(k).lower() == bet), None)
    if not by_bet:
        return None
    table = _rule(by_bet, perms)
    return table or None


def ibet_is_published(rules: PrizeRules, bet: str, perms: int) -> bool:
    """True when the prize table for this iBet came from the official page."""
    return _published_ibet(rules, _bet_name(bet), int(perms)) is not None


def ibet_table(rules: PrizeRules, bet: str, perms: int) -> dict[str, float]:
    """iBet prize per $1 for each tier, for a number with ``perms`` arrangements.

    Uses the official table (``rules.ibet_prizes``) when present, else the straight prize
    divided by ``perms`` and rounded down to a whole dollar, which is how Singapore Pools
    publishes it (iBet 24 Big: 1st $83, 2nd $41, 3rd $20, Starter $10, Consolation $2).
    """
    bet = _bet_name(bet)
    perms = int(perms)
    if perms < 1:
        raise ValueError(f"permutation count must be at least 1, got {perms}")
    published = _published_ibet(rules, bet, perms) or {}
    out = {}
    for tier, prize in straight_table(rules, bet).items():
        if tier in published:
            out[tier] = float(published[tier])
        else:
            # Tiny epsilon so 3000 / 24 style exact quotients never floor one dollar low.
            out[tier] = float(math.floor(prize / perms + 1e-9))
    return out


def fourd_expected_return(rules: PrizeRules, bet_type: str, perms: int | None = None) -> float:
    """Average prize per $1 staked, from the prize table (23 of 10,000 numbers win per draw).

    Straight: sum over tiers of (numbers in tier) x prize / 10,000. iBet with ``perms``
    arrangements: each of the ``perms`` numbers covered has the same chance, so
    perms x sum(numbers in tier x iBet prize) / 10,000.
    """
    ibet, bet = parse_fourd_bet(bet_type)
    if ibet:
        if perms is None:
            raise ValueError("iBet needs the permutation count")
        table, factor = ibet_table(rules, bet, perms), int(perms)
    else:
        table, factor = straight_table(rules, bet), 1
    total = sum(C.FOURD_TIER_COUNTS[t] * amount for t, amount in table.items())
    return round(factor * total / C.FOURD_SPACE, 10)


def fourd_hits(number: str, row: Any) -> list[str]:
    """Tiers this exact number won in the draw (e.g. ["starter"]); a repeat counts twice."""
    num = normalise_fourd_number(number)
    return [tier for tier, n in fourd_row_numbers(row) if n == num]


def fourd_detail(groups: Mapping[str, int]) -> str:
    """Plain summary such as "1st Prize x1, Starter x1", or "No prize"."""
    parts = [f"{FOURD_TIER_LABELS[t]} x{groups[t]}" for t in C.FOURD_TIERS if groups.get(t, 0) > 0]
    return ", ".join(parts) if parts else "No prize"


def fourd_ticket_prize(number: str, bet_type: str, stake: float, row: Any,
                       rules: PrizeRules) -> PrizeResult:
    """What a 4D ticket won.

    Big pays all five tiers, Small only 1st, 2nd and 3rd. iBet pays the ``ibet_table``
    amount for every winning number that is any arrangement of the ticket's digits. The
    per $1 prize is multiplied by ``stake`` (dollars on the ticket).
    """
    num = normalise_fourd_number(number)
    ibet, bet = parse_fourd_bet(bet_type)
    stake = float(stake)
    if not math.isfinite(stake) or stake < 0:
        raise ValueError(f"stake must be a positive amount, got {stake}")
    if ibet:
        prizes = ibet_table(rules, bet, permutations_count(num))
        covered = set(permutations(num))
        hits = [tier for tier, n in fourd_row_numbers(row) if n in covered]
    else:
        prizes = straight_table(rules, bet)
        hits = fourd_hits(num, row)
    counts = Counter(t for t in hits if prizes.get(t, 0.0) > 0)
    groups = {t: counts[t] for t in C.FOURD_TIERS if counts.get(t)}
    amount = round(sum(prizes[t] * n for t, n in groups.items()) * stake, 2)
    return PrizeResult(amount=amount, groups=groups, detail=fourd_detail(groups))


def fourd_prize_lookup(row: Any, rules: PrizeRules, bet: str = "big") -> np.ndarray:
    """(10000,) array: $1 straight prize of every number 0000 to 9999 in this draw."""
    ibet, bet = parse_fourd_bet(bet)
    if ibet:
        raise ValueError("the vectorised 4D prize handles Big and Small only")
    prizes = straight_table(rules, bet)
    table = np.zeros(C.FOURD_SPACE, dtype=np.float64)
    for tier, number in fourd_row_numbers(row):
        table[int(number)] += prizes.get(tier, 0.0)
    return table


def fourd_prize_vector(numbers: Any, row: Any, rules: PrizeRules, bet: str = "big") -> np.ndarray:
    """Vectorised Big / Small $1 prizes for int numbers 0..9999 (any array shape)."""
    arr = np.asarray(numbers, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= C.FOURD_SPACE):
        raise ValueError("4D numbers run from 0 to 9999")
    return fourd_prize_lookup(row, rules, bet)[arr]
