"""TOTO prize rules: what a board or a ticket won in a given draw.

* ``toto_group`` / ``toto_group_counts`` say which prize group each board of a bet lands in.
* ``toto_ticket_prize`` is what a real ticket won, from the share amounts published for the
  draw (``tickets.settle_ledger`` uses it).
* ``toto_share_if_won`` / ``toto_group_amounts`` are what ONE extra winning board in a group
  would have received, for estimates of draws a ticket did not play.

Rows are ``pd.Series`` rows of the toto.csv frame (``models.TOTO_COLUMNS``) or plain dicts
with the same keys. Missing values (None, NaN) are allowed anywhere.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

import pandas as pd

from . import constants as C
from .models import PrizeResult, PrizeRules

log = logging.getLogger(__name__)

TOTO_GROUPS = tuple(range(1, 8))
_POOL_GROUPS = (2, 3, 4)  # groups paid as a percentage of the prize pool
_FIXED_GROUPS = (5, 6, 7)  # groups paid a fixed amount per board
_CASCADE_TYPES = ("cascade", "hongbao")  # draw types whose unwon jackpot cascades down

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

_SYSTEM_BET = re.compile(r"system\s*(\d+)")


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
    try:
        f = float(_get(row, key))
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
        return [int(p) for p in re.split(r"[\s,]+", numbers.strip()) if p]
    return [int(x) for x in numbers]


# Boards and groups


def toto_winning(row: Any) -> tuple[list[int], int]:
    """(six winning numbers ascending, additional number) of a toto.csv row."""
    winning = sorted(int(_get(row, f"n{i}", 0)) for i in range(1, C.TOTO_PICK + 1))
    return winning, int(_get(row, "additional", 0))


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


def toto_bet_size(bet_type: str) -> int:
    """Numbers a bet type takes: 6 for "Ordinary" ("Ord" or blank), n for "System n" (7 to 12).

    Case and spacing do not matter. ValueError for anything else.
    """
    s = str(bet_type or "").strip().lower()
    if s in ("", "ordinary", "ord"):
        return C.TOTO_PICK
    m = _SYSTEM_BET.fullmatch(s)
    if m and int(m.group(1)) in C.TOTO_SYSTEM_BOARDS:
        return int(m.group(1))
    raise ValueError(f"unknown TOTO bet type {bet_type!r}")


def toto_boards(numbers: Any) -> list[tuple[int, ...]]:
    """Every 6 number board a bet covers: 1 for an Ordinary bet, C(n, 6) for System n."""
    nums = sorted(_parse_toto_numbers(numbers))
    if len(set(nums)) != len(nums):
        raise ValueError(f"TOTO numbers must be distinct, got {nums}")
    if not C.TOTO_PICK <= len(nums) <= max(C.TOTO_SYSTEM_BOARDS):
        raise ValueError(f"a TOTO bet has 6 to {max(C.TOTO_SYSTEM_BOARDS)} numbers, got {len(nums)}")
    if nums[0] < 1 or nums[-1] > C.TOTO_MAX_NUMBER:
        raise ValueError(f"TOTO numbers run from 1 to {C.TOTO_MAX_NUMBER}, got {nums}")
    return list(combinations(nums, C.TOTO_PICK))


def toto_group_counts(numbers: Any, winning: Iterable[int], additional: int) -> dict[int, int]:
    """{group: boards} over every board of the bet, ascending by group; groups with no board
    are left out."""
    winning = list(winning)
    counts = Counter(toto_group(board, winning, additional) for board in toto_boards(numbers))
    counts.pop(None, None)
    return dict(sorted(counts.items()))


def toto_detail(groups: Mapping[int, int]) -> str:
    """Plain summary such as "Group 7 x3, Group 6 x1" (lower prizes first), or "No prize"."""
    parts = [f"Group {g} x{n}" for g, n in sorted(groups.items(), reverse=True) if n > 0]
    return ", ".join(parts) if parts else "No prize"


# Prize pool


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

    ``prev`` (optional) is the row of the previous draw. It is only used when its draw number
    is exactly one less; without it no snowball can be seen and none is assumed, as
    ``crowd_table`` does for a draw missing from the frame.
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


# What one more winning board would have received


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
        group_total = float(pct) * pool if pool else 0.0
        # Winners but no share amount captured: the group money is still split with them.
        return group_total / (winners + 1) if winners > 0 else group_total
    if group in _FIXED_GROUPS:
        return float(_rule(rules.fixed_prizes, group, C.TOTO_FIXED_PRIZES[group]))
    raise ValueError(f"TOTO prize groups are 1 to 7, got {group}")


def toto_share_if_won(group: int | None, row: Any, rules: PrizeRules) -> float:
    """What ONE extra winning board in ``group`` would have received in this draw.

    Group 1: winners > 0 -> g1_share x w / (w + 1), else the row's jackpot.
    Groups 2 to 4: winners > 0 -> share x w / (w + 1), else group percentage x pool estimate
    (0 when the pool cannot be estimated); winners > 0 with no share amount captured ->
    group percentage x pool estimate / (w + 1). Groups 5 to 7: the fixed prize. None -> 0.
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


# What a real ticket won


def _published_share(group: int, row: Any) -> float | None:
    """The share amount printed on the result page, when the page shows winners for it."""
    share = _num(row, f"g{group}_share")
    if share is None or share <= 0 or _int(row, f"g{group}_winners") <= 0:
        return None
    return share


def _share_as_published_winner(group: int, row: Any, rules: PrizeRules) -> float | None:
    """Per board amount for a real ticket when the page shows winners for Group 1 to 4 but
    no share amount: the ticket is one of those winners, so the group money is divided by the
    published number of winners (not one more). None when the page shows no winners, and for
    the fixed prize groups."""
    winners = _int(row, f"g{group}_winners")
    if winners <= 0:
        return None
    if group == 1:
        total = _num(row, "jackpot") or float(rules.min_group1) * winners
        return total / winners
    if group in _POOL_GROUPS:
        pool = toto_pool_estimate(row, rules)
        pct = _rule(rules.group_pool_pct, group, 0.0) or 0.0
        return float(pct) * pool / winners if pool else 0.0
    return None


def _ticket_share(group: int, row: Any, rules: PrizeRules) -> float:
    """Per board amount a real winning board in ``group`` earned (see ``toto_ticket_prize``)."""
    share = _published_share(group, row)
    if share is None:
        share = _share_as_published_winner(group, row, rules)
    if share is None:
        share = toto_share_if_won(group, row, rules)
        log.debug("draw %s: no published Group %d share, using %.2f",
                  _get(row, "draw_number"), group, share)
    return share


def _check_toto_bet(bet_type: str, count: int) -> None:
    """Raise ValueError if ``bet_type`` ("Ordinary", "System 7".."System 12") does not fit."""
    expected = toto_bet_size(bet_type)
    if count != expected:
        raise ValueError(f"{bet_type} needs {expected} numbers, got {count}")


def toto_ticket_prize(numbers: Any, bet_type: str, units: float, row: Any,
                      rules: PrizeRules) -> PrizeResult:
    """What a real ticket that played this draw won.

    Each winning board earns the share amount published for its group. If the page shows
    winners but no share amount, the group money is split between the published winners (the
    ticket is one of them). If the page shows no winner for that group (missing data),
    ``toto_share_if_won`` is used instead. The total is multiplied by ``units`` (the stake per
    board, cost / boards).
    """
    nums = _parse_toto_numbers(numbers)
    _check_toto_bet(bet_type, len(nums))
    winning, additional = toto_winning(row)
    groups = toto_group_counts(nums, winning, additional)
    amount = sum(_ticket_share(group, row, rules) * boards for group, boards in groups.items())
    return PrizeResult(amount=round(amount * float(units), 2), groups=groups,
                       detail=toto_detail(groups))
