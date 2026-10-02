"""4D history analysis: digit spread per position, repeat winners, digit sets, fairness test
and the average return per $1 of each bet type.

Every function takes a fourd.csv frame (see ``models.FOURD_COLUMNS``) and counts all 23
winning numbers of each draw (1st, 2nd, 3rd, 10 Starter, 10 Consolation); blank cells are
skipped. Nothing here predicts a draw: every draw is independent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import constants as C
from . import prizes
from .models import FOURD_NUMBER_COLUMNS, PrizeRules

log = logging.getLogger(__name__)

DIGITS = tuple(range(10))
DOF_PER_POSITION = len(DIGITS) - 1  # 9
DOF_OVERALL = DOF_PER_POSITION * len(C.FOURD_POSITIONS)  # 36
# Chi square needs about 5 expected per cell: 10 digits per position -> 50 numbers.
MIN_NUMBERS_FOR_TEST = 5 * len(DIGITS)
SIGNIFICANCE = 0.05

_LONG_COLUMNS = ["draw_number", "draw_date", "tier", "number"]
_SLOT_ORDER = {col: i for i, col in enumerate(FOURD_NUMBER_COLUMNS)}


# Long form


def all_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """One row per winning number: draw_number, draw_date, tier, number (4 digit string).

    Ordered by draw number, then 1st, 2nd, 3rd, Starter 1 to 10, Consolation 1 to 10.
    Blank or unreadable cells are skipped.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame({
            "draw_number": pd.Series(dtype="int64"),
            "draw_date": pd.Series(dtype="datetime64[ns]"),
            "tier": pd.Series(dtype=object),
            "number": pd.Series(dtype=object),
        })
    slots = [c for c in FOURD_NUMBER_COLUMNS if c in df.columns]
    base = df[["draw_number", "draw_date"] + slots]
    long = base.melt(id_vars=["draw_number", "draw_date"], value_vars=slots,
                     var_name="slot", value_name="number")
    long = long.assign(number=long["number"].map(prizes.clean_fourd_number))
    long = long[long["number"] != ""].assign(
        tier=lambda d: d["slot"].map(prizes.FOURD_SLOT_TIER),
        order=lambda d: d["slot"].map(_SLOT_ORDER),
    )
    long = long.sort_values(["draw_number", "order"], kind="stable")
    out = long[_LONG_COLUMNS].reset_index(drop=True)
    out["number"] = out["number"].astype(object)
    return out


def _digit_matrix(numbers: pd.Series) -> np.ndarray:
    """(n, 4) int array of the digits of 4 digit strings."""
    if len(numbers) == 0:
        return np.zeros((0, len(C.FOURD_POSITIONS)), dtype=np.int64)
    raw = np.frombuffer("".join(numbers).encode("ascii"), dtype=np.uint8)
    return raw.reshape(-1, len(C.FOURD_POSITIONS)).astype(np.int64) - ord("0")


# Frequencies


def position_digit_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of each digit 0..9 (rows) in each position (columns thousands .. units)."""
    digits = _digit_matrix(all_numbers(df)["number"])
    data = {pos: np.bincount(digits[:, i], minlength=10).astype(np.int64)
            for i, pos in enumerate(C.FOURD_POSITIONS)}
    return pd.DataFrame(data, index=pd.Index(DIGITS, name="digit"))


def repeat_winners(df: pd.DataFrame, min_count: int = 2, k: int = 15) -> list[tuple[str, int, pd.Timestamp]]:
    """Numbers that won at least ``min_count`` times: (number, times won, last date won).

    Most wins first; ties go to the most recent win, then the lower number.
    """
    long = all_numbers(df)
    if long.empty:
        return []
    grouped = long.groupby("number", sort=False).agg(times=("number", "size"), last=("draw_date", "max"))
    grouped = grouped[grouped["times"] >= max(int(min_count), 1)].reset_index()
    grouped = grouped.sort_values(["times", "last", "number"], ascending=[False, False, True],
                                  na_position="last", kind="stable")
    return [(str(r.number), int(r.times), pd.Timestamp(r.last))
            for r in grouped.head(max(int(k), 0)).itertuples(index=False)]


def digit_set_freq(df: pd.DataFrame, k: int = 10) -> list[tuple[str, int]]:
    """Most frequent digit sets ignoring order ("1347" covers 3417, 7431, ...): (set, count).

    Sets with four different digits cover 24 numbers, so they naturally turn up more often
    than sets with repeated digits. Ties go to the lower digit string.
    """
    numbers = all_numbers(df)["number"]
    if numbers.empty:
        return []
    counts = numbers.map(lambda s: "".join(sorted(s))).value_counts()
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(str(s), int(n)) for s, n in ordered[: max(int(k), 0)]]


# Fairness test


def format_p(p: float) -> str:
    """p value for prose, formatted exactly like the TOTO verdict (shared rules)."""
    from .analysis_toto import format_p as _format_p

    return _format_p(p)


def chi_square_digits(df: pd.DataFrame) -> dict:
    """Chi square test of each position's digit counts against an even spread.

    per_position: one test per position (9 degrees of freedom). overall: the four statistics
    added up, compared with 36 degrees of freedom (the positions of a fair random number are
    independent). The verdict is one plain sentence on the overall result.
    """
    freq = position_digit_freq(df)
    n_numbers = int(freq.iloc[:, 0].sum())
    n_draws = int(len(df)) if df is not None else 0
    nan = float("nan")
    per_position: dict[str, dict[str, float]] = {}
    if n_numbers == 0:
        for pos in C.FOURD_POSITIONS:
            per_position[pos] = {"stat": nan, "dof": DOF_PER_POSITION, "p_value": nan}
        return {
            "per_position": per_position,
            "overall": {"stat": nan, "dof": DOF_OVERALL, "p_value": nan},
            "n_numbers": 0,
            "n_draws": n_draws,
            "verdict": "There are no 4D draws yet, so the spread of digits cannot be tested.",
        }

    for pos in C.FOURD_POSITIONS:
        stat, p = stats.chisquare(freq[pos].to_numpy(dtype=float))  # even spread expected
        per_position[pos] = {"stat": float(stat), "dof": DOF_PER_POSITION, "p_value": float(p)}
    total = float(sum(v["stat"] for v in per_position.values()))
    p_all = float(stats.chi2.sf(total, DOF_OVERALL))

    p_text = format_p(p_all)
    p_part = f"p {p_text}" if p_text.startswith(("above", "below")) else f"p = {p_text}"
    if n_numbers < MIN_NUMBERS_FOR_TEST:
        verdict = (f"Only {n_numbers} winning numbers so far, too few for a reliable test, "
                   "so hot and cold digits mean nothing yet.")
    elif p_all < SIGNIFICANCE:
        worst = min(per_position, key=lambda pos: per_position[pos]["p_value"])
        verdict = (f"The spread of digits is more uneven than chance usually gives ({p_part}, "
                   f"most uneven in the {worst} position), but this alone does not make any "
                   "digit more likely in the next draw.")
    else:
        verdict = (f"The spread of digits is consistent with pure chance ({p_part}), "
                   "so hot and cold digits are just noise.")
    return {
        "per_position": per_position,
        "overall": {"stat": total, "dof": DOF_OVERALL, "p_value": p_all},
        "n_numbers": n_numbers,
        "n_draws": n_draws,
        "verdict": verdict,
    }


# Bet type value


def _cents(x: float) -> str:
    return f"${x:,.2f}"


def _dollars(x: float) -> str:
    """Whole dollar amounts without cents ("$83"), others with cents."""
    return f"${x:,.0f}" if float(x).is_integer() else _cents(x)


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _span(values: dict[int, float]) -> str:
    """"$0.63 to $0.66", or a single amount when both ends round the same."""
    lo, hi = _cents(min(values.values())), _cents(max(values.values()))
    return lo if lo == hi else f"{lo} to {hi}"


def _ibet_note(rules: PrizeRules) -> str:
    """Plain sentence on where the iBet prizes came from."""
    official = [f"iBet {perms} {bet.title()}"
                for bet in ("big", "small") for perms in reversed(prizes.IBET_PERMUTATIONS)
                if prizes.ibet_is_published(rules, bet, perms)]
    total = 2 * len(prizes.IBET_PERMUTATIONS)
    rule = ("the straight prize divided by the number of permutations, rounded down to whole "
            "dollars")
    if len(official) == total:
        return "iBet prizes are taken from the official iBet prize table."
    if official:
        return (f"The {_and_list(official)} prizes are taken from the official prize table; "
                f"other iBet prizes are {rule}.")
    example = prizes.ibet_table(rules, "big", 24).get("first", 0.0)
    return (f"iBet prizes here are {rule}, because the official page did not give an iBet table "
            f"(for example iBet 24 Big pays {_dollars(example)} for a "
            "1st Prize hit).")


def bet_type_value(rules: PrizeRules) -> dict:
    """Average return per $1 for Big, Small and iBet (by permutation count) from the prize table.

    Returns {"Big": 0.659, "Small": 0.58, "iBet Big": {4: .., 6: .., 12: .., 24: ..},
    "iBet Small": {...}, "best": "Big", "explanation": str, "ibet_note": str}. "best" is the
    bet type with the highest return (iBet counted at its best permutation count; on a tie
    the straight bet wins).
    """
    big = prizes.fourd_expected_return(rules, "Big")
    small = prizes.fourd_expected_return(rules, "Small")
    ibet_big = {p: prizes.fourd_expected_return(rules, "iBet Big", p) for p in prizes.IBET_PERMUTATIONS}
    ibet_small = {p: prizes.fourd_expected_return(rules, "iBet Small", p) for p in prizes.IBET_PERMUTATIONS}

    candidates = [("Big", big), ("Small", small),
                  ("iBet Big", max(ibet_big.values())), ("iBet Small", max(ibet_small.values()))]
    best, best_value = candidates[0]
    for name, value in candidates[1:]:
        if value > best_value + 1e-12:
            best, best_value = name, value

    if best_value < 1:
        tail = "Every bet type returns less than the $1 it costs."
    else:
        tail = "Check the prize table, a return above $1 per $1 is unusual."
    explanation = (
        f"For every $1 staked, Big returns {_cents(big)} on average, Small {_cents(small)}, "
        f"iBet Big {_span(ibet_big)} and iBet Small {_span(ibet_small)} depending on how many "
        f"ways the digits can be arranged, so {best} gives the most back. {tail}"
    )
    return {
        "Big": big,
        "Small": small,
        "iBet Big": ibet_big,
        "iBet Small": ibet_small,
        "best": best,
        "explanation": explanation,
        "ibet_note": _ibet_note(rules),
    }
