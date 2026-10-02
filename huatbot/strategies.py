"""Number picking strategies for TOTO and 4D.

Each strategy turns a history frame into one suggestion plus a one line reason built from real
figures. A strategy only ever looks at the frame it is given, so the backtest can pass the draws
before a target draw and get exactly what the strategy would have said at the time. Picks are
deterministic for a given (history, seed): all randomness comes from ``numpy.random.default_rng``.

None of this changes the odds. Every draw is independent and every set of six numbers (or every
4D number) is as likely as any other. Hot, Overdue and Balanced only describe the past. Low Crowd
is the one idea with a real basis: it cannot win more often, but numbers other players buy less
mean fewer people to split a prize with when it does win.

TOTO strategies: Hot, Overdue, Balanced, Low Crowd (plus a System 7 built from a pick).
4D strategies: Hot Digits, Repeat Winner, Digit Set, Cold Digits, Random.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from . import constants as C
from . import prizes
from .analysis_toto import frequency, frequency_table, overdue, shape_stats
from .models import FOURD_NUMBER_COLUMNS, FourDPick, TotoPick
from .textfmt import fmt_date, fmt_num, pct, plural

log = logging.getLogger(__name__)

NUMBERS = np.arange(1, C.TOTO_MAX_NUMBER + 1)
_NUM_COLS = [f"n{i}" for i in range(1, C.TOTO_PICK + 1)]

# TOTO strategy settings
HOT_WINDOW = 50  # Hot ranks numbers by how often they came up in the last 50 draws
HOT_TIE_WINDOW = 100  # first tie break: the last 100 draws

SAMPLE_BATCH = 2_000  # random sets drawn per numpy batch
BALANCED_MAX_TRIES = 20_000

LOW_CROWD_CANDIDATES = 24  # Low Crowd picks from the 24 numbers with the lowest crowd score
LOW_CROWD_VALID_SAMPLES = 2_000  # best (least crowded) of up to this many valid sets
LOW_CROWD_MAX_TRIES = 20_000  # random sets tried per relaxation level
LOW_CROWD_ODD = (2, 4)  # odd count range of a Low Crowd set
LOW_CROWD_LOW = (2, 4)  # low (1 to 24) count range of a Low Crowd set

# Pattern rules: a set looks "patterned" (and is likely shared with many other players) when
# it breaks one of these. Used by low_crowd_set, pattern_problems and system7_from.
MAX_RUN = 2  # no 3 consecutive numbers
MIN_ABOVE_BIRTHDAY = 2  # at least 2 numbers above 31, so not all birthday numbers
MAX_SHARED_WITH_LAST = 1  # at most 1 number repeated from the last draw
MAX_SAME_LAST_DIGIT = 3  # at most 3 numbers ending in the same digit

TOTO_PICK_ORDER = ("Hot", "Overdue", "Balanced", "Low Crowd")

# 4D strategy settings
FOURD_RECENT_WINDOW = 100  # Hot Digits and Cold Digits look at the last 100 draws
FOURD_PICK_ORDER = ("Hot Digits", "Repeat Winner", "Digit Set", "Cold Digits", "Random")
_PLACE = np.array([1000, 100, 10, 1], dtype=np.int64)


# Shared helpers


def _sorted(hist: pd.DataFrame) -> pd.DataFrame:
    """Oldest draw first, so "last N draws" and "latest draw" are well defined."""
    if len(hist) and not hist["draw_number"].is_monotonic_increasing:
        hist = hist.sort_values("draw_number", kind="stable")
    return hist


def _numbers_of(value: Any) -> list[int] | None:
    """Numbers from a list of ints, a toto.csv row (n1..n6) or None, as a sorted list."""
    if value is None:
        return None
    if isinstance(value, (pd.Series, dict)):
        try:
            return sorted(int(value[c]) for c in _NUM_COLS)
        except (KeyError, TypeError, ValueError):
            pass
    if isinstance(value, str):
        return sorted({int(x) for x in value.split()})
    try:
        return sorted({int(x) for x in value})
    except (TypeError, ValueError):
        return None


def _last_draw_numbers(hist: pd.DataFrame) -> list[int] | None:
    """The six winning numbers of the latest draw in ``hist`` (None when empty)."""
    if len(hist) == 0:
        return None
    return _numbers_of(hist.iloc[-1])


def _score_array(crowd_scores: Any) -> tuple[np.ndarray, bool]:
    """Crowd score per number as an array (index 0 is number 1) and whether any score is set.

    Accepts a Series or dict keyed by number (string keys from a JSON cache work too). Missing
    or non finite scores count as zero. ``available`` is False when every score is zero.
    """
    arr = np.zeros(C.TOTO_MAX_NUMBER, dtype=float)
    if crowd_scores is None:
        return arr, False
    s = crowd_scores if isinstance(crowd_scores, pd.Series) else pd.Series(crowd_scores, dtype=float)
    if len(s) == 0:
        return arr, False
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_numeric(pd.Index(s.index), errors="coerce")
    s = s[s.index.notna()]
    s = s[~s.index.duplicated(keep="last")]
    s = s.reindex(NUMBERS.astype(float))
    arr = np.array(s.to_numpy(dtype=float), dtype=float)  # writable copy
    arr[~np.isfinite(arr)] = 0.0
    return arr, bool(np.any(np.abs(arr) > 1e-12))


def crowd_scores_available(crowd_scores: Any) -> bool:
    """True when ``crowd_scores`` holds at least one non zero score."""
    return _score_array(crowd_scores)[1]


def _sum_bounds(shape: dict) -> tuple[int, int]:
    """Whole number sum range [q25, q75] (sums are integers, so round inwards)."""
    q25, q75 = float(shape["sum_q25"]), float(shape["sum_q75"])
    lo, hi = math.ceil(q25 - 1e-9), math.floor(q75 + 1e-9)
    if lo > hi:  # quartiles between two integers: keep the nearest integers either side
        lo, hi = math.floor(q25), math.ceil(q75)
    return lo, hi


def _spaced(nums: Iterable[int]) -> str:
    return " ".join(str(int(n)) for n in nums)


def _range_text(lo: float, hi: float, unit: str) -> str:
    """ "9 to 12 times" or "9 times" when both ends match."""
    lo_i, hi_i = int(lo), int(hi)
    if lo_i == hi_i:
        return plural(hi_i, unit)
    return f"{fmt_num(lo_i)} to {plural(hi_i, unit)}"


class _Sampler:
    """Random 6 number sets drawn from ``pool``, generated lazily in batches and cached.

    The cache lets a relaxed constraint level look at the very same samples as the strict level
    before it, so the outcome depends only on (pool, rng state) and is fully deterministic.
    """

    def __init__(self, rng: np.random.Generator, pool: np.ndarray, max_tries: int,
                 batch: int = SAMPLE_BATCH) -> None:
        self.rng = rng
        self.pool = np.asarray(pool, dtype=np.int64)
        self.max_tries = int(max_tries)
        self.batch = int(batch)
        self._cache: list[np.ndarray] = []
        self._drawn = 0

    def _new_batch(self) -> np.ndarray:
        size = min(self.batch, self.max_tries - self._drawn)
        keys = self.rng.random((size, len(self.pool)))
        # The 6 smallest random keys per row pick a uniformly random 6 number subset.
        idx = np.argpartition(keys, C.TOTO_PICK - 1, axis=1)[:, : C.TOTO_PICK]
        sets = np.sort(self.pool[idx], axis=1)
        self._drawn += size
        self._cache.append(sets)
        return sets

    def batches(self) -> Iterator[np.ndarray]:
        yield from list(self._cache)
        while self._drawn < self.max_tries:
            yield self._new_batch()


def _rule_masks(sets: np.ndarray, last: list[int] | None, sum_lo: int, sum_hi: int,
                odd_range: tuple[int, int], low_range: tuple[int, int]) -> dict[str, np.ndarray]:
    """Which rows of ``sets`` (k numbers each, ascending) pass each named rule.

    Balance rules: "balance" (odd and low counts in range) and "sum" (sum in range).
    Pattern rules (the same ones ``pattern_problems`` explains): "consecutive", "birthday",
    "last_draw", "progression", "last_digit".
    """
    odd = (sets % 2 == 1).sum(axis=1)
    low = (sets <= C.TOTO_LOW_MAX).sum(axis=1)
    total = sets.sum(axis=1)
    gaps = np.diff(sets, axis=1)
    ones = np.ones(len(sets), dtype=bool)

    run3 = ((gaps[:, :-1] == 1) & (gaps[:, 1:] == 1)).any(axis=1) if gaps.shape[1] >= 2 else ~ones
    last_digits = sets % 10
    digit_counts = (last_digits[:, :, None] == np.arange(10)).sum(axis=1)
    if last:
        shared_ok = np.isin(sets, np.asarray(last)).sum(axis=1) <= MAX_SHARED_WITH_LAST
    else:
        shared_ok = ones
    return {
        "balance": (odd >= odd_range[0]) & (odd <= odd_range[1])
        & (low >= low_range[0]) & (low <= low_range[1]),
        "sum": (total >= sum_lo) & (total <= sum_hi),
        "consecutive": ~run3,
        "birthday": (sets > C.TOTO_BIRTHDAY_MAX).sum(axis=1) >= MIN_ABOVE_BIRTHDAY,
        "last_draw": shared_ok,
        "progression": ~(gaps == gaps[:, :1]).all(axis=1),
        "last_digit": digit_counts.max(axis=1) <= MAX_SAME_LAST_DIGIT,
    }


def _combine(masks: dict[str, np.ndarray], rules: Iterable[str], n: int) -> np.ndarray:
    ok = np.ones(n, dtype=bool)
    for rule in rules:
        ok &= masks[rule]
    return ok


def _shape_text(numbers: list[int]) -> str:
    """ "3 odd, 2 low (1 to 24), sum 160"."""
    odd = sum(n % 2 for n in numbers)
    low = sum(n <= C.TOTO_LOW_MAX for n in numbers)
    return f"{odd} odd, {low} low (1 to {C.TOTO_LOW_MAX}), sum {sum(numbers)}"


# TOTO strategies


def hot_set(hist: pd.DataFrame) -> TotoPick:
    """The 6 numbers drawn most often in the last 50 draws.

    Ties go to the count over the last 100 draws, then all history, then the lower number.
    """
    hist = _sorted(hist)
    table = frequency_table(hist)
    nums = table.index.to_numpy()
    # lexsort sorts by the last key first.
    order = np.lexsort((nums, -table["all"].to_numpy(), -table["last100"].to_numpy(),
                        -table["last50"].to_numpy()))
    chosen = sorted(int(n) for n in nums[order[: C.TOTO_PICK]])

    window = min(HOT_WINDOW, len(hist))
    if window == 0:
        reason = "No draw history yet, so no number is hot; these are simply the lowest numbers"
    else:
        counts = table.loc[chosen, "last50"]
        average = float(table["last50"].mean())
        reason = (f"Drawn {_range_text(counts.min(), counts.max(), 'time')} each in the last "
                  f"{plural(window, 'draw')}, against an average of {average:.1f} per number")
    return TotoPick(name="Hot", numbers=chosen, reason=reason)


def overdue_set(hist: pd.DataFrame) -> TotoPick:
    """The 6 numbers with the longest gap since they were last drawn.

    Ties go to the lower all time frequency, then the lower number.
    """
    hist = _sorted(hist)
    gaps = overdue(hist)
    freq_all = frequency(hist)
    nums = gaps.index.to_numpy()
    order = np.lexsort((nums, freq_all.to_numpy(), -gaps.to_numpy()))
    chosen = sorted(int(n) for n in nums[order[: C.TOTO_PICK]])

    if len(hist) == 0:
        reason = "No draw history yet, so no number is overdue; these are simply the lowest numbers"
    else:
        g = gaps.loc[chosen]
        typical_gap = C.TOTO_MAX_NUMBER / C.TOTO_PICK
        reason = (f"Not drawn for {_range_text(g.min(), g.max(), 'draw')} each, while a number "
                  f"turns up about once every {typical_gap:.1f} draws on average")
    return TotoPick(name="Overdue", numbers=chosen, reason=reason)


# Relaxation order for balanced_set: (what was relaxed, odd slack, low slack, use sum range).
_BALANCED_LEVELS: tuple[tuple[str, int, int, bool], ...] = (
    ("", 0, 0, True),
    ("sum range dropped", 0, 0, False),
    ("odd and low counts allowed 1 away from the usual, sum range dropped", 1, 1, False),
    ("no usual shape could be met", 6, 6, False),
)


def balanced_set(hist: pd.DataFrame, seed: int) -> TotoPick:
    """A random set with the usual shape of a winning set.

    Odd count == the most common odd count, low (1 to 24) count == the most common low count and
    the sum inside the middle 50% of past sums [q25, q75]. Random sets come from
    ``default_rng(seed)``; the first of up to 20,000 that fits wins. If none fits, the rules are
    relaxed in this order: drop the sum range, then allow odd and low counts 1 away from the
    usual, then take the first random set.
    """
    hist = _sorted(hist)
    shape = shape_stats(hist)
    t_odd, t_low = int(shape["typical_odd"]), int(shape["typical_low"])
    sum_lo, sum_hi = _sum_bounds(shape)
    sampler = _Sampler(np.random.default_rng(seed), NUMBERS, BALANCED_MAX_TRIES)

    chosen: list[int] | None = None
    relaxed = ""
    for relaxed, odd_slack, low_slack, use_sum in _BALANCED_LEVELS:
        rules = ["balance"] + (["sum"] if use_sum else [])
        for batch in sampler.batches():
            masks = _rule_masks(batch, None, sum_lo, sum_hi,
                                (t_odd - odd_slack, t_odd + odd_slack),
                                (t_low - low_slack, t_low + low_slack))
            ok = _combine(masks, rules, len(batch))
            if ok.any():
                chosen = [int(n) for n in batch[int(np.argmax(ok))]]
                break
        if chosen is not None:
            break
    assert chosen is not None  # the last level accepts any set

    odd = sum(n % 2 for n in chosen)
    low = sum(n <= C.TOTO_LOW_MAX for n in chosen)
    if shape["n_draws"] == 0:
        reason = (f"No draw history yet, so this follows the theoretical shape: {_shape_text(chosen)}, "
                  f"inside the middle half sum range {sum_lo} to {sum_hi}")
    else:
        lead = "Close to the usual shape" if relaxed else "Matches the usual shape"
        reason = (f"{lead}: {odd} odd ({pct(shape['odd_dist'].get(odd, 0.0), 0)} of draws), "
                  f"{low} low, 1 to {C.TOTO_LOW_MAX} ({pct(shape['low_dist'].get(low, 0.0), 0)} of draws), "
                  f"sum {sum(chosen)} against the middle half range {sum_lo} to {sum_hi}")
    if relaxed:
        reason += f"; rules relaxed: {relaxed}"
    return TotoPick(name="Balanced", numbers=sorted(chosen), reason=reason)


_BALANCE_RULES = ("balance", "sum")
_PATTERN_RULES = ("consecutive", "birthday", "last_draw", "progression", "last_digit")
_ALL_RULES = _BALANCE_RULES + _PATTERN_RULES

# Relaxation order for low_crowd_set: (use all 49 numbers, rules kept, what was dropped).
_LOW_CROWD_LEVELS: tuple[tuple[bool, tuple[str, ...], str], ...] = (
    (False, _ALL_RULES, ""),
    (True, _ALL_RULES, ""),
    (True, ("balance",) + _PATTERN_RULES, "sum range dropped"),
    (True, _PATTERN_RULES, "sum range and odd/low balance dropped"),
    (True, ("consecutive", "birthday", "progression"),
     "only the sequence, birthday and spacing rules kept"),
    (True, (), "no rule could be met"),
)


def _crowd_effect(total: float) -> str:
    """How a set's total crowd score changes the number of people sharing a prize.

    The crowd score is fitted on the crowd ratio (actual / expected Group 7 winners), so the sum
    of a set's six scores is roughly the change in that ratio if the set were drawn.
    """
    if abs(total) < 0.005:
        return "about as many people would share a prize as with an average set"
    word = "fewer" if total < 0 else "more"
    return f"roughly {pct(abs(total), 0)} {word} people would share a prize than with an average set"


def low_crowd_set(hist: pd.DataFrame, crowd_scores: pd.Series | None, seed: int) -> TotoPick:
    """A balanced, pattern free set built from numbers other players buy least.

    Candidates: the 24 numbers with the lowest crowd score (all 49 when scores are missing or
    all zero). Rules: odd count 2 to 4, low count 2 to 4, sum in [q25, q75], no 3 consecutive
    numbers, at least 2 numbers above 31, at most 1 number shared with the last draw in
    ``hist``, not an arithmetic progression, at most 3 numbers with the same last digit. Random
    sets come from ``default_rng(seed)``; among up to 2,000 valid ones (from up to 20,000
    tried) the set with the lowest total crowd score wins (ties: the first one found).

    If nothing qualifies, rules are relaxed in this order (see ``_LOW_CROWD_LEVELS``): use all 49
    numbers, drop the sum range, drop the odd/low balance, drop the last digit and last draw
    rules, and finally take any set.
    """
    hist = _sorted(hist)
    scores, available = _score_array(crowd_scores)
    shape = shape_stats(hist)
    sum_lo, sum_hi = _sum_bounds(shape)
    last = _last_draw_numbers(hist)
    rng = np.random.default_rng(seed)

    if available:
        order = np.lexsort((NUMBERS, scores))  # lowest score first, ties lower number
        candidates = np.sort(NUMBERS[order[:LOW_CROWD_CANDIDATES]])
    else:
        candidates = NUMBERS
    samplers: dict[bool, _Sampler] = {}

    def sampler_for(all_numbers: bool) -> _Sampler:
        key = all_numbers or not available  # without scores both pools are all 49 numbers
        if key not in samplers:
            pool = NUMBERS if key else candidates
            samplers[key] = _Sampler(rng, pool, LOW_CROWD_MAX_TRIES)
        return samplers[key]

    chosen: list[int] | None = None
    relaxed = ""
    for level, (all_numbers, rules, dropped) in enumerate(_LOW_CROWD_LEVELS):
        if level == 1 and not available:
            continue  # without scores level 0 already uses all 49 numbers
        valid: list[np.ndarray] = []
        found = 0
        for batch in sampler_for(all_numbers).batches():
            masks = _rule_masks(batch, last, sum_lo, sum_hi, LOW_CROWD_ODD, LOW_CROWD_LOW)
            ok_rows = batch[_combine(masks, rules, len(batch))]
            if len(ok_rows):
                valid.append(ok_rows[: LOW_CROWD_VALID_SAMPLES - found])
                found += len(valid[-1])
            if found >= LOW_CROWD_VALID_SAMPLES:
                break
        if found:
            pool = np.concatenate(valid)
            totals = scores[pool - 1].sum(axis=1)
            chosen = [int(n) for n in pool[int(np.argmin(totals))]]
            parts = [f"picked from all {C.TOTO_MAX_NUMBER} numbers"] if all_numbers and available else []
            relaxed = ", ".join(parts + ([dropped] if dropped else []))
            break
    assert chosen is not None  # the last level accepts any set

    problems = pattern_problems(chosen, last)
    if available:
        total = float(scores[np.asarray(chosen) - 1].sum())
        source = (f"from the {LOW_CROWD_CANDIDATES} numbers other players buy least"
                  if not relaxed else f"of all {C.TOTO_MAX_NUMBER} numbers")
        pattern = f"but {problems[0]}" if problems else "no obvious pattern"
        reason = (f"Least crowded mix {source}: crowd score {fmt_num(total, 2)}, so "
                  f"{_crowd_effect(total)}; {_shape_text(chosen)}, {pattern}")
    elif not problems:
        reason = (f"Crowd scores not available, so this is a pattern free balanced set: "
                  f"{_shape_text(chosen)}, no runs of 3, at least {MIN_ABOVE_BIRTHDAY} numbers above "
                  f"{C.TOTO_BIRTHDAY_MAX}, at most {MAX_SHARED_WITH_LAST} from the last draw")
    else:
        reason = (f"Crowd scores not available, so this is a balanced set: {_shape_text(chosen)}, "
                  f"but {problems[0]}")
    if relaxed:
        reason += f"; rules relaxed: {relaxed}"
    return TotoPick(name="Low Crowd", numbers=sorted(chosen), reason=reason)


def toto_picks(hist: pd.DataFrame, crowd_scores: pd.Series | None, seed: int) -> list[TotoPick]:
    """The four TOTO suggestions in display order: Hot, Overdue, Balanced, Low Crowd."""
    return [
        hot_set(hist),
        overdue_set(hist),
        balanced_set(hist, seed),
        low_crowd_set(hist, crowd_scores, seed),
    ]


def pattern_problems(numbers: Iterable[int], last_draw: Any = None) -> list[str]:
    """Plain reasons a set of numbers looks patterned (empty list when it looks random).

    Checks: 3 or more consecutive numbers, fewer than 2 numbers above 31 (birthday picks), more
    than 1 number repeated from ``last_draw`` (a list of numbers or a toto.csv row), evenly
    spaced numbers, more than 3 numbers ending in the same digit.
    """
    nums = sorted({int(n) for n in numbers})
    problems: list[str] = []

    # Runs of consecutive numbers.
    run = [nums[0]] if nums else []
    runs: list[list[int]] = []
    for n in nums[1:]:
        if n == run[-1] + 1:
            run.append(n)
        else:
            runs.append(run)
            run = [n]
    if run:
        runs.append(run)
    for r in runs:
        if len(r) > MAX_RUN:
            problems.append(f"{len(r)} numbers in a row ({_spaced(r)})")

    above = sum(n > C.TOTO_BIRTHDAY_MAX for n in nums)
    if above < MIN_ABOVE_BIRTHDAY:
        if above == 0:
            problems.append(f"every number is {C.TOTO_BIRTHDAY_MAX} or below, like birthday picks")
        else:
            problems.append(f"only {above} number above {C.TOTO_BIRTHDAY_MAX}, so it looks like "
                            "birthday picks")

    last = _numbers_of(last_draw)
    if last:
        shared = sorted(set(nums) & set(last))
        if len(shared) > MAX_SHARED_WITH_LAST:
            problems.append(f"repeats {len(shared)} numbers from the last draw ({_spaced(shared)})")

    if len(nums) >= 3:
        steps = {b - a for a, b in zip(nums, nums[1:])}
        if len(steps) == 1:
            problems.append(f"evenly spaced numbers (every {steps.pop()})")

    for digit, count in sorted(Counter(n % 10 for n in nums).items()):
        if count > MAX_SAME_LAST_DIGIT:
            problems.append(f"{count} numbers end in {digit}")
    return problems


def system7_from(pick: TotoPick, crowd_scores: pd.Series | None, last_draw: Any = None) -> list[int]:
    """Seven numbers for a System 7: the pick's six plus one more.

    The extra number is the least crowded number not in the pick (ties: lower number) that keeps
    ``pattern_problems`` empty for all seven. If no number manages that, the least crowded one
    with the fewest problems is used.
    """
    base = sorted({int(n) for n in pick.numbers})
    if len(base) != C.TOTO_PICK:
        raise ValueError(f"a System 7 is built from 6 different numbers, got {pick.numbers}")
    scores, _ = _score_array(crowd_scores)
    others = [int(n) for n in NUMBERS if int(n) not in base]
    others.sort(key=lambda n: (scores[n - 1], n))

    best, best_count = others[0], None
    for n in others:
        count = len(pattern_problems(base + [n], last_draw))
        if count == 0:
            return sorted(base + [n])
        if best_count is None or count < best_count:
            best, best_count = n, count
    return sorted(base + [best])


# 4D strategies


def _fourd_values(hist: pd.DataFrame) -> np.ndarray:
    """(n_draws, 23) int array of the winning numbers, columns as FOURD_NUMBER_COLUMNS; blank = -1."""
    cols = [c for c in FOURD_NUMBER_COLUMNS if c in hist.columns]
    if len(hist) == 0 or not cols:
        return np.full((len(hist), len(cols)), -1, dtype=np.int64)
    raw = hist[cols].to_numpy(dtype=object)
    text = np.ascontiguousarray(raw.astype(str))
    # Fast path for normalised frames, where every cell is a 4 character ASCII digit string:
    # read the 4 code points of each cell directly (shorter strings are padded with code 0).
    codes = np.ascontiguousarray(text.astype("U4")).view(np.uint32).reshape(*text.shape, 4).astype(np.int64) - ord("0")
    ok = (np.char.str_len(text) == 4) & ((codes >= 0) & (codes <= 9)).all(axis=-1)
    vals = np.where(ok, codes @ _PLACE, -1).astype(np.int64)
    # Anything else (ints, "42.0", blanks) goes through the shared cleaner.
    for i, j in zip(*np.nonzero(~ok)):
        cleaned = prizes.clean_fourd_number(raw[i, j])
        if cleaned:
            vals[i, j] = int(cleaned)
    return vals


def _digits(values: np.ndarray) -> np.ndarray:
    """(n, 4) digit matrix of int numbers 0..9999 (thousands first)."""
    return (values[:, None] // _PLACE) % 10


def _position_counts(values: np.ndarray) -> np.ndarray:
    """(10, 4) counts of each digit per position over the valid numbers in ``values``."""
    flat = values[values >= 0]
    digits = _digits(flat)
    return np.stack([np.bincount(digits[:, p], minlength=10) for p in range(len(_PLACE))], axis=1)


def _digit_pick(name: str, counts: np.ndarray, n_draws: int, most: bool) -> FourDPick:
    """Hot Digits (most common digit per position) or Cold Digits (least common), Big."""
    pick = counts.argmax(axis=0) if most else counts.argmin(axis=0)  # ties: lower digit
    number = "".join(str(int(d)) for d in pick)
    total = int(counts[:, 0].sum())
    if total == 0:
        reason = "No 4D history yet, so every digit count is zero and this is simply 0000"
    else:
        chosen = counts[pick, np.arange(counts.shape[1])]
        word = "Most" if most else "Least"
        reason = (f"{word} common digit in each position over the last {plural(n_draws, 'draw')}, "
                  f"seen {_range_text(chosen.min(), chosen.max(), 'time')} each against an "
                  f"average of {fmt_num(total / 10)}")
    return FourDPick(name=name, number=number, bet_type="Big", reason=reason)


def _repeat_pick(hist: pd.DataFrame, values: np.ndarray, fallback: str) -> FourDPick:
    """The number that won most often (ties: most recent win, then lower number), Big.

    Fallback when no number won twice: the latest 1st Prize; with no history at all, ``fallback``.
    """
    n_draws = len(values)
    valid = values >= 0
    nums = values[valid]
    if len(nums):
        counts = np.bincount(nums, minlength=C.FOURD_SPACE)
        best = int(counts.max())
    else:
        counts, best = None, 0

    if counts is not None and best >= 2:
        draw_idx = np.broadcast_to(np.arange(n_draws)[:, None], values.shape)[valid]
        last_seen = np.full(C.FOURD_SPACE, -1, dtype=np.int64)
        np.maximum.at(last_seen, nums, draw_idx)
        tied = np.flatnonzero(counts == best)
        # Most recent win first, then the lower number.
        winner = int(tied[np.lexsort((tied, -last_seen[tied]))[0]])
        when = fmt_date(hist["draw_date"].iloc[int(last_seen[winner])])
        if len(tied) > 1:
            lead = (f"Won {plural(best, 'time')} in the last {plural(n_draws, 'draw')}, tied with "
                    f"{plural(len(tied) - 1, 'other number')} for the most wins")
        else:
            lead = (f"Won {plural(best, 'time')} in the last {plural(n_draws, 'draw')}, "
                    "more than any other number")
        reason = f"{lead}; latest win {when}"
        return FourDPick(name="Repeat Winner", number=f"{winner:04d}", bet_type="Big", reason=reason)

    first_col = FOURD_NUMBER_COLUMNS.index("first")
    if n_draws and values.shape[1] > first_col:
        firsts = np.flatnonzero(values[:, first_col] >= 0)
        if len(firsts):
            i = int(firsts[-1])
            reason = (f"No number has won twice in the last {plural(n_draws, 'draw')}, so this is "
                      f"the latest 1st Prize ({fmt_date(hist['draw_date'].iloc[i])})")
            return FourDPick(name="Repeat Winner", number=f"{int(values[i, first_col]):04d}",
                             bet_type="Big", reason=reason)
    reason = "No 4D history yet, so this is a random number"
    return FourDPick(name="Repeat Winner", number=fallback, bet_type="Big", reason=reason)


def _digit_set_pick(values: np.ndarray, fallback: str) -> FourDPick:
    """The most frequent digit set ignoring order (ties: lower digit string).

    Bet type iBet Big so the stake covers every order of the digits; Big when the digits have
    only one order (e.g. 7777).
    """
    n_draws = len(values)
    nums = values[values >= 0]
    if len(nums):
        keys = (np.sort(_digits(nums), axis=1) * _PLACE).sum(axis=1)
        counts = np.bincount(keys, minlength=C.FOURD_SPACE)
        key = int(np.argmax(counts))
        number, count = f"{key:04d}", int(counts[key])
        tied = int((counts == count).sum()) > 1
    else:
        number, count, tied = "".join(sorted(fallback)), 0, False

    perms = prizes.permutations_count(number)
    bet = "iBet Big" if perms > 1 else "Big"
    if count == 0:
        reason = "No 4D history yet, so these are the digits of a random number"
    else:
        rank = "tied for the most" if tied else "the most"
        reason = (f"Digits {_spaced(number)} in any order won {plural(count, 'time')} in the last "
                  f"{plural(n_draws, 'draw')}, {rank} of any digit set")
    if perms > 1:
        reason += f"; iBet Big covers all {perms} orders for the same stake"
    else:
        reason += "; its digits have only one order, so a plain Big bet"
    return FourDPick(name="Digit Set", number=number, bet_type=bet, reason=reason)


def _random_pick(number: str) -> FourDPick:
    reason = (f"A random number for comparison: like every number it has "
              f"{C.FOURD_NUMBERS_PER_DRAW} chances in {fmt_num(C.FOURD_SPACE)} of some prize")
    return FourDPick(name="Random", number=number, bet_type="Big", reason=reason)


def _fourd_build(hist: pd.DataFrame, seed: int) -> list[FourDPick]:
    hist = _sorted(hist)
    values = _fourd_values(hist)
    rng = np.random.default_rng(seed)
    random_number = f"{int(rng.integers(0, C.FOURD_SPACE)):04d}"
    spare = f"{int(rng.integers(0, C.FOURD_SPACE)):04d}"  # stand in when there is no history

    recent = values[-FOURD_RECENT_WINDOW:]
    counts = _position_counts(recent)
    return [
        _digit_pick("Hot Digits", counts, len(recent), most=True),
        _repeat_pick(hist, values, spare),
        _digit_set_pick(values, spare),
        _digit_pick("Cold Digits", counts, len(recent), most=False),
        _random_pick(random_number),
    ]


def fourd_strategy_numbers(hist: pd.DataFrame, seed: int) -> dict[str, tuple[str, str]]:
    """Strategy name -> (4 digit number, bet type), in FOURD_PICK_ORDER.

    "Hot Digits": most frequent digit per position over the last 100 draws, Big.
    "Repeat Winner": the number won most often (ties: most recent), Big; fallback latest 1st Prize.
    "Digit Set": most frequent sorted digit set, iBet Big (Big if it has 1 permutation).
    "Cold Digits": least frequent digit per position over the last 100 draws, Big.
    "Random": ``default_rng(seed)``, Big.
    """
    return {p.name: (p.number, p.bet_type) for p in _fourd_build(hist, seed)}


def fourd_picks(hist: pd.DataFrame, seed: int) -> list[FourDPick]:
    """The five 4D suggestions with one line reasons, in FOURD_PICK_ORDER."""
    return _fourd_build(hist, seed)

