"""Backtests: how each strategy would have done on past draws, compared with random players.

For every tested draw the strategy picks are rebuilt from the draws before it only (no look
ahead), then scored against the real result with the real prize amounts of that draw:

* TOTO: each set is scored as one extra $1 board with ``prizes.toto_prize_matrix`` (the share
  amount one more winner would have received). Crowd scores for Low Crowd are fitted on the
  earlier draws' ``crowd_table`` rows only.
* 4D: each pick stakes $1, Big with ``prizes.fourd_prize_vector`` and iBet Big (Digit Set) with
  ``prizes.fourd_ticket_prize``.

The comparison is a crowd of random players. Random player ``i`` buys random set ``i`` of every
draw (TOTO: one random board, 4D: one random Big $1 number), so after the test window there is a
distribution of ``n_random`` player totals. A strategy's ``percentile_vs_random`` is the mid rank
of its total in that distribution (ties count half), and the "Random" row reports the average
player. None of this changes the odds: every draw is independent, so a strategy that did well in
the sample is not expected to keep doing so.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import analysis_toto, prizes, strategies
from . import constants as C
from .models import BacktestResult, PrizeRules, StrategyScore
from .textfmt import fmt_num, money, per_dollar, plural

log = logging.getLogger(__name__)

RANDOM_NAME = "Random"  # the baseline row: the average random player
RANDOM_VERDICT = "Baseline"
STAKE = 1.0  # dollars per strategy per draw (one TOTO board, or one 4D $1 bet)

TOTO_STRATEGIES: tuple[str, ...] = tuple(strategies.TOTO_PICK_ORDER)  # Hot, Overdue, Balanced, Low Crowd
# The 4D "Random" pick is itself a random Big number, exactly what the random players buy, so
# its row is the Random baseline (one row per name keeps lookups by pick name unambiguous).
FOURD_STRATEGIES: tuple[str, ...] = tuple(n for n in strategies.FOURD_PICK_ORDER if n != RANDOM_NAME)

# verdict_for bands (percent of random players beaten).
WORSE_BELOW = 5.0
BETTER_ABOVE = 95.0

RESULT_FORMAT = 1  # bump when result_to_dict changes shape, so old JSON caches are ignored
_EPS = 1e-6  # totals closer than this count as a tie (sums of float prize amounts)
SKEW_RATIO = 1.25  # explain the Random average when it is more than 1.25 x the median player


# Verdicts and percentiles


def verdict_for(percentile: float | None) -> str:
    """One plain line for a strategy's place among the random players.

    5 to 95: "No better than random (beat N% of random players)"; above 95: "Beat random in this
    sample (top N%), but every draw is independent so do not expect it to last"; below 5: "Worse
    than random in this sample". None (no random players to compare with) says so.
    """
    if percentile is None:
        return "Not compared with random players"
    try:
        p = float(percentile)
    except (TypeError, ValueError):
        return "Not compared with random players"
    if not math.isfinite(p):
        return "Not compared with random players"
    if p > BETTER_ABOVE:
        top = max(1, math.ceil(100.0 - p - 1e-9))  # 97.3 -> top 3%, 99.99 -> top 1%
        return (f"Beat random in this sample (top {top}%), but every draw is independent "
                "so do not expect it to last")
    if p < WORSE_BELOW:
        return "Worse than random in this sample"
    return f"No better than random (beat {int(math.floor(p + 0.5))}% of random players)"


def percentile_vs_random(total: float, random_totals: Any) -> float | None:
    """Mid rank of ``total`` among the random players' totals, 0 to 100 (None if there are none).

    Players below count fully, players with the same total count half, so a strategy that ties
    with every random player sits at 50.
    """
    totals = np.asarray(random_totals, dtype=float).ravel()
    if totals.size == 0:
        return None
    below = int(np.count_nonzero(totals < total - _EPS))
    equal = int(np.count_nonzero(np.abs(totals - total) <= _EPS))
    return 100.0 * (below + 0.5 * equal) / totals.size


# Shared plumbing


def _sorted_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Oldest draw first with a clean 0..n index, so position i means "the i th draw"."""
    if len(df) and not df["draw_number"].is_monotonic_increasing:
        df = df.sort_values("draw_number", kind="stable")
    return df.reset_index(drop=True)


def _verb(n: int) -> str:
    return "has" if n == 1 else "have"


def _test_window(n_rows: int, n_draws: int, min_history: int) -> tuple[int, int, list[str]]:
    """(draws to test, history each one needs, notes). Shrinks the window when history is short.

    Every tested draw needs ``min_history`` earlier draws, so at most ``n_rows - min_history``
    draws can be tested.
    """
    n_draws = max(int(n_draws), 0)
    min_history = max(int(min_history), 0)
    available = max(n_rows - min_history, 0)
    n_test = min(n_draws, available)
    notes: list[str] = []
    if n_draws == 0:
        notes.append("Backtest skipped: no draws were asked for.")
    elif n_test == 0:
        notes.append(f"Not enough history to backtest: {plural(n_rows, 'draw')} stored, and each "
                     f"tested draw needs {plural(min_history, 'earlier draw')}.")
    elif n_test < n_draws:
        notes.append(f"History is short: only {plural(available, 'draw')} {_verb(available)} at least "
                     f"{plural(min_history, 'earlier draw')}, so the backtest covers "
                     f"{plural(n_test, 'draw')} instead of {fmt_num(n_draws)}.")
    return n_test, min_history, notes


def _round_pct(p: float | None) -> float | None:
    return None if p is None else round(float(p), 1)


def _strategy_score(name: str, per_draw: np.ndarray, random_totals: np.ndarray) -> StrategyScore:
    """Score row for one strategy from its prize in every tested draw ($1 per draw)."""
    per_draw = np.asarray(per_draw, dtype=float)
    n = int(per_draw.size)
    total = float(per_draw.sum())
    # Rounded once so the shown percentile and the verdict always agree.
    pctl = _round_pct(percentile_vs_random(total, random_totals))
    return StrategyScore(
        name=name,
        draws=n,
        cost=n * STAKE,
        winnings=round(total, 2),
        wins=int(np.count_nonzero(per_draw > 0)),
        best_prize=round(float(per_draw.max()), 2) if n else 0.0,
        percentile_vs_random=pctl,
        verdict=verdict_for(pctl),
    )


def _random_score(random_won: np.ndarray) -> StrategyScore:
    """The average random player: mean total winnings, mean prize draws (rounded) and the median
    player's best prize, all over the same tested draws."""
    n_test = int(random_won.shape[0])
    totals = random_won.sum(axis=0)
    wins = np.count_nonzero(random_won > 0, axis=0)
    best = random_won.max(axis=0) if n_test else np.zeros(random_won.shape[1])
    return StrategyScore(
        name=RANDOM_NAME,
        draws=n_test,
        cost=n_test * STAKE,
        winnings=round(float(totals.mean()), 2),
        wins=int(round(float(wins.mean()))),
        best_prize=round(float(np.median(best)), 2),
        percentile_vs_random=None,
        verdict=RANDOM_VERDICT,
    )


def _scores(names: tuple[str, ...], won: np.ndarray, random_won: np.ndarray) -> list[StrategyScore]:
    """One row per strategy, plus the Random row when there were random players."""
    random_totals = random_won.sum(axis=0)
    rows = [_strategy_score(name, won[:, j], random_totals) for j, name in enumerate(names)]
    if random_won.shape[1] > 0:
        rows.append(_random_score(random_won))
    return rows


def _skew_note(random_won: np.ndarray) -> str | None:
    """A plain warning when a few big random wins pull the average player far above the middle one.

    Lottery prizes are lopsided: one Group 1 or 1st Prize among the random players lifts the
    mean of every player, so the Random row's return can look much better than a typical
    player's. Returns None when the mean and the median are close.
    """
    if random_won.size == 0:
        return None
    totals = random_won.sum(axis=0)
    mean, median = float(totals.mean()), float(np.median(totals))
    if mean <= SKEW_RATIO * median + _EPS:
        return None
    cost = random_won.shape[0] * STAKE
    return (f"A few random players won big prizes (the largest single prize was "
            f"{money(float(random_won.max()))}), which lifts the Random average; the middle random "
            f"player won {money(median)}, {per_dollar(median / cost)} per $1.")


def _window_note(n_test: int, first: int, last: int, what: str) -> str:
    return (f"Replayed the last {plural(n_test, 'draw')} (draw {first} to {last}). Every pick was "
            f"built only from the draws before it, and {what} with that draw's real prize amounts.")


def _random_note(n_random: int, what: str) -> str:
    if n_random <= 0:
        return "No random players were simulated, so the strategies are not compared with random."
    return (f"Random is the average of {fmt_num(n_random)} random players who each bought {what} "
            "every draw. A strategy's percentile is the share of those players it beat.")


# TOTO


def _random_toto_sets(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 6) uniformly random TOTO boards: the 6 smallest of 49 random keys per row."""
    if n <= 0:
        return np.zeros((0, C.TOTO_PICK), dtype=np.int64)
    keys = rng.random((n, C.TOTO_MAX_NUMBER))
    return np.argpartition(keys, C.TOTO_PICK - 1, axis=1)[:, : C.TOTO_PICK].astype(np.int64) + 1


def _toto_pick_sets(hist: pd.DataFrame, crowd: pd.Series | None, draw_no: int) -> np.ndarray:
    """(4, 6) sets in TOTO_STRATEGIES order, built only from ``hist``.

    Seeds match the live bot (seed = the draw number being picked for), so the backtest replays
    exactly what the bot would have suggested. Looked up on the module at call time so tests can
    swap a strategy out.
    """
    builders: dict[str, Callable[[], Any]] = {
        "Hot": lambda: strategies.hot_set(hist),
        "Overdue": lambda: strategies.overdue_set(hist),
        "Balanced": lambda: strategies.balanced_set(hist, draw_no),
        "Low Crowd": lambda: strategies.low_crowd_set(hist, crowd, draw_no),
    }
    return np.array([sorted(int(n) for n in builders[name]().numbers) for name in TOTO_STRATEGIES],
                    dtype=np.int64)


def backtest_toto(
    df: pd.DataFrame,
    rules: PrizeRules,
    n_draws: int = 300,
    n_random: int = 1000,
    seed: int = 0,
    min_history: int = 100,
    refit_every: int = 1,
) -> BacktestResult:
    """Replay Hot, Overdue, Balanced and Low Crowd over the last ``n_draws`` TOTO draws.

    For the draw at position i only ``df.iloc[:i]`` is used: the four picks (Balanced and Low
    Crowd seeded with the draw number) and the crowd scores (refitted every ``refit_every`` test
    draws on the earlier draws' ``crowd_table`` rows). Each set and ``n_random`` random sets are
    scored with ``prizes.toto_prize_matrix``. ``seed`` drives the random players only. If fewer
    than ``n_draws`` draws have ``min_history`` earlier draws the window shrinks, with a note.
    """
    started = time.perf_counter()
    df = _sorted_frame(df)
    n_random = max(int(n_random), 0)
    refit_every = max(int(refit_every), 1)
    n_test, _, notes = _test_window(len(df), n_draws, min_history)
    if n_test == 0:
        return BacktestResult(game="toto", draws_tested=0, first_draw=None, last_draw=None,
                              random_sets_per_draw=n_random, scores=[], notes=notes)

    start = len(df) - n_test
    # Row j of the crowd table depends only on draws j and j - 1, and each fit below gets the
    # rows of earlier draws only (table.iloc[:i]), so no crowd figure leaks from the future.
    table = analysis_toto.crowd_table(df, rules)
    rng = np.random.default_rng(seed)
    won = np.zeros((n_test, len(TOTO_STRATEGIES)))
    random_won = np.zeros((n_test, n_random))
    crowd: pd.Series | None = None
    thin_crowd = 0  # tested draws whose crowd fit had too few usable draws (all zero scores)
    thin = False

    for k, i in enumerate(range(start, len(df))):
        hist = df.iloc[:i]
        row = df.iloc[i]
        draw_no = int(row["draw_number"])
        if crowd is None or k % refit_every == 0:
            crowd, diag = analysis_toto.crowd_scores(hist, rules, table=table.iloc[:i])
            thin = int(diag.get("n_draws") or 0) < analysis_toto.MIN_CROWD_DRAWS
        thin_crowd += int(thin)

        sets = np.vstack([_toto_pick_sets(hist, crowd, draw_no), _random_toto_sets(rng, n_random)])
        prize = prizes.toto_prize_matrix(sets, row, rules) * STAKE
        won[k] = prize[: len(TOTO_STRATEGIES)]
        random_won[k] = prize[len(TOTO_STRATEGIES):]

    first, last = int(df["draw_number"].iloc[start]), int(df["draw_number"].iloc[-1])
    notes = [_window_note(n_test, first, last, "each set was scored as one extra $1 board")] + notes
    notes.append(_random_note(n_random, "one random set"))
    skew = _skew_note(random_won)
    if skew:
        notes.append(skew)
    if refit_every > 1:
        notes.append(f"Crowd scores were refitted every {plural(refit_every, 'draw')} to save time.")
    if thin_crowd:
        notes.append(f"For {plural(thin_crowd, 'tested draw')} the crowd score had too little usable "
                     "history, so Low Crowd was a plain pattern free balanced set there.")

    log.info("TOTO backtest: %d draws x %d random sets in %.1f s", n_test, n_random,
             time.perf_counter() - started)
    return BacktestResult(
        game="toto",
        draws_tested=n_test,
        first_draw=first,
        last_draw=last,
        random_sets_per_draw=n_random,
        scores=_scores(TOTO_STRATEGIES, won, random_won),
        notes=notes,
    )


# 4D


def _fourd_prize(number: str, bet_type: str, row: Any, rules: PrizeRules) -> float:
    """Prize of a $1 (``STAKE``) 4D bet in this draw: Big / Small via the vectorised table,
    iBet via ``fourd_ticket_prize`` (it pays for any arrangement of the digits)."""
    ibet, bet = prizes.parse_fourd_bet(bet_type)
    if ibet:
        return float(prizes.fourd_ticket_prize(number, bet_type, STAKE, row, rules).amount)
    value = int(prizes.normalise_fourd_number(number))
    return float(prizes.fourd_prize_vector(np.array([value]), row, rules, bet)[0]) * STAKE


def backtest_fourd(
    df: pd.DataFrame,
    rules: PrizeRules,
    n_draws: int = 300,
    n_random: int = 1000,
    seed: int = 0,
    min_history: int = 100,
) -> BacktestResult:
    """Replay the 4D picks over the last ``n_draws`` draws against random Big numbers.

    For the draw at position i the picks come from ``strategies.fourd_strategy_numbers`` on
    ``df.iloc[:i]`` (seeded with the draw number, as the live bot does), $1 each with their own
    bet type (iBet Big for Digit Set). Random player j buys one random Big $1 number per draw.
    The "Random" pick is a random number too, so the Random row (the average random player)
    stands for it. ``seed`` drives the random players only.
    """
    started = time.perf_counter()
    df = _sorted_frame(df)
    n_random = max(int(n_random), 0)
    n_test, _, notes = _test_window(len(df), n_draws, min_history)
    if n_test == 0:
        return BacktestResult(game="4d", draws_tested=0, first_draw=None, last_draw=None,
                              random_sets_per_draw=n_random, scores=[], notes=notes)

    start = len(df) - n_test
    rng = np.random.default_rng(seed)
    won = np.zeros((n_test, len(FOURD_STRATEGIES)))
    random_won = np.zeros((n_test, n_random))
    bet_types: dict[str, set[str]] = {name: set() for name in FOURD_STRATEGIES}

    for k, i in enumerate(range(start, len(df))):
        hist = df.iloc[:i]
        row = df.iloc[i]
        draw_no = int(row["draw_number"])
        picks: Mapping[str, tuple[str, str]] = strategies.fourd_strategy_numbers(hist, seed=draw_no)
        for j, name in enumerate(FOURD_STRATEGIES):
            number, bet_type = picks[name]
            bet_types[name].add(str(bet_type))
            won[k, j] = _fourd_prize(number, bet_type, row, rules)
        if n_random:
            numbers = rng.integers(0, C.FOURD_SPACE, size=n_random)
            random_won[k] = prizes.fourd_prize_vector(numbers, row, rules, "big") * STAKE

    first, last = int(df["draw_number"].iloc[start]), int(df["draw_number"].iloc[-1])
    bets = ", ".join(f"{name} {' or '.join(sorted(bet_types[name]))}" for name in FOURD_STRATEGIES)
    notes = [_window_note(n_test, first, last, f"each pick staked $1 ({bets}) and was scored")] + notes
    notes.append(_random_note(n_random, "one random Big $1 number"))
    if n_random:
        notes.append("The Random pick is a random number like the random players buy, so the "
                     "Random row stands for it.")
    skew = _skew_note(random_won)
    if skew:
        notes.append(skew)

    log.info("4D backtest: %d draws x %d random numbers in %.1f s", n_test, n_random,
             time.perf_counter() - started)
    return BacktestResult(
        game="4d",
        draws_tested=n_test,
        first_draw=first,
        last_draw=last,
        random_sets_per_draw=n_random,
        scores=_scores(FOURD_STRATEGIES, won, random_won),
        notes=notes,
    )


# JSON cache


def _json_float(x: Any) -> float | None:
    """A finite float, or None (JSON has no NaN)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _opt_int(x: Any) -> int | None:
    f = _json_float(x)
    return None if f is None else int(f)


def _score_to_dict(s: StrategyScore) -> dict:
    return {
        "name": str(s.name),
        "draws": int(s.draws),
        "cost": _json_float(s.cost) or 0.0,
        "winnings": _json_float(s.winnings) or 0.0,
        "wins": int(s.wins),
        "best_prize": _json_float(s.best_prize) or 0.0,
        "percentile_vs_random": _json_float(s.percentile_vs_random),
        "verdict": str(s.verdict),
        # Convenience for readers of the cache file; ignored when loading (it is a property).
        "return_per_dollar": _json_float(s.return_per_dollar),
    }


def _score_from_dict(d: Mapping[str, Any]) -> StrategyScore:
    return StrategyScore(
        name=str(d.get("name", "")),
        draws=_opt_int(d.get("draws")) or 0,
        cost=_json_float(d.get("cost")) or 0.0,
        winnings=_json_float(d.get("winnings")) or 0.0,
        wins=_opt_int(d.get("wins")) or 0,
        best_prize=_json_float(d.get("best_prize")) or 0.0,
        percentile_vs_random=_json_float(d.get("percentile_vs_random")),
        verdict=str(d.get("verdict", "")),
    )


def result_to_dict(r: BacktestResult) -> dict:
    """JSON safe dict of a backtest result (plain ints, floats, strings, None)."""
    return {
        "format": RESULT_FORMAT,
        "game": str(r.game),
        "draws_tested": int(r.draws_tested),
        "first_draw": _opt_int(r.first_draw),
        "last_draw": _opt_int(r.last_draw),
        "random_sets_per_draw": int(r.random_sets_per_draw),
        "scores": [_score_to_dict(s) for s in r.scores],
        "notes": [str(n) for n in r.notes],
    }


def result_from_dict(d: Mapping[str, Any]) -> BacktestResult:
    """Inverse of ``result_to_dict``. Unknown keys are ignored, missing ones get defaults.

    Raises ValueError for a dict written by a newer, incompatible format, so the caller can
    treat the cache as stale and rerun the backtest.
    """
    fmt = d.get("format", RESULT_FORMAT)
    if _opt_int(fmt) != RESULT_FORMAT:
        raise ValueError(f"unsupported backtest cache format {fmt!r}")
    return BacktestResult(
        game=str(d.get("game", "")),
        draws_tested=_opt_int(d.get("draws_tested")) or 0,
        first_draw=_opt_int(d.get("first_draw")),
        last_draw=_opt_int(d.get("last_draw")),
        random_sets_per_draw=_opt_int(d.get("random_sets_per_draw")) or 0,
        scores=[_score_from_dict(s) for s in d.get("scores") or []],
        notes=[str(n) for n in d.get("notes") or []],
    )
