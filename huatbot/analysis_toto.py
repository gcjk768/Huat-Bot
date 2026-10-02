"""TOTO history analysis: frequencies, gaps, pairs, set shape, fairness test and crowd score.

Every function takes a toto.csv frame (see ``models.TOTO_COLUMNS``) and returns plain pandas /
numpy / dict results that the report, notes and strategies turn into text. Nothing here
predicts a draw: every draw is independent. The crowd score only estimates which numbers
*other players* like, which changes how many people a prize would be split with.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from math import floor, isfinite

import numpy as np
import pandas as pd
from scipy import stats

from . import constants as C
from .models import PrizeRules

log = logging.getLogger(__name__)

NUMBERS = np.arange(1, C.TOTO_MAX_NUMBER + 1)
_NUM_COLS = [f"n{i}" for i in range(1, C.TOTO_PICK + 1)]

# Draw types whose unwon Group 1 jackpot cascades into the next group with winners.
CASCADE_TYPES = ("cascade", "hongbao")

# Crowd score settings.
MIN_CROWD_DRAWS = 30
# Ridge penalty. With n usable draws every number's centred indicator adds about 0.11 * n to
# the diagonal of X'X, so a penalty of 5 shrinks the scores by roughly 5 / (0.11 n + 5): about
# 60% with 30 draws, 23% with 150, under 4% with the full history since 2014. Thin histories
# are pulled hard towards "no preference" while a long history speaks for itself. The design
# is close to isotropic, so the ranking of numbers barely depends on this value.
DEFAULT_RIDGE_ALPHA = 5.0
CROWD_OK_TOLERANCE = 0.10


# Basic counts


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Oldest draw first, so "last N" and "draws since" are well defined."""
    if len(df) and not df["draw_number"].is_monotonic_increasing:
        df = df.sort_values("draw_number", kind="stable")
    return df


def number_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_draws, 49) bool matrix in frame order; column k is True when number k + 1 was drawn.

    The additional number is not included. Out of range values (a damaged row) are ignored.
    """
    n = len(df)
    m = np.zeros((n, C.TOTO_MAX_NUMBER), dtype=bool)
    if n == 0:
        return m
    nums = df[_NUM_COLS].to_numpy(dtype=np.int64)
    valid = (nums >= 1) & (nums <= C.TOTO_MAX_NUMBER)
    rows = np.repeat(np.arange(n), C.TOTO_PICK).reshape(n, C.TOTO_PICK)
    m[rows[valid], nums[valid] - 1] = True
    return m


def frequency(df: pd.DataFrame, last: int | None = None) -> pd.Series:
    """How often each number 1..49 was drawn, over all draws or the most recent ``last``."""
    df = _sorted(df)
    if last is not None:
        df = df.iloc[-last:] if last > 0 else df.iloc[0:0]
    counts = number_matrix(df).sum(axis=0).astype(np.int64)
    return pd.Series(counts, index=pd.Index(NUMBERS, name="number"), name="count")


def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per number over all history, the last 100 and the last 50 draws."""
    return pd.DataFrame(
        {
            "all": frequency(df),
            "last100": frequency(df, last=100),
            "last50": frequency(df, last=50),
        }
    )


def overdue(df: pd.DataFrame) -> pd.Series:
    """Draws since each number last appeared: 0 = in the latest draw; never seen = len(df)."""
    m = number_matrix(_sorted(df))
    n = len(m)
    gaps = np.full(C.TOTO_MAX_NUMBER, n, dtype=np.int64)
    if n:
        seen = m.any(axis=0)
        # Index of the last True per column: flip rows and take the first True.
        last_idx = n - 1 - np.argmax(m[::-1], axis=0)
        gaps[seen] = (n - 1 - last_idx)[seen]
    return pd.Series(gaps, index=pd.Index(NUMBERS, name="number"), name="draws_since")


def top_pairs(df: pd.DataFrame, k: int = 10) -> list[tuple[tuple[int, int], int]]:
    """The ``k`` pairs of numbers drawn together most often (ties: lower pair first)."""
    m = number_matrix(df).astype(np.int64)
    co = m.T @ m  # co[i, j] = draws containing both i + 1 and j + 1
    iu, ju = np.triu_indices(C.TOTO_MAX_NUMBER, k=1)
    counts = co[iu, ju]
    # lexsort sorts by the last key first: count descending, then pair ascending.
    order = np.lexsort((ju, iu, -counts))
    out = []
    for idx in order[: max(k, 0)]:
        out.append(((int(iu[idx] + 1), int(ju[idx] + 1)), int(counts[idx])))
    return out


# Shape of a winning set


@lru_cache(maxsize=1)
def _theoretical_sum_quartiles() -> tuple[float, float, float]:
    """Quartiles of the sum of 6 distinct numbers from 1..49, exact by counting subsets.

    Used only when there is no history, so callers always get usable numbers.
    """
    max_sum = sum(range(C.TOTO_MAX_NUMBER - C.TOTO_PICK + 1, C.TOTO_MAX_NUMBER + 1))
    ways = np.zeros((C.TOTO_PICK + 1, max_sum + 1), dtype=np.float64)
    ways[0, 0] = 1.0
    for x in range(1, C.TOTO_MAX_NUMBER + 1):
        for size in range(C.TOTO_PICK, 0, -1):  # backwards so each number is used once
            ways[size, x:] += ways[size - 1, : max_sum + 1 - x]
    cdf = np.cumsum(ways[C.TOTO_PICK]) / ways[C.TOTO_PICK].sum()
    return tuple(float(np.searchsorted(cdf, q)) for q in (0.25, 0.5, 0.75))  # type: ignore[return-value]


def _mode_near_three(dist: dict[int, float]) -> int:
    """Most common count; ties go to the count closest to 3 (the expected split), then lower."""
    best = max(dist.values()) if dist else 0.0
    candidates = [k for k, v in dist.items() if v == best]
    return min(candidates, key=lambda k: (abs(k - 3), k))


def shape_stats(df: pd.DataFrame) -> dict:
    """Usual odd/even split, low (1 to 24) / high split and middle 50% sum range of past sets."""
    n = len(df)
    keys = range(C.TOTO_PICK + 1)
    if n == 0:
        q25, med, q75 = _theoretical_sum_quartiles()
        zero = {k: 0.0 for k in keys}
        return {
            "n_draws": 0, "odd_dist": dict(zero), "low_dist": dict(zero),
            "typical_odd": 3, "typical_low": 3,
            "sum_q25": q25, "sum_q75": q75, "sum_median": med,
        }
    nums = df[_NUM_COLS].to_numpy(dtype=np.int64)
    odd = (nums % 2 == 1).sum(axis=1)
    low = (nums <= C.TOTO_LOW_MAX).sum(axis=1)
    sums = nums.sum(axis=1)
    odd_dist = {k: float(np.mean(odd == k)) for k in keys}
    low_dist = {k: float(np.mean(low == k)) for k in keys}
    return {
        "n_draws": int(n),
        "odd_dist": odd_dist,
        "low_dist": low_dist,
        "typical_odd": _mode_near_three(odd_dist),
        "typical_low": _mode_near_three(low_dist),
        "sum_q25": float(np.percentile(sums, 25)),
        "sum_q75": float(np.percentile(sums, 75)),
        "sum_median": float(np.median(sums)),
    }


# Fairness test


def format_p(p: float) -> str:
    """p value for prose. Never scientific notation (that would put a dash in the text).

    Below 0.05 it is rounded down (3 places from 0.01), so a result under the 5% cut never
    prints as "0.05"; above 0.99 it reads "above 0.99" rather than an impossible "1.00".
    """
    if not isfinite(p):
        return "n/a"
    if p > 0.99:
        return "above 0.99"
    if p >= 0.05:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{floor(p * 1000) / 1000:.3f}"
    if p >= 0.0001:
        return f"{p:.4f}".rstrip("0")
    return "below 0.0001"


def chi_square_numbers(df: pd.DataFrame) -> dict:
    """Chi square test of the 49 number counts against the even spread 6n / 49.

    Each draw holds 6 distinct numbers (drawn without replacement), so under a fair draw the
    plain Pearson statistic tends to (49 - 6) / (49 - 1) = 43/48 of a chi square with 48
    degrees of freedom. It is scaled by 48/43 before the p value is read from chi2(48).
    Damaged rows (a number out of range or repeated) are left out, and ``n_draws`` counts the
    draws actually tested.
    """
    dof = C.TOTO_MAX_NUMBER - 1
    m = number_matrix(df)
    valid = m.sum(axis=1) == C.TOTO_PICK
    n = int(valid.sum())
    if n == 0:
        return {"stat": float("nan"), "dof": dof, "p_value": float("nan"), "n_draws": 0,
                "verdict": "There are no draws yet, so the spread of numbers cannot be tested."}
    counts = m[valid].sum(axis=0).astype(float)
    expected = np.full(C.TOTO_MAX_NUMBER, C.TOTO_PICK * n / C.TOTO_MAX_NUMBER)
    raw_stat, _ = stats.chisquare(counts, f_exp=expected)
    stat = float(raw_stat) * (C.TOTO_MAX_NUMBER - 1) / (C.TOTO_MAX_NUMBER - C.TOTO_PICK)
    p = float(stats.chi2.sf(stat, dof))
    p_text = format_p(p)
    p_part = f"p {p_text}" if p_text.startswith(("above", "below")) else f"p = {p_text}"
    if p < 0.05:
        verdict = (f"The spread of numbers is more uneven than chance usually gives ({p_part}), "
                   "but this alone does not make any number more likely in the next draw.")
    else:
        verdict = (f"The spread of numbers is consistent with pure chance ({p_part}), "
                   "so hot and cold numbers are just noise.")
    return {"stat": stat, "dof": dof, "p_value": p, "n_draws": int(n), "verdict": verdict}


# Crowd score


def _group_problem(
    g: int, share: pd.Series, winners: pd.Series, prev_winners: pd.Series, lands: pd.Series
) -> pd.Series:
    """Plain reason why group ``g`` cannot give a clean pool estimate ("" when it can)."""
    reason = pd.Series("", index=share.index, dtype=object)
    # Assign in reverse priority so the most basic problem wins.
    reason[lands.to_numpy()] = f"Group {g} received the cascaded jackpot"
    reason[(prev_winners == 0).to_numpy()] = f"Group {g} held a snowball from the previous draw"
    bad_share = ~np.isfinite(share.to_numpy(dtype=float)) | (share.to_numpy(dtype=float) <= 0)
    reason[bad_share] = f"Group {g} share amount is missing"
    reason[(winners <= 0).to_numpy()] = f"Group {g} had no winner"
    return reason


def crowd_table(df: pd.DataFrame, rules: PrizeRules) -> pd.DataFrame:
    """Per draw estimate of sales and how crowded the winning numbers were.

    pool = Group 3 share x Group 3 winners / Group 3 pool percentage (Group 4 as fallback),
    boards = pool / pool share of sales, expected Group 7 winners = boards x 229,600 / 13,983,816,
    crowd ratio = actual Group 7 winners / expected. A group is only used when its amount is
    pure pool money: it must have winners, not carry a snowball from the previous draw (that
    draw had no winner in the group) and not have received a cascaded jackpot. Draws where
    neither Group 3 nor Group 4 is clean get basis "", NaN figures and a plain
    ``excluded_reason``. A draw with a clean pool but no Group 7 figures keeps its pool and
    boards (still useful sales data) and gets a NaN crowd ratio with a reason.
    """
    cols = ["basis", "pool", "boards", "expected_g7", "crowd_ratio", "excluded_reason"]
    df = _sorted(df)
    index = pd.Index(df["draw_number"].to_numpy(dtype=np.int64), name="draw_number")
    if len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype=object if c in ("basis", "excluded_reason") else float)
                             for c in cols}, index=index)

    d = df.reset_index(drop=True)
    w = {g: d[f"g{g}_winners"].astype(np.int64) for g in range(1, 8)}
    share = {g: d[f"g{g}_share"].astype(float) for g in (3, 4)}

    # Winners in the same group one draw earlier (looked up by draw number, NaN when that
    # draw is not in the frame: then we cannot see a snowball and assume none).
    prev = {}
    for g in (3, 4):
        lookup = pd.Series(w[g].to_numpy(), index=d["draw_number"].to_numpy())
        prev[g] = pd.Series((d["draw_number"] - 1).map(lookup).to_numpy(dtype=float))

    # An unwon cascade / Hongbao jackpot goes to the highest group below Group 1 with winners.
    unwon_cascade = d["draw_type"].isin(CASCADE_TYPES) & (w[1] == 0) & (w[2] == 0)
    lands = {3: unwon_cascade & (w[3] > 0), 4: unwon_cascade & (w[3] == 0) & (w[4] > 0)}

    problem = {g: _group_problem(g, share[g], w[g], prev[g], lands[g]) for g in (3, 4)}
    use3 = (problem[3] == "").to_numpy()
    use4 = ~use3 & (problem[4] == "").to_numpy()

    pct = rules.group_pool_pct
    pool = np.full(len(d), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pool[use3] = (share[3] * w[3] / pct[3]).to_numpy()[use3]
        pool[use4] = (share[4] * w[4] / pct[4]).to_numpy()[use4]
    basis = np.where(use3, "G3", np.where(use4, "G4", "")).astype(object)
    reason = np.where(use3 | use4, "", problem[3] + " and " + problem[4]).astype(object)

    # A pool that is not a positive number (e.g. nonsense percentages) gives no estimate at all.
    bad_pool = (use3 | use4) & ~(np.isfinite(pool) & (pool > 0))
    pool[bad_pool] = np.nan
    basis[bad_pool] = ""
    reason[bad_pool] = "prize pool figures are not usable"

    with np.errstate(divide="ignore", invalid="ignore"):
        boards = pool / rules.pool_share_of_sales
        expected_g7 = boards * C.TOTO_GROUP_COMBOS[7] / C.TOTO_COMBOS
        g7 = w[7].to_numpy(dtype=float)
        ratio = g7 / expected_g7

    # Millions of boards always produce thousands of Group 7 winners, so zero means the share
    # table was not captured. The pool (and boards) estimate is still valid sales data and is
    # kept, but the draw gets no crowd ratio.
    no_g7 = np.isfinite(pool) & (g7 <= 0)
    reason[no_g7] = "no Group 7 winners recorded"
    ratio[no_g7 | ~np.isfinite(pool)] = np.nan

    out = pd.DataFrame(
        {
            "basis": basis,
            "pool": pool,
            "boards": boards,
            "expected_g7": expected_g7,
            "crowd_ratio": ratio,
            "excluded_reason": reason,
        },
        index=index,
    )
    n_bad = int(np.isnan(ratio).sum())
    if n_bad:
        log.debug("crowd_table: %d of %d draws excluded", n_bad, len(out))
    return out


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed form ridge on centred data: beta = (Xc'Xc + alpha I)^-1 Xc'yc. Returns (beta, r2)."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    a = xc.T @ xc + alpha * np.eye(x.shape[1])
    beta = np.linalg.solve(a, xc.T @ yc)
    ss_tot = float(yc @ yc)
    resid = yc - xc @ beta
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def crowd_scores(
    df: pd.DataFrame,
    rules: PrizeRules,
    alpha: float | None = None,
    table: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    """Score each number by how much it raises the crowd ratio when drawn.

    Ridge regression of the crowd ratio on the 49 "was this number drawn" indicators, with X
    and y centred (no intercept needed). Because every draw has exactly six numbers, the
    centred indicators sum to zero across numbers, and the ridge solution therefore also sums
    to zero: positive = more people buy it than average, negative = fewer.

    ``table`` may be a precomputed ``crowd_table``; only its rows for draws in ``df`` are used
    (handy for backtests that fit on earlier draws only).
    """
    alpha = DEFAULT_RIDGE_ALPHA if alpha is None else float(alpha)
    if table is None:
        table = crowd_table(df, rules)
    df = _sorted(df)
    ratios = table["crowd_ratio"].reindex(df["draw_number"].to_numpy())
    usable = np.isfinite(ratios.to_numpy(dtype=float))
    y = ratios.to_numpy(dtype=float)[usable]
    n_used, n_excluded = int(usable.sum()), int((~usable).sum())

    mean_ratio = float(np.mean(y)) if n_used else None
    median_ratio = float(np.median(y)) if n_used else None
    ok = mean_ratio is not None and abs(mean_ratio - 1.0) <= CROWD_OK_TOLERANCE

    warnings = []
    if mean_ratio is not None and not ok:
        pool_pct = f"{rules.pool_share_of_sales * 100:g}%"
        warnings.append(f"Average crowd ratio is {mean_ratio:.2f}, not close to 1, so the "
                        f"{pool_pct} prize pool assumption may be off.")

    index = pd.Index(NUMBERS, name="number")
    if n_used < MIN_CROWD_DRAWS:
        warnings.insert(0, f"Only {n_used} usable draws for the crowd score ({MIN_CROWD_DRAWS} "
                           "needed), so every number gets a crowd score of zero.")
        scores = pd.Series(np.zeros(C.TOTO_MAX_NUMBER), index=index, name="crowd_score")
        r2 = 0.0
    else:
        x = number_matrix(df)[usable].astype(float)
        beta, r2 = _ridge(x, y, alpha)
        scores = pd.Series(beta, index=index, name="crowd_score")

    diag = {
        "n_draws": n_used,
        "n_excluded": n_excluded,
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
        "ok": bool(ok),
        "warning": " ".join(warnings) if warnings else None,
        "alpha": alpha,
        "r2": float(r2),
    }
    return scores, diag
