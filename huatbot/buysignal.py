"""TOTO buy signal: expected return per $1 at the advertised jackpot, and a HIGH / MEDIUM / LOW label.

The expected value model treats the number of other boards that hit each group as Poisson
with mean boards x (group combinations) / 13,983,816, which gives closed forms for sharing a
pool prize. Sales ("boards") at a given jackpot are estimated from history via the crowd
table's pool estimates. No label ever means a draw is more likely to be won by you: the
label only reflects the jackpot size and the user's alert settings.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from . import constants as C
from .analysis_toto import CASCADE_TYPES, crowd_table
from .models import BuySignal, NextToto, PrizeRules, Settings
from .store import no_winner_streak

log = logging.getLogger(__name__)

NEAR_JACKPOT_FACTOR = 1.25  # "near" = within x1.25 either way
MIN_NEAR_DRAWS = 8
MIN_FIT_DRAWS = 20
SPECIAL_TYPES = ("cascade", "hongbao", "special")
_TYPE_NAMES = {"cascade": "cascade", "hongbao": "Hongbao", "special": "special", "normal": "normal"}


def _money(x: float) -> str:
    """Whole dollars with thousands separators, e.g. "$1,000,000" (local, no dashes)."""
    return f"${x:,.0f}" if x >= 0 else f"minus ${-x:,.0f}"


def _per_dollar(x: float) -> str:
    return f"${x:,.2f}" if x >= 0 else f"minus ${-x:,.2f}"


# Sales estimate


def estimate_boards(
    df: pd.DataFrame, jackpot: float, rules: PrizeRules, table: pd.DataFrame | None = None
) -> tuple[float | None, str]:
    """Typical number of boards sold when the jackpot is about ``jackpot``.

    1. At least 8 past draws with a jackpot within x1.25 of it: the median of their boards.
    2. Else a straight line fit of log(boards) on log(jackpot) over all usable draws (at least 20).
    3. Else (None, "not enough history").
    Usable draws are those with a clean pool estimate in the crowd table and a known jackpot.
    """
    jackpot = _clean_amount(jackpot)
    if jackpot is None or jackpot <= 0:
        return None, "no jackpot to compare with"
    if table is None:
        table = crowd_table(df, rules)
    jp = pd.Series(df["jackpot"].to_numpy(dtype=float), index=df["draw_number"].to_numpy())
    boards = table["boards"].astype(float).reindex(jp.index)
    ok = (np.isfinite(boards) & (boards > 0) & np.isfinite(jp) & (jp > 0)).to_numpy()
    b = boards.to_numpy()[ok]
    j = jp.to_numpy()[ok]

    near = (j >= jackpot / NEAR_JACKPOT_FACTOR) & (j <= jackpot * NEAR_JACKPOT_FACTOR)
    n_near = int(near.sum())
    if n_near >= MIN_NEAR_DRAWS:
        est = float(np.median(b[near]))
        return est, f"median of {n_near} past draws with a jackpot near {_money(jackpot)}"

    if len(b) >= MIN_FIT_DRAWS and np.ptp(np.log(j)) > 0:
        slope, intercept = np.polyfit(np.log(j), np.log(b), 1)
        est = float(np.exp(intercept + slope * math.log(jackpot)))
        if n_near == 0:
            why = "no past draw had"
        else:
            why = f"only {n_near} past draw{'s' if n_near > 1 else ''} had"
        return est, (f"trend of sales against jackpot over {len(b)} past draws, as {why} "
                     f"a jackpot near {_money(jackpot)}")
    return None, "not enough history"


# Expected value


def _share_factor(lam: float) -> float:
    """E[1 / (1 + K)] for K ~ Poisson(lam): the average fraction of a prize you keep."""
    if lam < 1e-12:
        return 1.0
    return float(-math.expm1(-lam) / lam)


def toto_ev_per_dollar(jackpot: float, boards: float, rules: PrizeRules, draw_type: str = "normal") -> dict:
    """Expected return per $1 board, split by group (keys g1, g2, g3, g4, fixed, cascade, total).

    Group 1: jackpot / C times the share factor (1 - e^(-lam1)) / lam1 for the other expected
    Group 1 winners (lam1 = boards / C; the factor tends to 1 as sales tend to 0).
    Groups 2 to 4: P(win) x group pool x share factor, with group pool = pct_g x pool share x
    boards, simplifies to pct_g x pool share x (1 - e^(-lam_g)): the group pool is only paid
    out (rather than snowballed) when somebody wins it.
    Groups 5 to 7: fixed prizes times their odds (about $0.24). On cascade and Hongbao draws an
    unwon jackpot drops into Group 2, which adds the "cascade" term.
    Snowballs already sitting in Groups 2 to 4 are not counted, so this is slightly conservative.
    """
    combos = C.TOTO_GROUP_COMBOS
    big_c = float(C.TOTO_COMBOS)
    boards = max(float(boards), 0.0)
    jackpot = max(float(jackpot), 0.0)

    lam1 = boards / big_c
    out = {"g1": jackpot * _share_factor(lam1) / big_c}
    for g in (2, 3, 4):
        lam = boards * combos[g] / big_c
        out[f"g{g}"] = rules.group_pool_pct[g] * rules.pool_share_of_sales * float(-math.expm1(-lam))
    out["fixed"] = sum(rules.fixed_prizes[g] * combos[g] / big_c for g in (5, 6, 7))
    if draw_type in CASCADE_TYPES:
        lam2 = boards * combos[2] / big_c
        out["cascade"] = math.exp(-lam1) * jackpot * (combos[2] / big_c) * _share_factor(lam2)
    else:
        out["cascade"] = 0.0
    out["total"] = out["g1"] + out["g2"] + out["g3"] + out["g4"] + out["fixed"] + out["cascade"]
    return {k: float(v) for k, v in out.items()}


# Signal


def _clean_amount(value) -> float | None:
    """A finite float, or None for missing / unreadable values."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _predicted_draw_type(df: pd.DataFrame) -> str:
    """Next draw type from history alone (used when the next draw page could not be read)."""
    if len(df) and no_winner_streak(df, reset_on_cascade=True) == C.TOTO_SNOWBALL_LIMIT - 1:
        return "cascade"
    return "normal"


def buy_signal(
    next_toto: NextToto | None,
    df: pd.DataFrame,
    settings: Settings,
    rules: PrizeRules,
    table: pd.DataFrame | None = None,
) -> BuySignal:
    """Label the next TOTO draw HIGH, MEDIUM or LOW and estimate the return per $1.

    HIGH: jackpot at or above the user's alert, or a cascade / Hongbao / special draw when
    the user wants those flagged. LOW: jackpot at the $1,000,000 minimum. Otherwise MEDIUM,
    including when the jackpot estimate is not available.
    """
    jackpot = _clean_amount(next_toto.jackpot_estimate if next_toto is not None else None)
    draw_type = next_toto.draw_type if next_toto is not None and next_toto.draw_type else _predicted_draw_type(df)
    streak = no_winner_streak(df) if len(df) else 0

    boards, method, ev = None, "jackpot estimate not available", {}
    if jackpot is not None:
        if table is None and len(df):
            table = crowd_table(df, rules)
        boards, method = estimate_boards(df, jackpot, rules, table) if len(df) else (None, "not enough history")
        if boards is not None:
            ev = toto_ev_per_dollar(jackpot, boards, rules, draw_type)
    total = ev.get("total")

    special = settings.alert_on_special_draws and draw_type in SPECIAL_TYPES
    type_name = _TYPE_NAMES.get(draw_type, draw_type)
    ev_part = (f", and each $1 returns about {_per_dollar(total)} on average" if total is not None
               else ", but there is not enough history to estimate the return per $1")

    if jackpot is not None and jackpot >= settings.jackpot_alert:
        label = "HIGH"
        reason = (f"The estimated jackpot of {_money(jackpot)} meets your "
                  f"{_money(settings.jackpot_alert)} alert{ev_part}.")
    elif special:
        label = "HIGH"
        tail = ev_part if jackpot is not None else " (jackpot estimate not available)"
        reason = f"This is a {type_name} draw, which your settings treat as an alert{tail}."
    elif jackpot is None:
        label = "MEDIUM"
        reason = "Signal set to MEDIUM by default (jackpot estimate not available)."
    elif jackpot <= rules.min_group1 * 1.001:
        label = "LOW"
        reason = f"The jackpot is at the {_money(rules.min_group1)} minimum{ev_part}."
    else:
        label = "MEDIUM"
        reason = (f"The estimated jackpot of {_money(jackpot)} is above the minimum but below your "
                  f"{_money(settings.jackpot_alert)} alert{ev_part}.")

    log.debug("buy signal %s: jackpot=%s type=%s boards=%s ev=%s", label, jackpot, draw_type, boards, total)
    return BuySignal(
        label=label,
        jackpot=float(jackpot) if jackpot is not None else None,
        draw_type=draw_type,
        no_winner_streak=int(streak),
        boards_estimate=boards,
        boards_method=method,
        ev_per_dollar=total,
        ev_breakdown=ev,
        reason=reason,
    )
