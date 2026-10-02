"""Synthetic but internally consistent TOTO history.

Used by the tests and by ``python -m huatbot demo`` so the whole pipeline can run
without reaching the Singapore Pools site. The TOTO model plants a known crowd
preference so the crowd score can be checked against the truth.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import constants as C
from .models import NextToto
from .store import normalise_toto

SG = ZoneInfo(C.SG_TZ_NAME)

# Numbers the simulated crowd over buys (multiplier on top of the birthday bias).
DEFAULT_POPULAR = {7: 2.4, 8: 2.2, 9: 2.0, 18: 1.8, 28: 1.7, 13: 0.55, 44: 0.6, 49: 0.6}


def popularity_weights(popular: dict[int, float] | None = None) -> np.ndarray:
    """Weight per number, index 0 is number 1. Birthday numbers (1 to 31) get a mild boost."""
    w = np.ones(C.TOTO_MAX_NUMBER)
    w[: C.TOTO_BIRTHDAY_MAX] *= 1.15
    for n, m in (DEFAULT_POPULAR if popular is None else popular).items():
        w[n - 1] *= m
    return w


def draw_dates(start: date, weekdays: tuple[int, ...], n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() in weekdays:
            out.append(d)
        d += timedelta(days=1)
    return out


def _boards_for_jackpot(jackpot: float, rng: np.random.Generator) -> float:
    base = 2_300_000 + 0.55 * jackpot
    return float(base * rng.lognormal(0.0, 0.06))


def synth_toto(
    n_draws: int = 400,
    start_draw: int = 3800,
    start_date: date = date(2022, 1, 3),
    seed: int = 7,
    popular: dict[int, float] | None = None,
    hongbao_every: int = 104,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = popularity_weights(popular)
    w_mean = w.mean()
    pool_pct = C.TOTO_GROUP_POOL_PCT
    rows = []
    snow1 = 0.0
    snow = {2: 0.0, 3: 0.0, 4: 0.0}
    streak = 0
    dates = draw_dates(start_date, C.TOTO_WEEKDAYS, n_draws)
    for i, d in enumerate(dates):
        draw_no = start_draw + i
        nums = np.sort(rng.choice(np.arange(1, 50), size=7, replace=False)[:6])
        rest = np.setdiff1d(np.arange(1, 50), nums)
        additional = int(rng.choice(rest))
        crowd = float(w[nums - 1].mean() / w_mean)

        draw_type = "normal"
        guaranteed = None
        if hongbao_every and i > 0 and i % hongbao_every == 0:
            draw_type, guaranteed = "hongbao", 12_000_000.0
        elif streak == C.TOTO_SNOWBALL_LIMIT - 1:
            draw_type = "cascade"

        prospective = guaranteed or max(C.TOTO_MIN_GROUP1, snow1 + 2_000_000)
        boards = _boards_for_jackpot(prospective, rng)
        pool = C.TOTO_POOL_SHARE_OF_SALES * boards
        jackpot = max(C.TOTO_MIN_GROUP1, snow1 + pool_pct[1] * pool)
        if guaranteed:
            jackpot = max(jackpot, guaranteed)

        winners = {}
        for g in range(1, 8):
            lam = boards * C.TOTO_GROUP_COMBOS[g] / C.TOTO_COMBOS * crowd
            winners[g] = int(rng.poisson(lam))

        amounts = {1: jackpot}
        for g in (2, 3, 4):
            amounts[g] = pool_pct[g] * pool + snow[g]
        if draw_type in ("cascade", "hongbao") and winners[1] == 0:
            for g in (2, 3, 4):  # the jackpot cascades to the next group with winners
                if winners[g] > 0:
                    amounts[g] += jackpot
                    break

        shares = {}
        for g in range(1, 5):
            shares[g] = float(np.floor(amounts[g] / winners[g])) if winners[g] else np.nan
        for g in (5, 6, 7):
            shares[g] = C.TOTO_FIXED_PRIZES[g] if winners[g] else np.nan

        # Snowballs for the next draw.
        if winners[1] > 0 or draw_type in ("cascade", "hongbao"):
            snow1, streak = 0.0, 0
        else:
            snow1, streak = jackpot, streak + 1
        for g in (2, 3, 4):
            snow[g] = 0.0 if winners[g] else amounts[g]

        row = {
            "draw_number": draw_no,
            "draw_date": pd.Timestamp(d),
            **{f"n{k + 1}": int(v) for k, v in enumerate(nums)},
            "additional": additional,
            "jackpot": float(np.floor(jackpot)),
            "draw_type": draw_type,
            "fetched_at": "synthetic",
        }
        for g in range(1, 8):
            row[f"g{g}_share"] = shares[g]
            row[f"g{g}_winners"] = winners[g]
        rows.append(row)
    return normalise_toto(pd.DataFrame(rows))


def _next_date(last: date, weekdays: tuple[int, ...]) -> date:
    d = last + timedelta(days=1)
    while d.weekday() not in weekdays:
        d += timedelta(days=1)
    return d


def synth_next_toto(df: pd.DataFrame) -> NextToto:
    last = df.iloc[-1]
    d = _next_date(last["draw_date"].date(), C.TOTO_WEEKDAYS)
    if int(last["g1_winners"]) > 0 or last["draw_type"] in ("cascade", "hongbao"):
        jackpot = C.TOTO_MIN_GROUP1
    else:
        jackpot = float(last["jackpot"]) + 1_100_000
    streak = 0
    for _, r in df.iloc[::-1].iterrows():
        if int(r["g1_winners"]) > 0 or r["draw_type"] in ("cascade", "hongbao"):
            break
        streak += 1
    draw_type = "cascade" if streak == C.TOTO_SNOWBALL_LIMIT - 1 else "normal"
    return NextToto(
        draw_datetime=datetime.combine(d, C.DRAW_TIME, tzinfo=SG),
        jackpot_estimate=jackpot,
        draw_type=draw_type,
        draw_type_hint=None,
        raw_text=f"Next Jackpot ${jackpot:,.0f} est Next Draw {d:%a, %d %b %Y} , 6.30pm",
    )
