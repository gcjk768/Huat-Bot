"""What the next TOTO draws are likely to do, and where the next big prize is.

The jackpot snowballs: when nobody wins Group 1, the whole Group 1 prize rolls into the next
draw and 38% of that draw's prize pool is added. After the 4th draw in a row with no winner
the jackpot cascades: it is shared by the next group with winners (usually Group 2).

For each draw up to the cascade this module estimates

* the jackpot if nobody has won it before,
* how many boards that jackpot usually sells (from the stored history, see buysignal),
* the chance at least one board wins Group 1: 1 minus e^(-boards / 13,983,816), which treats
  every board as an independent random pick (people's favourite numbers are ignored),
* the chance the jackpot is still unwon when that draw comes.

Nothing here says which numbers will be drawn: every draw is independent.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import constants as C
from .buysignal import estimate_boards, toto_ev_per_dollar
from .models import JackpotHistory, JackpotOutlook, NextToto, OutlookStep, PrizeRules
from .sales import CASCADE_TYPES, sales_table
from .store import no_winner_streak

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)

# Used only when the history has no sales figures at all (a brand new install).
FALLBACK_BOARDS = 3_000_000.0
GUARANTEED_TYPES = ("hongbao", "special")  # draws with their own advertised jackpot
BIGGEST_SHOWN = 5


# Dates


def next_regular_draw_days(after: date, count: int) -> list[date]:
    """The next ``count`` regular TOTO draw days (Monday and Thursday) strictly after ``after``."""
    out, d = [], after
    while len(out) < count:
        d += timedelta(days=1)
        if d.weekday() in C.TOTO_WEEKDAYS:
            out.append(d)
    return out


def _sg_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=SG)).astimezone(SG).date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value if isinstance(value, date) else None


def _last_stored_date(df: pd.DataFrame) -> date | None:
    if df is None or df.empty:
        return None
    return pd.Timestamp(df["draw_date"].max()).date()


# Jackpot growth


def _boards(df: pd.DataFrame, jackpot: float, rules: PrizeRules, table: pd.DataFrame,
            notes: list[str]) -> float:
    boards, _ = estimate_boards(df, jackpot, rules, table) if len(df) else (None, "")
    if boards is None:
        known = table["boards"].dropna() if len(table) else pd.Series(dtype=float)
        if len(known):
            boards = float(known.median())
        else:
            boards = FALLBACK_BOARDS
            note = "There is no sales history yet, so the chances use a rough 3 million boards per draw."
            if note not in notes:
                notes.append(note)
    return float(boards)


def _added_to_jackpot(boards: float, rules: PrizeRules) -> float:
    """Group 1's 38% of a draw's prize pool (54% of sales)."""
    return rules.group_pool_pct[1] * rules.pool_share_of_sales * boards


def _grow(jackpot: float, df: pd.DataFrame, rules: PrizeRules, table: pd.DataFrame,
          notes: list[str]) -> float:
    """Jackpot at the following draw if nobody wins this one: all of it rolls over and 38% of
    the next pool is added. Sales depend on the jackpot, so the estimate is refined once."""
    guess = jackpot + _added_to_jackpot(_boards(df, jackpot, rules, table, notes), rules)
    better = jackpot + _added_to_jackpot(_boards(df, guess, rules, table, notes), rules)
    return max(round(better, -3), rules.min_group1)  # an estimate: whole thousands


def _chance_won(boards: float) -> float:
    """Chance at least one of ``boards`` random boards matches all six numbers."""
    return float(-math.expm1(-max(boards, 0.0) / C.TOTO_COMBOS))


def estimate_next_jackpot(df: pd.DataFrame, rules: PrizeRules, table: pd.DataFrame | None = None) -> float | None:
    """The next jackpot worked out from the stored history, for when the next draw page could not
    be read: the minimum after a win or a cascade, otherwise the last jackpot plus 38% of the pool."""
    if df is None or df.empty:
        return None
    table = sales_table(df, rules) if table is None else table
    last = df.sort_values("draw_number").iloc[-1]
    notes: list[str] = []
    if int(last["g1_winners"]) > 0 or last["draw_type"] in CASCADE_TYPES:
        start = rules.min_group1
        return max(rules.min_group1, round(_added_to_jackpot(_boards(df, start, rules, table, notes), rules), -3))
    jackpot = float(last["jackpot"]) if np.isfinite(last["jackpot"]) else rules.min_group1
    return _grow(jackpot, df, rules, table, notes)


# Outlook


def _current_next_toto(next_toto: NextToto | None, df: pd.DataFrame, today: date) -> NextToto | None:
    """The next draw page info, or None when it is about a draw already held or already past."""
    if next_toto is None:
        return None
    when = _sg_date(next_toto.draw_datetime)
    last = _last_stored_date(df)
    if when is not None and ((last is not None and when <= last) or when < today):
        return None
    return next_toto


def jackpot_outlook(
    df: pd.DataFrame,
    rules: PrizeRules,
    next_toto: NextToto | None = None,
    *,
    table: pd.DataFrame | None = None,
    today: date | None = None,
    announced: list[tuple[date, str]] | tuple = (),
) -> JackpotOutlook:
    """Projection of the next draws up to the cascade, plus announced special draws.

    ``announced`` holds (date, draw type) pairs the next draw pages have announced so far
    (kept in state.json); future Hongbao, special and cascade draws among them are listed.
    """
    today = today or datetime.now(SG).date()
    table = sales_table(df, rules) if table is None else table
    notes: list[str] = []
    nt = _current_next_toto(next_toto, df, today)

    snowball = no_winner_streak(df, reset_on_cascade=True) if len(df) else 0
    if nt is not None and nt.draw_type:
        draw_type = nt.draw_type
    else:
        draw_type = "cascade" if snowball == C.TOTO_SNOWBALL_LIMIT - 1 else "normal"

    jackpot = None
    if nt is not None and nt.jackpot_estimate is not None and math.isfinite(nt.jackpot_estimate):
        jackpot = float(nt.jackpot_estimate)
    elif len(df):
        jackpot = estimate_next_jackpot(df, rules, table)
        if jackpot is not None:
            notes.append("The next draw page gave no usable jackpot, so the next jackpot is worked out "
                         "from the stored results.")

    first_day = _sg_date(nt.draw_datetime) if nt is not None else None
    if first_day is None:
        last = _last_stored_date(df)
        yesterday = today - timedelta(days=1)
        first_day = next_regular_draw_days(max(last or yesterday, yesterday), 1)[0]

    if draw_type in GUARANTEED_TYPES:
        draws = 1
        to_cascade = None
        notes.append(f"The next draw is a {'Hongbao' if draw_type == 'hongbao' else 'special'} draw with its "
                     "own advertised jackpot; if nobody wins it, it goes to the next group with winners.")
    else:
        draws = 1 if draw_type == "cascade" else max(1, C.TOTO_SNOWBALL_LIMIT - snowball)
        to_cascade = draws

    steps: list[OutlookStep] = []
    if jackpot is not None:
        days = [first_day] + next_regular_draw_days(first_day, draws - 1)
        reached, current = 1.0, jackpot
        for k in range(1, draws + 1):
            boards = _boards(df, current, rules, table, notes)
            won = _chance_won(boards)
            last_step = k == draws
            cascade = last_step and draw_type != "special"
            ev_type = "cascade" if cascade else "normal"
            ev = toto_ev_per_dollar(current, boards, rules, ev_type)["total"]
            steps.append(OutlookStep(index=k, draw_date=days[k - 1], jackpot=float(current), boards=boards,
                                     chance_reached=reached, chance_won=won, cascade=cascade,
                                     ev_per_dollar=ev))
            reached *= 1.0 - won
            if not last_step:
                current = _grow(current, df, rules, table, notes)

    special = sorted({(d, t) for d, t in announced if d is not None and d >= today and t != "normal"})
    if nt is not None and draw_type in GUARANTEED_TYPES and (first_day, draw_type) not in special:
        special = sorted(special + [(first_day, draw_type)])

    return JackpotOutlook(jackpot=jackpot, draw_type=draw_type, snowball_draws=snowball,
                          draws_to_cascade=to_cascade, steps=steps, special_draws=special, notes=notes)


def chance_won_by_cascade(outlook: JackpotOutlook) -> float | None:
    """Chance somebody wins Group 1 at one of the projected draws (before it cascades)."""
    if not outlook.steps:
        return None
    last = outlook.steps[-1]
    return 1.0 - last.chance_reached * (1.0 - last.chance_won)


# History


def _row_dict(row) -> dict:
    return {
        "draw_number": int(row["draw_number"]),
        "draw_date": pd.Timestamp(row["draw_date"]).date(),
        "jackpot": float(row["jackpot"]) if np.isfinite(row["jackpot"]) else None,
        "winners": int(row["g1_winners"]),
        "draw_type": str(row["draw_type"]),
    }


def jackpot_history(df: pd.DataFrame) -> JackpotHistory:
    """How often Group 1 is won, how long jackpots snowball and the biggest jackpots so far."""
    if df is None or df.empty:
        return JackpotHistory()
    d = df.sort_values("draw_number")
    won = d["g1_winners"].astype(int) > 0
    paid_out = won | d["draw_type"].isin(CASCADE_TYPES)

    runs, run = [], 0
    for paid in paid_out:
        run += 1
        if paid:
            runs.append(run)
            run = 0

    jackpots = d["jackpot"].astype(float)
    top = d.assign(_j=jackpots).dropna(subset=["_j"]).sort_values(["_j", "draw_number"], ascending=[False, False])
    won_rows = d[won]
    return JackpotHistory(
        draws=int(len(d)),
        won_draws=int(won.sum()),
        cascades=int(((d["draw_type"] == "cascade") & ~won).sum()),
        average_run=float(np.mean(runs)) if runs else None,
        typical_won=float(jackpots[won].median()) if won.any() else None,
        biggest=[_row_dict(r) for _, r in top.head(BIGGEST_SHOWN).iterrows()],
        last_won=_row_dict(won_rows.iloc[-1]) if len(won_rows) else None,
    )
