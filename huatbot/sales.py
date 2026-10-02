"""Sales per TOTO draw, worked out from the published winning shares.

Singapore Pools does not publish sales, but the prize rules give them away: Group 3 gets
5.5% of the prize pool and the pool is 54% of sales, so

    pool   = Group 3 share x Group 3 winners / 5.5%   (Group 4 and 3% as the fallback)
    boards = pool / 54%

A group is only used when its amount is pure pool money: it must have winners, must not
hold a snowball from the previous draw (that draw had no winner in the group) and must not
have received a cascaded jackpot. The buy signal and the jackpot outlook use these figures
to estimate how many boards a draw sells at a given jackpot.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .models import PrizeRules

log = logging.getLogger(__name__)

# Draw types whose unwon Group 1 jackpot cascades into the next group with winners.
CASCADE_TYPES = ("cascade", "hongbao")

COLUMNS = ["basis", "pool", "boards", "excluded_reason"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) and not df["draw_number"].is_monotonic_increasing:
        df = df.sort_values("draw_number", kind="stable")
    return df


def _group_problem(g: int, share: pd.Series, winners: pd.Series, prev_winners: pd.Series,
                   lands: pd.Series) -> pd.Series:
    """Plain reason why group ``g`` cannot give a clean pool estimate ("" when it can)."""
    reason = pd.Series("", index=share.index, dtype=object)
    # Assign in reverse priority so the most basic problem wins.
    reason[lands.to_numpy()] = f"Group {g} received the cascaded jackpot"
    reason[(prev_winners == 0).to_numpy()] = f"Group {g} held a snowball from the previous draw"
    amounts = share.to_numpy(dtype=float)
    reason[~np.isfinite(amounts) | (amounts <= 0)] = f"Group {g} share amount is missing"
    reason[(winners <= 0).to_numpy()] = f"Group {g} had no winner"
    return reason


def sales_table(df: pd.DataFrame, rules: PrizeRules) -> pd.DataFrame:
    """Per draw prize pool and boards sold, indexed by draw number.

    Columns: basis ("G3", "G4" or "" when neither group is clean), pool, boards (NaN when
    unknown) and excluded_reason (plain text, "" when the draw has figures).
    """
    df = _sorted(df)
    index = pd.Index(df["draw_number"].to_numpy(dtype=np.int64), name="draw_number")
    if len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype=object if c in ("basis", "excluded_reason") else float)
                             for c in COLUMNS}, index=index)

    d = df.reset_index(drop=True)
    w = {g: d[f"g{g}_winners"].astype(np.int64) for g in (1, 2, 3, 4)}
    share = {g: d[f"g{g}_share"].astype(float) for g in (3, 4)}

    # Winners in the same group one draw earlier, looked up by draw number. NaN when that draw
    # is not stored: then a snowball cannot be seen and none is assumed.
    prev = {}
    for g in (3, 4):
        lookup = pd.Series(w[g].to_numpy(), index=d["draw_number"].to_numpy())
        prev[g] = pd.Series((d["draw_number"] - 1).map(lookup).to_numpy(dtype=float))

    # An unwon cascade or Hongbao jackpot goes to the highest group below Group 1 with winners.
    unwon = d["draw_type"].isin(CASCADE_TYPES) & (w[1] == 0) & (w[2] == 0)
    lands = {3: unwon & (w[3] > 0), 4: unwon & (w[3] == 0) & (w[4] > 0)}

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

    # A pool that is not a positive number (nonsense percentages) gives no estimate at all.
    bad = (use3 | use4) & ~(np.isfinite(pool) & (pool > 0))
    pool[bad] = np.nan
    basis[bad] = ""
    reason[bad] = "prize pool figures are not usable"

    with np.errstate(divide="ignore", invalid="ignore"):
        boards = pool / rules.pool_share_of_sales

    out = pd.DataFrame({"basis": basis, "pool": pool, "boards": boards, "excluded_reason": reason},
                       index=index)
    n_bad = int(np.isnan(pool).sum())
    if n_bad:
        log.debug("sales_table: no sales figures for %d of %d draws", n_bad, len(out))
    return out
