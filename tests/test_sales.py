"""Sales per draw from the published winning shares (sales.sales_table)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from huatbot.models import PrizeRules
from huatbot.sales import sales_table
from huatbot.store import normalise_toto
from huatbot.synth import synth_toto

DASHES = re.compile("[-–—]")

CLEAN = {
    1: (np.nan, 0), 2: (90_000.0, 2), 3: (2_000.0, 120), 4: (300.0, 450),
    5: (50.0, 9000), 6: (25.0, 12000), 7: (10.0, 160_000),
}


def frame(specs: list[dict]) -> pd.DataFrame:
    """Rows from {"draw": n, "type": ..., g: (share, winners)} overriding the CLEAN groups."""
    rows = []
    for k, spec in enumerate(specs):
        row = {"draw_number": spec["draw"], "draw_date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=3 * k),
               "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "additional": 7,
               "jackpot": spec.get("jackpot", 1_000_000.0), "draw_type": spec.get("type", "normal")}
        for g in range(1, 8):
            share, winners = spec.get(g, CLEAN[g])
            row[f"g{g}_share"], row[f"g{g}_winners"] = share, winners
        rows.append(row)
    return normalise_toto(pd.DataFrame(rows))


def test_formulas_on_a_clean_draw(rules):
    table = sales_table(frame([{"draw": 10}]), rules)
    assert list(table.columns) == ["basis", "pool", "boards", "excluded_reason"]
    assert table.index.name == "draw_number" and list(table.index) == [10]
    row = table.loc[10]
    pool = 2_000.0 * 120 / 0.055
    assert row["basis"] == "G3" and row["excluded_reason"] == ""
    assert row["pool"] == pytest.approx(pool)
    assert row["boards"] == pytest.approx(pool / 0.54)


def test_uses_the_rules_percentages():
    rules = PrizeRules(pool_share_of_sales=0.5, group_pool_pct={1: 0.38, 2: 0.08, 3: 0.06, 4: 0.03})
    row = sales_table(frame([{"draw": 10}]), rules).loc[10]
    assert row["pool"] == pytest.approx(2_000.0 * 120 / 0.06)
    assert row["boards"] == pytest.approx(row["pool"] / 0.5)


def test_switches_to_group_4_when_group_3_is_not_clean(rules):
    df = frame([
        {"draw": 20},
        {"draw": 21, 3: (np.nan, 0)},                       # G3 no winner
        {"draw": 22},                                       # previous draw G3 unwon: snowball in G3
        {"draw": 23, "type": "cascade", 2: (np.nan, 0)},    # unwon cascade jackpot landed in G3
        {"draw": 24, "type": "hongbao"},                    # jackpot landed in G2: G3 still clean
        {"draw": 25, "type": "cascade", 1: (5e6, 1), 2: (np.nan, 0)},  # Group 1 won: no cascade
    ])
    t = sales_table(df, rules)
    assert t["basis"].tolist() == ["G3", "G4", "G4", "G4", "G3", "G3"]
    assert (t["excluded_reason"] == "").all()
    g4_pool = 300.0 * 450 / 0.03
    assert t.loc[21, "pool"] == pytest.approx(g4_pool)
    assert t.loc[23, "pool"] == pytest.approx(g4_pool)


def test_no_figures_when_no_group_is_clean(rules):
    df = frame([
        {"draw": 30, 4: (np.nan, 0)},                                         # G4 unwon, G3 fine
        {"draw": 31, 3: (np.nan, 0)},                                         # G3 unwon, G4 snowball
        {"draw": 32, "type": "hongbao", 2: (np.nan, 0), 3: (np.nan, 0)},      # G3 unwon, jackpot into G4
        {"draw": 34, 3: (np.nan, 12), 4: (np.nan, 5)},                        # share amounts missing
    ])
    t = sales_table(df, rules)
    assert t.loc[30, "basis"] == "G3" and np.isfinite(t.loc[30, "boards"])
    for d in (31, 32, 34):
        assert t.loc[d, "basis"] == "" and np.isnan(t.loc[d, "pool"]) and np.isnan(t.loc[d, "boards"]), d
        assert t.loc[d, "excluded_reason"], d
    assert t.loc[31, "excluded_reason"] == ("Group 3 had no winner and Group 4 held a snowball "
                                            "from the previous draw")
    assert t.loc[32, "excluded_reason"] == "Group 3 had no winner and Group 4 received the cascaded jackpot"
    assert "share amount is missing" in t.loc[34, "excluded_reason"]
    for reason in t["excluded_reason"]:
        assert not DASHES.search(reason)


def test_gap_before_a_draw_means_no_snowball_is_seen(rules):
    # Draw 40 had no G3 winner but 42 follows a gap (41 missing), so 42 keeps G3.
    t = sales_table(frame([{"draw": 40, 3: (np.nan, 0)}, {"draw": 42}]), rules)
    assert t.loc[42, "basis"] == "G3"


def test_empty_history(rules):
    t = sales_table(frame([]), rules)
    assert len(t) == 0 and list(t.columns) == ["basis", "pool", "boards", "excluded_reason"]


def test_synthetic_sales_are_plausible_and_grow_with_the_jackpot(rules):
    # synth_toto sells more boards when the jackpot is bigger; the estimates must show that.
    df = synth_toto(n_draws=400)
    t = sales_table(df, rules)
    boards = t["boards"].dropna()
    assert len(boards) > 0.9 * len(df)
    assert boards.between(1_000_000, 30_000_000).all()
    jackpots = df.set_index("draw_number").loc[boards.index, "jackpot"]
    assert np.corrcoef(jackpots, boards)[0, 1] > 0.5
