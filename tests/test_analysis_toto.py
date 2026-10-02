"""Tests for huatbot.analysis_toto: handmade frames with known answers plus synthetic history."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from huatbot import analysis_toto as A
from huatbot import constants as C
from huatbot.models import PrizeRules
from huatbot.store import normalise_toto
from huatbot.synth import DEFAULT_POPULAR, synth_toto

DASHES = re.compile("[-–—]")


def frame(draws: list[list[int]], start: int = 1000, **extra_cols) -> pd.DataFrame:
    """A toto frame from lists of six numbers (other columns filled by normalise_toto)."""
    rows = []
    for i, nums in enumerate(draws):
        row = {"draw_number": start + i, "draw_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)}
        row.update({f"n{k + 1}": v for k, v in enumerate(sorted(nums))})
        row["additional"] = next(x for x in range(1, 50) if x not in nums)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col, values in extra_cols.items():
        df[col] = values
    return normalise_toto(df)


@pytest.fixture(scope="module")
def big_synth():
    return synth_toto(n_draws=1200)


# Counts


def test_number_matrix_marks_the_six_numbers():
    df = frame([[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 45, 49]])
    m = A.number_matrix(df)
    assert m.shape == (2, 49) and m.dtype == bool
    assert m.sum(axis=1).tolist() == [6, 6]
    assert np.flatnonzero(m[0]).tolist() == [0, 1, 2, 3, 4, 5]
    assert (np.flatnonzero(m[1]) + 1).tolist() == [10, 20, 30, 40, 45, 49]
    assert A.number_matrix(frame([])).shape == (0, 49)


def test_frequency_and_table_windows():
    # 20 draws of 1..6, then 50 of 13..18, then the latest 50 of 7..12.
    draws = [[1, 2, 3, 4, 5, 6]] * 20 + [[13, 14, 15, 16, 17, 18]] * 50 + [[7, 8, 9, 10, 11, 12]] * 50
    df = frame(draws)
    freq = A.frequency(df)
    assert list(freq.index) == list(range(1, 50))
    assert freq[1] == 20 and freq[13] == 50 and freq[7] == 50 and freq[30] == 0
    assert freq.sum() == 6 * len(draws)
    assert A.frequency(df, last=1)[7] == 1 and A.frequency(df, last=1)[1] == 0

    table = A.frequency_table(df)
    assert list(table.columns) == ["all", "last100", "last50"]
    assert list(table.index) == list(range(1, 50))
    assert table.loc[1].tolist() == [20, 0, 0]
    assert table.loc[13].tolist() == [50, 50, 0]
    assert table.loc[7].tolist() == [50, 50, 50]
    assert table.loc[49].tolist() == [0, 0, 0]


def test_overdue_counts_draws_since_last_seen():
    df = frame([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 12, 13, 14, 15, 16]])
    # Shuffled rows must give the same answer: the function orders by draw number.
    shuffled = df.sample(frac=1.0, random_state=3)
    for data in (df, shuffled):
        gaps = A.overdue(data)
        assert list(gaps.index) == list(range(1, 50))
        assert gaps[2] == 0 and gaps[16] == 0  # in the latest draw
        assert gaps[1] == 1 and gaps[7] == 1
        assert gaps[3] == 2 and gaps[6] == 2
        assert gaps[17] == 3 and gaps[49] == 3  # never seen = len(df)
    assert (A.overdue(frame([])) == 0).all()


def test_top_pairs_known_counts_and_tie_order():
    df = frame([[1, 2, 3, 4, 5, 6], [1, 2, 3, 10, 11, 12], [1, 2, 20, 21, 22, 23]])
    pairs = A.top_pairs(df, k=4)
    assert pairs == [((1, 2), 3), ((1, 3), 2), ((2, 3), 2), ((1, 4), 1)]
    assert len(A.top_pairs(df)) == 10
    assert A.top_pairs(df, k=0) == []


# Shape


def test_shape_stats_known_answers():
    df = frame([
        [1, 2, 3, 26, 27, 28],      # 3 odd, 3 low, sum 87
        [5, 6, 7, 30, 31, 32],      # 3 odd, 3 low, sum 111
        [2, 4, 6, 8, 10, 12],       # 0 odd, 6 low, sum 42
        [25, 27, 29, 31, 33, 35],   # 6 odd, 0 low, sum 180
    ])
    s = A.shape_stats(df)
    assert s["n_draws"] == 4
    assert s["odd_dist"] == {0: 0.25, 1: 0.0, 2: 0.0, 3: 0.5, 4: 0.0, 5: 0.0, 6: 0.25}
    assert s["low_dist"] == {0: 0.25, 1: 0.0, 2: 0.0, 3: 0.5, 4: 0.0, 5: 0.0, 6: 0.25}
    assert s["typical_odd"] == 3 and s["typical_low"] == 3
    assert s["sum_q25"] == pytest.approx(75.75)
    assert s["sum_median"] == pytest.approx(99.0)
    assert s["sum_q75"] == pytest.approx(128.25)


def test_shape_stats_mode_ties_go_towards_three():
    # Odd counts 2,4,2,4 and low counts 2,4,4,2: equal distance from 3, so the lower count wins.
    df = frame([[1, 3, 26, 28, 30, 32], [1, 3, 5, 7, 26, 28], [1, 2, 3, 4, 26, 28], [1, 3, 25, 27, 30, 32]])
    s = A.shape_stats(df)
    assert s["odd_dist"][2] == s["odd_dist"][4] == 0.5
    assert s["low_dist"][2] == s["low_dist"][4] == 0.5
    assert s["typical_odd"] == 2 and s["typical_low"] == 2
    # Odd counts 1,1,4,4: 4 is closer to the expected 3, so it wins the tie.
    df = frame([[1, 2, 4, 6, 8, 10], [3, 2, 4, 6, 8, 10], [1, 3, 5, 7, 2, 4], [1, 3, 5, 7, 2, 6]])
    assert A.shape_stats(df)["typical_odd"] == 4


def test_shape_stats_empty_history_uses_exact_theory():
    s = A.shape_stats(frame([]))
    assert s["n_draws"] == 0 and s["typical_odd"] == 3 and s["typical_low"] == 3
    # The sum of 6 numbers from 1..49 is symmetric around 150.
    assert s["sum_median"] == 150
    assert s["sum_q25"] + s["sum_q75"] == 300
    assert 120 < s["sum_q25"] < 135


def test_shape_stats_on_synthetic_history(toto_df):
    s = A.shape_stats(toto_df)
    assert sum(s["odd_dist"].values()) == pytest.approx(1.0)
    assert sum(s["low_dist"].values()) == pytest.approx(1.0)
    assert s["typical_odd"] in (2, 3, 4) and s["typical_low"] in (2, 3, 4)
    assert s["sum_q25"] < s["sum_median"] < s["sum_q75"]


# Fairness test


def test_chi_square_even_spread_is_chance():
    # 49 draws of 6 consecutive numbers (mod 49): every number appears exactly 6 times.
    draws = [[(6 * i + k) % 49 + 1 for k in range(6)] for i in range(49)]
    res = A.chi_square_numbers(frame(draws))
    assert res["stat"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["dof"] == 48 and res["n_draws"] == 49
    assert res["verdict"] == ("The spread of numbers is consistent with pure chance (p = 1.00), "
                              "so hot and cold numbers are just noise.")


def test_chi_square_lopsided_spread_is_flagged_but_not_predictive():
    df = frame([[1, 2, 3, 4, 5, 6]] * 100)
    res = A.chi_square_numbers(df)
    expected = stats.chisquare(A.frequency(df).to_numpy(), np.full(49, 600 / 49))
    assert res["stat"] == pytest.approx(expected.statistic)
    assert res["p_value"] < 0.05
    assert "unusual" in res["verdict"] or "more uneven" in res["verdict"]
    assert "does not make any number more likely in the next draw" in res["verdict"]
    assert "p below 0.0001" in res["verdict"]
    assert not DASHES.search(res["verdict"])


def test_chi_square_on_synthetic_history(toto_df):
    res = A.chi_square_numbers(toto_df)
    assert res["n_draws"] == len(toto_df) and res["dof"] == 48
    assert 0.0 <= res["p_value"] <= 1.0
    assert re.search(r"\(p = \d\.\d\d\)|\(p = 0\.\d+\)|\(p below 0\.0001\)", res["verdict"])
    assert res["verdict"].endswith(".") and res["verdict"].count(".") <= 3
    assert not DASHES.search(res["verdict"])


def test_chi_square_empty_history():
    res = A.chi_square_numbers(frame([]))
    assert res["n_draws"] == 0 and np.isnan(res["p_value"])
    assert not DASHES.search(res["verdict"])


def test_format_p_never_uses_scientific_notation():
    assert A.format_p(0.4123) == "0.41"
    assert A.format_p(0.0012) == "0.0012"
    assert A.format_p(3e-9) == "below 0.0001"
    for p in (0.5, 0.049, 0.002, 1e-5, 1e-300):
        assert "e" not in A.format_p(p) or A.format_p(p).startswith("below")
        assert not DASHES.search(A.format_p(p))


# Crowd table


CLEAN = {g: (s, w) for g, (s, w) in {
    1: (np.nan, 0), 2: (90_000.0, 2), 3: (2_000.0, 120), 4: (300.0, 450),
    5: (50.0, 9000), 6: (25.0, 12000), 7: (10.0, 160_000)}.items()}


def crowd_frame(specs: list[dict]) -> pd.DataFrame:
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


def test_crowd_table_formulas_on_clean_draw(rules):
    table = A.crowd_table(crowd_frame([{"draw": 10}]), rules)
    assert list(table.columns) == ["basis", "pool", "boards", "expected_g7", "crowd_ratio", "excluded_reason"]
    assert table.index.name == "draw_number" and list(table.index) == [10]
    row = table.loc[10]
    pool = 2_000.0 * 120 / 0.055
    boards = pool / 0.54
    expected = boards * 229_600 / 13_983_816
    assert row["basis"] == "G3" and row["excluded_reason"] == ""
    assert row["pool"] == pytest.approx(pool)
    assert row["boards"] == pytest.approx(boards)
    assert row["expected_g7"] == pytest.approx(expected)
    assert row["crowd_ratio"] == pytest.approx(160_000 / expected)


def test_crowd_table_uses_rules_percentages():
    rules = PrizeRules(pool_share_of_sales=0.5, group_pool_pct={1: 0.38, 2: 0.08, 3: 0.06, 4: 0.03})
    row = A.crowd_table(crowd_frame([{"draw": 10}]), rules).loc[10]
    assert row["pool"] == pytest.approx(2_000.0 * 120 / 0.06)
    assert row["boards"] == pytest.approx(row["pool"] / 0.5)


def test_crowd_table_switches_to_group_4(rules):
    df = crowd_frame([
        {"draw": 20},
        {"draw": 21, 3: (np.nan, 0)},                       # G3 no winner
        {"draw": 22},                                       # previous draw G3 unwon: snowball in G3
        {"draw": 23, "type": "cascade", 2: (np.nan, 0)},    # unwon cascade jackpot landed in G3
        {"draw": 24, "type": "hongbao"},                    # jackpot landed in G2: G3 still clean
        {"draw": 25, "type": "cascade", 1: (5e6, 1), 2: (np.nan, 0)},  # Group 1 won: no cascade
    ])
    t = A.crowd_table(df, rules)
    assert t["basis"].tolist() == ["G3", "G4", "G4", "G4", "G3", "G3"]
    assert (t["excluded_reason"] == "").all()
    g4_pool = 300.0 * 450 / 0.03
    assert t.loc[21, "pool"] == pytest.approx(g4_pool)
    assert t.loc[23, "pool"] == pytest.approx(g4_pool)
    assert t["crowd_ratio"].notna().all()


def test_crowd_table_excludes_when_no_clean_group(rules):
    df = crowd_frame([
        {"draw": 30, 4: (np.nan, 0)},                                         # G4 unwon, G3 fine
        {"draw": 31, 3: (np.nan, 0)},                                         # G3 unwon, G4 snowball
        {"draw": 32, "type": "hongbao", 2: (np.nan, 0), 3: (np.nan, 0)},      # G3 unwon, jackpot into G4
        {"draw": 33, 7: (np.nan, 0)},                                         # Group 7 not captured
        {"draw": 34, 3: (np.nan, 12), 4: (np.nan, 5)},                        # share amounts missing
    ])
    t = A.crowd_table(df, rules)
    assert t.loc[30, "basis"] == "G3" and np.isfinite(t.loc[30, "crowd_ratio"])
    for d in (31, 32, 33, 34):
        assert np.isnan(t.loc[d, "crowd_ratio"]), d
        assert t.loc[d, "excluded_reason"], d
    for d in (31, 32, 34):
        assert t.loc[d, "basis"] == "" and np.isnan(t.loc[d, "pool"]), d
    assert t.loc[31, "excluded_reason"] == ("Group 3 had no winner and Group 4 held a snowball "
                                            "from the previous draw")
    assert t.loc[32, "excluded_reason"] == "Group 3 had no winner and Group 4 received the cascaded jackpot"
    # Draw 33: G3 holds the snowball from draw 32, so the pool comes from G4 and is still a
    # valid sales estimate, but without Group 7 figures there is no crowd ratio.
    assert t.loc[33, "basis"] == "G4" and np.isfinite(t.loc[33, "boards"])
    assert t.loc[33, "excluded_reason"] == "no Group 7 winners recorded"
    assert "share amount is missing" in t.loc[34, "excluded_reason"]
    for reason in t["excluded_reason"]:
        assert not DASHES.search(reason)


def test_crowd_table_gap_before_draw_means_no_snowball_seen(rules):
    # Draw 40 had no G3 winner but 42 follows a gap (41 missing), so 42 keeps G3.
    df = crowd_frame([{"draw": 40, 3: (np.nan, 0)}, {"draw": 42}])
    t = A.crowd_table(df, rules)
    assert t.loc[42, "basis"] == "G3"


def test_crowd_table_empty(rules):
    t = A.crowd_table(frame([]), rules)
    assert len(t) == 0 and "crowd_ratio" in t.columns


def test_crowd_ratio_mean_close_to_one_after_exclusions(big_synth, rules):
    table = A.crowd_table(big_synth, rules)
    ratios = table["crowd_ratio"].dropna()
    assert len(ratios) > 1100
    assert abs(ratios.mean() - 1.0) <= 0.02
    # The synthetic history contains cascade draws whose jackpot fell into Group 3. They must
    # have been moved to Group 4, otherwise those draws look massively under crowded.
    landed_g3 = (big_synth["draw_type"].isin(["cascade", "hongbao"]) & (big_synth["g1_winners"] == 0)
                 & (big_synth["g2_winners"] == 0)).to_numpy()
    assert landed_g3.sum() > 5
    assert (table["basis"].to_numpy()[landed_g3] != "G3").all()
    # A naive Group 3 estimate on those draws is badly biased (sanity check of the test itself).
    naive = big_synth["g7_winners"] / (big_synth["g3_share"] * big_synth["g3_winners"] / 0.055 / 0.54
                                       * 229_600 / 13_983_816)
    assert naive.to_numpy()[landed_g3].mean() < 0.5
    assert ratios.to_numpy()[landed_g3[table["crowd_ratio"].notna().to_numpy()]].mean() > 0.8


# Crowd scores


def test_crowd_scores_find_planted_preferences(big_synth, rules):
    scores, diag = A.crowd_scores(big_synth, rules)
    assert list(scores.index) == list(range(1, 50))
    ranked = scores.sort_values(ascending=False).index.tolist()
    popular = [7, 8, 9, 18, 28]
    unpopular = [13, 44, 49]
    assert set(popular) <= set(ranked[:10])
    assert set(unpopular) <= set(ranked[-12:])
    # Planted strengths (DEFAULT_POPULAR x birthday boost) leave a wide gap to every other number,
    # so they should be exactly the top 5 and bottom 3.
    assert all(DEFAULT_POPULAR[n] > 1.5 for n in popular) and all(DEFAULT_POPULAR[n] < 0.7 for n in unpopular)
    assert set(ranked[:5]) == set(popular)
    assert set(ranked[-3:]) == set(unpopular)
    # Centred indicators sum to zero, so the ridge scores do too: positive = crowded.
    assert scores.sum() == pytest.approx(0.0, abs=1e-9)
    assert scores[7] > 0 > scores[44]

    assert diag["n_draws"] + diag["n_excluded"] == len(big_synth)
    assert diag["ok"] is True and diag["warning"] is None
    assert diag["alpha"] == A.DEFAULT_RIDGE_ALPHA
    assert 0.5 < diag["r2"] <= 1.0
    assert abs(diag["mean_ratio"] - 1) <= 0.02
    assert diag["median_ratio"] > 0


def test_crowd_scores_match_closed_form_ridge(toto_df, rules):
    alpha = 7.5
    scores, diag = A.crowd_scores(toto_df, rules, alpha=alpha)
    table = A.crowd_table(toto_df, rules)
    use = table["crowd_ratio"].notna().to_numpy()
    x = A.number_matrix(toto_df)[use].astype(float)
    y = table["crowd_ratio"].to_numpy()[use]
    xc, yc = x - x.mean(axis=0), y - y.mean()
    # Ridge = least squares on the data stacked with sqrt(alpha) * I.
    aug_x = np.vstack([xc, np.sqrt(alpha) * np.eye(49)])
    aug_y = np.concatenate([yc, np.zeros(49)])
    beta = np.linalg.lstsq(aug_x, aug_y, rcond=None)[0]
    np.testing.assert_allclose(scores.to_numpy(), beta, atol=1e-9)
    assert diag["alpha"] == alpha
    resid = yc - xc @ beta
    assert diag["r2"] == pytest.approx(1 - resid @ resid / (yc @ yc))


def test_crowd_scores_more_alpha_shrinks(toto_df, rules):
    small, _ = A.crowd_scores(toto_df, rules, alpha=0.5)
    large, _ = A.crowd_scores(toto_df, rules, alpha=500.0)
    assert np.abs(large).sum() < np.abs(small).sum()


def test_crowd_scores_accepts_precomputed_table_for_earlier_draws(toto_df, rules):
    full_table = A.crowd_table(toto_df, rules)
    early = toto_df.iloc[:300]
    a, da = A.crowd_scores(early, rules, table=full_table)
    b, db = A.crowd_scores(early, rules)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
    assert da == db


def test_crowd_scores_under_30_usable_draws(rules):
    df = synth_toto(n_draws=40, seed=3)
    df.loc[df.index[:15], "g7_winners"] = 0  # 15 draws become unusable
    scores, diag = A.crowd_scores(df, rules)
    assert (scores == 0).all() and list(scores.index) == list(range(1, 50))
    assert diag["n_draws"] == 25 and diag["n_excluded"] == 15
    assert diag["r2"] == 0.0
    assert "Only 25 usable draws" in diag["warning"]
    assert not DASHES.search(diag["warning"])

    scores, diag = A.crowd_scores(frame([]), rules)
    assert (scores == 0).all()
    assert diag["n_draws"] == 0 and diag["mean_ratio"] is None and diag["ok"] is False
    assert diag["warning"]


def test_crowd_scores_warns_when_ratio_far_from_one(toto_df, rules):
    df = toto_df.copy()
    df["g7_winners"] = (df["g7_winners"] * 0.8).round().astype("int64")
    _, diag = A.crowd_scores(df, rules)
    assert diag["ok"] is False
    assert diag["mean_ratio"] == pytest.approx(0.8, abs=0.03)
    assert diag["warning"].startswith(f"Average crowd ratio is {diag['mean_ratio']:.2f}, not close to 1")
    assert "54% prize pool assumption may be off" in diag["warning"]
    assert not DASHES.search(diag["warning"])


def test_constants_used_by_crowd_math():
    assert C.TOTO_GROUP_COMBOS[7] == 229_600 and C.TOTO_COMBOS == 13_983_816
