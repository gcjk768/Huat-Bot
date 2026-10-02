"""Parsers: static fixtures that look like the real site, plus htmlgen round trips."""
from __future__ import annotations

import math
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import parse as P
from huatbot.models import FOURD_NUMBER_COLUMNS
from huatbot.synth import SG, synth_fourd, synth_toto
from tests import htmlgen as H

TOTO_RESULT_KEYS = (
    ["draw_number", "draw_date"] + [f"n{i}" for i in range(1, 7)] + ["additional", "jackpot"]
    + [f"g{g}_{k}" for g in range(1, 8) for k in ("share", "winners")]
)


def _same(a, b) -> bool:
    """Equality that treats NaN == NaN and compares dates with Timestamps."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(b, pd.Timestamp):
        b = b.date()
    if isinstance(a, pd.Timestamp):
        a = a.date()
    if isinstance(b, (np.integer, np.floating)):
        b = b.item()
    return a == b


# sppl


def test_sppl_matches_site_encoding():
    assert P.sppl(4123) == "RHJhd051bWJlcj00MTIz"
    assert P.sppl(5432) == "RHJhd051bWJlcj01NDMy"


@pytest.mark.parametrize("n", [1, 99, 2995, 4123, 5432, 10000, 123456])
def test_sppl_round_trip(n):
    assert P.draw_number_from_sppl(P.sppl(n)) == n


@pytest.mark.parametrize("value", [
    "RHJhd051bWJlcj00MTIz",
    "sppl=RHJhd051bWJlcj00MTIz",
    "?sppl=RHJhd051bWJlcj00MTIz",
    "https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl=RHJhd051bWJlcj00MTIz",
    "RHJhd051bWJlcj00MTIz%3D%3D",
    "  RHJhd051bWJlcj00MTIz \n",
])
def test_draw_number_from_sppl_forms(value):
    assert P.draw_number_from_sppl(value) == 4123


def test_draw_number_from_sppl_without_padding():
    # 5 digit numbers need padding; the site may drop the "=" characters
    raw = P.sppl(12345)
    assert P.draw_number_from_sppl(raw.rstrip("=")) == 12345


@pytest.mark.parametrize("bad", ["", "sppl=", "hello", "SGVsbG8gd29ybGQ=", None])
def test_draw_number_from_sppl_rejects_garbage(bad):
    with pytest.raises(P.ParseError):
        P.draw_number_from_sppl(bad)


# small value parsers


@pytest.mark.parametrize("text, expected", [
    ("Thu, 01 Oct 2026", date(2026, 10, 1)),
    ("Thu,01 Oct 2026", date(2026, 10, 1)),
    ("  Mon,\n   05 Oct 2026  ", date(2026, 10, 5)),
    ("Thursday, 1 October 2026", date(2026, 10, 1)),
    ("1 Oct 2026", date(2026, 10, 1)),
    ("01-Oct-2026", date(2026, 10, 1)),
    ("1st Sept 2026", date(2026, 9, 1)),
    ("Oct 1, 2026", date(2026, 10, 1)),
    ("2026-10-01", date(2026, 10, 1)),
    ("01/10/2026", date(2026, 10, 1)),
    ("Draw date: Sat, 03 Oct 2026, 6.30pm", date(2026, 10, 3)),
])
def test_parse_draw_date_variants(text, expected):
    assert P.parse_draw_date(text) == expected


@pytest.mark.parametrize("bad", ["", "To be announced", "Draw No. 4123", "31 Feb 2026"])
def test_parse_draw_date_rejects(bad):
    with pytest.raises(P.ParseError):
        P.parse_draw_date(bad)


@pytest.mark.parametrize("text, expected", [
    ("$1,234,567", 1234567.0),
    (" $1,185,926 ", 1185926.0),
    ("S$50", 50.0),
    ("$12.50", 12.5),
    ("$1,000,000 est", 1000000.0),
    ("$1.5 million", 1500000.0),
    ("$2 Million", 2000000.0),
    ("$10 per winning share", 10.0),
    ("-", None),
    ("", None),
    ("  ", None),
    (None, None),
    ("To be announced", None),
])
def test_parse_money(text, expected):
    assert P.parse_money(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("1,234", 1234), ("2", 2), (" 84,113 ", 84113), ("-", 0), ("", 0), (None, 0), ("none", 0),
])
def test_parse_int(text, expected):
    assert P.parse_int(text) == expected


# draw lists


def test_toto_draw_list_fixture(fixture_html):
    draws = P.parse_draw_list(fixture_html("toto_draw_list.html"))
    assert len(draws) == 12
    assert draws[0] == (4123, date(2026, 10, 1))
    assert draws[-1] == (4112, date(2026, 8, 24))
    assert [n for n, _ in draws] == sorted((n for n, _ in draws), reverse=True)
    for n, d in draws:  # TOTO draws are on Mondays and Thursdays
        assert d.weekday() in C.TOTO_WEEKDAYS


def test_fourd_draw_list_fixture(fixture_html):
    draws = P.parse_draw_list(fixture_html("fourd_draw_list.html"))
    assert draws[0] == (5432, date(2026, 9, 30))
    assert len(draws) == 10
    assert all(d.weekday() in C.FOURD_WEEKDAYS for _, d in draws)


def test_cascade_list_fixture(fixture_html):
    draws = P.parse_draw_list(fixture_html("toto_cascade_list.html"))
    assert [n for n, _ in draws] == [4108, 4072, 4061, 4033, 3998]


def test_draw_list_decodes_sppl_when_value_is_not_a_number():
    html = """<select>
      <option querystring="sppl=RHJhd051bWJlcj00MTIz" value="RHJhd051bWJlcj00MTIz">Thu, 01 Oct 2026</option>
      <option value="RHJhd051bWJlcj00MTIy">Mon, 28 Sep 2026</option>
      <option querystring="sppl=RHJhd051bWJlcj00MTIx" value="">Thu, 24 Sep 2026</option>
      <option value="">Select a draw</option>
      <option value="4120">not a date</option>
      <option value="4120">Mon, 21 Sep 2026</option>
    </select>"""
    assert P.parse_draw_list(html) == [
        (4123, date(2026, 10, 1)), (4122, date(2026, 9, 28)), (4121, date(2026, 9, 24)), (4120, date(2026, 9, 21)),
    ]


def test_draw_list_sorted_descending_even_if_page_is_not():
    html = H.draw_list_html([(10, date(2026, 1, 1)), (12, date(2026, 1, 8)), (11, date(2026, 1, 5))])
    html = html.replace("\n", " ")
    assert [n for n, _ in P.parse_draw_list(html)] == [12, 11, 10]


def test_draw_list_round_trip(toto_df):
    rows = toto_df.tail(150)
    for as_sppl in (False, True):
        parsed = P.parse_draw_list(H.draw_list_html(rows, value_as_sppl=as_sppl))
        expected = sorted(((int(n), d.date()) for n, d in zip(rows["draw_number"], rows["draw_date"])), reverse=True)
        assert parsed == expected


def test_empty_draw_list():
    assert P.parse_draw_list("") == []
    assert P.parse_draw_list("<html><body>Service unavailable</body></html>") == []


# TOTO result pages


def test_toto_result_with_winner_fixture(fixture_html):
    r = P.parse_toto_result(fixture_html("toto_result_with_winner.html"))
    assert set(r) == set(TOTO_RESULT_KEYS)
    assert r["draw_number"] == 4123
    assert r["draw_date"] == date(2026, 10, 1)
    assert [r[f"n{i}"] for i in range(1, 7)] == [3, 11, 19, 27, 38, 45]
    assert r["additional"] == 7
    assert r["jackpot"] == 2371852.0
    assert (r["g1_share"], r["g1_winners"]) == (1185926.0, 2)
    assert (r["g2_share"], r["g2_winners"]) == (98765.0, 4)
    assert (r["g3_share"], r["g3_winners"]) == (1523.0, 92)
    assert (r["g4_share"], r["g4_winners"]) == (412.0, 198)
    assert (r["g5_share"], r["g5_winners"]) == (50.0, 4872)
    assert (r["g6_share"], r["g6_winners"]) == (25.0, 6301)
    assert (r["g7_share"], r["g7_winners"]) == (10.0, 84113)
    assert all(isinstance(r[f"g{g}_winners"], int) for g in range(1, 8))


def test_toto_result_no_winner_fixture(fixture_html):
    r = P.parse_toto_result(fixture_html("toto_result_no_winner.html"))
    assert r["draw_number"] == 4122
    assert r["draw_date"] == date(2026, 9, 28)
    assert [r[f"n{i}"] for i in range(1, 7)] == [2, 14, 23, 31, 40, 49]
    assert r["additional"] == 18
    assert r["jackpot"] == 1523040.0
    assert math.isnan(r["g1_share"]) and r["g1_winners"] == 0
    assert (r["g2_share"], r["g2_winners"]) == (284113.0, 1)
    assert (r["g7_share"], r["g7_winners"]) == (10.0, 41906)


def test_toto_numbers_are_sorted_even_if_page_order_is_not(toto_df):
    row = toto_df.iloc[10].copy()
    html = H.toto_result_html(row)
    nums = [int(row[f"n{i}"]) for i in range(1, 7)]
    # swap the first and last winning cells in the page
    html = html.replace(f"<td class='win1'>\n            {nums[0]}\n", "<td class='win1'>\n            XX\n")
    html = html.replace(f"<td class='win6'>\n            {nums[5]}\n", f"<td class='win6'>\n            {nums[0]}\n")
    html = html.replace("<td class='win1'>\n            XX\n", f"<td class='win1'>\n            {nums[5]}\n")
    r = P.parse_toto_result(html)
    assert [r[f"n{i}"] for i in range(1, 7)] == sorted(nums)


def test_toto_round_trip_every_synthetic_row(toto_df):
    """parse(render(row)) equals the row for every result column of every synthetic draw."""
    for _, row in toto_df.iterrows():
        parsed = P.parse_toto_result(H.toto_result_html(row))
        for key in TOTO_RESULT_KEYS:
            assert _same(parsed[key], row[key]), (int(row["draw_number"]), key, parsed[key], row[key])


def test_toto_round_trip_other_seed_and_cents():
    df = synth_toto(n_draws=120, start_draw=2995, seed=99)
    df.loc[5, "g4_share"] = 401.55  # a share with cents
    df.loc[6, "g7_winners"] = 1234567  # thousands separators
    for _, row in df.iterrows():
        parsed = P.parse_toto_result(H.toto_result_html(row))
        for key in TOTO_RESULT_KEYS:
            assert _same(parsed[key], row[key]), (key, parsed[key], row[key])


def test_toto_result_missing_pieces_raise():
    row = synth_toto(n_draws=5).iloc[-1]
    html = H.toto_result_html(row)
    with pytest.raises(P.ParseError):
        P.parse_toto_result(html.replace("class='drawNumber'", "class='somethingElse'"))
    with pytest.raises(P.ParseError):
        P.parse_toto_result(html.replace("class='win4'", "class='win4x'"))
    with pytest.raises(P.ParseError):
        P.parse_toto_result(html.replace("class='additional'", "class='extra'"))
    with pytest.raises(P.ParseError):
        P.parse_toto_result(html.replace("class='drawDate'", "class='notADate'"))
    with pytest.raises(P.ParseError):
        P.parse_toto_result("<html><body><p>The page is under maintenance</p></body></html>")


def test_toto_result_duplicate_numbers_rejected():
    row = synth_toto(n_draws=5).iloc[-1].copy()
    row["n2"] = row["n1"]
    with pytest.raises(P.ParseError):
        P.parse_toto_result(H.toto_result_html(row))


def test_toto_result_without_shares_table_raises():
    # Not published yet or a changed layout: the draw must not be stored as "no winners".
    row = synth_toto(n_draws=5).iloc[-1]
    html = H.toto_result_html(row).replace("tableWinningShares", "tableSomethingElse")
    with pytest.raises(P.ParseError, match="winning shares"):
        P.parse_toto_result(html)
    html = H.toto_result_html(row).replace("class='jackpotPrize'", "class='other'")
    assert math.isnan(P.parse_toto_result(html)["jackpot"])


def test_toto_result_with_unpublished_shares_raises():
    row = synth_toto(n_draws=5).iloc[-1].copy()
    for g in range(1, 8):
        row[f"g{g}_share"] = float("nan")
        row[f"g{g}_winners"] = 0  # every cell renders as "-"
    with pytest.raises(P.ParseError, match="not published yet"):
        P.parse_toto_result(H.toto_result_html(row))


def test_toto_result_with_missing_group_rows_raises():
    row = synth_toto(n_draws=5).iloc[-1]
    html = H.toto_result_html(row).replace("<td>Group 7</td>", "<td>Something else</td>")
    with pytest.raises(P.ParseError):
        P.parse_toto_result(html)


def test_toto_result_group_label_in_a_th_cell_is_read():
    row = synth_toto(n_draws=5).iloc[-1]
    html = H.toto_result_html(row)
    for g in range(1, 8):
        html = html.replace(f"<td>Group {g}</td>", f"<th>Group {g}</th>")
    r = P.parse_toto_result(html)
    for g in range(1, 8):
        assert r[f"g{g}_winners"] == int(row[f"g{g}_winners"])
    assert r["g7_share"] == float(row["g7_share"])


# 4D result pages


def test_fourd_result_fixture(fixture_html):
    r = P.parse_fourd_result(fixture_html("fourd_result.html"))
    assert r["draw_number"] == 5432
    assert r["draw_date"] == date(2026, 9, 30)
    assert (r["first"], r["second"], r["third"]) == ("0417", "8826", "3095")
    assert [r[f"starter_{i}"] for i in range(1, 11)] == [
        "1203", "4410", "5567", "6721", "7790", "0038", "2264", "3389", "8152", "9903"]
    assert [r[f"consolation_{i}"] for i in range(1, 11)] == [
        "0105", "1447", "2398", "3561", "4672", "5783", "6894", "7015", "8126", "9237"]
    assert set(r) == {"draw_number", "draw_date", *FOURD_NUMBER_COLUMNS}


def test_fourd_round_trip_every_synthetic_row(fourd_df):
    for _, row in fourd_df.iterrows():
        parsed = P.parse_fourd_result(H.fourd_result_html(row))
        assert parsed["draw_number"] == row["draw_number"]
        assert parsed["draw_date"] == row["draw_date"].date()
        for key in FOURD_NUMBER_COLUMNS:
            assert parsed[key] == row[key], (int(row["draw_number"]), key)


def test_fourd_blank_cells_become_empty_strings():
    row = synth_fourd(n_draws=3).iloc[-1].copy()
    row["starter_10"] = ""
    row["consolation_3"] = ""
    row["third"] = ""
    r = P.parse_fourd_result(H.fourd_result_html(row))
    assert r["starter_10"] == "" and r["consolation_3"] == "" and r["third"] == ""
    assert r["first"] == row["first"]


def test_fourd_renamed_prize_sections_raise(fixture_html):
    html = fixture_html("fourd_result.html")
    for cls in ("tbodyStarterPrizes", "tbodyConsolationPrizes"):
        with pytest.raises(P.ParseError, match="not found"):
            P.parse_fourd_result(html.replace(cls, "tbodySomethingElse"))


def test_fourd_extra_label_and_spacer_cells_do_not_shift_numbers():
    row = synth_fourd(n_draws=3).iloc[-1]
    html = H.fourd_result_html(row)
    noisy = html.replace("<tbody class='tbodyStarterPrizes'>",
                         "<tbody class='tbodyStarterPrizes'><tr><td>Starter Prizes</td><td> </td></tr>")
    r = P.parse_fourd_result(noisy)
    assert [r[f"starter_{i}"] for i in range(1, 11)] == [row[f"starter_{i}"] for i in range(1, 11)]


def test_fourd_wrong_count_or_long_number_raises():
    row = synth_fourd(n_draws=3).iloc[-1]
    html = H.fourd_result_html(row)
    nine = html.replace(f"<td>{row['starter_10']}</td>", "", 1)
    with pytest.raises(P.ParseError, match="expected 10"):
        P.parse_fourd_result(nine)
    longer = html.replace(f"<td>{row['consolation_1']}</td>", "<td>12345</td>", 1)
    with pytest.raises(P.ParseError, match="4 digit"):
        P.parse_fourd_result(longer)
    all_blank = row.copy()
    for i in range(1, 11):
        all_blank[f"starter_{i}"] = ""
    with pytest.raises(P.ParseError):
        P.parse_fourd_result(H.fourd_result_html(all_blank))


def test_fourd_missing_draw_number_raises(fixture_html):
    html = fixture_html("fourd_result.html").replace("class='drawNumber'", "class='x'")
    with pytest.raises(P.ParseError):
        P.parse_fourd_result(html)
    with pytest.raises(P.ParseError):
        P.parse_fourd_result("<table><tr><th class='drawNumber'>Draw No. 5</th>"
                             "<th class='drawDate'>Wed, 30 Sep 2026</th></tr></table>")


# next draw pages


def test_toto_next_draw_fixture(fixture_html):
    info = P.parse_toto_next_draw(fixture_html("toto_next_draw.html"))
    assert info["draw_datetime"] == datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    assert info["jackpot_estimate"] == 1_000_000.0  # not the $3,500,000 inside the script
    assert info["draw_type_hint"] is None
    assert "Next Jackpot" in info["raw_text"] and "3,500,000" not in info["raw_text"]


def test_fourd_next_draw_fixture(fixture_html):
    info = P.parse_fourd_next_draw(fixture_html("fourd_next_draw.html"))
    assert info["draw_datetime"] == datetime(2026, 10, 3, 18, 30, tzinfo=SG)
    assert "Next Draw" in info["raw_text"]


@pytest.mark.parametrize("style", ["dot", "colon", "upper"])
def test_next_draw_time_styles(style):
    dt = datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    info = P.parse_toto_next_draw(H.toto_next_draw_html(dt, 3_500_000, time_style=style))
    assert info["draw_datetime"] == dt
    assert info["jackpot_estimate"] == 3_500_000
    assert P.parse_fourd_next_draw(H.fourd_next_draw_html(dt, time_style=style))["draw_datetime"] == dt


@pytest.mark.parametrize("snippet, expected", [
    ("Next Draw Thu, 08 Oct 2026 , 9.30pm", time(21, 30)),
    ("Next Draw Thu, 08 Oct 2026, 7pm", time(19, 0)),
    ("Next Draw Thu, 08 Oct 2026 at 6:30 p.m.", time(18, 30)),
    ("Next Draw Thu, 08 Oct 2026", C.DRAW_TIME),
    ("Next Draw Thu, 08 Oct 2026 , 12.15am", time(0, 15)),
])
def test_next_draw_time_text(snippet, expected):
    info = P.parse_fourd_next_draw(f"<div><p>{snippet}</p></div>")
    assert info["draw_datetime"] == datetime.combine(date(2026, 10, 8), expected, tzinfo=SG)


@pytest.mark.parametrize("hint, word", [
    ("hongbao", "Hong Bao Draw"), ("hongbao", "Hongbao Draw"), ("hongbao", "HONG BAO DRAW"),
    ("cascade", "Cascade Draw"), ("special", "Special Draw"), (None, "Specials menu"),
])
def test_next_draw_type_hint(hint, word):
    html = f"<div><p>Next Jackpot</p><span>$12,000,000 est</span><b>{word}</b>" \
           "<p>Next Draw</p><p>Thu, 08 Oct 2026 , 6.30pm</p></div>"
    info = P.parse_toto_next_draw(html)
    assert info["draw_type_hint"] == hint
    assert info["jackpot_estimate"] == 12_000_000


def test_next_draw_hint_from_htmlgen():
    dt = datetime(2026, 2, 12, 18, 30, tzinfo=SG)
    for hint in ("cascade", "hongbao", "special"):
        assert P.parse_toto_next_draw(H.toto_next_draw_html(dt, 12_000_000, hint))["draw_type_hint"] == hint


def test_next_draw_missing_values():
    info = P.parse_toto_next_draw(H.toto_next_draw_html(None, None))
    assert info["draw_datetime"] is None and info["jackpot_estimate"] is None
    assert P.parse_fourd_next_draw("")["draw_datetime"] is None


def test_next_draw_million_and_label_preference():
    html = "<div>Last draw paid $2,000,000. Next Jackpot $1.2 million est. Next Draw Mon, 05 Oct 2026 , 6.30pm</div>"
    assert P.parse_toto_next_draw(html)["jackpot_estimate"] == 1_200_000


# prize structure pages


def test_toto_prize_structure_fixture(fixture_html):
    r = P.parse_toto_prize_structure(fixture_html("toto_prize_structure.html"))
    assert r == {
        "pool_share_of_sales": 0.54,
        "group_pool_pct": {1: 0.38, 2: 0.08, 3: 0.055, 4: 0.03},
        "fixed_prizes": {5: 50.0, 6: 25.0, 7: 10.0},
        "min_group1": 1_000_000.0,
    }


def test_toto_prize_structure_htmlgen_table_with_decoy_banner():
    # the htmlgen wrapper has a promo table "Group 1 | $5,000,000" before the real table
    r = P.parse_toto_prize_structure(H.toto_prize_structure_html())
    assert r["group_pool_pct"] == C.TOTO_GROUP_POOL_PCT
    assert r["fixed_prizes"] == C.TOTO_FIXED_PRIZES
    assert r["pool_share_of_sales"] == C.TOTO_POOL_SHARE_OF_SALES
    assert r["min_group1"] == C.TOTO_MIN_GROUP1


def test_toto_prize_structure_reads_changed_values():
    html = H.toto_prize_structure_html(group_pool_pct={1: 0.37, 2: 0.085, 3: 0.06, 4: 0.03},
                                       fixed_prizes={5: 60.0, 6: 30.0, 7: 10.0}, pool_share=None)
    r = P.parse_toto_prize_structure(html)
    assert r["group_pool_pct"] == {1: 0.37, 2: 0.085, 3: 0.06, 4: 0.03}
    assert r["fixed_prizes"] == {5: 60.0, 6: 30.0, 7: 10.0}
    assert r["pool_share_of_sales"] is None


def test_toto_prize_structure_javascript_shell_is_none():
    assert P.parse_toto_prize_structure(H.toto_prize_structure_html(rendered=False)) is None
    assert P.parse_toto_prize_structure("") is None
    assert P.parse_toto_prize_structure("<html><body><div id='root'></div></body></html>") is None


def test_toto_prize_structure_column_layout_is_not_guessed():
    # labels in one row and figures in another: cannot be matched safely, so None
    html = ("<table><tr><th>Group 1</th><th>Group 2</th><th>Group 3</th><th>Group 4</th></tr>"
            "<tr><td>38%</td><td>8%</td><td>5.5%</td><td>3%</td></tr></table>")
    assert P.parse_toto_prize_structure(html) is None


def test_toto_prize_structure_partial_is_none():
    html = "<p>Group 1: 38% of the prize pool. Group 2: 8% of the prize pool.</p>"
    assert P.parse_toto_prize_structure(html) is None


def test_toto_prize_structure_plain_text_paragraphs():
    html = ("<p>Group 1 gets 38% of the Prize Pool, at least $1 million.</p>"
            "<p>Group 2 gets 8%.</p><p>Group 3 gets 5.5%.</p><p>Group 4 gets 3%.</p>"
            "<p>Group 5 pays $50, Group 6 pays $25 and Group 7 pays $10 per winning share.</p>")
    r = P.parse_toto_prize_structure(html)
    assert r["group_pool_pct"] == {1: 0.38, 2: 0.08, 3: 0.055, 4: 0.03}
    assert r["fixed_prizes"] == {5: 50.0, 6: 25.0, 7: 10.0}
    assert r["min_group1"] == 1_000_000.0


@pytest.mark.parametrize("amount", ["$1m", "S$1M", "$1.0 mil", "$1 million", "$1,000,000"])
def test_toto_prize_structure_minimum_short_forms(amount):
    html = (f"<p>Group 1 gets 38% of the Prize Pool, at least {amount}.</p>"
            "<p>Group 2 gets 8%.</p><p>Group 3 gets 5.5%.</p><p>Group 4 gets 3%.</p>")
    assert P.parse_toto_prize_structure(html)["min_group1"] == 1_000_000.0


def test_toto_prize_structure_implausible_minimum_is_not_used():
    html = ("<p>Group 1 gets 38% of the Prize Pool, minimum $1.</p>"
            "<p>Group 2 gets 8%.</p><p>Group 3 gets 5.5%.</p><p>Group 4 gets 3%.</p>")
    assert P.parse_toto_prize_structure(html)["min_group1"] is None


def test_fourd_prize_structure_fixture(fixture_html):
    r = P.parse_fourd_prize_structure(fixture_html("fourd_prize_structure.html"))
    assert r["big"] == C.FOURD_PRIZES["big"]
    assert r["small"] == C.FOURD_PRIZES["small"]
    assert set(r["ibet"]) == {"big"}
    assert r["ibet"]["big"][24] == {"first": 83.0, "second": 41.0, "third": 20.0, "starter": 10.0, "consolation": 2.0}
    assert r["ibet"]["big"][4]["first"] == 500.0


def test_fourd_prize_structure_htmlgen_separate_tables():
    r = P.parse_fourd_prize_structure(H.fourd_prize_structure_html())
    assert r["big"] == C.FOURD_PRIZES["big"]
    assert r["small"] == C.FOURD_PRIZES["small"]
    for bet in ("big", "small"):
        for perms in (24, 12, 6, 4):
            expected = {t: float(math.floor(v / perms)) for t, v in C.FOURD_PRIZES[bet].items()}
            assert r["ibet"][bet][perms] == expected
    no_ibet = P.parse_fourd_prize_structure(H.fourd_prize_structure_html(ibet=False))
    assert "ibet" not in no_ibet


def test_fourd_prize_structure_reads_changed_values():
    big = {"first": 2500.0, "second": 1000.0, "third": 500.0, "starter": 250.0, "consolation": 60.0}
    r = P.parse_fourd_prize_structure(H.fourd_prize_structure_html(big=big, ibet=False))
    assert r["big"] == big


def test_fourd_prize_structure_text_fallback():
    html = """<div><h2>Big Forecast</h2><p>1st Prize $2,000 2nd Prize $1,000 3rd Prize $490
      Starter Prizes $250 Consolation Prizes $60</p>
      <h2>Small Forecast</h2><p>1st Prize $3,000, 2nd Prize $2,000, 3rd Prize $800</p>
      <h2>iBet Big</h2><p>1st Prize $83 2nd Prize $41</p></div>"""
    r = P.parse_fourd_prize_structure(html)
    assert r["big"] == C.FOURD_PRIZES["big"] and r["small"] == C.FOURD_PRIZES["small"]


def test_fourd_prize_structure_missing_or_js_is_none():
    assert P.parse_fourd_prize_structure(H.fourd_prize_structure_html(rendered=False)) is None
    assert P.parse_fourd_prize_structure("") is None
    only_big = "<h2>Big</h2><table><tr><td>1st Prize</td><td>$2,000</td></tr><tr><td>2nd Prize</td><td>$1,000</td></tr></table>"
    assert P.parse_fourd_prize_structure(only_big) is None


def test_page_text_drops_scripts_and_styles():
    text = P.page_text("<html><head><title>T</title><script>var x='$1';</script></head>"
                       "<body><style>p{}</style><p>Hello\n  world</p></body></html>")
    assert text == "Hello world"


def test_next_draw_impossible_date_is_none():
    info = P.parse_toto_next_draw("<p>Next Jackpot $1,000,000 est</p><p>Next Draw Mon, 31 Feb 2026 , 6.30pm</p>")
    assert info["draw_datetime"] is None and info["jackpot_estimate"] == 1_000_000


def test_check_site_fails_when_the_result_layout_changes():
    """A renamed shares table or prize section must make check site FAIL, not PASS with 0 groups."""
    from huatbot.fetch import check_site

    site = H.fake_site(synth_toto(n_draws=20), synth_fourd(n_draws=20))
    assert check_site(site, out=lambda s: None) is True
    for url, html in list(site.pages.items()):
        if isinstance(html, str):
            site.pages[url] = (html.replace("tableWinningShares", "tableSomethingElse")
                               .replace("tbodyStarterPrizes", "tbodySomethingElse"))
    lines: list[str] = []
    assert check_site(site, out=lines.append) is False
    assert any(line.startswith("FAIL  TOTO latest result page") and "winning shares" in line for line in lines)
    assert any(line.startswith("FAIL  4D latest result page") and "starter" in line for line in lines)
