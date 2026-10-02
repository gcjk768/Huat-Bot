"""Tests for huatbot.textfmt: money, dates, tables and the no dash rules."""
from __future__ import annotations

import random
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from huatbot.textfmt import (
    DASH_CHARS,
    contains_dash,
    find_prose_dashes,
    fmt_date,
    fmt_datetime,
    fmt_num,
    fmt_time,
    has_prose_dashes,
    html_escape,
    md_table,
    money,
    pct,
    per_dollar,
    plural,
    pre_block,
    pre_table,
    remove_dashes,
    toto_nums,
)

SG = ZoneInfo("Asia/Singapore")
BANNED = ("-", "–", "—")


def no_dash(s: str) -> bool:
    return not any(ch in s for ch in DASH_CHARS)


# money, per_dollar, pct, fmt_num


@pytest.mark.parametrize(
    "value, expected",
    [
        (1234567, "$1,234,567"),
        (1234567.4, "$1,234,567"),
        (0, "$0"),
        (999, "$999"),
        (1000, "$1,000"),
        (2.5, "$3"),  # half up, not banker's rounding
        (np.float64(13983816.0), "$13,983,816"),
        (np.int64(50), "$50"),
    ],
)
def test_money_whole_dollars(value, expected):
    assert money(value) == expected


def test_money_negative_has_no_dash():
    assert money(-20) == "minus $20"
    assert money(-1234567.0) == "minus $1,234,567"
    assert money(-0.66, cents=True) == "minus $0.66"
    for v in (-20, -0.5, -1e9):
        assert no_dash(money(v)) and no_dash(money(v, cents=True))


def test_money_rounding_to_zero_is_not_negative():
    assert money(-0.4) == "$0"
    assert money(-0.004, cents=True) == "$0.00"


def test_money_cents():
    assert money(0.659, cents=True) == "$0.66"
    assert money(1234.5, cents=True) == "$1,234.50"
    assert money(0.005, cents=True) == "$0.01"


@pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NA, float("inf"), "abc"])
def test_money_missing(value):
    assert money(value) == "n/a"


def test_per_dollar():
    assert per_dollar(0.659) == "$0.66"
    assert per_dollar(1) == "$1.00"
    assert per_dollar(None) == "n/a"
    assert per_dollar(-0.25) == "minus $0.25"


def test_pct():
    assert pct(0.054) == "5.4%"
    assert pct(0.5, 0) == "50%"
    assert pct(0.12345, 2) == "12.35%"
    assert pct(-0.0123) == "minus 1.2%"
    assert pct(None) == "n/a"
    assert pct(float("nan")) == "n/a"


def test_fmt_num_and_plural():
    assert fmt_num(13983816) == "13,983,816"
    assert fmt_num(-5) == "minus 5"
    assert fmt_num(1.256, 2) == "1.26"
    assert fmt_num(None) == "n/a"
    assert plural(1, "draw") == "1 draw"
    assert plural(3, "draw") == "3 draws"
    assert plural(1200, "board") == "1,200 boards"
    assert plural(2, "match", "matches") == "2 matches"


# Dates and times


def test_fmt_date_variants():
    assert fmt_date(date(2026, 10, 1)) == "Thu 1 Oct 2026"
    assert fmt_date(pd.Timestamp("2026-10-05")) == "Mon 5 Oct 2026"
    assert fmt_date("2026-10-05") == "Mon 5 Oct 2026"
    assert fmt_date(datetime(2026, 12, 25, 9, 0)) == "Fri 25 Dec 2026"
    assert fmt_date(np.datetime64("2026-01-03")) == "Sat 3 Jan 2026"


def test_fmt_date_converts_aware_datetimes_to_singapore():
    # 20:00 UTC on 1 Oct is 4am on 2 Oct in Singapore.
    assert fmt_date(datetime(2026, 10, 1, 20, 0, tzinfo=timezone.utc)) == "Fri 2 Oct 2026"


@pytest.mark.parametrize("value", [None, pd.NaT, "", "not a date", float("nan")])
def test_fmt_date_missing(value):
    assert fmt_date(value) == "n/a"


def test_fmt_datetime_spec_example():
    assert fmt_datetime(datetime(2026, 10, 5, 18, 30, tzinfo=SG)) == "Mon 5 Oct 2026, 6.30pm"


def test_fmt_datetime_converts_to_singapore_time():
    assert fmt_datetime(datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc)) == "Mon 5 Oct 2026, 6.30pm"
    ts = pd.Timestamp("2026-10-05 10:30", tz="UTC")
    assert fmt_datetime(ts) == "Mon 5 Oct 2026, 6.30pm"


@pytest.mark.parametrize(
    "hour, minute, expected",
    [(0, 5, "12.05am"), (9, 5, "9.05am"), (11, 59, "11.59am"), (12, 0, "12.00pm"), (13, 0, "1.00pm"), (23, 45, "11.45pm")],
)
def test_fmt_time_clock(hour, minute, expected):
    dt = datetime(2026, 10, 5, hour, minute, tzinfo=SG)
    assert fmt_time(dt) == expected
    assert fmt_datetime(dt).endswith(", " + expected)


def test_fmt_datetime_naive_is_taken_as_singapore_time():
    assert fmt_datetime(datetime(2026, 10, 5, 19, 30)) == "Mon 5 Oct 2026, 7.30pm"


def test_fmt_datetime_missing():
    assert fmt_datetime(None) == "n/a"
    assert fmt_datetime(pd.NaT) == "n/a"
    assert fmt_time(None) == "n/a"


def test_toto_nums():
    assert toto_nums([45, 3, 11, 19, 27, 38]) == "3 11 19 27 38 45"
    assert toto_nums(np.array([7, 1, 49])) == "1 7 49"
    assert toto_nums(["3", "11.0"]) == "3 11"
    assert toto_nums("38, 3 11") == "3 11 38"
    assert toto_nums([]) == ""
    assert toto_nums(None) == ""


# pre_table


def test_pre_table_layout_and_alignment():
    out = pre_table(["Group", "Share", "Winners"], [["Group 1", "$1,234,567", 2], ["Group 7", "$10", 12345]], "lrr")
    lines = out.split("\n")
    assert lines[0].startswith("Group")
    assert len(lines) == 3  # header plus rows, no rule line
    assert lines[1] == "Group 1  $1,234,567        2"
    assert lines[2] == "Group 7         $10    12345"
    assert all(line == line.rstrip() for line in lines)
    # right aligned columns end at the same place
    assert len(lines[1]) == len(lines[2])


def test_pre_table_has_no_dash_characters_at_all():
    rows = [
        ["Group 1", "-", -3, "2026-10-01"],
        ["Group 2", "—", -1234.5, "5 – 6"],
        ["Net", money(-20), None, float("nan")],
        ["well-known", "− 7", "--", "a - b"],
    ]
    out = pre_table(["Name", "Share", "Delta", "When"], rows)
    assert no_dash(out)
    for ch in BANNED:
        assert ch not in out
    assert "minus 3" in out and "minus 1234.5" in out
    assert "Thu 1 Oct 2026" in out
    assert "5 to 6" in out


def test_pre_table_default_alignment_right_aligns_numbers():
    out = pre_table(["Bet", "Return"], [["Big", "$0.66"], ["Small", "$0.58"], ["iBet 24", "n/a"]])
    lines = out.split("\n")
    assert lines[1].endswith("$0.66") and lines[2].endswith("$0.58")
    # header of a right aligned column is right aligned too
    assert lines[0].endswith("Return")
    assert lines[1].index("$0.66") == lines[2].index("$0.58")


def test_pre_table_ragged_rows_and_bad_align():
    out = pre_table(["A", "B", "C"], [["1"], ["2", "3", "4", "5"]], align="l")
    assert len(out.split("\n")) == 3
    with pytest.raises(ValueError):
        pre_table(["A"], [["x"]], align="lx")


def test_pre_table_no_rows():
    assert pre_table(["Draw", "Date"], []) == "Draw  Date"


def test_pre_block_escapes_html():
    out = pre_block(["Name"], [["<b>&"]])
    assert out.startswith("<pre>") and out.endswith("</pre>")
    assert "&lt;b&gt;&amp;" in out


# md_table


def test_md_table_is_valid_markdown():
    out = md_table(["Group", "Share", "Winners"], [["Group 1", "$1,234,567", 2], ["Group 2", "", 0]], "lrr")
    lines = out.split("\n")
    assert len(lines) == 4
    assert all(line.startswith("| ") and line.endswith(" |") for line in lines)
    sep = [c.strip() for c in lines[1].strip("|").split("|")]
    assert sep[0].strip("-") == "" and sep[0].startswith("---")
    assert sep[1].endswith(":") and set(sep[1]) == {"-", ":"}
    assert all(line.count("|") == 4 for line in lines)
    assert not has_prose_dashes(out)


def test_md_table_centre_and_default_alignment():
    out = md_table(["Name", "Count"], [["a", 1], ["b", 22]], "cr")
    sep = out.split("\n")[1]
    assert "| :" in sep and sep.rstrip(" |").endswith(":")
    auto = md_table(["Name", "Count"], [["a", 1], ["b", 22]])
    assert auto.split("\n")[1].rstrip(" |").endswith(":")


def test_md_table_escapes_pipes_and_newlines():
    out = md_table(["Note"], [["[[2026-10-01 TOTO 4123|Draw 4123]]"], ["two\nlines"]])
    assert "[[2026-10-01 TOTO 4123\\|Draw 4123]]" in out
    assert "two lines" in out
    assert len(out.split("\n")) == 4
    assert not has_prose_dashes(out)


def test_md_table_negative_numbers_and_missing_values():
    out = md_table(["Net", "Other"], [[-5, None], [np.int64(-7), float("nan")], [-1.25, True]])
    assert "minus 5" in out and "minus 7" in out and "minus 1.25" in out
    assert "yes" in out and "nan" not in out
    assert not has_prose_dashes(out)


def test_md_table_header_only():
    out = md_table(["Game", "Draw date", "Numbers", "Bet type", "Cost"], [])
    lines = out.split("\n")
    assert len(lines) == 2 and lines[0].startswith("| Game")


def test_html_escape():
    assert html_escape("<b>Tom & Jerry</b>") == "&lt;b&gt;Tom &amp; Jerry&lt;/b&gt;"
    assert html_escape('say "hi"') == 'say "hi"'
    assert html_escape(None) == ""
    assert html_escape(42) == "42"


# remove_dashes


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The jackpot — the biggest — is large.", "The jackpot, the biggest, is large."),
        ("Good – not great", "Good, not great"),
        ("Hot - recent draws", "Hot, recent draws"),
        ("Net is -$20 today", "Net is minus $20 today"),
        ("value: -5", "value: minus 5"),
        ("(-20)", "(minus 20)"),
        ("-20 overall", "minus 20 overall"),
        ("Numbers 1-24 are low", "Numbers 1 to 24 are low"),
        ("range 1 – 24", "range 1 to 24"),
        ("draws 4123 - 4130", "draws 4123 to 4130"),
        ("$1-$2 each", "$1 to $2 each"),
        ("Drawn 2026-10-01 at 6pm", "Drawn Thu 1 Oct 2026 at 6pm"),
        ("Log for 2026-10", "Log for Oct 2026"),
        ("well-known e-mail", "well known e mail"),
        ("word —", "word"),
        ("A plain sentence, with commas.", "A plain sentence, with commas."),
        ("Hello, world,\nnext", "Hello, world,\nnext"),
        ("Yes, — no", "Yes, no"),
        ("end — .", "end."),
        ("a—b", "a, b"),
        ("x − y", "x, y"),
    ],
)
def test_remove_dashes_examples(text, expected):
    assert remove_dashes(text) == expected


def test_remove_dashes_bullets_and_rules():
    assert remove_dashes("- first\n- second") == "• first\n• second"
    assert remove_dashes("Title\n---\nBody") == "Title\n\nBody"
    assert remove_dashes("- first", markdown=True) == "* first"


def test_remove_dashes_empty_and_none():
    assert remove_dashes("") == ""
    assert remove_dashes(None) == ""


def test_remove_dashes_never_leaves_a_dash():
    rng = random.Random(1234)
    alphabet = list("ab 1$,.\n") + list(DASH_CHARS) + ["— ", " - ", "2026-10-01", "| --- |"]
    for _ in range(2000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        out = remove_dashes(s)
        assert no_dash(out), (s, out)


def test_remove_dashes_markdown_keeps_structure():
    note = (
        "---\ntags: [huatbot, toto]\ndate: 2026-10-01\n---\n"
        "# Draw 4123\n\n"
        "| Group | Share |\n| --- | ---: |\n| Group 1 | $1,000,000 |\n\n"
        "Back to [[Dashboard]] — see [[2026-10-01 TOTO 4123]] and `a-b`.\n"
        "- a bullet\n"
        "```\ncode-block\n```\n"
    )
    out = remove_dashes(note, markdown=True)
    assert out.startswith("---\ntags: [huatbot, toto]\ndate: 2026-10-01\n---\n")
    assert "| --- | ---: |" in out
    assert "[[2026-10-01 TOTO 4123]]" in out and "`a-b`" in out and "code-block" in out
    assert "Back to [[Dashboard]], see" in out
    assert "* a bullet" in out
    assert not has_prose_dashes(out)
    # Plain mode removes everything, including the table rule.
    assert no_dash(remove_dashes(note))


# has_prose_dashes


def test_has_prose_dashes_detects_prose():
    assert has_prose_dashes("a - b")
    assert has_prose_dashes("well–known")
    assert has_prose_dashes("pause — here")
    assert has_prose_dashes("net -20")
    assert has_prose_dashes("- bullet")
    assert has_prose_dashes("---")  # a rule in the body is not allowed
    assert not has_prose_dashes("Plain text, no dashes. Minus $20.")
    assert not has_prose_dashes("")
    assert not has_prose_dashes(None)


def test_has_prose_dashes_ignores_table_separator_rows():
    table = "| A | B |\n| --- | :---: |\n|---|---|\n--- | ---\n| x | y |"
    assert not has_prose_dashes(table)
    # but a dash in a table cell is prose
    assert has_prose_dashes("| A |\n| --- |\n| -5 |")


def test_has_prose_dashes_ignores_leading_frontmatter_only():
    fm = "---\ndate: 2026-10-01\nrange: 1-24\n---\nBody text."
    assert not has_prose_dashes(fm)
    assert not has_prose_dashes("﻿" + fm)
    assert not has_prose_dashes("---\n---\nBody")
    assert has_prose_dashes(fm + "\nBad - prose")
    # a frontmatter like block that is not at the very start is prose
    assert has_prose_dashes("Intro\n---\ndate: 2026-10-01\n---\n")
    # an unclosed block is not frontmatter
    assert has_prose_dashes("---\ndate: 2026-10-01\nBody")


def test_has_prose_dashes_ignores_wikilinks_code_urls_comments():
    assert not has_prose_dashes("See [[2026-10-01 TOTO 4123]] and ![[2026-10 Chart.png]].")
    assert not has_prose_dashes("Alias [[Reports/2026-10-02 1930 Report|latest report]].")
    assert not has_prose_dashes("Run `python -m huatbot run --dry-run` now.")
    assert not has_prose_dashes("Use ``a-b`` too.")
    assert not has_prose_dashes("```bash\ndocker compose up -d\n```\nDone.")
    assert not has_prose_dashes("~~~\na-b\n~~~")
    assert not has_prose_dashes("Rules at https://online2.singaporepools.com/en/lottery/toto-prize-structure today.")
    assert not has_prose_dashes("[prize rules](https://x.com/toto-prize-structure)")
    assert not has_prose_dashes("[report](<Reports/2026-10-02 1930 Report.md>)")
    assert not has_prose_dashes("<!-- huatbot-start -->\nText\n%% a-b %%")
    # prose around them still counts
    assert has_prose_dashes("See [[2026-10-01 TOTO 4123]] - now")
    assert has_prose_dashes("```\ncode\n```\nafter - this")


def test_has_prose_dashes_unclosed_fence_runs_to_end():
    assert not has_prose_dashes("Text\n```\na-b\nc-d")


def test_find_prose_dashes_reports_lines():
    text = "---\nk: a-b\n---\nok line\nbad - line\n| --- |\n[[2026-10-01]]\nanother—bad"
    hits = find_prose_dashes(text)
    assert [n for n, _ in hits] == [5, 8]
    assert hits[0][1] == "bad - line"


def test_contains_dash():
    assert contains_dash("a-b")
    assert contains_dash("a—b")
    assert not contains_dash("minus $20")
    assert not contains_dash("")
    assert not contains_dash(None)


def test_formatters_never_emit_dashes():
    samples = [
        money(-1234567.89), money(-0.5, cents=True), pct(-0.25), fmt_num(-12345.678, 2),
        fmt_date(date(2026, 1, 1)), fmt_datetime(datetime(2026, 1, 1, 0, 0, tzinfo=SG)),
        toto_nums([1, 2, 3, 4, 5, 6]),
    ]
    for s in samples:
        assert no_dash(s), s
