"""Tests for huatbot.tickets: reading Tickets.md, ticket ids and the ledger.

Results come from the synthetic history in conftest (no network). Expected prize amounts
are taken from the synthetic rows themselves, never hard coded.
"""
from __future__ import annotations

import re
import warnings
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import prizes
from huatbot.models import LEDGER_COLUMNS, PrizeRules, Ticket
from huatbot.store import empty_ledger, load_ledger, save_ledger
from huatbot.tickets import (
    REMOVED_RESULT,
    REPLACED_PREFIX,
    TICKETS_TEMPLATE,
    has_ticket_table,
    ledger_totals,
    parse_date,
    parse_tickets,
    settle_ledger,
    settled_rows,
    sync_ledger,
    ticket_id,
    ticket_ids,
)

SG = ZoneInfo(C.SG_TZ_NAME)
NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)
HEADER = "| Game | Draw date | Numbers | Bet type | Cost |\n| --- | --- | --- | --- | --- |\n"


def note(*rows: str) -> str:
    """A Tickets.md body: the standard table with these rows."""
    return "# My tickets\n\n" + HEADER + "".join(r + "\n" for r in rows)


def valid(tickets):
    return [t for t in tickets if t.error is None]


def one(text: str) -> Ticket:
    tickets = parse_tickets(text)
    assert len(tickets) == 1, tickets
    return tickets[0]


def code_block_lines(text: str) -> list[str]:
    """Every line inside fenced code blocks of a note."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        if inside:
            out.append(line)
    return out


def iso(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def toto_text(row) -> str:
    return " ".join(str(int(row[f"n{i}"])) for i in range(1, 7))


def ledger_for(text: str) -> pd.DataFrame:
    return sync_ledger(empty_ledger(), parse_tickets(text), NOW)


# Template


def test_template_reads_as_no_tickets():
    assert parse_tickets(TICKETS_TEMPLATE) == []


def test_template_layout():
    t = TICKETS_TEMPLATE
    assert t.startswith("# My tickets\n")
    assert "| Game | Draw date | Numbers | Bet type | Cost |" in t
    assert "| --- | --- | --- | --- | --- |" in t
    assert "## Examples" in t
    # The table comes before the examples, which sit inside code blocks.
    assert t.index("| Game |") < t.index("## Examples") < t.index("```")


def test_template_has_no_prose_dashes():
    separator = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")
    for line in TICKETS_TEMPLATE.splitlines():
        if any(ch in line for ch in "-–—"):
            assert separator.match(line.strip()), line


def test_template_examples_are_valid_tickets():
    tickets = parse_tickets("\n".join(code_block_lines(TICKETS_TEMPLATE)))
    assert [t.error for t in tickets] == [None] * 5
    got = [(t.game, t.bet_type, t.cost) for t in tickets]
    assert got == [
        ("TOTO", "Ordinary", 1.0),
        ("TOTO", "System 7", 7.0),
        ("4D", "Big", 2.0),
        ("4D", "iBet Big", 1.0),
        ("TOTO", "Ordinary", 1.0),
    ]
    assert tickets[1].numbers == "3 11 19 27 38 45 49"


# parse_tickets: tables


def test_table_row_normalised():
    t = one(note("|  toto |  Mon 5 Oct 2026 | 45, 3,11 , 19 27 38 |  ord  | $1.00 |"))
    assert t.error is None
    assert (t.game, t.draw_date, t.numbers, t.bet_type, t.cost) == (
        "TOTO", date(2026, 10, 5), "3 11 19 27 38 45", "Ordinary", 1.0)
    assert t.line_no == 5
    assert t.source.startswith("|  toto |")


@pytest.mark.parametrize("bet", ["", "Ord", "ordinary", "ORDINARY", " Ordinary "])
def test_toto_ordinary_spellings(bet):
    t = one(note(f"| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | {bet} | 1 |"))
    assert t.error is None and t.bet_type == "Ordinary"


@pytest.mark.parametrize("bet", ["System 7", "Sys 7", "System7", "S7", "sys7", "SYSTEM 7", "s 7", "Sys. 7"])
def test_toto_system_spellings(bet):
    t = one(note(f"| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | {bet} | $7 |"))
    assert t.error is None and t.bet_type == "System 7"


def test_blank_bet_with_more_numbers_is_a_system_bet():
    t = one(note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 8 | | 28 |"))
    assert t.error is None and t.bet_type == "System 8"


@pytest.mark.parametrize("bet, expected", [
    ("Big", "Big"), ("big", "Big"), ("SMALL", "Small"), ("small", "Small"),
    ("iBet", "iBet Big"), ("ibet", "iBet Big"), ("i-Bet", "iBet Big"), ("I BET", "iBet Big"),
    ("iBet Big", "iBet Big"), ("i-bet big", "iBet Big"), ("ibet small", "iBet Small"),
    ("i-Bet Small", "iBet Small"), ("IBETSMALL", "iBet Small"),
])
def test_fourd_bet_spellings(bet, expected):
    t = one(note(f"| 4D | 4 Oct 2026 | 1234 | {bet} | 1 |"))
    assert t.error is None and t.bet_type == expected


def test_fourd_keeps_leading_zeros():
    t = one(note("| 4d | 4 Oct 2026 | 0042 | Big | $2 |"))
    assert t.error is None
    assert (t.game, t.numbers, t.bet_type, t.cost) == ("4D", "0042", "Big", 2.0)


def test_table_columns_mapped_by_header():
    text = ("| Cost | Game | Numbers | Draw date | Bet type | Notes |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| $2 | 4D | 0007 | 4 Oct 2026 | Small | birthday |\n")
    t = one(text)
    assert t.error is None
    assert (t.game, t.numbers, t.bet_type, t.cost, t.draw_date) == (
        "4D", "0007", "Small", 2.0, date(2026, 10, 4))


def test_rows_without_header_and_without_outer_formatting():
    t = one("| **TOTO** | 5 Oct 2026 | `3 11 19 27 38 45` | Ordinary | 1 |")
    assert t.error is None and t.numbers == "3 11 19 27 38 45"


def test_ignores_headers_separators_blank_rows_headings_and_prose():
    text = (
        "---\ntags: [huatbot]\nnote: TOTO, 5 Oct 2026, 1 2 3 4 5 6, Ordinary, 1\n---\n"
        "# My tickets\n"
        "Some words about TOTO and 4D, with commas, and more.\n"
        "## TOTO, 5 Oct 2026, 1 2 3 4 5 6, Ordinary, 1\n"
        + HEADER
        + "|  |  |  |  |  |\n"
        + "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\n"
        + "| Game | Draw date | Numbers | Bet type | Cost |\n"
        + "<!-- | TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 | -->\n"
        + "%% TOTO, 5 Oct 2026, 1 2 3 4 5 6, Ordinary, 1 %%\n"
    )
    tickets = parse_tickets(text)
    assert len(tickets) == 1 and tickets[0].error is None
    assert tickets[0].line_no == text.splitlines().index(
        "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |") + 1


def test_other_tables_are_ignored():
    text = ("| Shop | Area |\n| --- | --- |\n| Lucky Store | Bedok |\n\n"
            + note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |"))
    assert len(valid(parse_tickets(text))) == 1
    assert len(parse_tickets(text)) == 1


def test_rows_outside_a_ticket_table_need_a_game():
    # Without a ticket header, only rows that start with TOTO or 4D are read.
    text = "| Lotto | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\n| 4D | 4 Oct 2026 | 1234 | Big | 1 |\n"
    assert [t.game for t in parse_tickets(text)] == ["4D"]


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_code_blocks_are_ignored(fence):
    text = note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |") + (
        f"\n{fence}\n| TOTO | 5 Oct 2026 | 7 8 9 10 11 12 | Ordinary | 1 |\n"
        f"TOTO, 5 Oct 2026, 7 8 9 10 11 12, Ordinary, 1\n| broken |\n{fence}\n"
        "4D, 4 Oct 2026, 1234, Big, 1\n"
    )
    tickets = parse_tickets(text)
    assert [(t.game, t.numbers) for t in tickets] == [("TOTO", "1 2 3 4 5 6"), ("4D", "1234")]


# parse_tickets: plain lines


@pytest.mark.parametrize("line", [
    "TOTO, 5 Oct 2026, 3 11 19 27 38 45, Ordinary, 1",
    "- TOTO, 5 Oct 2026, 3 11 19 27 38 45, Ordinary, 1",
    "* toto ,  5 Oct 2026 ,  45 38 27 19 11 3 , ORD , $1",
    "- [x] TOTO, Mon, 5 Oct 2026, 3, 11, 19, 27, 38, 45, Ordinary, $1.00",
    "1. TOTO, 2026-10-05, 3 11 19 27 38 45, , 1",
    "TOTO, 5/10/2026, 3 11 19 27 38 45, 1",
])
def test_plain_lines(line):
    t = one(line)
    assert t.error is None, t.error
    assert (t.game, t.draw_date, t.numbers, t.bet_type, t.cost) == (
        "TOTO", date(2026, 10, 5), "3 11 19 27 38 45", "Ordinary", 1.0)
    assert not t.source.startswith(("-", "*", "1."))


def test_plain_fourd_line():
    t = one("* 4D, Sun 4 Oct 2026, 0042, i-Bet, $1")
    assert t.error is None
    assert (t.numbers, t.bet_type, t.cost) == ("0042", "iBet Big", 1.0)


def test_windows_line_endings_and_byte_order_mark():
    text = "\ufeff| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\r\n4D, 4 Oct 2026, 1234, Big, 1\r\n"
    tickets = parse_tickets(text)
    assert [(t.game, t.error, t.line_no) for t in tickets] == [("TOTO", None, 1), ("4D", None, 2)]


def test_mixed_note_keeps_file_order():
    text = note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |") + (
        "\nAlso bought:\n- 4D, 4 Oct 2026, 9999, Big, 3\n"
    )
    tickets = parse_tickets(text)
    assert [t.game for t in tickets] == ["TOTO", "4D"]
    assert tickets[0].line_no < tickets[1].line_no


# Dates


@pytest.mark.parametrize("text", [
    "5 Oct 2026", "05 Oct 2026", "5 October 2026", "Mon 5 Oct 2026", "Mon, 05 Oct 2026",
    "Monday 5 October 2026", "2026-10-05", "2026/10/05", "5/10/2026", "05/10/2026", "5/10/26",
    "5.10.2026", "Oct 5 2026", "Oct 5, 2026", "5th Oct 2026", "5 Oct 26", "5-Oct-2026",
])
def test_date_formats(text):
    assert parse_date(text) == date(2026, 10, 5)


def test_date_september_spellings():
    assert parse_date("1 Sept 2026") == parse_date("1 Sep 2026") == date(2026, 9, 1)


@pytest.mark.parametrize("text, words", [
    ("", "missing"),
    ("Tue 5 Oct 2026", "Monday"),
    ("31 Sep 2026", "not a real date"),
    ("5 Oct", "year"),
    ("next Monday", "could not be read"),
    ("2026-13-05", "not a real date"),
])
def test_bad_dates(text, words):
    with pytest.raises(ValueError, match=words):
        parse_date(text)


# Bad lines


@pytest.mark.parametrize("row, words", [
    ("| Lotto | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |", "TOTO or 4D"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 | Ordinary | 1 |", "6 numbers"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 | Ordinary | 1 |", "6 numbers"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | System 7 | 7 |", "System 7 needs 7 numbers"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 8 9 10 11 12 13 | | 1 |", "this line has 13"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 50 | Ordinary | 1 |", "50 is not a TOTO number"),
    ("| TOTO | 5 Oct 2026 | 0 2 3 4 5 6 | Ordinary | 1 |", "0 is not a TOTO number"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 5 | Ordinary | 1 |", "5 appears more than once"),
    ("| TOTO | 5 Oct 2026 | 1 2 three 4 5 6 | Ordinary | 1 |", "whole numbers"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Big | 1 |", "not a TOTO bet"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 | System 14 | 1 |", "System 7 to System 12"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 | System 7 | $1 |", "costs at least $7"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 0 |", "above 0"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | free |", "above 0"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | |", "cost is missing"),
    ("| TOTO | | 1 2 3 4 5 6 | Ordinary | 1 |", "date is missing"),
    ("| TOTO | 5 Oct 2026 | | Ordinary | 1 |", "numbers are missing"),
    ("| 4D | 4 Oct 2026 | 42 | Big | 1 |", "4 digits"),
    ("| 4D | 4 Oct 2026 | 12345 | Big | 1 |", "4 digits"),
    ("| 4D | 4 Oct 2026 | 1234 5678 | Big | 1 |", "4 digits"),
    ("| 4D | 4 Oct 2026 | 1234 | | 1 |", "bet type is missing"),
    ("| 4D | 4 Oct 2026 | 1234 | Ordinary | 1 |", "not a 4D bet"),
    ("| 4D | 4 Oct 2026 | 7777 | iBet | 1 |", "two different digits"),
    ("| 4D | 4 Oct 2026 | 1234 | Big | $0 |", "above 0"),
])
def test_bad_rows_give_plain_errors(row, words):
    tickets = parse_tickets(note(row))
    assert len(tickets) == 1
    err = tickets[0].error
    assert err and words in err, err
    assert not any(ch in err for ch in "-–—"), err  # user facing: no dashes
    assert tickets[0].source == row


def test_bad_plain_lines_give_errors():
    tickets = parse_tickets("- TOTO, 5 Oct 2026, 1 2 3, Ordinary, 1\n- 4D, someday, 1234, Big, 1\n")
    assert [bool(t.error) for t in tickets] == [True, True]
    assert "6 numbers" in tickets[0].error
    assert "could not be read" in tickets[1].error


@pytest.mark.parametrize("text", [
    "", "|", "||||", "| | |\n|---|", "```", "```\nTOTO, 1, 2, 3", "TOTO, , , , , , ,",
    "| TOTO |", "| 4D | x | y | z | w | v |", "---\n", "---\nbad: [\n---\nTOTO, 5 Oct 2026",
    "\x00\x01 | TOTO | — |", "4D,,,", "| TOTO | 99/99/9999 | 1 | 2 | 3 |",
])
def test_never_raises_on_garbage(text):
    for t in parse_tickets(text):
        assert isinstance(t, Ticket)


# ticket_id


def test_ticket_id_shape_and_stability():
    a = one(note("| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | $1 |"))
    b = one(note("|toto|Mon, 05 Oct 2026|45 38 27 19 11 3|ord|1.00|"))
    c = one("- TOTO, 2026-10-05, 3, 11, 19, 27, 38, 45, , 1")
    tid = ticket_id(a, 0)
    assert re.fullmatch(r"[0-9a-f]{12}", tid)
    assert tid == ticket_id(b, 0) == ticket_id(c, 0)
    assert ticket_id(a, 0) == ticket_id(a, 0)  # deterministic across calls (and runs: sha1)


def test_ticket_id_ibet_spellings_and_hand_built_tickets():
    a = one(note("| 4D | 4 Oct 2026 | 0042 | i-bet | $1 |"))
    hand = Ticket(game="4d", draw_date=date(2026, 10, 4), numbers=" 0042", bet_type="iBet Big",
                  cost=1, line_no=99, source="anything")
    assert ticket_id(a, 0) == ticket_id(hand, 0)


def test_ticket_id_distinguishes_tickets():
    base = one(note("| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |"))
    others = [
        one(note("| TOTO | 5 Oct 2026 | 3 11 19 27 38 46 | Ordinary | 1 |")),
        one(note("| TOTO | 8 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |")),
        one(note("| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 2 |")),
    ]
    ids = {ticket_id(base, 0), ticket_id(base, 1)} | {ticket_id(t, 0) for t in others}
    assert len(ids) == 5


def test_ticket_ids_count_occurrences_of_identical_lines():
    row = "| 4D | 4 Oct 2026 | 1234 | Big | 1 |"
    tickets = parse_tickets(note(row, "| TOTO | 5 Oct 2026 | 1 2 3 4 5 | Ordinary | 1 |", row, row))
    ids = ticket_ids(tickets)
    assert ids[1] is None  # the bad line has no id
    assert ids[0] == ticket_id(tickets[0], 0)
    assert ids[2] == ticket_id(tickets[2], 1)
    assert ids[3] == ticket_id(tickets[3], 2)
    assert len({ids[0], ids[2], ids[3]}) == 3


# sync_ledger


def test_sync_adds_valid_tickets_as_pending():
    text = note(
        "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | System 7 | $7 |",
        "| 4D | 4 Oct 2026 | 0042 | Big | $2 |",
        "| 4D | 4 Oct 2026 | 42 | Big | $2 |",  # bad: not added
    )
    led = ledger_for(text)
    assert list(led.columns) == LEDGER_COLUMNS
    assert len(led) == 2
    assert set(led["status"]) == {"pending"}
    toto, fourd = led.iloc[0], led.iloc[1]
    assert (toto["game"], toto["draw_date"], toto["bet_type"]) == ("TOTO", "2026-10-05", "System 7")
    assert toto["units"] == pytest.approx(1.0)  # $7 over 7 boards
    assert fourd["numbers"] == "0042" and fourd["units"] == pytest.approx(2.0)
    assert toto["added_at"] == NOW.isoformat(timespec="seconds")
    assert pd.isna(toto["draw_number"])
    assert toto["source"].startswith("| TOTO")


def test_sync_is_idempotent_and_keeps_duplicates():
    row = "| 4D | 4 Oct 2026 | 1234 | Big | 1 |"
    tickets = parse_tickets(note(row, row))
    first = sync_ledger(empty_ledger(), tickets, NOW)
    second = sync_ledger(first, tickets, NOW + timedelta(days=1))
    assert len(first) == 2  # two identical lines are two tickets
    pd.testing.assert_frame_equal(first, second)
    # Cosmetic edits do not add rows either.
    edited = parse_tickets(note("|4d| Sun 4 Oct 2026 |1234| big | $1.00 |", row))
    pd.testing.assert_frame_equal(sync_ledger(first, edited, NOW), first)


def test_sync_survives_csv_round_trip(tmp_path):
    text = note("| 4D | 4 Oct 2026 | 0042 | iBet | 1 |", "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | | 1 |")
    led = ledger_for(text)
    path = tmp_path / "ledger.csv"
    save_ledger(led, path)
    loaded = load_ledger(path)
    assert loaded.loc[0, "numbers"] == "0042"
    again = sync_ledger(loaded, parse_tickets(text), NOW)
    assert len(again) == 2
    assert list(again["ticket_id"]) == list(led["ticket_id"])


def test_sync_handles_none_and_empty_inputs():
    assert len(sync_ledger(None, [], NOW)) == 0
    led = sync_ledger(None, parse_tickets(note("| 4D | 4 Oct 2026 | 1234 | Big | 1 |")), None)
    assert len(led) == 1 and led.loc[0, "added_at"]


def test_sync_never_deletes_and_retires_removed_unchecked_rows():
    typo = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 44 | Ordinary | 1 |"
    fixed = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |"
    other = "| 4D | 4 Oct 2026 | 1234 | Big | 1 |"
    led = ledger_for(note(typo, other))
    led.loc[led["game"] == "4D", "status"] = "settled"  # already checked: never touched again
    led = sync_ledger(led, parse_tickets(note(fixed)), NOW)
    assert len(led) == 3
    by_numbers = led.set_index("numbers")
    assert by_numbers.loc["3 11 19 27 38 44", "status"] == "invalid"
    assert by_numbers.loc["3 11 19 27 38 44", "result"] == REMOVED_RESULT
    assert by_numbers.loc["3 11 19 27 38 45", "status"] == "pending"
    assert by_numbers.loc["1234", "status"] == "settled"
    assert ledger_totals(led)["spent"] == pytest.approx(2.0)  # the typo is not counted

    # The line comes back: the row is pending again. An empty note retires nothing.
    led = sync_ledger(led, parse_tickets(note(typo, fixed)), NOW)
    assert set(led["status"]) == {"pending", "settled"}
    assert len(sync_ledger(led, [], NOW)) == 3
    assert (sync_ledger(led, [], NOW)["status"] == led["status"]).all()
    kept = sync_ledger(led, parse_tickets(note(fixed)), NOW, retire_missing=False)
    assert (kept["status"] == led["status"]).all()


def test_emptying_the_ticket_table_retires_pending_rows():
    pending = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |"
    checked = "| 4D | 4 Oct 2026 | 1234 | Big | 1 |"
    led = ledger_for(note(pending, checked))
    led.loc[led["game"] == "4D", "status"] = "settled"
    # The user deleted every row but kept the table: the pending ticket leaves the totals.
    assert has_ticket_table(note())
    out = sync_ledger(led, parse_tickets(note()), NOW, note_read=True)
    by_numbers = out.set_index("numbers")
    assert by_numbers.loc["3 11 19 27 38 45", "status"] == "invalid"
    assert by_numbers.loc["3 11 19 27 38 45", "result"] == REMOVED_RESULT
    assert by_numbers.loc["1234", "status"] == "settled"
    assert ledger_totals(out)["spent"] == pytest.approx(1.0)
    # A missing note, or one without a ticket table, retires nothing.
    for text in (None, "", "# My tickets\n\nNothing here yet.\n", "```\n" + HEADER + "```\n"):
        assert not has_ticket_table(text)
    assert has_ticket_table(TICKETS_TEMPLATE)
    kept = sync_ledger(led, [], NOW, note_read=False)
    assert (kept["status"] == led["status"]).all()


# settle_ledger


@pytest.fixture
def toto_win_rows(toto_df):
    with_winner = toto_df[toto_df["g1_winners"] > 0]
    no_winner = toto_df[toto_df["g1_winners"] == 0]
    if with_winner.empty or no_winner.empty:
        pytest.skip("synthetic history lacks a Group 1 case")
    return with_winner.iloc[-1], no_winner.iloc[-1]


def test_toto_ticket_matching_the_draw_wins_group_1(toto_df, fourd_df, rules, toto_win_rows):
    shared, unwon = toto_win_rows
    text = note(
        f"| TOTO | {iso(shared['draw_date'])} | {toto_text(shared)} | Ordinary | 1 |",
        f"| TOTO | {iso(unwon['draw_date'])} | {toto_text(unwon)} | Ordinary | $2 |",
    )
    led, settled = settle_ledger(ledger_for(text), toto_df, fourd_df, rules, NOW)
    assert len(settled) == 2
    assert set(led["status"]) == {"settled"}
    first, second = settled
    assert first["result"] == "Group 1 x1"
    assert first["winnings"] == pytest.approx(float(shared["g1_share"]))
    assert first["draw_number"] == int(shared["draw_number"])
    assert isinstance(first["draw_number"], int)
    # No Group 1 winner on the page: the ticket would have taken the whole jackpot, twice.
    assert second["winnings"] == pytest.approx(2 * float(unwon["jackpot"]))
    assert second["units"] == pytest.approx(2.0)
    assert second["checked_at"] == NOW.isoformat(timespec="seconds")
    assert set(first) == set(LEDGER_COLUMNS)


def test_toto_system7_ticket_uses_one_unit(toto_df, fourd_df, rules, toto_win_rows):
    shared, _ = toto_win_rows
    winning = [int(shared[f"n{i}"]) for i in range(1, 7)]
    extra = next(n for n in range(1, 50) if n not in winning and n != int(shared["additional"]))
    numbers = " ".join(str(n) for n in sorted(winning + [extra]))
    text = note(f"| TOTO | {iso(shared['draw_date'])} | {numbers} | Sys 7 | $7 |")
    led, settled = settle_ledger(ledger_for(text), toto_df, fourd_df, rules, NOW)
    expected = prizes.toto_ticket_prize(numbers, "System 7", 1.0, shared, rules)
    assert settled[0]["winnings"] == pytest.approx(expected.amount)
    assert settled[0]["result"] == expected.detail == "Group 3 x6, Group 1 x1"
    assert led.loc[0, "units"] == pytest.approx(1.0)


def test_fourd_tickets_settle(toto_df, fourd_df, rules):
    row = fourd_df.iloc[-5]
    d = iso(row["draw_date"])
    first = row["first"]
    shuffled = first[::-1] if first[::-1] != first else None
    starter = row["starter_1"]
    rows = [
        f"| 4D | {d} | {first} | Big | $2 |",
        f"| 4D | {d} | {starter} | Small | $1 |",
    ]
    if shuffled:
        rows.append(f"| 4D | {d} | {shuffled} | iBet Big | $1 |")
    led, settled = settle_ledger(ledger_for(note(*rows)), toto_df, fourd_df, rules, NOW)
    assert len(settled) == len(rows)
    big, small = settled[0], settled[1]
    assert big["winnings"] == pytest.approx(2 * rules.fourd_prizes["big"]["first"])  # $2,000 per $1
    assert big["result"] == "1st Prize x1"
    assert small["winnings"] == 0 and small["result"] == "No prize"  # Small pays top 3 only
    if shuffled:
        perms = prizes.permutations_count(first)
        ibet = settled[2]
        assert ibet["winnings"] == pytest.approx(prizes.ibet_table(rules, "big", perms)["first"])
    totals = ledger_totals(led)
    assert totals["settled"] == len(rows)
    assert totals["won"] == pytest.approx(sum(r["winnings"] for r in settled))


def test_future_and_non_draw_dates(toto_df, fourd_df, rules):
    latest = toto_df.iloc[-1]["draw_date"].date()
    some_draw = toto_df.iloc[-10]["draw_date"].date()
    gap_day = some_draw + timedelta(days=1)  # Mon or Thu draws: the next day has no draw
    assert gap_day not in set(toto_df["draw_date"].dt.date)
    future = latest + timedelta(days=3)
    text = note(
        f"| TOTO | {future.isoformat()} | 1 2 3 4 5 6 | Ordinary | 1 |",
        f"| TOTO | {gap_day.isoformat()} | 1 2 3 4 5 6 | Ordinary | 1 |",
        f"| 4D | {future.isoformat()} | 1234 | Big | 1 |",
    )
    led, settled = settle_ledger(ledger_for(text), toto_df, fourd_df, rules, NOW)
    assert settled == []
    assert list(led["status"]) == ["pending", "no_draw", "pending"]
    assert led.loc[1, "result"] == "No draw on this date"
    totals = ledger_totals(led)
    assert totals["pending_cost"] == pytest.approx(2.0)
    assert totals["spent"] == pytest.approx(3.0)  # no_draw tickets still count as spent


def test_missing_draw_in_data_stays_pending(toto_df, fourd_df, rules):
    target = toto_df.iloc[-20]
    holed = toto_df[toto_df["draw_number"] != target["draw_number"]]
    text = note(f"| TOTO | {iso(target['draw_date'])} | {toto_text(target)} | Ordinary | 1 |")
    led, settled = settle_ledger(ledger_for(text), holed, fourd_df, rules, NOW)
    assert settled == [] and led.loc[0, "status"] == "pending"
    # Once the draw is stored the same row settles.
    led, settled = settle_ledger(led, toto_df, fourd_df, rules, NOW)
    assert len(settled) == 1 and settled[0]["result"] == "Group 1 x1"


def test_date_before_history_stays_pending(toto_df, fourd_df, rules):
    early = toto_df.iloc[0]["draw_date"].date() - timedelta(days=30)
    text = note(f"| TOTO | {early.isoformat()} | 1 2 3 4 5 6 | Ordinary | 1 |")
    led, settled = settle_ledger(ledger_for(text), toto_df, fourd_df, rules, NOW)
    assert settled == [] and led.loc[0, "status"] == "pending"
    assert "older than the stored history" in led.loc[0, "result"]


def test_settle_is_idempotent(toto_df, fourd_df, rules):
    row = toto_df.iloc[-3]
    text = note(f"| TOTO | {iso(row['draw_date'])} | 1 2 3 4 5 6 | Ordinary | 1 |",
                f"| 4D | {iso(fourd_df.iloc[-1]['draw_date'])} | 0000 | Big | 1 |")
    led, settled = settle_ledger(ledger_for(text), toto_df, fourd_df, rules, NOW)
    assert len(settled) == 2
    again, settled_again = settle_ledger(led, toto_df, fourd_df, rules, NOW + timedelta(days=1))
    assert settled_again == []
    pd.testing.assert_frame_equal(led, again)
    # Sync after settle changes nothing either.
    pd.testing.assert_frame_equal(sync_ledger(again, parse_tickets(text), NOW), again)


def test_settle_without_results_or_ledger(rules, toto_df, fourd_df):
    led = ledger_for(note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |"))
    out, settled = settle_ledger(led, None, None, rules, NOW)
    assert settled == [] and out.loc[0, "status"] == "pending"
    out, settled = settle_ledger(None, toto_df, fourd_df, rules, NOW)
    assert len(out) == 0 and settled == []


def test_hand_edited_bad_rows_become_invalid(toto_df, fourd_df, rules):
    row = toto_df.iloc[-2]
    led = ledger_for(note(f"| TOTO | {iso(row['draw_date'])} | 1 2 3 4 5 6 | Ordinary | 1 |",
                          "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 2 |"))
    led.loc[0, "numbers"] = "1 2 3"  # broken by hand in ledger.csv
    led.loc[1, "draw_date"] = "soon"
    out, settled = settle_ledger(led, toto_df, fourd_df, rules, NOW)
    assert settled == []
    assert list(out["status"]) == ["invalid", "invalid"]
    assert all(r and "-" not in r for r in out["result"])
    assert ledger_totals(out)["spent"] == 0


def test_settle_raises_no_warnings(toto_df, fourd_df, rules):
    row = toto_df.iloc[-4]
    text = note(f"| TOTO | {iso(row['draw_date'])} | {toto_text(row)} | Ordinary | 1 |",
                "| 4D | 4 Oct 2030 | 1234 | Big | 1 |")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        led = sync_ledger(empty_ledger(), parse_tickets(text), NOW)
        led, _ = settle_ledger(led, toto_df, fourd_df, rules, NOW)
        ledger_totals(led)


# ledger_totals


def test_ledger_totals():
    led = ledger_for(note(
        "| TOTO | 1 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |",
        "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 | System 7 | 7 |",
        "| 4D | 30 Sep 2026 | 1234 | Big | 3 |",
        "| 4D | 3 Oct 2026 | 1234 | Small | 2 |",
    ))
    led.loc[0, ["status", "winnings"]] = ["settled", 10.0]
    led.loc[2, ["status", "winnings"]] = ["settled", 0.0]
    led.loc[3, "status"] = "invalid"
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(11.0)
    assert totals["won"] == pytest.approx(10.0)
    assert totals["net"] == pytest.approx(-1.0)
    assert totals["pending_cost"] == pytest.approx(7.0)
    assert (totals["tickets"], totals["settled"], totals["pending"], totals["invalid"]) == (3, 2, 1, 1)
    assert totals["wins"] == 1


def test_ledger_totals_empty():
    totals = ledger_totals(empty_ledger())
    assert totals["spent"] == totals["won"] == totals["net"] == totals["pending_cost"] == 0
    assert totals["tickets"] == totals["settled"] == 0
    assert ledger_totals(None)["tickets"] == 0


def test_rules_default_used_when_missing(toto_df, fourd_df):
    row = fourd_df.iloc[-1]
    text = note(f"| 4D | {iso(row['draw_date'])} | {row['first']} | Big | 1 |")
    _, settled = settle_ledger(ledger_for(text), toto_df, fourd_df, PrizeRules(), NOW)
    assert settled[0]["winnings"] == pytest.approx(C.FOURD_PRIZES["big"]["first"])


def test_incomplete_results_keep_tickets_pending(toto_df, fourd_df, rules):
    # A draw stored before its winning shares (TOTO) or all 23 numbers (4D) were published
    # must not settle a ticket; the ticket waits for the complete result.
    toto = toto_df.copy()
    fourd = fourd_df.copy()
    t_row = toto.index[-1]
    f_row = fourd.index[-1]
    toto.loc[t_row, "g7_winners"] = 0
    fourd.loc[f_row, "consolation_10"] = ""
    text = note(
        f"| TOTO | {iso(toto.loc[t_row, 'draw_date'])} | {toto_text(toto.loc[t_row])} | Ordinary | 1 |",
        f"| 4D | {iso(fourd.loc[f_row, 'draw_date'])} | {fourd.loc[f_row, 'first']} | Big | 1 |",
    )
    led, settled = settle_ledger(ledger_for(text), toto, fourd, rules, NOW)
    assert settled == []
    assert list(led["status"]) == ["pending", "pending"]
    # Once the complete result is stored, both settle.
    led, settled = settle_ledger(led, toto_df, fourd_df, rules, NOW)
    assert len(settled) == 2


# Editing a ticket that was already checked


def group7_numbers(row) -> str:
    """A TOTO set matching exactly 3 winning numbers of ``row`` (a Group 7 win)."""
    win = [int(row[f"n{i}"]) for i in range(1, 7)]
    others = [n for n in range(1, 50) if n not in win and n != int(row["additional"])]
    return " ".join(str(n) for n in sorted(win[:3] + others[:3]))


def settle_note(led, text, toto_df, fourd_df, rules, when):
    led = sync_ledger(led, parse_tickets(text), when, note_read=has_ticket_table(text))
    return settle_ledger(led, toto_df, fourd_df, rules, when)


def test_editing_the_cost_of_a_checked_ticket_counts_it_once(toto_df, fourd_df, rules):
    row = toto_df.iloc[-3]
    g7 = float(row["g7_share"])
    nums = group7_numbers(row)
    before = note(f"| TOTO | {iso(row['draw_date'])} | {nums} | Ordinary | $1 |")
    after = note(f"| TOTO | {iso(row['draw_date'])} | {nums} | Ordinary | $2 |")
    led, settled = settle_note(empty_ledger(), before, toto_df, fourd_df, rules, NOW)
    assert [r["winnings"] for r in settled] == [pytest.approx(g7)]

    led, settled = settle_note(led, after, toto_df, fourd_df, rules, NOW + timedelta(days=1))
    assert len(led) == 2
    old, new = led.iloc[0], led.iloc[1]
    assert old["status"] == "invalid"
    assert old["result"] == f"{REPLACED_PREFIX} (was: Group 7 x1, won ${g7:,.0f})"
    assert old["winnings"] == pytest.approx(g7)  # kept for the record, not counted
    assert new["status"] == "settled" and new["winnings"] == pytest.approx(2 * g7)
    assert [r["ticket_id"] for r in settled] == [new["ticket_id"]]
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(2.0) and totals["won"] == pytest.approx(2 * g7)
    assert totals["tickets"] == 1
    # Syncing the same note again changes nothing.
    again, settled_again = settle_note(led, after, toto_df, fourd_df, rules, NOW + timedelta(days=2))
    assert settled_again == []
    pd.testing.assert_frame_equal(again, led)

    # Putting the original line back brings the original figures back.
    led, settled = settle_note(led, before, toto_df, fourd_df, rules, NOW + timedelta(days=3))
    assert len(led) == 2
    assert list(led["status"]) == ["settled", "invalid"]
    assert led.iloc[1]["result"].startswith(REPLACED_PREFIX)
    assert [r["ticket_id"] for r in settled] == [led.iloc[0]["ticket_id"]]
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(1.0) and totals["won"] == pytest.approx(g7)
    assert totals["tickets"] == 1


def test_deleting_one_checked_row_and_editing_another_retires_only_the_edited_one(toto_df, fourd_df, rules):
    row = toto_df.iloc[-3]
    day = iso(row["draw_date"])
    a = group7_numbers(row)
    win = {int(row[f"n{i}"]) for i in range(1, 7)} | {int(row["additional"])}
    spare = [n for n in range(1, 50) if n not in win and str(n) not in a.split()]
    b = " ".join(str(n) for n in sorted(spare[:6]))
    b_typo = " ".join(str(n) for n in sorted(spare[:5] + spare[6:7]))  # one number corrected
    led, _ = settle_note(empty_ledger(), note(f"| TOTO | {day} | {a} | Ordinary | $1 |",
                                              f"| TOTO | {day} | {b} | Ordinary | $1 |"),
                         toto_df, fourd_df, rules, NOW)
    assert set(led["status"]) == {"settled"}

    led, settled = settle_note(led, note(f"| TOTO | {day} | {b_typo} | Ordinary | $1 |"),
                               toto_df, fourd_df, rules, NOW + timedelta(days=1))
    by_numbers = led.set_index("numbers")
    assert by_numbers.loc[a, "status"] == "settled"  # deleted only: still counted
    assert by_numbers.loc[b, "status"] == "invalid" and by_numbers.loc[b, "result"].startswith(REPLACED_PREFIX)
    assert by_numbers.loc[b_typo, "status"] == "settled"
    assert ledger_totals(led)["spent"] == pytest.approx(2.0) and ledger_totals(led)["tickets"] == 2


def test_editing_one_of_two_identical_checked_lines(toto_df, fourd_df, rules):
    row = toto_df.iloc[-3]
    line = f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |"
    led, _ = settle_note(empty_ledger(), note(line, line), toto_df, fourd_df, rules, NOW)
    edited = line.replace("$1", "$2")
    led, _ = settle_note(led, note(edited, line), toto_df, fourd_df, rules, NOW + timedelta(days=1))
    assert list(led["status"]) == ["settled", "invalid", "settled"]
    g7 = float(row["g7_share"])
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(3.0) and totals["won"] == pytest.approx(3 * g7)


def test_deleting_checked_rows_without_a_matching_new_line_keeps_them(toto_df, fourd_df, rules):
    row = toto_df.iloc[-3]
    day = iso(row["draw_date"])
    nums = group7_numbers(row)
    led, _ = settle_note(empty_ledger(), note(f"| TOTO | {day} | {nums} | Ordinary | $1 |"),
                         toto_df, fourd_df, rules, NOW)
    # Tidied away, while tickets for another draw and an unrelated set for the same draw came in.
    win = {int(row[f"n{i}"]) for i in range(1, 7)} | {int(row["additional"])}
    unrelated = " ".join(str(n) for n in [n for n in range(1, 50) if n not in win and str(n) not in nums.split()][:6])
    led, _ = settle_note(led, note(f"| TOTO | 5 Oct 2026 | {nums} | Ordinary | $1 |",
                                   f"| TOTO | {day} | {unrelated} | Ordinary | $1 |"),
                         toto_df, fourd_df, rules, NOW + timedelta(days=1))
    first = led.iloc[0]
    assert first["numbers"] == nums and first["draw_date"] == day and first["status"] == "settled"
    assert not led["result"].str.startswith(REPLACED_PREFIX).any()
    assert ledger_totals(led)["tickets"] == 3


def test_settled_rows_returns_checked_rows_in_order(toto_df, fourd_df, rules):
    row = toto_df.iloc[-3]
    text = note(f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |",
                "| TOTO | 5 Oct 2030 | 1 2 3 4 5 6 | Ordinary | $1 |")
    led, settled = settle_note(empty_ledger(), text, toto_df, fourd_df, rules, NOW)
    ids = list(led["ticket_id"])
    assert settled_rows(led, [ids[1], ids[0], "missing"]) == settled  # pending and unknown ids skipped
