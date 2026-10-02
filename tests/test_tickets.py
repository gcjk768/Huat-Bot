"""Tests for huatbot.tickets: reading Tickets.md, ticket ids and the ledger (TOTO only).

Results come from the synthetic history in conftest (no network). Expected prize amounts
are taken from the synthetic rows themselves, never hard coded.
"""
from __future__ import annotations

import re
import warnings
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import prizes
from huatbot.models import LEDGER_COLUMNS, PrizeRules, Ticket
from huatbot.store import empty_ledger, load_ledger, normalise_ledger, save_ledger
from huatbot.tickets import (
    FOURD_NOT_TRACKED,
    LEFT_NOTE_MARK,
    OTHER_GAME_NOT_TRACKED,
    REMOVED_RESULT,
    REPLACED_PREFIX,
    TICKETS_TEMPLATE,
    has_ticket_table,
    ledger_totals,
    parse_date,
    parse_tickets,
    replaced_rows,
    settle_ledger,
    settled_rows,
    sync_ledger,
    ticket_id,
    ticket_ids,
    ticket_units,
    toto_boards_for,
)

SG = ZoneInfo(C.SG_TZ_NAME)
NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)
STAMP = NOW.isoformat(timespec="seconds")
HEADER = "| Game | Draw date | Numbers | Bet type | Cost |\n| --- | --- | --- | --- | --- |\n"
DASHES = "-–—"


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


def old_fourd_row(status: str, **fields) -> dict:
    """A 4D ledger row as an older version of the bot wrote it."""
    row = {
        "ticket_id": f"old4d{status}", "game": "4D", "draw_date": "2026-09-30", "draw_number": None,
        "numbers": "0042", "bet_type": "Big", "cost": 2.0, "units": 2.0, "status": status,
        "result": "", "winnings": 0.0, "added_at": "2026-09-29T19:30:00+08:00", "checked_at": "",
        "source": "| 4D | 30 Sep 2026 | 0042 | Big | $2 |",
    }
    row.update(fields)
    return row


OLD_SETTLED = {"draw_number": 5432, "result": "3rd Prize x1", "winnings": 980.0,
               "checked_at": "2026-09-30T19:30:00+08:00"}


def with_rows(led: pd.DataFrame, *rows: dict) -> pd.DataFrame:
    """The ledger with these rows added at the end (written by hand, or by an older version)."""
    extra = normalise_ledger(pd.DataFrame(list(rows), columns=LEDGER_COLUMNS))
    return extra if len(led) == 0 else normalise_ledger(pd.concat([led, extra], ignore_index=True))


def row_of(led: pd.DataFrame, tid: str) -> dict:
    return led.set_index("ticket_id").loc[tid].to_dict()


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
    assert "4D" not in t and "iBet" not in t


def test_template_has_no_prose_dashes():
    separator = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")
    for line in TICKETS_TEMPLATE.splitlines():
        if any(ch in line for ch in DASHES):
            assert separator.match(line.strip()), line


def test_template_examples_are_valid_tickets():
    tickets = parse_tickets("\n".join(code_block_lines(TICKETS_TEMPLATE)))
    assert [t.error for t in tickets] == [None] * 3
    got = [(t.game, t.bet_type, t.cost) for t in tickets]
    assert got == [("TOTO", "Ordinary", 1.0), ("TOTO", "System 7", 7.0), ("TOTO", "Ordinary", 1.0)]
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


@pytest.mark.parametrize("bet", ["System 7", "Sys 7", "System7", "S7", "sys7", "SYSTEM 7", "s 7", "Sys. 7",
                                 "System-7", "System–7"])
def test_toto_system_spellings(bet):
    t = one(note(f"| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | {bet} | $7 |"))
    assert t.error is None and t.bet_type == "System 7"


def test_blank_bet_with_more_numbers_is_a_system_bet():
    t = one(note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 8 | | 28 |"))
    assert t.error is None and t.bet_type == "System 8"


def test_table_columns_mapped_by_header():
    text = ("| Cost | Game | Numbers | Draw date | Bet type | Notes |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| $7 | TOTO | 7 6 5 4 3 2 1 | 5 Oct 2026 | Sys 7 | birthday |\n")
    t = one(text)
    assert t.error is None
    assert (t.game, t.numbers, t.bet_type, t.cost, t.draw_date) == (
        "TOTO", "1 2 3 4 5 6 7", "System 7", 7.0, date(2026, 10, 5))


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
    # Without a ticket header, only rows that start with TOTO (or 4D, to say it is not tracked)
    # are read.
    text = ("| Lotto | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\n"
            "| 4D | 4 Oct 2026 | 1234 | Big | 1 |\n"
            "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\n")
    assert [(t.game, t.error) for t in parse_tickets(text)] == [("4D", FOURD_NOT_TRACKED), ("TOTO", None)]


def test_quoted_table_rows_are_read():
    text = "> " + HEADER.replace("\n", "\n> ") + "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\n"
    t = one(text)
    assert t.error is None and t.source.startswith("| TOTO")


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_code_blocks_are_ignored(fence):
    text = note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |") + (
        f"\n{fence}\n| TOTO | 5 Oct 2026 | 7 8 9 10 11 12 | Ordinary | 1 |\n"
        f"TOTO, 5 Oct 2026, 7 8 9 10 11 12, Ordinary, 1\n| broken |\n{fence}\n"
        "TOTO, 8 Oct 2026, 13 14 15 16 17 18, Ordinary, 1\n"
    )
    tickets = parse_tickets(text)
    assert [(t.game, t.numbers) for t in tickets] == [("TOTO", "1 2 3 4 5 6"), ("TOTO", "13 14 15 16 17 18")]


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


def test_plain_system_line():
    t = one("* TOTO, Mon 5 Oct 2026, 3 11 19 27 38 45 49, Sys 7, $7")
    assert t.error is None
    assert (t.numbers, t.bet_type, t.cost) == ("3 11 19 27 38 45 49", "System 7", 7.0)


def test_windows_line_endings_and_byte_order_mark():
    text = ("\ufeff| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |\r\n"
            "TOTO, 8 Oct 2026, 7 8 9 10 11 12, Ordinary, 1\r\n")
    tickets = parse_tickets(text)
    assert [(t.game, t.error, t.line_no) for t in tickets] == [("TOTO", None, 1), ("TOTO", None, 2)]


def test_mixed_note_keeps_file_order():
    text = note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |") + (
        "\nAlso bought:\n- TOTO, 8 Oct 2026, 9 10 11 12 13 14, Ordinary, 3\n- 4D, 4 Oct 2026, 9999, Big, 3\n"
    )
    tickets = parse_tickets(text)
    assert [(t.game, t.error is None) for t in tickets] == [("TOTO", True), ("TOTO", True), ("4D", False)]
    assert tickets[0].line_no < tickets[1].line_no < tickets[2].line_no


# 4D lines: the bot follows TOTO only


@pytest.mark.parametrize("line", [
    "| 4D | 4 Oct 2026 | 0042 | Big | $2 |",
    "| 4d | 4 Oct 2026 | 1234 | iBet Big | 1 |",
    "| four d | 4 Oct 2026 | 1234 | Small | 1 |",
    "| 4D | someday | 12 | Huge | free |",  # the game is the reason, whatever else is wrong
    "- 4D, Sun 4 Oct 2026, 0042, i-Bet, $1",
    "4D, 4 Oct 2026, 1234, Big, 1",
])
def test_fourd_lines_give_a_plain_error(line):
    tickets = parse_tickets(note(line))
    assert len(tickets) == 1
    t = tickets[0]
    assert t.game == "4D" and t.error == FOURD_NOT_TRACKED
    assert t.source == line.removeprefix("- ")
    assert not any(ch in t.error for ch in DASHES)
    assert ticket_ids(tickets) == [None]
    assert len(sync_ledger(empty_ledger(), tickets, NOW)) == 0


def test_fourd_lines_do_not_stop_the_toto_tickets():
    text = note(
        "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |",
        "| 4D | 4 Oct 2026 | 0042 | Big | $2 |",
    ) + "\n- TOTO, 8 Oct 2026, 7 8 9 10 11 12, Ordinary, 2\n"
    tickets = parse_tickets(text)
    assert [t.error for t in tickets] == [None, FOURD_NOT_TRACKED, None]
    led = sync_ledger(empty_ledger(), tickets, NOW)
    assert list(led["game"]) == ["TOTO", "TOTO"]
    assert ledger_totals(led)["spent"] == pytest.approx(3.0)


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
    ("| Lotto | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |", "the game must be TOTO"),
    ("| | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |", "the game is missing"),
    ("| 4D | 4 Oct 2026 | 1234 | Big | 1 |", "4D is not tracked any more"),
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
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 | System 7 | $1 |", "covers 7 boards at $1 each, so it costs at least $7"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 0.50 |", "an Ordinary ticket covers 1 board"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 0 |", "above 0"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | free |", "above 0"),
    ("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | |", "cost is missing"),
    ("| TOTO | | 1 2 3 4 5 6 | Ordinary | 1 |", "date is missing"),
    ("| TOTO | 5 Oct 2026 | | Ordinary | 1 |", "numbers are missing"),
])
def test_bad_rows_give_plain_errors(row, words):
    tickets = parse_tickets(note(row))
    assert len(tickets) == 1
    err = tickets[0].error
    assert err and words in err, err
    assert not any(ch in err for ch in DASHES), err  # user facing: no dashes
    assert tickets[0].source == row


def test_bad_line_keeps_what_was_read_before_the_problem():
    t = one(note("| toto | 5 Oct 2026 | 45 3 11 19 27 38 | Ordinary | free |"))
    assert t.error and (t.game, t.draw_date, t.numbers, t.bet_type) == (
        "TOTO", date(2026, 10, 5), "3 11 19 27 38 45", "Ordinary")


def test_bad_plain_lines_give_errors():
    tickets = parse_tickets("- TOTO, 5 Oct 2026, 1 2 3, Ordinary, 1\n"
                            "- TOTO, someday, 1 2 3 4 5 6, Ordinary, 1\n"
                            "- 4D, 4 Oct 2026, 1234, Big, 1\n")
    assert [bool(t.error) for t in tickets] == [True, True, True]
    assert "6 numbers" in tickets[0].error
    assert "could not be read" in tickets[1].error
    assert tickets[2].error == FOURD_NOT_TRACKED


@pytest.mark.parametrize("text", [
    "", "|", "||||", "| | |\n|---|", "```", "```\nTOTO, 1, 2, 3", "TOTO, , , , , , ,",
    "| TOTO |", "| 4D | x | y | z | w | v |", "---\n", "---\nbad: [\n---\nTOTO, 5 Oct 2026",
    "\x00\x01 | TOTO | — |", "4D,,,", "| TOTO | 99/99/9999 | 1 | 2 | 3 |", "> > |",
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


def test_ticket_id_system_spellings_and_hand_built_tickets():
    a = one(note("| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | Sys 7 | $7 |"))
    hand = Ticket(game="toto", draw_date=date(2026, 10, 5), numbers=" 49, 45 38 27 19 11 3",
                  bet_type="s7", cost=7, line_no=99, source="anything")
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
    row = "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |"
    tickets = parse_tickets(note(row, "| TOTO | 5 Oct 2026 | 1 2 3 4 5 | Ordinary | 1 |", row,
                                 "| 4D | 4 Oct 2026 | 1234 | Big | 1 |", row))
    ids = ticket_ids(tickets)
    assert ids[1] is None and ids[3] is None  # bad lines have no id
    assert ids[0] == ticket_id(tickets[0], 0)
    assert ids[2] == ticket_id(tickets[2], 1)
    assert ids[4] == ticket_id(tickets[4], 2)
    assert len({ids[0], ids[2], ids[4]}) == 3


# Units


@pytest.mark.parametrize("bet, cost, units", [
    ("Ordinary", 1, 1.0), ("Ordinary", 2, 2.0), ("System 7", 7, 1.0), ("System 8", 56, 2.0),
    ("System 12", 924, 1.0), ("ordinary", 3, 3.0),
])
def test_ticket_units(bet, cost, units):
    assert ticket_units(bet, cost) == pytest.approx(units)


def test_boards_for_every_bet_type():
    assert toto_boards_for("Ordinary") == 1
    for n, boards in C.TOTO_SYSTEM_BOARDS.items():
        assert toto_boards_for(f"System {n}") == boards
    with pytest.raises(ValueError):
        toto_boards_for("Big")


# sync_ledger


def test_sync_adds_valid_tickets_as_pending():
    text = note(
        "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | System 7 | $7 |",
        "| TOTO | 8 Oct 2026 | 1 2 3 4 5 6 | Ordinary | $2 |",
        "| TOTO | 8 Oct 2026 | 1 2 3 4 5 | Ordinary | $2 |",  # bad: not added
        "| 4D | 4 Oct 2026 | 0042 | Big | $2 |",  # not tracked: not added
    )
    led = ledger_for(text)
    assert list(led.columns) == LEDGER_COLUMNS
    assert len(led) == 2
    assert set(led["status"]) == {"pending"} and set(led["game"]) == {"TOTO"}
    system, ordinary = led.iloc[0], led.iloc[1]
    assert (system["game"], system["draw_date"], system["bet_type"]) == ("TOTO", "2026-10-05", "System 7")
    assert system["units"] == pytest.approx(1.0)  # $7 over 7 boards
    assert ordinary["numbers"] == "1 2 3 4 5 6" and ordinary["units"] == pytest.approx(2.0)
    assert system["added_at"] == STAMP
    assert pd.isna(system["draw_number"])
    assert system["source"].startswith("| TOTO")


def test_sync_is_idempotent_and_keeps_duplicates():
    row = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |"
    tickets = parse_tickets(note(row, row))
    first = sync_ledger(empty_ledger(), tickets, NOW)
    second = sync_ledger(first, tickets, NOW + timedelta(days=1))
    assert len(first) == 2  # two identical lines are two tickets
    pd.testing.assert_frame_equal(first, second)
    # Cosmetic edits do not add rows either.
    edited = parse_tickets(note("|toto| Mon 5 Oct 2026 |45 38 27 19 11 3| ord | $1.00 |", row))
    pd.testing.assert_frame_equal(sync_ledger(first, edited, NOW), first)


def test_sync_survives_csv_round_trip(tmp_path):
    text = note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 | Sys 7 | 7 |", "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | | 1 |")
    led = with_rows(ledger_for(text), old_fourd_row("settled", **OLD_SETTLED))
    path = tmp_path / "ledger.csv"
    save_ledger(led, path)
    loaded = load_ledger(path)
    assert loaded.loc[0, "numbers"] == "1 2 3 4 5 6 7"
    assert loaded.loc[2, "numbers"] == "0042"  # an old 4D row keeps its leading zeros
    again = sync_ledger(loaded, parse_tickets(text), NOW)
    assert len(again) == 3
    assert list(again["ticket_id"]) == list(led["ticket_id"])
    pd.testing.assert_frame_equal(again, loaded)


def test_sync_handles_none_and_empty_inputs():
    assert len(sync_ledger(None, [], NOW)) == 0
    led = sync_ledger(None, parse_tickets(note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |")), None)
    assert len(led) == 1 and led.loc[0, "added_at"]


def test_sync_never_deletes_and_retires_removed_unchecked_rows():
    typo = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 44 | Ordinary | 1 |"
    fixed = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |"
    other = "| TOTO | 1 Oct 2026 | 7 8 9 10 11 12 | Ordinary | 1 |"
    led = ledger_for(note(typo, other))
    led.loc[led["numbers"] == "7 8 9 10 11 12", "status"] = "settled"  # already checked: never touched again
    led = sync_ledger(led, parse_tickets(note(fixed, other)), NOW)
    assert len(led) == 3
    by_numbers = led.set_index("numbers")
    assert by_numbers.loc["3 11 19 27 38 44", "status"] == "invalid"
    assert by_numbers.loc["3 11 19 27 38 44", "result"] == REMOVED_RESULT
    assert by_numbers.loc["3 11 19 27 38 45", "status"] == "pending"
    assert by_numbers.loc["7 8 9 10 11 12", "status"] == "settled"
    assert ledger_totals(led)["spent"] == pytest.approx(2.0)  # the typo is not counted

    # The line comes back: the row is pending again. An empty note retires nothing.
    led = sync_ledger(led, parse_tickets(note(typo, fixed, other)), NOW)
    assert set(led["status"]) == {"pending", "settled"}
    assert len(sync_ledger(led, [], NOW)) == 3
    assert (sync_ledger(led, [], NOW)["status"] == led["status"]).all()
    kept = sync_ledger(led, parse_tickets(note(fixed)), NOW, retire_missing=False)
    assert (kept["status"] == led["status"]).all()


def test_emptying_the_ticket_table_retires_pending_rows():
    pending = "| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | 1 |"
    checked = "| TOTO | 1 Oct 2026 | 7 8 9 10 11 12 | Ordinary | 1 |"
    led = ledger_for(note(pending, checked))
    led.loc[led["numbers"] == "7 8 9 10 11 12", "status"] = "settled"
    # The user deleted every row but kept the table: the pending ticket leaves the totals.
    assert has_ticket_table(note())
    out = sync_ledger(led, parse_tickets(note()), NOW, note_read=True)
    by_numbers = out.set_index("numbers")
    assert by_numbers.loc["3 11 19 27 38 45", "status"] == "invalid"
    assert by_numbers.loc["3 11 19 27 38 45", "result"] == REMOVED_RESULT
    assert by_numbers.loc["7 8 9 10 11 12", "status"] == "settled"
    assert ledger_totals(out)["spent"] == pytest.approx(1.0)
    # A missing note, or one without a ticket table, retires nothing.
    for text in (None, "", "# My tickets\n\nNothing here yet.\n", "```\n" + HEADER + "```\n",
                 "<!--\n" + HEADER + "-->\n", "---\ntable: |\n" + HEADER + "---\n"):
        assert not has_ticket_table(text)
    assert has_ticket_table(TICKETS_TEMPLATE)
    assert has_ticket_table("> " + HEADER)
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


def test_toto_ticket_matching_the_draw_wins_group_1(toto_df, rules, toto_win_rows):
    shared, unwon = toto_win_rows
    text = note(
        f"| TOTO | {iso(shared['draw_date'])} | {toto_text(shared)} | Ordinary | 1 |",
        f"| TOTO | {iso(unwon['draw_date'])} | {toto_text(unwon)} | Ordinary | $2 |",
    )
    led, settled = settle_ledger(ledger_for(text), toto_df, rules, NOW)
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
    assert second["checked_at"] == STAMP
    assert set(first) == set(LEDGER_COLUMNS)


def test_toto_system7_ticket_uses_one_unit(toto_df, rules, toto_win_rows):
    shared, _ = toto_win_rows
    winning = [int(shared[f"n{i}"]) for i in range(1, 7)]
    extra = next(n for n in range(1, 50) if n not in winning and n != int(shared["additional"]))
    numbers = " ".join(str(n) for n in sorted(winning + [extra]))
    text = note(f"| TOTO | {iso(shared['draw_date'])} | {numbers} | Sys 7 | $7 |")
    led, settled = settle_ledger(ledger_for(text), toto_df, rules, NOW)
    expected = prizes.toto_ticket_prize(numbers, "System 7", 1.0, shared, rules)
    assert settled[0]["winnings"] == pytest.approx(expected.amount)
    assert settled[0]["result"] == expected.detail == "Group 3 x6, Group 1 x1"
    assert led.loc[0, "units"] == pytest.approx(1.0)


def test_future_and_non_draw_dates(toto_df, rules):
    latest = toto_df.iloc[-1]["draw_date"].date()
    some_draw = toto_df.iloc[-10]["draw_date"].date()
    gap_day = some_draw + timedelta(days=1)  # Mon or Thu draws: the next day has no draw
    assert gap_day not in set(toto_df["draw_date"].dt.date)
    future = latest + timedelta(days=3)
    text = note(
        f"| TOTO | {future.isoformat()} | 1 2 3 4 5 6 | Ordinary | 1 |",
        f"| TOTO | {gap_day.isoformat()} | 1 2 3 4 5 6 | Ordinary | 1 |",
        f"| TOTO | {future.isoformat()} | 7 8 9 10 11 12 | Ordinary | 1 |",
    )
    led, settled = settle_ledger(ledger_for(text), toto_df, rules, NOW)
    assert settled == []
    assert list(led["status"]) == ["pending", "no_draw", "pending"]
    assert led.loc[1, "result"] == "No draw on this date"
    totals = ledger_totals(led)
    assert totals["pending_cost"] == pytest.approx(2.0)
    assert totals["spent"] == pytest.approx(3.0)  # no_draw tickets still count as spent


def test_missing_draw_in_data_stays_pending(toto_df, rules):
    target = toto_df.iloc[-20]
    holed = toto_df[toto_df["draw_number"] != target["draw_number"]]
    text = note(f"| TOTO | {iso(target['draw_date'])} | {toto_text(target)} | Ordinary | 1 |")
    led, settled = settle_ledger(ledger_for(text), holed, rules, NOW)
    assert settled == [] and led.loc[0, "status"] == "pending"
    assert led.loc[0, "result"] == "Result not in the stored data yet"
    # Once the draw is stored the same row settles.
    led, settled = settle_ledger(led, toto_df, rules, NOW)
    assert len(settled) == 1 and settled[0]["result"] == "Group 1 x1"


def test_date_before_history_stays_pending(toto_df, rules):
    early = toto_df.iloc[0]["draw_date"].date() - timedelta(days=30)
    text = note(f"| TOTO | {early.isoformat()} | 1 2 3 4 5 6 | Ordinary | 1 |")
    led, settled = settle_ledger(ledger_for(text), toto_df, rules, NOW)
    assert settled == [] and led.loc[0, "status"] == "pending"
    assert "older than the stored history" in led.loc[0, "result"]


def test_settle_is_idempotent(toto_df, rules):
    row = toto_df.iloc[-3]
    text = note(f"| TOTO | {iso(row['draw_date'])} | 1 2 3 4 5 6 | Ordinary | 1 |",
                f"| TOTO | {iso(toto_df.iloc[-5]['draw_date'])} | 7 8 9 10 11 12 13 | System 7 | 7 |")
    led, settled = settle_ledger(ledger_for(text), toto_df, rules, NOW)
    assert len(settled) == 2
    again, settled_again = settle_ledger(led, toto_df, rules, NOW + timedelta(days=1))
    assert settled_again == []
    pd.testing.assert_frame_equal(led, again)
    # Sync after settle changes nothing either.
    pd.testing.assert_frame_equal(sync_ledger(again, parse_tickets(text), NOW), again)


def test_settle_without_results_or_ledger(rules, toto_df):
    led = ledger_for(note("| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |"))
    out, settled = settle_ledger(led, None, rules, NOW)
    assert settled == [] and out.loc[0, "status"] == "pending"
    out, settled = settle_ledger(None, toto_df, rules, NOW)
    assert len(out) == 0 and settled == []


def test_hand_edited_bad_rows_become_invalid(toto_df, rules):
    row = toto_df.iloc[-2]
    day = iso(row["draw_date"])
    led = ledger_for(note(f"| TOTO | {day} | 1 2 3 4 5 6 | Ordinary | 1 |",
                          "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 2 |",
                          f"| TOTO | {day} | 7 8 9 10 11 12 | Ordinary | 3 |",
                          f"| TOTO | {day} | 13 14 15 16 17 18 | Ordinary | 4 |"))
    led.loc[0, "numbers"] = "1 2 3"  # broken by hand in ledger.csv
    led.loc[1, "draw_date"] = "soon"
    led.loc[2, "bet_type"] = "Lucky Dip"
    led.loc[3, "cost"] = 0.0
    out, settled = settle_ledger(led, toto_df, rules, NOW)
    assert settled == []
    assert list(out["status"]) == ["invalid"] * 4
    assert out.loc[1, "result"] == "Draw date could not be read"
    assert all(r and not any(ch in r for ch in DASHES) for r in out["result"])
    assert ledger_totals(out)["spent"] == 0


def test_settle_raises_no_warnings(toto_df, rules):
    row = toto_df.iloc[-4]
    text = note(f"| TOTO | {iso(row['draw_date'])} | {toto_text(row)} | Ordinary | 1 |",
                "| TOTO | 5 Oct 2030 | 1 2 3 4 5 6 | Ordinary | 1 |")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        led = sync_ledger(empty_ledger(), parse_tickets(text), NOW)
        led = with_rows(led, old_fourd_row("pending"), old_fourd_row("settled", **OLD_SETTLED))
        led, _ = settle_ledger(led, toto_df, rules, NOW)
        ledger_totals(led)


def test_rules_default_used_when_missing(toto_df):
    # The page shows Group 7 winners but no share amount: the fixed prize from the rules is paid.
    toto = toto_df.copy()
    i = toto.index[-3]
    toto.loc[i, "g7_share"] = np.nan
    text = note(f"| TOTO | {iso(toto.loc[i, 'draw_date'])} | {group7_numbers(toto.loc[i])} | Ordinary | 1 |")
    _, settled = settle_ledger(ledger_for(text), toto, None, NOW)
    assert settled[0]["winnings"] == pytest.approx(C.TOTO_FIXED_PRIZES[7])
    _, settled = settle_ledger(ledger_for(text), toto, PrizeRules(fixed_prizes={5: 50.0, 6: 25.0, 7: 11.0}), NOW)
    assert settled[0]["winnings"] == pytest.approx(11.0)


def test_incomplete_results_keep_tickets_pending(toto_df, rules):
    # A draw stored before its winning shares were published, or with a number missing, must
    # not settle a ticket; the ticket waits for the complete result.
    toto = toto_df.copy()
    shares_missing, number_missing = toto.index[-1], toto.index[-2]
    toto.loc[shares_missing, "g7_winners"] = 0
    toto.loc[number_missing, "additional"] = 0
    text = note(*(f"| TOTO | {iso(toto.loc[i, 'draw_date'])} | {toto_text(toto.loc[i])} | Ordinary | 1 |"
                  for i in (shares_missing, number_missing)))
    led, settled = settle_ledger(ledger_for(text), toto, rules, NOW)
    assert settled == []
    assert list(led["status"]) == ["pending", "pending"]
    assert list(led["result"]) == ["", ""]
    # Once the complete result is stored, both settle.
    led, settled = settle_ledger(led, toto_df, rules, NOW)
    assert len(settled) == 2


# Ledger rows of other games from older versions


def test_unchecked_rows_of_other_games_become_invalid_and_checked_ones_stay(toto_df, rules):
    row = toto_df.iloc[-3]
    led = ledger_for(note(f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |"))
    old_settled = old_fourd_row("settled", **OLD_SETTLED)
    led = with_rows(
        led,
        old_fourd_row("pending"),
        old_fourd_row("no_draw", ticket_id="old4dnodraw", draw_date="2026-10-01", result="No draw on this date"),
        old_fourd_row("pending", ticket_id="old4dlower", game="4d"),
        old_fourd_row("pending", ticket_id="oldlotto", game="Lotto", numbers="1 2 3 4 5 6"),
        old_settled,
    )
    before = led.copy()
    out, settled = settle_ledger(led, toto_df, rules, NOW)

    assert [r["game"] for r in settled] == ["TOTO"]  # only the TOTO ticket is checked
    for tid in ("old4dpending", "old4dnodraw", "old4dlower"):
        got = row_of(out, tid)
        assert (got["status"], got["result"], got["checked_at"]) == ("invalid", FOURD_NOT_TRACKED, STAMP)
        assert got["winnings"] == 0
    assert row_of(out, "oldlotto")["result"] == OTHER_GAME_NOT_TRACKED
    assert not any(ch in OTHER_GAME_NOT_TRACKED + FOURD_NOT_TRACKED for ch in DASHES)
    # The checked 4D ticket is real history: left exactly as it was.
    assert row_of(out, "old4dsettled") == row_of(before, "old4dsettled")
    totals = ledger_totals(out)
    assert totals["tickets"] == 2 and totals["settled"] == 2 and totals["invalid"] == 4
    assert totals["spent"] == pytest.approx(1.0 + 2.0)
    assert totals["won"] == pytest.approx(settled[0]["winnings"] + 980.0)
    assert totals["wins"] == 2

    again, settled_again = settle_ledger(out, toto_df, rules, NOW + timedelta(days=1))
    assert settled_again == []
    pd.testing.assert_frame_equal(again, out)


def test_sync_leaves_rows_of_other_games_to_settle(toto_df, rules):
    toto_line = "| TOTO | 5 Oct 2030 | 1 2 3 4 5 6 | Ordinary | $1 |"
    old_line = "| 4D | 30 Sep 2026 | 0042 | Big | $2 |"  # the old line is still in the note
    led = with_rows(ledger_for(note(toto_line)), old_fourd_row("pending"),
                    old_fourd_row("settled", **OLD_SETTLED))
    for text in (note(toto_line, old_line), note(toto_line), note()):
        synced = sync_ledger(led, parse_tickets(text), NOW + timedelta(days=1), note_read=True)
        assert row_of(synced, "old4dpending")["status"] == "pending"  # not "removed from Tickets.md"
        assert row_of(synced, "old4dsettled") == row_of(led, "old4dsettled")  # no LEFT_NOTE_MARK
    synced = sync_ledger(led, parse_tickets(note(toto_line, old_line)), NOW, note_read=True)
    out, _ = settle_ledger(synced, toto_df, rules, NOW)
    assert row_of(out, "old4dpending")["result"] == FOURD_NOT_TRACKED
    assert row_of(out, "old4dsettled")["status"] == "settled"


def test_a_fourd_line_is_never_taken_for_an_edit_of_a_toto_row(toto_df, rules):
    row = toto_df.iloc[-3]
    day = iso(row["draw_date"])
    led, _ = settle_note(empty_ledger(), note(f"| TOTO | {day} | {group7_numbers(row)} | Ordinary | $1 |"),
                         toto_df, rules, NOW)
    # The checked row is deleted while a 4D line for the same day sits in the note: unlike an
    # unreadable TOTO line, it cannot be the row being fixed, so the row is marked as deleted.
    led, _ = settle_note(led, note(f"| 4D | {day} | 1234 | Big | $1 |"), toto_df, rules, NOW + timedelta(days=1))
    assert led.loc[0, "status"] == "settled" and LEFT_NOTE_MARK in led.loc[0, "checked_at"]


# ledger_totals


def test_ledger_totals():
    led = ledger_for(note(
        "| TOTO | 1 Oct 2026 | 1 2 3 4 5 6 | Ordinary | 1 |",
        "| TOTO | 5 Oct 2026 | 1 2 3 4 5 6 7 | System 7 | 7 |",
        "| TOTO | 28 Sep 2026 | 7 8 9 10 11 12 | Ordinary | 3 |",
        "| TOTO | 1 Oct 2026 | 13 14 15 16 17 18 | Ordinary | 2 |",
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


# Editing a ticket that was already checked


def group7_numbers(row) -> str:
    """A TOTO set matching exactly 3 winning numbers of ``row`` (a Group 7 win)."""
    win = [int(row[f"n{i}"]) for i in range(1, 7)]
    others = [n for n in range(1, 50) if n not in win and n != int(row["additional"])]
    return " ".join(str(n) for n in sorted(win[:3] + others[:3]))


def drawn_numbers(row) -> set[str]:
    return {str(int(row[f"n{i}"])) for i in range(1, 7)} | {str(int(row["additional"]))}


def spare_number(nums: str, row) -> str:
    """The lowest number that was not drawn and is not in ``nums``."""
    taken = drawn_numbers(row) | set(nums.split())
    return next(str(n) for n in range(1, 50) if str(n) not in taken)


def alike_numbers(nums: str, row) -> str:
    """``nums`` with one number that was not drawn swapped for another that was not drawn
    either: it looks like a typo fix and wins the same group."""
    parts = nums.split()
    out = next(p for p in reversed(parts) if p not in drawn_numbers(row))
    return " ".join(sorted([p for p in parts if p != out] + [spare_number(nums, row)], key=int))


def settle_note(led, text, toto_df, rules, when):
    led = sync_ledger(led, parse_tickets(text), when, note_read=has_ticket_table(text))
    return settle_ledger(led, toto_df, rules, when)


def test_editing_the_cost_of_a_checked_ticket_counts_it_once(toto_df, rules):
    row = toto_df.iloc[-3]
    g7 = float(row["g7_share"])
    nums = group7_numbers(row)
    before = note(f"| TOTO | {iso(row['draw_date'])} | {nums} | Ordinary | $1 |")
    after = note(f"| TOTO | {iso(row['draw_date'])} | {nums} | Ordinary | $2 |")
    led, settled = settle_note(empty_ledger(), before, toto_df, rules, NOW)
    assert [r["winnings"] for r in settled] == [pytest.approx(g7)]

    led, settled = settle_note(led, after, toto_df, rules, NOW + timedelta(days=1))
    assert len(led) == 2
    old, new = led.iloc[0], led.iloc[1]
    assert old["status"] == "invalid"
    assert old["result"] == (f"{REPLACED_PREFIX}: {nums} Ordinary $2 (was: Group 7 x1, won ${g7:,.0f}). "
                             "If both are real tickets, put this row back in Tickets.md")
    assert old["winnings"] == pytest.approx(g7)  # kept for the record, not counted
    assert new["status"] == "settled" and new["winnings"] == pytest.approx(2 * g7)
    assert [r["ticket_id"] for r in settled] == [new["ticket_id"]]
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(2.0) and totals["won"] == pytest.approx(2 * g7)
    assert totals["tickets"] == 1
    # Syncing the same note again changes nothing.
    again, settled_again = settle_note(led, after, toto_df, rules, NOW + timedelta(days=2))
    assert settled_again == []
    pd.testing.assert_frame_equal(again, led)

    # Putting the original line back brings the original figures back.
    led, settled = settle_note(led, before, toto_df, rules, NOW + timedelta(days=3))
    assert len(led) == 2
    assert list(led["status"]) == ["settled", "invalid"]
    assert led.iloc[1]["result"].startswith(REPLACED_PREFIX)
    assert [r["ticket_id"] for r in settled] == [led.iloc[0]["ticket_id"]]
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(1.0) and totals["won"] == pytest.approx(g7)
    assert totals["tickets"] == 1


def test_editing_the_bet_type_of_a_checked_ticket(toto_df, rules):
    row = toto_df.iloc[-3]
    nums = group7_numbers(row)
    system = " ".join(sorted(nums.split() + [spare_number(nums, row)], key=int))
    led, _ = settle_note(empty_ledger(), note(f"| TOTO | {iso(row['draw_date'])} | {nums} | Ordinary | $7 |"),
                         toto_df, rules, NOW)
    # The ticket was really a System 7 with one more number: numbers and bet type both change,
    # so it is not read as an edit; the cost is the same and the numbers look alike.
    led, settled = settle_note(led, note(f"| TOTO | {iso(row['draw_date'])} | {system} | System 7 | $7 |"),
                               toto_df, rules, NOW + timedelta(days=1))
    assert list(led["status"]) == ["settled", "settled"]
    assert replaced_rows(led, settled) == {}


def test_deleting_one_checked_row_and_editing_another_retires_only_the_edited_one(toto_df, rules):
    row = toto_df.iloc[-3]
    day = iso(row["draw_date"])
    a = group7_numbers(row)
    win = {int(row[f"n{i}"]) for i in range(1, 7)} | {int(row["additional"])}
    spare = [n for n in range(1, 50) if n not in win and str(n) not in a.split()]
    b = " ".join(str(n) for n in sorted(spare[:6]))
    b_typo = " ".join(str(n) for n in sorted(spare[:5] + spare[6:7]))  # one number corrected
    led, _ = settle_note(empty_ledger(), note(f"| TOTO | {day} | {a} | Ordinary | $1 |",
                                              f"| TOTO | {day} | {b} | Ordinary | $1 |"),
                         toto_df, rules, NOW)
    assert set(led["status"]) == {"settled"}

    led, settled = settle_note(led, note(f"| TOTO | {day} | {b_typo} | Ordinary | $1 |"),
                               toto_df, rules, NOW + timedelta(days=1))
    by_numbers = led.set_index("numbers")
    assert by_numbers.loc[a, "status"] == "settled"  # deleted only: still counted
    assert by_numbers.loc[b, "status"] == "invalid" and by_numbers.loc[b, "result"].startswith(REPLACED_PREFIX)
    assert by_numbers.loc[b_typo, "status"] == "settled"
    assert ledger_totals(led)["spent"] == pytest.approx(2.0) and ledger_totals(led)["tickets"] == 2


def test_editing_one_of_two_identical_checked_lines(toto_df, rules):
    row = toto_df.iloc[-3]
    line = f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |"
    led, _ = settle_note(empty_ledger(), note(line, line), toto_df, rules, NOW)
    edited = line.replace("$1", "$2")
    led, _ = settle_note(led, note(edited, line), toto_df, rules, NOW + timedelta(days=1))
    assert list(led["status"]) == ["settled", "invalid", "settled"]
    g7 = float(row["g7_share"])
    totals = ledger_totals(led)
    assert totals["spent"] == pytest.approx(3.0) and totals["won"] == pytest.approx(3 * g7)


def test_deleting_checked_rows_without_a_matching_new_line_keeps_them(toto_df, rules):
    row = toto_df.iloc[-3]
    day = iso(row["draw_date"])
    nums = group7_numbers(row)
    led, _ = settle_note(empty_ledger(), note(f"| TOTO | {day} | {nums} | Ordinary | $1 |"),
                         toto_df, rules, NOW)
    # Tidied away, while tickets for another draw and an unrelated set for the same draw came in.
    win = {int(row[f"n{i}"]) for i in range(1, 7)} | {int(row["additional"])}
    unrelated = " ".join(str(n) for n in [n for n in range(1, 50) if n not in win and str(n) not in nums.split()][:6])
    led, _ = settle_note(led, note(f"| TOTO | 5 Oct 2026 | {nums} | Ordinary | $1 |",
                                   f"| TOTO | {day} | {unrelated} | Ordinary | $1 |"),
                         toto_df, rules, NOW + timedelta(days=1))
    first = led.iloc[0]
    assert first["numbers"] == nums and first["draw_date"] == day and first["status"] == "settled"
    assert not led["result"].str.startswith(REPLACED_PREFIX).any()
    assert ledger_totals(led)["tickets"] == 3


# A checked row deleted in one run, and a similar ticket for the same draw added later


def test_a_checked_row_deleted_earlier_is_never_replaced_by_a_later_ticket(toto_df, rules):
    row = toto_df.iloc[-3]
    d = iso(row["draw_date"])
    nums = group7_numbers(row)
    led, settled = settle_note(empty_ledger(), note(f"| TOTO | {d} | {nums} | Ordinary | $1 |"),
                               toto_df, rules, NOW)
    won = settled[0]["winnings"]
    assert won == pytest.approx(float(row["g7_share"]))
    checked_at = led.loc[0, "checked_at"]

    # The user tidies Tickets.md: the checked row is deleted, the table stays.
    led, _ = settle_note(led, note(), toto_df, rules, NOW + timedelta(days=1))
    assert led.loc[0, "status"] == "settled"
    assert led.loc[0, "checked_at"].startswith(checked_at) and LEFT_NOTE_MARK in led.loc[0, "checked_at"]
    # An empty or unreadable note leaves the mark alone; syncing the same note again changes nothing.
    pd.testing.assert_frame_equal(sync_ledger(led, [], NOW + timedelta(days=2), note_read=False), led)
    pd.testing.assert_frame_equal(settle_note(led, note(), toto_df, rules, NOW + timedelta(days=2))[0], led)

    # Days later, two more tickets bought for the same draw are recorded: the same numbers at
    # another cost, and a set one number apart. Both look like edits of the deleted row.
    later = [f"| TOTO | {d} | {nums} | Ordinary | $2 |", f"| TOTO | {d} | {alike_numbers(nums, row)} | Ordinary | $1 |"]
    led, settled = settle_note(led, note(*later), toto_df, rules, NOW + timedelta(days=3))
    assert len(settled) == len(later)
    assert led.loc[0, "status"] == "settled" and led.loc[0, "winnings"] == pytest.approx(won)
    assert not led["result"].str.startswith(REPLACED_PREFIX).any()
    assert replaced_rows(led, settled) == {}
    totals = ledger_totals(led)
    assert totals["tickets"] == 3 and totals["spent"] == pytest.approx(1.0 + 2.0 + 1.0)
    assert totals["won"] == pytest.approx(won + sum(r["winnings"] for r in settled))


def test_deleting_a_checked_row_and_later_adding_another_ticket_keeps_both(toto_df, rules):
    row = toto_df.iloc[-3]
    d = iso(row["draw_date"])
    nums = group7_numbers(row)
    similar = alike_numbers(nums, row)  # one number apart: looks like a typo fix
    other = "| TOTO | 5 Oct 2030 | 1 2 3 4 5 6 | Ordinary | $1 |"
    led, settled = settle_note(empty_ledger(), note(f"| TOTO | {d} | {nums} | Ordinary | $1 |", other),
                               toto_df, rules, NOW)
    won = settled[0]["winnings"]
    led, _ = settle_note(led, note(other), toto_df, rules, NOW + timedelta(days=1))
    led, settled = settle_note(led, note(other, f"| TOTO | {d} | {similar} | Ordinary | $1 |"),
                               toto_df, rules, NOW + timedelta(days=2))
    assert led.loc[0, "status"] == "settled"
    totals = ledger_totals(led)
    assert totals["tickets"] == 3 and totals["spent"] == pytest.approx(3.0)
    assert totals["won"] == pytest.approx(won + settled[0]["winnings"])


def test_deleting_a_checked_row_and_adding_a_similar_one_together_is_a_correction(toto_df, rules):
    row = toto_df.iloc[-3]
    d = iso(row["draw_date"])
    nums = group7_numbers(row)
    similar = alike_numbers(nums, row)
    led, settled = settle_note(empty_ledger(), note(f"| TOTO | {d} | {nums} | Ordinary | $1 |"),
                               toto_df, rules, NOW)
    won = settled[0]["winnings"]
    led, settled = settle_note(led, note(f"| TOTO | {d} | {similar} | Ordinary | $1 |"),
                               toto_df, rules, NOW + timedelta(days=1))
    old = led.iloc[0]
    assert old["status"] == "invalid"
    assert old["result"] == (f"{REPLACED_PREFIX}: {similar} Ordinary $1 (was: Group 7 x1, won ${won:,.0f}). "
                             "If both are real tickets, put this row back in Tickets.md")
    assert not any(ch in old["result"] for ch in DASHES)
    pairs = replaced_rows(led, settled)
    assert list(pairs) == [settled[0]["ticket_id"]]
    was = pairs[settled[0]["ticket_id"]]
    assert was["numbers"] == nums and was["winnings"] == pytest.approx(won) and was["status"] == "invalid"
    assert replaced_rows(led, []) == {}


def test_a_deleted_checked_row_that_comes_back_can_be_edited_again(toto_df, rules):
    row = toto_df.iloc[-3]
    line = f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |"
    led, _ = settle_note(empty_ledger(), note(line), toto_df, rules, NOW)
    checked_at = led.loc[0, "checked_at"]
    led, _ = settle_note(led, note(), toto_df, rules, NOW + timedelta(days=1))
    assert LEFT_NOTE_MARK in led.loc[0, "checked_at"]
    led, _ = settle_note(led, note(line), toto_df, rules, NOW + timedelta(days=2))
    assert led.loc[0, "checked_at"] == checked_at  # back in the note: the mark is gone
    led, _ = settle_note(led, note(line.replace("$1", "$2")), toto_df, rules, NOW + timedelta(days=3))
    assert list(led["status"]) == ["invalid", "settled"]
    assert ledger_totals(led)["tickets"] == 1


def test_an_edit_first_saved_with_a_typo_still_replaces_the_checked_row(toto_df, rules):
    row = toto_df.iloc[-3]
    line = f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |"
    led, _ = settle_note(empty_ledger(), note(line), toto_df, rules, NOW)
    # The bot runs while the edited row cannot be read (the cost is being retyped).
    typo = line.replace("$1", "$")
    assert parse_tickets(note(typo))[0].error
    led, _ = settle_note(led, note(typo), toto_df, rules, NOW + timedelta(days=1))
    assert led.loc[0, "status"] == "settled" and LEFT_NOTE_MARK not in led.loc[0, "checked_at"]
    led, _ = settle_note(led, note(line.replace("$1", "$2")), toto_df, rules, NOW + timedelta(days=2))
    assert list(led["status"]) == ["invalid", "settled"]
    totals = ledger_totals(led)
    assert totals["tickets"] == 1 and totals["spent"] == pytest.approx(2.0)


def test_settled_rows_returns_checked_rows_in_order(toto_df, rules):
    row = toto_df.iloc[-3]
    text = note(f"| TOTO | {iso(row['draw_date'])} | {group7_numbers(row)} | Ordinary | $1 |",
                "| TOTO | 5 Oct 2030 | 1 2 3 4 5 6 | Ordinary | $1 |")
    led, settled = settle_note(empty_ledger(), text, toto_df, rules, NOW)
    ids = list(led["ticket_id"])
    assert settled_rows(led, [ids[1], ids[0], "missing"]) == settled  # pending and unknown ids skipped
