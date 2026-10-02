"""My tickets: read Tickets.md from the vault, keep ledger.csv, check tickets against results.

The user writes tickets in the Obsidian note ``Tickets.md`` (one per table row, or one per
line in the comma format). This module:

* ``parse_tickets`` reads the note into ``Ticket`` objects, normalising cosmetic differences
  (spacing, "$", case, "i-Bet" / "ibet", "Sys 7" / "S7", number order, ...). A line that
  cannot be read becomes a ``Ticket`` with ``error`` set to a plain reason; it never raises.
* ``ticket_id`` gives each ticket a stable id, so editing a line cosmetically does not create
  a second ledger row, while two identical lines (two real tickets) stay two rows.
* ``sync_ledger`` adds new tickets to the ledger, ``settle_ledger`` checks pending ones
  against the stored results using ``prizes``, and ``ledger_totals`` sums it all up.

Error texts end up in the vault (Ledger.md), so they are plain English with no dashes.
"""
from __future__ import annotations

import calendar
import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from . import constants as C
from . import prizes
from .models import LEDGER_COLUMNS, PrizeRules, Ticket
from .store import empty_ledger, normalise_ledger

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)

GAMES = ("TOTO", "4D")
FOURD_BET_TYPES = prizes.FOURD_BET_TYPES  # ("Big", "Small", "iBet Big", "iBet Small")
TOTO_BET_TYPES = ("Ordinary",) + tuple(f"System {n}" for n in sorted(C.TOTO_SYSTEM_BOARDS))

# Result text of a ledger row retired by ``sync_ledger`` because its line left Tickets.md
# before the draw was checked. Rows with this text come back if the line comes back.
REMOVED_RESULT = "Removed from Tickets.md before it was checked"

# Template

TICKETS_TEMPLATE: str = """# My tickets

Add every ticket you buy as a row of the table below. Huat Bot reads this note on each run, \
checks every ticket once its draw result is out, and keeps the running totals in [[Ledger]].

* **Game**: TOTO or 4D.
* **Draw date**: the date of the draw, such as 5 Oct 2026, Mon 5 Oct 2026 or 5/10/2026 \
(day first).
* **Numbers**: TOTO numbers separated by spaces or commas (6 for Ordinary, 7 for System 7 and so \
on, up to System 12). For 4D, the 4 digit number with any leading zeros, such as 0042.
* **Bet type**: TOTO Ordinary or System 7 to System 12. 4D Big, Small, iBet Big or iBet Small.
* **Cost**: what you paid in dollars, such as 1 or $7.

Old rows can stay: a ticket that has been checked stays in the ledger even if you delete its \
row later. Deleting a row before its draw is checked takes it out of the totals. If a row cannot \
be read, [[Ledger]] lists it with the reason, so you can fix it.

| Game | Draw date | Numbers | Bet type | Cost |
| --- | --- | --- | --- | --- |

## Examples

These sit inside a code block, so they are never read as tickets. Copy a row into the table \
above and change it to match your ticket.

```
| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | $1 |
| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | System 7 | $7 |
| 4D | 4 Oct 2026 | 0042 | Big | $2 |
| 4D | 4 Oct 2026 | 1234 | iBet Big | $1 |
```

You can also write one ticket per line anywhere outside the table, with commas between the parts:

```
TOTO, 5 Oct 2026, 3 11 19 27 38 45, Ordinary, 1
```
"""

# Markdown structure

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_COMMENTS = re.compile(r"<!--.*?-->|%%.*?%%", re.S)
_SEPARATOR_CELL = re.compile(r"^:?-+:?$")
# List markers in front of a plain ticket line: "- ", "* ", "+ ", "1. ", "1) ", checkboxes.
_LIST_MARKER = re.compile(r"^(?:[-*+]\s+|\d{1,3}[.)]\s+)(?:\[[ xX]\]\s+)?")
_QUOTE_MARKER = re.compile(r"^(?:>\s?)+")
_PIPE_SPLIT = re.compile(r"(?<!\\)\|")

FIELDS = ("game", "date", "numbers", "bet", "cost")
_FIELD_LABELS = {
    "game": "game",
    "date": "draw date",
    "numbers": "numbers",
    "bet": "bet type",
    "cost": "cost",
}
# Header cell text (lower case, punctuation removed) -> field.
_HEADER_NAMES = {
    "game": "game", "lottery": "game",
    "draw date": "date", "date": "date", "draw": "date",
    "numbers": "numbers", "number": "numbers", "nums": "numbers", "num": "numbers",
    "no": "numbers", "nos": "numbers",
    "bet type": "bet", "bet": "bet", "type": "bet",
    "cost": "cost", "price": "cost", "stake": "cost", "amount": "cost", "paid": "cost",
}

# Dates

_WEEKDAY_NAMES = tuple(calendar.day_name[i].lower() for i in range(7))  # monday .. sunday
_WEEKDAY_SHORT = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3,
                  "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
_MONTH_NAMES = tuple(calendar.month_name[i].lower() for i in range(1, 13))  # january ..
_ISO_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_DAY_FIRST = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})$")
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$")


class _LineError(ValueError):
    """A plain reason why a ticket line cannot be used (shown to the user as is)."""


# Small helpers


def _clean(text: Any) -> str:
    """Cell text without markdown emphasis or inline code marks, single spaced."""
    s = "" if text is None else str(text)
    s = re.sub(r"[`*]+|(?<!\w)_+|_+(?!\w)", "", s)
    s = s.replace("\\|", "|").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def _dollars(x: float) -> str:
    """"$7" or "$1.50" for messages (no dashes, no sign handling needed: costs are positive)."""
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def _weekday_of(token: str) -> int | None:
    """0 for "mon" / "monday" / "Mon.", ... ; None when the token is not a weekday."""
    t = token.lower().rstrip(".")
    if t in _WEEKDAY_SHORT:
        return _WEEKDAY_SHORT[t]
    return _WEEKDAY_NAMES.index(t) if t in _WEEKDAY_NAMES else None


def _month_of(token: str) -> int | None:
    """1 for "jan" / "january", 9 for "sep" / "sept" / "september"; None otherwise."""
    t = token.lower().rstrip(".")
    if len(t) < 3:
        return None
    for i, name in enumerate(_MONTH_NAMES, start=1):
        if name.startswith(t):
            return i
    return None


def _year(value: str) -> int:
    y = int(value)
    return 2000 + y if len(value) == 2 else y


def _make_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError:
        raise _LineError("the draw date is not a real date") from None


def parse_date(text: str) -> date:
    """A draw date from the forms people write.

    "5 Oct 2026", "05 October 2026", "Mon 5 Oct 2026", "Mon, 05 Oct 2026", "Oct 5 2026",
    "2026-10-05" (ISO) and "5/10/2026" or "5/10/26" (day first). A weekday, when given,
    must match the date. Raises ``ValueError`` with a plain reason otherwise.
    """
    s = _clean(text)
    if not s:
        raise _LineError("the draw date is missing")
    compact = s.replace(" ", "")
    if m := _ISO_DATE.match(compact):
        return _make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if m := _DAY_FIRST.match(compact):
        return _make_date(_year(m.group(3)), int(m.group(2)), int(m.group(1)))

    tokens = [t for t in re.split(r"[\s,/.\-]+", s.lower()) if t]
    weekday = None
    if tokens and _weekday_of(tokens[0]) is not None:
        weekday = _weekday_of(tokens[0])
        tokens = tokens[1:]
    day = month = year = None
    for tok in tokens:
        if m := _ORDINAL.match(tok):
            tok = m.group(1)
        if tok.isdigit():
            if len(tok) == 4 and year is None:
                year = int(tok)
            elif len(tok) <= 2 and day is None:
                day = int(tok)
            elif len(tok) == 2 and year is None:
                year = _year(tok)
            else:
                raise _LineError("the draw date could not be read, write it like 5 Oct 2026")
        elif (mo := _month_of(tok)) is not None and month is None:
            month = mo
        else:
            raise _LineError("the draw date could not be read, write it like 5 Oct 2026")
    if day is None or month is None:
        raise _LineError("the draw date could not be read, write it like 5 Oct 2026")
    if year is None:
        raise _LineError("the draw date needs a year, such as 5 Oct 2026")
    d = _make_date(year, month, day)
    if weekday is not None and weekday != d.weekday():
        raise _LineError(
            f"{d.day} {calendar.month_abbr[d.month]} {d.year} is a {calendar.day_name[d.weekday()]}, "
            f"but the line says {calendar.day_name[weekday]}"
        )
    return d


def parse_game(text: str) -> str:
    """"TOTO" or "4D" (any case, spaces ignored); ValueError otherwise."""
    s = _clean(text).lower().replace(" ", "")
    if s == "toto":
        return "TOTO"
    if s in ("4d", "fourd"):
        return "4D"
    if not s:
        raise _LineError("the game is missing, write TOTO or 4D")
    raise _LineError("the game must be TOTO or 4D")


def parse_cost(text: str) -> float:
    """Dollars paid: "$1", "1", "1.00", "S$ 7", "SGD 2" -> float above 0; ValueError otherwise."""
    s = _clean(text).lower()
    s = re.sub(r"^(?:sgd|s\$|\$)|(?:sgd|dollars?)$", "", s).strip()
    s = s.lstrip("$").replace(",", "").replace(" ", "")
    if not s:
        raise _LineError("the cost is missing, write what you paid such as $1")
    if not re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", s):
        raise _LineError("the cost must be a dollar amount above 0, such as $1")
    value = float(s)
    if not math.isfinite(value) or value <= 0:
        raise _LineError("the cost must be a dollar amount above 0, such as $1")
    return round(value, 2)


def _bet_key(text: str) -> str:
    """Lower case bet text with spaces, dots, dashes and underscores removed."""
    return re.sub(r"[\s.\-_‐‑–—]+", "", _clean(text).lower())


def normalise_toto_bet(text: str, count: int) -> str:
    """"Ordinary" or "System N" from "Ord", "ordinary", "", "Sys 7", "System7", "S7", ...

    A blank bet type means Ordinary with 6 numbers and System N with N numbers. The number
    count must match the bet type. Raises ValueError with a plain reason.
    """
    key = _bet_key(text)
    lowest, highest = min(C.TOTO_SYSTEM_BOARDS), max(C.TOTO_SYSTEM_BOARDS)
    if key == "":
        if count == C.TOTO_PICK:
            return "Ordinary"
        if count in C.TOTO_SYSTEM_BOARDS:
            return f"System {count}"
        raise _LineError(
            f"a TOTO ticket has 6 numbers (Ordinary) or {lowest} to {highest} numbers "
            f"(System {lowest} to System {highest}), this line has {count}"
        )
    if key in ("ord", "ordinary", "o"):
        if count != C.TOTO_PICK:
            raise _LineError(f"an Ordinary ticket has 6 numbers, this line has {count}")
        return "Ordinary"
    m = re.fullmatch(r"(?:system|sys|s)(\d{1,2})", key)
    if not m:
        raise _LineError(f"the bet type is not a TOTO bet, use Ordinary or System {lowest} "
                         f"to System {highest}")
    n = int(m.group(1))
    if n not in C.TOTO_SYSTEM_BOARDS:
        raise _LineError(f"System bets run from System {lowest} to System {highest}")
    if count != n:
        raise _LineError(f"System {n} needs {n} numbers, this line has {count}")
    return f"System {n}"


def normalise_fourd_bet(text: str) -> str:
    """"Big", "Small", "iBet Big" or "iBet Small".

    Accepts any case and the spellings "i-Bet", "ibet", "i bet", "iBet Big", "Big iBet".
    Plain "iBet" means "iBet Big". Raises ValueError with a plain reason.
    """
    key = _bet_key(text)
    if not key:
        raise _LineError("the 4D bet type is missing, use Big, Small, iBet Big or iBet Small")
    table = {
        "big": "Big", "small": "Small",
        "ibet": "iBet Big", "ibetbig": "iBet Big", "bigibet": "iBet Big",
        "ibetsmall": "iBet Small", "smallibet": "iBet Small",
    }
    if key in table:
        return table[key]
    raise _LineError("the bet type is not a 4D bet, use Big, Small, iBet Big or iBet Small")


def normalise_toto_numbers(text: str) -> list[int]:
    """Ascending TOTO numbers from "3 11 19 27 38 45", "3,11,19", "03, 11; 19", ...

    Checks the range and that no number repeats (the count is checked with the bet type).
    """
    s = _clean(text)
    tokens = [t for t in re.split(r"[\s,;/+]+|(?<=\d)[-.](?=\d)", s) if t]
    if not tokens:
        raise _LineError("the numbers are missing")
    if not all(t.isdigit() for t in tokens):
        raise _LineError("TOTO numbers must be whole numbers from 1 to 49")
    nums = [int(t) for t in tokens]
    for n in nums:
        if not 1 <= n <= C.TOTO_MAX_NUMBER:
            raise _LineError(f"{n} is not a TOTO number, they run from 1 to {C.TOTO_MAX_NUMBER}")
    repeats = sorted(n for n, k in Counter(nums).items() if k > 1)
    if repeats:
        raise _LineError(f"the number {repeats[0]} appears more than once")
    return sorted(nums)


def normalise_fourd_number_text(text: str) -> str:
    """The 4 digit number as written, leading zeros kept ("0042"); ValueError otherwise."""
    s = _clean(text).replace(" ", "")
    if not s:
        raise _LineError("the 4D number is missing")
    if not re.fullmatch(r"[0-9]{4}", s):
        raise _LineError("a 4D number has exactly 4 digits, such as 0042")
    return s


def toto_boards_for(bet_type: str) -> int:
    """Boards covered by a normalised TOTO bet type: 1 for Ordinary, C(n, 6) for System n."""
    m = re.fullmatch(r"System (\d+)", str(bet_type).strip())
    return C.TOTO_SYSTEM_BOARDS[int(m.group(1))] if m else 1


def ticket_units(game: str, bet_type: str, cost: float) -> float:
    """Stake multiplier used by the prize functions.

    TOTO: cost / boards (a $7 System 7 is 1 unit, $2 on an Ordinary set is 2 units).
    4D: the dollars staked.
    """
    if str(game).upper() == "TOTO":
        return float(cost) / (toto_boards_for(bet_type) * C.TOTO_BOARD_COST)
    return float(cost)


# Line parsing


@dataclass
class _Fields:
    """Raw text of the five parts of one ticket line."""

    game: str = ""
    date: str = ""
    numbers: str = ""
    bet: str = ""
    cost: str = ""


def _build_ticket(f: _Fields, line_no: int, source: str) -> Ticket:
    """Validate and normalise the five parts. Bad parts give a Ticket with ``error`` set."""
    game: str = _clean(f.game)
    draw_date: date | None = None
    numbers: str = _clean(f.numbers)
    bet_type: str = _clean(f.bet)
    cost: float = 0.0
    try:
        game = parse_game(f.game)
        draw_date = parse_date(f.date)
        if game == "TOTO":
            nums = normalise_toto_numbers(f.numbers)
            numbers = " ".join(str(n) for n in nums)
            bet_type = normalise_toto_bet(f.bet, len(nums))
        else:
            numbers = normalise_fourd_number_text(f.numbers)
            bet_type = normalise_fourd_bet(f.bet)
            if bet_type.startswith("iBet") and prizes.permutations_count(numbers) == 1:
                raise _LineError("iBet needs a number with at least two different digits")
        cost = parse_cost(f.cost)
        if game == "TOTO":
            boards = toto_boards_for(bet_type)
            least = boards * C.TOTO_BOARD_COST
            if cost + 1e-9 < least:
                what = "an Ordinary ticket" if boards == 1 else f"{bet_type}"
                raise _LineError(f"{what} covers {boards} {'board' if boards == 1 else 'boards'} "
                                 f"at {_dollars(C.TOTO_BOARD_COST)} each, so it costs at least "
                                 f"{_dollars(least)}")
    except _LineError as exc:
        return Ticket(game=game, draw_date=draw_date, numbers=numbers, bet_type=bet_type,
                      cost=cost, line_no=line_no, source=source, error=str(exc))
    return Ticket(game=game, draw_date=draw_date, numbers=numbers, bet_type=bet_type,
                  cost=cost, line_no=line_no, source=source)


def _split_row(line: str) -> list[str]:
    """Cells of a markdown table row (outer pipes optional, "\\|" kept inside a cell)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in _PIPE_SPLIT.split(s)]


def _is_separator_row(cells: list[str]) -> bool:
    filled = [c.replace(" ", "") for c in cells if c.strip()]
    return bool(filled) and all(_SEPARATOR_CELL.match(c) for c in filled)


def _header_map(cells: list[str]) -> dict[str, int] | None:
    """{field: column} when the row names the ticket columns (3 or more recognised), else None.

    Unrecognised columns fill the missing fields from left to right, so a renamed column
    still works as long as the order is the usual one.
    """
    found: dict[str, int] = {}
    for i, cell in enumerate(cells):
        key = re.sub(r"[^a-z ]+", " ", _clean(cell).lower())
        key = re.sub(r"\s+", " ", key).strip()
        name = _HEADER_NAMES.get(key)
        if name and name not in found:
            found[name] = i
    if len(found) < 3:
        return None
    free = [i for i in range(len(cells)) if i not in found.values()]
    for name in FIELDS:
        if name not in found and free:
            found[name] = free.pop(0)
    return found


def _fields_from_cells(cells: list[str], mapping: dict[str, int] | None) -> _Fields:
    if mapping is None:
        mapping = {name: i for i, name in enumerate(FIELDS)}
    values = {name: (cells[i] if i < len(cells) else "") for name, i in mapping.items()}
    return _Fields(**{name: values.get(name, "") for name in FIELDS})


def _looks_like_game(text: str) -> bool:
    try:
        parse_game(text)
    except _LineError:
        return False
    return True


def _fields_from_plain(parts: list[str]) -> _Fields:
    """Five parts from a comma separated line, tolerating commas inside the date and numbers.

    "TOTO, Mon, 5 Oct 2026, 3, 11, 19, 27, 38, 45, Ordinary, $1" works: the date may span up
    to 3 parts, the numbers take everything between the date and the bet type, and a missing
    bet type (numbers right before the cost) is allowed.
    """
    parts = [p.strip() for p in parts]
    f = _Fields(game=parts[0])
    rest = parts[1:]
    # The date: the shortest run of 1 to 3 parts that reads as a date.
    take = 1
    for k in (1, 2, 3):
        if k > len(rest):
            break
        try:
            parse_date(" ".join(rest[:k]))
        except ValueError:
            continue
        take = k
        break
    f.date = " ".join(rest[:take])
    rest = rest[take:]
    if not rest:
        return f
    f.cost = rest[-1]
    middle = rest[:-1]
    if middle and (re.search(r"[A-Za-z]", middle[-1]) or not middle[-1].strip()):
        f.bet = middle[-1]
        middle = middle[:-1]
    f.numbers = " ".join(p for p in middle if p)
    return f


def _strip_comments(text: str) -> str:
    """Blank out HTML and Obsidian comments, keeping line numbers."""
    return _COMMENTS.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _frontmatter_lines(lines: list[str]) -> int:
    """Number of leading lines taken by a YAML frontmatter block (0 if none)."""
    if not lines or lines[0].lstrip("﻿").rstrip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].rstrip() in ("---", "..."):
            return i + 1
    return 0


def parse_tickets(md_text: str) -> list[Ticket]:
    """Every ticket in the Tickets.md text, in file order.

    Read: markdown table rows (``| Game | Draw date | Numbers | Bet type | Cost |``, the
    columns may be reordered when the header names them) and plain lines in the comma format
    "TOTO, 5 Oct 2026, 3 11 19 27 38 45, Ordinary, 1", optionally with a list marker
    ("- ", "* ", "1. ", "- [x] ").
    Ignored: fenced code blocks, comments, frontmatter, headings, header and separator rows,
    blank rows, prose, and rows of other tables whose first cell is not TOTO or 4D.
    A line that is clearly a ticket but cannot be used gives a Ticket with ``error`` set.
    Never raises.
    """
    if not md_text:
        return []
    try:
        return _parse_tickets(str(md_text))
    except Exception:  # pragma: no cover, defensive: a bad note must never stop a run
        log.exception("could not read the tickets note")
        return []


def _parse_tickets(md_text: str) -> list[Ticket]:
    text = md_text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = _strip_comments(text).split("\n")
    tickets: list[Ticket] = []
    fence: tuple[str, int] | None = None  # (fence char, length) while inside a code block
    mapping: dict[str, int] | None = None  # column map of the current ticket table
    in_table = False
    start = _frontmatter_lines(lines)

    for idx in range(start, len(lines)):
        line_no = idx + 1
        raw = lines[idx]
        if m := _FENCE.match(raw):
            mark = m.group(1)
            if fence is None:
                fence = (mark[0], len(mark))
            elif mark[0] == fence[0] and len(mark) >= fence[1] and not raw[m.end():].strip():
                fence = None
            in_table, mapping = False, None
            continue
        if fence is not None:
            continue

        line = _QUOTE_MARKER.sub("", raw.strip()).strip()
        if not line or line.startswith("#"):
            in_table, mapping = False, None
            continue

        if line.startswith("|"):
            cells = _split_row(line)
            if _is_separator_row(cells):
                continue
            if not any(_clean(c) for c in cells):
                continue  # blank row
            if not in_table:
                in_table, mapping = True, None
            header = _header_map(cells)
            if header is not None:
                mapping = header  # a header row starts (or restarts) a ticket table
                continue
            nxt = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            if nxt.startswith("|") and _is_separator_row(_split_row(nxt)) \
                    and not _looks_like_game(cells[0]):
                continue  # header row of some other table
            f = _fields_from_cells(cells, mapping)
            if mapping is None and not _looks_like_game(f.game):
                log.debug("line %d: table row without a game, skipped", line_no)
                continue
            tickets.append(_build_ticket(f, line_no, line))
            continue

        in_table, mapping = False, None
        body = _LIST_MARKER.sub("", line, count=1).strip()
        parts = body.split(",")
        if len(parts) < 3 or not _looks_like_game(parts[0]):
            continue  # prose
        tickets.append(_build_ticket(_fields_from_plain(parts), line_no, body))

    bad = sum(1 for t in tickets if t.error)
    log.debug("read %d ticket lines, %d unreadable", len(tickets), bad)
    return tickets


# Ticket ids


def _canonical(t: Ticket) -> tuple[str, str, str, str, str]:
    """(game, ISO date, numbers, bet type, cost) normalised again, so hand built Tickets agree."""
    game = str(t.game or "").strip().upper()
    iso = t.draw_date.isoformat() if isinstance(t.draw_date, date) else str(t.draw_date or "")
    numbers = _clean(t.numbers)
    bet = _clean(t.bet_type)
    try:
        if game == "TOTO":
            nums = normalise_toto_numbers(numbers)
            numbers = " ".join(str(n) for n in nums)
            bet = normalise_toto_bet(bet, len(nums))
        elif game == "4D":
            numbers = normalise_fourd_number_text(numbers)
            bet = normalise_fourd_bet(bet)
    except ValueError:
        pass  # keep the cleaned text; the id is still stable for the same line
    try:
        cost = f"{float(t.cost):.2f}"
    except (TypeError, ValueError):
        cost = str(t.cost)
    return game, iso, numbers, bet, cost


def ticket_id(t: Ticket, occurrence: int) -> str:
    """Stable 12 hex character id: sha1 of the normalised fields plus the occurrence index.

    Cosmetic edits (spacing, "$", case, number order, "i-Bet" vs "iBet") keep the id.
    ``occurrence`` is 0 for the first identical ticket in the note, 1 for the second, ...
    so two identical lines are two tickets.
    """
    key = "|".join(_canonical(t) + (str(int(occurrence)),))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def ticket_ids(tickets: list[Ticket]) -> list[str | None]:
    """Ids for a parsed note, aligned with ``tickets`` (None for lines with an error)."""
    seen: Counter[tuple[str, ...]] = Counter()
    out: list[str | None] = []
    for t in tickets:
        if t.error:
            out.append(None)
            continue
        key = _canonical(t)
        out.append(ticket_id(t, seen[key]))
        seen[key] += 1
    return out


# Ledger


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(SG)


def _stamp(now: datetime | None) -> str:
    return _now(now).isoformat(timespec="seconds")


def _ledger_frame(ledger: pd.DataFrame | None) -> pd.DataFrame:
    if ledger is None or len(ledger) == 0:
        return empty_ledger()
    return normalise_ledger(ledger)


def sync_ledger(ledger: pd.DataFrame | None, tickets: list[Ticket], now: datetime | None,
                retire_missing: bool = True) -> pd.DataFrame:
    """Add every new valid ticket to the ledger as "pending". Never deletes a row.

    Idempotent: a ticket already in the ledger (same ``ticket_id``) is left alone, so
    syncing the same note twice changes nothing.

    ``retire_missing`` (an addition to the SPEC signature, on by default): a row that is
    still unchecked (pending or no_draw) whose line is no longer in the note is marked
    "invalid" with ``REMOVED_RESULT``, so fixing a typo in Tickets.md does not count the
    ticket twice. It is kept in ledger.csv and comes back as pending if the line returns.
    Nothing is retired when the note gave no ticket lines at all (an empty or unreadable
    note must not wipe the pending tickets).
    """
    df = _ledger_frame(ledger)
    ids = ticket_ids(tickets)
    current = {i for i in ids if i}
    stamp = _stamp(now)

    if retire_missing and tickets and len(df):
        status = df["status"]
        retire = status.isin(["pending", "no_draw"]) & ~df["ticket_id"].isin(current)
        restore = (status == "invalid") & (df["result"] == REMOVED_RESULT) & df["ticket_id"].isin(current)
        if retire.any():
            df.loc[retire, "status"] = "invalid"
            df.loc[retire, "result"] = REMOVED_RESULT
            df.loc[retire, "checked_at"] = stamp
            log.info("retired %d unchecked tickets no longer in the note", int(retire.sum()))
        if restore.any():
            df.loc[restore, "status"] = "pending"
            df.loc[restore, "result"] = ""
            df.loc[restore, "checked_at"] = ""

    known = set(df["ticket_id"])
    new_rows = []
    for t, tid in zip(tickets, ids):
        if tid is None or tid in known:
            continue
        known.add(tid)
        new_rows.append({
            "ticket_id": tid,
            "game": t.game,
            "draw_date": t.draw_date.isoformat() if t.draw_date else "",
            "draw_number": None,
            "numbers": t.numbers,
            "bet_type": t.bet_type,
            "cost": float(t.cost),
            "units": ticket_units(t.game, t.bet_type, t.cost),
            "status": "pending",
            "result": "",
            "winnings": 0.0,
            "added_at": stamp,
            "checked_at": "",
            "source": t.source,
        })
    if not new_rows:
        return df
    log.info("added %d new tickets to the ledger", len(new_rows))
    added = normalise_ledger(pd.DataFrame(new_rows, columns=LEDGER_COLUMNS))
    if len(df) == 0:
        return added
    return normalise_ledger(pd.concat([df, added], ignore_index=True))


@dataclass
class _DrawIndex:
    """Draws of one game by date, plus what is needed to prove there was no draw on a date."""

    by_date: dict[date, Any]  # date -> row (the highest draw number that day)
    dates: list[date]  # ascending by draw number
    numbers: list[int]

    @classmethod
    def build(cls, df: pd.DataFrame | None) -> "_DrawIndex":
        if df is None or len(df) == 0 or "draw_date" not in df or "draw_number" not in df:
            return cls({}, [], [])
        frame = df.copy()
        frame["draw_date"] = pd.to_datetime(frame["draw_date"], errors="coerce")
        frame["draw_number"] = pd.to_numeric(frame["draw_number"], errors="coerce")
        frame = frame[frame["draw_date"].notna() & (frame["draw_number"] > 0)]
        frame = frame.sort_values("draw_number")
        by_date: dict[date, Any] = {}
        for _, row in frame.iterrows():
            by_date[row["draw_date"].date()] = row
        return cls(by_date, [d.date() for d in frame["draw_date"]],
                   [int(n) for n in frame["draw_number"]])

    def status_without_draw(self, d: date) -> tuple[str, str]:
        """("no_draw" | "pending", result text) for a date with no draw in the data.

        "no_draw" only when the stored draws just before and just after the date have
        consecutive draw numbers, so no draw can be missing from the data in between.
        """
        before = [i for i, x in enumerate(self.dates) if x < d]
        after = [i for i, x in enumerate(self.dates) if x > d]
        if not after:
            return "pending", ""  # a future draw, or one not fetched yet
        if not before:
            return "pending", "Draw is older than the stored history"
        prev_no = max(self.numbers[i] for i in before)
        next_no = min(self.numbers[i] for i in after)
        if next_no == prev_no + 1:
            return "no_draw", "No draw on this date"
        return "pending", "Result not in the stored data yet"


def _result_ready(game: str, row: Any) -> bool:
    """True when the stored result row is complete enough to check a ticket against."""
    if game == "TOTO":
        try:
            winning, additional = prizes.toto_winning(row)
        except (TypeError, ValueError):
            return False
        ok = len(set(winning)) == C.TOTO_PICK and all(1 <= n <= C.TOTO_MAX_NUMBER for n in winning)
        # The winning shares table must be in: every TOTO draw has thousands of Group 7
        # winners, so zero means the prize amounts were not published (or read) yet.
        try:
            g7 = int(row["g7_winners"])
        except (KeyError, TypeError, ValueError):
            g7 = 0
        return ok and 1 <= additional <= C.TOTO_MAX_NUMBER and g7 > 0
    # All 23 numbers, so a ticket is never checked against a half published 4D result.
    return len(prizes.fourd_row_numbers(row)) == C.FOURD_NUMBERS_PER_DRAW


def _py(value: Any) -> Any:
    """Plain Python value for a ledger cell (NA -> None, numpy scalars -> int/float)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def settle_ledger(ledger: pd.DataFrame | None, toto_df: pd.DataFrame | None,
                  fourd_df: pd.DataFrame | None, rules: PrizeRules,
                  now: datetime | None) -> tuple[pd.DataFrame, list[dict]]:
    """Check unchecked tickets against the stored results.

    For each pending (or no_draw) row: the draw is the one with exactly the ticket's draw
    date. If it is stored, the row is settled with ``prizes.toto_ticket_prize`` (units =
    cost / boards) or ``prizes.fourd_ticket_prize`` (stake = cost). A date with no draw,
    proven by consecutive draw numbers around it, becomes "no_draw"; anything else stays
    pending (future draws, or a draw missing from the data). A row that cannot be checked at
    all (hand edited ledger.csv with bad values) becomes "invalid". Never raises for a row.

    Returns the ledger and the rows settled this run, as plain dicts of ledger columns.
    """
    df = _ledger_frame(ledger)
    if len(df) == 0:
        return df, []
    unchecked = df.index[df["status"].isin(["pending", "no_draw"])]
    if len(unchecked) == 0:
        return df, []
    rules = rules or PrizeRules()
    stamp = _stamp(now)
    index = {"TOTO": _DrawIndex.build(toto_df), "4D": _DrawIndex.build(fourd_df)}
    settled_idx: list[int] = []

    for i in unchecked:
        game = str(df.at[i, "game"]).strip().upper()
        try:
            d = date.fromisoformat(str(df.at[i, "draw_date"]).strip())
        except ValueError:
            df.at[i, "status"] = "invalid"
            df.at[i, "result"] = "Draw date could not be read"
            df.at[i, "checked_at"] = stamp
            continue
        if game not in index:
            df.at[i, "status"] = "invalid"
            df.at[i, "result"] = "Game must be TOTO or 4D"
            df.at[i, "checked_at"] = stamp
            continue

        draws = index[game]
        row = draws.by_date.get(d)
        if row is None or not _result_ready(game, row):
            status, result = draws.status_without_draw(d) if row is None else ("pending", "")
            if status != df.at[i, "status"] or result != df.at[i, "result"]:
                df.at[i, "status"] = status
                df.at[i, "result"] = result
                df.at[i, "checked_at"] = stamp
            continue

        cost = float(df.at[i, "cost"])
        bet_type = str(df.at[i, "bet_type"])
        numbers = str(df.at[i, "numbers"])
        try:
            if cost <= 0:
                raise ValueError("cost must be above 0")
            units = ticket_units(game, bet_type, cost)
            if game == "TOTO":
                res = prizes.toto_ticket_prize(numbers, bet_type, units, row, rules)
            else:
                res = prizes.fourd_ticket_prize(numbers, bet_type, cost, row, rules)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("ticket %s could not be checked: %s", df.at[i, "ticket_id"], exc)
            df.at[i, "status"] = "invalid"
            df.at[i, "result"] = "Ticket details could not be checked against the result"
            df.at[i, "checked_at"] = stamp
            continue

        df.at[i, "draw_number"] = int(row["draw_number"])
        df.at[i, "units"] = float(units)
        df.at[i, "status"] = "settled"
        df.at[i, "result"] = res.detail
        df.at[i, "winnings"] = float(res.amount)
        df.at[i, "checked_at"] = stamp
        settled_idx.append(i)

    df = normalise_ledger(df)
    settled = [
        {col: _py(df.at[i, col]) for col in LEDGER_COLUMNS} for i in settled_idx
    ]
    if settled:
        log.info("settled %d tickets, won %s in total", len(settled),
                 _dollars(sum(r["winnings"] or 0.0 for r in settled)))
    return df, settled


def ledger_totals(ledger: pd.DataFrame | None) -> dict:
    """Totals over the ledger.

    spent: cost of every ticket that is not invalid. won: winnings of settled tickets.
    net: won minus spent. pending_cost: cost of tickets still waiting for their draw.
    tickets: count of tickets that are not invalid. settled: count of settled tickets.
    Also: pending, no_draw, invalid (counts) and wins (settled tickets that won something).
    """
    df = _ledger_frame(ledger)
    status = df["status"]
    valid = status != "invalid"
    settled = status == "settled"
    spent = float(df.loc[valid, "cost"].sum())
    won = float(df.loc[settled, "winnings"].sum())
    return {
        "spent": round(spent, 2),
        "won": round(won, 2),
        "net": round(won - spent, 2),
        "pending_cost": round(float(df.loc[status == "pending", "cost"].sum()), 2),
        "tickets": int(valid.sum()),
        "settled": int(settled.sum()),
        "pending": int((status == "pending").sum()),
        "no_draw": int((status == "no_draw").sum()),
        "invalid": int((status == "invalid").sum()),
        "wins": int((settled & (df["winnings"] > 0)).sum()),
    }
