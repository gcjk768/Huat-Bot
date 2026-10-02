"""My tickets: read Tickets.md from the vault, keep ledger.csv, check TOTO tickets against results.

The user writes tickets in the Obsidian note ``Tickets.md`` (one per table row, or one per
line in the comma format). This module:

* ``parse_tickets`` reads the note into ``Ticket`` objects, normalising cosmetic differences
  (spacing, "$", case, "Sys 7" / "S7", number order, ...). A line that cannot be read becomes
  a ``Ticket`` with ``error`` set to a plain reason; it never raises. The bot follows TOTO
  only, so a 4D line is one of those, with ``FOURD_NOT_TRACKED`` as the reason.
* ``ticket_id`` gives each ticket a stable id, so editing a line cosmetically does not create
  a second ledger row, while two identical lines (two real tickets) stay two rows.
* ``sync_ledger`` adds new tickets to the ledger, ``settle_ledger`` checks pending ones
  against the stored results using ``prizes``, and ``ledger_totals`` sums it all up.

Ledger rows of other games written by older versions are kept: checked (settled) ones are
real history and are never touched; unchecked ones become "invalid" when settling.

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
from .textfmt import DASH_CHARS, money, plural, toto_nums

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)

GAMES = ("TOTO",)
TOTO_BET_TYPES = ("Ordinary",) + tuple(f"System {n}" for n in sorted(C.TOTO_SYSTEM_BOARDS))

# Reason given for a 4D line in Tickets.md, and result text of an unchecked 4D ledger row
# from an older version once ``settle_ledger`` retires it.
FOURD_NOT_TRACKED = "4D is not tracked any more, this bot follows TOTO only"
# Result text of an unchecked ledger row of any other game (a hand edited ledger.csv).
OTHER_GAME_NOT_TRACKED = "This game is not tracked, this bot follows TOTO only"
# Result text of a ledger row retired by ``sync_ledger`` because its line left Tickets.md
# before the draw was checked. Rows with this text come back if the line comes back.
REMOVED_RESULT = "Removed from Tickets.md before it was checked"
# Start of the result text of a checked ledger row that ``sync_ledger`` retired because the
# user edited its line in Tickets.md (a corrected ticket). The corrected line, the old result
# and winnings follow. Rows with this text come back if the original line comes back.
REPLACED_PREFIX = "Replaced by an edited line in Tickets.md"
# Added to the checked_at stamp of a checked (settled) ledger row once a sync finds its line
# gone from Tickets.md, with the time of that sync. The row was deleted, not edited: a ticket
# for the same draw added later is a separate ticket and never replaces it. The mark goes
# when the line comes back. ledger.csv has no column for it, so it rides on checked_at.
LEFT_NOTE_MARK = "left Tickets.md"

_UNCHECKED = ("pending", "no_draw")

# Template

TICKETS_TEMPLATE: str = """# My tickets

Add every TOTO ticket you buy as a row of the table below. Huat Bot reads this note on each \
run, checks every ticket once its draw result is out, and keeps the running totals in [[Ledger]].

* **Game**: TOTO (Huat Bot follows TOTO only).
* **Draw date**: the date of the draw, such as 5 Oct 2026, Mon 5 Oct 2026 or 5/10/2026 \
(day first).
* **Numbers**: your numbers separated by spaces or commas, 6 for Ordinary, 7 for System 7 and \
so on, up to System 12.
* **Bet type**: Ordinary, or System 7 to System 12.
* **Cost**: what you paid in dollars, such as 1 or $7.

Old rows can stay: a ticket that has been checked stays in the ledger even if you delete its \
row later. To fix a ticket that was already checked, edit its row, changing only the numbers, \
the bet type or the cost: the bot then replaces the old check with the corrected one. It can \
only tell an edit by timing: a checked row deleted and a similar ticket for the same draw added \
before the next run is read as a correction, and the bot says so in its results message. If \
both are real tickets, put the old row back. A wrong draw date, or a fix to more than one of \
these, is the exception: fix the row here, then delete the old row from Data/ledger.csv while \
the bot is stopped. Deleting a row before its draw is checked takes it out of the totals. If a \
row cannot be read, [[Ledger]] lists it with the reason, so you can fix it.

| Game | Draw date | Numbers | Bet type | Cost |
| --- | --- | --- | --- | --- |

## Examples

These sit inside a code block, so they are never read as tickets. Copy a row into the table \
above and change it to match your ticket.

```
| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | $1 |
| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | System 7 | $7 |
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
# Header cell text (lower case, punctuation removed) -> field.
_HEADER_NAMES = {
    "game": "game", "lottery": "game",
    "draw date": "date", "date": "date", "draw": "date",
    "numbers": "numbers", "number": "numbers", "nums": "numbers", "num": "numbers",
    "no": "numbers", "nos": "numbers",
    "bet type": "bet", "bet": "bet", "type": "bet",
    "cost": "cost", "price": "cost", "stake": "cost", "amount": "cost", "paid": "cost",
}
# Spaces, dots, underscores and dashes, which do not matter in a bet type ("Sys. 7", "S_7").
_BET_NOISE = re.compile(rf"[\s._{re.escape(DASH_CHARS)}]+")

# Dates

_WEEKDAY_NAMES = tuple(calendar.day_name[i].lower() for i in range(7))  # monday .. sunday
_WEEKDAY_SHORT = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3,
                  "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
_MONTH_NAMES = tuple(calendar.month_name[i].lower() for i in range(1, 13))  # january ..
_ISO_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_DAY_FIRST = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})$")
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$")
_UNREADABLE_DATE = "the draw date could not be read, write it like 5 Oct 2026"


class _LineError(ValueError):
    """A plain reason why a ticket line cannot be used (shown to the user as is)."""


# Small helpers


def _clean(text: Any) -> str:
    """Cell text without markdown emphasis or inline code marks, single spaced."""
    s = "" if text is None else str(text)
    s = re.sub(r"[`*]+|(?<!\w)_+|_+(?!\w)", "", s)
    s = s.replace("\\|", "|").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _dollars(x: float) -> str:
    """ "$7" for whole dollars, "$1.50" otherwise."""
    return money(x, cents=not float(x).is_integer())


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
    weekday = _weekday_of(tokens[0]) if tokens else None
    if weekday is not None:
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
                raise _LineError(_UNREADABLE_DATE)
        elif (mo := _month_of(tok)) is not None and month is None:
            month = mo
        else:
            raise _LineError(_UNREADABLE_DATE)
    if day is None or month is None:
        raise _LineError(_UNREADABLE_DATE)
    if year is None:
        raise _LineError("the draw date needs a year, such as 5 Oct 2026")
    d = _make_date(year, month, day)
    if weekday is not None and weekday != d.weekday():
        raise _LineError(
            f"{d.day} {calendar.month_abbr[d.month]} {d.year} is a {calendar.day_name[d.weekday()]}, "
            f"but the line says {calendar.day_name[weekday]}"
        )
    return d


def _game_key(text: Any) -> str | None:
    """ "TOTO" or "4D" for the ways people write those games (any case, spaces ignored), else
    None. 4D is still recognised so its lines get a clear reason instead of being skipped."""
    s = _clean(text).lower().replace(" ", "")
    if s == "toto":
        return "TOTO"
    if s in ("4d", "fourd"):
        return "4D"
    return None


def _not_tracked(game: Any) -> str:
    """Result text for a ledger row of a game the bot does not follow."""
    return FOURD_NOT_TRACKED if _game_key(game) == "4D" else OTHER_GAME_NOT_TRACKED


def parse_game(text: str) -> str:
    """ "TOTO" (any case, spaces ignored); ValueError with a plain reason otherwise, including
    ``FOURD_NOT_TRACKED`` for 4D."""
    game = _game_key(text)
    if game == "TOTO":
        return game
    if game == "4D":
        raise _LineError(FOURD_NOT_TRACKED)
    if not _clean(text):
        raise _LineError("the game is missing, write TOTO")
    raise _LineError("the game must be TOTO")


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


def normalise_toto_bet(text: str, count: int) -> str:
    """ "Ordinary" or "System N" from "Ord", "ordinary", "", "Sys 7", "System7", "S7", ...

    A blank bet type means Ordinary with 6 numbers and System N with N numbers. The number
    count must match the bet type. Raises ValueError with a plain reason.
    """
    key = _BET_NOISE.sub("", _clean(text).lower())
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


def normalise_toto_numbers(text: str) -> list[int]:
    """Ascending TOTO numbers from "3 11 19 27 38 45", "3,11,19", "03, 11; 19", ...

    Checks the range and that no number repeats (the count is checked with the bet type).
    """
    tokens = [t for t in re.split(r"[\s,;/+]+|(?<=\d)[-.](?=\d)", _clean(text)) if t]
    if not tokens:
        raise _LineError("the numbers are missing")
    if not all(t.isdigit() for t in tokens):
        raise _LineError(f"TOTO numbers must be whole numbers from 1 to {C.TOTO_MAX_NUMBER}")
    nums = [int(t) for t in tokens]
    for n in nums:
        if not 1 <= n <= C.TOTO_MAX_NUMBER:
            raise _LineError(f"{n} is not a TOTO number, they run from 1 to {C.TOTO_MAX_NUMBER}")
    repeats = sorted(n for n, k in Counter(nums).items() if k > 1)
    if repeats:
        raise _LineError(f"the number {repeats[0]} appears more than once")
    return sorted(nums)


def toto_boards_for(bet_type: str) -> int:
    """Boards a TOTO bet type covers: 1 for Ordinary, C(n, 6) for System n.

    ValueError for a bet type that is not one of ``TOTO_BET_TYPES`` (any case or spacing).
    """
    return math.comb(prizes.toto_bet_size(bet_type), C.TOTO_PICK)


def ticket_units(bet_type: str, cost: float) -> float:
    """Stake per board used by the prize functions: cost / boards.

    A $7 System 7 is 1 unit, $2 on an Ordinary set is 2 units.
    """
    return float(cost) / (toto_boards_for(bet_type) * C.TOTO_BOARD_COST)


def _check_cost(bet_type: str, cost: float) -> None:
    """Raise a plain reason when ``cost`` is less than the boards of the bet cost."""
    boards = toto_boards_for(bet_type)
    least = boards * C.TOTO_BOARD_COST
    if cost + 1e-9 < least:
        what = "an Ordinary ticket" if boards == 1 else bet_type
        raise _LineError(f"{what} covers {plural(boards, 'board')} at {_dollars(C.TOTO_BOARD_COST)} "
                         f"each, so it costs at least {_dollars(least)}")


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
    """Validate and normalise the five parts. Bad parts give a Ticket with ``error`` set to
    the first problem found, keeping whatever was read before it."""
    game = _game_key(f.game) or _clean(f.game)
    draw_date: date | None = None
    numbers = _clean(f.numbers)
    bet_type = _clean(f.bet)
    cost = 0.0
    error: str | None = None
    try:
        game = parse_game(f.game)
        draw_date = parse_date(f.date)
        nums = normalise_toto_numbers(f.numbers)
        numbers = toto_nums(nums)
        bet_type = normalise_toto_bet(f.bet, len(nums))
        cost = parse_cost(f.cost)
        _check_cost(bet_type, cost)
    except _LineError as exc:
        error = str(exc)
    return Ticket(game=game, draw_date=draw_date, numbers=numbers, bet_type=bet_type,
                  cost=cost, line_no=line_no, source=source, error=error)


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
        name = _HEADER_NAMES.get(re.sub(r"\s+", " ", key).strip())
        if name and name not in found:
            found[name] = i
    if len(found) < 3:
        return None
    free = [i for i in range(len(cells)) if i not in found.values()]
    for name in FIELDS:
        if name not in found and free:
            found[name] = free.pop(0)
    return found


def _is_header_row(line: str) -> bool:
    return line.startswith("|") and _header_map(_split_row(line)) is not None


def _fields_from_cells(cells: list[str], mapping: dict[str, int] | None) -> _Fields:
    if mapping is None:
        mapping = {name: i for i, name in enumerate(FIELDS)}
    return _Fields(**{name: (cells[i] if i < len(cells) else "") for name, i in mapping.items()})


def _names_a_game(text: str) -> bool:
    """True when the text is TOTO or 4D, so the line is meant as a ticket."""
    return _game_key(text) is not None


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
    for k in range(1, min(3, len(rest)) + 1):
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
    if not lines or lines[0].lstrip("\ufeff").rstrip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].rstrip() in ("---", "..."):
            return i + 1
    return 0


def _note_lines(md_text: str) -> list[tuple[int, str | None]]:
    """(line number from 1, text) for every line of the note that could hold a ticket.

    Frontmatter, comments and the inside of fenced code blocks are left out; quote markers
    and outer spaces are stripped. A fence line gives None for its text (it ends a table).
    """
    text = md_text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = _strip_comments(text).split("\n")
    out: list[tuple[int, str | None]] = []
    fence: tuple[str, int] | None = None  # (fence char, length) while inside a code block
    for idx in range(_frontmatter_lines(lines), len(lines)):
        raw = lines[idx]
        if m := _FENCE.match(raw):
            mark = m.group(1)
            if fence is None:
                fence = (mark[0], len(mark))
            elif mark[0] == fence[0] and len(mark) >= fence[1] and not raw[m.end():].strip():
                fence = None
            out.append((idx + 1, None))
        elif fence is None:
            out.append((idx + 1, _QUOTE_MARKER.sub("", raw.strip()).strip()))
    return out


def parse_tickets(md_text: str) -> list[Ticket]:
    """Every ticket in the Tickets.md text, in file order.

    Read: markdown table rows (``| Game | Draw date | Numbers | Bet type | Cost |``, the
    columns may be reordered when the header names them) and plain lines in the comma format
    "TOTO, 5 Oct 2026, 3 11 19 27 38 45, Ordinary, 1", optionally with a list marker
    ("- ", "* ", "1. ", "- [x] ").
    Ignored: fenced code blocks, comments, frontmatter, headings, header and separator rows,
    blank rows, prose, and rows of other tables whose first cell is not TOTO or 4D.
    A line that is clearly a ticket but cannot be used gives a Ticket with ``error`` set; a
    4D line is one, with ``FOURD_NOT_TRACKED``. Never raises.
    """
    if not md_text:
        return []
    try:
        return _parse_tickets(str(md_text))
    except Exception:  # pragma: no cover, defensive: a bad note must never stop a run
        log.exception("could not read the tickets note")
        return []


def _parse_tickets(md_text: str) -> list[Ticket]:
    entries = _note_lines(md_text)
    tickets: list[Ticket] = []
    mapping: dict[str, int] | None = None  # column map of the current ticket table

    for k, (line_no, line) in enumerate(entries):
        if not line or line.startswith("#"):  # fence, blank line or heading: a table ends
            mapping = None
            continue

        if line.startswith("|"):
            cells = _split_row(line)
            if _is_separator_row(cells) or not any(_clean(c) for c in cells):
                continue
            header = _header_map(cells)
            if header is not None:
                mapping = header  # a header row starts (or restarts) a ticket table
                continue
            nxt = entries[k + 1][1] if k + 1 < len(entries) else None
            if nxt and nxt.startswith("|") and _is_separator_row(_split_row(nxt)) \
                    and not _names_a_game(cells[0]):
                continue  # header row of some other table
            f = _fields_from_cells(cells, mapping)
            if mapping is None and not _names_a_game(f.game):
                log.debug("line %d: table row without a game, skipped", line_no)
                continue
            tickets.append(_build_ticket(f, line_no, line))
            continue

        mapping = None
        body = _LIST_MARKER.sub("", line, count=1).strip()
        parts = body.split(",")
        if len(parts) < 3 or not _names_a_game(parts[0]):
            continue  # prose
        tickets.append(_build_ticket(_fields_from_plain(parts), line_no, body))

    bad = sum(1 for t in tickets if t.error)
    log.debug("read %d ticket lines, %d unreadable", len(tickets), bad)
    return tickets


def has_ticket_table(md_text: str | None) -> bool:
    """True when the note has a ticket table header row (``| Game | Draw date | ...``) outside
    code blocks, comments and frontmatter. ``sync_ledger`` uses it to tell a table the user
    emptied on purpose from a missing or unreadable note."""
    if not md_text:
        return False
    return any(line and _is_header_row(line) for _, line in _note_lines(str(md_text)))


# Ticket ids


def _canonical(t: Ticket) -> tuple[str, str, str, str, str]:
    """(game, ISO date, numbers, bet type, cost) normalised again, so hand built Tickets agree."""
    game = str(t.game or "").strip().upper()
    iso = t.draw_date.isoformat() if isinstance(t.draw_date, date) else str(t.draw_date or "")
    numbers = _clean(t.numbers)
    bet = _clean(t.bet_type)
    if game == "TOTO":
        try:
            nums = normalise_toto_numbers(numbers)
            numbers = toto_nums(nums)
            bet = normalise_toto_bet(bet, len(nums))
        except ValueError:
            pass  # keep the cleaned text; the id is still stable for the same line
    try:
        cost = f"{float(t.cost):.2f}"
    except (TypeError, ValueError):
        cost = str(t.cost)
    return game, iso, numbers, bet, cost


def ticket_id(t: Ticket, occurrence: int) -> str:
    """Stable 12 hex character id: sha1 of the normalised fields plus the occurrence index.

    Cosmetic edits (spacing, "$", case, number order, "Sys 7" vs "System 7") keep the id.
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


def _stamp(now: datetime | None) -> str:
    return (now if now is not None else datetime.now(SG)).isoformat(timespec="seconds")


def _ledger_frame(ledger: pd.DataFrame | None) -> pd.DataFrame:
    if ledger is None or len(ledger) == 0:
        return empty_ledger()
    return normalise_ledger(ledger)


def _is_toto(df: pd.DataFrame) -> pd.Series:
    """Rows of the ledger that are TOTO tickets (older versions also kept other games)."""
    return df["game"].astype(str).str.strip().str.upper().eq("TOTO")


def _row_key(row: Any) -> tuple[str, str, str, str, str]:
    """``_canonical`` of a ledger row (its draw date is already ISO text)."""
    return _canonical(Ticket(game=str(row["game"]), draw_date=str(row["draw_date"]).strip(),  # type: ignore[arg-type]
                             numbers=str(row["numbers"]), bet_type=str(row["bet_type"]), cost=row["cost"],
                             line_no=0, source=""))


def _in_common(a: str, b: str) -> int:
    """How many numbers two TOTO sets ("3 11 19 27 38 45") share."""
    return len(set(a.split()) & set(b.split()))


def _numbers_alike(a: str, b: str) -> bool:
    """True when one TOTO set could be a typo of the other: they share at least half their
    numbers."""
    return 2 * _in_common(a, b) >= max(len(a.split()), len(b.split()))


def _checked_stamp(value: Any) -> str:
    """A ledger checked_at value without its ``LEFT_NOTE_MARK`` part."""
    return str(value or "").partition(LEFT_NOTE_MARK)[0].rstrip(" ;")


def _left_note(value: Any) -> bool:
    """True when a ledger checked_at value carries ``LEFT_NOTE_MARK``."""
    return LEFT_NOTE_MARK in str(value or "")


def _key_text(key: tuple) -> str:
    """ "3 11 19 27 38 45 Ordinary $2": numbers, bet type and cost of a ``_canonical`` key."""
    try:
        cost = _dollars(float(key[4]))
    except (TypeError, ValueError):
        cost = str(key[4])
    return f"{key[2]} {key[3]} {cost}"


def _match_edits(incoming: list[tuple], old: dict[int, tuple]) -> list[tuple[int, int]]:
    """Pairs (position in ``incoming``, ledger row of ``old``). Each incoming ticket key, in
    order, takes at most one old row (ledger index -> key): same game and draw date, and at
    most one of numbers, bet type and cost different (different numbers must still look
    alike). Closest first (fewest differences, then most numbers in common), then the
    earliest ledger row. One to one."""
    left = list(old)
    pairs: list[tuple[int, int]] = []
    for pos, key in enumerate(incoming):
        best = None
        for i in left:
            prev = old[i]
            if prev[:2] != key[:2]:
                continue
            diff = sum(prev[k] != key[k] for k in (2, 3, 4))
            if diff > 1 or (prev[2] != key[2] and not _numbers_alike(prev[2], key[2])):
                continue
            rank = (diff, -_in_common(prev[2], key[2]), i)
            if best is None or rank < best:
                best = rank
        if best is not None:
            left.remove(best[2])
            pairs.append((pos, best[2]))
    return pairs


def _edited_rows(df: pd.DataFrame, incoming: list[tuple], current: set[str]) -> list[tuple[int, int]]:
    """Checked rows replaced by an edited line, as (position in ``incoming``, ledger row).

    Each incoming ticket key (a new or restored line, in note order) can take a settled TOTO
    row whose line left the note in this sync (see ``_match_edits``). A row whose line was
    already gone at an earlier sync carries ``LEFT_NOTE_MARK``: it was deleted, not edited,
    so a ticket for the same draw added later is a separate ticket and never replaces it."""
    candidates = (df["status"] == "settled") & _is_toto(df) & ~df["ticket_id"].isin(current)
    gone = {i: _row_key(df.loc[i]) for i in df.index[candidates]
            if not _left_note(df.at[i, "checked_at"])}
    return _match_edits(incoming, gone)


def _could_be(t: Ticket, row: Any) -> bool:
    """True when the unreadable line ``t`` could be the TOTO ledger row: its draw date is the
    row's or could not be read. A 4D line never could, since it can never be fixed into a
    TOTO ticket."""
    if _game_key(t.game) == "4D":
        return False
    day = t.draw_date.isoformat() if isinstance(t.draw_date, date) else ""
    return not day or day == str(row["draw_date"]).strip()


def _mark_left_note(df: pd.DataFrame, current: set[str], stamp: str, unreadable: list[Ticket]) -> None:
    """After a sync that read the note: add ``LEFT_NOTE_MARK`` to settled TOTO rows whose line
    is gone and clear it from rows whose line is back. A row an unreadable line could be is
    not marked yet, since that line may be its edit (it is marked once the note reads
    cleanly). Rows of other games are history and are left as they are."""
    for i in df.index[(df["status"] == "settled") & _is_toto(df)]:
        value = str(df.at[i, "checked_at"] or "")
        if df.at[i, "ticket_id"] in current:
            if _left_note(value):
                df.at[i, "checked_at"] = _checked_stamp(value)
        elif not _left_note(value) and not any(_could_be(t, df.loc[i]) for t in unreadable):
            df.at[i, "checked_at"] = f"{value}; {LEFT_NOTE_MARK} {stamp}" if value else f"{LEFT_NOTE_MARK} {stamp}"


def sync_ledger(ledger: pd.DataFrame | None, tickets: list[Ticket], now: datetime | None,
                retire_missing: bool = True, note_read: bool = False) -> pd.DataFrame:
    """Add every new valid ticket to the ledger as "pending". Never deletes a row.

    Idempotent: a ticket already in the ledger (same ``ticket_id``) is left alone, so
    syncing the same note twice changes nothing.

    ``retire_missing`` (on by default): a TOTO row that is still unchecked (pending or
    no_draw) whose line is no longer in the note is marked "invalid" with ``REMOVED_RESULT``,
    so fixing a typo in Tickets.md does not count the ticket twice. It is kept in ledger.csv
    and comes back as pending if the line returns.
    A checked (settled) row whose line was edited, so that a new line for the same draw date
    differs from it in only one of numbers, bet type and cost, is marked "invalid" with
    ``REPLACED_PREFIX``, the corrected line and its old result: the corrected line is checked
    instead and the ticket is counted once (``replaced_rows`` pairs them again). Only a row
    whose line left the note in this sync counts as edited. A checked row whose line was only
    deleted stays settled: its checked_at gets ``LEFT_NOTE_MARK``, so a similar ticket for the
    same draw added in a later sync is a new ticket, not an edit. A retired row comes back
    (pending, checked again) if its line returns, and then retires the edited row in the same
    way.
    Nothing is retired when the note gave no ticket lines, unless ``note_read`` says the
    note was read and still has its ticket table (the user deleted every row); an empty or
    unreadable note must not wipe the pending tickets. Rows of other games from older
    versions are never touched here (``settle_ledger`` retires the unchecked ones).
    """
    df = _ledger_frame(ledger)
    ids = ticket_ids(tickets)
    current = {i for i in ids if i}
    stamp = _stamp(now)
    check = retire_missing and (bool(tickets) or note_read)
    restored: set[str] = set()

    if check and len(df):
        status = df["status"]
        result = df["result"].astype(str)
        listed = df["ticket_id"].isin(current)
        retire = status.isin(_UNCHECKED) & _is_toto(df) & ~listed
        restore = ((status == "invalid") & (result.eq(REMOVED_RESULT) | result.str.startswith(REPLACED_PREFIX))
                   & listed)
        if retire.any():
            df.loc[retire, "status"] = "invalid"
            df.loc[retire, "result"] = REMOVED_RESULT
            df.loc[retire, "checked_at"] = stamp
            log.info("retired %d unchecked tickets no longer in the note", int(retire.sum()))
        if restore.any():
            restored = set(df.loc[restore, "ticket_id"])
            df.loc[restore, "status"] = "pending"
            df.loc[restore, "result"] = ""
            df.loc[restore, "winnings"] = 0.0
            df.loc[restore, "draw_number"] = pd.NA
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
            "units": ticket_units(t.bet_type, t.cost),
            "status": "pending",
            "result": "",
            "winnings": 0.0,
            "added_at": stamp,
            "checked_at": "",
            "source": t.source,
        })
    if new_rows:
        log.info("added %d new tickets to the ledger", len(new_rows))
        added = normalise_ledger(pd.DataFrame(new_rows, columns=LEDGER_COLUMNS))
        df = added if len(df) == 0 else normalise_ledger(pd.concat([df, added], ignore_index=True))

    if check and len(df):
        fresh = {r["ticket_id"] for r in new_rows} | restored
        incoming = [_canonical(t) for t, tid in zip(tickets, ids) if tid in fresh]
        replaced = _edited_rows(df, incoming, current)
        for pos, i in replaced:
            won = _dollars(float(df.at[i, "winnings"] or 0.0))
            was = str(df.at[i, "result"] or "") or "No prize"
            now_text = _key_text(incoming[pos])
            log.info("%s %s %s for %s was treated as corrected to %s (it had won %s)", df.at[i, "game"],
                     df.at[i, "numbers"], df.at[i, "bet_type"], df.at[i, "draw_date"], now_text, won)
            df.at[i, "status"] = "invalid"
            df.at[i, "result"] = (f"{REPLACED_PREFIX}: {now_text} (was: {was}, won {won}). If both are "
                                  "real tickets, put this row back in Tickets.md")
            df.at[i, "checked_at"] = stamp
        if replaced:
            log.info("replaced %d checked tickets by their edited lines in the note", len(replaced))
        _mark_left_note(df, current, stamp, [t for t in tickets if t.error])
    return df


def _py(value: Any) -> Any:
    """Plain Python value for a ledger cell (NA -> None, numpy scalars -> int/float)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_dict(df: pd.DataFrame, i: Any) -> dict:
    """Ledger row ``i`` as a plain dict of the ledger columns."""
    return {col: _py(df.at[i, col]) for col in LEDGER_COLUMNS}


def replaced_rows(ledger: pd.DataFrame | None, rows: list[dict]) -> dict[str, dict]:
    """The checked row each of ``rows`` replaced as an edited line, keyed by its ticket_id.

    ``rows`` are ledger rows as plain dicts (as ``settle_ledger`` and ``settled_rows`` give
    them). ``sync_ledger`` stamps a replaced row with the time of the sync, and the corrected
    line is checked in the same run with the same time, so the pairs are found again by
    matching rows with the same stamp the way ``sync_ledger`` matched them. Rows that
    replaced nothing are left out. The replaced rows come back as plain dicts of the ledger
    columns (status "invalid", their old result in ``result``, the old ``winnings``).
    """
    df = _ledger_frame(ledger)
    if not rows or len(df) == 0:
        return {}
    gone = df.index[(df["status"] == "invalid") & df["result"].astype(str).str.startswith(REPLACED_PREFIX)]
    by_stamp: dict[str, dict[int, tuple]] = {}
    for i in gone:
        by_stamp.setdefault(_checked_stamp(df.at[i, "checked_at"]), {})[i] = _row_key(df.loc[i])
    groups: dict[str, list[dict]] = {}
    for r in rows:
        stamp = _checked_stamp(r.get("checked_at"))
        if stamp and stamp in by_stamp:
            groups.setdefault(stamp, []).append(r)
    out: dict[str, dict] = {}
    for stamp, group in groups.items():
        for pos, i in _match_edits([_row_key(r) for r in group], by_stamp[stamp]):
            out[str(group[pos]["ticket_id"])] = _row_dict(df, i)
    return out


@dataclass
class _DrawIndex:
    """TOTO draws by date, plus what is needed to prove there was no draw on a date."""

    by_date: dict[date, Any]  # date -> row (the highest draw number that day)
    draws: list[tuple[date, int]]  # (date, draw number), ascending by draw number

    @classmethod
    def build(cls, df: pd.DataFrame | None) -> _DrawIndex:
        if df is None or len(df) == 0 or "draw_date" not in df or "draw_number" not in df:
            return cls({}, [])
        frame = df.copy()
        frame["draw_date"] = pd.to_datetime(frame["draw_date"], errors="coerce")
        frame["draw_number"] = pd.to_numeric(frame["draw_number"], errors="coerce")
        frame = frame[frame["draw_date"].notna() & (frame["draw_number"] > 0)].sort_values("draw_number")
        by_date: dict[date, Any] = {}
        for _, row in frame.iterrows():
            by_date[row["draw_date"].date()] = row
        draws = [(d.date(), int(n)) for d, n in zip(frame["draw_date"], frame["draw_number"])]
        return cls(by_date, draws)

    def status_without_draw(self, d: date) -> tuple[str, str]:
        """("no_draw" | "pending", result text) for a date with no draw in the data.

        "no_draw" only when the stored draws just before and just after the date have
        consecutive draw numbers, so no draw can be missing from the data in between.
        """
        prev_no = max((n for x, n in self.draws if x < d), default=None)
        next_no = min((n for x, n in self.draws if x > d), default=None)
        if next_no is None:
            return "pending", ""  # a future draw, or one not fetched yet
        if prev_no is None:
            return "pending", "Draw is older than the stored history"
        if next_no == prev_no + 1:
            return "no_draw", "No draw on this date"
        return "pending", "Result not in the stored data yet"


def _result_ready(row: Any) -> bool:
    """True when the stored TOTO result row is complete enough to check a ticket against."""
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


def _set_status(df: pd.DataFrame, i: Any, status: str, result: str, stamp: str) -> None:
    df.at[i, "status"] = status
    df.at[i, "result"] = result
    df.at[i, "checked_at"] = stamp


def settle_ledger(ledger: pd.DataFrame | None, toto_df: pd.DataFrame | None,
                  rules: PrizeRules | None, now: datetime | None) -> tuple[pd.DataFrame, list[dict]]:
    """Check unchecked tickets against the stored TOTO results.

    For each pending (or no_draw) TOTO row: the draw is the one with exactly the ticket's draw
    date. If it is stored with its complete result (the winning shares table included), the
    row is settled with ``prizes.toto_ticket_prize`` (units = cost / boards). A date with no
    draw, proven by consecutive draw numbers around it, becomes "no_draw"; anything else
    stays pending (future draws, or a draw missing or incomplete in the data). A row that
    cannot be checked at all (hand edited ledger.csv with bad values) becomes "invalid".
    An unchecked row of another game from an older version becomes "invalid" with
    ``FOURD_NOT_TRACKED`` (or ``OTHER_GAME_NOT_TRACKED``); settled rows of other games are
    real history and stay as they are. Never raises for a row.

    Returns the ledger and the rows settled this run, as plain dicts of ledger columns.
    """
    df = _ledger_frame(ledger)
    unchecked = df.index[df["status"].isin(_UNCHECKED)]
    if len(unchecked) == 0:
        return df, []
    rules = rules or PrizeRules()
    stamp = _stamp(now)
    draws = _DrawIndex.build(toto_df)
    settled_idx: list[Any] = []

    for i in unchecked:
        game = str(df.at[i, "game"]).strip().upper()
        if game != "TOTO":
            _set_status(df, i, "invalid", _not_tracked(game), stamp)
            continue
        try:
            d = date.fromisoformat(str(df.at[i, "draw_date"]).strip())
        except ValueError:
            _set_status(df, i, "invalid", "Draw date could not be read", stamp)
            continue

        row = draws.by_date.get(d)
        if row is None or not _result_ready(row):
            status, result = draws.status_without_draw(d) if row is None else ("pending", "")
            if status != df.at[i, "status"] or result != df.at[i, "result"]:
                _set_status(df, i, status, result, stamp)
            continue

        try:
            cost = float(df.at[i, "cost"])
            if not cost > 0:
                raise ValueError("cost must be above 0")
            bet_type = str(df.at[i, "bet_type"])
            units = ticket_units(bet_type, cost)
            res = prizes.toto_ticket_prize(str(df.at[i, "numbers"]), bet_type, units, row, rules)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("ticket %s could not be checked: %s", df.at[i, "ticket_id"], exc)
            _set_status(df, i, "invalid", "Ticket details could not be checked against the result", stamp)
            continue

        df.at[i, "draw_number"] = int(row["draw_number"])
        df.at[i, "units"] = float(units)
        df.at[i, "winnings"] = float(res.amount)
        _set_status(df, i, "settled", res.detail, stamp)
        settled_idx.append(i)

    df = normalise_ledger(df)
    settled = [_row_dict(df, i) for i in settled_idx]
    if settled:
        log.info("settled %d tickets, won %s in total", len(settled),
                 _dollars(sum(r["winnings"] or 0.0 for r in settled)))
    return df, settled


def settled_rows(ledger: pd.DataFrame | None, ids: list[str]) -> list[dict]:
    """Settled ledger rows whose ticket_id is in ``ids``, in that order, as plain dicts of the
    ledger columns (the shape ``settle_ledger`` returns). Ids that are not in the ledger, or
    whose row is no longer settled, are skipped."""
    df = _ledger_frame(ledger)
    where = {str(tid): i for i, tid in zip(df.index, df["ticket_id"])}
    out = []
    for tid in ids:
        i = where.get(str(tid))
        if i is not None and df.at[i, "status"] == "settled":
            out.append(_row_dict(df, i))
    return out


def ledger_totals(ledger: pd.DataFrame | None) -> dict:
    """Totals over the ledger (every game, so checked 4D tickets from older versions count).

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
