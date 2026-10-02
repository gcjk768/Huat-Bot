"""Formatting helpers for every piece of text the user reads.

House style (SPEC, "Global conventions"): user facing text has no dashes. Negative money is
written "minus $20", ranges "1 to 24", dates "Thu 1 Oct 2026" and times "6.30pm", always in
Singapore time. Telegram messages must contain no dash at all; vault notes may only carry
hyphens in markdown table separator rows, the YAML frontmatter block and [[wikilinks]] (which
hold ISO dates in file names). ``has_prose_dashes`` checks exactly that, and
``remove_dashes`` is the sanitiser for prose that may contain dashes (for example text written
by a model).
"""
from __future__ import annotations

import html
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import constants as C

SG_TZ = ZoneInfo(C.SG_TZ_NAME)

NA = "n/a"

# Every character treated as a dash. The SPEC bans "-", "–" and "—"; the look alikes (hyphen,
# non breaking hyphen, figure dash, horizontal bar, minus sign, small and fullwidth hyphen minus)
# render as dashes too, so they get the same treatment.
DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
_D = "[" + re.escape(DASH_CHARS) + "]"

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# Small value helpers


def _is_missing(x: Any) -> bool:
    """True for None, NaN, NaT and pd.NA (scalars only; containers are never "missing")."""
    if x is None:
        return True
    if isinstance(x, (str, bytes, list, tuple, dict, set)):
        return False
    try:
        res = pd.isna(x)
    except (TypeError, ValueError):
        return False
    return bool(res) if isinstance(res, (bool, np.bool_)) else False


def _decimal(x: Any) -> Decimal | None:
    """Finite Decimal for a number like value, else None."""
    if _is_missing(x) or isinstance(x, (bool, np.bool_)):
        return None
    try:
        if isinstance(x, Decimal):
            d = x
        elif isinstance(x, (int, np.integer)):
            d = Decimal(int(x))  # Decimal does not accept numpy integers directly
        else:
            d = Decimal(repr(float(x)))
    except (TypeError, ValueError, InvalidOperation):
        return None
    return d if d.is_finite() else None


def _round(d: Decimal, digits: int) -> Decimal:
    """Round half up (so $2.50 shows as $3, not banker's $2)."""
    return d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def _signed(body: str, negative: bool) -> str:
    return f"minus {body}" if negative else body


# Numbers and money


def money(x: float | None, cents: bool = False) -> str:
    """Dollars with thousands separators: "$1,234,567"; cents=True gives "$0.66".

    None or NaN gives "n/a". Negative amounts read "minus $20" (never a dash). An amount that
    rounds to zero is "$0", never "minus $0".
    """
    d = _decimal(x)
    if d is None:
        return NA
    digits = 2 if cents else 0
    r = _round(d, digits)
    return _signed(f"${abs(r):,.{digits}f}", r < 0)


def per_dollar(x: float | None) -> str:
    """A return per $1 staked, e.g. 0.659 -> "$0.66"."""
    return money(x, cents=True)


def pct(x: float | None, digits: int = 1) -> str:
    """A fraction as a percentage: 0.054 -> "5.4%"; negative -> "minus 1.2%"."""
    d = _decimal(x)
    if d is None:
        return NA
    r = _round(d * 100, digits)
    return _signed(f"{abs(r):,.{digits}f}%", r < 0)


def fmt_num(x: float | None, digits: int = 0) -> str:
    """A plain number with thousands separators: 13983816 -> "13,983,816"; negative -> "minus 5"."""
    d = _decimal(x)
    if d is None:
        return NA
    r = _round(d, digits)
    return _signed(f"{abs(r):,.{digits}f}", r < 0)


def plural(n: int | float, singular: str, plural_form: str | None = None) -> str:
    """ "1 draw", "3 draws" (counts get thousands separators)."""
    word = singular if n == 1 else (plural_form or singular + "s")
    return f"{fmt_num(n)} {word}"


# Dates and times (always shown in Singapore time)


def _to_sg(dt: datetime) -> datetime:
    """Aware datetimes are converted to Singapore time; naive ones are taken as Singapore time."""
    return dt.astimezone(SG_TZ) if dt.tzinfo is not None else dt


def _as_date(d: Any) -> date | None:
    if _is_missing(d):
        return None
    if isinstance(d, datetime):  # includes pd.Timestamp
        return _to_sg(d).date()
    if isinstance(d, date):
        return d
    if isinstance(d, np.datetime64):
        return pd.Timestamp(d).date()
    if isinstance(d, str):
        s = d.strip()
        if not s:
            return None
        try:
            parsed = datetime.fromisoformat(s)
        except ValueError:
            return None
        return _to_sg(parsed).date()
    return None


def fmt_date(d: date | datetime | pd.Timestamp | str | None) -> str:
    """ "Thu 1 Oct 2026". Accepts date, datetime, pd.Timestamp or an ISO string; missing -> "n/a"."""
    dd = _as_date(d)
    if dd is None:
        return NA
    return f"{_WEEKDAYS[dd.weekday()]} {dd.day} {_MONTHS[dd.month - 1]} {dd.year}"


def fmt_time(dt: datetime | None) -> str:
    """ "6.30pm": no leading zero on the hour, minutes always two digits, lower case am/pm."""
    if _is_missing(dt) or not isinstance(dt, datetime):
        return NA
    t = _to_sg(dt)
    hour = t.hour % 12 or 12
    return f"{hour}.{t.minute:02d}{'am' if t.hour < 12 else 'pm'}"


def fmt_datetime(dt: datetime | None) -> str:
    """ "Mon 5 Oct 2026, 6.30pm" in Singapore time; missing -> "n/a"."""
    if _is_missing(dt) or not isinstance(dt, datetime):
        return NA
    t = _to_sg(dt)
    return f"{fmt_date(t)}, {fmt_time(t)}"


def toto_nums(nums: Iterable[Any] | str | None) -> str:
    """TOTO numbers ascending, single spaced: "3 11 19 27 38 45". Accepts ints, floats or strings."""
    if nums is None:
        return ""
    if isinstance(nums, str):
        items: list[Any] = re.findall(r"\d+", nums)
    else:
        items = [n for n in nums if not _is_missing(n)]
    return " ".join(str(n) for n in sorted(int(float(n)) for n in items))


# Tables

_NUMERIC_CELL = re.compile(r"^(?:minus )?\$?\d[\d,]*(?:\.\d+)?(?:%|x)?$")


def _plain_value(v: Any) -> str:
    """One table cell as text. Numbers keep their digits (no grouping); negatives say "minus"."""
    if _is_missing(v):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return _signed(str(abs(int(v))), int(v) < 0)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isinf(f):
            return NA
        if f.is_integer():
            return _signed(str(abs(int(f))), f < 0)
        body = f"{abs(round(f, 4)):f}".rstrip("0").rstrip(".")
        return _signed(body, f < 0)
    return " ".join(str(v).split())  # one line, single spaced


def _width(s: str) -> int:
    """Display width in a monospace font (wide East Asian characters take two cells)."""
    total = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _grid(headers: Sequence[Any], rows: Iterable[Sequence[Any]], cell) -> tuple[list[str], list[list[str]]]:
    """Headers and rows as strings, every row padded to the same number of columns."""
    head = [cell(h) for h in headers]
    body = [[cell(v) for v in row] for row in rows]
    ncols = max([len(head)] + [len(r) for r in body])
    head += [""] * (ncols - len(head))
    body = [r + [""] * (ncols - len(r)) for r in body]
    return head, body


def _alignments(align: str | None, body: list[list[str]], ncols: int) -> str:
    """Resolve the align string. None means: right align columns whose cells are all numbers."""
    if align is None:
        out = []
        for c in range(ncols):
            cells = [r[c] for r in body if r[c] not in ("", NA)]
            out.append("r" if cells and all(_NUMERIC_CELL.match(x) for x in cells) else "l")
        return "".join(out)
    align = align.lower()
    bad = set(align) - set("lrc")
    if bad:
        raise ValueError(f"align may only contain l, r or c, got {align!r}")
    return (align + "l" * ncols)[:ncols]


def _pad(s: str, width: int, how: str) -> str:
    gap = max(0, width - _width(s))
    if how == "r":
        return " " * gap + s
    if how == "c":
        return " " * (gap // 2) + s + " " * (gap - gap // 2)
    return s + " " * gap


def _pre_cell(v: Any) -> str:
    """Cell text for a Telegram <pre> table: no dashes at all (a lone "-" means empty)."""
    s = _plain_value(v)
    if s.strip(DASH_CHARS + " ") == "":
        return ""
    return remove_dashes(s).strip()


def pre_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    """Monospace table for a Telegram <pre> block: columns padded with spaces, no rule lines.

    ``align`` holds one letter per column: "l" left, "r" right ("c" centre also works). None
    right aligns columns that hold only numbers. The result is plain text with no dashes; escape
    it (``html_escape``) when wrapping in <pre>, or use ``pre_block``.
    """
    head, body = _grid(headers, rows, _pre_cell)
    ncols = len(head)
    how = _alignments(align, body, ncols)
    widths = [max(_width(r[c]) for r in [head] + body) for c in range(ncols)]
    lines = ["  ".join(_pad(r[c], widths[c], how[c]) for c in range(ncols)).rstrip() for r in [head] + body]
    return "\n".join(lines)


def pre_block(headers: list[str], rows: list[list], align: str | None = None) -> str:
    """``pre_table`` escaped and wrapped in <pre> tags, ready for a Telegram HTML message."""
    return f"<pre>{html_escape(pre_table(headers, rows, align))}</pre>"


def _md_cell(v: Any) -> str:
    return _plain_value(v).replace("|", "\\|")


def md_table(headers: list[Any], rows: list[list], align: str | None = None) -> str:
    """Markdown pipe table for Obsidian, e.g. "| Group | Share |" then "| --- | ---: |".

    Cells are single line, pipes are escaped (so [[note\\|alias]] links work inside tables).
    ``align`` works as in ``pre_table``. With no rows the header and separator are still emitted.
    """
    head, body = _grid(headers, rows, _md_cell)
    ncols = len(head)
    how = _alignments(align, body, ncols)
    widths = [max([3] + [_width(r[c]) for r in [head] + body]) for c in range(ncols)]
    rules = {"l": lambda w: "-" * w, "r": lambda w: "-" * (w - 1) + ":", "c": lambda w: ":" + "-" * (w - 2) + ":"}

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    out = [line([_pad(head[c], widths[c], how[c]) for c in range(ncols)])]
    out.append(line([rules[how[c]](widths[c]) for c in range(ncols)]))
    out += [line([_pad(r[c], widths[c], how[c]) for c in range(ncols)]) for r in body]
    return "\n".join(out)


def html_escape(s: Any) -> str:
    """Escape &, < and > for Telegram HTML (quotes are left alone, they are fine in text)."""
    if s is None:
        return ""
    return html.escape(str(s), quote=False)


# Dashes

# A markdown table separator row: pipes, colons, spaces and at least one hyphen.
_TABLE_SEP = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*(?::?-+:?)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# Spans inside a line that may carry hyphens: inline code, wikilinks and embeds, link targets,
# autolinks and bare URLs (file names and URLs are not prose).
_INLINE_ALLOWED = re.compile(
    r"(`+)[^\n]*?\1"
    r"|!?\[\[[^\]\n]*\]\]"
    r"|\]\(<[^>\n]*>\)"
    r"|\]\([^)\s]*\)"
    r"|<https?://[^>\s]+>"
    r"|https?://[^\s<>()\[\]]+"
)
# Comments that never render: HTML comments and Obsidian %% comments (may span lines).
_COMMENTS = re.compile(r"<!--.*?-->|%%.*?%%", re.S)


def _frontmatter_end(text: str) -> int:
    """Index just past a leading YAML frontmatter block (fence lines included), or 0 if none."""
    start = 1 if text.startswith("\ufeff") else 0
    first_nl = text.find("\n", start)
    if first_nl < 0 or text[start:first_nl].rstrip() != "---":
        return 0
    pos = first_nl + 1
    while pos <= len(text):
        nl = text.find("\n", pos)
        end = len(text) if nl < 0 else nl
        if text[pos:end].rstrip() in ("---", "..."):
            return end
        if nl < 0:
            break
        pos = nl + 1
    return 0  # never closed: not frontmatter


def _allowed_spans(text: str) -> list[tuple[int, int]]:
    """Merged (start, end) spans of markdown where hyphens are allowed (not prose)."""
    spans: list[tuple[int, int]] = []
    fm = _frontmatter_end(text)
    if fm:
        spans.append((0, fm))
    spans += [m.span() for m in _COMMENTS.finditer(text)]

    pos, fence = fm, None  # fence = (char, length) while inside a fenced code block
    if fm and fm < len(text) and text[fm] == "\n":
        pos = fm + 1
    while pos < len(text):
        nl = text.find("\n", pos)
        end = len(text) if nl < 0 else nl
        line = text[pos:end]
        m = _FENCE.match(line)
        if fence is not None:
            spans.append((pos, end))
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not line[m.end():].strip():
                fence = None
        elif m:
            fence = (m.group(1)[0], len(m.group(1)))
            spans.append((pos, end))
        elif "|" in line and _TABLE_SEP.match(line):
            spans.append((pos, end))
        else:
            spans += [(pos + a, pos + b) for a, b in (mm.span() for mm in _INLINE_ALLOWED.finditer(line))]
        pos = end + 1

    merged: list[tuple[int, int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def contains_dash(text: str | None) -> bool:
    """True if any dash character appears anywhere (the Telegram rule)."""
    return bool(text) and any(ch in DASH_CHARS for ch in text)


def find_prose_dashes(text: str | None) -> list[tuple[int, str]]:
    """(line number from 1, line) for every line with a dash in prose (see ``has_prose_dashes``)."""
    if not text:
        return []
    spans = _allowed_spans(text)
    hits: list[tuple[int, str]] = []
    seen: set[int] = set()
    for m in re.finditer(_D, text):
        i = m.start()
        if any(a <= i < b for a, b in spans):
            continue
        line_no = text.count("\n", 0, i) + 1
        if line_no not in seen:
            seen.add(line_no)
            hits.append((line_no, text.split("\n")[line_no - 1]))
    return hits


def has_prose_dashes(text: str | None) -> bool:
    """True if a dash (-, –, — or a look alike) appears in prose.

    Ignored: markdown table separator rows ("| --- | ---: |"), a leading YAML frontmatter
    block, [[wikilinks]] and embeds, inline code, fenced code blocks, markdown link targets,
    URLs, and HTML / Obsidian comments.
    """
    return bool(find_prose_dashes(text))


# remove_dashes rules, applied in this order.
_ISO_DATE = re.compile(rf"(?<![\w.])((?:19|20)\d{{2}}){_D}(\d{{2}}){_D}(\d{{2}})(?![\w])")
_YEAR_MONTH = re.compile(rf"(?<![\w.])((?:19|20)\d{{2}}){_D}(0[1-9]|1[0-2])(?![\w]|{_D})")
_TABLE_SEP_LINE = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*(?::?-+:?)?[ \t]*(?:\n|$)", re.M)
_RULE_LINE = re.compile(rf"^[ \t]*{_D}{{3,}}[ \t]*$", re.M)  # markdown horizontal rule
_BULLET = re.compile(rf"^([ \t]*){_D}+[ \t]+", re.M)  # "- item"
_NEGATIVE = re.compile(rf"(^|[\s(\[:=,/]){_D}(?=\$?\d)", re.M)  # "-20", "-$20"
_LEAD = re.compile(rf"^([ \t]*){_D}+(?=\S)", re.M)  # "-word" at a line start
_RANGE = re.compile(rf"(\d)[ \t]*{_D}+[ \t]*(?=\$?\d)")  # "1-24", "1 - 24", "$1-$2"
_SPACED = re.compile(rf"[ \t]+{_D}+[ \t]+")  # " - ", " — "
_EM = re.compile(r"[ \t]*[\u2014\u2015]+[ \t]*")  # unspaced em dash: a clause break
_TRAILING = re.compile(rf"[ \t]*{_D}+[ \t]*$", re.M)
_ANY = re.compile(rf"[ \t]*{_D}+[ \t]*")
_COMMA = "\x01"  # marks commas made by the rules, so only those get tidied


def _iso_to_words(m: re.Match) -> str:
    try:
        return fmt_date(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    except ValueError:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"


def _year_month_to_words(m: re.Match) -> str:
    return f"{_MONTHS[int(m.group(2)) - 1]} {m.group(1)}"


def _strip_dashes(text: str, markdown: bool) -> str:
    """The prose rules. Allowed markdown spans are already masked out when ``markdown`` is True."""
    text = _ISO_DATE.sub(_iso_to_words, text)
    text = _YEAR_MONTH.sub(_year_month_to_words, text)
    if not markdown:
        text = _TABLE_SEP_LINE.sub("", text)
    text = _RULE_LINE.sub("", text)
    text = _BULLET.sub(r"\1* " if markdown else "\\1\u2022 ", text)
    text = _NEGATIVE.sub(r"\1minus ", text)
    text = _LEAD.sub(r"\1", text)
    text = _RANGE.sub(r"\1 to ", text)
    text = _SPACED.sub(_COMMA + " ", text)
    text = _EM.sub(_COMMA + " ", text)
    text = _TRAILING.sub("", text)
    text = _ANY.sub(" ", text)
    # Tidy the commas made above: merge with neighbouring punctuation, drop them at line ends.
    text = re.sub(rf",[ \t]*{_COMMA}|{_COMMA}(?:[ \t]*{_COMMA})*[ \t]*,", ",", text)
    text = re.sub(rf"{_COMMA}(?:[ \t]*{_COMMA})+", _COMMA, text)
    text = re.sub(rf"{_COMMA}[ \t]*([.;:!?)])", r"\1", text)
    text = re.sub(rf"^([ \t]*){_COMMA}[ \t]*", r"\1", text, flags=re.M)
    text = re.sub(rf"[ \t]*{_COMMA}[ \t]*$", "", text, flags=re.M)
    return text.replace(_COMMA, ",")


def remove_dashes(text: str | None, markdown: bool = False) -> str:
    """Prose sanitiser: returns the text with every dash rewritten.

    " — ", " – " and " - " become ", "; "-20" becomes "minus 20"; "1-24" becomes "1 to 24";
    ISO dates become "Thu 1 Oct 2026"; "- item" bullets become "• item"; any other dash
    becomes a space. The result never contains a dash.

    With ``markdown=True`` the spans ``has_prose_dashes`` allows (frontmatter, table separator
    rows, wikilinks, code, URLs, comments) are kept as they are and bullets become "* item",
    so a whole note can be cleaned without breaking it.
    """
    if not text:
        return ""
    text = str(text).replace("\x00", "").replace(_COMMA, "")
    if not markdown:
        return _strip_dashes(text, markdown=False)
    stash: list[str] = []
    parts, last = [], 0
    for a, b in _allowed_spans(text):
        parts.append(text[last:a])
        parts.append(f"\x00{len(stash)}\x00")
        stash.append(text[a:b])
        last = b
    parts.append(text[last:])
    cleaned = _strip_dashes("".join(parts), markdown=True)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], cleaned)
