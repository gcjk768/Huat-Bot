"""Pure parsers for the Singapore Pools TOTO pages (HTML string in, plain Python values out).

Result pages are read with the CSS classes the user verified on the live site, so the
parsers do not depend on where things sit on the page. The next draw and prize
structure pages are read from their visible text, because their markup is loose and
the online2 prize page may be drawn by JavaScript (then the figures are simply not
in the HTML and the prize parser returns None).
"""
from __future__ import annotations

import base64
import binascii
import math
import re
from collections.abc import Callable
from datetime import date, datetime, time
from urllib.parse import parse_qs, unquote
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from . import constants as C

SG = ZoneInfo(C.SG_TZ_NAME)


class ParseError(ValueError):
    """The page did not contain what we expected."""


# small helpers


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "lxml")


def _clean(text: str | None) -> str:
    """Collapse all whitespace (including non breaking spaces) to single spaces."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def _cell_text(tag) -> str:
    return _clean(tag.get_text(" ")) if tag is not None else ""


def _visible_soup(html: str) -> BeautifulSoup:
    """The page with everything a reader never sees (scripts, styles, templates, head) removed."""
    soup = _soup(html)
    for tag in soup(["script", "style", "noscript", "template", "head"]):
        tag.decompose()
    return soup


def page_text(html: str) -> str:
    """Visible text of a page: scripts, styles and templates removed, whitespace collapsed."""
    return _clean(_visible_soup(html).get_text(" "))


# draw numbers and the sppl query value


def sppl(draw_number: int) -> str:
    """Query value for a result page: base64 of ``DrawNumber=NNNN``."""
    return base64.b64encode(f"DrawNumber={int(draw_number)}".encode("ascii")).decode("ascii")


def draw_number_from_sppl(value: str) -> int:
    """Draw number inside an sppl value.

    Accepts the bare value, ``sppl=...``, a whole query string or URL, and URL quoted forms.
    """
    if value is None:
        raise ParseError("empty sppl value")
    text = unquote(str(value).strip())
    if "sppl=" in text:
        query = text.split("?", 1)[-1]
        found = parse_qs(query, keep_blank_values=True).get("sppl")
        text = found[0] if found else text.split("sppl=", 1)[1].split("&", 1)[0]
    text = text.strip()
    if not text:
        raise ParseError("empty sppl value")
    padded = text + "=" * (-len(text) % 4)
    try:
        decoded = base64.b64decode(padded, validate=False).decode("ascii", errors="ignore")
    except (binascii.Error, ValueError) as exc:
        raise ParseError(f"sppl value {value!r} is not base64") from exc
    m = re.search(r"DrawNumber\s*=\s*(\d+)", decoded, re.I)
    if not m:
        raise ParseError(f"sppl value {value!r} does not hold a draw number")
    return int(m.group(1))


def _draw_number_from_text(text: str) -> int | None:
    m = re.search(r"(\d[\d,]*)", text or "")
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


# dates, money and integers

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_WORD = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"

# "Thu, 01 Oct 2026", "1 October 2026", "01-Oct-2026", "1st Oct 2026"
_DMY_NAMED = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-/.,]*{_MONTH_WORD}[\s\-/.,]*(\d{{4}})\b", re.I)
# "Oct 1, 2026"
_MDY_NAMED = re.compile(rf"\b{_MONTH_WORD}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def _make_date(y: int, m: int, d: int, text: str) -> date:
    try:
        return date(y, m, d)
    except ValueError as exc:
        raise ParseError(f"not a real date: {text!r}") from exc


def _search_date(text: str) -> tuple[date, int] | None:
    """First date in ``text`` and the index just after it, or None."""
    candidates = []
    m = _DMY_NAMED.search(text)
    if m:
        candidates.append((m.start(), m.end(), int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))))
    m = _MDY_NAMED.search(text)
    if m:
        candidates.append((m.start(), m.end(), int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))))
    m = _ISO.search(text)
    if m:
        candidates.append((m.start(), m.end(), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    m = _DMY_NUMERIC.search(text)
    if m:
        candidates.append((m.start(), m.end(), int(m.group(3)), int(m.group(2)), int(m.group(1))))
    if not candidates:
        return None
    start, end, y, mo, d = min(candidates)  # the earliest one in the text
    return _make_date(y, mo, d, text), end


def parse_draw_date(text: str) -> date:
    """Draw date from text like "Thu, 01 Oct 2026" (also "1 October 2026", "2026-10-01", "01/10/2026")."""
    found = _search_date(_clean(text))
    if found is None:
        raise ParseError(f"no date found in {text!r}")
    return found[0]


_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(million|mil|m\b|billion|bn\b)?", re.I)


def parse_money(text: str | None) -> float | None:
    """Dollar amount: "$1,234,567" -> 1234567.0, "$1.2 million" -> 1200000.0; "-", "" or None -> None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return None if (isinstance(text, float) and math.isnan(text)) else float(text)
    m = _AMOUNT.search(_clean(text))  # "-", "$" or words have no digit, so no match
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in ("million", "mil", "m"):
        value *= 1_000_000
    elif unit in ("billion", "bn"):
        value *= 1_000_000_000
    return value


def parse_int(text: str | None) -> int:
    """Whole number: "1,234" -> 1234; "-", "" or None -> 0."""
    if text is None:
        return 0
    m = re.search(r"\d[\d,]*", _clean(text))
    return int(m.group(0).replace(",", "")) if m else 0


# draw lists


def parse_draw_list(html: str) -> list[tuple[int, date | None]]:
    """(draw number, draw date) for every <option> of a draw list page, newest first.

    The option value is normally the draw number; if it is not, the draw number is
    decoded from the ``querystring`` attribute (``sppl=...``) or the value itself.
    """
    soup = _soup(html)
    found: dict[int, date | None] = {}
    for opt in soup.find_all("option"):
        value = (opt.get("value") or "").strip()
        number = int(value) if value.isdigit() else None
        if number is None:
            for candidate in (opt.get("querystring"), value):
                if not candidate:
                    continue
                try:
                    number = draw_number_from_sppl(candidate)
                    break
                except ParseError:
                    continue
        if number is None:
            continue
        try:
            when = parse_draw_date(opt.get_text(" "))
        except ParseError:
            when = None
        if found.get(number) is None:
            found[number] = when
    return sorted(found.items(), key=lambda item: item[0], reverse=True)


# result pages


def _draw_header(soup: BeautifulSoup) -> tuple[int, date]:
    num_tag = soup.select_one("th.drawNumber") or soup.select_one(".drawNumber")
    number = _draw_number_from_text(_cell_text(num_tag)) if num_tag is not None else None
    if number is None:
        raise ParseError("draw number not found on the page")
    date_tag = soup.select_one("th.drawDate") or soup.select_one(".drawDate")
    if date_tag is None:
        raise ParseError(f"draw date not found on the page for draw {number}")
    return number, parse_draw_date(_cell_text(date_tag))


def parse_toto_result(html: str) -> dict:
    """One TOTO result page -> dict with the toto.csv result columns (see models.TOTO_COLUMNS)."""
    soup = _soup(html)
    draw_number, draw_date = _draw_header(soup)

    numbers = []
    for k in range(1, 7):
        tag = soup.select_one(f"td.win{k}")
        text = _cell_text(tag)
        if tag is None or not re.search(r"\d", text):
            raise ParseError(f"winning number {k} not found for draw {draw_number}")
        numbers.append(parse_int(text))
    add_tag = soup.select_one("td.additional")
    if add_tag is None or not re.search(r"\d", _cell_text(add_tag)):
        raise ParseError(f"additional number not found for draw {draw_number}")
    additional = parse_int(_cell_text(add_tag))

    numbers.sort()
    if len(set(numbers)) != 6 or not all(1 <= n <= C.TOTO_MAX_NUMBER for n in numbers):
        raise ParseError(f"winning numbers for draw {draw_number} look wrong: {numbers}")
    if not 1 <= additional <= C.TOTO_MAX_NUMBER or additional in numbers:
        raise ParseError(f"additional number for draw {draw_number} looks wrong: {additional}")

    jackpot = parse_money(_cell_text(soup.select_one("td.jackpotPrize")))
    out: dict = {
        "draw_number": draw_number,
        "draw_date": draw_date,
        **{f"n{i + 1}": n for i, n in enumerate(numbers)},
        "additional": additional,
        "jackpot": float("nan") if jackpot is None else float(jackpot),
    }
    for g in range(1, 8):
        out[f"g{g}_share"] = float("nan")
        out[f"g{g}_winners"] = 0

    table = soup.select_one("table.tableWinningShares")
    groups_read: set[int] = set()
    if table is not None:
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])  # the group label may be a th cell
            if len(cells) < 3:
                continue
            m = re.search(r"Group\s*(\d)", _cell_text(cells[0]), re.I)
            if not m or not 1 <= int(m.group(1)) <= 7:
                continue  # the header row ("Prize Group", "Share Amount", ...)
            g = int(m.group(1))
            share = parse_money(_cell_text(cells[1]))
            out[f"g{g}_share"] = float("nan") if share is None else float(share)
            out[f"g{g}_winners"] = parse_int(_cell_text(cells[2]))
            groups_read.add(g)
    # A single "-" row is a real result (Group 1 with no winner), but every draw has Group 7
    # winners, so a missing table, missing group rows or an all "-" table means the shares are
    # not published yet (or the layout changed). Raising keeps the draw out of toto.csv, so it
    # is fetched again on the next run instead of being stored as a draw with no winners.
    if len(groups_read) < 7 or out["g7_winners"] <= 0:
        raise ParseError(f"winning shares table for draw {draw_number} is missing or not published yet")
    return out


# next draw page

_TIME = re.compile(r"\b(\d{1,2})(?:\s*[.:]\s*(\d{2}))?\s*([ap])\.?\s*m\b\.?", re.I)


def _parse_time(text: str) -> time | None:
    m = _TIME.search(text)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        return None
    if m.group(3).lower() == "p" and hour != 12:
        hour += 12
    elif m.group(3).lower() == "a" and hour == 12:
        hour = 0
    return time(hour, minute)


def _next_draw_datetime(text: str) -> datetime | None:
    """Draw date and time from next draw text, preferring what follows a "Next Draw" label."""
    label = re.search(r"next\s+draw", text, re.I)
    scopes = [text[label.end():], text] if label else [text]
    for scope in scopes:
        try:
            found = _search_date(scope)
        except ParseError:  # an impossible date such as "31 Feb": treat as no date
            found = None
        if found is None:
            continue
        d, end = found
        when = _parse_time(scope[end:end + 40]) or C.DRAW_TIME
        return datetime.combine(d, when, tzinfo=SG)
    return None


def _draw_type_hint(text: str) -> str | None:
    low = text.lower()
    if re.search(r"hong\s*bao", low):
        return "hongbao"
    if "cascade" in low:
        return "cascade"
    if re.search(r"\bspecial\b", low):
        return "special"
    return None


_DOLLAR = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|mil|m)\b)?", re.I)


def parse_toto_next_draw(html: str) -> dict:
    """Next TOTO draw: {draw_datetime, jackpot_estimate, draw_type_hint, raw_text}."""
    text = page_text(html)
    jackpot = None
    label = re.search(r"jackpot", text, re.I)
    for scope in ([text[label.end():]] if label else []) + [text]:
        m = _DOLLAR.search(scope)
        if m:
            jackpot = parse_money(m.group(0))
            break
    return {
        "draw_datetime": _next_draw_datetime(text),
        "jackpot_estimate": jackpot,
        "draw_type_hint": _draw_type_hint(text),
        "raw_text": text,
    }


# prize structure page (online2, possibly drawn by JavaScript)

_GROUP_LABEL = re.compile(r"\bGroup\s*([1-7])\b", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _group_segments(text: str) -> dict[int, list[str]]:
    """Text following each "Group N" label up to the next group label (capped at 300 chars)."""
    marks = list(_GROUP_LABEL.finditer(text))
    out: dict[int, list[str]] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.setdefault(int(m.group(1)), []).append(text[m.end():min(end, m.end() + 300)])
    return out


_OF_SALES = re.compile(r"\s*of\s+(?:the\s+)?(?:total\s+|gross\s+)?(?:ticket\s+)?(?:sales|turnover)", re.I)
_MINIMUM = re.compile(r"(?:minimum|min\.|at least|guaranteed)[^$%]{0,30}(" + _DOLLAR.pattern + ")", re.I)
MIN_GROUP1_RANGE = (100_000.0, 50_000_000.0)  # a Group 1 minimum outside this is a misread


def _min_group1(match: re.Match | None) -> float | None:
    """The Group 1 minimum from a _MINIMUM match, or None when it is missing or implausible."""
    value = parse_money(match.group(1)) if match else None
    if value is None or not MIN_GROUP1_RANGE[0] <= value <= MIN_GROUP1_RANGE[1]:
        return None
    return float(value)


def _group_pct(text: str, strict: bool) -> float | None:
    """First pool percentage in text that belongs to a group (not the "54% of sales" figure).

    ``strict`` (used for loose page text) also rejects a percentage that comes after a dollar
    amount, which is how a stray "Group 1 $5,000,000" banner would otherwise borrow a figure.
    """
    for m in _PERCENT.finditer(text):
        if _OF_SALES.match(text, m.end()):
            continue
        if strict and "$" in text[: m.start()]:
            return None
        value = float(m.group(1))
        return round(value / 100, 6) if 0 < value < 100 else None
    return None


def _strict_group_pct(text: str) -> float | None:
    return _group_pct(text, strict=True)


def _group_fixed(text: str) -> float | None:
    m = _DOLLAR.search(text)
    if not m or "%" in text[: m.start()]:
        return None
    value = parse_money(m.group(0))
    return float(value) if value is not None and 0 < value < 10_000 else None


def _first_value(texts: list[str], read: Callable[[str], float | None]) -> float | None:
    """The first value ``read`` finds in ``texts``, or None."""
    return next((v for v in map(read, texts) if v is not None), None)


def parse_toto_prize_structure(html: str) -> dict | None:
    """Prize percentages from the official TOTO prize structure page, or None if not in the HTML.

    Returns {"pool_share_of_sales": 0.54 | None, "group_pool_pct": {1: .38, 2: .08, 3: .055, 4: .03},
    "fixed_prizes": {5: 50.0, 6: 25.0, 7: 10.0} (the groups found), "min_group1": 1000000.0 | None}.
    Table rows that start with a "Group N" cell are read first; loose page text near each
    group label fills any group the tables did not give.
    """
    soup = _visible_soup(html)
    text = _clean(soup.get_text(" "))
    if not text:
        return None

    pct: dict[int, float] = {}
    fixed: dict[int, float] = {}
    min_g1 = None
    for tr in soup.find_all("tr"):
        cells = [_cell_text(c) for c in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        m = re.match(r"(?:prize\s+)?group\s*([1-7])\b", cells[0], re.I)
        if not m:
            continue
        g, rest = int(m.group(1)), " ".join(cells[1:])
        if g <= 4 and g not in pct:
            value = _group_pct(rest, strict=False)
            if value is not None:
                pct[g] = value
            if g == 1 and min_g1 is None:
                min_g1 = _min_group1(_MINIMUM.search(rest))
        elif g >= 5 and g not in fixed:
            value = _group_fixed(rest)
            if value is not None:
                fixed[g] = value

    # Loose text near each group label fills what the tables did not give.
    segments = _group_segments(text)
    for g in range(1, 8):
        found, read = (pct, _strict_group_pct) if g <= 4 else (fixed, _group_fixed)
        if g not in found:
            value = _first_value(segments.get(g, []), read)
            if value is not None:
                found[g] = value
    if min_g1 is None:
        min_g1 = _first_value(segments.get(1, []), lambda seg: _min_group1(_MINIMUM.search(seg)))

    if len(pct) < 4 or sum(pct.values()) >= 1 or pct[1] <= max(pct[2], pct[3], pct[4]):
        return None  # incomplete or implausible (Group 1 always takes the largest share)
    # Fixed prizes must fall from Group 5 to Group 7; anything else is a misread.
    ordered = [fixed[g] for g in (5, 6, 7) if g in fixed]
    if ordered != sorted(ordered, reverse=True):
        fixed = {}

    share = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%" + _OF_SALES.pattern, text, re.I)
    if m and 0 < float(m.group(1)) < 100:
        share = round(float(m.group(1)) / 100, 6)

    return {"pool_share_of_sales": share, "group_pool_pct": pct, "fixed_prizes": fixed, "min_group1": min_g1}
