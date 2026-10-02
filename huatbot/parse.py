"""Pure parsers for Singapore Pools pages (HTML string in, plain Python values out).

Result pages are read with the CSS classes the user verified on the live site, so the
parsers do not depend on where things sit on the page. The next draw and prize
structure pages are read from their visible text, because their markup is loose and
the online2 prize pages may be drawn by JavaScript (then the figures are simply not
in the HTML and the prize parsers return None).
"""
from __future__ import annotations

import base64
import binascii
import math
import re
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


def page_text(html: str) -> str:
    """Visible text of a page: scripts, styles and templates removed, whitespace collapsed."""
    soup = _soup(html)
    for tag in soup(["script", "style", "noscript", "template", "head"]):
        tag.decompose()
    return _clean(soup.get_text(" "))


def _first(soup: BeautifulSoup, selector: str):
    return soup.select_one(selector)


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
_WEEKDAY_WORD = r"(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)[a-z]*\.?,?"

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
    s = _clean(text)
    if not s or set(s) <= set("-–—$ "):
        return None
    m = _AMOUNT.search(s)
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
    num_tag = _first(soup, "th.drawNumber") or _first(soup, ".drawNumber")
    number = _draw_number_from_text(_cell_text(num_tag)) if num_tag is not None else None
    if number is None:
        raise ParseError("draw number not found on the page")
    date_tag = _first(soup, "th.drawDate") or _first(soup, ".drawDate")
    if date_tag is None:
        raise ParseError(f"draw date not found on the page for draw {number}")
    return number, parse_draw_date(_cell_text(date_tag))


def parse_toto_result(html: str) -> dict:
    """One TOTO result page -> dict with the toto.csv result columns (see models.TOTO_COLUMNS)."""
    soup = _soup(html)
    draw_number, draw_date = _draw_header(soup)

    numbers = []
    for k in range(1, 7):
        tag = _first(soup, f"td.win{k}")
        text = _cell_text(tag)
        if tag is None or not re.search(r"\d", text):
            raise ParseError(f"winning number {k} not found for draw {draw_number}")
        numbers.append(parse_int(text))
    add_tag = _first(soup, "td.additional")
    if add_tag is None or not re.search(r"\d", _cell_text(add_tag)):
        raise ParseError(f"additional number not found for draw {draw_number}")
    additional = parse_int(_cell_text(add_tag))

    numbers.sort()
    if len(set(numbers)) != 6 or not all(1 <= n <= C.TOTO_MAX_NUMBER for n in numbers):
        raise ParseError(f"winning numbers for draw {draw_number} look wrong: {numbers}")
    if not 1 <= additional <= C.TOTO_MAX_NUMBER or additional in numbers:
        raise ParseError(f"additional number for draw {draw_number} looks wrong: {additional}")

    jackpot = parse_money(_cell_text(_first(soup, "td.jackpotPrize")))
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

    table = _first(soup, "table.tableWinningShares")
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


_FOURD_PLACEHOLDER = re.compile(r"[-\u2013\u2014]+")  # "-", "----" and long dash variants


def _fourd_number(text: str, what: str = "4D number") -> str:
    """A 4D cell as a 4 character string; "" for an empty cell or a dash placeholder.

    Raises ParseError for anything else (text, or more than 4 digits), so a changed layout is
    not stored as wrong numbers.
    """
    s = re.sub(r"\s+", "", text or "")
    if not s or _FOURD_PLACEHOLDER.fullmatch(s):
        return ""
    if not s.isdigit() or len(s) > 4:
        raise ParseError(f"{what} {s!r} is not a 4 digit number")
    return s.zfill(4)


def _fourd_prize_list(body, label: str, draw_number: int) -> list[str]:
    """The 10 starter or consolation numbers of a 4D page, in page order.

    Cells holding a number or a dash placeholder count (a placeholder stays a blank in its
    place); label cells are skipped, and empty spacer cells too when there are more than 10
    cells. Anything other than exactly 10 cells, or 10 with no number at all, raises.
    """
    if body is None:
        raise ParseError(f"{label} numbers not found for 4D draw {draw_number}")
    items: list[tuple[str, bool]] = []  # (value, empty cell)
    for td in body.find_all("td"):
        s = re.sub(r"\s+", "", _cell_text(td))
        if not s:
            items.append(("", True))
        elif _FOURD_PLACEHOLDER.fullmatch(s):
            items.append(("", False))
        elif s.isdigit():
            items.append((_fourd_number(s, f"{label} number"), False))
        # any other text is a label or heading cell
    if len(items) > 10:
        items = [item for item in items if not item[1]]  # drop empty spacer cells
    values = [v for v, _ in items]
    if len(values) != 10 or not any(values):
        found = sum(1 for v in values if v)
        raise ParseError(f"{found} {label} numbers were read for 4D draw {draw_number} "
                         f"({len(values)} cells), expected 10; the layout may have changed")
    return values


def parse_fourd_result(html: str) -> dict:
    """One 4D result page -> dict with the fourd.csv number columns (4 character strings)."""
    soup = _soup(html)
    draw_number, draw_date = _draw_header(soup)
    out: dict = {"draw_number": draw_number, "draw_date": draw_date}
    for key, cls in (("first", "tdFirstPrize"), ("second", "tdSecondPrize"), ("third", "tdThirdPrize")):
        out[key] = _fourd_number(_cell_text(_first(soup, f"td.{cls}")), f"4D {key} prize")
    for key, cls in (("starter", "tbodyStarterPrizes"), ("consolation", "tbodyConsolationPrizes")):
        values = _fourd_prize_list(_first(soup, f"tbody.{cls}"), key, draw_number)
        for i, v in enumerate(values, start=1):
            out[f"{key}_{i}"] = v
    if not any(out[k] for k in out if k not in ("draw_number", "draw_date")):
        raise ParseError(f"no winning numbers found for 4D draw {draw_number}")
    return out


# next draw pages

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


def parse_fourd_next_draw(html: str) -> dict:
    """Next 4D draw: {draw_datetime, raw_text}."""
    text = page_text(html)
    return {"draw_datetime": _next_draw_datetime(text), "raw_text": text}


# prize structure pages (online2, possibly drawn by JavaScript)

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


def _group_fixed(text: str) -> float | None:
    m = _DOLLAR.search(text)
    if not m or "%" in text[: m.start()]:
        return None
    value = parse_money(m.group(0))
    return float(value) if value is not None and 0 < value < 10_000 else None


def parse_toto_prize_structure(html: str) -> dict | None:
    """Prize percentages from the official TOTO prize structure page, or None if not in the HTML.

    Returns {"pool_share_of_sales": 0.54 | None, "group_pool_pct": {1: .38, 2: .08, 3: .055, 4: .03},
    "fixed_prizes": {5: 50.0, 6: 25.0, 7: 10.0} (the groups found), "min_group1": 1000000.0 | None}.
    Table rows that start with a "Group N" cell are read first; loose page text near each
    group label fills any group the tables did not give.
    """
    soup = _soup(html)
    for tag in soup(["script", "style", "noscript", "template", "head"]):
        tag.decompose()
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

    segments = _group_segments(text)
    for g in (1, 2, 3, 4):
        for seg in segments.get(g, []) if g not in pct else []:
            value = _group_pct(seg, strict=True)
            if value is not None:
                pct[g] = value
                break
    for g in (5, 6, 7):
        for seg in segments.get(g, []) if g not in fixed else []:
            value = _group_fixed(seg)
            if value is not None:
                fixed[g] = value
                break
    if min_g1 is None:
        for seg in segments.get(1, []):
            min_g1 = _min_group1(_MINIMUM.search(seg))
            if min_g1 is not None:
                break

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


_TIER_PATTERNS = (
    ("first", re.compile(r"\b(?:1st|first)\b", re.I)),
    ("second", re.compile(r"\b(?:2nd|second)\b", re.I)),
    ("third", re.compile(r"\b(?:3rd|third)\b", re.I)),
    ("starter", re.compile(r"\bstarter", re.I)),
    ("consolation", re.compile(r"\bconsolation", re.I)),
)
_BIG_TIERS = ("first", "second", "third", "starter", "consolation")
_SMALL_TIERS = ("first", "second", "third")
_PERMS = (24, 12, 6, 4)


def _tier_of(label: str) -> str | None:
    for tier, pattern in _TIER_PATTERNS:
        if pattern.search(label):
            return tier
    return None


def _bet_of(text: str) -> str | None:
    """"big" or "small" if the text names exactly one of them (last mention wins)."""
    hits = re.findall(r"\b(big|small)\b", text or "", re.I)
    return hits[-1].lower() if hits else None


def _perm_of(header: str) -> int | None:
    header = re.sub(r"\$\s*[\d,.]+", " ", header)  # "($1 bet)" is not a permutation count
    m = re.search(r"\b(24|12|6|4)\b", header)
    if not m:
        return None
    low = header.lower()
    if "perm" in low or "ibet" in low or header.strip() == m.group(1):
        return int(m.group(1))
    return None


def _complete(table: dict[str, float], tiers: tuple[str, ...]) -> bool:
    """All tiers present and amounts strictly falling from 1st prize down."""
    if not all(t in table for t in tiers):
        return False
    values = [table[t] for t in tiers]
    return all(v > 0 for v in values) and all(a > b for a, b in zip(values, values[1:]))


def _fourd_tables(soup: BeautifulSoup) -> tuple[dict, dict, dict]:
    """Read Big, Small and iBet amounts from <table> elements."""
    straight: dict[str, dict[str, float]] = {"big": {}, "small": {}}
    ibet: dict[str, dict[int, dict[str, float]]] = {"big": {}, "small": {}}
    for table in soup.find_all("table"):
        rows = [[_cell_text(c) for c in tr.find_all(["td", "th"])] for tr in table.find_all("tr")]
        rows = [r for r in rows if r]
        if not rows:
            continue
        # Context: caption, else the nearest text before the table that names Big or Small.
        caption = table.find("caption")
        context = _cell_text(caption) if caption is not None else ""
        if not _bet_of(context):
            prev = table.find_previous(string=re.compile(r"\b(big|small)\b", re.I))
            context = _clean(str(prev)) if prev else ""
        context_bet = _bet_of(context)
        context_ibet = "ibet" in context.lower()

        # Column meaning from the header rows (rows whose first cell is not a prize tier).
        col_bet: dict[int, str] = {}
        col_perm: dict[int, int] = {}
        for row in rows:
            if _tier_of(row[0]):
                continue
            for i, cell in enumerate(row):
                if i == 0:
                    continue
                bet = _bet_of(cell)
                if bet:
                    col_bet[i] = bet
                perm = _perm_of(cell)
                if perm:
                    col_perm[i] = perm
            if "ibet" in " ".join(row).lower():
                context_ibet = True
            if not context_bet:
                context_bet = _bet_of(" ".join(row))

        for row in rows:
            tier = _tier_of(row[0]) if row else None
            if tier is None:
                continue
            money_cells = {i: parse_money(c) for i, c in enumerate(row) if i > 0 and "$" in c}
            money_cells = {i: v for i, v in money_cells.items() if v is not None}
            if not money_cells:
                continue
            if col_perm and context_bet:
                for i, v in money_cells.items():
                    if i in col_perm:
                        ibet[context_bet].setdefault(col_perm[i], {})[tier] = v
            elif col_bet:
                for i, v in money_cells.items():
                    if i in col_bet:
                        straight[col_bet[i]].setdefault(tier, v)
            elif context_bet and not context_ibet:
                straight[context_bet].setdefault(tier, list(money_cells.values())[-1])
    return straight["big"], straight["small"], ibet


_TIER_AMOUNT = re.compile(
    r"\b(1st|first|2nd|second|3rd|third|starter|consolation)\b[^$]{0,60}?(\$\s*\d[\d,]*(?:\.\d+)?)", re.I
)


def _fourd_text(text: str) -> tuple[dict, dict]:
    """Fallback: tier amounts from running text, each assigned to the last Big/Small label before it."""
    labels = [(m.start(), m.group(1).lower(), bool(m.group(0).lower().startswith("ibet")))
              for m in re.finditer(r"(?:\biBet\s+)?\b(big|small)\b", text, re.I)]
    out: dict[str, dict[str, float]] = {"big": {}, "small": {}}
    for m in _TIER_AMOUNT.finditer(text):
        before = [lab for lab in labels if lab[0] < m.start()]
        if not before or before[-1][2]:
            continue  # no label yet, or inside an iBet section
        tier = _tier_of(m.group(1))
        value = parse_money(m.group(2))
        if tier and value is not None:
            out[before[-1][1]].setdefault(tier, value)
    return out["big"], out["small"]


def parse_fourd_prize_structure(html: str) -> dict | None:
    """Big and Small prize per $1 from the official 4D prize structure page, or None if not in the HTML.

    Returns {"big": {tier: amount}, "small": {tier: amount}} and, when the page lists them,
    "ibet": {"big": {24: {tier: amount}, 12: ..., 6: ..., 4: ...}, "small": {...}}.
    """
    soup = _soup(html)
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    big, small, ibet = _fourd_tables(soup)
    if not (_complete(big, _BIG_TIERS) and _complete(small, _SMALL_TIERS)):
        tbig, tsmall = _fourd_text(page_text(str(soup)))
        if not _complete(big, _BIG_TIERS):
            big = tbig
        if not _complete(small, _SMALL_TIERS):
            small = tsmall
    if not (_complete(big, _BIG_TIERS) and _complete(small, _SMALL_TIERS)):
        return None
    out: dict = {
        "big": {t: float(big[t]) for t in _BIG_TIERS},
        "small": {t: float(small[t]) for t in _SMALL_TIERS},
    }
    ibet_clean: dict[str, dict[int, dict[str, float]]] = {}
    for bet, tiers in (("big", _BIG_TIERS), ("small", _SMALL_TIERS)):
        for perm, table in sorted(ibet.get(bet, {}).items(), reverse=True):
            if _complete(table, tiers):
                ibet_clean.setdefault(bet, {})[perm] = {t: float(table[t]) for t in tiers}
    if ibet_clean:
        out["ibet"] = ibet_clean
    return out
