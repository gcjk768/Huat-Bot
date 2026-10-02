"""The full markdown report and the two Telegram messages, built from a ``models.Context``.

Every figure shown here was computed elsewhere in Python (sales, buy signal, jackpot outlook,
ledger); this module only lays it out. House style:

* Telegram messages follow the fleet card style: an emoji + <b>TITLE</b> · subtitle header (one
  fixed emoji per section in ``SECTION_TITLES``), one emoji led line per detail, a divider
  between sections and background detail last in an expandable blockquote. They contain no
  dash characters, escape all text, and stay within ``constants.TELEGRAM_MAX_CHARS``. When a message would
  be too long, detail is dropped in whole blocks so a tag is never cut in half.
* The markdown report has no dashes in prose (tables, frontmatter and [[wikilinks]] may carry
  hyphens) and uses the section headings in ``SECTION_HEADINGS``, in that order.

The bot never suggests numbers: every draw is independent. It reports the latest result, checks
the user's tickets, and says what the next draws are likely to do with the jackpot.

A few view helpers live here because ``notes`` shares them: ``latest_row``, ``next_draw``, the
note names and links, and the markdown blocks for a draw result and the jackpot outlook.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from . import constants as C
from . import tickets as ticket_ledger
from .models import Context, JackpotHistory, JackpotOutlook, PrizeRules
from .outlook import chance_won_by_cascade
from .store import no_winner_streak, toto_numbers
from .textfmt import (
    contains_dash,
    fmt_date,
    fmt_datetime,
    fmt_num,
    fmt_time,
    html_escape,
    md_table,
    money,
    pct,
    per_dollar,
    plural,
    pre_block,
    remove_dashes,
    toto_nums,
)
from .vault import log_note_path

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)
LIMIT = C.TELEGRAM_MAX_CHARS

SECTION_HEADINGS = (
    "## 1. Next draw and the next big prize",
    "## 2. Latest result and my ticket check",
    "## 3. Jackpot history",
    "## 4. Odds note",
)

DRAW_FOLDER = "Draws/TOTO"  # one note per draw, in the bot folder
DRAW_TYPE_NAMES = {"normal": "Normal", "cascade": "Cascade", "hongbao": "Hongbao", "special": "Special"}
EV_PARTS = (
    ("g1", "Group 1"),
    ("g2", "Group 2"),
    ("g3", "Group 3"),
    ("g4", "Group 4"),
    ("fixed", "Groups 5 to 7 (fixed prizes)"),
    ("cascade", "Cascaded jackpot"),
)

# Telegram: how many detail levels the message builders know (0 = everything). Each step drops
# more: 1 the losing ticket lines and the sales estimate; 2 winning tickets beyond the first 10
# and the history line; 3 every ticket line and the buy signal reason; 4 the return breakdown,
# the projection table and the commentary.
_TG_LEVELS = 5
_TG_MORE = "<i>More detail is in the report note in the vault.</i>"
DIVIDER = "━━━━━━━━━━━━━━━━"
# One fixed emoji per Telegram section title.
SECTION_TITLES = {
    "result": "🎱", "winner": "🎉", "tickets": "🎫", "next": "🔮", "big": "💎", "more": "📜",
}
SIGNAL_MARKERS = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}


def title(section: str, name: str, subtitle: str = "") -> str:
    """ "🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct": a section header in the fleet card style."""
    head = f"{SECTION_TITLES[section]} <b>{_h(name.upper())}</b>"
    return f"{head} · {_h(subtitle)}" if subtitle else head
_TG_WINNERS_CAP = 10


# Small value helpers


def clean_text(text: Any) -> str:
    """Prose from another module, made dash free and single line (None -> "")."""
    if text is None:
        return ""
    return " ".join(remove_dashes(str(text)).split())


def sentence(text: Any) -> str:
    """Prose from another module as a sentence: capital first letter, one full stop."""
    s = clean_text(text).rstrip(" .")
    return f"{s[:1].upper()}{s[1:]}." if s else ""


def _h(text: Any) -> str:
    """Prose for Telegram: dash free and HTML escaped."""
    return html_escape(clean_text(text))


def as_float(x: Any) -> float | None:
    """A finite float, or None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def as_int(x: Any, default: int = 0) -> int:
    f = as_float(x)
    return int(f) if f is not None else default


def dollars(x: Any) -> str:
    """ "$10" for whole dollars, "$2.50" otherwise, "n/a" when missing."""
    f = as_float(x)
    if f is None:
        return "n/a"
    return money(f, cents=abs(f - round(f)) >= 0.005)


def boards_text(x: Any) -> str:
    """An estimated board count rounded to the nearest thousand: "3,920,000"."""
    f = as_float(x)
    return "n/a" if f is None else fmt_num(round(f, -3))


def short_money(x: Any) -> str:
    """Compact amount for narrow phone tables: "$4.02m" from a million up, else "$950,000"."""
    f = as_float(x)
    if f is None:
        return "n/a"
    return f"${f / 1e6:,.2f}m" if abs(f) >= 1e6 else money(f)


def field(row: Any, key: str, default: Any = None) -> Any:
    """Field of a pd.Series row or a dict."""
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def as_date(value: Any) -> date | None:
    """A date from a Timestamp, datetime, date or ISO string (None when missing)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value != value:  # NaT
            return None
        return value.astimezone(SG).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts.date()


def to_sg(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(SG) if dt.tzinfo else dt.replace(tzinfo=SG)


def _and_list(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _frame(df: Any) -> pd.DataFrame | None:
    return df if isinstance(df, pd.DataFrame) and len(df) else None


def latest_row(df: Any) -> pd.Series | None:
    """The newest draw (highest draw number) of a toto.csv frame, or None."""
    df = _frame(df)
    if df is None or "draw_number" not in df.columns:
        return None
    return df.loc[pd.to_numeric(df["draw_number"], errors="coerce").idxmax()]


def draw_type_name(draw_type: Any) -> str:
    key = str(draw_type or "normal").strip().lower()
    return DRAW_TYPE_NAMES.get(key, clean_text(draw_type).title() or "Normal")


def short_date(d: Any) -> str:
    """ "Mon 5 Oct" for narrow phone tables."""
    day = as_date(d)
    return f"{day:%a} {day.day} {day:%b}" if day else "n/a"


# Next draw and note names


@dataclass(frozen=True)
class NextDraw:
    """The next TOTO draw as far as the context knows it."""

    # Latest stored draw number + 1 (draw numbers run on without gaps), for a date from the
    # next draw page only: a date worked out from the regular schedule gets None, since a
    # special draw before it would take that number.
    number: int | None
    day: date | None
    when: datetime | None  # Singapore time
    from_schedule: bool = False  # True when the date was worked out from the regular draw days
    stale: bool = False  # True when stored results are behind: draws were held after the newest one
    held: bool = False  # True when its draw time has passed but its result is not stored yet

    @property
    def when_text(self) -> str:
        if self.when is None:
            return "not announced yet"
        text = fmt_datetime(self.when)
        if self.stale:
            if self.from_schedule:
                return f"{text} (worked out from the regular schedule, results are not up to date)"
            return f"{text} (results are not up to date)"
        if self.held:
            return f"{text} (draw held, result not out yet)"
        return f"{text} (regular schedule, not announced yet)" if self.from_schedule else text


def _last_stored(ctx: Context) -> tuple[int, date | None]:
    row = latest_row(ctx.toto)
    if row is None:
        return 0, None
    return as_int(field(row, "draw_number"), 0), as_date(field(row, "draw_date"))


def next_info_is_current(ctx: Context) -> bool:
    """False when the next draw page info is about a draw already held: its date is not after
    the newest stored draw (the page still shows the draw just held) or it is before today
    (stored info from an earlier run while the site could not be read). True when there is no
    such date."""
    when = to_sg(getattr(ctx.next_toto, "draw_datetime", None))
    if when is None:
        return True
    _, last_day = _last_stored(ctx)
    return (last_day is None or when.date() > last_day) and when.date() >= to_sg(ctx.now).date()


# Start of the run warning (runner) when the newest stored draw is not the latest one on the
# site: "TOTO: the newest stored draw (4124) does not match ...".
_SITE_AHEAD = "the newest stored draw ("


def _fetched_up_to_date(ctx: Context) -> bool:
    """True when this run read the site and no warning says it has a newer draw than the stored
    ones: the newest stored draw is then the latest one held, whatever the regular days say."""
    return ctx.fetched and not any(_SITE_AHEAD in str(w) for w in ctx.warnings or [])


def _draw_day_between(after: date, before: date) -> bool:
    """True when a regular TOTO draw day falls strictly between ``after`` and ``before``."""
    d = after + timedelta(days=1)
    while d < before:
        if d.weekday() in C.TOTO_WEEKDAYS:
            return True
        d += timedelta(days=1)
    return False


def _next_regular_day(after: date) -> date:
    d = after + timedelta(days=1)
    while d.weekday() not in C.TOTO_WEEKDAYS:
        d += timedelta(days=1)
    return d


def next_draw(ctx: Context) -> NextDraw:
    """Number and date of the next TOTO draw.

    The date comes from the next draw page when it is later than the newest stored draw;
    otherwise it is the next regular draw day after the newest stored draw, with no number (a
    special draw before it would take that number). When that day is already past (stored
    results are behind, for example because the site could not be read), the next regular day
    from today is shown instead, marked ``stale``. The page date is marked ``stale`` too, with
    no number, when a regular draw day falls between the newest stored draw and it, unless this
    run read the site and found nothing newer (the draw was moved or cancelled). A draw earlier
    today still counts as the next draw, marked ``held``: its sales are closed.
    """
    last_no, last_day = _last_stored(ctx)
    number = last_no + 1 if last_no > 0 else None
    now = to_sg(ctx.now)
    today = now.date()

    when = to_sg(getattr(ctx.next_toto, "draw_datetime", None))
    if when is not None and (last_day is None or when.date() > last_day):
        nd = NextDraw(number, when.date(), when)
        if (last_day is not None and nd.day >= today and _draw_day_between(last_day, nd.day)
                and not _fetched_up_to_date(ctx)):
            return replace(nd, number=None, stale=True)
    elif last_day is None:
        return NextDraw(number, None, None)
    else:
        d = _next_regular_day(last_day)
        nd = NextDraw(None, d, datetime.combine(d, C.DRAW_TIME, tzinfo=SG), from_schedule=True)
    if nd.day is None or nd.day >= today:
        return replace(nd, held=True) if nd.when is not None and nd.when < now else nd
    d = _next_regular_day(today - timedelta(days=1))
    return NextDraw(None, d, datetime.combine(d, C.DRAW_TIME, tzinfo=SG), from_schedule=True, stale=True)


def draw_note_name(number: int, day: date | None) -> str:
    """ "2026-10-01 TOTO 4123" (the file name of a draw note, without .md)."""
    return f"{day:%Y-%m-%d} TOTO {number}" if day else f"TOTO {number}"


def report_note_name(now: datetime) -> str:
    """ "2026-10-02 1930 Report" for a run at 7.30pm Singapore time on 2 Oct 2026."""
    return f"{to_sg(now):%Y-%m-%d %H%M} Report"


def activity_note_name(now: datetime) -> str:
    """ "2026-10-02": today's activity log note (see ``vault.log_note_path``)."""
    return log_note_path(now).rsplit("/", 1)[-1].removesuffix(".md")


def link(name: str, alias: str | None = None) -> str:
    """An Obsidian [[wikilink]]."""
    return f"[[{name}|{alias}]]" if alias else f"[[{name}]]"


def draw_note_rel(number: int, day: date | None) -> str:
    """ "Draws/TOTO/2026-10-01 TOTO 4123.md": a draw note's path in the bot folder."""
    return f"{DRAW_FOLDER}/{draw_note_name(number, day)}.md"


def note_link(rel: str) -> str:
    """Wikilink to a note by its path in the bot folder, with the file name as the alias."""
    name = str(rel).removesuffix(".md")
    return link(name, name.rsplit("/", 1)[-1]) if "/" in name else link(name)


# Odds


def odds_note(rules: PrizeRules | None = None) -> str:
    """The plain odds statement, said once per report: independence and the real odds."""
    rules = rules or PrizeRules()
    return (
        "Every draw is independent, so past results do not change the odds. "
        f"TOTO Group 1 is 1 in {fmt_num(C.TOTO_COMBOS)} per board, and any TOTO prize is about "
        f"1 in {fmt_num(C.TOTO_COMBOS / C.TOTO_ANY_PRIZE_COMBOS)}. Across all draws TOTO pays "
        f"{pct(rules.pool_share_of_sales, 0)} of sales back as prizes. A snowballed jackpot can lift "
        "one draw's average return per $1, even above $1, but almost every ticket still wins nothing. "
        "The chances in this report are the chances that anybody wins Group 1, not that you do."
    )


def odds_line() -> str:
    """Short odds reminder for Telegram."""
    return (
        f"Every draw is independent. Group 1 is 1 in {fmt_num(C.TOTO_COMBOS)} per board, any prize "
        f"about 1 in {fmt_num(C.TOTO_COMBOS / C.TOTO_ANY_PRIZE_COMBOS)}. The chances above are for "
        "anybody winning, not you."
    )


# Sentences that restate the odds note, which the report and message 2 already carry once.
_ODDS_REPEAT = re.compile(r"independen|\bodds\b", re.IGNORECASE)


def commentary_text(text: Any) -> str:
    """The optional commentary without sentences that restate independence or the odds (the
    odds statement is said once, by ``odds_note`` / ``odds_line``). "" when nothing is left."""
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return " ".join(s for s in sentences if s and not _ODDS_REPEAT.search(s))


# The next draw in one place


def toto_signal(ctx: Context) -> dict:
    """Next draw figures: jackpot, draw type, rollovers, buy signal and where it comes from.

    A draw held earlier today has closed sales, so it gets no buy signal. When the next draw
    page is about a draw already held its jackpot belongs to that draw: the outlook (built
    without it) gives the jackpot instead.
    """
    bs, out = ctx.buy_signal, ctx.outlook
    jackpot = as_float(out.jackpot) if out is not None else None
    if jackpot is None and bs is not None:
        jackpot = as_float(bs.jackpot)
    page = as_float(ctx.next_toto.jackpot_estimate) if ctx.next_toto is not None and next_info_is_current(ctx) \
        else None
    if jackpot is None:
        jackpot = page
    draw_type = ((out.draw_type if out is not None else None)
                 or (bs.draw_type if bs is not None else None) or "normal")
    if out is not None:
        rollovers = int(out.snowball_draws)
    else:
        rollovers = no_winner_streak(ctx.toto, reset_on_cascade=True) if _frame(ctx.toto) is not None else 0
    if next_draw(ctx).held:
        bs = None
    return {
        "jackpot": jackpot,
        # True when the jackpot was worked out from the stored results, not read from the page
        "jackpot_worked_out": jackpot is not None and page is None,
        "draw_type": draw_type,
        "rollovers": rollovers,
        "label": bs.label if bs is not None else None,
        "reason": bs.reason if bs is not None else "",
        "ev": as_float(bs.ev_per_dollar) if bs is not None else None,
        "breakdown": dict(bs.ev_breakdown) if bs is not None and bs.ev_breakdown else {},
        "boards": as_float(bs.boards_estimate) if bs is not None else None,
        "boards_method": bs.boards_method if bs is not None else "",
    }


JACKPOT_WORKED_OUT = "worked out from past results"


def jackpot_text(sig: dict, missing: str = "not available yet") -> str:
    """The next jackpot, marked when it was worked out from the stored results."""
    if sig["jackpot"] is None:
        return missing
    return money(sig["jackpot"]) + (f" ({JACKPOT_WORKED_OUT})" if sig.get("jackpot_worked_out") else "")


def rollover_text(sig: dict) -> str:
    """How many draws the jackpot has rolled over, and how many it can before it cascades."""
    n = fmt_num(sig["rollovers"])
    if sig["draw_type"] in ("hongbao", "special"):
        return n
    if sig["draw_type"] == "cascade":
        top = C.TOTO_SNOWBALL_LIMIT - 1
        return f"{fmt_num(max(sig['rollovers'], top))} of {top}, this is the cascade draw"
    return f"{n} of {C.TOTO_SNOWBALL_LIMIT - 1}, then it cascades"


def held_text(ctx: Context) -> str | None:
    """Plain notice when the next draw was held earlier today but its result is not stored."""
    nd = next_draw(ctx)
    if not nd.held:
        return None
    draw = f"TOTO draw {nd.number}" if nd.number else "The TOTO draw"
    return f"{draw} was held at {fmt_time(nd.when)} today, result not out yet."


def _ev_rows(breakdown: dict) -> list[list[str]]:
    rows = []
    for key, label in EV_PARTS:
        value = as_float(breakdown.get(key))
        if value is None or (key == "cascade" and value <= 0):
            continue
        rows.append([label, per_dollar(value)])
    if as_float(breakdown.get("total")) is not None:
        rows.append(["Total", per_dollar(breakdown["total"])])
    return rows


def _group_rows(row: Any) -> list[list[str]]:
    """Winning shares table of a TOTO row: group, share amount, number of winning shares."""
    out = []
    for g in range(1, 8):
        share = as_float(field(row, f"g{g}_share"))
        winners = as_int(field(row, f"g{g}_winners"))
        out.append([f"Group {g}", money(share) if share is not None and winners else "n/a",
                    fmt_num(winners)])
    return out


def _group1_parts(row: Any) -> tuple[str, str]:
    """(Group 1 prize amount, plain detail on its winners) of a TOTO row."""
    jackpot = as_float(field(row, "jackpot"))
    winners = as_int(field(row, "g1_winners"))
    share = as_float(field(row, "g1_share"))
    if winners > 0:
        each = f" of {money(share)} each" if share is not None and winners > 1 else ""
        return money(jackpot), f"{plural(winners, 'winning share')}{each}"
    return money(jackpot), "no winner"


def _group1_text(row: Any) -> str:
    amount, detail = _group1_parts(row)
    return f"{amount}, {detail}"


# Jackpot outlook and history text (shared with the notes)


def outlook_rows(out: JackpotOutlook) -> list[list[str]]:
    """Projection rows for the report: draw date, jackpot, boards, chances, return, note."""
    rows = []
    for s in out.steps:
        rows.append([
            fmt_date(s.draw_date), money(s.jackpot), boards_text(s.boards),
            pct(s.chance_reached, 0), pct(s.chance_won, 0), per_dollar(s.ev_per_dollar),
            "cascade draw" if s.cascade else "",
        ])
    return rows


def big_prize_text(out: JackpotOutlook | None) -> str | None:
    """One plain sentence on the next big prize, or None when there is no projection."""
    if out is None or not out.steps:
        return None
    big = out.biggest
    if out.draws_to_cascade is None:
        return (f"The next draw is a {draw_type_name(out.draw_type)} draw with a jackpot of about "
                f"{money(big.jackpot)}.")
    if len(out.steps) == 1:
        return (f"The next draw is the cascade draw: about {money(big.jackpot)}. If nobody wins it, "
                "the jackpot goes to the Group 2 winners.")
    chance = chance_won_by_cascade(out)
    return (f"If nobody wins Group 1 first, the jackpot snowballs to about {money(big.jackpot)} at the "
            f"cascade draw on {fmt_date(big.draw_date)} ({pct(big.chance_reached, 0)} chance it gets "
            f"that far). The chance somebody wins it before then is about {pct(chance, 0)}.")


def specials_text(out: JackpotOutlook | None) -> str | None:
    if out is None or not out.special_draws:
        return None
    items = [f"{fmt_date(d)} ({draw_type_name(t)})" for d, t in out.special_draws]
    return f"Announced special draws: {_and_list(items)}."


def history_line(h: JackpotHistory | None) -> str | None:
    """One plain sentence summing up the jackpot history, or None when there is none."""
    if h is None or not h.draws:
        return None
    parts = [f"Over {plural(h.draws, 'stored draw')} Group 1 was won in {pct(h.won_share, 0)} of draws"]
    if h.average_run is not None:
        parts.append(f"a jackpot lasted {fmt_num(h.average_run, 1)} draws on average")
    if h.biggest:
        top = h.biggest[0]
        parts.append(f"the biggest was {money(top['jackpot'])} on {fmt_date(top['draw_date'])}")
    return _and_list(parts) + "."


# Markdown blocks (shared with notes)


def toto_result_md(row: Any) -> str:
    """Markdown for one TOTO result: numbers, Group 1 prize and the full winning shares table."""
    nums = toto_numbers(row)
    additional = as_int(field(row, "additional"))
    return "\n".join([
        f"Winning numbers: **{toto_nums(nums)}**, additional number **{additional}**.",
        "",
        f"Draw type: {draw_type_name(field(row, 'draw_type'))}. Group 1 prize: {_group1_text(row)}.",
        "",
        md_table(["Group", "Share amount", "Winning shares"], _group_rows(row), align="lrr"),
    ])


def outlook_md(out: JackpotOutlook | None) -> str:
    """Markdown for the jackpot projection up to the cascade draw."""
    if out is None or not out.steps:
        return "There is no jackpot projection in this run (no jackpot estimate and no stored results)."
    parts = [md_table(
        ["Draw", "Jackpot if unwon before", "Boards sold (estimate)", "Still unwon by then", "Won at this draw",
         "Return per $1", ""], outlook_rows(out), align="lrrrrrl")]
    for text in (big_prize_text(out), specials_text(out)):
        if text:
            parts.append(text)
    parts.append("\"Won at this draw\" is the chance that at least one board in the whole draw matches all six "
                 "numbers, with the boards spread evenly over every combination; it is not your chance.")
    parts.extend(clean_text(n) for n in out.notes)
    return "\n\n".join(parts)


def history_md(h: JackpotHistory | None) -> str:
    if h is None or not h.draws:
        return "No results are stored yet."
    rows = [
        ["Draws stored", fmt_num(h.draws)],
        ["Draws where Group 1 was won", f"{fmt_num(h.won_draws)} ({pct(h.won_share, 0)})"],
        ["Cascades (nobody won it by the 4th draw)", fmt_num(h.cascades)],
        ["Average draws a jackpot lasts", fmt_num(h.average_run, 1) if h.average_run is not None else "n/a"],
        ["Typical Group 1 prize when won (median)", money(h.typical_won)],
    ]
    if h.last_won:
        lw = h.last_won
        rows.append(["Last won", f"draw {lw['draw_number']}, {fmt_date(lw['draw_date'])}, {money(lw['jackpot'])}, "
                                 f"{plural(lw['winners'], 'winning share')}"])
    parts = [md_table(["Item", "Value"], rows, align="ll")]
    if h.biggest:
        parts.append("Biggest Group 1 prizes stored:")
        parts.append(md_table(
            ["Draw", "Date", "Group 1 prize", "Winning shares", "Draw type"],
            [[str(b["draw_number"]), fmt_date(b["draw_date"]), money(b["jackpot"]), fmt_num(b["winners"]),
              draw_type_name(b["draw_type"])] for b in h.biggest], align="llrrl"))
    return "\n\n".join(parts)


# Tickets


# Corrected tickets: a checked ticket whose row was edited in Tickets.md is checked again and
# replaces the old check (tickets.sync_ledger). The old result was announced before, so only
# the difference is new money.
CORRECTION_HINT = ("A corrected row replaces the ticket already checked, which is no longer counted. "
                   "If both are real tickets, put the old row back in Tickets.md.")


def corrections(ctx: Context) -> dict[str, dict]:
    """The checked ticket each ticket checked in this run replaced as a corrected row, keyed by
    the new row's ticket_id (see ``tickets.replaced_rows``). Empty when there is none."""
    settled = list(ctx.settled_this_run or [])
    if not settled or not isinstance(ctx.ledger, pd.DataFrame) or ctx.ledger.empty:
        return {}
    try:
        return ticket_ledger.replaced_rows(ctx.ledger, settled)
    except Exception as exc:  # never let a layout helper break the report
        log.warning("Corrected tickets could not be matched to the tickets they replaced: %s", exc)
        return {}


def _was(fixes: dict[str, dict], r: dict) -> dict | None:
    return fixes.get(str(r.get("ticket_id")))


def _run_winnings(settled: Sequence[dict], fixes: dict[str, dict]) -> tuple[float, int]:
    total, count = 0.0, 0
    for r in settled:
        won = as_float(r.get("winnings")) or 0.0
        was = _was(fixes, r)
        if was is not None:
            won = max(0.0, won - (as_float(was.get("winnings")) or 0.0))
        if won > 0:
            total, count = total + won, count + 1
    return total, count


def run_winnings(ctx: Context) -> tuple[float, int]:
    """(money won by the tickets checked in this run, how many tickets won it). A corrected
    ticket counts only what it won above the ticket it replaced."""
    return _run_winnings(list(ctx.settled_this_run or []), corrections(ctx))


def _correction_note(was: dict) -> str:
    """ "corrected from 1 2 3 4 5 6 Ordinary $1, was won $10": the ticket a corrected row replaced."""
    won = as_float(was.get("winnings")) or 0.0
    return (f"corrected from {clean_text(was.get('numbers'))} {clean_text(was.get('bet_type'))} "
            f"{dollars(was.get('cost'))}, " + (f"was won {dollars(won)}" if won > 0 else "was no prize"))


def _ticket_rows(rows: Sequence[dict], fixes: dict[str, dict] | None = None) -> list[list[str]]:
    out = []
    for r in rows:
        result = clean_text(r.get("result")) or "n/a"
        was = _was(fixes or {}, r)
        if was is not None:
            result = f"{result}, {_correction_note(was)}"
        out.append([fmt_date(r.get("draw_date")), clean_text(r.get("numbers")), clean_text(r.get("bet_type")),
                    dollars(r.get("cost")), result, dollars(r.get("winnings") or 0.0)])
    return out


def _totals_md(totals: dict) -> str:
    if not totals or not as_int(totals.get("tickets")):
        return f"No tickets in the ledger yet. Add the tickets you buy to {link('Tickets')}."
    rows = [[fmt_num(as_int(totals.get("tickets"))), dollars(totals.get("spent")), dollars(totals.get("won")),
             dollars(totals.get("net")), fmt_num(as_int(totals.get("pending"))),
             dollars(totals.get("pending_cost"))]]
    return "Totals so far:\n\n" + md_table(
        ["Tickets", "Spent", "Won", "Net", "Waiting for draw", "Waiting cost"], rows, align="rrrrrr")


# Full report


def _md_section1(ctx: Context) -> str:
    sig = toto_signal(ctx)
    nd = next_draw(ctx)
    parts = [SECTION_HEADINGS[0]]
    first = ctx.outlook.steps[0] if ctx.outlook is not None and ctx.outlook.steps else None
    rows = [
        ["Draw", str(nd.number) if nd.number else "n/a"],
        ["Date and time", nd.when_text],
        ["Estimated jackpot", jackpot_text(sig)],
        ["Draw type", draw_type_name(sig["draw_type"])],
        ["Jackpot rollovers so far", rollover_text(sig)],
        ["Chance somebody wins Group 1 at this draw", pct(first.chance_won, 0) if first else "not available"],
        ["Buy signal", f"**{sig['label']}**" if sig["label"] else "not available"],
        ["Return per $1", per_dollar(sig["ev"]) if sig["ev"] is not None else "not available"],
    ]
    if sig["boards"] is not None:
        rows.append(["Sales estimate", f"about {boards_text(sig['boards'])} boards ({clean_text(sig['boards_method'])})"])
    parts.append(md_table(["Item", "Value"], rows, align="ll"))
    if nd.held:
        parts.append(f"This draw was held at {fmt_time(nd.when)} today and its result is not out yet, so its "
                     "sales are closed and there is no buy signal for it.")
    if sig["reason"]:
        parts.append(clean_text(sig["reason"]))
    ev_rows = _ev_rows(sig["breakdown"])
    if ev_rows:
        parts.append("Where the return per $1 comes from (average over many draws, including the chance of "
                     "sharing a prize with other winners at typical sales for this jackpot):")
        parts.append(md_table(["Prize group", "Return per $1"], ev_rows, align="lr"))
    parts.append("### The next big prize")
    parts.append(outlook_md(ctx.outlook))
    return "\n\n".join(parts)


def _md_section2(ctx: Context) -> str:
    parts = [SECTION_HEADINGS[1]]
    row = latest_row(ctx.toto)
    if row is None:
        parts.append("No TOTO results are stored yet.")
    else:
        number, day = as_int(field(row, "draw_number")), as_date(field(row, "draw_date"))
        parts.append(f"### Latest draw {number}, {fmt_date(day)}")
        if not ctx.new_draws:
            parts.append("No new draw in this run; this is the newest stored result.")
        parts.append(toto_result_md(row))
        parts.append(f"Draw note: {note_link(draw_note_rel(number, day))}.")

    parts.append("### My ticket check")
    settled = list(ctx.settled_this_run or [])
    if settled:
        fixes = corrections(ctx)
        won, winners = _run_winnings(settled, fixes)
        if won > 0:
            lead = f"**You won {dollars(won)}** with {plural(winners, 'ticket')} in this run."
        elif any((as_float(r.get("winnings")) or 0.0) > 0 for r in settled):
            lead = "Nothing more was won in this run."
        else:
            lead = "None of the tickets checked in this run won a prize."
        parts.append(f"{plural(len(settled), 'ticket')} checked in this run. {lead}")
        parts.append(md_table(["Draw date", "Numbers", "Bet type", "Cost", "Result", "Won"],
                              _ticket_rows(settled, fixes), align="lllrlr"))
        if fixes:
            what = ("1 ticket checked in this run is a corrected row of a ticket already checked, so only "
                    "what it won above the old check counts" if len(fixes) == 1 else
                    f"{plural(len(fixes), 'ticket')} checked in this run are corrected rows of tickets already "
                    "checked, so only what they won above the old checks counts")
            parts.append(f"{what} as won in this run. {CORRECTION_HINT}")
    else:
        parts.append("No tickets were checked in this run.")
    parts.append(_totals_md(ctx.ledger_totals))
    if ctx.bad_ticket_lines:
        parts.append(f"{plural(len(ctx.bad_ticket_lines), 'line')} in {link('Tickets')} could not be read; "
                     f"{link('Ledger')} lists them with the reason.")
    return "\n\n".join(parts)


def _rules_md(rules: PrizeRules) -> str:
    status = ("confirmed on the official prize page" if rules.toto_confirmed
              else "not confirmed, built in values used")
    checked = "n/a"
    if rules.checked_at:
        try:
            checked = fmt_datetime(datetime.fromisoformat(str(rules.checked_at)))
        except ValueError:
            checked = clean_text(rules.checked_at)
    return "\n\n".join([
        "### Prize rules used",
        f"TOTO prize rules: {status}. {sentence(rules.source_note) or 'Built in values.'} Last checked: {checked}.",
        f"Prize pool {pct(rules.pool_share_of_sales, 0)} of sales; Group 1 "
        f"{pct(rules.group_pool_pct.get(1), 0)} (at least {money(rules.min_group1)}), Group 2 "
        f"{pct(rules.group_pool_pct.get(2), 0)}, Group 3 {pct(rules.group_pool_pct.get(3), 1)}, Group 4 "
        f"{pct(rules.group_pool_pct.get(4), 0)}; Group 5 {money(rules.fixed_prizes.get(5))}, Group 6 "
        f"{money(rules.fixed_prizes.get(6))}, Group 7 {money(rules.fixed_prizes.get(7))}. An unwon Group 1 "
        f"prize snowballs for up to {C.TOTO_SNOWBALL_LIMIT} draws, then cascades to the next group with "
        "winners.",
    ])


def _md_section3(ctx: Context) -> str:
    return "\n\n".join([SECTION_HEADINGS[2], history_md(ctx.history), _rules_md(ctx.rules)])


def report_warnings(ctx: Context) -> list[str]:
    """Run warnings, settings problems and unconfirmed prize rules, each once, dash free."""
    items = [clean_text(w) for w in list(ctx.warnings or []) + list(getattr(ctx.settings, "warnings", []) or [])]
    if not ctx.rules.confirmed:
        items.append("The TOTO prize rules could not be confirmed on the official page, so built in values "
                     "were used.")
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def stale_text(ctx: Context) -> str | None:
    """Plain notice when the stored results are behind (draws were held after the newest one)."""
    last_no, _ = _last_stored(ctx)
    if not last_no or not next_draw(ctx).stale:
        return None
    return (f"Results not up to date: the newest stored result is draw {last_no}, and newer draws have been "
            "held since. The Singapore Pools site could not be read or did not have every result yet, or "
            "fetching was turned off.")


def full_report(ctx: Context) -> str:
    """The run report as Obsidian markdown."""
    parts = [f"# Huat Bot report, {fmt_datetime(to_sg(ctx.now))}"]
    if ctx.new_draws:
        parts.append(f"New in this run: {plural(len(ctx.new_draws), 'draw')} "
                     f"({_and_list([str(n) for n in ctx.new_draws[-5:]])}{' and earlier' if len(ctx.new_draws) > 5 else ''}).")
    else:
        parts.append("No new draws in this run.")
    for notice in (stale_text(ctx), held_text(ctx)):
        if notice:
            parts.append(f"**{notice}**")
    warnings = report_warnings(ctx)
    if warnings:
        parts.append("> [!warning] Warnings\n" + "\n".join(f"> * {w}" for w in warnings))
    for build in (_md_section1, _md_section2, _md_section3):
        parts.append(build(ctx))
    parts.append(f"{SECTION_HEADINGS[3]}\n\n{odds_note(ctx.rules)}")
    comment = commentary_text(ctx.commentary)
    if comment:
        parts.append(f"*Commentary:* {comment}")
    return "\n\n".join(parts).rstrip() + "\n"


# Telegram


def _tg_join(blocks: Sequence[str]) -> str:
    return "\n\n".join(b for b in blocks if b)


def _scrub(text: str) -> str:
    """Final safety net: Telegram messages carry no dash at all."""
    if contains_dash(text):
        log.warning("A Telegram message still held a dash; removing it")
        text = remove_dashes(text)
    return text


def _fit(build: Callable[[int], list[str]]) -> str:
    """The most detailed version of a message that fits the Telegram limit.

    ``build(level)`` returns whole, self contained HTML blocks; higher levels carry less detail.
    If even the last level is too long, whole blocks are dropped from the end and a pointer to
    the vault report is added, so a tag is never cut.
    """
    for level in range(_TG_LEVELS):
        text = _tg_join(build(level))
        if len(text) <= LIMIT:
            return _scrub(text)
    blocks = build(_TG_LEVELS - 1)
    while blocks and len(_tg_join(blocks + [_TG_MORE])) > LIMIT:
        blocks.pop()
    return _scrub(_tg_join(blocks + [_TG_MORE]))


def _tg_result(ctx: Context) -> str:
    row = latest_row(ctx.toto)
    if row is None:
        return "No TOTO result is stored yet."
    lines = [
        f"🔢 <b>Winning numbers</b> · <code>{toto_nums(toto_numbers(row))}</code>",
        f"➕ Additional number · <code>{as_int(field(row, 'additional'))}</code>",
        "🏆 Group 1 · <b>{}</b>, {}".format(*(_h(x) for x in _group1_parts(row))),
        pre_block(["Group", "Share", "Winners"], _group_rows(row), "lrr"),
    ]
    if not ctx.new_draws:
        lines.insert(0, "<i>No new draw in this run, this is the newest stored result.</i>")
    return "\n".join(lines)


def _result_title(ctx: Context) -> str:
    row = latest_row(ctx.toto)
    if row is None:
        return title("result", "TOTO result", fmt_date(to_sg(ctx.now)))
    dtype = str(field(row, "draw_type", "normal"))
    kind = f", {draw_type_name(dtype)} draw" if dtype != "normal" else ""
    return title("result", "TOTO result",
                 f"Draw {as_int(field(row, 'draw_number'))}, {fmt_date(as_date(field(row, 'draw_date')))}{kind}")


def _ticket_line(r: dict, was: dict | None = None) -> str:
    won = as_float(r.get("winnings")) or 0.0
    head = f"{fmt_date(r.get('draw_date'))}, {_h(r.get('numbers'))}, {_h(r.get('bet_type'))} {dollars(r.get('cost'))}"
    fix = f", {_h(_correction_note(was))}" if was is not None else ""
    if won > 0:
        return f"{head}: {_h(r.get('result'))}{fix}, {'now won' if fix else 'won'} <b>{dollars(won)}</b>"
    return f"{head}: {_h(r.get('result') or 'No prize')}{fix}"


def _ticket_mark(r: dict) -> str:
    return "🟢" if (as_float(r.get("winnings")) or 0.0) > 0 else "⚪"


def _no_draw_cost(ctx: Context) -> float:
    """Cost of the ledger tickets whose date had no draw (status "no_draw")."""
    ledger = ctx.ledger
    if not isinstance(ledger, pd.DataFrame) or ledger.empty or "status" not in ledger.columns:
        return 0.0
    cost = pd.to_numeric(ledger.loc[ledger["status"] == "no_draw", "cost"], errors="coerce")
    return float(cost.fillna(0.0).sum())


def _tg_tickets(ctx: Context, level: int) -> str:
    lines = [title("tickets", "My tickets")]
    settled = list(ctx.settled_this_run or [])
    fixes = corrections(ctx)
    winners = sorted((r for r in settled if (as_float(r.get("winnings")) or 0.0) > 0),
                     key=lambda r: -(as_float(r.get("winnings")) or 0.0))
    losers = [r for r in settled if (as_float(r.get("winnings")) or 0.0) <= 0]
    total_won, n_won = _run_winnings(settled, fixes)
    if not settled:
        lines.append("No tickets were checked in this run.")
    elif level == 0:
        lines += [f"{_ticket_mark(r)} {_ticket_line(r, _was(fixes, r))}" for r in winners + losers]
    elif level <= 2:
        # A corrected ticket is always listed: it changes a result already announced.
        shown = [r for k, r in enumerate(winners) if level == 1 or k < _TG_WINNERS_CAP or _was(fixes, r)]
        fixed_losers = [r for r in losers if _was(fixes, r)]
        lines += [f"{_ticket_mark(r)} {_ticket_line(r, _was(fixes, r))}" for r in shown + fixed_losers]
        if len(shown) < len(winners):
            lines.append(f"🟢 and {plural(len(winners) - len(shown), 'more winning ticket')}")
        if len(losers) > len(fixed_losers):
            lines.append(f"{plural(len(losers) - len(fixed_losers), 'other ticket')} checked won nothing.")
    else:
        lines.append(f"{plural(len(settled), 'ticket')} checked, {fmt_num(n_won)} won, "
                     f"{dollars(total_won)} in total.")
        if fixes:
            lines.append("1 of them is a corrected row of a ticket already checked, see Ledger.md."
                         if len(fixes) == 1 else
                         f"{fmt_num(len(fixes))} of them are corrected rows of tickets already checked, see "
                         "Ledger.md.")
    if fixes:
        lines.append(f"<i>{_h(CORRECTION_HINT)}</i>")

    totals = ctx.ledger_totals or {}
    if as_int(totals.get("tickets")):
        lines.append(f"📒 All tickets so far: spent {dollars(totals.get('spent'))}, won "
                     f"{dollars(totals.get('won'))}, net <b>{dollars(totals.get('net'))}</b>.")
        pending = as_int(totals.get("pending"))
        if pending:
            lines.append(f"{plural(pending, 'ticket')} ({dollars(totals.get('pending_cost'))}) "
                         f"{'waits' if pending == 1 else 'wait'} for the draw.")
        no_draw = as_int(totals.get("no_draw"))
        if no_draw:
            lines.append(f"{plural(no_draw, 'ticket')} ({dollars(_no_draw_cost(ctx))}) "
                         f"{'has' if no_draw == 1 else 'have'} no draw on {'its' if no_draw == 1 else 'their'} "
                         f"date and still {'counts' if no_draw == 1 else 'count'} as spent. Check the date in "
                         "Tickets.md.")
    elif not settled:
        lines.append("Add the tickets you buy to Tickets.md in the vault and the bot will check them.")
    if ctx.bad_ticket_lines:
        lines.append(f"{plural(len(ctx.bad_ticket_lines), 'line')} in Tickets.md could not be read, "
                     "see Ledger.md.")
    return "\n".join(lines)


def _tg_message1(ctx: Context, level: int) -> list[str]:
    won, n = run_winnings(ctx)
    head = _result_title(ctx)
    if won > 0:
        head = (title("winner", "Winner", f"your tickets won {dollars(won)}") + "\n"
                + f"{plural(n, 'winning ticket')}, details below.\n\n" + head)
    blocks = [head]
    for notice in (stale_text(ctx), held_text(ctx)):
        if notice:
            blocks.append(f"<i>{_h(notice)}</i>")
    blocks.append(_tg_result(ctx))
    blocks.append(DIVIDER)
    blocks.append(_tg_tickets(ctx, level))
    return blocks


def _jackpot_change(ctx: Context, jackpot: float | None) -> str:
    """ " 🟢 <i>UP ▲$900,000</i>": the next jackpot against the last draw's Group 1 prize. A
    bigger jackpot is good news for the reader (green), a reset after a win is not (red)."""
    row = latest_row(ctx.toto)
    last = as_float(field(row, "jackpot")) if row is not None else None
    if jackpot is None or last is None or abs(jackpot - last) < 1:
        return ""
    if jackpot > last:
        return f" 🟢 <i>UP ▲{_h(money(jackpot - last))}</i>"
    return f" 🔴 <i>RESET ▼{_h(money(last - jackpot))}</i>"


def _tg_next_draw(ctx: Context, level: int) -> str:
    sig = toto_signal(ctx)
    nd = next_draw(ctx)
    out = ctx.outlook
    first = out.steps[0] if out is not None and out.steps else None
    sub = nd.when_text + (f", draw {nd.number}" if nd.number else "")
    lines = [title("next", "Next TOTO draw", sub), ""]
    worked_out = f" <i>({JACKPOT_WORKED_OUT})</i>" if sig["jackpot_worked_out"] else ""
    dtype = draw_type_name(sig["draw_type"])
    dtype_text = f"<b>{_h(dtype)} draw</b>" if sig["draw_type"] != "normal" else f"{_h(dtype)} draw"
    if sig["jackpot"] is not None:
        lines.append(f"💰 <b>Jackpot {money(sig['jackpot'])}</b>{worked_out}{_jackpot_change(ctx, sig['jackpot'])}")
    else:
        lines.append("💰 Jackpot not available yet")
    lines.append(f"🗓 {dtype_text} · rollovers {_h(rollover_text(sig))}")
    if first is not None:
        lines.append(f"🎯 Somebody wins Group 1: <b>{pct(first.chance_won, 0)}</b>")
    if nd.held:
        lines.append("⏸ Its sales are closed, so there is no buy signal for it.")
        return "\n".join(lines)
    label = sig["label"]
    marker = SIGNAL_MARKERS.get(str(label or "").upper(), "⚪")
    signal = f"{marker} Buy signal <b>{_h(label or 'not available')}</b>"
    if sig["ev"] is not None:
        signal += f" · <b>{per_dollar(sig['ev'])}</b> back per $1 on average"
    lines.append(signal)
    if sig["reason"] and level < 3:
        lines.append(f"<i>{_h(sig['reason'])}</i>")
    ev_rows = _ev_rows(sig["breakdown"]) if sig["ev"] is not None else []
    if ev_rows and level < 4:
        lines.append(pre_block(["Part", "Per $1"],
                               [[r[0].replace(" (fixed prizes)", ""), r[1]] for r in ev_rows], "lr"))
    if sig["boards"] is not None and level < 1:
        lines.append(f"🎟 Sales estimate: about {boards_text(sig['boards'])} boards ({_h(sig['boards_method'])}).")
    return "\n".join(lines)


def _tg_big_prize(ctx: Context, level: int) -> str | None:
    out = ctx.outlook
    if out is None or not out.steps:
        return None
    big = out.biggest
    when = f" on {short_date(big.draw_date)}" if big is not None and big.draw_date and len(out.steps) > 1 else ""
    lines = [title("big", "Next big prize", f"about {short_money(big.jackpot)}{when}" if big is not None else "")]
    text = big_prize_text(out)
    if text:
        lines.append(_h(text))
    if len(out.steps) > 1 and level < 4:
        rows = [[short_date(s.draw_date), short_money(s.jackpot), pct(s.chance_reached, 0), pct(s.chance_won, 0)]
                for s in out.steps]
        lines.append(pre_block(["Draw", "Jackpot", "Unwon", "Won"], rows, "lrrr"))
        lines.append("<i>Unwon: chance nobody has won it by then. Won: chance somebody wins at that draw.</i>")
    special = specials_text(out)
    if special:
        lines.append(f"🧧 {_h(special)}")
    return "\n".join(lines)


def _tg_message2(ctx: Context, level: int) -> list[str]:
    blocks = [_tg_next_draw(ctx, level)]
    big = _tg_big_prize(ctx, level)
    if big:
        blocks += [DIVIDER, big]
    more = []
    hist = history_line(ctx.history)
    if hist and level < 2:
        more.append(f"{SECTION_TITLES['more']} {_h(hist)}")
    more.append(f"⚖️ {_h(odds_line())}")
    comment = commentary_text(ctx.commentary)
    if comment and level < 4:
        more.append(f"💬 {html_escape(comment)}")
    blocks.append("<blockquote expandable>" + "\n\n".join(more) + "</blockquote>")
    return [b for b in blocks if b]


def telegram_messages(ctx: Context) -> list[str]:
    """Exactly two Telegram HTML messages: the latest result with the ticket check, then the next
    draw and the next big prize. Each fits ``constants.TELEGRAM_MAX_CHARS`` and has no dash."""
    return [
        _fit(lambda level: _tg_message1(ctx, level)),
        _fit(lambda level: _tg_message2(ctx, level)),
    ]
