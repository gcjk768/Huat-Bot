"""The full markdown report and the three Telegram messages, built from a ``models.Context``.

Every figure shown here was computed elsewhere in Python (analysis, plans, backtests, ledger);
this module only lays it out. House style (SPEC, "Global conventions"):

* Telegram messages contain no dash characters at all, use only the <b>, <i>, <pre> and <code>
  tags, escape all text, and stay within ``constants.TELEGRAM_MAX_CHARS``. When a message would
  be too long, detail is dropped in whole blocks (reasons first, then ticket lines) so a tag is
  never cut in half.
* The markdown report has no dashes in prose (tables, frontmatter and [[wikilinks]] may carry
  hyphens) and uses the section headings in ``SECTION_HEADINGS``, in that order.

Text that comes from other modules (strategy reasons, plan notes, verdicts, the commentary) is
passed through ``textfmt.remove_dashes`` as a last line of defence.

A few view helpers live here because ``notes`` shares them: ``latest_row``, ``next_draw``,
the note names and links (``draw_note_name``, ``draw_note_rel``, ``note_link``,
``report_note_name`` ...) and the markdown blocks for a draw result, a purchase plan and a
backtest scoreboard.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from . import buysignal
from . import constants as C
from . import prizes
from . import tickets as ticket_ledger
from .analysis_fourd import bet_type_value
from .analysis_toto import format_p
from .backtest import BETTER_ABOVE, RANDOM_NAME, WORSE_BELOW
from .models import BacktestResult, Context, Plan, PrizeRules
from .store import fourd_numbers, no_winner_streak, toto_numbers
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
    "## 1. Next draws, prizes and buy signal",
    "## 2. Latest results and my ticket check",
    "## 3. Suggested numbers and total cost",
    "## 4. Backtest scoreboard",
    "## 5. Key stats tables",
    "## 6. Odds note",
)

GAME_NAMES = {"toto": "TOTO", "4d": "4D"}
DRAW_FOLDERS = {"toto": "Draws/TOTO", "4d": "Draws/4D"}  # one note per draw, in the bot folder
DRAW_TYPE_NAMES = {"normal": "Normal", "cascade": "Cascade", "hongbao": "Hongbao", "special": "Special"}
EV_PARTS = (
    ("g1", "Group 1"),
    ("g2", "Group 2"),
    ("g3", "Group 3"),
    ("g4", "Group 4"),
    ("fixed", "Groups 5 to 7 (fixed prizes)"),
    ("cascade", "Cascaded jackpot"),
)
BET_TYPES = ("Big", "Small", "iBet Big", "iBet Small")

# Telegram: how many detail levels the message builders know (0 = everything). Each step
# drops more: 1 the strategy reasons and the losing ticket lines; 2 the sales estimate, the
# picks left out of the plan, plan notes and winning tickets beyond the first 10; 3 every
# ticket line, the buy signal reason and the System 7 option; 4 the return breakdown, the
# backtest line and the commentary.
_TG_LEVELS = 5
_TG_MORE = "<i>More detail is in the report note in the vault.</i>"
_TG_WINNERS_CAP = 10  # winning tickets listed one by one at the tighter detail levels


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
    """The newest draw (highest draw number) of a toto.csv / fourd.csv frame, or None."""
    df = _frame(df)
    if df is None or "draw_number" not in df.columns:
        return None
    return df.loc[pd.to_numeric(df["draw_number"], errors="coerce").idxmax()]


def draw_type_name(draw_type: Any) -> str:
    key = str(draw_type or "normal").strip().lower()
    return DRAW_TYPE_NAMES.get(key, clean_text(draw_type).title() or "Normal")


def games_text(games: Iterable[str]) -> str:
    return _and_list([GAME_NAMES.get(g, str(g).upper()) for g in games])


# Next draws and note names


@dataclass(frozen=True)
class NextDraw:
    """The next draw of one game as far as the context knows it."""

    game: str  # "toto" or "4d"
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


def _last_stored(ctx: Context, game: str) -> tuple[int, date | None]:
    row = latest_row(ctx.toto if game == "toto" else ctx.fourd)
    if row is None:
        return 0, None
    return as_int(field(row, "draw_number"), 0), as_date(field(row, "draw_date"))


def next_info_is_current(ctx: Context, game: str) -> bool:
    """False when the next draw page info in the context is about a draw already held: its
    date is not after the newest stored draw (the page still shows the draw just held) or it
    is before today (stored info from an earlier run while the site could not be read). Its
    date, jackpot and draw type must not be used then. True when there is no such date."""
    info = ctx.next_toto if game == "toto" else ctx.next_fourd
    when = to_sg(getattr(info, "draw_datetime", None))
    if when is None:
        return True
    _, last_day = _last_stored(ctx, game)
    return (last_day is None or when.date() > last_day) and when.date() >= to_sg(ctx.now).date()


# Start of the run warning (runner._update_from_site) for a game whose newest stored draw is
# not the latest draw on the site: "TOTO: the newest stored draw (4124) does not match ...".
_SITE_AHEAD = "the newest stored draw ("


def _fetched_up_to_date(ctx: Context, game: str) -> bool:
    """True when this run read ``game`` from the site (it is reported and has an entry in
    ``ctx.new_draws``) and no warning says the site has a draw newer than the stored ones:
    the newest stored draw is then the latest one held, whatever the regular draw days say."""
    if game not in (ctx.games_drawn or ()) or game not in (ctx.new_draws or {}):
        return False
    prefix = f"{GAME_NAMES[game]}: {_SITE_AHEAD}"
    return not any(str(w).startswith(prefix) for w in ctx.warnings or [])


def _draw_day_between(after: date, before: date, weekdays: Iterable[int]) -> bool:
    """True when a day strictly between ``after`` and ``before`` falls on one of ``weekdays``."""
    days = set(weekdays)
    d = after + timedelta(days=1)
    while d < before:
        if d.weekday() in days:
            return True
        d += timedelta(days=1)
    return False


def next_draw(ctx: Context, game: str) -> NextDraw:
    """Number and date of the next draw of ``game`` ("toto" or "4d").

    The date comes from the next draw page when it is later than the newest stored draw;
    otherwise it is the next regular draw day after the newest stored draw, with no draw
    number (a special draw before it would take that number). When that day is already past
    (the stored results are behind, for example because the site could not be read), the
    next regular draw day from today is shown instead, marked ``stale``, also with no draw
    number. The page date is marked ``stale`` the same way, keeping its date but not a
    number, when a regular draw day falls between the newest stored draw and it (a draw was
    held but is not stored, for example because this run did not fetch that game). That
    rule is skipped for a game this run read from the site with nothing newer there (see
    ``_fetched_up_to_date``): its stored results are up to date, so a regular draw day in
    between had no draw (moved or cancelled, as for a Hongbao draw or Chinese New Year). A
    draw earlier today still counts as the next draw, since its result may simply not be
    out yet, but it is marked ``held``: its sales are closed, so it gets no buy signal and
    no suggestions.
    """
    last_no, last_day = _last_stored(ctx, game)
    number = last_no + 1 if last_no > 0 else None
    weekdays = C.TOTO_WEEKDAYS if game == "toto" else C.FOURD_WEEKDAYS
    now = to_sg(ctx.now)
    today = now.date()

    info = ctx.next_toto if game == "toto" else ctx.next_fourd
    when = to_sg(getattr(info, "draw_datetime", None))
    if when is not None and (last_day is None or when.date() > last_day):
        nd = NextDraw(game, number, when.date(), when, False)
        if (last_day is not None and nd.day >= today and _draw_day_between(last_day, nd.day, weekdays)
                and not _fetched_up_to_date(ctx, game)):
            return replace(nd, number=None, stale=True)
    elif last_day is None:
        return NextDraw(game, number, None, None, False)
    else:
        d = last_day + timedelta(days=1)
        while d.weekday() not in weekdays:
            d += timedelta(days=1)
        nd = NextDraw(game, None, d, datetime.combine(d, C.DRAW_TIME, tzinfo=SG), True)
    if nd.day is None or nd.day >= today:
        return replace(nd, held=True) if nd.when is not None and nd.when < now else nd
    d = today
    while d.weekday() not in weekdays:
        d += timedelta(days=1)
    return NextDraw(game, None, d, datetime.combine(d, C.DRAW_TIME, tzinfo=SG), True, stale=True)


def draw_note_name(game: str, number: int, day: date | None) -> str:
    """ "2026-10-01 TOTO 4123" (the file name of a draw or suggestion note, without .md)."""
    label = GAME_NAMES.get(game, str(game).upper())
    return f"{day:%Y-%m-%d} {label} {number}" if day else f"{label} {number}"


def report_note_name(now: datetime) -> str:
    """ "2026-10-02 1930 Report" for a run at 7.30pm Singapore time on 2 Oct 2026."""
    return f"{to_sg(now):%Y-%m-%d %H%M} Report"


def activity_note_name(now: datetime) -> str:
    """ "2026-10 Activity": this month's activity log (see ``vault.log_note_path``)."""
    return log_note_path(now).rsplit("/", 1)[-1].removesuffix(".md")


def link(name: str, alias: str | None = None) -> str:
    """An Obsidian [[wikilink]]."""
    return f"[[{name}|{alias}]]" if alias else f"[[{name}]]"


def draw_note_rel(game: str, number: int, day: date | None) -> str:
    """ "Draws/TOTO/2026-10-01 TOTO 4123.md": a draw note's path in the bot folder."""
    return f"{DRAW_FOLDERS[game]}/{draw_note_name(game, number, day)}.md"


def note_link(rel: str) -> str:
    """Wikilink to a note by its path in the bot folder, with the folder in the link and the
    file name as the alias: ``[[Draws/TOTO/2026-10-01 TOTO 4123|2026-10-01 TOTO 4123]]``.

    A draw note and the suggestion note made before that draw share a file name, so a bare
    ``[[2026-10-01 TOTO 4123]]`` could open either one; the folder makes the link exact.
    """
    name = str(rel).removesuffix(".md")
    return link(name, name.rsplit("/", 1)[-1]) if "/" in name else link(name)


# Odds


def odds_note(rules: PrizeRules | None = None) -> str:
    """The plain odds statement, said once per report: independence and the real odds."""
    rules = rules or PrizeRules()
    big = prizes.fourd_expected_return(rules, "Big")
    return (
        "Every draw is independent, so past results do not change the odds. "
        f"TOTO Group 1 is 1 in {fmt_num(C.TOTO_COMBOS)} per board, and any TOTO prize is about "
        f"1 in {fmt_num(C.TOTO_COMBOS / C.TOTO_ANY_PRIZE_COMBOS)}. A 4D Big bet wins some prize "
        f"{C.FOURD_NUMBERS_PER_DRAW} times in {fmt_num(C.FOURD_SPACE)}. Across all draws TOTO pays "
        f"{pct(rules.pool_share_of_sales, 0)} of sales back as prizes and a 4D Big bet returns "
        f"{per_dollar(big)} per $1. A snowballed jackpot can lift one draw's average return per $1, "
        "even above $1, but almost every ticket still wins nothing, and no number or strategy "
        "changes the odds. Never spend above your budget."
    )


def odds_line() -> str:
    """Short odds reminder for Telegram (no rules needed)."""
    return (
        f"Every draw is independent. TOTO Group 1 is 1 in {fmt_num(C.TOTO_COMBOS)} per board, any "
        f"prize about 1 in {fmt_num(C.TOTO_COMBOS / C.TOTO_ANY_PRIZE_COMBOS)}. 4D Big wins some "
        f"prize {C.FOURD_NUMBERS_PER_DRAW} times in {fmt_num(C.FOURD_SPACE)}."
    )


# Shared figures


def _bet_values(ctx: Context) -> dict:
    values = ctx.fourd_bet_values or {}
    if values.get("Big") is None:
        try:
            values = bet_type_value(ctx.rules)
        except Exception as exc:  # never let a layout helper break the report
            log.warning("Bet type values could not be worked out: %s", exc)
            values = {}
    return values


def _best_ibet(values: Any) -> float | None:
    if isinstance(values, dict) and values:
        nums = [as_float(v) for v in values.values()]
        nums = [v for v in nums if v is not None]
        return max(nums) if nums else None
    return as_float(values)


def _signal_without_estimate(ctx: Context):
    """The buy signal worked out without the next draw page (no jackpot estimate)."""
    if _frame(ctx.toto) is None:
        return None
    try:
        return buysignal.buy_signal(None, ctx.toto, ctx.settings, ctx.rules)
    except Exception as exc:  # never let a layout helper break the report
        log.warning("The buy signal could not be worked out without the next draw page: %s", exc)
        return None


def toto_signal(ctx: Context) -> dict:
    """Next TOTO draw figures in one place (buy signal first, next draw page second).

    When the next draw page info is about a draw already held (see ``next_info_is_current``),
    its jackpot estimate belongs to that draw, not the next one: the figures are then worked
    out without it, so the jackpot and the return per $1 show as not available.
    """
    bs, nt = ctx.buy_signal, ctx.next_toto
    if nt is not None and not next_info_is_current(ctx, "toto"):
        bs, nt = _signal_without_estimate(ctx), None
    jackpot = as_float(bs.jackpot) if bs is not None else None
    if jackpot is None and nt is not None:
        jackpot = as_float(nt.jackpot_estimate)
    draw_type = (bs.draw_type if bs is not None and bs.draw_type else None) or (
        nt.draw_type if nt is not None and nt.draw_type else "normal")
    if bs is not None:
        streak = int(bs.no_winner_streak)
    else:
        streak = no_winner_streak(ctx.toto) if _frame(ctx.toto) is not None else 0
    if next_draw(ctx, "toto").held:
        # Sales for a draw already held are closed: its jackpot and buy signal are of no use.
        jackpot, bs = None, None
    return {
        "jackpot": jackpot,
        "draw_type": draw_type,
        "streak": streak,
        "label": bs.label if bs is not None else None,
        "reason": bs.reason if bs is not None else "",
        "ev": as_float(bs.ev_per_dollar) if bs is not None else None,
        "breakdown": dict(bs.ev_breakdown) if bs is not None and bs.ev_breakdown else {},
        "boards": as_float(bs.boards_estimate) if bs is not None else None,
        "boards_method": bs.boards_method if bs is not None else "",
    }


def held_text(ctx: Context) -> str | None:
    """Plain notice for each game whose next draw was held earlier today but whose result is
    not stored yet (see ``NextDraw.held``), or None."""
    items = []
    for game in GAME_NAMES:
        nd = next_draw(ctx, game)
        if nd.held:
            draw = f"{GAME_NAMES[game]} draw {nd.number}" if nd.number else f"The {GAME_NAMES[game]} draw"
            items.append(f"{draw} was held at {fmt_time(nd.when)} today, result not out yet.")
    return " ".join(items) or None


def _held_plan_text(nd: NextDraw) -> str:
    return (f"No suggestions: this draw was held at {fmt_time(nd.when)} today and its sales are closed. "
            "Suggestions for the draw after it come once its result is stored.")


def picks_not_in_plan(plan: Plan | None, picks: Sequence[Any], game: str) -> list[Any]:
    """Picks the plan does not buy, either on their own or as a board inside a System 7 line.

    The one rule for "is this pick bought", shared by the report, message 3 and the notes.
    """
    if plan is None:
        return list(picks)
    labels = {ln.label for ln in plan.lines}
    sys7 = [{int(x) for x in str(ln.numbers).split()} for ln in plan.lines if ln.label == "System 7"]

    def inside_sys7(pick: Any) -> bool:
        if game != "toto" or not sys7:
            return False
        nums = {int(x) for x in toto_nums(pick.numbers).split()}
        return any(nums <= s for s in sys7)

    return [p for p in picks if p.name not in labels and not inside_sys7(p)]


# Sentences that restate the odds note, which the report and message 3 already carry once.
_ODDS_REPEAT = re.compile(r"independen|\bodds\b", re.IGNORECASE)


def commentary_text(text: Any) -> str:
    """The optional commentary without sentences that restate independence or the odds (the
    odds statement is said once, by ``odds_note`` / ``odds_line``). "" when nothing is left."""
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return " ".join(s for s in sentences if s and not _ODDS_REPEAT.search(s))


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


def _prize_table_rows(rules: PrizeRules) -> list[list[str]]:
    big = rules.fourd_prizes.get("big", {})
    small = rules.fourd_prizes.get("small", {})
    rows = []
    for tier in C.FOURD_TIERS:
        b, s = as_float(big.get(tier)), as_float(small.get(tier))
        rows.append([prizes.FOURD_TIER_LABELS[tier], money(b) if b else "", money(s) if s else ""])
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


def shape_text(numbers: list[int]) -> str:
    odd = sum(1 for n in numbers if n % 2)
    low = sum(1 for n in numbers if n <= C.TOTO_LOW_MAX)
    return (f"{odd} odd and {len(numbers) - odd} even, {low} low (1 to {C.TOTO_LOW_MAX}) and "
            f"{len(numbers) - low} high, sum {sum(numbers)}")


# Markdown blocks (shared with notes)


def toto_result_md(row: Any) -> str:
    """Markdown for one TOTO result: numbers, Group 1 prize and the full winning shares table."""
    nums = toto_numbers(row)
    additional = as_int(field(row, "additional"))
    lines = [
        f"Winning numbers: **{toto_nums(nums)}**, additional number **{additional}**.",
        "",
        f"Draw type: {draw_type_name(field(row, 'draw_type'))}. Group 1 prize: {_group1_text(row)}.",
        "",
        md_table(["Group", "Share amount", "Winning shares"], _group_rows(row), align="lrr"),
    ]
    return "\n".join(lines)


def fourd_result_md(row: Any) -> str:
    """Markdown for one 4D result: all 23 winning numbers."""
    nums = fourd_numbers(row)
    rows = [[prizes.FOURD_TIER_LABELS[t], " ".join(nums[t]) or "n/a"] for t in C.FOURD_TIERS]
    return md_table(["Prize", "Numbers"], rows, align="ll")


def plan_md(plan: Plan | None, game: str) -> str:
    """Markdown for a purchase plan: lines with reasons, total against the budget, notes and
    the alternative plan when there is one."""
    label = GAME_NAMES.get(game, game)
    if plan is None:
        return f"No {label} plan was made in this run."
    parts = []
    if plan.lines:
        head = ["Strategy", "Numbers", "Bet type", "Cost", "Reason"] if game == "toto" else \
            ["Pick", "Number", "Bet type", "Stake", "Reason"]
        rows = [[clean_text(ln.label), ln.numbers, clean_text(ln.bet_type), dollars(ln.cost), clean_text(ln.reason)]
                for ln in plan.lines]
        parts.append(md_table(head, rows, align="lllrl"))
        parts.append(f"**Total: {dollars(plan.total)} of the {dollars(plan.budget)} {label} budget.**")
    else:
        parts.append(f"Nothing to buy for {label} (budget {dollars(plan.budget)}).")
    if plan.notes:
        parts.append("\n".join(f"* {clean_text(n)}" for n in plan.notes))
    alt = plan.alternative
    if alt is not None and alt.lines:
        rows = [[clean_text(ln.label), ln.numbers, clean_text(ln.bet_type), dollars(ln.cost)] for ln in alt.lines]
        parts.append(f"Alternative for the same budget ({dollars(alt.total)}):")
        parts.append(md_table(["Strategy", "Numbers", "Bet type", "Cost"], rows, align="lllr"))
        if alt.notes:
            parts.append("\n".join(f"* {clean_text(n)}" for n in alt.notes))
    return "\n\n".join(parts)


def _percentile_text(p: Any) -> str:
    f = as_float(p)
    return "n/a" if f is None else f"{fmt_num(f, 1 if f % 1 else 0)}%"


def backtest_md(result: BacktestResult | None, game: str) -> str:
    """Markdown scoreboard: cost, winnings, return per $1 and verdict per strategy (and Random)."""
    label = GAME_NAMES.get(game, game)
    if result is None:
        return f"The {label} backtest is not available in this run."
    if not result.scores:
        notes = "\n".join(f"* {clean_text(n)}" for n in result.notes)
        return f"The {label} backtest could not run.\n\n{notes}".rstrip()
    rows = []
    for s in result.scores:
        verdict = "Baseline, the average random player" if s.name == RANDOM_NAME else clean_text(s.verdict)
        rows.append([
            clean_text(s.name), fmt_num(s.draws), dollars(s.cost), dollars(s.winnings),
            per_dollar(s.return_per_dollar), fmt_num(s.wins), dollars(s.best_prize),
            _percentile_text(s.percentile_vs_random), verdict,
        ])
    span = ""
    if result.first_draw is not None and result.last_draw is not None:
        span = f", draw {result.first_draw} to {result.last_draw}"
    head = (f"{label}: last {plural(result.draws_tested, 'draw')}{span}, compared with "
            f"{fmt_num(result.random_sets_per_draw)} random players.")
    table = md_table(
        ["Strategy", "Draws", "Cost", "Winnings", "Return per $1", "Prize draws", "Best prize",
         "Rank vs random", "Verdict"], rows, align="lrrrrrrrl")
    notes = "\n".join(f"* {clean_text(n)}" for n in result.notes)
    return "\n\n".join(p for p in (head, table, notes) if p)


def backtest_summary(result: BacktestResult | None, game: str) -> str | None:
    """One plain line on how the strategies did against random players (None when missing)."""
    label = GAME_NAMES.get(game, game)
    if result is None or not result.scores:
        return None
    better, worse, same, unknown = [], [], [], []
    for s in result.scores:
        if s.name == RANDOM_NAME:
            continue
        p = as_float(s.percentile_vs_random)
        if p is None:
            unknown.append(s.name)
        elif p > BETTER_ABOVE:
            better.append(s.name)
        elif p < WORSE_BELOW:
            worse.append(s.name)
        else:
            same.append(s.name)
    bits = []
    if same:
        bits.append(f"{_and_list(same)} did no better than random")
    if better:
        bits.append(f"{_and_list(better)} beat random in this sample, which should not be expected to last")
    if worse:
        bits.append(f"{_and_list(worse)} did worse than random")
    if unknown:
        bits.append(f"{_and_list(unknown)} not compared with random")
    if not bits:
        return None
    return f"{label}, last {plural(result.draws_tested, 'draw')}: {'; '.join(bits)}."


# Full report


def _md_section1(ctx: Context) -> str:
    sig = toto_signal(ctx)
    nd = next_draw(ctx, "toto")
    parts = [SECTION_HEADINGS[0], "### Next TOTO draw"]
    rows = [
        ["Draw", str(nd.number) if nd.number else "n/a"],
        ["Date and time", nd.when_text],
        ["Estimated jackpot", money(sig["jackpot"]) if sig["jackpot"] is not None else "not available yet"],
        ["Draw type", draw_type_name(sig["draw_type"])],
        ["Draws in a row with no Group 1 winner", fmt_num(sig["streak"])],
        ["Buy signal", f"**{sig['label']}**" if sig["label"] else "not available"],
        ["Return per $1", per_dollar(sig["ev"]) if sig["ev"] is not None else "not available"],
    ]
    if sig["boards"] is not None:
        rows.append(["Sales estimate", f"about {fmt_num(sig['boards'])} boards ({clean_text(sig['boards_method'])})"])
    parts.append(md_table(["Item", "Value"], rows, align="ll"))
    if nd.held:
        parts.append(f"This draw was held at {fmt_time(nd.when)} today and its result is not out yet, so its "
                     "sales are closed and there is no buy signal for it.")
    if sig["reason"]:
        parts.append(clean_text(sig["reason"]))
    ev_rows = _ev_rows(sig["breakdown"])
    if ev_rows:
        parts.append("Where the return per $1 comes from (average over many draws, including the "
                     "chance of sharing a prize with other winners at typical sales for this jackpot):")
        parts.append(md_table(["Prize group", "Return per $1"], ev_rows, align="lr"))

    nd4 = next_draw(ctx, "4d")
    parts.append("### Next 4D draw")
    parts.append(f"Draw {nd4.number if nd4.number else 'n/a'}, {nd4.when_text}.")
    parts.append("Prize for each $1 staked:")
    parts.append(md_table(["Prize", "Big", "Small"], _prize_table_rows(ctx.rules), align="lrr"))
    values = _bet_values(ctx)
    if values.get("Big") is not None:
        best = clean_text(values.get("best") or "Big")
        parts.append(f"Average return per $1: Big {per_dollar(values.get('Big'))}, Small "
                     f"{per_dollar(values.get('Small'))}. {best} gives the most back.")
    return "\n\n".join(parts)


# Corrected tickets: a checked ticket whose row was edited in Tickets.md is checked again and
# replaces the old check (tickets.sync_ledger). The old result was announced before, so only
# the difference is new money.
CORRECTION_HINT = ("A corrected row replaces the ticket already checked, which is no longer counted. "
                   "If both are real tickets, put the old row back in Tickets.md.")


def corrections(ctx: Context) -> dict[str, dict]:
    """The checked ticket each ticket checked in this run replaced as a corrected row, keyed
    by the new row's ticket_id (see ``tickets.replaced_rows``). Empty when there is none."""
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
    ticket counts only what it won above the ticket it replaced, whose win was announced
    before (never less than nothing)."""
    return _run_winnings(list(ctx.settled_this_run or []), corrections(ctx))


def _correction_note(was: dict) -> str:
    """ "corrected from 0601 Big $2, was won $500": the ticket a corrected row replaced."""
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
        out.append([
            clean_text(r.get("game")), fmt_date(r.get("draw_date")), clean_text(r.get("numbers")),
            clean_text(r.get("bet_type")), dollars(r.get("cost")), result, dollars(r.get("winnings") or 0.0),
        ])
    return out


def _totals_md(totals: dict) -> str:
    if not totals or not as_int(totals.get("tickets")):
        return f"No tickets in the ledger yet. Add the tickets you buy to {link('Tickets')}."
    rows = [[fmt_num(as_int(totals.get("tickets"))), dollars(totals.get("spent")), dollars(totals.get("won")),
             dollars(totals.get("net")), fmt_num(as_int(totals.get("pending"))),
             dollars(totals.get("pending_cost"))]]
    return "Totals so far:\n\n" + md_table(
        ["Tickets", "Spent", "Won", "Net", "Waiting for draw", "Waiting cost"], rows, align="rrrrrr")


def _md_section2(ctx: Context) -> str:
    parts = [SECTION_HEADINGS[1]]
    drawn = [g for g in ctx.games_drawn if g in GAME_NAMES]
    order = drawn + [g for g in GAME_NAMES if g not in drawn]
    for game in order:
        row = latest_row(ctx.toto if game == "toto" else ctx.fourd)
        label = GAME_NAMES[game]
        if row is None:
            parts.append(f"### Latest {label} draw\n\nNo {label} results are stored yet.")
            continue
        number, day = as_int(field(row, "draw_number")), as_date(field(row, "draw_date"))
        parts.append(f"### Latest {label} draw {number}, {fmt_date(day)}")
        if game not in drawn:
            parts.append(f"No new {label} draw in this run; this is the newest stored result.")
        parts.append(toto_result_md(row) if game == "toto" else fourd_result_md(row))
        parts.append(f"Draw note: {note_link(draw_note_rel(game, number, day))}.")

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
        parts.append(md_table(["Game", "Draw date", "Numbers", "Bet type", "Cost", "Result", "Won"],
                              _ticket_rows(settled, fixes), align="llllrlr"))
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


def _md_section3(ctx: Context) -> str:
    parts = [SECTION_HEADINGS[2]]
    total, budget = 0.0, 0.0
    for game, plan in (("toto", ctx.toto_plan), ("4d", ctx.fourd_plan)):
        nd = next_draw(ctx, game)
        label = GAME_NAMES[game]
        when = f"draw {nd.number}, {fmt_date(nd.day)}" if nd.number else "the next draw"
        parts.append(f"### {label} for {when}")
        if nd.held:
            parts.append(_held_plan_text(nd))
            continue
        parts.append(plan_md(plan, game))
        if plan is not None:
            total += plan.total
            budget += plan.budget
        picks = ctx.toto_picks if game == "toto" else ctx.fourd_picks
        extra = picks_not_in_plan(plan, picks, game)
        if extra:
            if game == "toto":
                rows = [[clean_text(p.name), toto_nums(p.numbers), clean_text(p.reason)] for p in extra]
                head = ["Strategy", "Numbers", "Reason"]
            else:
                rows = [[clean_text(p.name), p.number, clean_text(p.bet_type), clean_text(p.reason)] for p in extra]
                head = ["Pick", "Number", "Bet type", "Reason"]
            parts.append("Also suggested, not in the plan:")
            parts.append(md_table(head, rows))
    parts.append(f"**Total for both games: {dollars(total)} of the {dollars(budget)} budget.**")
    return "\n\n".join(parts)


def _md_section4(ctx: Context) -> str:
    return "\n\n".join([
        SECTION_HEADINGS[3],
        "### TOTO",
        backtest_md(ctx.toto_backtest, "toto"),
        "### 4D",
        backtest_md(ctx.fourd_backtest, "4d"),
    ])


def _chi_row(label: str, result: dict) -> list[str]:
    """One chi square table row: test, statistic, degrees of freedom, p value (never scientific)."""
    p = as_float(result.get("p_value"))
    return [label, fmt_num(as_float(result.get("stat")), 1), fmt_num(as_int(result.get("dof"))),
            format_p(p if p is not None else float("nan"))]


def _chi_rows(ctx: Context) -> list[list[str]]:
    rows = []
    if ctx.toto_chi:
        rows.append(_chi_row("TOTO numbers", ctx.toto_chi))
    if ctx.fourd_chi:
        rows.append(_chi_row("4D digits, all positions", ctx.fourd_chi.get("overall") or {}))
        for pos, res in (ctx.fourd_chi.get("per_position") or {}).items():
            rows.append(_chi_row(f"4D digits, {pos} position", res or {}))
    return rows


def _crowd_md(ctx: Context) -> str:
    diag = ctx.crowd_diag or {}
    parts = ["### Crowd score"]
    if not diag:
        parts.append("Crowd scores were not worked out in this run.")
        return "\n\n".join(parts)
    rows = [
        ["Draws used", fmt_num(as_int(diag.get("n_draws")))],
        ["Draws left out", fmt_num(as_int(diag.get("n_excluded")))],
        ["Average crowd ratio", fmt_num(as_float(diag.get("mean_ratio")), 2)],
        ["Median crowd ratio", fmt_num(as_float(diag.get("median_ratio")), 2)],
        ["Fit (R squared)", fmt_num(as_float(diag.get("r2")), 2)],
    ]
    parts.append(md_table(["Item", "Value"], rows, align="lr"))
    parts.append("The crowd ratio compares the real number of Group 7 winners with the number expected "
                 "from sales. A positive crowd score means many people buy that number, so its prizes "
                 "are shared more ways.")
    mean = as_float(diag.get("mean_ratio"))
    if not diag.get("ok", True) or diag.get("warning"):
        warning = diag.get("warning") or (
            f"Average crowd ratio is {fmt_num(mean, 2)}, not close to 1, so the "
            f"{pct(ctx.rules.pool_share_of_sales, 0)} prize pool assumption may be off.")
        parts.append(f"**Warning:** {clean_text(warning)}")
    elif mean is not None:
        parts.append(f"The average crowd ratio is {fmt_num(mean, 2)}, close to 1 as expected.")
    scores = ctx.crowd_scores
    if isinstance(scores, pd.Series) and len(scores) and float(scores.abs().sum()) > 0:
        ordered = scores.sort_values()
        most = [f"{int(n)} ({fmt_num(v, 2)})" for n, v in ordered.iloc[::-1].head(6).items()]
        least = [f"{int(n)} ({fmt_num(v, 2)})" for n, v in ordered.head(6).items()]
        parts.append(f"Most crowded: {', '.join(most)}. Least crowded: {', '.join(least)}.")
    return "\n\n".join(parts)


def _toto_stats_md(ctx: Context) -> list[str]:
    parts = []
    freq = ctx.toto_frequency
    if isinstance(freq, pd.DataFrame) and len(freq):
        overdue = ctx.toto_overdue if isinstance(ctx.toto_overdue, pd.Series) else None
        crowd = ctx.crowd_scores if isinstance(ctx.crowd_scores, pd.Series) else None
        rows = []
        for n in freq.index:
            rows.append([
                int(n), fmt_num(as_int(freq.at[n, "all"])), fmt_num(as_int(freq.at[n, "last100"])),
                fmt_num(as_int(freq.at[n, "last50"])),
                fmt_num(as_int(overdue.get(n))) if overdue is not None and n in overdue.index else "n/a",
                fmt_num(as_float(crowd.get(n)), 3) if crowd is not None and n in crowd.index else "n/a",
            ])
        parts.append("### TOTO numbers")
        parts.append(md_table(["Number", "All draws", "Last 100", "Last 50", "Draws since seen", "Crowd score"],
                              rows, align="rrrrrr"))
    if ctx.toto_pairs:
        parts.append("### Most common TOTO pairs")
        parts.append(md_table(["Pair", "Times drawn together"],
                              [[f"{a} and {b}", fmt_num(c)] for (a, b), c in ctx.toto_pairs], align="lr"))
    shape = ctx.toto_shape or {}
    if shape:
        odd, low = shape.get("odd_dist") or {}, shape.get("low_dist") or {}
        rows = [[k, pct(as_float(odd.get(k, odd.get(str(k)))), 1), pct(as_float(low.get(k, low.get(str(k)))), 1)]
                for k in range(C.TOTO_PICK + 1)]
        parts.append("### Usual shape of a winning TOTO set")
        head = ["How many", "Share with that many odd", f"Share with that many low (1 to {C.TOTO_LOW_MAX})"]
        parts.append(md_table(head, rows, align="rrr"))
        parts.append(
            f"Most common: {shape.get('typical_odd', 'n/a')} odd and {shape.get('typical_low', 'n/a')} low numbers. "
            f"The middle 50% of winning sets add up to {fmt_num(as_float(shape.get('sum_q25')))} to "
            f"{fmt_num(as_float(shape.get('sum_q75')))} (median {fmt_num(as_float(shape.get('sum_median')))}), "
            f"over {plural(as_int(shape.get('n_draws')), 'draw')}.")
    return parts


def _fourd_stats_md(ctx: Context) -> list[str]:
    parts = []
    freq = ctx.fourd_position_freq
    if isinstance(freq, pd.DataFrame) and len(freq):
        rows = [[int(d)] + [fmt_num(as_int(freq.at[d, pos])) for pos in freq.columns] for d in freq.index]
        parts.append("### 4D digit frequency by position (all 23 prizes per draw)")
        parts.append(md_table(["Digit"] + [str(c).title() for c in freq.columns], rows,
                              align="r" * (len(freq.columns) + 1)))
    if ctx.fourd_repeats:
        parts.append("### 4D numbers that won more than once")
        parts.append(md_table(["Number", "Times won", "Last won"],
                              [[str(n), fmt_num(c), fmt_date(d)] for n, c, d in ctx.fourd_repeats], align="lrl"))
    if ctx.fourd_digit_sets:
        parts.append("### Most frequent 4D digit sets (any order)")
        parts.append(md_table(["Digit set", "Times won"],
                              [[" ".join(str(s)), fmt_num(c)] for s, c in ctx.fourd_digit_sets], align="lr"))
    values = _bet_values(ctx)
    if values.get("Big") is not None:
        rows = []
        for bet in BET_TYPES:
            v = values.get(bet)
            if isinstance(v, dict):
                for perms in sorted(v, key=lambda k: int(k)):
                    rows.append([f"{bet}, {perms} permutations", per_dollar(as_float(v[perms]))])
            elif v is not None:
                rows.append([bet, per_dollar(as_float(v))])
        parts.append("### 4D bet type value")
        parts.append(md_table(["Bet type", "Average return per $1"], rows, align="lr"))
        best = clean_text(values.get("best") or "n/a")
        parts.append(f"Best bet type: **{best}**. {clean_text(values.get('explanation'))}")
        if values.get("ibet_note"):
            parts.append(clean_text(values["ibet_note"]))
    return parts


def _rules_md(rules: PrizeRules) -> str:
    def status(ok: bool) -> str:
        return "confirmed on the official prize page" if ok else "not confirmed, built in values used"

    rows = [["TOTO", status(rules.toto_confirmed)], ["4D", status(rules.fourd_confirmed)]]
    checked = "n/a"
    if rules.checked_at:
        try:
            checked = fmt_datetime(datetime.fromisoformat(str(rules.checked_at)))
        except ValueError:
            checked = clean_text(rules.checked_at)
    parts = [
        "### Prize rules used",
        md_table(["Game", "Prize rules"], rows, align="ll"),
        f"{sentence(rules.source_note) or 'Built in values.'} Last checked: {checked}.",
        f"TOTO: prize pool {pct(rules.pool_share_of_sales, 0)} of sales; Group 1 "
        f"{pct(rules.group_pool_pct.get(1), 0)} (at least {money(rules.min_group1)}), Group 2 "
        f"{pct(rules.group_pool_pct.get(2), 0)}, Group 3 {pct(rules.group_pool_pct.get(3), 1)}, Group 4 "
        f"{pct(rules.group_pool_pct.get(4), 0)}; Group 5 {money(rules.fixed_prizes.get(5))}, Group 6 "
        f"{money(rules.fixed_prizes.get(6))}, Group 7 {money(rules.fixed_prizes.get(7))}.",
    ]
    return "\n\n".join(parts)


def _md_section5(ctx: Context) -> str:
    parts = [SECTION_HEADINGS[4]]
    parts.append("### Fairness tests (chi square)")
    rows = _chi_rows(ctx)
    if rows:
        parts.append(md_table(["Test", "Chi square", "Degrees of freedom", "p value"], rows, align="lrrr"))
    verdicts = []
    if (ctx.toto_chi or {}).get("verdict"):
        verdicts.append(f"* TOTO: {clean_text(ctx.toto_chi['verdict'])}")
    if (ctx.fourd_chi or {}).get("verdict"):
        verdicts.append(f"* 4D: {clean_text(ctx.fourd_chi['verdict'])}")
    parts.append("\n".join(verdicts) if verdicts else "The fairness tests were not run in this run.")
    parts.append(_crowd_md(ctx))
    parts.extend(_toto_stats_md(ctx))
    parts.extend(_fourd_stats_md(ctx))
    parts.append(_rules_md(ctx.rules))
    return "\n\n".join(parts)


def report_warnings(ctx: Context) -> list[str]:
    """Run warnings, settings problems, the crowd ratio warning and unconfirmed prize rules,
    each once, dash free."""
    items: list[str] = []
    for w in list(ctx.warnings or []) + list(getattr(ctx.settings, "warnings", []) or []):
        items.append(clean_text(w))
    diag = ctx.crowd_diag or {}
    if diag.get("warning"):
        items.append(clean_text(diag["warning"]))
    if not ctx.rules.confirmed:
        missing = [g for g, ok in (("TOTO", ctx.rules.toto_confirmed), ("4D", ctx.rules.fourd_confirmed)) if not ok]
        items.append(f"The {_and_list(missing)} prize rules could not be confirmed on the official pages, "
                     "so built in values were used.")
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _stale_text(ctx: Context) -> str | None:
    """Plain notice when the stored results of a game are behind (draws were held after the
    newest one), or None when they are up to date. A game this run did not fetch (not in
    ``ctx.games_drawn``) gets its own sentence: its newer draws come with its next run."""
    behind, others = [], []
    for game in GAME_NAMES:
        last_no, _ = _last_stored(ctx, game)
        if not last_no or not next_draw(ctx, game).stale:
            continue
        if game in ctx.games_drawn:
            behind.append(f"{GAME_NAMES[game]} draw {last_no}")
        else:
            others.append(f"{GAME_NAMES[game]} results are not up to date: the newest stored draw is {last_no}, "
                          f"and newer draws have been held since. They are fetched on the next run that "
                          f"covers {GAME_NAMES[game]}.")
    parts = []
    if behind:
        parts.append(f"Results not up to date: the newest stored {'results are' if len(behind) > 1 else 'result is'} "
                     f"{_and_list(behind)}, and newer draws have been held since. The Singapore Pools site could "
                     "not be read or did not have every result yet, or fetching was turned off.")
    return " ".join(parts + others) or None


def full_report(ctx: Context) -> str:
    """The run report as Obsidian markdown, in the order the analyst prompt asks for."""
    title = f"# Huat Bot report, {fmt_datetime(to_sg(ctx.now))}"
    parts = [title]
    drawn = [g for g in ctx.games_drawn if g in GAME_NAMES]
    parts.append(f"Games reported in this run: {games_text(drawn)}." if drawn else "No new draws in this run.")
    stale = _stale_text(ctx)
    if stale:
        parts.append(f"**{stale}**")
    held = held_text(ctx)
    if held:
        parts.append(f"**{held}**")
    warnings = report_warnings(ctx)
    if warnings:
        parts.append("> [!warning] Warnings\n" + "\n".join(f"> * {w}" for w in warnings))
    builders: list[Callable[[Context], str]] = [_md_section1, _md_section2, _md_section3, _md_section4,
                                                _md_section5]
    for build in builders:
        parts.append(build(ctx))
    parts.append(f"{SECTION_HEADINGS[5]}\n\n{odds_note(ctx.rules)}")
    comment = commentary_text(ctx.commentary)
    if comment:
        parts.append(f"*Commentary:* {comment}")
    return "\n\n".join(parts).rstrip() + "\n"


# Telegram


def _tg_join(blocks: Sequence[str]) -> str:
    return "\n\n".join(b for b in blocks if b)


def _fit(build: Callable[[int], list[str]]) -> str:
    """The most detailed version of a message that fits the Telegram limit.

    ``build(level)`` returns whole, self contained HTML blocks; higher levels carry less
    detail. If even the last level is too long, whole blocks are dropped from the end and a
    pointer to the vault report is added, so a tag is never cut.
    """
    for level in range(_TG_LEVELS):
        text = _tg_join(build(level))
        if len(text) <= LIMIT:
            return _scrub(text)
    blocks = build(_TG_LEVELS - 1)
    while blocks and len(_tg_join(blocks + [_TG_MORE])) > LIMIT:
        blocks.pop()
    return _scrub(_tg_join(blocks + [_TG_MORE]))


def _scrub(text: str) -> str:
    """Final safety net: Telegram messages carry no dash at all."""
    if contains_dash(text):
        log.warning("A Telegram message still held a dash; removing it")
        text = remove_dashes(text)
    return text


def _tg_toto_result(ctx: Context) -> str:
    row = latest_row(ctx.toto)
    if row is None:
        return "<b>TOTO</b>\nNo TOTO result is stored yet."
    number, day = as_int(field(row, "draw_number")), as_date(field(row, "draw_date"))
    dtype = str(field(row, "draw_type", "normal"))
    kind = f" ({_h(draw_type_name(dtype))} draw)" if dtype != "normal" else ""
    lines = [
        f"<b>TOTO draw {number}</b>, {fmt_date(day)}{kind}",
        f"Winning numbers: <b>{toto_nums(toto_numbers(row))}</b>",
        f"Additional number: <b>{as_int(field(row, 'additional'))}</b>",
        "Group 1 prize: <b>{}</b>, {}".format(*(_h(x) for x in _group1_parts(row))),
        pre_block(["Group", "Share", "Winners"], _group_rows(row), "lrr"),
    ]
    return "\n".join(lines)


def _tg_fourd_result(ctx: Context) -> str:
    row = latest_row(ctx.fourd)
    if row is None:
        return "<b>4D</b>\nNo 4D result is stored yet."
    number, day = as_int(field(row, "draw_number")), as_date(field(row, "draw_date"))
    nums = fourd_numbers(row)
    lines = [f"<b>4D draw {number}</b>, {fmt_date(day)}"]
    for tier in ("first", "second", "third"):
        lines.append(f"{prizes.FOURD_TIER_LABELS[tier]}: <b>{_h(' '.join(nums[tier]) or 'n/a')}</b>")
    grid = []
    for tier in ("starter", "consolation"):
        values = nums[tier]
        grid.append(prizes.FOURD_TIER_LABELS[tier])
        grid += ["  ".join(values[i:i + 5]) for i in range(0, len(values), 5)] or ["n/a"]
    lines.append(f"<pre>{html_escape(chr(10).join(grid))}</pre>")
    return "\n".join(lines)


def _ticket_line(r: dict, was: dict | None = None) -> str:
    won = as_float(r.get("winnings")) or 0.0
    head = (f"{_h(r.get('game'))} {fmt_date(r.get('draw_date'))}, {_h(r.get('numbers'))}, "
            f"{_h(r.get('bet_type'))} {dollars(r.get('cost'))}")
    fix = f", {_h(_correction_note(was))}" if was is not None else ""
    if won > 0:
        return f"{head}: {_h(r.get('result'))}{fix}, {'now won' if fix else 'won'} <b>{dollars(won)}</b>"
    return f"{head}: {_h(r.get('result') or 'No prize')}{fix}"


def _no_draw_cost(ctx: Context) -> float:
    """Cost of the ledger tickets whose date had no draw (status "no_draw")."""
    ledger = ctx.ledger
    if not isinstance(ledger, pd.DataFrame) or ledger.empty or "status" not in ledger.columns:
        return 0.0
    cost = pd.to_numeric(ledger.loc[ledger["status"] == "no_draw", "cost"], errors="coerce")
    return float(cost.fillna(0.0).sum())


def _tg_tickets(ctx: Context, level: int) -> str:
    lines = ["<b>My tickets</b>"]
    settled = list(ctx.settled_this_run or [])
    fixes = corrections(ctx)
    winners = sorted((r for r in settled if (as_float(r.get("winnings")) or 0.0) > 0),
                     key=lambda r: -(as_float(r.get("winnings")) or 0.0))
    losers = [r for r in settled if (as_float(r.get("winnings")) or 0.0) <= 0]
    total_won, n_won = _run_winnings(settled, fixes)
    if not settled:
        lines.append("No tickets were checked in this run.")
    elif level == 0:
        lines += [f"• {_ticket_line(r, _was(fixes, r))}" for r in winners + losers]
    elif level <= 2:
        # A corrected ticket is always listed: it changes a result already announced.
        shown = [r for k, r in enumerate(winners) if level == 1 or k < _TG_WINNERS_CAP or _was(fixes, r)]
        fixed_losers = [r for r in losers if _was(fixes, r)]
        lines += [f"• {_ticket_line(r, _was(fixes, r))}" for r in shown + fixed_losers]
        if len(shown) < len(winners):
            lines.append(f"• and {plural(len(winners) - len(shown), 'more winning ticket')}")
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
        lines.append(f"All tickets so far: spent {dollars(totals.get('spent'))}, won "
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
    if won > 0:
        head = (f"<b>WINNER! Your tickets won {dollars(won)} in this run</b>\n"
                f"{plural(n, 'winning ticket')}, details below.")
    else:
        head = f"<b>Huat Bot results</b>, {fmt_date(to_sg(ctx.now))}"
    blocks = [head]
    stale = _stale_text(ctx)
    if stale:
        blocks.append(f"<i>{_h(stale)}</i>")
    held = held_text(ctx)
    if held:
        blocks.append(f"<i>{_h(held)}</i>")
    drawn = [g for g in ctx.games_drawn if g in GAME_NAMES]
    if "toto" in drawn:
        blocks.append(_tg_toto_result(ctx))
    if "4d" in drawn:
        blocks.append(_tg_fourd_result(ctx))
    if not drawn:
        blocks.append("No new draw results in this run.")
    blocks.append(_tg_tickets(ctx, level))
    return blocks


def _tg_message2(ctx: Context, level: int) -> list[str]:
    sig = toto_signal(ctx)
    nd = next_draw(ctx, "toto")
    draw = f" (draw {nd.number})" if nd.number else ""
    special = str(sig["draw_type"]) != "normal"
    dtype = draw_type_name(sig["draw_type"])
    if nd.held:
        lines = [f"<b>Next TOTO draw</b>{draw}: {_h(nd.when_text)}",
                 "Its sales are closed, so there is no buy signal for it."]
        return ["\n".join(lines), _tg_fourd_next(ctx)]
    lines = [
        f"<b>Next TOTO draw</b>{draw}: {_h(nd.when_text)}",
        "Estimated jackpot: " + (f"<b>{money(sig['jackpot'])}</b>" if sig["jackpot"] is not None
                                 else "not available yet"),
        "Draw type: " + (f"<b>{_h(dtype)} draw</b>" if special else _h(dtype)),
        f"Draws in a row with no Group 1 winner: {fmt_num(sig['streak'])}",
        f"Buy signal: <b>{_h(sig['label'] or 'not available')}</b>",
    ]
    if sig["reason"] and level < 3:
        lines.append(f"<i>{_h(sig['reason'])}</i>")
    if sig["ev"] is not None:
        lines.append(f"Return per $1: <b>{per_dollar(sig['ev'])}</b> on average")
        ev_rows = _ev_rows(sig["breakdown"])
        if ev_rows and level < 4:
            lines.append(pre_block(["Part", "Per $1"],
                                   [[r[0].replace(" (fixed prizes)", ""), r[1]] for r in ev_rows], "lr"))
    else:
        lines.append("Return per $1: not available")
    if sig["boards"] is not None and level < 2:
        lines.append(f"Sales estimate: about {fmt_num(sig['boards'])} boards ({_h(sig['boards_method'])}).")
    return ["\n".join(lines), _tg_fourd_next(ctx)]


def _tg_fourd_next(ctx: Context) -> str:
    nd4 = next_draw(ctx, "4d")
    draw4 = f" (draw {nd4.number})" if nd4.number else ""
    lines4 = [
        f"<b>Next 4D draw</b>{draw4}: {_h(nd4.when_text)}",
        "Prize for each $1:",
        pre_block(["Prize", "Big", "Small"], _prize_table_rows(ctx.rules), "lrr"),
    ]
    values = _bet_values(ctx)
    if values.get("Big") is not None:
        lines4.append(f"Return per $1: Big <b>{per_dollar(values.get('Big'))}</b>, Small "
                      f"<b>{per_dollar(values.get('Small'))}</b>. {_h(values.get('best') or 'Big')} gives "
                      "the most back.")
    return "\n".join(lines4)


def _tg_plan(plan: Plan | None, game: str, nd: NextDraw, picks: Sequence[Any], level: int) -> str:
    label = GAME_NAMES[game]
    when = f"draw {nd.number}, {fmt_date(nd.day)}" if nd.number else "next draw"
    if nd.held:
        return f"<b>{label} {when}</b>\n{_h(_held_plan_text(nd))}"
    if plan is None:
        return f"<b>{label}</b> ({when})\nNo {label} plan this run."
    lines = [f"<b>{label} {when}</b>, budget {dollars(plan.budget)}"]
    if plan.lines:
        if game == "toto":
            rows = [[ln.label, ln.numbers, dollars(ln.cost)] for ln in plan.lines]
            lines.append(pre_block(["Set", "Numbers", "Cost"], rows, "llr"))
        else:
            rows = [[ln.label, ln.numbers, ln.bet_type, dollars(ln.cost)] for ln in plan.lines]
            lines.append(pre_block(["Pick", "Number", "Bet", "Stake"], rows, "lllr"))
        if level == 0:
            lines += [f"• {_h(ln.label)}: {_h(ln.reason)}" for ln in plan.lines if ln.reason]
    else:
        lines += [_h(n) for n in plan.notes[:1]] or ["Nothing to buy."]
    alt = plan.alternative
    if alt is not None and alt.lines and level < 3:
        sys7 = next((ln for ln in alt.lines if ln.label == "System 7"), None)
        others = [ln.label for ln in alt.lines if ln is not sys7]
        if sys7 is not None:
            extra = f" plus {_h(_and_list(others))}" if others else ""
            lines.append(f"System 7 option: <b>{_h(sys7.numbers)}</b> for {dollars(sys7.cost)}{extra}, "
                         f"{dollars(alt.total)} in total, instead of the sets above.")
    elif plan.lines and not any(ln.label == "System 7" for ln in plan.lines) and level < 2:
        sys_note = next((n for n in plan.notes if "System 7" in n), None)
        if sys_note:
            lines.append(_h(sys_note))
    if plan.lines:
        lines.append(f"{label} total: <b>{dollars(plan.total)}</b> of your {dollars(plan.budget)} budget")
    unbought = picks_not_in_plan(plan, picks, game)
    if unbought and level < 2:
        lines.append(f"Not in the plan (details in the report): {_h(_and_list([p.name for p in unbought]))}.")
    return "\n".join(lines)


def _tg_message3(ctx: Context, level: int) -> list[str]:
    blocks = ["<b>Suggested numbers</b>"]
    total = budget = 0.0
    for game, plan, picks in (("toto", ctx.toto_plan, ctx.toto_picks), ("4d", ctx.fourd_plan, ctx.fourd_picks)):
        nd = next_draw(ctx, game)
        blocks.append(_tg_plan(plan, game, nd, picks, level))
        if plan is not None and not nd.held:
            total += plan.total
            budget += plan.budget
    blocks.append(f"Total for both games: <b>{dollars(total)}</b> of your {dollars(budget)} budget.")
    bt = [backtest_summary(ctx.toto_backtest, "toto"), backtest_summary(ctx.fourd_backtest, "4d")]
    bt = [b for b in bt if b]
    if bt and level < 4:
        blocks.append("<b>Backtest</b>\n" + "\n".join(_h(b) for b in bt))
    blocks.append(f"<i>{_h(odds_line())}</i>")
    comment = commentary_text(ctx.commentary)
    if comment and level < 4:
        blocks.append(f"<i>{html_escape(comment)}</i>")
    return blocks


def telegram_messages(ctx: Context) -> list[str]:
    """Exactly three Telegram HTML messages: results and tickets, next draws and buy signal,
    suggested numbers and cost. Each fits ``constants.TELEGRAM_MAX_CHARS`` and has no dash."""
    return [
        _fit(lambda level: _tg_message1(ctx, level)),
        _fit(lambda level: _tg_message2(ctx, level)),
        _fit(lambda level: _tg_message3(ctx, level)),
    ]
