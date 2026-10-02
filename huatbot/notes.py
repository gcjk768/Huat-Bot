"""Obsidian notes the bot writes into the vault on the NAS.

Each builder returns ``(relative path, frontmatter, body)``; ``write_all`` hands them to
``Vault.write_note`` (atomic, skipped when unchanged) and logs every note it actually wrote as
a NOTE event in this month's activity log.

The notes are linked into one small graph:

* ``Dashboard.md`` links to the newest report, ``Ledger``, ``Tickets``, ``Settings``, this
  month's activity log, the latest draw notes and the suggestion notes of the upcoming draws.
  Draw and suggestion notes are linked with their folder (``report.note_link``), because the
  suggestion note made before a draw and that draw's note share a file name.
* Draw, suggestion and report notes link back to ``[[Dashboard]]``.

Frontmatter uses plain Obsidian properties: ``tags`` (always ``huatbot`` plus the kind of note),
``game``, ``draw``, ``date`` (ISO) and the figures of that note. Bodies follow the house style:
no dashes in prose (wikilinks with ISO dates and table separator rows are fine).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from .models import LEDGER_COLUMNS, Context
from .report import (
    GAME_NAMES,
    NextDraw,
    activity_note_name,
    as_date,
    as_float,
    as_int,
    backtest_summary,
    clean_text,
    dollars,
    draw_note_name,
    draw_note_rel,
    draw_type_name,
    field,
    fourd_result_md,
    latest_row,
    link,
    next_draw,
    note_link,
    odds_line,
    plan_md,
    report_note_name,
    report_warnings,
    sentence,
    shape_text,
    to_sg,
    toto_result_md,
    toto_signal,
)
from .store import fourd_numbers, toto_numbers
from .textfmt import (
    fmt_date,
    fmt_datetime,
    md_table,
    money,
    per_dollar,
    plural,
    toto_nums,
)

log = logging.getLogger(__name__)

DASHBOARD = "Dashboard"
LEDGER = "Ledger"
TICKETS = "Tickets"
SETTINGS = "Settings"

SUGGESTION_FOLDER = "Suggestions"
REPORT_FOLDER = "Reports"

STATUS_TEXT = {
    "settled": "Checked",
    "pending": "Waiting for the draw",
    "no_draw": "No draw that day",
    "invalid": "Not counted",
}

NoteSpec = tuple[str, dict, str]


# Helpers


def _iso(day: Any) -> str | None:
    d = as_date(day)
    return d.isoformat() if d else None


def _stamp(now: datetime) -> str:
    """ISO time of the run in Singapore time (frontmatter only)."""
    return to_sg(now).isoformat(timespec="seconds")


def _code(text: Any) -> str:
    """Inline code span for raw user text (may hold dashes, backticks or pipes)."""
    s = " ".join(str(text or "").split())
    if not s:
        return ""
    ticks = "``" if "`" in s else "`"
    return f"{ticks} {s} {ticks}" if ticks == "``" else f"`{s}`"


def suggestion_note_path(nd: NextDraw) -> str | None:
    """ "Suggestions/2026-10-05 TOTO 4124.md", or None when the next draw number is unknown."""
    if not nd.number:
        return None
    return f"{SUGGESTION_FOLDER}/{draw_note_name(nd.game, nd.number, nd.day)}.md"


def report_note_path(now: datetime) -> str:
    return f"{REPORT_FOLDER}/{report_note_name(now)}.md"


def draw_note_path(game: str, row: Any) -> str:
    return draw_note_rel(game, as_int(field(row, "draw_number")), as_date(field(row, "draw_date")))


# Draw notes


def toto_draw_note(row: Any) -> NoteSpec:
    """("Draws/TOTO/2026-10-01 TOTO 4123.md", frontmatter, body) for one TOTO result row."""
    number = as_int(field(row, "draw_number"))
    day = as_date(field(row, "draw_date"))
    nums = toto_numbers(row)
    jackpot = as_float(field(row, "jackpot"))
    frontmatter = {
        "tags": ["huatbot", "toto", "draw"],
        "game": "TOTO",
        "draw": number,
        "date": _iso(day),
        "numbers": nums,
        "additional": as_int(field(row, "additional")),
        "draw_type": str(field(row, "draw_type", "normal")),
        "group1_prize": jackpot,
        "group1_winners": as_int(field(row, "g1_winners")),
    }
    body = "\n\n".join([
        f"# TOTO draw {number}, {fmt_date(day)}",
        f"Back to {link(DASHBOARD)}.",
        toto_result_md(row),
        f"Shape of this set: {shape_text(nums)}.",
    ])
    return draw_note_path("toto", row), frontmatter, body


def fourd_draw_note(row: Any) -> NoteSpec:
    """("Draws/4D/2026-09-30 4D 5432.md", frontmatter, body) for one 4D result row."""
    number = as_int(field(row, "draw_number"))
    day = as_date(field(row, "draw_date"))
    nums = fourd_numbers(row)
    # 4D numbers go in as space separated text: a lone "0698" would be read as the number 698
    # by YAML 1.2 readers such as Obsidian, while "2751 6755 9745" is text in every reader.
    frontmatter = {
        "tags": ["huatbot", "4d", "draw"],
        "game": "4D",
        "draw": number,
        "date": _iso(day),
        "top_three": " ".join(nums["first"] + nums["second"] + nums["third"]),
        "starter": " ".join(nums["starter"]),
        "consolation": " ".join(nums["consolation"]),
    }
    body = "\n\n".join([
        f"# 4D draw {number}, {fmt_date(day)}",
        f"Back to {link(DASHBOARD)}.",
        fourd_result_md(row),
    ])
    return draw_note_path("4d", row), frontmatter, body


# Dashboard


def _next_rows(ctx: Context) -> list[list[str]]:
    sig = toto_signal(ctx)
    rows = []
    for game in ("toto", "4d"):
        nd = next_draw(ctx, game)
        path = suggestion_note_path(nd)
        if game == "toto":
            jackpot = sig["jackpot"]
            prize = f"{money(jackpot)} estimated jackpot" if jackpot is not None else "jackpot not announced yet"
            kind = draw_type_name(sig["draw_type"])
        else:
            first = as_float(ctx.rules.fourd_prizes.get("big", {}).get("first"))
            prize = f"Big 1st Prize {money(first)} per $1"
            kind = "Normal"
        rows.append([GAME_NAMES[game], str(nd.number or "n/a"), nd.when_text, kind, prize,
                     note_link(path) if path else "n/a"])
    return rows


def _latest_rows(ctx: Context) -> list[list[str]]:
    rows = []
    for game in ("toto", "4d"):
        row = latest_row(ctx.toto if game == "toto" else ctx.fourd)
        if row is None:
            continue
        number, day = as_int(field(row, "draw_number")), as_date(field(row, "draw_date"))
        if game == "toto":
            result = f"{toto_nums(toto_numbers(row))}, additional {as_int(field(row, 'additional'))}"
            winners = as_int(field(row, "g1_winners"))
            extra = (f"Group 1 {money(as_float(field(row, 'jackpot')))}, "
                     f"{plural(winners, 'winner') if winners else 'no winner'}")
        else:
            nums = fourd_numbers(row)
            result = ", ".join(f"{label} {(nums[t] or ['n/a'])[0]}"
                               for t, label in (("first", "1st"), ("second", "2nd"), ("third", "3rd")))
            extra = "All 23 numbers in the draw note"
        rows.append([GAME_NAMES[game], str(number), fmt_date(day), result, extra,
                     note_link(draw_note_rel(game, number, day))])
    return rows


def _plan_rows(ctx: Context) -> list[list[str]]:
    rows = []
    for plan in (ctx.toto_plan, ctx.fourd_plan):
        if plan is None:
            continue
        for ln in plan.lines:
            rows.append([ln.game, clean_text(ln.label), ln.numbers, clean_text(ln.bet_type), dollars(ln.cost)])
    return rows


def _totals_row(totals: dict) -> list[str]:
    return [str(as_int(totals.get("tickets"))), dollars(totals.get("spent") or 0.0),
            dollars(totals.get("won") or 0.0), dollars(totals.get("net") or 0.0),
            str(as_int(totals.get("pending"))), dollars(totals.get("pending_cost") or 0.0)]


def dashboard_note(ctx: Context) -> NoteSpec:
    """("Dashboard.md", frontmatter, body): next draws, buy signal, latest results, plan, ledger
    totals and links to the newest report, ledger, tickets, settings, log and suggestion notes."""
    sig = toto_signal(ctx)
    nd_toto, nd_4d = next_draw(ctx, "toto"), next_draw(ctx, "4d")
    totals = ctx.ledger_totals or {}
    report = report_note_name(ctx.now)
    suggestion_links = [note_link(p) for p in (suggestion_note_path(nd_toto), suggestion_note_path(nd_4d)) if p]
    latest_toto, latest_4d = latest_row(ctx.toto), latest_row(ctx.fourd)

    frontmatter = {
        "tags": ["huatbot", "dashboard"],
        "updated": _stamp(ctx.now),
        "next_toto_draw": nd_toto.number,
        "next_toto_date": nd_toto.day.isoformat() if nd_toto.day else None,
        "next_toto_jackpot": sig["jackpot"],
        "next_toto_draw_type": sig["draw_type"],
        "buy_signal": sig["label"],
        "return_per_dollar": round(sig["ev"], 4) if sig["ev"] is not None else None,
        "next_4d_draw": nd_4d.number,
        "next_4d_date": nd_4d.day.isoformat() if nd_4d.day else None,
        "latest_toto_draw": as_int(field(latest_toto, "draw_number")) if latest_toto is not None else None,
        "latest_4d_draw": as_int(field(latest_4d, "draw_number")) if latest_4d is not None else None,
        "spent": float(totals.get("spent") or 0.0),
        "won": float(totals.get("won") or 0.0),
        "net": float(totals.get("net") or 0.0),
    }

    parts = [
        "# Huat Bot dashboard",
        f"Updated {fmt_datetime(to_sg(ctx.now))}. Newest report: {link(report)}. "
        f"Activity log: {link(activity_note_name(ctx.now))}.",
        f"Your notes: {link(TICKETS)} (add tickets here), {link(LEDGER)} (every ticket and the totals), "
        f"{link(SETTINGS)} (budgets and alerts).",
    ]
    if suggestion_links:
        parts.append(f"Suggestions for the upcoming draws: {', '.join(suggestion_links)}.")

    parts += ["## Next draws",
              md_table(["Game", "Draw", "Date and time", "Draw type", "Prize", "Suggestions"], _next_rows(ctx))]

    parts.append("## Buy signal")
    if sig["label"]:
        ev = f" Return per $1: about {per_dollar(sig['ev'])}." if sig["ev"] is not None else ""
        parts.append(f"**{sig['label']}**. {sentence(sig['reason'])}{ev}".rstrip())
    else:
        parts.append("Not worked out in this run.")
    parts.append(f"Draws in a row with no Group 1 winner: {sig['streak']}.")

    parts.append("## Latest results")
    latest = _latest_rows(ctx)
    parts.append(md_table(["Game", "Draw", "Date", "Winning numbers", "Prizes", "Note"], latest)
                 if latest else "No results are stored yet.")

    parts.append("## Suggested purchases")
    plan_rows = _plan_rows(ctx)
    if plan_rows:
        parts.append(md_table(["Game", "Strategy", "Numbers", "Bet type", "Cost"], plan_rows, align="llllr"))
        budget = sum(p.budget for p in (ctx.toto_plan, ctx.fourd_plan) if p is not None)
        total = sum(p.total for p in (ctx.toto_plan, ctx.fourd_plan) if p is not None)
        parts.append(f"Total {dollars(total)} of the {dollars(budget)} budget. Reasons and options are in the "
                     "suggestion notes.")
    else:
        parts.append("Nothing to buy in this run.")

    parts.append("## My tickets")
    parts.append(md_table(["Tickets", "Spent", "Won", "Net", "Waiting", "Waiting cost"], [_totals_row(totals)],
                          align="rrrrrr"))
    settled = list(ctx.settled_this_run or [])
    if settled:
        won = sum(as_float(r.get("winnings")) or 0.0 for r in settled)
        parts.append(f"Checked in this run: {plural(len(settled), 'ticket')}, won {dollars(won)}. "
                     f"Details in {link(LEDGER)}.")
    if ctx.bad_ticket_lines:
        parts.append(f"{plural(len(ctx.bad_ticket_lines), 'line')} in {link(TICKETS)} could not be read, "
                     f"see {link(LEDGER)}.")

    warnings = report_warnings(ctx)
    if warnings:
        parts.append("## Warnings")
        parts.append("\n".join(f"* {w}" for w in warnings))
    return "Dashboard.md", frontmatter, "\n\n".join(parts)


# Suggestions


def _suggestion_note(ctx: Context, game: str) -> NoteSpec | None:
    nd = next_draw(ctx, game)
    path = suggestion_note_path(nd)
    if path is None:
        return None
    label = GAME_NAMES[game]
    plan = ctx.toto_plan if game == "toto" else ctx.fourd_plan
    picks = ctx.toto_picks if game == "toto" else ctx.fourd_picks
    backtest = ctx.toto_backtest if game == "toto" else ctx.fourd_backtest

    frontmatter: dict[str, Any] = {
        "tags": ["huatbot", "suggestion", game],
        "game": label,
        "draw": nd.number,
        "date": nd.day.isoformat() if nd.day else None,
        "budget": plan.budget if plan is not None else None,
        "total_cost": plan.total if plan is not None else None,
    }
    if game == "toto":
        frontmatter["sets"] = [toto_nums(p.numbers) for p in picks]
        frontmatter["buy_signal"] = toto_signal(ctx)["label"]
    else:
        frontmatter["numbers"] = " ".join(p.number for p in picks)  # text, see fourd_draw_note
    frontmatter["generated"] = _stamp(ctx.now)

    parts = [
        f"# {label} suggestions for draw {nd.number}",
        f"{nd.when_text}. Back to {link(DASHBOARD)}. Full analysis: {link(report_note_name(ctx.now))}.",
        "## What to buy",
        plan_md(plan, game),
    ]
    bought = {ln.label for ln in plan.lines} if plan is not None else set()
    others = [p for p in picks if p.name not in bought]
    if others:
        parts.append("## Other suggestions")
        if game == "toto":
            rows = [[clean_text(p.name), toto_nums(p.numbers), clean_text(p.reason)] for p in others]
            parts.append(md_table(["Strategy", "Numbers", "Reason"], rows))
        else:
            rows = [[clean_text(p.name), p.number, clean_text(p.bet_type), clean_text(p.reason)] for p in others]
            parts.append(md_table(["Pick", "Number", "Bet type", "Reason"], rows))
    if game == "toto":
        sig = toto_signal(ctx)
        parts.append("## Buy signal")
        jackpot = money(sig["jackpot"]) if sig["jackpot"] is not None else "not announced yet"
        parts.append(f"**{sig['label'] or 'n/a'}**. Estimated jackpot {jackpot}, "
                     f"{draw_type_name(sig['draw_type']).lower()} draw. {sentence(sig['reason'])}".rstrip())
    summary = backtest_summary(backtest, game)
    if summary:
        parts.append("## Backtest")
        parts.append(clean_text(summary))
    parts.append(f"Bought any of these? Add them to {link(TICKETS)} so the bot can check them. {odds_line()}")
    return path, frontmatter, "\n\n".join(parts)


def suggestions_notes(ctx: Context) -> list[NoteSpec]:
    """Suggestion notes for the upcoming TOTO and 4D draws (games without data are skipped)."""
    notes = []
    for game in ("toto", "4d"):
        try:
            note = _suggestion_note(ctx, game)
        except Exception as exc:  # one broken note must not stop the others
            log.warning("Could not build the %s suggestion note: %s", GAME_NAMES[game], exc)
            continue
        if note is not None:
            notes.append(note)
    return notes


# Ledger


def _ledger_rows(ledger: pd.DataFrame) -> list[list[str]]:
    df = ledger.copy()
    df["_order"] = range(len(df))
    df = df.sort_values(["draw_date", "_order"], ascending=[False, True])
    rows = []
    for _, r in df.iterrows():
        status = str(r.get("status") or "pending")
        draw = r.get("draw_number")
        rows.append([
            clean_text(r.get("game")), fmt_date(r.get("draw_date") or None),
            "" if draw is None or pd.isna(draw) else str(int(draw)),
            clean_text(r.get("numbers")), clean_text(r.get("bet_type")), dollars(r.get("cost")),
            STATUS_TEXT.get(status, status), clean_text(r.get("result")),
            dollars(r.get("winnings") or 0.0) if status == "settled" else "",
        ])
    return rows


def ledger_note(ctx: Context) -> NoteSpec:
    """("Ledger.md", frontmatter, body): totals, every ticket and the lines that could not be read."""
    totals = ctx.ledger_totals or {}
    ledger = ctx.ledger if isinstance(ctx.ledger, pd.DataFrame) else pd.DataFrame(columns=LEDGER_COLUMNS)
    frontmatter = {
        "tags": ["huatbot", "ledger"],
        "updated": _stamp(ctx.now),
        "tickets": as_int(totals.get("tickets")),
        "spent": float(totals.get("spent") or 0.0),
        "won": float(totals.get("won") or 0.0),
        "net": float(totals.get("net") or 0.0),
        "pending_cost": float(totals.get("pending_cost") or 0.0),
    }
    parts = [
        "# My ticket ledger",
        f"Back to {link(DASHBOARD)}. Add or fix tickets in {link(TICKETS)}; the bot updates this note on "
        "every run.",
        "## Totals",
        md_table(["Tickets", "Spent", "Won", "Net", "Waiting", "Waiting cost"], [_totals_row(totals)],
                 align="rrrrrr"),
        "Spent counts every ticket that is not marked Not counted. Won counts checked tickets only.",
        "## Tickets",
    ]
    if len(ledger):
        parts.append(md_table(["Game", "Draw date", "Draw", "Numbers", "Bet type", "Cost", "Status", "Result",
                               "Won"], _ledger_rows(ledger), align="lllllrllr"))
    else:
        parts.append(f"No tickets yet. Add the tickets you buy to {link(TICKETS)}.")
    bad = list(ctx.bad_ticket_lines or [])
    parts.append("## Lines that could not be read")
    if bad:
        # A list, not a table: the raw lines often hold "|" characters of their own.
        parts.append("\n".join(f"* Line {t.line_no}: {_code(t.source)} {sentence(t.error)}" for t in bad))
        parts.append(f"Fix these lines in {link(TICKETS)} and they will be added on the next run.")
    else:
        parts.append("None, every line was read.")
    return "Ledger.md", frontmatter, "\n\n".join(parts)


# Report


def report_note(ctx: Context, report_md: str) -> NoteSpec:
    """("Reports/2026-10-02 1930 Report.md", frontmatter, body) holding the full report."""
    sig = toto_signal(ctx)
    toto_row, fourd_row = latest_row(ctx.toto), latest_row(ctx.fourd)
    frontmatter = {
        "tags": ["huatbot", "report"],
        "date": to_sg(ctx.now).date().isoformat(),
        "created": _stamp(ctx.now),
        "games": [GAME_NAMES[g] for g in ctx.games_drawn if g in GAME_NAMES],
        "toto_draw": as_int(field(toto_row, "draw_number")) if toto_row is not None else None,
        "fourd_draw": as_int(field(fourd_row, "draw_number")) if fourd_row is not None else None,
        "buy_signal": sig["label"],
        "next_toto_jackpot": sig["jackpot"],
    }
    links = f"Back to {link(DASHBOARD)}. Activity log: {link(activity_note_name(ctx.now))}."
    text = (report_md or "").strip("\n")
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        body = "\n".join([lines[0], "", links] + lines[1:])
    else:
        body = f"{links}\n\n{text}"
    return report_note_path(ctx.now), frontmatter, body


# Writing


def _draw_rows(ctx: Context, game: str) -> list[Any]:
    """Rows of ``ctx.new_draws[game]``, newest first, capped at settings.draw_notes_backfill.
    The newest new draw always gets its note (even with a cap of 0), because the Dashboard
    and the report link to the latest draw note."""
    cap = max(int(getattr(ctx.settings, "draw_notes_backfill", 0) or 0), 1)
    wanted = {int(n) for n in (ctx.new_draws or {}).get(game, []) or []}
    df = ctx.toto if game == "toto" else ctx.fourd
    if not wanted or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    rows = df[df["draw_number"].astype("int64").isin(wanted)].sort_values("draw_number", ascending=False)
    return [r for _, r in rows.head(cap).iterrows()]


def _all_notes(ctx: Context, report_md: str) -> list[NoteSpec]:
    notes: list[NoteSpec] = []
    for game, build in (("toto", toto_draw_note), ("4d", fourd_draw_note)):
        # Oldest first, so the activity log reads in draw order.
        notes += [build(r) for r in reversed(_draw_rows(ctx, game))]
    notes.append(report_note(ctx, report_md))
    notes += suggestions_notes(ctx)
    notes.append(ledger_note(ctx))
    notes.append(dashboard_note(ctx))
    return notes


def write_all(vault, ctx: Context, report_md: str) -> list[str]:
    """Write every note of this run into the vault; return the relative paths actually written.

    Draw notes are written for ``ctx.new_draws`` (at most ``settings.draw_notes_backfill`` most
    recent per game, and always the newest one), then the report, the suggestion notes, the
    ledger and the dashboard. A note whose content is unchanged is not rewritten
    (``Vault.write_note`` returns False) and not logged. Every write is logged as a NOTE event
    with a link that includes the note's folder; a note that cannot be written is logged as an
    ERROR event and the others are still written.
    """
    written: list[str] = []
    for rel, frontmatter, body in _all_notes(ctx, report_md):
        name = note_link(rel)
        try:
            existed = vault.exists(rel)
            changed = vault.write_note(rel, body, frontmatter)
        except (OSError, ValueError) as exc:
            log.warning("Could not write %s: %s", rel, exc)
            vault.log("ERROR", f"Could not write {name} ({type(exc).__name__})", when=ctx.now)
            continue
        if changed:
            vault.log("NOTE", f"{'Updated' if existed else 'Created'} {name}", when=ctx.now)
            written.append(rel)
    return written
