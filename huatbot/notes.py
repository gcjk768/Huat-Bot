"""Obsidian notes the bot writes into the vault on the NAS.

Each builder returns ``(relative path, frontmatter, body)``; ``write_all`` hands them to
``Vault.write_note`` (atomic, skipped when unchanged) and logs every note it actually wrote as
a NOTE event in this month's activity log.

The notes are linked into one small graph: ``Dashboard.md`` links to the newest report,
``Ledger``, ``Tickets``, ``Settings``, this month's activity log and the latest draw note; draw
and report notes link back to ``[[Dashboard]]``.

Frontmatter uses plain Obsidian properties: ``tags`` (always ``huatbot`` plus the kind of
note), ``draw``, ``date`` (ISO) and the figures of that note. Bodies follow the house style: no
dashes in prose (wikilinks with ISO dates and table separator rows are fine).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import pandas as pd

from .models import LEDGER_COLUMNS, Context
from .report import (
    activity_note_name,
    as_date,
    as_float,
    as_int,
    clean_text,
    dollars,
    draw_note_rel,
    draw_type_name,
    field,
    history_line,
    jackpot_text,
    latest_row,
    link,
    next_draw,
    note_link,
    outlook_md,
    report_note_name,
    report_warnings,
    rollover_text,
    run_winnings,
    sentence,
    to_sg,
    toto_result_md,
    toto_signal,
)
from .store import toto_numbers
from .textfmt import fmt_date, fmt_datetime, md_table, money, pct, per_dollar, plural, toto_nums
from .vault import render_note

log = logging.getLogger(__name__)

DASHBOARD = "Dashboard"
LEDGER = "Ledger"
TICKETS = "Tickets"
SETTINGS = "Settings"
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
    return f"`` {s} ``" if "`" in s else f"`{s}`"


def report_note_path(now: datetime) -> str:
    return f"{REPORT_FOLDER}/{report_note_name(now)}.md"


def draw_note_path(row: Any) -> str:
    return draw_note_rel(as_int(field(row, "draw_number")), as_date(field(row, "draw_date")))


def _totals_row(totals: dict) -> list[str]:
    return [str(as_int(totals.get("tickets"))), dollars(totals.get("spent") or 0.0),
            dollars(totals.get("won") or 0.0), dollars(totals.get("net") or 0.0),
            str(as_int(totals.get("pending"))), dollars(totals.get("pending_cost") or 0.0)]


# Draw notes


def toto_draw_note(row: Any) -> NoteSpec:
    """("Draws/TOTO/2026-10-01 TOTO 4123.md", frontmatter, body) for one TOTO result row."""
    number = as_int(field(row, "draw_number"))
    day = as_date(field(row, "draw_date"))
    frontmatter = {
        "tags": ["huatbot", "toto", "draw"],
        "draw": number,
        "date": _iso(day),
        "numbers": toto_numbers(row),
        "additional": as_int(field(row, "additional")),
        "draw_type": str(field(row, "draw_type", "normal")),
        "group1_prize": as_float(field(row, "jackpot")),
        "group1_winners": as_int(field(row, "g1_winners")),
    }
    body = "\n\n".join([
        f"# TOTO draw {number}, {fmt_date(day)}",
        f"Back to {link(DASHBOARD)}.",
        toto_result_md(row),
    ])
    return draw_note_path(row), frontmatter, body


# Dashboard


def dashboard_note(ctx: Context, report_name: str | None = None) -> NoteSpec:
    """("Dashboard.md", frontmatter, body): next draw, buy signal, the next big prize, the latest
    result, ledger totals and links to the newest report (``report_name``, default this run's)."""
    sig = toto_signal(ctx)
    nd = next_draw(ctx)
    out = ctx.outlook
    first = out.steps[0] if out is not None and out.steps else None
    biggest = out.biggest if out is not None else None
    totals = ctx.ledger_totals or {}
    report = report_name or report_note_name(ctx.now)
    latest = latest_row(ctx.toto)

    frontmatter = {
        "tags": ["huatbot", "dashboard"],
        "updated": _stamp(ctx.now),
        "next_draw": nd.number,
        "next_date": nd.day.isoformat() if nd.day else None,
        "next_jackpot": sig["jackpot"],
        "next_draw_type": sig["draw_type"],
        "chance_jackpot_won": round(first.chance_won, 4) if first else None,
        "next_big_prize": round(biggest.jackpot) if biggest else None,
        "next_big_prize_date": biggest.draw_date.isoformat() if biggest and biggest.draw_date else None,
        "buy_signal": sig["label"],
        "return_per_dollar": round(sig["ev"], 4) if sig["ev"] is not None else None,
        "latest_draw": as_int(field(latest, "draw_number")) if latest is not None else None,
        "spent": float(totals.get("spent") or 0.0),
        "won": float(totals.get("won") or 0.0),
        "net": float(totals.get("net") or 0.0),
    }

    parts = [
        "# Huat Bot dashboard",
        f"Updated {fmt_datetime(to_sg(ctx.now))}. Newest report: {link(report)}. "
        f"Activity log: {link(activity_note_name(ctx.now))}.",
        f"Your notes: {link(TICKETS)} (add tickets here), {link(LEDGER)} (every ticket and the totals), "
        f"{link(SETTINGS)} (jackpot alert).",
        "## Next draw",
    ]
    jackpot = jackpot_text(sig, "not announced yet")
    rows = [
        ["Draw", str(nd.number or "n/a")],
        ["Date and time", nd.when_text],
        ["Estimated jackpot", "sales closed" if nd.held else jackpot],
        ["Draw type", draw_type_name(sig["draw_type"])],
        ["Jackpot rollovers so far", rollover_text(sig)],
        ["Chance somebody wins Group 1", pct(first.chance_won, 0) if first else "n/a"],
    ]
    parts.append(md_table(["Item", "Value"], rows, align="ll"))

    parts.append("## Buy signal")
    if nd.held:
        parts.append("No buy signal: this draw was held and its sales are closed.")
    elif sig["label"]:
        ev = f" Return per $1: about {per_dollar(sig['ev'])}." if sig["ev"] is not None else ""
        parts.append(f"**{sig['label']}**. {sentence(sig['reason'])}{ev}".rstrip())
    else:
        parts.append("Not worked out in this run.")

    parts.append("## The next big prize")
    parts.append(outlook_md(out))

    parts.append("## Latest result")
    if latest is None:
        parts.append("No results are stored yet.")
    else:
        number, day = as_int(field(latest, "draw_number")), as_date(field(latest, "draw_date"))
        winners = as_int(field(latest, "g1_winners"))
        prize = (f"Group 1 {money(as_float(field(latest, 'jackpot')))}, "
                 f"{plural(winners, 'winner') if winners else 'no winner'}")
        parts.append(f"Draw {number}, {fmt_date(day)}: **{toto_nums(toto_numbers(latest))}**, additional "
                     f"{as_int(field(latest, 'additional'))}. {prize}. {note_link(draw_note_rel(number, day))}")
    hist = history_line(ctx.history)
    if hist:
        parts.append(hist)

    parts.append("## My tickets")
    parts.append(md_table(["Tickets", "Spent", "Won", "Net", "Waiting", "Waiting cost"], [_totals_row(totals)],
                          align="rrrrrr"))
    settled = list(ctx.settled_this_run or [])
    if settled:
        won, _ = run_winnings(ctx)  # a corrected ticket counts only what it adds to the earlier win
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
            fmt_date(r.get("draw_date") or None),
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
        parts.append(md_table(["Draw date", "Draw", "Numbers", "Bet type", "Cost", "Status", "Result", "Won"],
                              _ledger_rows(ledger), align="llllrllr"))
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
    row = latest_row(ctx.toto)
    out = ctx.outlook
    biggest = out.biggest if out is not None else None
    frontmatter = {
        "tags": ["huatbot", "report"],
        "date": to_sg(ctx.now).date().isoformat(),
        "created": _stamp(ctx.now),
        "new_draws": len(ctx.new_draws or []),
        "latest_draw": as_int(field(row, "draw_number")) if row is not None else None,
        "buy_signal": sig["label"],
        "next_jackpot": sig["jackpot"],
        "next_big_prize": round(biggest.jackpot) if biggest else None,
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

# Frontmatter lines that only stamp the time of the run: a note whose only change is in these
# is not rewritten (or logged as updated).
_STAMP_LINE = re.compile(r"^(?:updated|generated): .*(?:\n|$)", re.M)


def _without_stamps(text: str) -> str:
    head, sep, body = text.partition("\n---\n") if text.startswith("---\n") else ("", "", text)
    return _STAMP_LINE.sub("", head) + sep + body


def _only_stamps_changed(vault, rel: str, body: str, frontmatter: dict) -> bool:
    """True when the note on disk differs from the new content only in its run time stamps."""
    old = vault.read_text(rel)
    if old is None:
        return False
    return _without_stamps(old) == _without_stamps(render_note(body, frontmatter))


def _draw_rows(ctx: Context, vault=None) -> list[Any]:
    """Rows of ``ctx.new_draws``, newest first, capped at settings.draw_notes_backfill. The
    newest new draw always gets its note (even with a cap of 0), because the Dashboard and the
    report link to it. With ``vault``, the newest stored draw is also included when its note is
    missing (a run stopped after saving the draw but before writing its note)."""
    cap = max(int(getattr(ctx.settings, "draw_notes_backfill", 0) or 0), 1)
    wanted = {int(n) for n in ctx.new_draws or []}
    df = ctx.toto
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    newest = latest_row(df)
    if vault is not None and newest is not None and not vault.exists(draw_note_path(newest)):
        wanted.add(as_int(field(newest, "draw_number")))
    if not wanted:
        return []
    rows = df[df["draw_number"].astype("int64").isin(wanted)].sort_values("draw_number", ascending=False)
    return [r for _, r in rows.head(cap).iterrows()]


# The parts of a report note that only stamp the time of the run.
_CREATED_LINE = re.compile(r"^created: .*(?:\n|$)", re.M)
_REPORT_TITLE = re.compile(r"^# Huat Bot report, .*$", re.M)


def _report_without_stamps(text: str) -> str:
    head, sep, body = text.partition("\n---\n") if text.startswith("---\n") else ("", "", text)
    return _CREATED_LINE.sub("", head) + sep + _REPORT_TITLE.sub("# Huat Bot report", body, count=1)


def _earlier_report(vault, ctx: Context, spec: NoteSpec) -> str | None:
    """The newest report note of the same day, when it differs from this run's report only in
    its time stamps, or None. A rerun that only retries the post (Telegram was down) then adds
    no near identical report."""
    rel, frontmatter, body = spec
    try:
        folder = vault.path(REPORT_FOLDER)
        names = sorted(p.name for p in folder.glob(f"{to_sg(ctx.now):%Y-%m-%d} * Report.md")) \
            if folder.is_dir() else []
    except OSError:
        return None
    if not names or f"{REPORT_FOLDER}/{names[-1]}" == rel:
        return None  # nothing earlier, or this run's own report (the usual stamps check applies)
    latest = f"{REPORT_FOLDER}/{names[-1]}"
    old = vault.read_text(latest)
    if old is None or _report_without_stamps(old) != _report_without_stamps(render_note(body, frontmatter)):
        return None
    return latest


def report_rel(vault, ctx: Context, report_md: str) -> str:
    """Vault path of this run's report note: an earlier report of the same day that differs
    only in its time stamps (see ``_earlier_report``), else ``report_note_path(ctx.now)``."""
    spec = report_note(ctx, report_md)
    return _earlier_report(vault, ctx, spec) or spec[0]


def _all_notes(ctx: Context, report_md: str, vault=None) -> list[NoteSpec]:
    # Draw notes oldest first, so the activity log reads in draw order.
    notes: list[NoteSpec] = [toto_draw_note(r) for r in reversed(_draw_rows(ctx, vault))]
    report = report_note(ctx, report_md)
    earlier = _earlier_report(vault, ctx, report) if vault is not None else None
    report_name = (earlier or report[0]).rsplit("/", 1)[-1].removesuffix(".md")
    if earlier is None:
        notes.append(report)
    notes.append(ledger_note(ctx))
    notes.append(dashboard_note(ctx, report_name))
    return notes


def write_all(vault, ctx: Context, report_md: str) -> list[str]:
    """Write every note of this run into the vault; return the relative paths actually written.

    Draw notes are written for ``ctx.new_draws`` (at most ``settings.draw_notes_backfill`` most
    recent, and always the newest one) and for the newest stored draw when its note is missing,
    then the report, the ledger and the dashboard. A note whose content is unchanged apart from
    the run time stamps in its properties is not rewritten and not logged. No new report is
    written when the newest report of the same day differs only in its time stamps. Every
    write is logged as a NOTE event; a note that cannot be written is logged as an ERROR event
    and the others are still written.
    """
    written: list[str] = []
    for rel, frontmatter, body in _all_notes(ctx, report_md, vault):
        name = note_link(rel)
        try:
            existed = vault.exists(rel)
            if existed and _only_stamps_changed(vault, rel, body, frontmatter):
                continue
            changed = vault.write_note(rel, body, frontmatter)
        except (OSError, ValueError) as exc:
            log.warning("Could not write %s: %s", rel, exc)
            vault.log("ERROR", f"Could not write {name} ({type(exc).__name__})", when=ctx.now)
            continue
        if changed:
            vault.log("NOTE", f"{'Updated' if existed else 'Created'} {name}", when=ctx.now)
            written.append(rel)
    return written
