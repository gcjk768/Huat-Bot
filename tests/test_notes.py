"""Tests for huatbot.notes: the Obsidian notes the bot writes into the vault."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import pytest
import yaml

from huatbot import notes, report
from huatbot.models import Settings
from huatbot.store import empty_ledger, fourd_numbers, toto_numbers
from huatbot.textfmt import find_prose_dashes, has_prose_dashes
from huatbot.tickets import ledger_totals
from huatbot.vault import Vault, parse_note, render_note
from tests.ctxgen import make_context, variant

REPORT_NAME = "2026-10-01 1930 Report"
LOG_NAME = "2026-10 Activity"
TOTO_SUGGESTION = "2026-10-05 TOTO 4124"
FOURD_SUGGESTION = "2026-10-03 4D 5433"


@pytest.fixture(scope="module")
def ctx():
    return make_context()


@pytest.fixture(scope="module")
def report_md(ctx):
    return report.full_report(ctx)


def _yaml_block(text: str) -> dict:
    """Parse the frontmatter block with plain yaml.safe_load (independent of the vault parser)."""
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    assert m, "note has no frontmatter block"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict)
    return data


def _check_note(spec):
    rel, frontmatter, body = spec
    assert rel.endswith(".md")
    assert isinstance(frontmatter, dict) and "huatbot" in frontmatter["tags"]
    text = render_note(body, frontmatter)
    parsed, parsed_body = parse_note(text)
    assert parsed["tags"] == frontmatter["tags"]
    assert _yaml_block(text)["tags"] == frontmatter["tags"]
    assert not has_prose_dashes(text), find_prose_dashes(text)
    return rel, parsed, parsed_body


# Draw notes


def test_toto_draw_note(ctx):
    row = report.latest_row(ctx.toto)
    rel, fm, body = _check_note(notes.toto_draw_note(row))
    assert rel == "Draws/TOTO/2026-10-01 TOTO 4123.md"
    assert fm["game"] == "TOTO" and fm["draw"] == 4123 and fm["date"] == "2026-10-01"
    assert fm["numbers"] == toto_numbers(row)
    assert fm["additional"] == int(row["additional"])
    assert fm["tags"] == ["huatbot", "toto", "draw"]
    assert body.startswith("# TOTO draw 4123, Thu 1 Oct 2026")
    assert "[[Dashboard]]" in body
    assert "| Group 7 |" in body and "Shape of this set" in body


def test_toto_draw_note_accepts_a_dict_row(ctx):
    row = report.latest_row(ctx.toto).to_dict()
    row["draw_date"] = "2026-10-01"
    rel, _, _ = notes.toto_draw_note(row)
    assert rel == "Draws/TOTO/2026-10-01 TOTO 4123.md"


def test_fourd_draw_note_keeps_leading_zeros(ctx):
    row = report.latest_row(ctx.fourd).copy()
    row["first"], row["starter_1"] = "0042", "0698"
    rel, fm, body = _check_note(notes.fourd_draw_note(row))
    assert rel == "Draws/4D/2026-09-30 4D 5432.md"
    assert fm["game"] == "4D" and fm["draw"] == 5432 and fm["date"] == "2026-09-30"
    assert fm["top_three"].split()[0] == "0042"
    assert fm["starter"].split()[0] == "0698"
    assert len(fm["consolation"].split()) == 10
    for n in (x for tier in fourd_numbers(row).values() for x in tier):
        assert n in body
    assert "[[Dashboard]]" in body


# Dashboard


def test_dashboard_links_everything(ctx):
    rel, fm, body = _check_note(notes.dashboard_note(ctx))
    assert rel == "Dashboard.md"
    for name in (REPORT_NAME, "Ledger", "Tickets", "Settings", LOG_NAME):
        assert f"[[{name}]]" in body, name
    # Draw and suggestion notes are linked with their folder (a draw note and the suggestion
    # note made before that draw share a file name); in tables the alias pipe is escaped.
    for name in (TOTO_SUGGESTION, FOURD_SUGGESTION):
        assert f"[[Suggestions/{name}|{name}]]" in body, name
        assert f"[[Suggestions/{name}\\|{name}]]" in body, name
    for folder, name in (("Draws/TOTO", "2026-10-01 TOTO 4123"), ("Draws/4D", "2026-09-30 4D 5432")):
        assert f"[[{folder}/{name}\\|{name}]]" in body, name
    assert fm["buy_signal"] == ctx.buy_signal.label
    assert fm["next_toto_draw"] == 4124 and fm["next_toto_date"] == "2026-10-05"
    assert fm["next_4d_draw"] == 5433
    assert fm["net"] == ctx.ledger_totals["net"]
    for heading in ("## Next draws", "## Buy signal", "## Latest results", "## Suggested purchases",
                    "## My tickets", "## Warnings"):
        assert heading in body


def test_dashboard_without_next_draw_info(ctx):
    bare = variant(ctx, next_toto=None, next_fourd=None, buy_signal=None)
    _, fm, body = _check_note(notes.dashboard_note(bare))
    assert "regular schedule" in body
    assert f"[[Suggestions/{TOTO_SUGGESTION}|{TOTO_SUGGESTION}]]" in body  # date worked out from the TOTO draw days
    assert fm["buy_signal"] is None


# Suggestions


def test_suggestions_notes(ctx):
    specs = notes.suggestions_notes(ctx)
    assert [s[0] for s in specs] == [f"Suggestions/{TOTO_SUGGESTION}.md", f"Suggestions/{FOURD_SUGGESTION}.md"]
    for spec, plan in zip(specs, (ctx.toto_plan, ctx.fourd_plan), strict=True):
        _, fm, body = _check_note(spec)
        assert fm["total_cost"] == plan.total <= fm["budget"] == plan.budget
        assert "[[Dashboard]]" in body and f"[[{REPORT_NAME}]]" in body and "[[Tickets]]" in body
        for line in plan.lines:
            assert line.numbers in body
    toto_fm = _check_note(specs[0])[1]
    assert toto_fm["tags"] == ["huatbot", "suggestion", "toto"] and toto_fm["draw"] == 4124
    assert len(toto_fm["sets"]) == 4
    fourd_fm = _check_note(specs[1])[1]
    assert fourd_fm["numbers"].split() == [p.number for p in ctx.fourd_picks]


def test_suggestions_skip_games_without_data(ctx):
    no_fourd = variant(ctx, fourd=ctx.fourd.iloc[0:0], next_fourd=None)
    paths = [s[0] for s in notes.suggestions_notes(no_fourd)]
    assert paths == [f"Suggestions/{TOTO_SUGGESTION}.md"]


# Ledger


def test_ledger_note_lists_every_ticket_and_bad_line(ctx):
    rel, fm, body = _check_note(notes.ledger_note(ctx))
    assert rel == "Ledger.md"
    assert fm["spent"] == ctx.ledger_totals["spent"] and fm["tickets"] == ctx.ledger_totals["tickets"]
    table_rows = [ln for ln in body.splitlines() if ln.startswith("| TOTO ") or ln.startswith("| 4D ")]
    assert len(table_rows) == len(ctx.ledger)
    for numbers in ctx.ledger["numbers"]:
        assert numbers in body
    assert "Waiting for the draw" in body and "Checked" in body
    assert len(ctx.bad_ticket_lines) == 2
    for t in ctx.bad_ticket_lines:
        assert f"* Line {t.line_no}: `{t.source}`" in body  # raw text kept as code, dashes and all
    assert "2026-10-03" in body  # the ISO date inside the bad line is code, not prose
    assert "[[Dashboard]]" in body and "[[Tickets]]" in body


def test_ledger_note_without_tickets(ctx):
    empty = empty_ledger()
    bare = variant(ctx, ledger=empty, ledger_totals=ledger_totals(empty), settled_this_run=[], bad_ticket_lines=[])
    _, fm, body = _check_note(notes.ledger_note(bare))
    assert "No tickets yet" in body and "None, every line was read." in body
    assert fm["tickets"] == 0


# Report note


def test_report_note(ctx, report_md):
    rel, fm, body = _check_note(notes.report_note(ctx, report_md))
    assert rel == f"Reports/{REPORT_NAME}.md"
    assert fm["tags"] == ["huatbot", "report"] and fm["date"] == "2026-10-01"
    assert fm["games"] == ["TOTO", "4D"]
    lines = body.splitlines()
    assert lines[0].startswith("# Huat Bot report")
    assert "[[Dashboard]]" in lines[2] and f"[[{LOG_NAME}]]" in lines[2]
    for heading in report.SECTION_HEADINGS:
        assert heading in body


# Writing to a vault


def _md_files(vault):
    return sorted(p for p in vault.base.rglob("*.md"))


def _log_rows(vault, event):
    text = vault.read_text(f"Logs/{LOG_NAME}.md") or ""
    return [ln for ln in text.splitlines() if f"| {event} |" in ln]


def test_write_all_creates_notes_and_logs(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    written = notes.write_all(vault, ctx, report_md)
    expected = {
        "Dashboard.md", "Ledger.md", f"Reports/{REPORT_NAME}.md",
        f"Suggestions/{TOTO_SUGGESTION}.md", f"Suggestions/{FOURD_SUGGESTION}.md",
        "Draws/TOTO/2026-09-28 TOTO 4122.md", "Draws/TOTO/2026-10-01 TOTO 4123.md",
        "Draws/4D/2026-09-27 4D 5431.md", "Draws/4D/2026-09-30 4D 5432.md",
    }
    assert set(written) == expected
    for rel in written:
        assert vault.exists(rel)
        text = vault.read_text(rel)
        assert _yaml_block(text)["tags"][0] == "huatbot"
        fm, _ = vault.read_note(rel)
        assert fm
    assert len(_log_rows(vault, "NOTE")) == len(written)
    assert any("Created [[Dashboard]]" in row for row in _log_rows(vault, "NOTE"))
    for path in _md_files(vault):  # notes and the activity log
        assert not has_prose_dashes(path.read_text(encoding="utf-8")), path
    # nothing written outside the bot folder
    assert {p.relative_to(vault.root).parts[0] for p in (tmp_path / "vault").rglob("*") if p.is_file()} == {"Huat Bot"}


def test_write_all_twice_changes_nothing(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    first = notes.write_all(vault, ctx, report_md)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in _md_files(vault)}
    again = notes.write_all(vault, ctx, report_md)
    assert first and again == []
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in _md_files(vault)}
    assert after == before  # not even the activity log changed
    for rel in first:
        rel_path, fm, body = next(s for s in notes._all_notes(ctx, report_md) if s[0] == rel)
        assert vault.write_note(rel_path, body, fm) is False


def test_write_all_updates_changed_notes_only(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report_md)
    later = variant(ctx, commentary="Second run.", new_draws={})
    written = notes.write_all(vault, later, report.full_report(later))
    assert written == [f"Reports/{REPORT_NAME}.md"]
    assert any("Updated [[Reports/2026-10-01 1930 Report\\|2026-10-01 1930 Report]]" in row
               for row in _log_rows(vault, "NOTE"))


def test_write_all_caps_draw_notes_at_backfill(tmp_path, ctx, report_md):
    settings = Settings(draw_notes_backfill=3)
    toto_new = [int(n) for n in ctx.toto["draw_number"].iloc[-10:]]
    fourd_new = [int(n) for n in ctx.fourd["draw_number"].iloc[-10:]]
    capped = variant(ctx, settings=settings, new_draws={"toto": toto_new, "4d": fourd_new})
    vault = Vault(tmp_path / "vault")
    written = notes.write_all(vault, capped, report_md)
    toto_notes = sorted(p for p in written if p.startswith("Draws/TOTO/"))
    fourd_notes = sorted(p for p in written if p.startswith("Draws/4D/"))
    assert [p.rsplit(" ", 1)[-1] for p in toto_notes] == ["4121.md", "4122.md", "4123.md"]
    assert [p.rsplit(" ", 1)[-1] for p in fourd_notes] == ["5430.md", "5431.md", "5432.md"]
    assert len(list(vault.path("Draws/TOTO").glob("*.md"))) == 3


def test_write_all_without_backfill_still_writes_the_newest_draw_note(tmp_path, ctx, report_md):
    # The Dashboard and the report link to the newest draw note, so it is always written.
    vault = Vault(tmp_path / "vault")
    written = notes.write_all(vault, variant(ctx, settings=Settings(draw_notes_backfill=0)), report_md)
    assert [p for p in written if p.startswith("Draws/")] == ["Draws/TOTO/2026-10-01 TOTO 4123.md",
                                                             "Draws/4D/2026-09-30 4D 5432.md"]
    assert "Dashboard.md" in written
    dashboard = vault.read_text("Dashboard.md")
    for rel in ("Draws/TOTO/2026-10-01 TOTO 4123", "Draws/4D/2026-09-30 4D 5432"):
        assert f"[[{rel}\\|" in dashboard and vault.exists(rel + ".md")


WIKILINK = re.compile(r"\[\[([^\]|\\]+)(?:\\?\|[^\]]*)?\]\]")


def test_links_to_draw_and_suggestion_notes_are_unambiguous(tmp_path, ctx):
    """After the draw a suggestion note was made for is held, the vault holds a draw note and a
    suggestion note with the same file name: every link to such a name must say which one."""
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report.full_report(ctx))  # Thu 1 Oct: suggestions for Mon 5 Oct, TOTO 4124
    held = ctx.toto.iloc[[-1]].copy()
    held["draw_number"] = 4124
    held["draw_date"] = pd.Timestamp("2026-10-05")
    later = variant(ctx, toto=pd.concat([ctx.toto, held], ignore_index=True), new_draws={"toto": [4124]},
                    now=datetime(2026, 10, 5, 19, 30, tzinfo=report.SG), next_toto=None, buy_signal=None)
    notes.write_all(vault, later, report.full_report(later))

    files: dict[str, list[str]] = {}
    for path in vault.base.rglob("*.md"):
        files.setdefault(path.stem, []).append(path.relative_to(vault.base).as_posix())
    assert len(files["2026-10-05 TOTO 4124"]) == 2  # Draws/TOTO and Suggestions
    checked = 0
    for path in vault.base.rglob("*.md"):
        for target in WIKILINK.findall(path.read_text()):
            name = target.rsplit("/", 1)[-1]
            if len(files.get(name, [])) > 1:
                checked += 1
                assert f"{target}.md" in files[name], (path.name, target)
    assert checked > 0
    dashboard = vault.read_text("Dashboard.md")
    assert "[[Draws/TOTO/2026-10-05 TOTO 4124\\|2026-10-05 TOTO 4124]]" in dashboard


def test_write_all_logs_an_error_and_carries_on(tmp_path, ctx, report_md, monkeypatch):
    vault = Vault(tmp_path / "vault")
    real = vault.write_note

    def flaky(rel, body, frontmatter=None):
        if rel == "Ledger.md":
            raise PermissionError("read only share")
        return real(rel, body, frontmatter)

    monkeypatch.setattr(vault, "write_note", flaky)
    written = notes.write_all(vault, ctx, report_md)
    assert "Ledger.md" not in written and "Dashboard.md" in written
    errors = _log_rows(vault, "ERROR")
    assert len(errors) == 1 and "[[Ledger]]" in errors[0]
