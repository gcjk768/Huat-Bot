"""Vault: layout, confinement to the bot folder, notes with properties, activity log, state."""
from __future__ import annotations

import json
import logging
import os
import stat
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from huatbot import vault as V
from huatbot.vault import SG, Vault, fmt_log_time, parse_note, render_note

LOG_REL = "Logs/2026-10 Activity.md"


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path / "My Vault")


def _tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


# Construction and paths


def test_paths_default_layout(tmp_path):
    v = Vault(tmp_path)
    assert v.root == tmp_path
    assert v.base == tmp_path / "Huat Bot"
    assert v.data_dir == tmp_path / "Huat Bot" / "Data"
    assert v.toto_csv == v.data_dir / "toto.csv"
    assert v.fourd_csv == v.data_dir / "fourd.csv"
    assert v.ledger_csv == v.data_dir / "ledger.csv"
    assert v.state_path == v.data_dir / "state.json"
    assert v.prize_rules_path == v.data_dir / "prize_rules.json"
    assert v.backtest_cache_path == v.data_dir / "backtest_cache.json"
    assert v.path("Draws/TOTO", "2026-10-01 TOTO 4123.md") == v.base / "Draws" / "TOTO" / "2026-10-01 TOTO 4123.md"
    assert v.path() == v.base


def test_custom_data_dir_and_empty_folder(tmp_path):
    v = Vault(tmp_path / "vault", folder="", data_dir=tmp_path / "data")
    assert v.base == tmp_path / "vault"
    assert v.data_dir == tmp_path / "data"
    nested = Vault(tmp_path, folder="Bots/Huat Bot")
    assert nested.base == tmp_path / "Bots" / "Huat Bot"


def test_from_env(monkeypatch, tmp_path):
    for name in ("VAULT_PATH", "VAULT_FOLDER", "DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    v = Vault.from_env()
    assert v.root == Path("./vault")
    assert v.base == Path("./vault") / "Huat Bot"
    assert v.data_dir == v.base / "Data"

    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "nas vault"))
    monkeypatch.setenv("VAULT_FOLDER", "Lottery")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "csv"))
    v = Vault.from_env()
    assert v.base == tmp_path / "nas vault" / "Lottery"
    assert v.data_dir == tmp_path / "csv"

    v = Vault.from_env({"VAULT_PATH": "", "VAULT_FOLDER": "  ", "DATA_DIR": ""})
    assert v.root == Path("./vault") and v.folder == "Huat Bot"


@pytest.mark.parametrize("bad", ["../outside.md", "/etc/passwd", ".obsidian/app.json", "Draws/../../x.md",
                                 "Logs/.hidden.md", "C:\\Windows\\x.md", ".trash/old.md"])
def test_path_refuses_escapes_and_hidden_folders(vault, bad):
    with pytest.raises(ValueError):
        vault.path(bad)
    with pytest.raises(ValueError):
        vault.write_note(bad, "x")


@pytest.mark.parametrize("bad_folder", ["..", "../Other", ".obsidian", "/abs"])
def test_folder_is_validated(tmp_path, bad_folder):
    with pytest.raises(ValueError):
        Vault(tmp_path, folder=bad_folder)


def test_rel_normalises_separators(vault):
    assert vault.rel("Draws\\TOTO\\a b.md") == "Draws/TOTO/a b.md"
    assert vault.rel("./Reports//x.md") == "Reports/x.md"


# Layout


def test_ensure_layout_creates_folders_and_templates_once(vault):
    templates = {"Settings.md": "---\na: 1\n---\n# Settings\n", "Tickets.md": "# Tickets\n"}
    created = vault.ensure_layout(templates)
    assert created == ["Settings.md", "Tickets.md"]
    for sub in ("Data", "Draws/TOTO", "Draws/4D", "Reports", "Suggestions", "Logs"):
        assert (vault.base / sub).is_dir(), sub
    assert vault.read_text("Settings.md") == templates["Settings.md"]

    # The user edits a note in Obsidian; a second layout pass must not overwrite it.
    (vault.base / "Tickets.md").write_text("# My tickets\nTOTO, 5 Oct 2026, 1 2 3 4 5 6, Ordinary, 1\n")
    assert vault.ensure_layout(templates) == []
    assert "My tickets" in vault.read_text("Tickets.md")


def test_ensure_layout_with_separate_data_dir(tmp_path):
    v = Vault(tmp_path / "vault", data_dir=tmp_path / "data")
    v.ensure_layout({})
    assert (tmp_path / "data").is_dir()
    assert not (v.base / "Data").exists()


def test_never_touches_obsidian_folder_or_outside_files(tmp_path):
    root = tmp_path / "vault"
    (root / ".obsidian").mkdir(parents=True)
    (root / ".obsidian" / "app.json").write_text('{"x": 1}')
    (root / "Personal.md").write_text("mine")
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}

    v = Vault(root)
    v.ensure_layout({"Settings.md": "x\n"})
    v.write_note("Dashboard.md", "# Dashboard", {"tags": ["huatbot"]})
    v.log("RUN", "Started", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    v.save_state({"a": 1})

    for p, mtime in before.items():
        assert p.stat().st_mtime_ns == mtime, p
    assert _tree(root / ".obsidian") == {"app.json"}
    outside = {p.name for p in root.iterdir()}
    assert outside == {".obsidian", "Personal.md", "Huat Bot"}


# Notes and properties


def test_write_and_read_note_round_trip(vault):
    fm = {
        "tags": ["huatbot", "toto"],
        "game": "TOTO",
        "draw": np.int64(4123),
        "date": pd.Timestamp("2026-10-01"),
        "drawn_at": datetime(2026, 10, 1, 18, 30, tzinfo=SG),
        "day": date(2026, 10, 1),
        "numbers": [np.int64(3), 11, 19, 27, 38, 45],
        "jackpot": np.float64(1_234_567.0),
        "missing": float("nan"),
        "detail": "Group 7 x3: yes",
        "words": "Huat ah",
    }
    body = "# TOTO draw 4123\n\nBack to [[Dashboard]]"
    assert vault.write_note("Draws/TOTO/2026-10-01 TOTO 4123.md", body, fm) is True

    path = vault.base / "Draws" / "TOTO" / "2026-10-01 TOTO 4123.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\ntags: [huatbot, toto]\ngame: TOTO\ndraw: 4123\n")
    assert "numbers: [3, 11, 19, 27, 38, 45]\n" in text
    assert text.endswith("---\n# TOTO draw 4123\n\nBack to [[Dashboard]]\n")

    # The properties block is plain YAML that a strict safe loader accepts.
    block = text.split("---\n")[1]
    loaded = yaml.safe_load(block)
    assert loaded["date"] == "2026-10-01"
    assert loaded["drawn_at"] == "2026-10-01T18:30:00+08:00"
    assert loaded["missing"] is None
    assert list(loaded) == list(fm)  # key order kept

    got_fm, got_body = vault.read_note("Draws/TOTO/2026-10-01 TOTO 4123.md")
    assert got_fm["draw"] == 4123 and got_fm["jackpot"] == 1_234_567.0
    assert got_fm["day"] == "2026-10-01"
    assert got_fm["numbers"] == [3, 11, 19, 27, 38, 45]
    assert got_body == body + "\n"


def test_unchanged_note_is_not_rewritten(vault, monkeypatch):
    assert vault.write_note("Dashboard.md", "# Dashboard\n", {"tags": ["huatbot"]}) is True
    calls = []
    monkeypatch.setattr(V, "atomic_write_text", lambda *a, **k: calls.append(a))
    assert vault.write_note("Dashboard.md", "# Dashboard\n", {"tags": ["huatbot"]}) is False
    assert vault.write_note("Dashboard.md", "# Dashboard", {"tags": ["huatbot"]}) is False  # same after newline fix
    assert calls == []
    assert vault.write_note("Dashboard.md", "# Dashboard v2\n", {"tags": ["huatbot"]}) is True
    assert len(calls) == 1


def test_unchanged_check_ignores_windows_line_endings(vault):
    vault.write_note("Ledger.md", "# Ledger\n\nline", {"a": 1})
    path = vault.path("Ledger.md")
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))  # a Windows PC synced it back
    assert vault.write_note("Ledger.md", "# Ledger\n\nline", {"a": 1}) is False


def test_note_without_frontmatter(vault):
    vault.write_note("Plain.md", "Just text")
    assert vault.read_text("Plain.md") == "Just text\n"
    assert vault.read_note("Plain.md") == ({}, "Just text\n")


def test_read_missing_note(vault):
    assert vault.read_text("Nope.md") is None
    assert vault.read_note("Nope.md") == ({}, "")


def test_read_note_bad_yaml_keeps_body_and_warns(vault, caplog):
    vault.write_text("Settings.md", "---\ntoto_budget: [10\n---\n# Settings\n")
    with caplog.at_level(logging.WARNING, logger="huatbot.vault"):
        fm, body = vault.read_note("Settings.md")
    assert fm == {} and body == "# Settings\n"
    assert "not valid YAML" in caplog.text


def test_read_note_non_mapping_yaml(vault):
    vault.write_text("List.md", "---\n- a\n- b\n---\nbody\n")
    assert vault.read_note("List.md") == ({}, "body\n")


def test_read_note_obsidian_style_and_bom(vault):
    # What Obsidian itself writes: block lists, unquoted dates, CRLF from a Windows device, a BOM.
    raw = "\ufeff---\r\ntags:\r\n  - huatbot\r\ndate: 2026-10-01\r\nbudget: 10\r\n---\r\n# Hi\r\n"
    vault.path("Obs.md").parent.mkdir(parents=True, exist_ok=True)
    vault.path("Obs.md").write_bytes(raw.encode("utf-8"))
    fm, body = vault.read_note("Obs.md")
    assert fm == {"tags": ["huatbot"], "date": "2026-10-01", "budget": 10}
    assert body == "# Hi\n"


def test_empty_frontmatter_block(vault):
    vault.write_text("Empty.md", "---\n---\nbody\n")
    assert vault.read_note("Empty.md") == ({}, "body\n")


def test_render_and_parse_helpers():
    text = render_note("body", {"a": [1, 2], "b": {"c": "x"}, "e": []})
    assert text == "---\na: [1, 2]\nb:\n  c: x\ne: []\n---\nbody\n"
    assert parse_note(text) == ({"a": [1, 2], "b": {"c": "x"}, "e": []}, "body\n")
    assert render_note("", None) == ""
    assert parse_note("no fence here") == ({}, "no fence here")


def test_file_names_with_spaces_and_unicode(vault):
    rel = "Suggestions/2026-10-05 TOTO 4124.md"
    assert vault.write_note(rel, "Huat ah 发财", {"title": "发"})
    assert vault.read_note(rel) == ({"title": "发"}, "Huat ah 发财\n")
    assert "发" in vault.path(rel).read_text(encoding="utf-8")  # written as text, not escaped


def test_new_files_get_normal_permissions_and_existing_mode_is_kept(vault):
    umask = os.umask(0)
    os.umask(umask)
    vault.write_note("New.md", "x")
    mode = stat.S_IMODE(vault.path("New.md").stat().st_mode)
    assert mode == 0o666 & ~umask  # not the 0600 of a temp file
    os.chmod(vault.path("New.md"), 0o640)
    vault.write_note("New.md", "changed")
    assert stat.S_IMODE(vault.path("New.md").stat().st_mode) == 0o640


def test_no_temp_files_left_behind(vault):
    vault.write_note("A.md", "a")
    vault.write_note("A.md", "b")
    vault.save_state({"x": 1})
    leftovers = [p for p in vault.base.rglob("*.tmp")]
    assert leftovers == []


# Activity log


def test_log_creates_monthly_note_with_title_and_header(vault):
    vault.log("FETCH", "2 new TOTO draws", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    text = vault.read_text(LOG_REL)
    assert text == (
        "# Huat Bot activity, October 2026\n\n"
        "| Time | Event | Details |\n"
        "| --- | --- | --- |\n"
        "| Fri 2 Oct 2026 7.30pm | FETCH | 2 new TOTO draws |\n"
    )


def test_log_appends_rows_in_order(vault):
    vault.log("RUN", "Started", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    vault.log("note", "Wrote [[Dashboard]]", when=datetime(2026, 10, 2, 19, 31, tzinfo=SG))
    lines = vault.read_text(LOG_REL).splitlines()
    assert lines[-2] == "| Fri 2 Oct 2026 7.30pm | RUN | Started |"
    assert lines[-1] == "| Fri 2 Oct 2026 7.31pm | NOTE | Wrote [[Dashboard]] |"
    assert sum(1 for line in lines if line.startswith("| Time |")) == 1


def test_log_escapes_pipes_and_flattens_newlines(vault):
    vault.log("ERROR", "bad | value\nsecond line - with dash \u2014 here",
              when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    row = vault.read_text(LOG_REL).splitlines()[-1]
    assert row == "| Fri 2 Oct 2026 7.30pm | ERROR | bad \\| value second line, with dash, here |"
    assert vault.read_text(LOG_REL).count("\n") == 5  # title, blank, header, rule, one row


def test_log_keeps_iso_dates_and_wikilink_aliases(vault):
    vault.log("NOTE", "[[Draws/TOTO/2026-10-01 TOTO 4123|TOTO 4123]]", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    row = vault.read_text(LOG_REL).splitlines()[-1]
    assert "[[Draws/TOTO/2026-10-01 TOTO 4123\\|TOTO 4123]]" in row


def test_log_month_rollover_and_time_zones(vault):
    # 16:05 UTC on 31 Oct is 12.05am on 1 Nov in Singapore.
    vault.log("RUN", "late", when=datetime(2026, 10, 31, 16, 5, tzinfo=timezone.utc))
    text = vault.read_text("Logs/2026-11 Activity.md")
    assert text.startswith("# Huat Bot activity, November 2026\n")
    assert "| Sun 1 Nov 2026 12.05am | RUN | late |" in text
    # A naive time is taken as Singapore time.
    vault.log("RUN", "naive", when=datetime(2026, 10, 5, 12, 0))
    assert "| Mon 5 Oct 2026 12.00pm | RUN | naive |" in vault.read_text(LOG_REL)


def test_log_default_time_is_now(vault):
    vault.log("RUN", "now")
    files = list((vault.base / "Logs").glob("* Activity.md"))
    assert len(files) == 1 and "| RUN | now |" in files[0].read_text()


def test_log_starts_new_table_after_user_text(vault):
    vault.log("RUN", "one", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    path = vault.path(LOG_REL)
    path.write_text(path.read_text() + "\nMy own note under the table.\n")
    vault.log("RUN", "two", when=datetime(2026, 10, 2, 19, 40, tzinfo=SG))
    text = path.read_text()
    assert text.count("| Time | Event | Details |") == 2
    assert text.endswith("| --- | --- | --- |\n| Fri 2 Oct 2026 7.40pm | RUN | two |\n")


def test_log_never_raises(vault, monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(V, "atomic_write_text", boom)
    with caplog.at_level(logging.WARNING, logger="huatbot.vault"):
        vault.log("RUN", "x", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))
    assert "disk full" in caplog.text

    # A folder where the note should be is also survived.
    monkeypatch.undo()
    vault.path(LOG_REL).mkdir(parents=True)
    vault.log("RUN", "x", when=datetime(2026, 10, 2, 19, 30, tzinfo=SG))


def test_log_bad_when_does_not_raise(vault):
    vault.log("RUN", "x", when="not a datetime")  # type: ignore[arg-type]


@pytest.mark.parametrize("hour,minute,expected", [
    (0, 5, "Fri 2 Oct 2026 12.05am"),
    (9, 0, "Fri 2 Oct 2026 9.00am"),
    (12, 0, "Fri 2 Oct 2026 12.00pm"),
    (19, 30, "Fri 2 Oct 2026 7.30pm"),
    (23, 59, "Fri 2 Oct 2026 11.59pm"),
])
def test_fmt_log_time(hour, minute, expected):
    assert fmt_log_time(datetime(2026, 10, 2, hour, minute, tzinfo=SG)) == expected


# State


def test_state_round_trip_and_unchanged_skip(vault, monkeypatch):
    assert vault.load_state() == {}
    state = {
        "next_draws": {"toto": datetime(2026, 10, 5, 18, 30, tzinfo=SG), "4d": date(2026, 10, 3)},
        "skip": {"toto": {4001, 3999}},
        "last_posted": np.int64(4123),
    }
    vault.save_state(state)
    loaded = vault.load_state()
    assert loaded == {
        "next_draws": {"toto": "2026-10-05T18:30:00+08:00", "4d": "2026-10-03"},
        "skip": {"toto": [3999, 4001]},
        "last_posted": 4123,
    }
    assert json.loads(vault.state_path.read_text()) == loaded

    calls = []
    monkeypatch.setattr(V, "atomic_write_text", lambda *a, **k: calls.append(a))
    vault.save_state(loaded)
    assert calls == []


def test_load_state_survives_corrupt_file(vault, caplog):
    vault.data_dir.mkdir(parents=True)
    vault.state_path.write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="huatbot.vault"):
        assert vault.load_state() == {}
    assert "not valid JSON" in caplog.text
    vault.state_path.write_text("[1, 2]")
    assert vault.load_state() == {}


def test_number_shaped_strings_are_quoted_for_obsidian():
    # Obsidian reads YAML 1.2, where an unquoted 0042 is the number 42.
    from huatbot.vault import dump_frontmatter
    text = dump_frontmatter({"first": "0042", "numbers": ["0698", "1234"], "draw": 4123, "x": "1e3"})
    assert "first: '0042'" in text
    assert "['0698', '1234']" in text
    assert "draw: 4123" in text
    assert "x: '1e3'" in text
