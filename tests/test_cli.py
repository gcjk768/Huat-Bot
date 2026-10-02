"""Tests for the command line (huatbot.cli.main), all offline.

``cli.make_fetcher`` is replaced by a fake Singapore Pools site, Telegram by a recorder, and the
scheduler loop by a stub that captures the hooks ``serve`` wires up.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from huatbot import cli, runner, scheduler, telegram
from huatbot.models import RunResult
from huatbot.report import SECTION_HEADINGS
from huatbot.store import load_fourd, load_toto
from huatbot.textfmt import contains_dash
from huatbot.vault import SG, Vault
from tests.ctxgen import FOURD_LAST_DRAW, TOTO_LAST_DRAW, fourd_history, toto_history
from tests.htmlgen import FakeFetcher, fake_site

N_DRAWS = 60
SETTINGS_NOTE = f"""---
toto_start_draw: {TOTO_LAST_DRAW - N_DRAWS + 1}
fourd_history_draws: {N_DRAWS}
backtest_draws: 10
random_sets_per_draw: 20
draw_notes_backfill: 2
---
# Settings for the tests
"""


@pytest.fixture(scope="module")
def history():
    return toto_history(N_DRAWS), fourd_history(N_DRAWS)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "COMMENTARY", "DATA_DIR", "VAULT_FOLDER",
                 "VAULT_PATH", "DRY_RUN", "LOG_LEVEL", "RUN_AT", "RETRY_MINUTES", "RETRY_HOURS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sent(monkeypatch):
    """Telegram calls are recorded, never sent."""
    calls: list[dict] = []

    def fake_send(token, chat_id, text, parse_mode="HTML", **kw):
        calls.append({"text": text, "parse_mode": parse_mode})
        return {"ok": True}
    monkeypatch.setattr(telegram, "send_message", fake_send)
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)
    return calls


@pytest.fixture
def site(monkeypatch, history):
    """cli.make_fetcher returns a fresh fake site; the fetchers handed out are kept."""
    made: list[FakeFetcher] = []

    def make():
        fetcher = fake_site(*history)
        made.append(fetcher)
        return fetcher
    monkeypatch.setattr(cli, "make_fetcher", make)
    return made


@pytest.fixture
def offline(monkeypatch):
    """Any attempt to build a fetcher fails the test."""
    def make():
        raise AssertionError("this command must not touch the network")
    monkeypatch.setattr(cli, "make_fetcher", make)


@pytest.fixture
def vault_path(tmp_path):
    root = tmp_path / "vault"
    (root / "Huat Bot").mkdir(parents=True)
    (root / "Huat Bot" / "Settings.md").write_text(SETTINGS_NOTE, encoding="utf-8")
    return root


# run


def test_run_dry_run_prints_three_messages(vault_path, site, sent, capsys):
    code = cli.main(["run", "--dry-run", "--vault", str(vault_path)])
    out = capsys.readouterr().out
    assert code == 0
    for i in (1, 2, 3):
        assert f"Message {i} of 3" in out
    assert "Nothing was posted to Telegram." in out
    assert f"New draws: TOTO {N_DRAWS} draws, 4D {N_DRAWS} draws." in out
    assert sent == []
    vault = Vault(vault_path)
    assert int(load_toto(vault.toto_csv)["draw_number"].max()) == TOTO_LAST_DRAW
    assert int(load_fourd(vault.fourd_csv)["draw_number"].max()) == FOURD_LAST_DRAW
    assert vault.exists("Dashboard.md") and vault.exists("Tickets.md")


def test_dry_run_env_forces_a_dry_run(vault_path, site, sent, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("DRY_RUN", "1")
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert sent == []
    assert "Message 1 of 3" in capsys.readouterr().out


def test_run_posts_when_telegram_is_set_up(vault_path, site, sent, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert len(sent) == 3
    assert all(not contains_dash(c["text"]) for c in sent)
    assert "Posted 3 messages to Telegram." in capsys.readouterr().out
    # a manual run posts again even though nothing is new
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert len(sent) == 6


def test_run_flags_reach_the_runner(monkeypatch, tmp_path, offline, capsys):
    calls: list[tuple] = []

    def fake_run(games, **kw):
        calls.append((games, kw))
        return RunResult(ok=kw["fetch"], new_draws={"toto": [1, 2]})
    monkeypatch.setattr(runner, "run", fake_run)

    code = cli.main(["run", "--game", "TOTO", "--no-fetch", "--no-post", "--vault", str(tmp_path)])
    assert code == 1  # the fake reports a failure when fetch is off
    games, kw = calls[0]
    assert games == ("toto",)
    assert kw["fetch"] is False and kw["post"] is False and kw["dry_run"] is False
    assert kw["force_post"] is True and kw["fetcher"] is None
    assert kw["vault"].root == tmp_path
    out = capsys.readouterr().out
    assert "New draws: TOTO 2 draws." in out and "Finished with a problem" in out


def test_vault_flag_overrides_vault_path(monkeypatch, tmp_path, site):
    calls = []
    monkeypatch.setattr(runner, "run", lambda games, **kw: calls.append(kw) or RunResult(ok=True))
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "from env"))
    monkeypatch.setenv("VAULT_FOLDER", "Lottery")
    assert cli.main(["run", "--game", "4d"]) == 0
    assert calls[-1]["vault"].root == tmp_path / "from env"
    assert calls[-1]["vault"].folder == "Lottery"
    assert cli.main(["run", "--vault", str(tmp_path / "flag")]) == 0
    assert calls[-1]["vault"].root == tmp_path / "flag"
    assert calls[-1]["vault"].folder == "Lottery"


@pytest.mark.parametrize("argv", [[], ["run", "--game", "lotto"], ["dance"], ["run", "--bogus"]])
def test_bad_arguments_exit_1(argv, capsys):
    assert cli.main(argv) == 1


def test_help_and_version_exit_0(capsys):
    assert cli.main(["--help"]) == 0
    assert "check-site" in capsys.readouterr().out
    assert cli.main(["--version"]) == 0
    assert cli.main(["run", "--help"]) == 0
    assert "--dry-run" in capsys.readouterr().out


def test_unexpected_error_exits_1(monkeypatch, tmp_path, offline, capsys):
    def broken(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(runner, "run", broken)
    assert cli.main(["run", "--no-fetch", "--vault", str(tmp_path)]) == 1
    assert "disk on fire" in capsys.readouterr().err


# fetch, report


def test_fetch_then_report(vault_path, site, capsys):
    assert cli.main(["fetch", "--vault", str(vault_path)]) == 0
    out = capsys.readouterr().out
    assert f"TOTO {N_DRAWS} new draws, 4D {N_DRAWS} new draws." in out
    vault = Vault(vault_path)
    assert vault.load_state()["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-05")
    assert not vault.exists("Dashboard.md")  # fetch does not analyse or write notes

    site.clear()
    assert cli.main(["report", "--vault", str(vault_path)]) == 0
    assert site == []  # report never builds a fetcher
    out = capsys.readouterr().out
    for heading in SECTION_HEADINGS:
        assert heading in out
    assert "Mon 5 Oct 2026" in out


def test_fetch_failure_exits_1(vault_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_fetcher", lambda: FakeFetcher({}))
    assert cli.main(["fetch", "--vault", str(vault_path)]) == 1
    assert "could not be fetched" in capsys.readouterr().out


def test_report_with_no_data_exits_1(tmp_path, offline, capsys):
    assert cli.main(["report", "--vault", str(tmp_path)]) == 1
    assert "Run the fetch command first" in capsys.readouterr().err


# check-site, init-vault, demo


def test_check_site(monkeypatch, history, capsys):
    monkeypatch.setattr(cli, "make_fetcher", lambda: fake_site(*history))
    assert cli.main(["check-site"]) == 0
    out = capsys.readouterr().out
    assert "PASS  TOTO draw list" in out and "Summary: all" in out
    monkeypatch.setattr(cli, "make_fetcher", lambda: FakeFetcher({}))
    assert cli.main(["check-site"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_init_vault(tmp_path, offline, capsys):
    root = tmp_path / "nas" / "Obsidian"
    assert cli.main(["init-vault", "--vault", str(root)]) == 0
    vault = Vault(root)
    assert vault.exists("Settings.md") and vault.exists("Tickets.md")
    assert vault.data_dir.is_dir() and vault.path("Draws/TOTO").is_dir()
    assert "Created" in capsys.readouterr().out
    assert cli.main(["init-vault", "--vault", str(root)]) == 0
    assert "nothing new was needed" in capsys.readouterr().out
    assert not (root / ".obsidian").exists()


def test_demo_command(tmp_path, offline, sent, capsys):
    root = tmp_path / "demo"
    (root / "Huat Bot").mkdir(parents=True)
    settings = "".join(line for line in SETTINGS_NOTE.splitlines(keepends=True) if "toto_start_draw" not in line)
    (root / "Huat Bot" / "Settings.md").write_text(settings, encoding="utf-8")
    assert cli.main(["demo", "--vault", str(root)]) == 0
    out = capsys.readouterr().out
    assert "synthetic data" in out
    for i in (1, 2, 3):
        assert f"Message {i} of 3" in out
    assert sent == []
    assert Vault(root).exists("Dashboard.md")


# serve


def test_serve_wires_the_scheduler(vault_path, site, sent, monkeypatch, capsys):
    hooks: dict = {}

    def fake_serve(run_fn, check_fn, refresh_fn, notify_fn, log_fn, now_fn, sleep_fn, config=None, max_cycles=None,
                   done_fn=None, games=None):
        hooks.update(run_fn=run_fn, check_fn=check_fn, refresh_fn=refresh_fn, notify_fn=notify_fn,
                     log_fn=log_fn, now_fn=now_fn, sleep_fn=sleep_fn, config=config, done_fn=done_fn, games=games)
    monkeypatch.setattr(scheduler, "serve", fake_serve)
    monkeypatch.setenv("RUN_AT", "20:15")

    assert cli.main(["serve", "--dry-run", "--vault", str(vault_path)]) == 0
    assert "scheduler started" in capsys.readouterr().out
    assert hooks["config"].run_time.hour == 20 and hooks["config"].run_time.minute == 15
    assert hooks["now_fn"]().tzinfo is not None

    vault = Vault(vault_path)
    assert hooks["check_fn"]("toto") == date(2026, 10, 1)
    assert hooks["check_fn"]("4d") == date(2026, 9, 30)
    state = hooks["refresh_fn"]()
    assert state["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-05")
    assert vault.load_state()["next_draws"] == state["next_draws"]

    assert hooks["notify_fn"]("TOTO result is late") is False  # dry run: printed, not sent
    assert "Notice (not posted): TOTO result is late" in capsys.readouterr().out
    hooks["log_fn"]("WAIT", "Result not out yet")
    log_text = (vault.base / "Logs").glob("*Activity.md")
    assert any("Result not out yet" in p.read_text() for p in log_text)

    assert hooks["games"] == ("toto", "4d")
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is False  # nothing stored yet
    result = hooks["run_fn"](games=("toto",))
    assert result.ok and not result.posted
    assert result.new_draws["toto"]
    assert "Message 1 of 3" in capsys.readouterr().out
    assert sent == []
    # a dry run posts nothing, so the draw is done once it is stored
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is True
    assert hooks["done_fn"]("toto", date(2026, 10, 5)) is False  # Monday's draw is not stored yet


def test_serve_notifies_on_telegram_when_set_up(vault_path, site, sent, monkeypatch):
    hooks: dict = {}
    monkeypatch.setattr(scheduler, "serve", lambda run_fn, check_fn, refresh_fn, notify_fn, *a, **k:
                        hooks.update(notify_fn=notify_fn))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert cli.main(["serve", "--vault", str(vault_path)]) == 0
    assert hooks["notify_fn"]("4D result is late") is True
    assert sent == [{"text": "4D result is late", "parse_mode": None}]


def test_serve_refuses_no_fetch(vault_path, offline, capsys):
    assert cli.main(["serve", "--no-fetch", "--vault", str(vault_path)]) == 1
    assert "--no-fetch" in capsys.readouterr().err


def test_log_level_is_forgiving(monkeypatch, tmp_path, offline):
    monkeypatch.setenv("LOG_LEVEL", "very loud")
    assert cli.main(["init-vault", "--vault", str(tmp_path)]) == 0


def test_env_flag(monkeypatch):
    for value, expected in (("1", True), ("true", True), ("Yes", True), ("0", False), ("", False), ("no", False)):
        monkeypatch.setenv("DRY_RUN", value)
        assert cli.env_flag("DRY_RUN") is expected


def _capture_serve(monkeypatch) -> dict:
    hooks: dict = {}

    def fake_serve(run_fn, check_fn, refresh_fn, notify_fn, log_fn, now_fn, sleep_fn, config=None, **kw):
        hooks.update(run_fn=run_fn, check_fn=check_fn, refresh_fn=refresh_fn, notify_fn=notify_fn,
                     log_fn=log_fn, config=config, **kw)
    monkeypatch.setattr(scheduler, "serve", fake_serve)
    return hooks


def test_serve_done_only_after_the_draw_is_posted(vault_path, site, sent, monkeypatch):
    hooks = _capture_serve(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert cli.main(["serve", "--vault", str(vault_path)]) == 0
    real_send = telegram.send_message

    def down(*a, **k):
        raise telegram.TelegramError("Telegram is down")
    monkeypatch.setattr(telegram, "send_message", down)
    result = hooks["run_fn"](games=("toto",))
    assert not result.posted
    # stored but not posted: the scheduler must run it again
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is False
    monkeypatch.setattr(telegram, "send_message", real_send)
    assert hooks["run_fn"](games=("toto",)).posted
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is True


def test_serve_game_option_limits_the_scheduler(vault_path, site, monkeypatch, capsys):
    hooks = _capture_serve(monkeypatch)
    assert cli.main(["serve", "--dry-run", "--game", "toto", "--vault", str(vault_path)]) == 0
    assert hooks["games"] == ("toto",)
    assert "Only TOTO is followed." in capsys.readouterr().out


def test_serve_dry_run_cycle_runs_once_and_sends_no_notice(vault_path, site, sent, monkeypatch, capsys):
    """The real scheduler loop with the hooks serve wires up, on a fake clock (Thu 1 Oct 2026,
    the latest TOTO draw on the fake site): one run, no retry and no notice."""
    real_serve = scheduler.serve
    hooks = _capture_serve(monkeypatch)
    assert cli.main(["serve", "--dry-run", "--vault", str(vault_path)]) == 0
    clock = {"t": datetime(2026, 10, 1, 10, 0, tzinfo=SG)}

    def sleep(seconds):
        clock["t"] += timedelta(seconds=seconds)
    runs, notices = [], []

    def run_fn(games):
        runs.append((clock["t"], games))
        return hooks["run_fn"](games=games)

    def notify_fn(text):
        notices.append(text)
        return hooks["notify_fn"](text)
    real_serve(run_fn, hooks["check_fn"], hooks["refresh_fn"], notify_fn, hooks["log_fn"], lambda: clock["t"],
               sleep, scheduler.SchedulerConfig(), max_cycles=1, done_fn=hooks["done_fn"], games=hooks["games"])
    assert runs == [(datetime(2026, 10, 1, 19, 30, tzinfo=SG), ("toto",))]
    assert notices == [] and sent == []


def test_load_dotenv_reads_file_without_overriding(tmp_path):
    from huatbot.cli import load_dotenv
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Telegram\n"
        "TELEGRAM_BOT_TOKEN=123:abc\n"
        "export TELEGRAM_CHAT_ID='-1001234'\n"
        "VAULT_FOLDER=\"Huat Bot\"\n"
        "RUN_AT=19:30  # local time\n"
        "DRY_RUN=1\n"
        "not a line\n",
        encoding="utf-8",
    )
    env = {"DRY_RUN": "0"}
    loaded = load_dotenv(env_file, environ=env)
    assert env["TELEGRAM_BOT_TOKEN"] == "123:abc"
    assert env["TELEGRAM_CHAT_ID"] == "-1001234"
    assert env["VAULT_FOLDER"] == "Huat Bot"
    assert env["RUN_AT"] == "19:30"
    assert env["DRY_RUN"] == "0"  # an existing value wins
    assert "DRY_RUN" not in loaded


def test_load_dotenv_can_be_switched_off(tmp_path):
    from huatbot.cli import load_dotenv
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=x\n", encoding="utf-8")
    env = {"HUATBOT_NO_DOTENV": "1"}
    assert load_dotenv(env_file, environ=env) == []
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert load_dotenv(tmp_path / "missing.env", environ={}) == []
