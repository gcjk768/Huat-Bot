"""Tests for the command line (huatbot.cli.main), all offline.

``cli.make_fetcher`` is replaced by a fake Singapore Pools site, Telegram by a recorder, and the
scheduler loop by a stub that captures the hooks ``serve`` wires up. The runner entry points get
a fixed clock (Thu 1 Oct 2026, 7.45pm), so the output does not depend on the day the tests run.

The 60 draw history ends on Thu 1 Oct 2026 (draw 4123, won), so the next draw is a normal draw
on Mon 5 Oct 2026 with four draws to go before the jackpot cascades.
"""
from __future__ import annotations

import functools
import inspect
from datetime import date, datetime, timedelta

import pytest

from huatbot import cli, runner, scheduler, telegram
from huatbot.http import FetchError
from huatbot.models import RunResult
from huatbot.report import SECTION_HEADINGS
from huatbot.store import load_toto, save_toto
from huatbot.textfmt import contains_dash
from huatbot.vault import SG, Vault
from tests.ctxgen import TOTO_LAST_DRAW, toto_history
from tests.htmlgen import FakeFetcher, fake_site

N_DRAWS = 60
NOW = datetime(2026, 10, 1, 19, 45, tzinfo=SG)
SETTINGS_NOTE = f"""---
toto_start_draw: {TOTO_LAST_DRAW - N_DRAWS + 1}
draw_notes_backfill: 2
---
# Settings for the tests
"""
ENV_NAMES = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "COMMENTARY", "DATA_DIR", "VAULT_FOLDER", "VAULT_PATH",
             "DRY_RUN", "LOG_LEVEL", "RUN_AT", "RETRY_MINUTES", "RETRY_HOURS", "ENV_FILE")


@pytest.fixture(scope="module")
def history():
    return toto_history(N_DRAWS)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _at_now(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        kw.setdefault("now", NOW)
        return fn(*args, **kw)
    return wrapper


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """The runner entry points the CLI calls run at NOW unless a time is passed."""
    for name in ("run", "fetch_data", "build_report", "refresh_next_draws"):
        monkeypatch.setattr(runner, name, _at_now(getattr(runner, name)))


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
def telegram_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")


class ClosingSite(FakeFetcher):
    """A fake site that remembers being closed."""

    closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def site(monkeypatch, history):
    """cli.make_fetcher returns a fresh fake site; the fetchers handed out are kept."""
    made: list[ClosingSite] = []

    def make():
        fetcher = ClosingSite(fake_site(history).pages)
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


def activity(vault: Vault) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in (vault.base / "Logs").glob("*Activity.md"))


# run


def test_run_dry_run_prints_two_messages(vault_path, site, sent, capsys):
    code = cli.main(["run", "--dry-run", "--vault", str(vault_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Message 1 of 2" in out and "Message 2 of 2" in out and "Message 3" not in out
    assert "<b>Next big prize</b>" in out
    assert "Nothing was posted to Telegram." in out
    assert f"New draws: {N_DRAWS} TOTO draws." in out
    vault = Vault(vault_path)
    assert f"Report: {vault.path('Reports/2026-10-01 1945 Report.md')}" in out
    assert out.rstrip().endswith("Done.")
    assert sent == []
    assert int(load_toto(vault.toto_csv)["draw_number"].max()) == TOTO_LAST_DRAW
    # no tickets yet, so no ledger.csv; nothing for 4D or backtests either
    assert sorted(p.name for p in vault.data_dir.iterdir()) == ["prize_rules.json", "state.json", "toto.csv"]
    assert vault.exists("Dashboard.md") and vault.exists("Tickets.md")
    assert site[0].closed  # the fetcher run made was closed


def test_dry_run_env_forces_a_dry_run(vault_path, site, sent, telegram_env, monkeypatch, capsys):
    monkeypatch.setenv("DRY_RUN", "1")
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert sent == []
    out = capsys.readouterr().out
    assert "Message 1 of 2" in out and "Nothing was posted to Telegram." in out


def test_run_posts_when_telegram_is_set_up(vault_path, site, sent, telegram_env, capsys):
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert len(sent) == 2
    assert all(not contains_dash(c["text"]) and c["parse_mode"] == "HTML" for c in sent)
    assert sent[1]["text"].startswith(f"<b>Next TOTO draw</b> (draw {TOTO_LAST_DRAW + 1}): Mon 5 Oct 2026, 6.30pm")
    out = capsys.readouterr().out
    assert "Posted 2 messages to Telegram." in out
    assert Vault(vault_path).load_state()["last_posted"] == {"toto": TOTO_LAST_DRAW}
    # a manual run posts again even though nothing is new
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert len(sent) == 4
    assert "No new draws." in capsys.readouterr().out


def test_run_without_telegram_is_a_dry_run_with_a_warning(vault_path, site, sent, capsys):
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    out = capsys.readouterr().out
    assert "Warning: Telegram is not set up (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing)" in out
    assert "Message 2 of 2" in out and "Nothing was posted to Telegram." in out
    assert sent == []


def test_run_flags_reach_the_runner(monkeypatch, tmp_path, offline, capsys):
    calls: list[tuple] = []

    def fake_run(*args, **kw):
        calls.append((args, kw))
        return RunResult(ok=kw["fetch"], new_draws=[4122, 4123])
    monkeypatch.setattr(runner, "run", fake_run)

    code = cli.main(["run", "--no-fetch", "--no-post", "--vault", str(tmp_path)])
    assert code == 1  # the fake reports a failure when fetch is off
    args, kw = calls[0]
    assert args == ()  # everything by keyword, and no games
    assert set(kw) == {"dry_run", "fetch", "post", "vault", "fetcher", "out", "force_post"}
    assert kw["fetch"] is False and kw["post"] is False and kw["dry_run"] is False
    assert kw["force_post"] is True and kw["fetcher"] is None
    assert kw["vault"].root == tmp_path
    out = capsys.readouterr().out
    assert "New draws: 2 TOTO draws." in out and "Finished with a problem" in out


def test_run_summary_prints_the_warnings(monkeypatch, tmp_path, offline, capsys):
    monkeypatch.setattr(runner, "run", lambda **kw: RunResult(ok=True, warnings=["Something to know."],
                                                              messages=["a", "b"], posted=True))
    assert cli.main(["run", "--no-fetch", "--vault", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["Warning: Something to know.", "No new draws.", "Posted 2 messages to Telegram.", "Done."]


def test_vault_flag_overrides_vault_path(monkeypatch, tmp_path, site):
    calls = []
    monkeypatch.setattr(runner, "run", lambda **kw: calls.append(kw) or RunResult(ok=True))
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "from env"))
    monkeypatch.setenv("VAULT_FOLDER", "Lottery")
    assert cli.main(["run"]) == 0
    assert calls[-1]["vault"].root == tmp_path / "from env"
    assert calls[-1]["vault"].folder == "Lottery"
    assert cli.main(["run", "--vault", str(tmp_path / "flag")]) == 0
    assert calls[-1]["vault"].root == tmp_path / "flag"
    assert calls[-1]["vault"].folder == "Lottery"


def test_run_that_fails_exits_1(vault_path, monkeypatch, sent, capsys):
    monkeypatch.setattr(cli, "make_fetcher", lambda: FakeFetcher({}))  # site down, nothing stored
    assert cli.main(["run", "--dry-run", "--vault", str(vault_path)]) == 1
    out = capsys.readouterr().out
    assert "Warning: Huat Bot could not run: no TOTO results are stored yet" in out
    assert "Finished with a problem: see the warnings above and the activity log in the vault." in out


@pytest.mark.parametrize("argv", [
    [], ["run", "--game", "toto"], ["run", "--game", "4d"], ["serve", "--game", "toto"], ["fetch", "--game", "TOTO"],
    ["dance"], ["run", "--bogus"], ["run", "toto"],
])
def test_bad_arguments_exit_1(argv, offline, capsys):
    assert cli.main(argv) == 1


def test_there_is_no_game_option(offline, capsys):
    assert cli.main(["run", "--game", "toto"]) == 1
    assert "unrecognized arguments: --game" in capsys.readouterr().err
    parser = cli.build_parser()
    texts = [parser.format_help()]
    sub = next(a for a in parser._actions if a.dest == "command")
    texts += [p.format_help() for p in sub.choices.values()]
    assert set(sub.choices) == {"run", "serve", "fetch", "report", "check-site", "demo", "init-vault"}
    for text in texts:
        assert "--game" not in text
        assert "4D" not in text and "suggest" not in text.lower()


def test_help_and_version_exit_0(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "check-site" in out and "TOTO" in out
    assert cli.main(["--version"]) == 0
    assert cli.main(["run", "--help"]) == 0
    assert "--dry-run" in capsys.readouterr().out


def test_unexpected_error_exits_1(monkeypatch, tmp_path, offline, capsys):
    def broken(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(runner, "run", broken)
    assert cli.main(["run", "--no-fetch", "--vault", str(tmp_path)]) == 1
    assert "Error: run failed (RuntimeError: disk on fire)" in capsys.readouterr().err


def test_ctrl_c_exits_1(monkeypatch, tmp_path, offline, capsys):
    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "run", stop)
    assert cli.main(["run", "--no-fetch", "--vault", str(tmp_path)]) == 1
    assert "Stopped." in capsys.readouterr().err


# fetch, report


def test_fetch_then_report(vault_path, site, capsys):
    assert cli.main(["fetch", "--vault", str(vault_path)]) == 0
    out = capsys.readouterr().out
    assert f"TOTO: {N_DRAWS} new draws." in out
    vault = Vault(vault_path)
    assert f"Data folder: {vault.data_dir}" in out
    assert vault.load_state()["next_draws"]["toto"]["draw_datetime"] == "2026-10-05T18:30:00+08:00"
    assert not vault.exists("Dashboard.md")  # fetch does not analyse or write notes
    assert site[0].closed

    site.clear()
    assert cli.main(["report", "--vault", str(vault_path)]) == 0
    assert site == []  # report never builds a fetcher
    out = capsys.readouterr().out
    positions = [out.index(h) for h in SECTION_HEADINGS]
    assert positions == sorted(positions)
    assert "Mon 5 Oct 2026" in out and "### The next big prize" in out
    assert "Suggested" not in out and "4D" not in out


def test_fetch_failure_exits_1(vault_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_fetcher", lambda: FakeFetcher({}))
    assert cli.main(["fetch", "--vault", str(vault_path)]) == 1
    out = capsys.readouterr().out
    assert "could not be fetched" in out and "TOTO could not be updated." in out


def test_fetch_with_no_fetch_does_nothing(vault_path, offline, capsys):
    assert cli.main(["fetch", "--no-fetch", "--vault", str(vault_path)]) == 1
    assert "Nothing to do" in capsys.readouterr().err


def test_report_with_no_data_exits_1(tmp_path, offline, capsys):
    assert cli.main(["report", "--vault", str(tmp_path)]) == 1
    assert "Run the fetch command first" in capsys.readouterr().err


# check-site, init-vault, demo


def test_check_site(monkeypatch, history, capsys):
    made: list[ClosingSite] = []
    monkeypatch.setattr(cli, "make_fetcher", lambda: made.append(ClosingSite(fake_site(history).pages)) or made[-1])
    assert cli.main(["check-site"]) == 0
    out = capsys.readouterr().out
    assert f"PASS  TOTO draw list: {N_DRAWS} draws listed, the latest is draw {TOTO_LAST_DRAW}" in out
    assert "PASS  TOTO latest result page" in out and "PASS  TOTO next draw page" in out
    assert "Summary: all 6 critical checks passed." in out
    assert "4D" not in out
    assert made[0].closed

    # the prize page is not critical: built in values are used without it
    monkeypatch.setattr(cli, "make_fetcher", lambda: fake_site(history, prize_pages=None))
    assert cli.main(["check-site"]) == 0
    assert "FAIL  TOTO prize structure page (not critical)" in capsys.readouterr().out

    monkeypatch.setattr(cli, "make_fetcher", lambda: FakeFetcher({}))
    assert cli.main(["check-site"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  TOTO draw list" in out and "Summary: 6 of 6 critical checks failed" in out


def test_init_vault(tmp_path, offline, capsys):
    root = tmp_path / "nas" / "Obsidian"
    assert cli.main(["init-vault", "--vault", str(root)]) == 0
    vault = Vault(root)
    assert vault.exists("Settings.md") and vault.exists("Tickets.md")
    assert vault.data_dir.is_dir() and vault.path("Draws/TOTO").is_dir()
    assert not vault.base.joinpath("Draws", "4D").exists() and not vault.base.joinpath("Suggestions").exists()
    out = capsys.readouterr().out
    assert f"Created {vault.path('Settings.md')}" in out and f"Created {vault.path('Tickets.md')}" in out
    assert "add your TOTO tickets to Tickets.md" in out
    assert "Created [[Settings]] (starter note)" in activity(vault)
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
    assert out.startswith(f"Demo run with synthetic data in the vault at {root}. Nothing is fetched and nothing "
                          "is posted.")
    assert "Message 1 of 2" in out and "Message 2 of 2" in out and "Message 3" not in out
    assert "WINNER!" in out
    assert f"New draws: {runner.DEMO_TOTO_DRAWS} TOTO draws." in out
    assert "Nothing was posted to Telegram." in out and out.rstrip().endswith("Done.")
    assert sent == []
    vault = Vault(root)
    assert vault.exists("Dashboard.md")
    assert len(load_toto(vault.toto_csv)) == runner.DEMO_TOTO_DRAWS


def test_demo_command_default_vault(tmp_path, offline, sent, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo"]) == 0
    assert "in the vault at demo-vault." in capsys.readouterr().out
    assert (tmp_path / cli.DEMO_VAULT / "Huat Bot" / "Dashboard.md").is_file()


# serve


def _capture_serve(monkeypatch) -> dict:
    hooks: dict = {}

    def fake_serve(run_fn, check_fn, refresh_fn, notify_fn, log_fn, now_fn, sleep_fn, config=None, **kw):
        hooks.update(run_fn=run_fn, check_fn=check_fn, refresh_fn=refresh_fn, notify_fn=notify_fn,
                     log_fn=log_fn, now_fn=now_fn, sleep_fn=sleep_fn, config=config, kwargs=kw, **kw)
    monkeypatch.setattr(scheduler, "serve", fake_serve)
    return hooks


def test_serve_wires_the_scheduler(vault_path, site, sent, monkeypatch, capsys):
    hooks = _capture_serve(monkeypatch)
    monkeypatch.setenv("RUN_AT", "20:15")

    assert cli.main(["serve", "--dry-run", "--vault", str(vault_path)]) == 0
    out = capsys.readouterr().out
    assert "Huat Bot scheduler started." in out and f"Vault folder: {Vault(vault_path).base}" in out
    assert hooks["config"].run_time.hour == 20 and hooks["config"].run_time.minute == 15
    assert hooks["now_fn"]().tzinfo is not None
    assert set(hooks["kwargs"]) == {"done_fn"}  # every game the scheduler knows: TOTO
    assert scheduler.GAMES == ("toto",)
    assert site[0].closed  # closed once the scheduler returned

    vault = Vault(vault_path)
    assert vault.exists("Settings.md") and vault.exists("Tickets.md")
    assert hooks["check_fn"]("toto") == date(2026, 10, 1)
    state = hooks["refresh_fn"]()
    assert state["next_draws"]["toto"]["draw_datetime"] == "2026-10-05T18:30:00+08:00"
    assert vault.load_state()["next_draws"] == state["next_draws"]

    assert hooks["notify_fn"]("TOTO result is late") is False  # dry run: printed, not sent
    assert "Notice (not posted): TOTO result is late" in capsys.readouterr().out
    hooks["log_fn"]("WAIT", "Result not out yet")
    assert "| WAIT | Result not out yet |" in activity(vault)

    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is False  # nothing stored yet
    result = hooks["run_fn"](games=("toto",))
    assert result.ok and not result.posted
    assert result.new_draws == list(range(TOTO_LAST_DRAW - N_DRAWS + 1, TOTO_LAST_DRAW + 1))
    assert "Message 1 of 2" in capsys.readouterr().out
    assert sent == []
    # a dry run posts nothing, so the draw is done once it is stored
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is True
    assert hooks["done_fn"]("toto", date(2026, 10, 5)) is False  # Monday's draw is not stored yet
    # the games argument is accepted and ignored: a plain call runs the same
    again = hooks["run_fn"]()
    assert again.ok and again.new_draws == []


def test_serve_check_fn_tells_an_unreachable_site_from_a_late_result(vault_path, history, monkeypatch):
    hooks = _capture_serve(monkeypatch)
    fetcher = fake_site(history)
    pages = fetcher.pages  # changed below, the fetcher serve holds sees the change
    monkeypatch.setattr(cli, "make_fetcher", lambda: fetcher)
    assert cli.main(["serve", "--dry-run", "--vault", str(vault_path)]) == 0
    assert hooks["check_fn"]("toto") == date(2026, 10, 1)

    # the newest result page lists no winning shares yet: not out, keep waiting
    from huatbot.fetch import toto_result_url
    from tests.htmlgen import toto_result_html
    row = history.iloc[-1].copy()
    for g in range(1, 8):
        row[f"g{g}_share"] = float("nan")
        row[f"g{g}_winners"] = 0
    pages[toto_result_url(TOTO_LAST_DRAW)] = toto_result_html(row)
    assert hooks["check_fn"]("toto") is None

    # the draw list cannot be fetched at all: the scheduler must hear about it
    from huatbot import constants as C
    pages[C.TOTO_DRAW_LIST_URL] = FetchError("the site is busy (HTTP 503)", status=503)
    with pytest.raises(FetchError):
        hooks["check_fn"]("toto")


def test_serve_notifies_on_telegram_when_set_up(vault_path, site, sent, telegram_env, monkeypatch):
    hooks = _capture_serve(monkeypatch)
    assert cli.main(["serve", "--vault", str(vault_path)]) == 0
    assert hooks["notify_fn"]("TOTO result is late") is True
    assert sent == [{"text": "TOTO result is late", "parse_mode": None}]


def test_serve_with_no_post_prints_notices(vault_path, site, sent, telegram_env, monkeypatch, capsys):
    hooks = _capture_serve(monkeypatch)
    assert cli.main(["serve", "--no-post", "--vault", str(vault_path)]) == 0
    assert hooks["notify_fn"]("TOTO result is late") is False
    assert sent == []
    result = hooks["run_fn"]()
    assert result.ok and not result.posted and sent == []
    assert "Message 1 of 2" not in capsys.readouterr().out  # --no-post: not even printed
    # posting is off, so a stored draw is done
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is True


def test_serve_refuses_no_fetch(vault_path, offline, capsys):
    assert cli.main(["serve", "--no-fetch", "--vault", str(vault_path)]) == 1
    assert "--no-fetch" in capsys.readouterr().err


def test_serve_stops_cleanly_on_ctrl_c(vault_path, site, monkeypatch, capsys):
    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(scheduler, "serve", stop)
    assert cli.main(["serve", "--dry-run", "--vault", str(vault_path)]) == 0
    assert "Scheduler stopped." in capsys.readouterr().out
    assert site[0].closed


def test_serve_prints_config_warnings(vault_path, site, monkeypatch, capsys):
    _capture_serve(monkeypatch)
    monkeypatch.setenv("RUN_AT", "teatime")
    assert cli.main(["serve", "--dry-run", "--vault", str(vault_path)]) == 0
    out = capsys.readouterr().out
    assert any(line.startswith("Warning: ") and "RUN_AT" in line for line in out.splitlines())


def test_log_level_is_forgiving(monkeypatch, tmp_path, offline):
    monkeypatch.setenv("LOG_LEVEL", "very loud")
    assert cli.main(["init-vault", "--vault", str(tmp_path)]) == 0


def test_env_flag(monkeypatch):
    for value, expected in (("1", True), ("true", True), ("Yes", True), ("on", True), ("0", False), ("", False),
                            ("no", False)):
        monkeypatch.setenv("DRY_RUN", value)
        assert cli.env_flag("DRY_RUN") is expected
    monkeypatch.delenv("DRY_RUN")
    assert cli.env_flag("DRY_RUN") is False


def test_serve_done_only_after_the_draw_is_posted(vault_path, site, sent, telegram_env, monkeypatch):
    hooks = _capture_serve(monkeypatch)
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
    assert len(sent) == 2
    assert hooks["done_fn"]("toto", date(2026, 10, 1)) is True


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
               sleep, scheduler.SchedulerConfig(), max_cycles=1, **hooks["kwargs"])
    assert runs == [(datetime(2026, 10, 1, 19, 30, tzinfo=SG), ("toto",))]
    assert notices == [] and sent == []
    assert "TOTO draw on Thu 1 Oct 2026, checking for the result" in activity(Vault(vault_path))


# draw_done


def test_draw_done(tmp_path, history):
    vault = Vault(tmp_path / "vault")
    vault.ensure_layout()
    assert "game" in inspect.signature(cli.draw_done).parameters
    day = date(2026, 10, 1)
    assert cli.draw_done(vault, "toto", day, posting=False) is False  # nothing stored
    save_toto(history, vault.toto_csv)
    assert cli.draw_done(vault, "toto", day, posting=False) is True
    assert cli.draw_done(vault, "toto", date(2026, 9, 28), posting=False) is True  # a later draw is stored
    assert cli.draw_done(vault, "toto", date(2026, 10, 5), posting=False) is False  # not stored yet
    assert cli.draw_done(vault, "toto", day, posting=True) is False  # stored, not posted
    for posted, expected in (({"toto": TOTO_LAST_DRAW}, True), ({"toto": TOTO_LAST_DRAW + 1}, True),
                             ({"toto": TOTO_LAST_DRAW - 1}, False), ({"toto": "soon"}, False), ({}, False),
                             ({"4d": TOTO_LAST_DRAW}, False)):
        vault.save_state({"last_posted": posted})
        assert cli.draw_done(vault, "toto", day, posting=True) is expected, posted


# .env


def test_load_dotenv_reads_file_without_overriding(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Telegram\n"
        "TELEGRAM_BOT_TOKEN=123:abc\n"
        "export TELEGRAM_CHAT_ID='-1001234'\n"
        "VAULT_FOLDER=\"Huat Bot\"\n"
        "RUN_AT=19:30  # local time\n"
        "DRY_RUN=1\n"
        "BAD KEY=1\n"
        "not a line\n",
        encoding="utf-8",
    )
    env = {"DRY_RUN": "0"}
    loaded = cli.load_dotenv(env_file, environ=env)
    assert env["TELEGRAM_BOT_TOKEN"] == "123:abc"
    assert env["TELEGRAM_CHAT_ID"] == "-1001234"
    assert env["VAULT_FOLDER"] == "Huat Bot"
    assert env["RUN_AT"] == "19:30"
    assert env["DRY_RUN"] == "0"  # an existing value wins
    assert loaded == ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "VAULT_FOLDER", "RUN_AT"]
    assert "BAD KEY" not in env


def test_load_dotenv_can_be_switched_off(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=x\n", encoding="utf-8")
    env = {"HUATBOT_NO_DOTENV": "1"}
    assert cli.load_dotenv(env_file, environ=env) == []
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert cli.load_dotenv(tmp_path / "missing.env", environ={}) == []
    # ENV_FILE names the file when no path is given
    assert cli.load_dotenv(environ={"ENV_FILE": str(env_file)}) == ["TELEGRAM_BOT_TOKEN"]


def test_main_reads_the_env_file(vault_path, site, sent, telegram_env, tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "bot.env"
    env_file.write_text("DRY_RUN=1\n", encoding="utf-8")
    monkeypatch.delenv("HUATBOT_NO_DOTENV")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("DRY_RUN", "x")
    monkeypatch.delenv("DRY_RUN")  # recorded, so the value main loads is removed after the test
    assert cli.main(["run", "--vault", str(vault_path)]) == 0
    assert sent == []  # DRY_RUN=1 from the file
    assert "Message 1 of 2" in capsys.readouterr().out
