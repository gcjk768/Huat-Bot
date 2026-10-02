"""End to end tests of runner.run against an offline fake Singapore Pools site.

The fake site (tests.htmlgen.fake_site) serves result pages rendered from synthetic rows, so the
whole pipeline runs for real: fetch and parse, CSVs in the vault, analysis, backtests, tickets,
notes, the activity log, state.json and the three Telegram messages. Telegram itself is faked.
"""
from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from huatbot import backtest, runner, telegram
from huatbot import constants as C
from huatbot.fetch import fourd_result_url, toto_result_url
from huatbot.http import FetchError
from huatbot.models import PrizeRules, Settings
from huatbot.report import SECTION_HEADINGS
from huatbot.store import load_fourd, load_ledger, load_toto, save_toto
from huatbot.synth import SG, synth_next_fourd, synth_next_toto, synth_toto
from huatbot.textfmt import contains_dash
from huatbot.vault import Vault, log_note_path
from tests.ctxgen import (
    FOURD_LAST_DRAW,
    TOTO_LAST_DATE,
    TOTO_LAST_DRAW,
    fourd_history,
    start_date_for,
    tickets_markdown,
    toto_history,
)
from tests.htmlgen import FakeFetcher, fake_site

N_DRAWS = 150
NOW = datetime(2026, 10, 1, 19, 45, tzinfo=SG)  # Thu, after the TOTO draw 4123
NEXT_TOTO_DAY = date(2026, 10, 5)  # Mon
NEXT_FOURD_DAY = date(2026, 10, 3)  # Sat
FIRST_TOTO = TOTO_LAST_DRAW - N_DRAWS + 1
FIRST_FOURD = FOURD_LAST_DRAW - N_DRAWS + 1

SETTINGS_NOTE = f"""---
toto_start_draw: {FIRST_TOTO}
fourd_history_draws: {N_DRAWS}
backtest_draws: 20
random_sets_per_draw: 30
draw_notes_backfill: 5
---
# Settings for the tests
"""

TOTO_COLS = ["draw_number", "draw_date"] + [f"n{i}" for i in range(1, 7)] + ["additional", "draw_type"] + [
    f"g{g}_winners" for g in range(1, 8)
]


# fixtures


@pytest.fixture(scope="module")
def toto():
    return toto_history(N_DRAWS)


@pytest.fixture(scope="module")
def fourd():
    return fourd_history(N_DRAWS)


ENV_NAMES = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "COMMENTARY", "DATA_DIR", "VAULT_FOLDER", "VAULT_PATH",
             "DRY_RUN")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.fail = False

    def send_message(self, token, chat_id, text, parse_mode="HTML", **kw):
        if self.fail:
            raise telegram.TelegramError("Telegram refused the message: chat not found")
        self.sent.append({"token": token, "chat_id": chat_id, "text": text, "parse_mode": parse_mode})
        return {"ok": True, "result": {"message_id": len(self.sent)}}


@pytest.fixture
def tg(monkeypatch):
    """Telegram configured, but every send is recorded instead of going out."""
    fake = FakeTelegram()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@huat")
    monkeypatch.setattr(telegram, "send_message", fake.send_message)
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)
    return fake


@pytest.fixture
def no_send(monkeypatch):
    """Fail loudly if anything tries to reach Telegram."""
    def boom(*a, **k):
        raise AssertionError("Telegram must not be called")
    monkeypatch.setattr(telegram, "send_message", boom)


def make_vault(tmp_path, toto, fourd, tickets_text: str | None = "auto") -> Vault:
    vault = Vault(tmp_path / "vault")
    vault.base.mkdir(parents=True)
    (vault.base / "Settings.md").write_text(SETTINGS_NOTE, encoding="utf-8")
    if tickets_text == "auto":
        tickets_text = tickets_markdown(toto, fourd, NEXT_TOTO_DAY, NEXT_FOURD_DAY)
    if tickets_text is not None:
        (vault.base / "Tickets.md").write_text(tickets_text, encoding="utf-8")
    return vault


@pytest.fixture(scope="module")
def first_run(tmp_path_factory, toto, fourd):
    """One dry run from an empty vault against the fake site (shared, never modified by tests)."""
    root = tmp_path_factory.mktemp("first")
    printed: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        for name in ENV_NAMES:
            mp.delenv(name, raising=False)

        def boom(*a, **k):
            raise AssertionError("Telegram must not be called")
        mp.setattr(telegram, "send_message", boom)
        vault = make_vault(root, toto, fourd)
        result = runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto, fourd), now=NOW,
                            out=printed.append, force_post=True)
    return vault, result, printed


@pytest.fixture
def seeded(first_run, tmp_path) -> Vault:
    """A private copy of the vault as it was after the first run."""
    src = first_run[0]
    shutil.copytree(src.root, tmp_path / "vault")
    return Vault(tmp_path / "vault")


def log_rows(vault: Vault, when: datetime = NOW) -> list[tuple[str, str]]:
    text = vault.read_text(log_note_path(when)) or ""
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| Time") or line.startswith("| ---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        rows.append((cells[1], cells[2]))
    return rows


def events(vault: Vault, when: datetime = NOW) -> list[str]:
    return [e for e, _ in log_rows(vault, when)]


def result_pages(fetcher: FakeFetcher) -> list[str]:
    return [u for u in fetcher.requested if "toto_results.aspx" in u or "4d_results.aspx" in u]


def raising_fetcher():
    class Raising:
        def get(self, url):
            raise AssertionError(f"no network expected, asked for {url}")

        def get_many(self, urls):
            raise AssertionError("no network expected")
    return Raising()


def toto_151() -> pd.DataFrame:
    """The same synthetic history with one more draw (4124 on Mon 5 Oct 2026)."""
    start = start_date_for(TOTO_LAST_DATE, C.TOTO_WEEKDAYS, N_DRAWS)
    return synth_toto(n_draws=N_DRAWS + 1, start_draw=FIRST_TOTO, start_date=start, seed=7)


# end to end


def test_first_run_dry_run_end_to_end(first_run, toto, fourd):
    vault, result, printed = first_run

    assert result.ok, result.warnings
    assert not result.posted
    assert result.new_draws == {"toto": list(range(FIRST_TOTO, TOTO_LAST_DRAW + 1)),
                                "4d": list(range(FIRST_FOURD, FOURD_LAST_DRAW + 1))}
    assert not any("could not be worked out" in w for w in result.warnings)

    # three messages printed, none posted, all within the rules
    out = "\n".join(printed)
    for i in (1, 2, 3):
        assert f"Message {i} of 3" in out
    assert len(result.messages) == 3
    for m in result.messages:
        assert m in out
        assert not contains_dash(m)
        assert len(m) <= C.TELEGRAM_MAX_CHARS
    assert "WINNER" in result.messages[0]
    assert f"TOTO draw {TOTO_LAST_DRAW}" in result.messages[0]
    assert f"4D draw {FOURD_LAST_DRAW}" in result.messages[0]
    assert "Buy signal" in result.messages[1]
    assert "Suggested numbers" in result.messages[2]

    # CSVs in the vault match the site
    saved = load_toto(vault.toto_csv)
    pd.testing.assert_frame_equal(saved[TOTO_COLS], toto[TOTO_COLS], check_dtype=False)
    saved4 = load_fourd(vault.fourd_csv)
    cols = ["draw_number", "first", "second", "third", "starter_1", "consolation_10"]
    pd.testing.assert_frame_equal(saved4[cols], fourd[cols], check_dtype=False)

    # tickets: the Group 7 ticket and the 4D Big $2 ticket on the 1st prize were settled
    ledger = load_ledger(vault.ledger_csv)
    settled = ledger[ledger["status"] == "settled"]
    assert len(settled) >= 3
    fourd_win = settled[(settled["game"] == "4D") & (settled["bet_type"] == "Big")]
    assert fourd_win["winnings"].tolist() == [2 * 2000.0]
    toto_win = settled[(settled["game"] == "TOTO") & (settled["result"] == "Group 7 x1")]
    assert 10.0 in toto_win["winnings"].tolist()
    assert (ledger["status"] == "pending").sum() >= 3  # the tickets for the next draws

    # notes
    for rel in ("Dashboard.md", "Ledger.md", "Reports/2026-10-01 1945 Report.md",
                f"Suggestions/2026-10-05 TOTO {TOTO_LAST_DRAW + 1}.md",
                f"Suggestions/2026-10-03 4D {FOURD_LAST_DRAW + 1}.md",
                f"Draws/TOTO/2026-10-01 TOTO {TOTO_LAST_DRAW}.md",
                f"Draws/4D/2026-09-30 4D {FOURD_LAST_DRAW}.md"):
        assert vault.exists(rel), rel
    assert len(list(vault.path("Draws/TOTO").glob("*.md"))) == 5  # draw_notes_backfill
    assert result.report_path == str(vault.path("Reports/2026-10-01 1945 Report.md"))
    report_text = vault.read_text("Reports/2026-10-01 1945 Report.md")
    positions = [report_text.index(h) for h in SECTION_HEADINGS]
    assert positions == sorted(positions)

    # state.json
    state = vault.load_state()
    assert state["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-05T18:30")
    assert state["next_draws"]["4d"]["draw_datetime"].startswith("2026-10-03T18:30")
    assert "last_posted" not in state
    assert state["last_run"]["ok"] is True and state["last_run"]["dry_run"] is True

    # caches
    assert vault.prize_rules_path.exists()
    assert vault.backtest_cache_path.exists()

    # activity log: a row for every kind of action, no errors
    seen = set(events(vault))
    for ev in ("RUN", "SETTINGS", "FETCH", "NEW DRAW", "BACKTEST", "TICKETS", "LEDGER", "SUGGEST", "SIGNAL",
               "NOTE", "DRY RUN"):
        assert ev in seen, ev
    assert "ERROR" not in seen
    assert "POST" not in seen
    rows = log_rows(vault)
    assert rows[0][0] == "RUN" and rows[-1][0] == "RUN"
    assert any(e == "NEW DRAW" and f"TOTO draw {TOTO_LAST_DRAW} on Thu 1 Oct 2026" in d for e, d in rows)


def test_posting_is_idempotent_and_a_second_run_fetches_nothing(seeded, toto, fourd, tg):
    vault = seeded
    # nothing was posted by the dry run, so the first scheduled run posts the stored draws
    first = runner.run(vault=vault, fetcher=fake_site(toto, fourd), now=NOW + timedelta(minutes=5),
                       out=lambda s: None)
    assert first.ok and first.posted
    assert len(tg.sent) == 3
    assert [s["text"] for s in tg.sent] == first.messages
    assert all(s["chat_id"] == "@huat" for s in tg.sent)
    assert vault.load_state()["last_posted"] == {"toto": TOTO_LAST_DRAW, "4d": FOURD_LAST_DRAW}
    assert "POST" in events(vault)

    # scheduled rerun half an hour later: nothing new on the site, nothing posted again
    fetcher = fake_site(toto, fourd)
    second = runner.run(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=30), out=lambda s: None)
    assert second.ok and not second.posted
    assert second.new_draws == {"toto": [], "4d": []}
    assert result_pages(fetcher) == []
    assert len(tg.sent) == 3
    rows = log_rows(vault)
    assert any(e == "POST" and "Nothing new since the last post" in d for e, d in rows)
    assert any(e == "BACKTEST" and "reused from the cache" in d for e, d in rows)

    # a manual run posts again on purpose
    third = runner.run(vault=vault, fetcher=fake_site(toto, fourd), now=NOW + timedelta(minutes=40),
                       out=lambda s: None, force_post=True)
    assert third.posted and len(tg.sent) == 6


def test_new_draw_is_fetched_alone_and_posted(seeded, toto, fourd, tg):
    vault = seeded
    assert runner.run(vault=vault, fetcher=fake_site(toto, fourd), now=NOW, out=lambda s: None).posted
    sent_before = len(tg.sent)

    bigger = toto_151()
    pd.testing.assert_frame_equal(bigger.iloc[:N_DRAWS][TOTO_COLS], toto[TOTO_COLS], check_dtype=False)
    fetcher = fake_site(bigger, fourd)
    monday = datetime(2026, 10, 5, 19, 45, tzinfo=SG)

    result = runner.run(("toto",), vault=vault, fetcher=fetcher, now=monday, out=lambda s: None)

    assert result.ok and result.posted
    assert result.new_draws == {"toto": [TOTO_LAST_DRAW + 1]}
    assert result_pages(fetcher) == [toto_result_url(TOTO_LAST_DRAW + 1)]
    assert len(load_toto(vault.toto_csv)) == N_DRAWS + 1
    assert len(tg.sent) == sent_before + 3
    assert f"TOTO draw {TOTO_LAST_DRAW + 1}" in result.messages[0]
    assert f"4D draw {FOURD_LAST_DRAW}" not in result.messages[0]  # only TOTO was drawn
    assert vault.load_state()["last_posted"] == {"toto": TOTO_LAST_DRAW + 1, "4d": FOURD_LAST_DRAW}
    assert vault.exists(f"Draws/TOTO/2026-10-05 TOTO {TOTO_LAST_DRAW + 1}.md")
    # the tickets bought for Monday's draw are checked now
    ledger = load_ledger(vault.ledger_csv)
    monday_rows = ledger[(ledger["game"] == "TOTO") & (ledger["draw_date"] == "2026-10-05")]
    assert len(monday_rows) == 2 and set(monday_rows["status"]) == {"settled"}
    assert any(e == "NEW DRAW" and f"TOTO draw {TOTO_LAST_DRAW + 1}" in d for e, d in log_rows(vault, monday))


def test_site_down_uses_the_stored_data(seeded, no_send):
    vault = seeded
    printed: list[str] = []
    result = runner.run(dry_run=True, vault=vault, fetcher=FakeFetcher({}), now=NOW + timedelta(hours=1),
                        out=printed.append, force_post=True)

    assert result.ok
    assert len(result.messages) == 3
    assert "Message 3 of 3" in "\n".join(printed)
    assert any("could not be fetched" in w and "stored draw" in w for w in result.warnings)
    assert "ERROR" in events(vault)
    assert len(load_toto(vault.toto_csv)) == N_DRAWS
    # the next draws fall back to what state.json remembered
    assert "Mon 5 Oct 2026" in result.messages[1]


def test_site_down_and_nothing_stored_notifies_and_fails(tmp_path, toto, fourd, tg):
    vault = make_vault(tmp_path, toto, fourd, tickets_text=None)
    result = runner.run(vault=vault, fetcher=FakeFetcher({}), now=NOW, out=lambda s: None)

    assert not result.ok
    assert result.messages == []
    assert len(tg.sent) == 1
    notice = tg.sent[0]
    assert notice["parse_mode"] is None
    assert "no TOTO and 4D results are stored yet" in notice["text"]
    assert not contains_dash(notice["text"])
    rows = log_rows(vault)
    assert any(e == "ERROR" and "could not run" in d for e, d in rows)
    assert vault.load_state()["last_run"]["ok"] is False


def test_site_down_and_nothing_stored_in_a_dry_run_only_prints(tmp_path, toto, fourd, no_send):
    vault = make_vault(tmp_path, toto, fourd, tickets_text=None)
    printed: list[str] = []
    result = runner.run(dry_run=True, vault=vault, fetcher=FakeFetcher({}), now=NOW, out=printed.append)
    assert not result.ok
    assert not any("Notice" in p for p in printed)  # a dry run notifies nobody
    assert "ERROR" in events(vault)


def test_no_token_is_a_dry_run_with_a_warning(seeded, toto, fourd, no_send):
    vault = seeded
    printed: list[str] = []
    result = runner.run(vault=vault, fetcher=fake_site(toto, fourd), now=NOW, out=printed.append)
    assert result.ok and not result.posted
    assert "Message 1 of 3" in "\n".join(printed)
    assert any("Telegram is not set up" in w for w in result.warnings)
    assert "last_posted" not in vault.load_state()
    assert any(e == "DRY RUN" and "Telegram is not set up" in d for e, d in log_rows(vault))


def test_failed_post_is_retried_next_time(seeded, toto, fourd, tg):
    vault = seeded
    tg.fail = True
    result = runner.run(vault=vault, fetcher=fake_site(toto, fourd), now=NOW, out=lambda s: None)
    assert not result.ok and not result.posted
    assert "last_posted" not in vault.load_state()
    assert any(e == "ERROR" and "Posting to Telegram failed" in d for e, d in log_rows(vault))

    tg.fail = False
    again = runner.run(vault=vault, fetcher=fake_site(toto, fourd), now=NOW + timedelta(minutes=10),
                       out=lambda s: None)
    assert again.ok and again.posted and len(tg.sent) == 3


def test_no_fetch_and_no_post(seeded, tg):
    vault = seeded
    printed: list[str] = []
    later = NOW + timedelta(hours=2)
    result = runner.run(fetch=False, post=False, vault=vault, fetcher=raising_fetcher(), now=later,
                        out=printed.append, force_post=True)
    assert result.ok and not result.posted
    assert printed == []
    assert tg.sent == []
    assert len(result.messages) == 3
    assert "Mon 5 Oct 2026" in result.messages[1]  # next draw from state.json
    rows = log_rows(vault)
    assert any(e == "FETCH" and "Fetching is off" in d for e, d in rows)
    assert any(e == "POST" and "Posting is turned off" in d for e, d in rows)


def test_backtests_come_from_the_cache(seeded, first_run, monkeypatch):
    vault = seeded
    first = first_run[1]
    def boom(*a, **k):
        raise AssertionError("the cached backtest should be used")
    monkeypatch.setattr(backtest, "backtest_toto", boom)
    monkeypatch.setattr(backtest, "backtest_fourd", boom)
    second = runner.run(dry_run=True, fetch=False, vault=vault, now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert second.ok
    assert second.messages[2] == first.messages[2]


def drop_stored_draw(vault: Vault, number: int) -> None:
    df = load_toto(vault.toto_csv)
    save_toto(df[df["draw_number"] != number], vault.toto_csv)


def test_a_draw_that_keeps_failing_goes_on_the_skip_list(seeded, toto, fourd):
    vault = seeded
    broken = FIRST_TOTO + 5
    drop_stored_draw(vault, broken)
    site = fake_site(toto, fourd)
    del site.pages[toto_result_url(broken)]

    for i in range(runner.SKIP_AFTER_FAILED_RUNS):
        fetcher = FakeFetcher(site.pages)
        result = runner.fetch_data(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=i))
        assert toto_result_url(broken) in fetcher.requested
        assert any(str(broken) in w for w in result.warnings)
    state = vault.load_state()
    assert broken in state["skip"]["toto"]
    assert "fetch_failures" not in state

    fetcher = FakeFetcher(site.pages)
    result = runner.fetch_data(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=9))
    assert result.ok and toto_result_url(broken) not in fetcher.requested
    assert result.new_draws == {"toto": [], "4d": []}
    assert any(e == "FETCH" and "skip list" in d for e, d in log_rows(vault))


def test_newest_draws_are_never_skipped(seeded, toto, fourd):
    vault = seeded
    drop_stored_draw(vault, TOTO_LAST_DRAW)
    site = fake_site(toto, fourd)
    del site.pages[toto_result_url(TOTO_LAST_DRAW)]
    for i in range(runner.SKIP_AFTER_FAILED_RUNS + 1):
        runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=i))
    state = vault.load_state()
    assert TOTO_LAST_DRAW not in (state.get("skip") or {}).get("toto", [])
    assert state["fetch_failures"]["toto"][str(TOTO_LAST_DRAW)] == runner.SKIP_AFTER_FAILED_RUNS + 1


# demo


def test_demo_run_uses_synthetic_data_and_never_posts(tmp_path, monkeypatch, tg):
    elsewhere = tmp_path / "real data"
    monkeypatch.setenv("DATA_DIR", str(elsewhere))  # a demo must never write into the real data
    demo_root = tmp_path / "demo"
    (demo_root / "Huat Bot").mkdir(parents=True)
    (demo_root / "Huat Bot" / "Settings.md").write_text(SETTINGS_NOTE.replace(f"toto_start_draw: {FIRST_TOTO}\n", ""),
                                                        encoding="utf-8")
    printed: list[str] = []

    result = runner.run(demo=True, vault=demo_root, fetcher=raising_fetcher(), now=NOW, out=printed.append,
                        force_post=True)

    assert result.ok and not result.posted
    assert tg.sent == []
    out = "\n".join(printed)
    for i in (1, 2, 3):
        assert f"Message {i} of 3" in out
    assert not elsewhere.exists()
    vault = Vault(demo_root)
    toto = load_toto(vault.toto_csv)
    fourd = load_fourd(vault.fourd_csv)
    assert len(toto) == runner.DEMO_TOTO_DRAWS and int(toto["draw_number"].max()) == TOTO_LAST_DRAW
    assert int(fourd["draw_number"].max()) == FOURD_LAST_DRAW
    assert set(toto["fetched_at"]) == {"synthetic"}
    # demo tickets were made from the synthetic results and checked
    assert "demo tickets" in vault.read_text("Tickets.md")
    ledger = load_ledger(vault.ledger_csv)
    assert (ledger["winnings"] > 0).sum() >= 2
    assert "WINNER" in result.messages[0]
    assert any("synthetic" in w for w in result.warnings)
    state = vault.load_state()
    assert "last_posted" not in state and state["last_run"]["demo"] is True
    assert any(e == "DRY RUN" and "Demo run" in d for e, d in log_rows(vault))
    assert vault.exists("Dashboard.md")

    # a second demo run regenerates the same history: nothing new
    again = runner.run(demo=True, vault=demo_root, now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert again.ok and again.new_draws == {"toto": [], "4d": []}


def test_demo_history_ends_on_the_latest_draw_days():
    toto, fourd = runner.demo_history(datetime(2026, 10, 5, 12, 0, tzinfo=SG), fourd_draws=500)
    # before the Monday 7.30pm run the newest TOTO result is Thursday's
    assert toto["draw_date"].iloc[-1] == pd.Timestamp("2026-10-01")
    assert int(toto["draw_number"].iloc[-1]) == TOTO_LAST_DRAW
    assert fourd["draw_date"].iloc[-1] == pd.Timestamp("2026-10-04")
    assert int(fourd["draw_number"].iloc[-1]) == FOURD_LAST_DRAW + 2  # Sat 3 and Sun 4 Oct
    assert len(fourd) == 500
    toto_late, _ = runner.demo_history(datetime(2026, 10, 5, 19, 45, tzinfo=SG))
    assert int(toto_late["draw_number"].iloc[-1]) == TOTO_LAST_DRAW + 1


# pieces


def test_build_context_fills_every_part(toto, fourd):
    nt, nf = synth_next_toto(toto), synth_next_fourd(fourd)
    settings = Settings()
    ctx = runner.build_context(toto, fourd, settings, PrizeRules(), now=NOW, next_toto=nt, next_fourd=nf,
                               games_drawn=("toto",), new_draws={"toto": [TOTO_LAST_DRAW]})
    assert ctx.warnings == []
    assert [p.name for p in ctx.toto_picks] == ["Hot", "Overdue", "Balanced", "Low Crowd"]
    assert len(ctx.fourd_picks) == 5
    assert ctx.toto_plan.total <= settings.toto_budget
    assert ctx.fourd_plan.total <= settings.fourd_budget
    assert ctx.buy_signal is not None and ctx.buy_signal.label in ("HIGH", "MEDIUM", "LOW")
    assert ctx.crowd_scores is not None and len(ctx.crowd_scores) == 49
    assert ctx.games_drawn == ("toto",)
    assert ctx.fourd_bet_values["best"] == "Big"
    # picks are seeded with the next draw number: the same inputs give the same numbers
    again = runner.build_context(toto, fourd, settings, PrizeRules(), now=NOW, next_toto=nt, next_fourd=nf)
    assert [p.numbers for p in again.toto_picks] == [p.numbers for p in ctx.toto_picks]


def test_build_context_turns_a_broken_part_into_a_warning(toto, fourd, monkeypatch):
    from huatbot import analysis_toto

    def broken(df):
        raise ZeroDivisionError("boom")
    monkeypatch.setattr(analysis_toto, "chi_square_numbers", broken)
    ctx = runner.build_context(toto, fourd, Settings(), PrizeRules(), now=NOW)
    assert ctx.toto_chi == {}
    assert any("TOTO fairness test could not be worked out" in w for w in ctx.warnings)
    assert ctx.toto_picks  # the rest still ran


def test_refresh_next_draws_stores_the_dates(tmp_path, toto, fourd):
    vault = make_vault(tmp_path, toto, fourd)
    state = runner.refresh_next_draws(vault, fake_site(toto, fourd), now=NOW)
    assert state["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-05T18:30")
    assert state["next_draws"]["4d"]["draw_datetime"].startswith("2026-10-03T18:30")
    assert vault.load_state() == state
    # a failed refresh keeps what was known
    state = runner.refresh_next_draws(vault, FakeFetcher({}), now=NOW)
    assert state["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-05")
    assert any(e == "FETCH" and "Next draws" in d for e, d in log_rows(vault))


def test_next_draws_from_state_ignores_stale_entries(toto, fourd):
    state = {"next_draws": {"toto": {"draw_datetime": "2026-10-01T18:30:00+08:00", "jackpot_estimate": 1e6},
                            "4d": {"draw_datetime": "2026-10-03T18:30:00+08:00"}}}
    nt, nf = runner.next_draws_from_state(state, toto, fourd)
    assert nt is None  # Thursday's draw is already stored
    assert nf is not None and nf.draw_datetime.date() == date(2026, 10, 3)


def test_build_report_reads_only_stored_data(tmp_path, seeded, toto, fourd):
    assert runner.build_report(make_vault(tmp_path / "empty", toto, fourd), now=NOW) is None  # nothing stored
    vault = seeded
    ledger_before = vault.ledger_csv.read_text()
    log_before = vault.read_text(log_note_path(NOW))

    text = runner.build_report(vault, now=NOW + timedelta(hours=1))

    for heading in SECTION_HEADINGS:
        assert heading in text
    assert vault.ledger_csv.read_text() == ledger_before
    assert vault.read_text(log_note_path(NOW)) == log_before


def test_normalise_games():
    assert runner.normalise_games("both") == ("toto", "4d")
    assert runner.normalise_games(None) == ("toto", "4d")
    assert runner.normalise_games("TOTO") == ("toto",)
    assert runner.normalise_games(["4D", "toto"]) == ("toto", "4d")
    assert runner.normalise_games(("fourd",)) == ("4d",)
    with pytest.raises(ValueError):
        runner.normalise_games("lotto")


def test_notify_prints_in_a_dry_run(no_send):
    printed: list[str] = []
    assert runner.notify("Result not out - try later", dry_run=True, out=printed.append) is False
    assert printed and "Notice" in printed[0] and not contains_dash(printed[0])


def test_notify_sends_plain_text(tg):
    assert runner.notify("TOTO result is late", out=lambda s: None) is True
    assert tg.sent[0]["parse_mode"] is None and tg.sent[0]["text"] == "TOTO result is late"
    tg.fail = True
    assert runner.notify("again", out=lambda s: None) is False


def test_unexpected_error_is_logged_not_raised(seeded, toto, fourd, monkeypatch, no_send):
    from huatbot import report

    vault = seeded

    def broken(ctx):
        raise RuntimeError("layout bug")
    monkeypatch.setattr(report, "full_report", broken)
    result = runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto, fourd), now=NOW, out=lambda s: None)
    assert not result.ok
    assert any("layout bug" in w for w in result.warnings)
    assert any(e == "ERROR" and "Run failed" in d for e, d in log_rows(vault))
    assert vault.load_state()["last_run"]["ok"] is False


def test_fetch_error_type_is_reported_plainly(tmp_path, toto, fourd):
    vault = make_vault(tmp_path, toto, fourd)
    small = toto.tail(5)
    site = fake_site(small, fourd.tail(5))
    site.pages[C.TOTO_DRAW_LIST_URL] = FetchError("the site is busy (HTTP 503)", status=503)
    result = runner.fetch_data(("toto",), vault=vault, fetcher=site, now=NOW)
    assert not result.ok
    assert any("the site is busy (HTTP 503)" in w for w in result.warnings)
    assert len(result.new_draws["4d"]) == 5  # 4D had nothing stored, so it was fetched as well
    assert "toto" not in result.new_draws
