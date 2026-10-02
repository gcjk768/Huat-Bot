"""End to end tests of runner.run against an offline fake Singapore Pools site.

The fake site (tests.htmlgen.fake_site) serves result pages rendered from synthetic rows, so the
whole pipeline runs for real: fetch and parse, toto.csv in the vault, the buy signal, the jackpot
outlook and history, tickets, notes, the activity log, state.json and the two Telegram messages.
Telegram itself is faked.

The 150 draw history (seed 7) ends on Thu 1 Oct 2026 (draw 4123) after three draws in a row with
no Group 1 winner, so the next draw (Mon 5 Oct, 4124) is a cascade draw. With one more draw
(``toto_151``, 4124 is that cascade draw) the jackpot starts again and the outlook runs four
draws to the next cascade.
"""
from __future__ import annotations

import inspect
import json
import re
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pytest

from huatbot import outlook, prize_rules, report, runner, sales, store, telegram, tickets
from huatbot import constants as C
from huatbot.fetch import toto_result_url
from huatbot.http import FetchError
from huatbot.models import LEDGER_COLUMNS, NextToto, PrizeRules, Settings
from huatbot.report import SECTION_HEADINGS
from huatbot.store import load_ledger, load_toto, save_toto
from huatbot.synth import SG, synth_next_toto, synth_toto
from huatbot.textfmt import contains_dash, has_prose_dashes
from huatbot.vault import Vault, log_note_path
from tests.ctxgen import (
    TOTO_LAST_DATE,
    TOTO_LAST_DRAW,
    make_context,
    start_date_for,
    tickets_markdown,
    toto_history,
    variant,
)
from tests.htmlgen import FakeFetcher, fake_site

N_DRAWS = 150
NOW = datetime(2026, 10, 1, 19, 45, tzinfo=SG)  # Thu, after the TOTO draw 4123
MONDAY = datetime(2026, 10, 5, 19, 45, tzinfo=SG)  # after the TOTO draw 4124
NEXT_TOTO_DAY = date(2026, 10, 5)  # Mon
FIRST_TOTO = TOTO_LAST_DRAW - N_DRAWS + 1
NEXT_JACKPOT = "$3,695,567"  # synth_next_toto: the last jackpot ($2,595,567) plus $1,100,000

SETTINGS_NOTE = f"""---
toto_start_draw: {FIRST_TOTO}
draw_notes_backfill: 5
---
# Settings for the tests
"""

TOTO_COLS = ["draw_number", "draw_date"] + [f"n{i}" for i in range(1, 7)] + ["additional", "jackpot", "draw_type"] + [
    f"g{g}_winners" for g in range(1, 8)
]


# fixtures


@pytest.fixture(scope="module")
def toto():
    return toto_history(N_DRAWS)


ENV_NAMES = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "COMMENTARY", "DATA_DIR", "VAULT_FOLDER", "VAULT_PATH",
             "DRY_RUN", "ENV_FILE")


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


def make_vault(tmp_path, toto, tickets_text: str | None = "auto") -> Vault:
    vault = Vault(tmp_path / "vault")
    vault.base.mkdir(parents=True)
    (vault.base / "Settings.md").write_text(SETTINGS_NOTE, encoding="utf-8")
    if tickets_text == "auto":
        tickets_text = tickets_markdown(toto, NEXT_TOTO_DAY)
    if tickets_text is not None:
        (vault.base / "Tickets.md").write_text(tickets_text, encoding="utf-8")
    return vault


@pytest.fixture(scope="module")
def first_run(tmp_path_factory, toto):
    """One dry run from an empty vault against the fake site (shared, never modified by tests)."""
    root = tmp_path_factory.mktemp("first")
    printed: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        for name in ENV_NAMES:
            mp.delenv(name, raising=False)

        def boom(*a, **k):
            raise AssertionError("Telegram must not be called")
        mp.setattr(telegram, "send_message", boom)
        vault = make_vault(root, toto)
        result = runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto), now=NOW,
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
        m = re.match(r"^- \d\d:\d\d \S+ \*\*(.+?)\*\* · (.*)$", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def events(vault: Vault, when: datetime = NOW) -> list[str]:
    return [e for e, _ in log_rows(vault, when)]


def result_pages(fetcher: FakeFetcher) -> list[str]:
    return [u for u in fetcher.requested if "toto_results.aspx" in u]


def raising_fetcher():
    class Raising:
        def get(self, url):
            raise AssertionError(f"no network expected, asked for {url}")

        def get_many(self, urls):
            raise AssertionError("no network expected")
    return Raising()


def toto_151() -> pd.DataFrame:
    """The same synthetic history with one more draw (4124 on Mon 5 Oct 2026, a cascade draw)."""
    start = start_date_for(TOTO_LAST_DATE, C.TOTO_WEEKDAYS, N_DRAWS)
    return synth_toto(n_draws=N_DRAWS + 1, start_draw=FIRST_TOTO, start_date=start, seed=7)


def next_toto_on(day: date, jackpot: float | None, draw_type: str = "normal", hint: str | None = None) -> NextToto:
    return NextToto(draw_datetime=datetime.combine(day, time(18, 30), tzinfo=SG), jackpot_estimate=jackpot,
                    draw_type=draw_type, draw_type_hint=hint)


def new_draw_row(df: pd.DataFrame, n: int) -> str:
    """The NEW DRAW activity row the runner writes for draw ``n`` of ``df``."""
    r = df[df["draw_number"] == n].iloc[0]
    nums = " ".join(str(int(r[f"n{i}"])) for i in range(1, 7))
    winners = int(r["g1_winners"])
    who = "no winner" if not winners else f"{winners} winner" + ("s" if winners > 1 else "")
    return (f"TOTO draw {n} on {r['draw_date']:%a} {r['draw_date'].day} {r['draw_date']:%b %Y}: {nums}, "
            f"additional {int(r['additional'])}, Group 1 ${float(r['jackpot']):,.0f}, {who}")


# end to end


def test_first_run_dry_run_end_to_end(first_run, toto):
    vault, result, printed = first_run

    assert result.ok, result.warnings
    assert not result.posted
    assert result.new_draws == list(range(FIRST_TOTO, TOTO_LAST_DRAW + 1))
    assert result.warnings == []

    # exactly two messages printed, none posted, all within the rules
    out = "\n".join(printed)
    assert "Message 1 of 2" in out and "Message 2 of 2" in out
    assert not re.search(r"Message \d of 3|Message 3", out)
    assert len(result.messages) == 2
    for m in result.messages:
        assert m in out
        assert not contains_dash(m)
        assert len(m) <= C.TELEGRAM_MAX_CHARS
    m1, m2 = result.messages
    assert m1.startswith("🎉 <b>WINNER</b> · your tickets won $10\n1 winning ticket, details below.")
    assert f"<b>TOTO RESULT</b> · Draw {TOTO_LAST_DRAW}, Thu 1 Oct 2026" in m1
    assert "<b>MY TICKETS</b>" in m1 and "Group 7 x1, won <b>$10</b>" in m1
    assert "2 lines in Tickets.md could not be read, see Ledger.md." in m1
    assert "Next TOTO draw" not in m1

    assert m2.startswith(f"🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct 2026, 6.30pm, draw {TOTO_LAST_DRAW + 1}")
    assert f"<b>Jackpot {NEXT_JACKPOT}</b>" in m2
    assert "<b>Cascade draw</b>" in m2
    assert "rollovers 3 of 3, this is the cascade draw" in m2
    assert "Somebody wins Group 1: <b>" in m2
    assert "Buy signal <b>HIGH</b>" in m2
    assert "<b>$1.01</b> back per $1 on average" in m2
    assert "Cascaded jackpot" in m2  # the return breakdown
    assert "Sales estimate: about " in m2
    assert "<b>NEXT BIG PRIZE</b>" in m2
    assert (f"The next draw is the cascade draw: about {NEXT_JACKPOT}. If nobody wins it, the jackpot goes to "
            "the Group 2 winners.") in m2
    assert f"Over {N_DRAWS} stored draws Group 1 was won in " in m2
    assert "Every draw is independent." in m2

    # toto.csv in the vault matches the site
    saved = load_toto(vault.toto_csv)
    pd.testing.assert_frame_equal(saved[TOTO_COLS], toto[TOTO_COLS], check_dtype=False)

    # tickets: the Group 7 ticket and the losing one were checked, the next draw's wait
    ledger = load_ledger(vault.ledger_csv)
    assert set(ledger["game"]) == {"TOTO"}
    settled = ledger[ledger["status"] == "settled"]
    assert sorted(settled["result"]) == ["Group 7 x1", "No prize"]
    assert settled["winnings"].sum() == 10.0
    pending = ledger[ledger["status"] == "pending"]
    assert sorted(pending["bet_type"]) == ["Ordinary", "System 7"]
    assert set(pending["draw_date"]) == {"2026-10-05"}

    # notes: dashboard, ledger, report and draw notes; nothing for 4D or suggestions
    for rel in ("Home.md", "Ledger.md", "Reports/2026/10/2026-10-01 1945 Report.md",
                f"Draws/TOTO/2026-10-01 TOTO {TOTO_LAST_DRAW}.md", "Settings.md", "Tickets.md"):
        assert vault.exists(rel), rel
    assert len(list(vault.path("Draws/TOTO").glob("*.md"))) == 5  # draw_notes_backfill
    assert not vault.base.joinpath("Suggestions").exists()
    assert not vault.base.joinpath("Draws", "4D").exists()
    assert sorted(p.name for p in vault.data_dir.iterdir()) == ["last_messages.json", "ledger.csv",
                                                                 "prize_rules.json", "state.json", "toto.csv"]
    assert result.report_path == str(vault.path("Reports/2026/10/2026-10-01 1945 Report.md"))
    report_text = vault.read_text("Reports/2026/10/2026-10-01 1945 Report.md")
    positions = [report_text.index(h) for h in SECTION_HEADINGS]
    assert positions == sorted(positions)
    assert "### The next big prize" in report_text
    dashboard = vault.read_text("Home.md")
    assert "## The next big prize" in dashboard and "Checked in this run: 2 tickets, won $10." in dashboard

    # state.json
    state = vault.load_state()
    assert set(state["next_draws"]) == {"toto", "checked_at"}
    nxt = state["next_draws"]["toto"]
    assert nxt["draw_datetime"] == "2026-10-05T18:30:00+08:00"
    assert nxt["jackpot_estimate"] == 3_695_567.0 and nxt["draw_type"] == "cascade"
    assert state["upcoming_draws"] == {"toto": ["2026-10-05"]}
    assert "last_posted" not in state and "posting" not in state and "skip" not in state
    assert state["last_run"] == {"at": NOW.isoformat(), "ok": True, "posted": False, "dry_run": True,
                                 "demo": False, "new_draws": N_DRAWS}
    assert set(state["last_logged"]) == {runner.SIGNAL_LOG_KEY, runner.NEXT_DRAW_LOG_KEY}

    # prize rules cache
    assert vault.prize_rules_path.exists()

    # activity log: a row for every kind of action, no errors
    seen = set(events(vault))
    for ev in ("RUN", "SETTINGS", "FETCH", "NEW DRAW", "SIGNAL", "TICKETS", "LEDGER", "NOTE", "DRY RUN"):
        assert ev in seen, ev
    assert not seen & {"ERROR", "POST", "BACKTEST", "SUGGEST"}
    rows = log_rows(vault)
    assert rows[0] == ("RUN", "Run started (dry run, manual)")
    assert rows[-1][0] == "RUN" and rows[-1][1].startswith("Run finished in ")
    assert rows[-1][1].endswith(f"{N_DRAWS} new draws, report [[2026-10-01 1945 Report]]")
    assert ("SETTINGS", f"Read [[Settings]]: jackpot alert $3,000,000, special draws flagged, history from draw "
                        f"{FIRST_TOTO}") in rows
    assert ("FETCH", f"TOTO: {N_DRAWS} new draws ({FIRST_TOTO} to {TOTO_LAST_DRAW}), latest on the site is draw "
                     f"{TOTO_LAST_DRAW}, newest stored draw matches the site") in rows
    assert ("NEW DRAW", f"TOTO: 140 older draws added ({FIRST_TOTO} to {TOTO_LAST_DRAW - 10})") in rows
    assert [d for e, d in rows if e == "NEW DRAW"][1:] == [new_draw_row(toto, n)
                                                         for n in range(TOTO_LAST_DRAW - 9, TOTO_LAST_DRAW + 1)]
    assert ("FETCH", f"Next draw: TOTO Mon 5 Oct 2026, 6.30pm, estimated jackpot {NEXT_JACKPOT}, Cascade draw") in rows
    signal = [d for e, d in rows if e == "SIGNAL"]
    assert signal == [f"Buy signal HIGH for TOTO draw {TOTO_LAST_DRAW + 1}: jackpot {NEXT_JACKPOT}, cascade draw, "
                      "return per $1 about $1.01"]  # the cascade draw is the next big prize itself
    assert ("TICKETS", "Read [[Tickets]]: 4 tickets, 2 lines could not be read (listed in [[Ledger]])") in rows
    assert ("LEDGER", "4 new tickets, 2 tickets checked; all tickets: spent $10, won $10, net $0") in rows
    assert ("DRY RUN", "2 messages printed, nothing posted") in rows


def test_run_takes_no_games_argument():
    params = inspect.signature(runner.run).parameters
    assert "games" not in params
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    with pytest.raises(TypeError):
        runner.run(("toto",), dry_run=True)
    assert "games" not in inspect.signature(runner.fetch_data).parameters
    assert runner.GAME == "toto"
    for gone in ("EV_BACKTEST", "EV_SUGGEST", "normalise_games", "next_draws_from_state", "_log_suggestions",
                 "_backtest_key"):
        assert not hasattr(runner, gone), gone


def test_posting_is_idempotent_and_a_second_run_fetches_nothing(seeded, toto, tg):
    vault = seeded
    # nothing was posted by the dry run, so the first scheduled run posts the stored draws
    first = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert first.ok and first.posted
    assert len(tg.sent) == 2
    assert [s["text"] for s in tg.sent] == first.messages
    assert all(s["chat_id"] == "@huat" and s["parse_mode"] == "HTML" for s in tg.sent)
    state = vault.load_state()
    assert state["last_posted"] == {"toto": TOTO_LAST_DRAW}
    assert state["last_posted_at"] == (NOW + timedelta(minutes=5)).isoformat()
    assert "posting" not in state
    assert ("POST", f"Posted 2 messages to Telegram (TOTO {TOTO_LAST_DRAW})") in log_rows(vault)

    # scheduled rerun half an hour later: nothing new on the site, nothing posted again
    fetcher = fake_site(toto)
    logged = len(log_rows(vault))
    second = runner.run(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=30), out=lambda s: None)
    assert second.ok and not second.posted
    assert second.new_draws == []
    assert result_pages(fetcher) == []
    assert len(tg.sent) == 2
    rows = log_rows(vault)[logged:]
    assert ("POST", f"Nothing new since the last post (TOTO {TOTO_LAST_DRAW}), not posted again") in rows
    assert ("FETCH", f"TOTO: nothing new, latest on the site is draw {TOTO_LAST_DRAW}, newest stored draw matches "
                     "the site") in rows
    # nothing that merely restates the previous run: cached prize rules, the same signal, notes
    # whose only change would be their time stamp
    assert not any(e == "SIGNAL" for e, _ in rows), rows
    assert not any(e == "FETCH" and d.startswith("Prize rules") for e, d in rows), rows
    assert not any(e == "NOTE" and "[[Ledger]]" in d for e, d in rows), rows
    assert vault.load_state()["last_run"]["new_draws"] == 0

    # a manual run posts again on purpose
    third = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=40),
                       out=lambda s: None, force_post=True)
    assert third.posted and len(tg.sent) == 4


def test_new_draw_is_fetched_alone_and_posted(seeded, toto, tg):
    vault = seeded
    assert runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None).posted
    sent_before = len(tg.sent)

    bigger = toto_151()
    pd.testing.assert_frame_equal(bigger.iloc[:N_DRAWS][TOTO_COLS], toto[TOTO_COLS], check_dtype=False)
    assert bigger.iloc[-1]["draw_type"] == "cascade" and int(bigger.iloc[-1]["g1_winners"]) == 0
    fetcher = fake_site(bigger)
    logged = len(log_rows(vault, MONDAY))  # one log note per month: Thursday's rows are in it

    result = runner.run(vault=vault, fetcher=fetcher, now=MONDAY, out=lambda s: None)

    assert result.ok and result.posted
    assert result.new_draws == [TOTO_LAST_DRAW + 1]
    assert result_pages(fetcher) == [toto_result_url(TOTO_LAST_DRAW + 1)]
    assert len(load_toto(vault.toto_csv)) == N_DRAWS + 1
    assert len(tg.sent) == sent_before + 2
    m1, m2 = result.messages
    assert f"<b>TOTO RESULT</b> · Draw {TOTO_LAST_DRAW + 1}, Mon 5 Oct 2026, Cascade draw" in m1
    assert vault.load_state()["last_posted"] == {"toto": TOTO_LAST_DRAW + 1}
    assert vault.exists(f"Draws/TOTO/2026-10-05 TOTO {TOTO_LAST_DRAW + 1}.md")
    # the tickets bought for Monday's draw are checked now
    ledger = load_ledger(vault.ledger_csv)
    monday_rows = ledger[ledger["draw_date"] == "2026-10-05"]
    assert len(monday_rows) == 2 and set(monday_rows["status"]) == {"settled"}
    assert monday_rows["draw_number"].tolist() == [TOTO_LAST_DRAW + 1] * 2
    rows = log_rows(vault, MONDAY)[logged:]
    assert ("NEW DRAW", new_draw_row(bigger, TOTO_LAST_DRAW + 1)) in rows

    # the cascade paid out, so the jackpot starts again: four draws to the next cascade
    assert m2.startswith(f"🔮 <b>NEXT TOTO DRAW</b> · Thu 8 Oct 2026, 6.30pm, draw {TOTO_LAST_DRAW + 2}")
    assert "<b>Jackpot $1,000,000</b>" in m2
    assert "Normal draw · " in m2
    assert "rollovers 0 of 3, then it cascades" in m2
    assert "Buy signal <b>LOW</b>" in m2
    assert ("If nobody wins Group 1 first, the jackpot snowballs to about $3,714,000 at the cascade draw on "
            "Mon 19 Oct 2026") in m2
    table = re.search(r"<b>NEXT BIG PRIZE</b>.*?<pre>(.*?)</pre>", m2, re.S).group(1).splitlines()
    assert table[0].split() == ["Draw", "Jackpot", "Unwon", "Won"]
    assert [" ".join(line.split()[:3]) for line in table[1:]] == ["Thu 8 Oct", "Mon 12 Oct", "Thu 15 Oct",
                                                                    "Mon 19 Oct"]
    assert table[1].split()[3] == "$1.00m" and table[4].split()[3] == "$3.71m"
    assert table[1].split()[4] == "100%"

    # the SIGNAL row changed with the new draw and names the next big prize
    signal = [d for e, d in rows if e == "SIGNAL"]
    assert len(signal) == 1
    assert signal[0].startswith(f"Buy signal LOW for TOTO draw {TOTO_LAST_DRAW + 2}: jackpot $1,000,000, normal "
                                "draw, return per $1 about $0.")
    assert signal[0].endswith("; next big prize about $3,714,000 at the cascade draw on Mon 19 Oct 2026 if nobody "
                              "wins it first")


def test_signal_row_is_logged_once_per_change(seeded, toto, no_send):
    vault = seeded
    for minutes in (5, 10):
        runner.run(dry_run=True, fetch=False, vault=vault, now=NOW + timedelta(minutes=minutes), out=lambda s: None)
    assert [e for e in events(vault)].count("SIGNAL") == 1  # only the first run's row
    digest = vault.load_state()["last_logged"][runner.SIGNAL_LOG_KEY]

    # a new jackpot on the next draw page changes the signal: logged again, once
    site = fake_site(toto, next_toto=next_toto_on(NEXT_TOTO_DAY, 4_200_000.0))
    for minutes in (20, 30):
        runner.run(dry_run=True, vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=minutes),
                   out=lambda s: None)
    signal = [d for e, d in log_rows(vault) if e == "SIGNAL"]
    assert len(signal) == 2
    assert signal[1].startswith(f"Buy signal HIGH for TOTO draw {TOTO_LAST_DRAW + 1}: jackpot $4,200,000, "
                                "cascade draw")
    assert vault.load_state()["last_logged"][runner.SIGNAL_LOG_KEY] != digest


def test_site_down_uses_the_stored_data(seeded, no_send):
    vault = seeded
    printed: list[str] = []
    result = runner.run(dry_run=True, vault=vault, fetcher=FakeFetcher({}), now=NOW + timedelta(hours=1),
                        out=printed.append, force_post=True)

    assert result.ok
    assert len(result.messages) == 2
    assert "Message 2 of 2" in "\n".join(printed)
    assert any(w.startswith("TOTO results could not be fetched from the Singapore Pools site (") and
               f"using the {N_DRAWS} stored draws" in w for w in result.warnings)
    assert any(e == "ERROR" and "could not be fetched" in d for e, d in log_rows(vault))
    assert len(load_toto(vault.toto_csv)) == N_DRAWS
    # the next draw falls back to what state.json remembered
    assert "Mon 5 Oct 2026" in result.messages[1]
    assert f"<b>Jackpot {NEXT_JACKPOT}</b>" in result.messages[1]
    assert vault.load_state()["next_draws"]["toto"]["draw_datetime"] == "2026-10-05T18:30:00+08:00"


def test_site_down_and_nothing_stored_notifies_and_fails(tmp_path, toto, tg):
    vault = make_vault(tmp_path, toto, tickets_text=None)
    result = runner.run(vault=vault, fetcher=FakeFetcher({}), now=NOW, out=lambda s: None)

    assert not result.ok
    assert result.messages == [] and result.new_draws == []
    assert len(tg.sent) == 1
    notice = tg.sent[0]
    assert notice["parse_mode"] is None
    assert notice["text"].startswith("Huat Bot could not run: no TOTO results are stored yet and the Singapore "
                                     "Pools site could not be used (")
    assert notice["text"].endswith("). It will try again on the next run.")
    assert "4D" not in notice["text"]
    assert not contains_dash(notice["text"])
    rows = log_rows(vault)
    assert any(e == "ERROR" and "could not run" in d for e, d in rows)
    state = vault.load_state()
    assert state["last_run"]["ok"] is False and state["last_run"]["new_draws"] == 0
    assert not vault.exists("Home.md")


def test_site_down_and_nothing_stored_in_a_dry_run_only_prints(tmp_path, toto, no_send):
    vault = make_vault(tmp_path, toto, tickets_text=None)
    printed: list[str] = []
    result = runner.run(dry_run=True, vault=vault, fetcher=FakeFetcher({}), now=NOW, out=printed.append)
    assert not result.ok
    assert not any("Notice" in p for p in printed)  # a dry run notifies nobody
    assert "ERROR" in events(vault)


def test_fetch_off_and_nothing_stored_says_why(tmp_path, toto, tg):
    vault = make_vault(tmp_path, toto, tickets_text=None)
    result = runner.run(fetch=False, vault=vault, fetcher=raising_fetcher(), now=NOW, out=lambda s: None)
    assert not result.ok
    text = ("Huat Bot could not run: no TOTO results are stored yet and the Singapore Pools site could not be used "
            "(fetching was turned off). It will try again on the next run.")
    assert text in result.warnings
    assert [s["text"] for s in tg.sent] == [text]
    assert ("FETCH", "Fetching is off for this run, using the stored data (0 TOTO draws)") in log_rows(vault)


def test_no_token_is_a_dry_run_with_a_warning(seeded, toto, no_send):
    vault = seeded
    printed: list[str] = []
    result = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=printed.append)
    assert result.ok and not result.posted
    out = "\n".join(printed)
    assert "Message 1 of 2" in out and "Message 2 of 2" in out
    hint = [w for w in result.warnings if "Telegram is not set up" in w]
    assert hint == ["Telegram is not set up (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing), so this is a dry "
                    "run: the messages are printed, not posted."]
    state = vault.load_state()
    assert "last_posted" not in state and state["last_run"]["dry_run"] is True
    assert ("DRY RUN", "Telegram is not set up, treating this run as a dry run") in log_rows(vault)


def test_failed_post_is_retried_next_time(seeded, toto, tg):
    vault = seeded
    tg.fail = True
    result = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert not result.ok and not result.posted
    assert "last_posted" not in vault.load_state()
    assert any(e == "ERROR" and "Posting to Telegram failed" in d for e, d in log_rows(vault))
    assert any(w.startswith("Posting to Telegram failed: ") for w in result.warnings)

    tg.fail = False
    again = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=10), out=lambda s: None)
    assert again.ok and again.posted and len(tg.sent) == 2
    assert vault.load_state()["last_posted"] == {"toto": TOTO_LAST_DRAW}


def test_no_fetch_and_no_post(seeded, tg):
    vault = seeded
    printed: list[str] = []
    later = NOW + timedelta(hours=2)
    result = runner.run(fetch=False, post=False, vault=vault, fetcher=raising_fetcher(), now=later,
                        out=printed.append, force_post=True)
    assert result.ok and not result.posted
    assert printed == []
    assert tg.sent == []
    assert len(result.messages) == 2
    assert "Mon 5 Oct 2026" in result.messages[1]  # next draw from state.json
    rows = log_rows(vault)
    assert ("FETCH", f"Fetching is off for this run, using the stored data ({N_DRAWS} TOTO draws)") in rows
    assert ("POST", "Posting is turned off for this run, 2 messages not sent") in rows
    assert rows[-1][0] == "RUN" and rows[-1][1].startswith("Run finished in ")


def test_no_fetch_uses_the_cached_prize_rules(seeded, first_run, no_send):
    vault = seeded
    first = first_run[1]
    second = runner.run(dry_run=True, fetch=False, vault=vault, fetcher=raising_fetcher(),
                        now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert second.ok
    assert second.messages[1] == first.messages[1]  # same rules, same next draw, same outlook
    report_text = Path(second.report_path).read_text(encoding="utf-8")
    assert "TOTO prize rules: confirmed on the official prize page." in report_text
    assert not any("could not be confirmed" in w for w in second.warnings)
    # the tickets were checked by the first run, so this one checks none
    assert "No tickets were checked in this run." in second.messages[0]


def drop_stored_draw(vault: Vault, number: int) -> None:
    df = load_toto(vault.toto_csv)
    save_toto(df[df["draw_number"] != number], vault.toto_csv)


def test_a_draw_that_keeps_failing_goes_on_the_skip_list(seeded, toto):
    vault = seeded
    broken = FIRST_TOTO + 5
    drop_stored_draw(vault, broken)
    site = fake_site(toto)
    del site.pages[toto_result_url(broken)]

    for i in range(runner.SKIP_AFTER_FAILED_RUNS):
        fetcher = FakeFetcher(site.pages)
        result = runner.fetch_data(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=i))
        assert toto_result_url(broken) in fetcher.requested
        assert any(str(broken) in w for w in result.warnings)
        if i < runner.SKIP_AFTER_FAILED_RUNS - 1:
            assert vault.load_state()["fetch_failures"] == {"toto": {str(broken): i + 1}}
    state = vault.load_state()
    assert state["skip"] == {"toto": [broken]}
    assert "fetch_failures" not in state
    assert any(f"({broken}) failed in {runner.SKIP_AFTER_FAILED_RUNS} runs in a row and will not be fetched again"
               in w for w in result.warnings)

    fetcher = FakeFetcher(site.pages)
    result = runner.fetch_data(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=9))
    assert result.ok and toto_result_url(broken) not in fetcher.requested
    assert result.new_draws == []
    assert ("FETCH", f"TOTO: 1 draw ({broken}) failed in {runner.SKIP_AFTER_FAILED_RUNS} runs in a row and went on "
                     "the skip list in state.json") in log_rows(vault)


def test_newest_draws_are_never_skipped(seeded, toto):
    vault = seeded
    drop_stored_draw(vault, TOTO_LAST_DRAW)
    site = fake_site(toto)
    del site.pages[toto_result_url(TOTO_LAST_DRAW)]
    for i in range(runner.SKIP_AFTER_FAILED_RUNS + 1):
        runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=i))
    state = vault.load_state()
    assert TOTO_LAST_DRAW not in (state.get("skip") or {}).get("toto", [])
    assert state["fetch_failures"]["toto"][str(TOTO_LAST_DRAW)] == runner.SKIP_AFTER_FAILED_RUNS + 1
    assert runner.SKIP_PROTECT_NEWEST == 10


def test_fetch_trouble_never_puts_a_draw_on_the_skip_list(seeded, toto):
    """A 503, a timeout or a WAF 403 is not a reason to give up on a draw for good."""
    vault = seeded
    broken = FIRST_TOTO + 5
    drop_stored_draw(vault, broken)
    site = fake_site(toto)
    good_page = site.pages[toto_result_url(broken)]
    site.pages[toto_result_url(broken)] = FetchError("the site is busy (HTTP 503 after 6 attempts)", status=503)
    for i in range(runner.SKIP_AFTER_FAILED_RUNS + 1):
        result = runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=i))
        assert any(str(broken) in w and "tried again" in w for w in result.warnings)
    state = vault.load_state()
    assert broken not in (state.get("skip") or {}).get("toto", [])
    assert str(broken) not in (state.get("fetch_failures") or {}).get("toto", {})

    # one real failure (page gone) is counted, later fetch trouble keeps that count unchanged
    del site.pages[toto_result_url(broken)]
    runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=10))
    site.pages[toto_result_url(broken)] = FetchError("the request timed out", status=None)
    runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=11))
    site.pages[toto_result_url(broken)] = FetchError("access denied (HTTP 403)", status=403)
    runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW + timedelta(minutes=12))
    assert vault.load_state()["fetch_failures"]["toto"][str(broken)] == 1

    # the site recovers: the draw is fetched and stored
    site.pages[toto_result_url(broken)] = good_page
    fetcher = FakeFetcher(site.pages)
    result = runner.fetch_data(vault=vault, fetcher=fetcher, now=NOW + timedelta(minutes=13))
    assert toto_result_url(broken) in fetcher.requested
    assert result.new_draws == [broken]
    assert broken in set(load_toto(vault.toto_csv)["draw_number"])
    assert "fetch_failures" not in vault.load_state()


def test_a_skip_list_entry_that_is_not_a_number_is_ignored(seeded, toto):
    vault = seeded
    broken = FIRST_TOTO + 5
    drop_stored_draw(vault, broken)
    state = vault.load_state()
    state["skip"] = {"toto": [str(broken), "oops"]}
    vault.save_state(state)
    fetcher = fake_site(toto)
    result = runner.fetch_data(vault=vault, fetcher=fetcher, now=NOW)
    assert result.ok and result.new_draws == []
    assert toto_result_url(broken) not in fetcher.requested


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
    assert "Message 1 of 2" in out and "Message 2 of 2" in out and "Message 3" not in out
    assert len(result.messages) == 2
    assert not elsewhere.exists()
    vault = Vault(demo_root)
    toto = load_toto(vault.toto_csv)
    assert len(toto) == runner.DEMO_TOTO_DRAWS == 600
    assert int(toto["draw_number"].max()) == TOTO_LAST_DRAW
    assert toto["draw_date"].iloc[-1] == pd.Timestamp("2026-10-01")
    assert set(toto["fetched_at"]) == {"synthetic"}
    assert result.new_draws == list(range(TOTO_LAST_DRAW - 599, TOTO_LAST_DRAW + 1))
    assert sorted(p.name for p in vault.data_dir.iterdir()) == ["ledger.csv", "state.json", "toto.csv"]
    # demo tickets were made from the synthetic results and checked: one Group 7 winner
    assert "demo tickets" in vault.read_text("Tickets.md")
    ledger = load_ledger(vault.ledger_csv)
    settled = ledger[ledger["status"] == "settled"].sort_values("winnings")
    assert settled["result"].tolist() == ["No prize", "Group 7 x1"]
    assert settled["winnings"].tolist() == [0.0, 10.0]
    assert (ledger["status"] == "pending").sum() == 1
    assert result.messages[0].startswith("🎉 <b>WINNER</b> · your tickets won $10")
    assert "<b>NEXT BIG PRIZE</b>" in result.messages[1]
    assert result.warnings[0] == runner.DEMO_WARNING
    state = vault.load_state()
    assert "last_posted" not in state and state["last_run"]["demo"] is True
    assert state["next_draws"]["toto"]["draw_datetime"] == "2026-10-05T18:30:00+08:00"
    rows = log_rows(vault)
    assert ("DRY RUN", "Demo run, 2 messages printed, nothing posted") in rows
    assert ("FETCH", "Demo run: synthetic history from huatbot.synth, the site was not contacted (600 TOTO "
                     "draws)") in rows
    assert ("NOTE", "Created [[Tickets]] with demo tickets") in rows
    assert vault.exists("Home.md")

    # a second demo run regenerates the same history: nothing new
    again = runner.run(demo=True, vault=demo_root, now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert again.ok and again.new_draws == []
    assert len(load_toto(vault.toto_csv)) == 600


def test_demo_default_vault_folder(tmp_path, monkeypatch, no_send):
    monkeypatch.chdir(tmp_path)
    result = runner.run(demo=True, now=NOW, out=lambda s: None)
    assert result.ok
    assert (tmp_path / "demo-vault" / "Huat Bot" / "Home.md").is_file()


def test_demo_history_ends_on_the_latest_draw_days():
    toto = runner.demo_history(datetime(2026, 10, 5, 12, 0, tzinfo=SG))
    assert isinstance(toto, pd.DataFrame) and len(toto) == runner.DEMO_TOTO_DRAWS
    # before the Monday 7.30pm run the newest TOTO result is Thursday's
    assert toto["draw_date"].iloc[-1] == pd.Timestamp("2026-10-01")
    assert int(toto["draw_number"].iloc[-1]) == TOTO_LAST_DRAW
    assert runner.DEMO_TOTO_ANCHOR == (date(2026, 10, 1), TOTO_LAST_DRAW)
    late = runner.demo_history(datetime(2026, 10, 5, 19, 45, tzinfo=SG))
    assert int(late["draw_number"].iloc[-1]) == TOTO_LAST_DRAW + 1
    assert late["draw_date"].iloc[-1] == pd.Timestamp("2026-10-05")
    earlier = runner.demo_history(datetime(2026, 9, 30, 10, 0, tzinfo=SG))  # Wed: Mon 28 Sep is the newest
    assert int(earlier["draw_number"].iloc[-1]) == TOTO_LAST_DRAW - 1
    # every date is a regular draw day and the numbers run without gaps
    assert set(pd.to_datetime(toto["draw_date"]).dt.weekday) <= set(C.TOTO_WEEKDAYS)
    assert toto["draw_number"].diff().dropna().eq(1).all()


def test_demo_tickets_note_has_a_group_7_winner():
    toto = runner.demo_history(NOW)
    nt = synth_next_toto(toto)
    text = runner.demo_tickets_note(toto, nt)
    assert text.startswith(tickets.TICKETS_TEMPLATE.split("| Game |")[0])
    parsed = tickets.parse_tickets(text)
    assert len(parsed) == 3 and not any(t.error for t in parsed)
    assert [t.draw_date for t in parsed] == [date(2026, 10, 1), date(2026, 10, 1), date(2026, 10, 5)]
    ledger = tickets.sync_ledger(store.empty_ledger(), parsed, NOW)
    ledger, settled = tickets.settle_ledger(ledger, toto, PrizeRules(), NOW)
    assert sorted((r["result"], r["winnings"]) for r in settled) == [("Group 7 x1", 10.0), ("No prize", 0.0)]
    # without a next draw the waiting ticket is for 4 days after the newest draw
    no_next = tickets.parse_tickets(runner.demo_tickets_note(toto, None))
    assert no_next[-1].draw_date == date(2026, 10, 5)


# pieces


def test_build_context_fills_every_part(toto):
    nt = synth_next_toto(toto)
    settings = Settings()
    ctx = runner.build_context(toto, settings, PrizeRules(), now=NOW, next_toto=nt, new_draws=[TOTO_LAST_DRAW],
                               fetched=True)
    assert ctx.warnings == []
    assert ctx.next_toto is nt and ctx.fetched is True and ctx.new_draws == [TOTO_LAST_DRAW]
    assert ctx.buy_signal is not None and ctx.buy_signal.label == "HIGH"
    assert ctx.buy_signal.jackpot == nt.jackpot_estimate
    assert ctx.outlook is not None and ctx.outlook.jackpot == nt.jackpot_estimate
    assert ctx.outlook.draw_type == "cascade" and ctx.outlook.snowball_draws == 3
    assert len(ctx.outlook.steps) == 1 and ctx.outlook.steps[0].cascade
    assert ctx.history is not None and ctx.history.draws == N_DRAWS
    for gone in ("toto_picks", "fourd_picks", "toto_plan", "crowd_scores", "games_drawn", "fourd"):
        assert not hasattr(ctx, gone), gone
    # the same inputs give the same figures
    again = runner.build_context(toto, settings, PrizeRules(), now=NOW, next_toto=nt)
    assert again.outlook == ctx.outlook and again.history == ctx.history


def test_build_context_with_no_history_is_empty():
    ctx = runner.build_context(store.empty_toto(), Settings(), PrizeRules(), now=NOW)
    assert ctx.outlook is None and ctx.history is None and ctx.buy_signal is None and ctx.warnings == []


def test_build_context_turns_a_broken_part_into_a_warning(toto, monkeypatch):
    def broken(*a, **k):
        raise ZeroDivisionError("boom")
    monkeypatch.setattr(outlook, "jackpot_outlook", broken)
    ctx = runner.build_context(toto, Settings(), PrizeRules(), now=NOW, next_toto=synth_next_toto(toto))
    assert ctx.outlook is None
    assert ctx.warnings == ["The jackpot outlook could not be worked out in this run (ZeroDivisionError)."]
    assert ctx.buy_signal is not None and ctx.history is not None  # the rest still ran

    monkeypatch.setattr(sales, "sales_table", broken)
    ctx = runner.build_context(toto, Settings(), PrizeRules(), now=NOW, next_toto=synth_next_toto(toto))
    assert "The TOTO sales estimate could not be worked out in this run (ZeroDivisionError)." in ctx.warnings
    assert ctx.history is not None


def test_buy_signal_uses_the_outlook_jackpot_when_the_page_has_none(toto):
    rules = PrizeRules()
    expected = outlook.estimate_next_jackpot(toto, rules)
    assert expected is not None and expected > float(toto["jackpot"].iloc[-1])
    for nt in (next_toto_on(NEXT_TOTO_DAY, None, "cascade"), None):
        ctx = runner.build_context(toto, Settings(), rules, now=NOW, next_toto=nt)
        assert ctx.outlook.jackpot == expected
        assert ctx.buy_signal is not None and ctx.buy_signal.jackpot == expected
        assert ctx.buy_signal.draw_type == ctx.outlook.draw_type == "cascade"
        assert any("worked out from the stored results" in n for n in ctx.outlook.notes)
        assert report.toto_signal(ctx)["jackpot"] == expected


def test_a_stale_next_draw_page_is_ignored(toto):
    rules = PrizeRules()
    # the page still shows Thursday's draw (already stored) with its own jackpot
    stale = next_toto_on(TOTO_LAST_DATE, 9_999_999.0, "special", hint="special")
    ctx = runner.build_context(toto, Settings(), rules, now=NOW, next_toto=stale)
    expected = outlook.estimate_next_jackpot(toto, rules)
    assert ctx.next_toto is stale  # kept, the report says it is not current
    assert ctx.outlook.jackpot == expected != 9_999_999.0
    assert ctx.outlook.draw_type == "cascade"  # worked out from the history, not the old page
    assert ctx.buy_signal.jackpot == expected
    assert ctx.buy_signal.draw_type == "cascade"
    # stored info from an earlier run about a day already past is ignored the same way
    old = next_toto_on(date(2026, 10, 5), 7_777_777.0)
    later = runner.build_context(toto, Settings(), rules, now=datetime(2026, 10, 6, 9, 0, tzinfo=SG), next_toto=old)
    assert later.outlook.jackpot == expected and later.buy_signal.jackpot == expected


def test_announced_draws_marks_special_days():
    state = {"upcoming_draws": {"toto": ["2026-10-05", "2026-10-09T18:30:00+08:00", "not a date"],
                                "4d": ["2026-10-03"]}}
    assert runner.announced_draws(state, None) == [(date(2026, 10, 5), "normal"), (date(2026, 10, 9), "special")]
    # the next draw page names its draw type: a Hongbao draw on Friday
    hongbao = next_toto_on(date(2026, 10, 9), 12_000_000.0, "hongbao", hint="hongbao")
    assert runner.announced_draws(state, hongbao) == [(date(2026, 10, 5), "normal"), (date(2026, 10, 9), "hongbao")]
    special = next_toto_on(date(2026, 10, 12), 5_000_000.0, "special", hint="special")
    assert runner.announced_draws({}, special) == [(date(2026, 10, 12), "special")]
    # a cascade word is not a draw with its own advertised jackpot
    cascade = next_toto_on(date(2026, 10, 12), 4_000_000.0, "cascade", hint="cascade")
    assert runner.announced_draws({}, cascade) == []
    assert runner.announced_draws({}, None) == []
    assert runner.announced_draws({"upcoming_draws": {"toto": None}}, None) == []


def test_announced_special_draws_reach_the_outlook_and_message_2(seeded, toto, no_send):
    vault = seeded
    state = vault.load_state()
    state["upcoming_draws"] = {"toto": ["2026-10-05", "2026-10-09"]}
    vault.save_state(state)
    site = fake_site(toto, next_toto=next_toto_on(date(2026, 10, 9), 12_000_000.0, "hongbao", hint="hongbao"))
    result = runner.run(dry_run=True, vault=vault, fetcher=site, now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert result.ok
    m2 = result.messages[1]
    assert "Announced special draws: Fri 9 Oct 2026 (Hongbao)." in m2
    assert "<b>Hongbao draw</b>" in m2
    assert "The next draw is a Hongbao draw with a jackpot of about $12,000,000." in m2
    assert vault.load_state()["upcoming_draws"]["toto"] == ["2026-10-05", "2026-10-09"]


def test_load_state_drops_what_older_versions_kept(tmp_path):
    vault = Vault(tmp_path / "vault")
    vault.ensure_layout()
    old = {
        "next_draws": {"toto": {"draw_datetime": "2026-10-05T18:30:00+08:00"},
                       "4d": {"draw_datetime": "2026-10-03T18:30:00+08:00"}, "checked_at": "2026-10-01T19:45:00+08:00"},
        "skip": {"toto": [4000], "4d": [5000]},
        "fetch_failures": {"toto": {"4001": 1}, "4D": {"5001": 2}},
        "last_posted": {"toto": 4123, "4d": 5432},
        "upcoming_draws": {"toto": ["2026-10-05"], "4d": ["2026-10-03"], "fourd": ["2026-10-04"]},
        "last_logged": {"SUGGEST toto": "abc", "SUGGEST 4d": "def", "SIGNAL": "123"},
        "posting": {"draws": {"toto": 4123, "4d": 5432}, "sent": 1, "total": 3},
        "unposted_settled": ["t1"],
    }
    vault.state_path.write_text(json.dumps(old), encoding="utf-8")
    state = runner.load_state(vault)
    assert state["next_draws"] == {"toto": {"draw_datetime": "2026-10-05T18:30:00+08:00"},
                                   "checked_at": "2026-10-01T19:45:00+08:00"}
    assert state["skip"] == {"toto": [4000]}
    assert state["fetch_failures"] == {"toto": {"4001": 1}}
    assert state["last_posted"] == {"toto": 4123}
    assert state["upcoming_draws"] == {"toto": ["2026-10-05"]}
    assert state["last_logged"] == {"SIGNAL": "123"}
    assert "posting" not in state  # half of the old three messages is not resumed
    assert state["unposted_settled"] == ["t1"]

    # a partly posted set of this version's two messages is kept
    vault.state_path.write_text(json.dumps({"posting": {"draws": {"toto": 4123}, "sent": 1, "total": 2}}),
                                encoding="utf-8")
    assert runner.load_state(vault)["posting"] == {"draws": {"toto": 4123}, "sent": 1, "total": 2}


def test_an_old_state_is_cleaned_and_posts_the_new_two_messages(seeded, toto, tg):
    vault = seeded
    state = vault.load_state()
    state["last_posted"] = {"toto": TOTO_LAST_DRAW - 1, "4d": 5432}
    state["posting"] = {"draws": {"toto": TOTO_LAST_DRAW, "4d": 5432}, "sent": 1, "total": 3}
    state["next_draws"]["4d"] = {"draw_datetime": "2026-10-03T18:30:00+08:00"}
    state["last_logged"]["SUGGEST 4d"] = "abc"
    vault.state_path.write_text(json.dumps(state), encoding="utf-8")

    result = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=5), out=lambda s: None)
    assert result.ok and result.posted
    assert [s["text"] for s in tg.sent] == result.messages  # both messages, none skipped
    saved = json.loads(vault.state_path.read_text(encoding="utf-8"))
    assert saved["last_posted"] == {"toto": TOTO_LAST_DRAW}
    assert "posting" not in saved
    assert set(saved["next_draws"]) == {"toto", "checked_at"}
    assert set(saved["upcoming_draws"]) == {"toto"}
    assert not any(k.startswith("SUGGEST") for k in saved["last_logged"])


def test_an_old_ledger_with_4d_rows_is_kept(seeded, toto, no_send):
    vault = seeded
    ledger = load_ledger(vault.ledger_csv)
    base = {c: "" for c in LEDGER_COLUMNS}
    old = pd.DataFrame([
        {**base, "ticket_id": "old4dwin", "game": "4D", "draw_date": "2026-09-30", "draw_number": 5432,
         "numbers": "1234", "bet_type": "Big", "cost": 2.0, "units": 2.0, "status": "settled",
         "result": "1st prize", "winnings": 4000.0, "added_at": "2026-09-29T10:00:00+08:00",
         "checked_at": "2026-09-30T19:30:00+08:00", "source": "| 4D | 30 Sep 2026 | 1234 | Big | $2 |"},
        {**base, "ticket_id": "old4dwait", "game": "4D", "draw_date": "2026-10-03", "numbers": "5678",
         "bet_type": "Small", "cost": 1.0, "units": 1.0, "status": "pending", "result": "", "winnings": 0.0,
         "added_at": "2026-09-29T10:00:00+08:00", "source": "| 4D | 3 Oct 2026 | 5678 | Small | $1 |"},
    ])
    store.save_ledger(pd.concat([ledger, old], ignore_index=True), vault.ledger_csv)

    result = runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=5),
                        out=lambda s: None)
    assert result.ok, result.warnings
    after = load_ledger(vault.ledger_csv).set_index("ticket_id")
    assert after.loc["old4dwin", "status"] == "settled" and after.loc["old4dwin", "winnings"] == 4000.0
    assert after.loc["old4dwin", "result"] == "1st prize"
    assert after.loc["old4dwait", "status"] == "invalid"  # never checked, not counted
    assert after.loc["old4dwait", "result"] == tickets.FOURD_NOT_TRACKED
    totals = tickets.ledger_totals(after.reset_index())
    assert totals["won"] == 4010.0 and totals["spent"] == 12.0 and totals["invalid"] == 1
    # the old 4D win is history in the totals; the messages carry no 4D figures of their own
    assert "All tickets so far: spent $12, won $4,010, net <b>$3,998</b>." in result.messages[0]
    assert "4D" not in "\n".join(result.messages)
    ledger_note = vault.read_text("Ledger.md")
    assert "| Wed 30 Sep 2026 | 5432 | 1234                | Big      |   $2 | Checked              | 1st prize" \
        in ledger_note
    assert f"| Not counted          | {tickets.FOURD_NOT_TRACKED} |" in ledger_note


def test_refresh_next_draws_stores_the_date(tmp_path, toto):
    vault = make_vault(tmp_path, toto)
    state = runner.refresh_next_draws(vault, fake_site(toto), now=NOW)
    assert state["next_draws"]["toto"]["draw_datetime"] == "2026-10-05T18:30:00+08:00"
    assert state["next_draws"]["checked_at"] == NOW.isoformat()
    assert set(state["next_draws"]) == {"toto", "checked_at"}
    assert vault.load_state() == state
    rows = log_rows(vault)
    # nothing is stored yet, so the draw type is not predicted from a history
    assert rows == [("FETCH", "Next draw: TOTO Mon 5 Oct 2026, 6.30pm, estimated jackpot $3,695,567")]
    # a failed refresh keeps what was known and says so
    later = NOW + timedelta(minutes=10)
    state = runner.refresh_next_draws(vault, FakeFetcher({}), now=later)
    assert state["next_draws"]["toto"]["draw_datetime"] == "2026-10-05T18:30:00+08:00"
    assert state["next_draws"]["checked_at"] == NOW.isoformat()
    assert log_rows(vault)[-1] == ("FETCH", "Next draw: the next draw page could not be read")
    # log_activity=False writes no row
    runner.refresh_next_draws(vault, fake_site(toto), now=later + timedelta(minutes=10), log_activity=False)
    assert len(log_rows(vault)) == 2


def test_refresh_next_draws_takes_a_vault_path(tmp_path, toto):
    root = tmp_path / "vault"
    state = runner.refresh_next_draws(root, fake_site(toto), now=NOW)
    assert Vault(root).load_state() == state
    assert Vault(root).base.joinpath("Draws", "TOTO").is_dir()


def test_next_draw_from_state(toto):
    state = {"next_draws": {"toto": {"draw_datetime": "2026-10-01T18:30:00+08:00", "jackpot_estimate": 1e6}}}
    assert runner.next_draw_from_state(state, toto) is None  # Thursday's draw is already stored
    state = {"next_draws": {"toto": {"draw_datetime": "2026-10-05T18:30:00+08:00", "jackpot_estimate": 3_695_567,
                                     "draw_type": "cascade", "draw_type_hint": None, "raw_text": "Next Jackpot"},
                            "4d": {"draw_datetime": "2026-10-03T18:30:00+08:00"}}}
    nt = runner.next_draw_from_state(state, toto)
    assert nt == NextToto(draw_datetime=datetime(2026, 10, 5, 18, 30, tzinfo=SG), jackpot_estimate=3_695_567.0,
                          draw_type="cascade", draw_type_hint=None, raw_text="Next Jackpot")
    assert nt.draw_datetime.tzinfo is not None
    # odd values: no jackpot, no draw type, a naive time (Singapore time)
    odd = {"next_draws": {"toto": {"draw_datetime": "2026-10-05T18:30:00", "jackpot_estimate": "soon"}}}
    nt = runner.next_draw_from_state(odd, toto)
    assert nt.jackpot_estimate is None and nt.draw_type == "normal" and nt.raw_text == ""
    assert nt.draw_datetime == datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    # nothing stored yet: any date counts
    assert runner.next_draw_from_state(state, store.empty_toto()) is not None
    for broken in ({}, {"next_draws": {}}, {"next_draws": {"toto": "Mon"}},
                   {"next_draws": {"toto": {"draw_datetime": "later"}}}):
        assert runner.next_draw_from_state(broken, toto) is None


def test_build_report_reads_only_stored_data(tmp_path, seeded, toto):
    assert runner.build_report(make_vault(tmp_path / "empty", toto), now=NOW) is None  # nothing stored
    vault = seeded
    ledger_before = vault.ledger_csv.read_text()
    log_before = vault.read_text(log_note_path(NOW))
    state_before = vault.state_path.read_text()

    text = runner.build_report(vault, now=NOW + timedelta(hours=1))

    for heading in SECTION_HEADINGS:
        assert heading in text
    assert "Mon 5 Oct 2026" in text and NEXT_JACKPOT in text
    assert vault.ledger_csv.read_text() == ledger_before
    assert vault.read_text(log_note_path(NOW)) == log_before
    assert vault.state_path.read_text() == state_before


def test_notify_prints_in_a_dry_run(no_send):
    printed: list[str] = []
    assert runner.notify("Result not out - try later", dry_run=True, out=printed.append) is False
    assert printed == ["Notice (not posted): Result not out, try later"]


def test_notify_without_telegram_prints(no_send):
    printed: list[str] = []
    assert runner.notify("TOTO result is late", out=printed.append) is False
    assert printed == ["Notice (not posted): TOTO result is late"]


def test_notify_sends_plain_text(tg):
    assert runner.notify("TOTO result is late", out=lambda s: None) is True
    assert tg.sent[0]["parse_mode"] is None and tg.sent[0]["text"] == "TOTO result is late"
    tg.fail = True
    assert runner.notify("again", out=lambda s: None) is False


def test_unexpected_error_is_logged_not_raised(seeded, toto, monkeypatch, no_send):
    vault = seeded

    def broken(ctx):
        raise RuntimeError("layout bug")
    monkeypatch.setattr(report, "full_report", broken)
    result = runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert not result.ok
    assert "Run failed: RuntimeError: layout bug." in result.warnings
    assert any(e == "ERROR" and "Run failed" in d for e, d in log_rows(vault))
    assert vault.load_state()["last_run"]["ok"] is False
    assert log_rows(vault)[-1][1].startswith("Run ended with a problem")


def test_fetch_error_type_is_reported_plainly(tmp_path, toto):
    vault = make_vault(tmp_path, toto)
    site = fake_site(toto.tail(5))
    site.pages[C.TOTO_DRAW_LIST_URL] = FetchError("the site is busy (HTTP 503)", status=503)
    result = runner.fetch_data(vault=vault, fetcher=site, now=NOW)
    assert not result.ok
    assert result.new_draws == []
    assert any("the site is busy (HTTP 503)" in w and "no draws are stored yet" in w for w in result.warnings)
    assert all(not contains_dash(w) for w in result.warnings)
    rows = log_rows(vault)
    assert rows[0] == ("RUN", "Fetch started")
    assert ("ERROR", "TOTO results could not be fetched from the Singapore Pools site (the site is busy (HTTP "
                     "503)), and no draws are stored yet") in rows
    # the next draw page is read all the same (nothing stored: no draw type predicted)
    assert rows[-1] == ("FETCH", "Next draw: TOTO Mon 5 Oct 2026, 6.30pm, estimated jackpot $3,695,567")


def test_fetch_data_stores_the_draws_and_the_next_draw(tmp_path, toto):
    vault = make_vault(tmp_path, toto)
    result = runner.fetch_data(vault=vault, fetcher=fake_site(toto), now=NOW)
    assert result.ok and result.new_draws == list(range(FIRST_TOTO, TOTO_LAST_DRAW + 1))
    assert result.warnings == []
    assert len(load_toto(vault.toto_csv)) == N_DRAWS
    state = vault.load_state()
    assert state["next_draws"]["toto"]["draw_type"] == "cascade"
    assert not vault.exists("Home.md")  # fetch does not analyse or write notes
    rows = log_rows(vault)
    assert ("RUN", "Fetch started") in rows
    assert ("FETCH", f"Next draw: TOTO Mon 5 Oct 2026, 6.30pm, estimated jackpot {NEXT_JACKPOT}, Cascade draw") in rows


# fetch trouble, fetch messages and activity rows


def test_draw_type_list_failure_is_a_warning_and_an_error_row(seeded, toto):
    vault = seeded
    site = fake_site(toto)
    del site.pages[C.TOTO_HONGBAO_LIST_URL]
    result = runner.fetch_data(vault=vault, fetcher=FakeFetcher(site.pages), now=NOW)
    assert any("Hongbao draw list could not be used" in w for w in result.warnings)
    assert any(e == "ERROR" and "Hongbao draw list could not be used" in d for e, d in log_rows(vault))
    assert all(not contains_dash(w) for w in result.warnings)


def test_date_mismatch_is_worded_as_a_date_problem(seeded, toto):
    vault = seeded
    df = load_toto(vault.toto_csv)
    df.loc[df["draw_number"] == TOTO_LAST_DRAW, "draw_date"] = pd.Timestamp("2026-09-30")
    save_toto(df, vault.toto_csv)
    result = runner.fetch_data(vault=vault, fetcher=fake_site(toto), now=NOW)
    assert any(f"draw {TOTO_LAST_DRAW} is dated" in w and "on the site" in w for w in result.warnings)
    assert not any("does not match the latest draw on the site" in w for w in result.warnings)


def test_prize_rules_check_is_logged(first_run, seeded, toto):
    first_vault = first_run[0]
    assert any(e == "FETCH" and d.startswith("Prize rules: the official prize page was checked this run")
               for e, d in log_rows(first_vault))
    runner.run(dry_run=True, vault=seeded, fetcher=fake_site(toto), now=NOW + timedelta(hours=1),
               out=lambda s: None)
    later = log_rows(seeded)[len(log_rows(first_vault)):]
    assert not any(e == "FETCH" and d.startswith("Prize rules") for e, d in later)  # cache reused: no row

    # after PRIZE_RULES_MAX_AGE_DAYS the page is checked again and logged
    week = NOW + timedelta(days=runner.PRIZE_RULES_MAX_AGE_DAYS + 1)
    runner.run(dry_run=True, vault=seeded, fetcher=fake_site(toto), now=week, out=lambda s: None)
    assert any(e == "FETCH" and d.startswith("Prize rules: the official prize page was checked this run")
               for e, d in log_rows(seeded, week))


def test_prize_rules_row_says_when_the_page_could_not_be_read(tmp_path):
    rules = prize_rules.load_prize_rules(FakeFetcher({}), cache_path=tmp_path / "prize_rules.json", now=NOW)
    assert rules.checked_at == NOW.isoformat(timespec="seconds")
    text = runner._rules_text(rules, NOW)
    assert text.startswith("Prize rules: the official prize page could not be read this run.")
    assert "checked this run" not in text and not contains_dash(text)
    assert runner._rules_text(PrizeRules(), NOW).startswith("Prize rules: built in values")
    # rules checked at another time (reused from the cache) give no row
    assert runner._rules_text(rules, NOW + timedelta(hours=1)) is None


def test_analysis_failures_and_commentary_failures_get_error_rows(seeded, toto, monkeypatch, no_send):
    from huatbot import commentary

    def broken(*a, **k):
        raise ValueError("bad row")
    monkeypatch.setattr(outlook, "jackpot_history", broken)
    monkeypatch.setenv("COMMENTARY", "claude")
    monkeypatch.setattr(commentary, "build_commentary", lambda figures, **kw: None)
    result = runner.run(dry_run=True, vault=seeded, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert result.ok
    assert "The jackpot history could not be worked out in this run (ValueError)." in result.warnings
    rows = log_rows(seeded)
    assert ("ERROR", "The jackpot history could not be worked out in this run (ValueError)") in rows
    assert any(e == "ERROR" and "Commentary is turned on but none was added" in d for e, d in rows)
    assert "Over " not in result.messages[1]  # no history line without a history


def test_no_token_with_a_local_env_file_explains_why(seeded, toto, no_send, tmp_path, monkeypatch):
    # The CLI reads ./.env (cli.load_dotenv), so a .env without usable values is what to fix.
    workdir = tmp_path / "project"
    workdir.mkdir()
    (workdir / ".env").write_text("TELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=@huat\n", encoding="utf-8")
    monkeypatch.chdir(workdir)
    result = runner.run(vault=seeded, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert not result.posted
    hint = [w for w in result.warnings if "Telegram is not set up" in w]
    assert hint and "The .env file in the working folder has no TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID" in hint[0]
    assert "only docker compose" not in hint[0] and "export" not in hint[0]
    # ENV_FILE names another file
    other = workdir / "bot.env"
    other.write_text("TELEGRAM_CHAT_ID=@huat\n", encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(other))
    result = runner.run(vault=seeded, fetcher=fake_site(toto), now=NOW + timedelta(minutes=1), out=lambda s: None)
    assert any("The ENV_FILE settings file has no TELEGRAM_BOT_TOKEN" in w for w in result.warnings)


# posting


def test_a_partly_posted_set_is_finished_not_posted_again(seeded, toto, tg, monkeypatch):
    vault = seeded
    real_send = tg.send_message
    calls = {"n": 0}

    def flaky(token, chat_id, text, parse_mode="HTML", **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise telegram.TelegramError("Telegram server error (HTTP 502)")
        return real_send(token, chat_id, text, parse_mode=parse_mode, **kw)
    monkeypatch.setattr(telegram, "send_message", flaky)

    first = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert not first.ok and not first.posted
    assert [s["text"] for s in tg.sent] == first.messages[:1]
    state = vault.load_state()
    assert state["posting"] == {"draws": {"toto": TOTO_LAST_DRAW}, "sent": 1, "total": 2}
    assert "last_posted" not in state

    again = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=10), out=lambda s: None)
    assert again.ok and again.posted
    assert [s["text"] for s in tg.sent] == first.messages[:1] + again.messages[1:]
    assert len(tg.sent) == 2  # message 1 was not posted twice
    state = vault.load_state()
    assert "posting" not in state
    assert state["last_posted"] == {"toto": TOTO_LAST_DRAW}
    assert ("POST", f"Posted the remaining 1 message to Telegram (TOTO {TOTO_LAST_DRAW}); the first 1 message went "
                    "out in an earlier run, so they were not posted again") in log_rows(vault)


def test_a_partly_posted_set_of_an_older_draw_is_posted_in_full(seeded, toto, tg):
    vault = seeded
    state = vault.load_state()
    state["posting"] = {"draws": {"toto": TOTO_LAST_DRAW - 1}, "sent": 1, "total": 2}
    vault.save_state(state)
    result = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert result.posted and [s["text"] for s in tg.sent] == result.messages


def test_tickets_checked_before_a_failed_post_are_posted_on_the_retry(seeded, toto, tg):
    vault = seeded
    assert runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None).posted
    sent_before = len(tg.sent)
    bigger = toto_151()

    # Monday: the tickets for 5 Oct are checked and saved, then Telegram fails.
    tg.fail = True
    failed = runner.run(vault=vault, fetcher=fake_site(bigger), now=MONDAY, out=lambda s: None)
    assert not failed.ok and not failed.posted
    ledger = load_ledger(vault.ledger_csv)
    monday_rows = ledger[ledger["draw_date"] == "2026-10-05"]
    assert set(monday_rows["status"]) == {"settled"}
    assert sorted(vault.load_state()["unposted_settled"]) == sorted(monday_rows["ticket_id"])

    # The retry 10 minutes later checks nothing new, but its message 1 still lists them.
    tg.fail = False
    retry = runner.run(vault=vault, fetcher=fake_site(bigger), now=MONDAY + timedelta(minutes=10),
                       out=lambda s: None)
    assert retry.ok and retry.posted
    assert len(tg.sent) == sent_before + 2
    message1 = tg.sent[sent_before]["text"]
    assert "No tickets were checked in this run" not in message1
    assert message1.count("Mon 5 Oct 2026, 3 11 19 27 38 45") == 2  # the Ordinary and the System 7 ticket
    assert "unposted_settled" not in vault.load_state()
    assert "Checked in this run: 2 tickets" in vault.read_text("Home.md")

    # Once posted, they are not listed again.
    later = runner.run(vault=vault, fetcher=fake_site(bigger), now=MONDAY + timedelta(minutes=20),
                       out=lambda s: None, force_post=True)
    assert "No tickets were checked in this run" in later.messages[0]


def test_a_dry_run_leaves_unposted_tickets_alone(seeded, toto, no_send):
    vault = seeded
    state = vault.load_state()
    state["unposted_settled"] = ["abc"]
    vault.save_state(state)
    runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert vault.load_state()["unposted_settled"] == ["abc"]


def test_next_draw_page_not_updated_yet_is_warned_about(seeded, toto, no_send):
    # The next draw page still shows the draw just held (TOTO Thu 1 Oct) with a jackpot of its own.
    site = fake_site(toto, next_toto=next_toto_on(TOTO_LAST_DATE, 1_000_000.0))
    result = runner.run(dry_run=True, vault=seeded, fetcher=site, now=NOW, out=lambda s: None)
    assert ("The next TOTO draw page has not been updated yet (it still shows the draw on Thu 1 Oct 2026), "
            "so the next jackpot is worked out from the stored results." in result.warnings)
    assert not any("may be out of date" in w for w in result.warnings)
    expected = outlook.estimate_next_jackpot(load_toto(seeded.toto_csv), PrizeRules())
    m2 = result.messages[1]
    assert f"<b>Jackpot ${expected:,.0f}</b>" in m2
    assert "$1,000,000" not in m2
    signal = [d for e, d in log_rows(seeded) if e == "SIGNAL"]
    assert signal[-1].startswith(f"Buy signal HIGH for TOTO draw: jackpot ${expected:,.0f}, cascade draw")


def test_checked_tickets_are_recorded_before_the_ledger_is_saved(seeded, toto, tg, monkeypatch):
    # If the container is stopped right after ledger.csv is written, state.json already says
    # which tickets were checked but not posted, so the catch up run still posts them.
    vault = seeded
    assert runner.run(vault=vault, fetcher=fake_site(toto), now=NOW, out=lambda s: None).posted
    on_disk: list = []
    real = store.save_ledger

    def spy(df, path):
        on_disk.append(vault.load_state().get("unposted_settled"))
        return real(df, path)
    monkeypatch.setattr(store, "save_ledger", spy)
    assert runner.run(vault=vault, fetcher=fake_site(toto_151()), now=MONDAY, out=lambda s: None).posted
    assert on_disk and len(on_disk[0]) == 2
    assert "unposted_settled" not in vault.load_state()


def test_completed_draw_gets_its_note_rewritten(tmp_path):
    from huatbot import notes
    vault = Vault(tmp_path / "vault")
    df = synth_toto(n_draws=10)
    row = df.iloc[-1].copy()
    incomplete = row.copy()
    for g in range(1, 8):
        incomplete[f"g{g}_share"] = float("nan")
        incomplete[f"g{g}_winners"] = 0
    rel, fm, body = notes.toto_draw_note(incomplete)
    vault.write_note(rel, body, fm)
    before = vault.read_text(rel)
    rows = []
    data = runner._Data(toto=df, repaired=[int(row["draw_number"])])
    runner._rewrite_repaired_notes(vault, data, lambda event, text: rows.append((event, text)))
    after = vault.read_text(rel)
    assert after != before
    assert vault.read_note(rel)[1].strip() == notes.toto_draw_note(row)[2].strip()
    assert rows == [("NOTE", f"Updated [[{rel.rsplit('/', 1)[-1].removesuffix('.md')}]] with the complete result")]
    # A draw without a note is left alone (notes for old draws are not created here).
    data.repaired = [int(df.iloc[0]["draw_number"])]
    rows.clear()
    runner._rewrite_repaired_notes(vault, data, lambda event, text: rows.append((event, text)))
    assert rows == []
    assert not vault.exists(notes.toto_draw_note(df.iloc[0])[0])


def test_fetch_command_rewrites_the_note_of_a_draw_it_completes(seeded, toto):
    from huatbot import notes
    vault = seeded
    df = load_toto(vault.toto_csv)
    i = df.index[df["draw_number"] == TOTO_LAST_DRAW][0]
    for g in range(1, 8):  # stored while the winning shares table was not published yet
        df.loc[i, f"g{g}_share"] = float("nan")
        df.loc[i, f"g{g}_winners"] = 0
    save_toto(df, vault.toto_csv)
    rel, fm, body = notes.toto_draw_note(df.loc[i])
    vault.write_note(rel, body, fm)
    before = vault.read_text(rel)

    result = runner.fetch_data(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=5))
    assert result.ok
    assert result.new_draws == []  # completed, not new
    stored = load_toto(vault.toto_csv)
    assert int(stored.loc[stored["draw_number"] == TOTO_LAST_DRAW, "g7_winners"].iloc[0]) > 0
    complete = notes.toto_draw_note(stored[stored["draw_number"] == TOTO_LAST_DRAW].iloc[0])
    assert vault.read_text(rel) != before
    assert vault.read_note(rel)[1].strip() == complete[2].strip()
    assert any(e == "NOTE" and "2026-10-01 TOTO 4123" in d and "complete result" in d for e, d in log_rows(vault))


def test_a_run_rewrites_the_note_of_a_draw_it_completes(seeded, toto, no_send):
    from huatbot import notes
    vault = seeded
    df = load_toto(vault.toto_csv)
    i = df.index[df["draw_number"] == TOTO_LAST_DRAW][0]
    for g in range(1, 8):
        df.loc[i, f"g{g}_share"] = float("nan")
        df.loc[i, f"g{g}_winners"] = 0
    save_toto(df, vault.toto_csv)
    rel, fm, body = notes.toto_draw_note(df.loc[i])
    vault.write_note(rel, body, fm)

    result = runner.run(dry_run=True, vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=5),
                        out=lambda s: None)
    assert result.ok and result.new_draws == []
    stored = load_toto(vault.toto_csv)
    complete = notes.toto_draw_note(stored[stored["draw_number"] == TOTO_LAST_DRAW].iloc[0])
    assert vault.read_note(rel)[1].strip() == complete[2].strip()
    assert ("NOTE", f"Updated [[{rel.rsplit('/', 1)[-1].removesuffix('.md')}]] with the complete result") in \
        log_rows(vault)


# commentary


def test_commentary_dropped_for_restating_the_odds_is_not_logged_as_added(seeded, toto, monkeypatch, no_send):
    from huatbot import commentary
    monkeypatch.setenv("COMMENTARY", "claude")
    monkeypatch.setattr(commentary, "build_commentary",
                        lambda figures, **kw: "Nothing here beats the odds, every draw is independent.")
    result = runner.run(dry_run=True, vault=seeded, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    assert result.ok
    rows = log_rows(seeded)
    assert not any("Added a short commentary" in d for _, d in rows)
    assert any(e == "ERROR" and "only restated the odds" in d for e, d in rows)
    assert "beats the odds" not in "\n".join(result.messages)


def test_commentary_logged_as_added_is_the_text_that_is_shown(seeded, toto, monkeypatch, no_send):
    from huatbot import commentary
    seen: list[dict] = []

    def fake_commentary(figures, **kw):
        seen.append(figures)
        return "Every draw is independent. The jackpot has rolled over three times in a row."
    monkeypatch.setenv("COMMENTARY", "claude")
    monkeypatch.setattr(commentary, "build_commentary", fake_commentary)
    result = runner.run(dry_run=True, vault=seeded, fetcher=fake_site(toto), now=NOW, out=lambda s: None)
    rows = log_rows(seeded)
    assert ("NOTE", "Added a short commentary written from the computed figures") in rows
    assert not any(e == "ERROR" and "Commentary" in d for e, d in rows)
    assert "💬 The jackpot has rolled over three times in a row.</blockquote>" in result.messages[1]
    assert "rolled over" not in result.messages[0]
    assert ("*Commentary:* The jackpot has rolled over three times in a row." in
            Path(result.report_path).read_text(encoding="utf-8"))
    # the figures the commentary was written from: no number picks, no 4D
    assert {"next_toto", "buy_signal", "jackpot_outlook", "jackpot_history", "my_tickets"} <= set(seen[0])
    assert not {k for k in seen[0] if "4d" in k.lower() or "suggest" in k.lower() or "pick" in k.lower()}


def test_a_demo_never_asks_for_commentary(tmp_path, monkeypatch, no_send):
    from huatbot import commentary

    def boom(figures):
        raise AssertionError("a demo never calls out for commentary")
    monkeypatch.setenv("COMMENTARY", "claude")
    monkeypatch.setattr(commentary, "build_commentary", boom)
    result = runner.run(demo=True, vault=tmp_path / "demo", now=NOW, out=lambda s: None)
    assert result.ok
    assert not any(e == "ERROR" for e in events(Vault(tmp_path / "demo")))


# a draw held earlier today


def test_held_draw_gets_no_commentary_figures_or_signal_row():
    ctx = make_context()
    late = variant(ctx, now=datetime(2026, 10, 5, 21, 0, tzinfo=SG))  # TOTO 4124 held at 6.30pm
    figures = runner._commentary_figures(late)
    assert not {"next_toto", "buy_signal"} & set(figures)
    assert "jackpot_outlook" in figures
    rows: list[tuple[str, str]] = []
    runner._log_signal(late, lambda e, m: rows.append((e, m)))
    assert rows == []  # sales are closed

    before = variant(ctx, now=datetime(2026, 10, 5, 17, 0, tzinfo=SG))  # still open for sales
    assert {"next_toto", "buy_signal", "jackpot_outlook"} <= set(runner._commentary_figures(before))
    runner._log_signal(before, lambda e, m: rows.append((e, m)))
    assert [e for e, _ in rows] == ["SIGNAL"]
    assert rows[0][1].startswith(f"Buy signal {ctx.buy_signal.label} for TOTO draw {TOTO_LAST_DRAW + 1}: ")


def test_signal_row_names_the_next_big_prize_once_per_change():
    ctx = make_context()  # 260 draws: one rollover so far, three draws to the cascade
    assert len(ctx.outlook.steps) == 3
    big = ctx.outlook.biggest
    state: dict = {}
    rows: list[tuple[str, str]] = []
    for _ in range(2):
        runner._log_signal(ctx, lambda e, m: rows.append((e, m)), state)
    assert len(rows) == 1
    day = big.draw_date
    assert rows[0][1].endswith(f"; next big prize about ${big.jackpot:,.0f} at the cascade draw on "
                               f"{day:%a} {day.day} {day:%b %Y} if nobody wins it first")
    assert set(state["last_logged"]) == {runner.SIGNAL_LOG_KEY}
    # without state every call logs
    runner._log_signal(ctx, lambda e, m: rows.append((e, m)))
    assert len(rows) == 2


# activity log rows that should not repeat


def test_next_draw_row_is_logged_once_when_the_run_follows_the_refresh(seeded, toto, no_send):
    vault = seeded
    site = fake_site(toto)

    def next_rows():
        return [d for e, d in log_rows(vault) if e == "FETCH" and d.startswith("Next draw:")]

    start = len(next_rows())
    cycle = NOW + timedelta(hours=1)
    runner.refresh_next_draws(vault, site, now=cycle)  # the scheduler, at the start of the cycle
    runner.run(dry_run=True, vault=vault, fetcher=site, now=cycle + timedelta(minutes=1), out=lambda s: None)
    assert len(next_rows()) == start + 1
    # A retry 10 minutes later reads the page again and says so.
    runner.run(dry_run=True, vault=vault, fetcher=site, now=cycle + timedelta(minutes=11), out=lambda s: None)
    assert len(next_rows()) == start + 2
    assert vault.load_state()["last_logged"][runner.NEXT_DRAW_LOG_KEY].endswith(
        (cycle + timedelta(minutes=11)).isoformat())


def test_retry_after_a_failed_post_keeps_one_report_note(seeded, toto, tg):
    vault = seeded
    tg.fail = True
    first = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=30), out=lambda s: None)
    assert not first.ok and first.report_path
    reports = sorted(p.name for p in vault.path("Reports").rglob("*.md"))
    tg.fail = False
    again = runner.run(vault=vault, fetcher=fake_site(toto), now=NOW + timedelta(minutes=40), out=lambda s: None)
    assert again.ok and again.posted
    assert sorted(p.name for p in vault.path("Reports").rglob("*.md")) == reports
    assert again.report_path == first.report_path
    report_name = again.report_path.rsplit("/", 1)[-1].removesuffix(".md")
    assert f"Newest report: [[{report_name}]]" in vault.read_text("Home.md")
    assert any(e == "RUN" and f"report [[{report_name}]]" in d for e, d in log_rows(vault))


# guard: the removed features never come back in what the user reads

# Words for the features that were removed: 4D, number suggestions and picks, budgets and plans,
# the System 7 offer, backtests and hot, cold or overdue numbers.
REMOVED_FEATURES = re.compile(
    r"\b4\s?D\b|four\s?d|suggest|\bpick|budget|system 7 offer|back\s?test|\bhot\b|\bcold\b|overdue",
    re.IGNORECASE)


def _user_text_problems(name: str, text: str) -> list[str]:
    # A 4D line the user wrote in Tickets.md is listed in Ledger.md as their own line (inline
    # code) with the reason it is not counted; that line says 4D is not followed, not a feature.
    text = re.sub(r"`[^`\n]*`", "", text).replace(tickets.FOURD_NOT_TRACKED, "")
    return [f"{name}: {m.group(0)!r} in ...{text[max(0, m.start() - 60):m.end() + 60]}..."
            for m in REMOVED_FEATURES.finditer(text)]


def test_no_user_facing_text_mentions_a_removed_feature(first_run, seeded, toto, tg, tmp_path):
    vault, result, printed = first_run
    texts = {f"message {i}": m for i, m in enumerate(result.messages, 1)}
    texts["printed"] = "\n".join(printed)
    for path in sorted(vault.base.rglob("*.md")):
        if path.name != "Tickets.md":  # the user's own note (it holds a 4D line on purpose)
            texts[str(path.relative_to(vault.base))] = path.read_text(encoding="utf-8")
    assert {"Home.md", "Ledger.md", "Reports/2026/10/2026-10-01 1945 Report.md", "Activity/2026/10/2026-10-01.md",
            "Settings.md", f"Draws/TOTO/2026-10-01 TOTO {TOTO_LAST_DRAW}.md"} <= set(texts)
    texts["Tickets template"] = tickets.TICKETS_TEMPLATE
    texts["Settings template"] = runner.SETTINGS_TEMPLATE
    texts["warnings"] = "\n".join(result.warnings)

    # Monday: a normal next draw with the projection table, posted to (fake) Telegram
    monday = runner.run(vault=seeded, fetcher=fake_site(toto_151()), now=MONDAY, out=lambda s: None)
    assert monday.posted
    for i, sent in enumerate(tg.sent, 1):
        texts[f"posted {i}"] = sent["text"]
    texts["Monday dashboard"] = seeded.read_text("Home.md")
    texts["Monday report"] = Path(monday.report_path).read_text(encoding="utf-8")
    texts["Monday log"] = seeded.read_text(log_note_path(MONDAY))

    # a demo vault
    demo = tmp_path / "guard demo"
    printed_demo: list[str] = []
    runner.run(demo=True, vault=demo, now=NOW, out=printed_demo.append)
    texts["demo printed"] = "\n".join(printed_demo)
    texts["demo dashboard"] = Vault(demo).read_text("Home.md")
    texts["demo tickets"] = Vault(demo).read_text("Tickets.md")

    problems = [p for name, text in texts.items() for p in _user_text_problems(name, text or "")]
    assert problems == []
    # and the prose of every note and message stays dash free
    for name, text in texts.items():
        if name.startswith(("message", "posted")):
            assert not contains_dash(text), name
        elif name.endswith(".md") and not name.startswith("Activity/"):
            assert not has_prose_dashes(text), name


def test_guard_catches_the_removed_features():
    for text in ("4D draw 5432", "Suggested numbers", "Hot numbers", "your TOTO budget", "Backtest results",
                 "overdue numbers", "System 7 offer", "Cold numbers", "our picks"):
        assert _user_text_problems("x", text), text
    assert _user_text_problems("x", f"* Line 9: `| 4D | 3 Oct 2026 | 1234 | Big | $1 |` "
                                    f"{tickets.FOURD_NOT_TRACKED}.") == []
