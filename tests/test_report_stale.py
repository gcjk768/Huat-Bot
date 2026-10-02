"""End to end: stale next draw info, stale results and a draw held earlier today never reach
Telegram as if they were current.

Covers the report side (huatbot.report.next_draw and toto_signal) through runner.run on an
offline fake site, so the messages are the ones the bot would really post.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from huatbot import constants as C
from huatbot import outlook, runner, telegram
from huatbot.http import FetchError
from huatbot.models import NextToto, PrizeRules
from huatbot.synth import SG, synth_toto
from huatbot.textfmt import contains_dash, money
from tests.ctxgen import TOTO_LAST_DRAW, start_date_for, toto_history
from tests.htmlgen import FakeFetcher, fake_site
from tests.test_report import assert_valid_message, forbidden_mentions

N_DRAWS = 120
SETTINGS_NOTE = f"""---
toto_start_draw: {TOTO_LAST_DRAW - N_DRAWS + 1}
draw_notes_backfill: 2
---
# Settings for the tests
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "COMMENTARY", "DATA_DIR", "VAULT_FOLDER", "VAULT_PATH",
                 "DRY_RUN"):
        monkeypatch.delenv(name, raising=False)

    def boom(*a, **k):
        raise AssertionError("Telegram must not be called")
    monkeypatch.setattr(telegram, "send_message", boom)


@pytest.fixture
def vault_path(tmp_path):
    root = tmp_path / "vault"
    (root / "Huat Bot").mkdir(parents=True)
    (root / "Huat Bot" / "Settings.md").write_text(SETTINGS_NOTE, encoding="utf-8")
    return root


def _at(y, m, d, hh=19, mm=45):
    return datetime(y, m, d, hh, mm, tzinfo=SG)


def _run(vault_path, site, now):
    result = runner.run(dry_run=True, vault=vault_path, fetcher=site, out=lambda s: None, now=now)
    assert result.ok, result.warnings
    assert len(result.messages) == 2
    for msg in result.messages:
        assert_valid_message(msg)
    return result


def _history_to(day: date, last_draw: int):
    """N_DRAWS synthetic draws ending with ``last_draw`` on ``day``."""
    start = start_date_for(day, C.TOTO_WEEKDAYS, N_DRAWS)
    return synth_toto(n_draws=N_DRAWS, start_draw=last_draw - N_DRAWS + 1, start_date=start)


def _activity(vault_path) -> str:
    logs = sorted((vault_path / "Huat Bot").glob("Activity/**/*.md"))
    return "\n".join(p.read_text(encoding="utf-8") for p in logs)


def _notes(vault_path) -> dict[str, str]:
    base = vault_path / "Huat Bot"
    return {str(p.relative_to(base)): p.read_text(encoding="utf-8") for p in base.rglob("*.md")}


def test_next_draw_page_still_showing_the_draw_just_held(vault_path):
    # 7.45pm on Thu 1 Oct: TOTO 4123 is out, but the next draw page still shows it with its jackpot.
    held = NextToto(draw_datetime=_at(2026, 10, 1, 18, 30), jackpot_estimate=5e6, draw_type="normal",
                    draw_type_hint=None)
    history = toto_history(N_DRAWS)
    result = _run(vault_path, fake_site(history, next_toto=held), _at(2026, 10, 1))
    msg2 = result.messages[1]
    assert "$5,000,000" not in msg2
    assert msg2.startswith("🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)\n")
    expected = outlook.estimate_next_jackpot(history, PrizeRules())
    assert f"<b>Jackpot {money(expected)}</b> <i>(worked out from past results)</i>" in msg2
    assert "Buy signal <b>" in msg2 and "<b>NEXT BIG PRIZE</b>" in msg2
    assert any("still shows the draw on Thu 1 Oct 2026" in w for w in result.warnings)
    report_text = (vault_path / "Huat Bot" / "Reports" / "2026" / "10" / "2026-10-01 1945 Report.md").read_text(encoding="utf-8")
    assert "$5,000,000" not in report_text
    assert "still shows the draw on Thu 1 Oct 2026" in report_text
    assert "$5,000,000" not in _notes(vault_path)["Home.md"]


def test_site_unreachable_for_a_week(vault_path):
    first = _at(2026, 10, 2)
    assert runner.run(demo=True, vault=vault_path, out=lambda s: None, now=first).ok
    later = first + timedelta(days=7)  # Fri 9 Oct: nothing can be fetched
    result = _run(vault_path, FakeFetcher({}), later)
    assert any("could not be fetched" in w for w in result.warnings)
    msg1, msg2 = result.messages
    assert ("<i>Results not up to date: the newest stored result is draw 4123, and newer draws have been held "
            "since.") in msg1
    for past in ("Mon 5 Oct", "Thu 8 Oct", "Thu 1 Oct 2026, 6.30pm"):
        assert past not in msg2, past
    assert msg2.startswith("🔮 <b>NEXT TOTO DRAW</b> · Mon 12 Oct 2026, 6.30pm (worked out from the regular schedule, "
                           "results are not up to date)\n")
    assert "(draw " not in msg2.split("\n", 1)[0]
    assert "<i>(worked out from past results)</i>" in msg2  # the stored Mon 5 Oct jackpot is not used
    assert "Mon 12 Oct" in msg2[msg2.index("<b>NEXT BIG PRIZE</b>"):]


def test_draw_held_earlier_today_with_its_result_not_out(vault_path):
    # Mon 5 Oct 9pm: the 6.30pm draw was held but the site has no result for it yet.
    nt = NextToto(draw_datetime=_at(2026, 10, 5, 18, 30), jackpot_estimate=2_400_000.0, draw_type="normal",
                  draw_type_hint=None)
    result = _run(vault_path, fake_site(toto_history(N_DRAWS), next_toto=nt), _at(2026, 10, 5, 21, 0))
    msg1, msg2 = result.messages
    assert "<i>TOTO draw 4124 was held at 6.30pm today, result not out yet.</i>" in msg1
    assert msg2.startswith("🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct 2026, 6.30pm (draw held, result not out "
                           "yet), draw 4124\n")
    assert "Its sales are closed, so there is no buy signal for it." in msg2
    assert "Buy signal" not in msg2 and "back per $1" not in msg2
    activity = _activity(vault_path)
    assert "SIGNAL" not in activity  # sales are closed: no buy signal row either
    assert "Next draw: TOTO Mon 5 Oct 2026, 6.30pm, estimated jackpot $2,400,000" in activity


def test_site_with_a_newer_draw_that_could_not_be_read(vault_path):
    # Tue 6 Oct: the site lists TOTO 4124 (Mon 5 Oct) but its result page cannot be read, so the
    # stored results end at 4123. The next draw page says Thu 8 Oct: that is 4125, not 4124.
    history = _history_to(date(2026, 10, 5), 4124)
    nt = NextToto(draw_datetime=_at(2026, 10, 8, 18, 30), jackpot_estimate=1_500_000.0, draw_type="normal",
                  draw_type_hint=None)
    site = fake_site(history, next_toto=nt)
    url = runner.site.toto_result_url(4124)
    site.pages[url] = FetchError("page not found (HTTP 404)", url=url, status=404, attempts=1)
    result = _run(vault_path, site, _at(2026, 10, 6))
    assert any("does not match the latest draw on the site (4124)" in w for w in result.warnings)
    msg1, msg2 = result.messages
    assert "Results not up to date: the newest stored result is draw 4123" in msg1
    assert "<b>TOTO RESULT</b> · Draw 4123" in msg1
    assert msg2.startswith("🔮 <b>NEXT TOTO DRAW</b> · Thu 8 Oct 2026, 6.30pm (results are not up to date)\n")
    assert "4124" not in msg2 and "4125" not in msg2
    assert "<b>Jackpot $1,500,000</b>" in msg2  # the page jackpot is for that draw


def test_moved_hongbao_draw_keeps_its_number(vault_path):
    # Mon 5 Oct 7.45pm: TOTO 4124 is out and read from the site. The next draw page announces a
    # Hongbao draw on Fri 9 Oct 9.30pm in place of Thursday's: it is draw 4125 and nothing is stale.
    history = _history_to(date(2026, 10, 5), 4124)
    nt = NextToto(draw_datetime=_at(2026, 10, 9, 21, 30), jackpot_estimate=6_000_000.0, draw_type="hongbao",
                  draw_type_hint="hongbao")
    result = _run(vault_path, fake_site(history, next_toto=nt), _at(2026, 10, 5))
    msg1, msg2 = result.messages
    assert "not up to date" not in msg1 and "not up to date" not in msg2
    assert "<b>TOTO RESULT</b> · Draw 4124, Mon 5 Oct 2026" in msg1
    assert msg2.startswith("🔮 <b>NEXT TOTO DRAW</b> · Fri 9 Oct 2026, 9.30pm, draw 4125\n")
    assert "<b>Jackpot $6,000,000</b>" in msg2
    assert "<b>Hongbao draw</b>" in msg2
    assert "The next draw is a Hongbao draw with a jackpot of about $6,000,000." in msg2
    assert "Announced special draws: Fri 9 Oct 2026 (Hongbao)." in msg2


def test_old_fourd_state_and_ledger_are_handled(vault_path):
    # A vault left by the TOTO and 4D version: 4D entries in state.json, a half posted set of the
    # old three messages and a checked 4D ticket in ledger.csv.
    data = vault_path / "Huat Bot" / "Data"
    data.mkdir(parents=True)
    state = {
        "next_draws": {"toto": {"draw_datetime": "2026-10-05T18:30:00+08:00", "jackpot_estimate": 2_000_000,
                                "draw_type": "normal"},
                       "4d": {"draw_datetime": "2026-10-03T18:30:00+08:00"}, "checked_at": "2026-10-01T19:00:00"},
        "last_posted": {"toto": 4122, "4d": 5432},
        "posting": {"draws": {"toto": 4123, "4d": 5432}, "sent": 2, "total": 3},
        "skip": {"toto": [], "4d": [5000]},
        "last_logged": {"SUGGEST toto": "abc", "SIGNAL": "def"},
    }
    (data / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (data / "ledger.csv").write_text(
        "ticket_id,game,draw_date,draw_number,numbers,bet_type,cost,units,status,result,winnings,added_at,"
        "checked_at,source\n"
        "old4d,4D,2026-09-26,5430,1234,Big,1.0,1.0,settled,1st Prize x1,2000.0,2026-09-26T19:00:00+08:00,"
        "2026-09-26T19:30:00+08:00,| 4D | 26 Sep 2026 | 1234 | Big | $1 |\n", encoding="utf-8")
    result = _run(vault_path, fake_site(toto_history(N_DRAWS)), _at(2026, 10, 1))
    msg1, msg2 = result.messages
    assert "All tickets so far: spent $1, won $2,000, net <b>$1,999</b>." in msg1
    for msg in result.messages:
        assert forbidden_mentions(msg) == []
    saved = json.loads((data / "state.json").read_text(encoding="utf-8"))
    for key in ("next_draws", "last_posted", "skip"):
        assert "4d" not in saved.get(key, {}), key
    assert "posting" not in saved
    assert not any(k.startswith("SUGGEST") for k in saved.get("last_logged", {}))
    assert not (data / "fourd.csv").exists()
    assert not (vault_path / "Huat Bot" / "Suggestions").exists()


def test_guard_no_removed_feature_anywhere_in_the_vault(vault_path):
    # A full run, then every note in the bot folder (starter notes, dashboard, ledger, report,
    # draw notes and the activity log) and both messages are checked.
    result = _run(vault_path, fake_site(toto_history(N_DRAWS)), _at(2026, 10, 1))
    written = _notes(vault_path)
    assert {"Settings.md", "Tickets.md", "Home.md", "Ledger.md"} <= set(written)
    assert any(rel.startswith("Reports/") for rel in written)
    assert any(rel.startswith("Draws/TOTO/") for rel in written)
    assert not any("4D" in rel or "Suggestions" in rel for rel in written)
    for rel, text in written.items():
        assert forbidden_mentions(text) == [], rel
    for msg in result.messages:
        assert forbidden_mentions(msg) == []
        assert not contains_dash(msg)
