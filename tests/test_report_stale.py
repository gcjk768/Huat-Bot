"""End to end: stale next draw info and stale results never reach Telegram as if they were current.

Covers the report side (huatbot.report.next_draw and toto_signal) through runner.run on an
offline fake site, so the messages are the ones the bot would really post.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from huatbot import runner, telegram
from huatbot.models import NextToto
from huatbot.synth import SG
from tests.ctxgen import TOTO_LAST_DRAW, fourd_history, toto_history
from tests.htmlgen import FakeFetcher, fake_site

N_DRAWS = 120
SETTINGS_NOTE = f"""---
toto_start_draw: {TOTO_LAST_DRAW - N_DRAWS + 1}
fourd_history_draws: {N_DRAWS}
backtest_draws: 10
random_sets_per_draw: 20
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


def test_next_draw_page_still_showing_the_draw_just_held(vault_path):
    # 7.45pm on Thu 1 Oct: TOTO 4123 is out, but the next draw page still shows it with its jackpot.
    held = NextToto(draw_datetime=datetime(2026, 10, 1, 18, 30, tzinfo=SG), jackpot_estimate=5e6,
                    draw_type="normal", draw_type_hint=None)
    site = fake_site(toto_history(N_DRAWS), fourd_history(N_DRAWS), next_toto=held)
    result = runner.run(dry_run=True, vault=vault_path, fetcher=site, out=lambda s: None,
                        now=datetime(2026, 10, 1, 19, 45, tzinfo=SG))
    assert result.ok
    assert "$5,000,000" not in result.messages[1]
    assert "not available" in result.messages[1]
    assert any("still shows the draw on Thu 1 Oct 2026" in w for w in result.warnings)


def test_site_unreachable_for_a_week(vault_path):
    first = datetime(2026, 10, 2, 19, 45, tzinfo=SG)
    assert runner.run(demo=True, vault=vault_path, out=lambda s: None, now=first).ok
    later = first + timedelta(days=7)  # Fri 9 Oct: nothing can be fetched
    result = runner.run(dry_run=True, vault=vault_path, fetcher=FakeFetcher({}), out=lambda s: None, now=later)
    assert result.messages, result.warnings
    assert "Results not up to date" in result.messages[0]
    assert any("could not be fetched" in w for w in result.warnings)
    msg2 = result.messages[1]
    for past in ("Mon 5 Oct", "Sat 3 Oct", "Sun 4 Oct", "Wed 7 Oct", "Thu 8 Oct"):
        assert past not in msg2, past
    assert "Mon 12 Oct 2026" in msg2 and "Sat 10 Oct 2026" in msg2
    assert "results are not up to date" in msg2
