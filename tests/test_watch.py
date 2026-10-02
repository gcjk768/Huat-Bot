from datetime import datetime
from zoneinfo import ZoneInfo

from huatbot import watch

SG = ZoneInfo("Asia/Singapore")
MON = {"date": "2026-10-05T18:30+08:00", "jackpot": 1_000_000.0, "type": "normal"}


def test_alert_hours_and_window():
    assert watch.alert_hours({"ALERT_HOURS": "8-23"}) == (8, 23)
    assert watch.alert_hours({"ALERT_HOURS": "nonsense"}) == (9, 21)
    assert watch.in_hours(datetime(2026, 10, 3, 9, 0, tzinfo=SG), (9, 22))
    assert not watch.in_hours(datetime(2026, 10, 3, 22, 0, tzinfo=SG), (9, 22))


def test_findings():
    assert watch.findings(None, MON) == []  # first look only stores the snapshot
    assert watch.findings(MON, dict(MON)) == []
    up = watch.findings(MON, {**MON, "jackpot": 1_500_000.0})
    assert len(up) == 1 and "🟢" in up[0] and "▲$500,000" in up[0]
    assert "🔴" in watch.findings(MON, {**MON, "jackpot": 900_000.0})[0]
    # the regular next draw after a draw day is not a finding (the draw day post covers it)
    assert watch.findings(MON, {"date": "2026-10-08T18:30+08:00", "jackpot": 1_300_000.0, "type": "normal"}) == []
    hb = watch.findings(MON, {"date": "2026-10-09T18:30+08:00", "jackpot": 4_000_000.0, "type": "hongbao"})
    assert "Hongbao draw" in hb[0] and "$4,000,000" in hb[1]
    text = watch.card(MON, up)
    assert text.startswith("🆕 <b>TOTO UPDATE</b> · ") and "-" not in text.split("\n", 1)[1]
