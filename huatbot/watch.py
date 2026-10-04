"""Hourly light check of the next draw page: post a 🆕 card the moment something new turns up.

A finding is a change on the next draw page since the last card: the estimated jackpot moved,
or a Cascade, Hongbao or special draw was announced. Checks only run inside ALERT_HOURS
(default 09-21 Singapore time, so it stays clear of the trading desk's US session from 21:30); a change made overnight is found by the first check of the
morning, because each check compares against the last snapshot it alerted on
(``Data/watch.json``), not against the previous hour. The normal draw day posts stay with the
scheduler; this never touches state.json.
"""
from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import constants as C
from . import fetch as site
from . import listener, store, telegram
from .store import atomic_write_text
from .textfmt import fmt_datetime, money

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)
SNAPSHOT = "watch.json"
SPECIAL_TYPES = ("cascade", "hongbao", "special")
TYPE_EMOJI = {"cascade": "🌊", "hongbao": "🧧", "special": "⭐"}


def alert_hours(env: dict | None = None) -> tuple[int, int]:
    """ALERT_HOURS "9-21" -> (9, 21): checks run from 9:00 until 20:59. Bad values -> (9, 21)."""
    raw = (env if env is not None else os.environ).get("ALERT_HOURS", "9-21")
    try:
        start, end = (int(x) for x in str(raw).replace(" ", "").split("-"))
        if 0 <= start < end <= 24:
            return start, end
    except ValueError:
        pass
    log.warning("ALERT_HOURS %r is not like 9-21, using 9-21", raw)
    return 9, 21


def in_hours(now: datetime, hours: tuple[int, int]) -> bool:
    return hours[0] <= now.astimezone(SG).hour < hours[1]


def snapshot(nt) -> dict | None:
    if nt is None or nt.draw_datetime is None:
        return None
    return {"date": nt.draw_datetime.astimezone(SG).isoformat(timespec="minutes"),
            "jackpot": nt.jackpot_estimate, "type": nt.draw_type_hint or "normal"}


def findings(old: dict | None, new: dict | None) -> list[str]:
    """Telegram lines for what changed between two snapshots (empty when nothing worth a card)."""
    if not new or not old:
        return []
    lines = []
    if new["type"] in SPECIAL_TYPES and (new["type"] != old.get("type") or new["date"] != old.get("date")):
        lines.append(f"{TYPE_EMOJI[new['type']]} <b>{new['type'].title()} draw</b> announced")
    same_draw = new["date"] == old.get("date")
    j_old, j_new = old.get("jackpot"), new.get("jackpot")
    if j_new and (not same_draw or j_old) and j_new != j_old:
        if same_draw:
            up = j_new > j_old
            marker = (f"🟢 <i>UP ▲{html.escape(money(j_new - j_old))}</i>" if up
                      else f"🔴 <i>DOWN ▼{html.escape(money(j_old - j_new))}</i>")
            lines.append(f"💰 Jackpot estimate <b>{html.escape(money(j_new))}</b> {marker}")
        elif lines:  # a new special draw: say its jackpot too
            lines.append(f"💰 Jackpot estimate <b>{html.escape(money(j_new))}</b>")
    return lines


def card(new: dict, lines: list[str]) -> str:
    when = fmt_datetime(datetime.fromisoformat(new["date"]))
    return "\n".join([f"🆕 <b>TOTO UPDATE</b> · {html.escape(when)}", ""] + lines
                     + ["", "<i>Tap 🔮 Next draw for the full outlook.</i>"])


def check_once(vault, fetcher, *, send: Callable[..., Any] = telegram.send_message,
               now: datetime | None = None) -> bool:
    """One check: returns True when a card went out. Never raises past a logged warning."""
    path = vault.data_dir / SNAPSHOT
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    new = snapshot(site.fetch_next_draw(fetcher, store.load_toto(vault.toto_csv)))
    if new is None:
        return False
    lines = findings(old, new)
    sent = False
    if lines:
        token, chat_id = telegram.config_from_env()
        if not token or not chat_id:
            log.info("Finding not posted (no Telegram token): %s", lines)
            return False
        send(token, chat_id, card(new, lines), reply_markup=listener.BUTTONS)
        vault.log("POST", "Posted a TOTO update card: " + "; ".join(
            html.unescape(ln).replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
            for ln in lines))
        sent = True
    if new != old:
        atomic_write_text(path, json.dumps(new))
    return sent


def watch(vault, fetcher, stop: threading.Event | None = None, interval: float = 3600,
          sleep: Callable[[float], Any] = time.sleep) -> None:
    stop = stop or threading.Event()
    hours = alert_hours()
    log.info("Finding watcher started: every %d minutes, %02d:00 to %02d:00", interval // 60, *hours)
    while not stop.is_set():
        if in_hours(datetime.now(SG), hours):
            try:
                check_once(vault, fetcher)
            except Exception as exc:  # a bad hour must not stop the watcher
                log.warning("Finding check failed (%s: %s)", type(exc).__name__, exc)
        sleep(interval)


def start(vault, fetcher) -> threading.Thread:
    minutes = float(os.environ.get("WATCH_MINUTES") or 1440)
    t = threading.Thread(target=watch, args=(vault, fetcher), kwargs={"interval": max(15.0, minutes) * 60},
                         name="finding-watcher", daemon=True)
    t.start()
    return t
