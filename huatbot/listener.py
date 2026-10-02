"""Telegram commands and inline buttons: /huat, /huatnext and /huathelp.

Every answer comes from what the last run saved (``Data/last_messages.json``), so a command
never contacts Singapore Pools or Claude. Command names carry the bot's own prefix because the
James Channel forum is shared by many bots and Telegram has no per topic command menu.

Only updates from the configured chat and forum topic are answered: every bot in the group sees
every message, and button data is client supplied, so it is checked against ``COMMANDS``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from . import telegram
from .store import atomic_write_text

log = logging.getLogger(__name__)

LAST_MESSAGES = "last_messages.json"
HELP = (
    "🎱 <b>HUAT BOT</b> · TOTO commands\n\n"
    "🔄 /huat · latest result and next draw\n"
    "🔮 /huatnext · next draw, jackpot and buy signal\n"
    "❓ /huathelp · this list\n\n"
    "<i>Posts by itself after every TOTO draw (Mon and Thu, about 7.30pm).</i>"
)
# Cheap commands only: each one just resends saved text.
COMMANDS = ("huat", "huatnext", "huathelp")
BUTTONS = {"inline_keyboard": [[
    {"text": "🔄 Latest result", "callback_data": "huat"},
    {"text": "🔮 Next draw", "callback_data": "huatnext"},
]]}
NOTHING_YET = "🎱 <b>HUAT BOT</b> · nothing yet\n\n<i>No report has been built yet. The first one comes after the next TOTO draw.</i>"


def save_last_messages(vault, messages: list[str]) -> None:
    """Keep this run's messages for the commands. Best effort, never raises."""
    try:
        atomic_write_text(vault.data_dir / LAST_MESSAGES, json.dumps(messages, ensure_ascii=False))
    except Exception as exc:
        log.warning("Could not save the last messages (%s)", type(exc).__name__)


def load_last_messages(vault) -> list[str]:
    try:
        data = json.loads((vault.data_dir / LAST_MESSAGES).read_text(encoding="utf-8"))
        return [m for m in data if isinstance(m, str) and m.strip()] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def parse_command(text: Any, bot_username: str | None) -> str | None:
    """ "/huatnext@jameskoh_huat_bot extra" -> "huatnext"; None for anything else, including a
    command addressed to another bot."""
    if not isinstance(text, str) or not text.startswith("/"):
        return None
    word = text.split()[0][1:]
    name, _, target = word.partition("@")
    if target and bot_username and target.lower() != bot_username.lower():
        return None
    name = name.lower()
    return name if name in COMMANDS else None


def replies(command: str, vault) -> list[str]:
    """The message(s) a command answers with."""
    if command == "huathelp":
        return [HELP]
    saved = load_last_messages(vault)
    if not saved:
        return [NOTHING_YET]
    return saved[-1:] if command == "huatnext" else saved


def _in_scope(chat: dict | None, thread: Any, want_chat: str, want_topic: int | None) -> bool:
    if not chat or str(chat.get("id")) != want_chat:
        return False
    return want_topic is None or thread == want_topic


def handle(update: dict, *, vault, token: str, chat_id: str, bot_username: str | None,
           send: Callable[..., Any] = telegram.send_message, answer: Callable[..., Any] = telegram.answer_callback,
           log_fn: Callable[[str, str], Any] | None = None) -> str | None:
    """Answer one update; returns the command answered (None when it was ignored)."""
    want_chat, want_topic = telegram.split_chat(chat_id)
    query = update.get("callback_query")
    if query:
        msg = query.get("message") or {}
        command = query.get("data") if query.get("data") in COMMANDS else None
        ok = command is not None and _in_scope(msg.get("chat"), msg.get("message_thread_id"), want_chat, want_topic)
        answer(token, str(query.get("id")), "" if ok else "Not available here")
        if not ok:
            return None
    else:
        msg = update.get("message") or {}
        if not _in_scope(msg.get("chat"), msg.get("message_thread_id"), want_chat, want_topic):
            return None
        command = parse_command(msg.get("text"), bot_username)
        if command is None:
            return None
    out = replies(command, vault)
    for i, text in enumerate(out, 1):
        send(token, chat_id, text, reply_markup=BUTTONS if i == len(out) and command != "huathelp" else None)
    if log_fn is not None:
        log_fn("COMMAND", f"Answered /{command} in Telegram")
    return command


def listen(vault, token: str, chat_id: str, stop: threading.Event | None = None,
           sleep: Callable[[float], Any] = time.sleep) -> None:
    """Long poll until ``stop`` is set. Errors are logged and retried, never raised."""
    stop = stop or threading.Event()
    try:
        bot_username = telegram._call(token, "getMe", {}).get("result", {}).get("username")
    except telegram.TelegramError as exc:
        log.warning("getMe failed (%s), commands to @other bots may be answered", exc)
        bot_username = None
    offset = None
    log.info("Telegram command listener started (/huat, /huatnext, /huathelp)")
    while not stop.is_set():
        try:
            updates = telegram.get_updates(token, offset)
        except telegram.TelegramError as exc:
            log.warning("getUpdates failed: %s", exc)
            sleep(10)
            continue
        for update in updates:
            offset = int(update.get("update_id", 0)) + 1
            try:
                handle(update, vault=vault, token=token, chat_id=chat_id, bot_username=bot_username,
                       log_fn=vault.log)
            except Exception as exc:  # one bad update must not stop the listener
                log.warning("Could not answer a Telegram update (%s: %s)", type(exc).__name__, exc)


def start(vault) -> threading.Thread | None:
    """Run ``listen`` in a daemon thread when a token and chat are configured."""
    token, chat_id = telegram.config_from_env()
    if not token or not chat_id:
        return None
    t = threading.Thread(target=listen, args=(vault, token, chat_id), name="telegram-listener", daemon=True)
    t.start()
    return t
