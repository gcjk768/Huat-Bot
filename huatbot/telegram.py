"""Post messages to a Telegram chat or channel through the Bot API (plain ``requests``).

Only one endpoint is used: ``sendMessage``. Rate limits (HTTP 429) are honoured using the
``retry_after`` value Telegram returns; server errors and connection problems are retried with
a short exponential backoff. The bot token is never written to logs or error messages.
"""
from __future__ import annotations

import html
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

import requests

from . import constants as C
from .textfmt import contains_dash, fmt_num, remove_dashes

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT = 30.0  # seconds per HTTP request
MAX_RETRY_AFTER = 900.0  # give up rather than block a run for longer than this on a 429
BACKOFF_START = 2.0
BACKOFF_MAX = 60.0
DEFAULT_RETRY_AFTER = 5.0  # used when a 429 carries no retry_after


class TelegramError(Exception):
    """Telegram refused the message, or it could not be delivered after retrying."""


def _clean_env(value: str | None) -> str | None:
    """Strip whitespace and one pair of surrounding quotes (common in hand written .env files)."""
    if value is None:
        return None
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v or None


def config_from_env() -> tuple[str | None, str | None]:
    """(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) from the environment; missing or blank -> None."""
    return _clean_env(os.environ.get("TELEGRAM_BOT_TOKEN")), _clean_env(os.environ.get("TELEGRAM_CHAT_ID"))


def message_problems(text: str) -> list[str]:
    """Plain reasons a message breaks the posting rules (empty list when it is fine)."""
    problems = []
    if not isinstance(text, str) or not text.strip():
        return ["the message is empty"]
    if len(text) > C.TELEGRAM_MAX_CHARS:
        problems.append(
            f"the message is {fmt_num(len(text))} characters, over the {fmt_num(C.TELEGRAM_MAX_CHARS)} limit"
        )
    if contains_dash(text):
        problems.append("the message contains a dash")
    return problems


def html_to_plain(text: str) -> str:
    """Telegram HTML to plain text: drop tags, unescape entities (used when HTML parsing fails)."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def _redact(text: str, token: str | None) -> str:
    """Remove the bot token (and the /bot<token> URL part) from a message."""
    if token:
        text = text.replace(token, "<token>")
    return re.sub(r"/bot[^/\s]+/", "/bot<token>/", text)


def _short_error(exc: BaseException, token: str | None) -> str:
    """One line, dash free description of an exception, safe to log."""
    detail = _redact(str(exc), token).splitlines()[0] if str(exc) else ""
    detail = remove_dashes(detail)[:200]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _json(resp: Any) -> dict:
    try:
        data = resp.json()
    except Exception:  # not JSON (proxy error page and the like)
        return {}
    return data if isinstance(data, dict) else {}


def _description(data: dict, status: int) -> str:
    desc = str(data.get("description") or "").strip()
    return remove_dashes(desc) if desc else f"HTTP {status}"


def _retry_after(data: dict, resp: Any) -> float:
    """Seconds Telegram asked us to wait: parameters.retry_after, else the Retry-After header."""
    params = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
    for value in (params.get("retry_after"), (getattr(resp, "headers", None) or {}).get("Retry-After")):
        try:
            if value is not None and float(value) >= 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return DEFAULT_RETRY_AFTER


def send_message(
    token: str,
    chat_id: str | int,
    text: str,
    parse_mode: str | None = "HTML",
    session: Any = None,
    retries: int = 3,
    sleep: Callable[[float], Any] = time.sleep,
    *,
    timeout: float = REQUEST_TIMEOUT,
    api_base: str = API_BASE,
) -> dict:
    """Send one message and return Telegram's JSON reply ({"ok": true, "result": {...}}).

    ``retries`` is the number of extra attempts after the first one. HTTP 429 waits
    ``parameters.retry_after`` seconds (via ``sleep``); 5xx and connection errors back off
    2, 4, 8 ... seconds. Other refusals raise TelegramError at once, except an HTML parse error,
    which is retried once as plain text so the message still arrives. ``session`` is anything
    with ``post(url, json=..., timeout=...)``, by default the ``requests`` module.

    Raises ValueError for a missing token or chat id, empty text, or text longer than
    constants.TELEGRAM_MAX_CHARS.
    """
    if not token or chat_id is None or str(chat_id).strip() == "":
        raise ValueError("Telegram bot token and chat id are both required")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Telegram message text is empty")
    if len(text) > C.TELEGRAM_MAX_CHARS:
        raise ValueError(
            f"Telegram message is {fmt_num(len(text))} characters, over the {fmt_num(C.TELEGRAM_MAX_CHARS)} limit"
        )

    url = f"{api_base.rstrip('/')}/bot{token}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    http = session if session is not None else requests

    attempts = max(1, int(retries) + 1)
    backoff = BACKOFF_START
    last_error = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            resp = http.post(url, json=payload, timeout=timeout)
        except (requests.RequestException, OSError) as exc:
            last_error = _short_error(exc, token)
            log.warning("Telegram request failed (attempt %d of %d): %s", attempt, attempts, last_error)
            if attempt < attempts:
                sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        status = int(getattr(resp, "status_code", 0) or 0)
        data = _json(resp)
        if status == 200 and data.get("ok"):
            return data
        desc = _description(data, status)

        if status == 429:
            wait = _retry_after(data, resp)
            last_error = f"rate limited by Telegram ({desc})"
            if wait > MAX_RETRY_AFTER:
                raise TelegramError(f"Telegram asked to wait {fmt_num(wait)} seconds, which is too long; message not sent")
            log.warning("Telegram rate limit, waiting %s seconds (attempt %d of %d)", fmt_num(wait, 1), attempt, attempts)
            if attempt < attempts:
                sleep(wait)
            continue

        if status >= 500 or status == 0:
            last_error = f"Telegram server error ({desc})"
            log.warning("Telegram server error %s (attempt %d of %d): %s", status, attempt, attempts, desc)
            if attempt < attempts:
                sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        if status == 400 and parse_mode and "parse" in desc.lower():
            log.warning("Telegram could not parse the %s message (%s), sending it as plain text", parse_mode, desc)
            return send_message(
                token, chat_id, html_to_plain(text), parse_mode=None, session=session, retries=retries,
                sleep=sleep, timeout=timeout, api_base=api_base,
            )

        raise TelegramError(_redact(f"Telegram refused the message: {desc}", token))

    raise TelegramError(_redact(f"Telegram message not sent after {attempts} attempts: {last_error}", token))


def post_messages(
    messages: Sequence[str],
    token: str | None,
    chat_id: str | int | None,
    dry_run: bool = False,
    out: Callable[[str], Any] = print,
    pause: float = 1.0,
    **kw: Any,
) -> bool:
    """Post the messages in order. Returns True only when every message was posted.

    dry_run: print each message under a "Message 1 of 3" header (plus a warning line for any
    rule it breaks) and return False. A missing token or chat id falls back to a dry run with a
    warning. Before posting, every message is checked so nothing is half posted because of a
    bad message (ValueError). A send failure raises TelegramError saying how many went out.
    ``pause`` seconds pass between messages; other keywords go to ``send_message``.
    """
    messages = list(messages)
    total = len(messages)

    if not dry_run and (not token or chat_id is None or str(chat_id).strip() == ""):
        log.warning("Telegram bot token or chat id is not set, printing the messages instead of posting")
        dry_run = True

    if dry_run:
        for i, text in enumerate(messages, 1):
            out(f"Message {i} of {total}")
            out(text)
            for problem in message_problems(text):
                log.warning("Message %d of %d: %s", i, total, problem)
                out(f"Warning: {problem}")
            out("")
        return False

    if not messages:
        log.info("No Telegram messages to post")
        return False
    for i, text in enumerate(messages, 1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Message {i} of {total} is empty")
        if len(text) > C.TELEGRAM_MAX_CHARS:
            raise ValueError(
                f"Message {i} of {total} is {fmt_num(len(text))} characters, "
                f"over the {fmt_num(C.TELEGRAM_MAX_CHARS)} limit"
            )
        if contains_dash(text):
            log.warning("Message %d of %d contains a dash", i, total)

    sleep = kw.get("sleep", time.sleep)
    for i, text in enumerate(messages, 1):
        if i > 1 and pause:
            sleep(pause)
        try:
            send_message(token, chat_id, text, **kw)
        except TelegramError as exc:
            raise TelegramError(f"Posted {i - 1} of {total} messages, message {i} failed: {exc}") from exc
        log.info("Posted Telegram message %d of %d", i, total)
    return True
