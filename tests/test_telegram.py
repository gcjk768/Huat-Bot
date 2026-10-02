"""Tests for huatbot.telegram. No network: a fake session stands in for requests."""
from __future__ import annotations

import pytest
import requests

from huatbot import constants as C
from huatbot.telegram import (
    TelegramError,
    config_from_env,
    html_to_plain,
    message_problems,
    post_messages,
    send_message,
)

TOKEN = "123456:ABCdefSecretToken"
CHAT = "@huatbot_channel"


class FakeResponse:
    def __init__(self, status: int, data=None, headers=None, bad_json: bool = False):
        self.status_code = status
        self._data = data
        self.headers = headers or {}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._data


def ok(message_id: int = 1) -> FakeResponse:
    return FakeResponse(200, {"ok": True, "result": {"message_id": message_id}})


class FakeSession:
    """Replays queued responses (or raises queued exceptions) and records every call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.responses.pop(0) if self.responses else ok(len(self.calls))
        if isinstance(item, BaseException):
            raise item
        return item


class Sleeps(list):
    def __call__(self, seconds):
        self.append(seconds)


# config_from_env


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "  123:abc \n")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", '"@mychannel"')
    assert config_from_env() == ("123:abc", "@mychannel")


def test_config_from_env_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "   ")
    assert config_from_env() == (None, None)


# send_message


def test_send_message_posts_expected_request():
    session = FakeSession(ok(42))
    sleeps = Sleeps()
    reply = send_message(TOKEN, CHAT, "<b>Hello</b>", session=session, sleep=sleeps)
    assert reply == {"ok": True, "result": {"message_id": 42}}
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert call["json"] == {
        "chat_id": CHAT,
        "text": "<b>Hello</b>",
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    assert call["timeout"] > 0
    assert sleeps == []


def test_send_message_without_parse_mode():
    session = FakeSession(ok())
    send_message(TOKEN, CHAT, "plain", parse_mode=None, session=session)
    assert "parse_mode" not in session.calls[0]["json"]


def test_send_message_honours_retry_after():
    limited = FakeResponse(429, {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 7",
                                 "parameters": {"retry_after": 7}})
    session = FakeSession(limited, ok())
    sleeps = Sleeps()
    reply = send_message(TOKEN, CHAT, "hi", session=session, sleep=sleeps)
    assert reply["ok"] is True
    assert sleeps == [7.0]
    assert len(session.calls) == 2


def test_send_message_retry_after_header_fallback():
    limited = FakeResponse(429, {"ok": False}, headers={"Retry-After": "3"})
    session = FakeSession(limited, ok())
    sleeps = Sleeps()
    send_message(TOKEN, CHAT, "hi", session=session, sleep=sleeps)
    assert sleeps == [3.0]


def test_send_message_gives_up_after_retries():
    limited = lambda: FakeResponse(429, {"ok": False, "parameters": {"retry_after": 1}})  # noqa: E731
    session = FakeSession(*(limited() for _ in range(10)))
    sleeps = Sleeps()
    with pytest.raises(TelegramError) as err:
        send_message(TOKEN, CHAT, "hi", session=session, retries=2, sleep=sleeps)
    assert len(session.calls) == 3  # first try plus 2 retries
    assert sleeps == [1.0, 1.0]  # no pointless wait after the last attempt
    assert TOKEN not in str(err.value)


def test_send_message_refuses_huge_retry_after():
    session = FakeSession(FakeResponse(429, {"ok": False, "parameters": {"retry_after": 5000}}))
    sleeps = Sleeps()
    with pytest.raises(TelegramError):
        send_message(TOKEN, CHAT, "hi", session=session, sleep=sleeps)
    assert sleeps == []


def test_send_message_retries_server_errors_with_backoff():
    session = FakeSession(FakeResponse(502, None, bad_json=True), FakeResponse(500, {"ok": False}), ok())
    sleeps = Sleeps()
    send_message(TOKEN, CHAT, "hi", session=session, sleep=sleeps)
    assert sleeps == [2.0, 4.0]
    assert len(session.calls) == 3


def test_send_message_retries_connection_errors():
    session = FakeSession(requests.ConnectionError(f"failed for /bot{TOKEN}/sendMessage"), ok())
    sleeps = Sleeps()
    assert send_message(TOKEN, CHAT, "hi", session=session, sleep=sleeps)["ok"]
    assert sleeps == [2.0]


def test_send_message_connection_error_never_leaks_token():
    errors = [requests.ConnectionError(f"Max retries exceeded with url: /bot{TOKEN}/sendMessage") for _ in range(4)]
    session = FakeSession(*errors)
    with pytest.raises(TelegramError) as err:
        send_message(TOKEN, CHAT, "hi", session=session, sleep=Sleeps())
    assert TOKEN not in str(err.value)
    assert "ConnectionError" in str(err.value)
    assert "-" not in str(err.value)


def test_send_message_client_error_is_not_retried():
    bad = FakeResponse(400, {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"})
    session = FakeSession(bad)
    sleeps = Sleeps()
    with pytest.raises(TelegramError, match="chat not found"):
        send_message(TOKEN, CHAT, "hi", session=session, sleep=sleeps)
    assert len(session.calls) == 1 and sleeps == []


def test_send_message_unauthorised():
    session = FakeSession(FakeResponse(401, {"ok": False, "description": "Unauthorized"}))
    with pytest.raises(TelegramError, match="Unauthorized"):
        send_message(TOKEN, CHAT, "hi", session=session)


def test_send_message_falls_back_to_plain_text_on_parse_error():
    bad = FakeResponse(400, {"ok": False, "description": "Bad Request: can't parse entities: unclosed tag"})
    session = FakeSession(bad, ok())
    send_message(TOKEN, CHAT, "<b>Jackpot &amp; more<b>", session=session, sleep=Sleeps())
    assert len(session.calls) == 2
    second = session.calls[1]["json"]
    assert "parse_mode" not in second
    assert second["text"] == "Jackpot & more"


def test_send_message_rejects_long_or_empty_text():
    session = FakeSession()
    with pytest.raises(ValueError):
        send_message(TOKEN, CHAT, "x" * (C.TELEGRAM_MAX_CHARS + 1), session=session)
    with pytest.raises(ValueError):
        send_message(TOKEN, CHAT, "   ", session=session)
    with pytest.raises(ValueError):
        send_message("", CHAT, "hi", session=session)
    assert session.calls == []
    # exactly at the limit is fine
    send_message(TOKEN, CHAT, "x" * C.TELEGRAM_MAX_CHARS, session=session)
    assert len(session.calls) == 1


def test_send_message_ok_false_with_200_is_an_error():
    session = FakeSession(FakeResponse(200, {"ok": False, "description": "odd"}))
    with pytest.raises(TelegramError):
        send_message(TOKEN, CHAT, "hi", session=session)


# post_messages


def test_post_messages_dry_run_prints_and_posts_nothing():
    printed: list[str] = []
    session = FakeSession()
    msgs = ["first <b>one</b>", "second", "third"]
    result = post_messages(msgs, TOKEN, CHAT, dry_run=True, out=printed.append, session=session)
    assert result is False
    assert session.calls == []
    assert "Message 1 of 3" in printed and "Message 2 of 3" in printed and "Message 3 of 3" in printed
    for m in msgs:
        assert m in printed
    assert printed.index("Message 1 of 3") < printed.index(msgs[0]) < printed.index("Message 2 of 3")


def test_post_messages_dry_run_warns_about_rule_breaks():
    printed: list[str] = []
    post_messages(["has a dash - here", "y" * (C.TELEGRAM_MAX_CHARS + 5)], None, None, dry_run=True, out=printed.append)
    warnings = [line for line in printed if line.startswith("Warning:")]
    assert len(warnings) == 2
    assert any("dash" in w for w in warnings)
    assert any("4,005 characters" in w for w in warnings)


def test_post_messages_posts_in_order():
    session = FakeSession()
    sleeps = Sleeps()
    assert post_messages(["a", "b", "c"], TOKEN, CHAT, session=session, sleep=sleeps, out=lambda s: None) is True
    assert [c["json"]["text"] for c in session.calls] == ["a", "b", "c"]
    assert sleeps == [1.0, 1.0]  # pause between messages, not before the first


def test_post_messages_missing_token_falls_back_to_dry_run():
    printed: list[str] = []
    session = FakeSession()
    assert post_messages(["a"], None, CHAT, out=printed.append, session=session) is False
    assert session.calls == []
    assert "Message 1 of 1" in printed


def test_post_messages_checks_every_message_before_sending():
    session = FakeSession()
    with pytest.raises(ValueError, match="Message 2 of 2"):
        post_messages(["fine", "z" * (C.TELEGRAM_MAX_CHARS + 1)], TOKEN, CHAT, session=session, sleep=Sleeps())
    assert session.calls == []


def test_post_messages_reports_partial_failure():
    bad = FakeResponse(403, {"ok": False, "description": "Forbidden: bot is not a member of the channel chat"})
    session = FakeSession(ok(), bad)
    with pytest.raises(TelegramError, match="Posted 1 of 3 messages"):
        post_messages(["a", "b", "c"], TOKEN, CHAT, session=session, sleep=Sleeps())
    assert len(session.calls) == 2


def test_post_messages_reports_progress_after_each_message():
    bad = FakeResponse(403, {"ok": False, "description": "Forbidden"})
    session = FakeSession(ok(), bad)
    sent: list[int] = []
    with pytest.raises(TelegramError):
        post_messages(["a", "b", "c"], TOKEN, CHAT, session=session, sleep=Sleeps(), on_sent=sent.append)
    assert sent == [1]  # only the message that really went out
    sent.clear()
    assert post_messages(["a", "b"], TOKEN, CHAT, session=FakeSession(), sleep=Sleeps(), on_sent=sent.append)
    assert sent == [1, 2]


def test_post_messages_empty_list():
    assert post_messages([], TOKEN, CHAT, session=FakeSession()) is False


def test_message_problems_and_html_to_plain():
    assert message_problems("All good, minus $20") == []
    assert message_problems("") == ["the message is empty"]
    assert html_to_plain("<pre>a &lt; b</pre>") == "a < b"
