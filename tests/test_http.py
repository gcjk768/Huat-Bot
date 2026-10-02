"""Fetcher: retries, backoff, slow down, politeness and concurrency, all without network or real sleeps."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
import requests

from huatbot import constants as C
from huatbot import http as H
from huatbot.http import FetchError, Fetcher


@dataclass
class FakeResponse:
    status_code: int
    text: str = ""
    headers: dict = field(default_factory=dict)


class FakeSession:
    """Scripted responses per URL. Each script item is a status code, (status, text),
    (status, text, headers), a FakeResponse or an exception instance. The last item repeats."""

    def __init__(self, scripts: dict | None = None, default: object = (200, "<html>ok</html>")):
        self.scripts = {url: list(items) for url, items in (scripts or {}).items()}
        self.default = default
        self.calls: list[tuple[str, dict, float]] = []
        self._lock = threading.Lock()

    def _next(self, url):
        with self._lock:
            items = self.scripts.get(url)
            if not items:
                return self.default
            return items.pop(0) if len(items) > 1 else items[0]

    def get(self, url, headers=None, timeout=None):
        with self._lock:
            self.calls.append((url, dict(headers or {}), timeout))
        item = self._next(url)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, FakeResponse):
            return item
        if isinstance(item, int):
            return FakeResponse(item, f"<html>{item}</html>")
        return FakeResponse(*item)

    def count(self, url):
        return sum(1 for u, _, _ in self.calls if u == url)


class Sleeps:
    """Thread safe recorder used in place of time.sleep."""

    def __init__(self):
        self.calls: list[float] = []
        self._lock = threading.Lock()

    def __call__(self, seconds):
        with self._lock:
            self.calls.append(seconds)


URL = "https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_draw_list_en.html"


def make(session, **kw):
    sleeps = Sleeps()
    kw.setdefault("pause", 0.4)
    return Fetcher(session=session, sleep=sleeps, **kw), sleeps


def test_success_sends_browser_headers_and_pauses():
    session = FakeSession({URL: [(200, "<html>draw list</html>")]})
    fetcher, sleeps = make(session, timeout=12)
    assert fetcher.get(URL) == "<html>draw list</html>"
    url, headers, timeout = session.calls[0]
    assert headers["User-Agent"] == C.BROWSER_USER_AGENT
    assert headers["Accept-Language"].startswith("en")
    assert timeout == 12
    assert sleeps.calls == [0.4]  # one polite pause, no backoff


def test_retries_connection_error_then_succeeds():
    session = FakeSession({URL: [requests.ConnectionError("boom"), requests.Timeout("slow"), (200, "fine")]})
    fetcher, sleeps = make(session, pause=0.0)
    assert fetcher.get(URL) == "fine"
    assert session.count(URL) == 3
    # a polite pause follows every attempt, a backoff follows every failed attempt but the last
    assert sleeps.calls[1::2] == [2.0, 4.0]
    assert fetcher.stats["retries"] == 2


def test_server_errors_back_off_and_slow_down_then_ease():
    session = FakeSession({URL: [503, 500, (200, "ok")]})
    fetcher, sleeps = make(session, pause=0.4)
    assert fetcher.get(URL) == "ok"
    # pause doubled twice: 0.4 -> 0.8 -> 1.6; backoff 2 then 4
    assert sleeps.calls == [0.8, 2.0, 1.6, 4.0, 1.6]
    assert fetcher.pause == 1.6
    # after a few successes in a row the pause eases back to the base
    other = "https://example.invalid/other"
    for _ in range(3):
        fetcher.get(other)
    assert fetcher.pause == 0.8
    for _ in range(3):
        fetcher.get(other)
    assert fetcher.pause == 0.4
    for _ in range(3):
        fetcher.get(other)
    assert fetcher.pause == 0.4  # never below the base pause


def test_backoff_is_capped_and_gives_up_after_max_retries():
    session = FakeSession({URL: [502]})
    fetcher, sleeps = make(session, pause=0.0, max_retries=7)
    with pytest.raises(FetchError) as info:
        fetcher.get(URL)
    assert session.count(URL) == 8  # first attempt plus 7 retries
    assert sleeps.calls[1::2] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert max(sleeps.calls[0::2]) == H.MAX_PAUSE
    err = info.value
    assert err.status == 502 and err.attempts == 8 and err.url == URL
    assert "HTTP 502" in str(err) and "8 attempts" in str(err)
    assert fetcher.pause == H.MAX_PAUSE  # slowed down to the cap


def test_429_honours_retry_after_header():
    session = FakeSession({URL: [(429, "slow down", {"Retry-After": "17"}), (200, "ok")]})
    fetcher, sleeps = make(session, pause=0.0)
    assert fetcher.get(URL) == "ok"
    assert 17.0 in sleeps.calls


def test_429_retry_after_is_capped():
    session = FakeSession({URL: [(429, "", {"Retry-After": "3600"}), (200, "ok")]})
    fetcher, sleeps = make(session, pause=0.0)
    fetcher.get(URL)
    assert max(sleeps.calls) == H.BACKOFF_CAP


def test_404_fails_immediately():
    session = FakeSession({URL: [404]})
    fetcher, sleeps = make(session)
    with pytest.raises(FetchError) as info:
        fetcher.get(URL)
    assert session.count(URL) == 1
    assert info.value.status == 404
    assert "404" in str(info.value)
    assert sleeps.calls == [0.4]  # still polite, but no backoff


def test_other_client_errors_are_not_retried():
    session = FakeSession({URL: [403]})
    fetcher, _ = make(session)
    with pytest.raises(FetchError) as info:
        fetcher.get(URL)
    assert session.count(URL) == 1 and info.value.status == 403


def test_invalid_request_is_not_retried():
    session = FakeSession({URL: [requests.exceptions.InvalidURL("bad url")]})
    fetcher, _ = make(session)
    with pytest.raises(FetchError):
        fetcher.get(URL)
    assert session.count(URL) == 1


def test_empty_page_is_retried():
    session = FakeSession({URL: [(200, "   "), (200, "<html>real</html>")]})
    fetcher, _ = make(session, pause=0.0)
    assert fetcher.get(URL) == "<html>real</html>"


def test_zero_retries_means_one_attempt():
    session = FakeSession({URL: [requests.ConnectionError("down")]})
    fetcher, sleeps = make(session, pause=0.0, max_retries=0)
    with pytest.raises(FetchError) as info:
        fetcher.get(URL)
    assert session.count(URL) == 1
    assert "1 attempt" in str(info.value)
    assert sleeps.calls == [H.SLOW_DOWN_FLOOR]  # the polite pause only, no backoff


def test_error_messages_are_plain():
    session = FakeSession({URL: [requests.ConnectionError("HTTPSConnectionPool(host='x'): Max retries exceeded -- [Errno -2]")]})
    fetcher, _ = make(session, pause=0.0, max_retries=1)
    with pytest.raises(FetchError) as info:
        fetcher.get(URL)
    msg = str(info.value)
    assert "-" not in msg and "http" not in msg.lower().replace("http ", "")
    assert "Errno" in (info.value.detail or "")


def test_utf8_bytes_preferred_over_latin1_guess():
    class Resp:
        status_code = 200
        encoding = "ISO-8859-1"
        content = "Café – $1,000".encode("utf-8")
        text = content.decode("latin-1")
        headers = {}

    class Sess:
        def get(self, url, headers=None, timeout=None):
            return Resp()

    fetcher = Fetcher(session=Sess(), sleep=lambda s: None)
    assert fetcher.get(URL) == "Café – $1,000"


def test_get_many_returns_text_and_errors_in_order():
    urls = [f"https://example.invalid/{i}" for i in range(10)]
    session = FakeSession({urls[3]: [404], urls[7]: [requests.ConnectionError("down")]})
    fetcher, _ = make(session, pause=0.0, max_retries=1)
    out = fetcher.get_many(urls + [urls[0]])  # duplicate is fetched once
    assert list(out) == urls
    assert isinstance(out[urls[3]], FetchError) and out[urls[3]].status == 404
    assert isinstance(out[urls[7]], FetchError)
    assert all(isinstance(out[u], str) for i, u in enumerate(urls) if i not in (3, 7))
    assert session.count(urls[0]) == 1
    assert fetcher.get_many([]) == {}


class ConcurrencySession:
    """Proves at most max_workers requests are in flight, and that the pool really uses them all.

    The first ``workers`` requests wait on a barrier, so they must be in flight together."""

    def __init__(self, workers):
        self.barrier = threading.Barrier(workers, timeout=10)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.started = 0
        self.workers = workers

    def get(self, url, headers=None, timeout=None):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.started += 1
            first_wave = self.started <= self.workers
        try:
            if first_wave:
                self.barrier.wait()
            return FakeResponse(200, f"<html>{url}</html>")
        finally:
            with self.lock:
                self.in_flight -= 1


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_get_many_concurrency_limit(workers):
    session = ConcurrencySession(workers)
    fetcher, _ = make(session, max_workers=workers, pause=0.0)
    urls = [f"https://example.invalid/p{i}" for i in range(40)]
    out = fetcher.get_many(urls)
    assert all(out[u] == f"<html>{u}</html>" for u in urls)
    assert session.max_in_flight == workers


def test_shared_pause_is_thread_safe_under_errors():
    urls = [f"https://example.invalid/e{i}" for i in range(60)]
    scripts = {u: [503, (200, "ok")] if i % 2 else [(200, "ok")] for i, u in enumerate(urls)}
    session = FakeSession(scripts)
    fetcher, sleeps = make(session, pause=0.1, max_workers=4)
    out = fetcher.get_many(urls)
    assert all(v == "ok" for v in out.values())
    assert 0.1 <= fetcher.pause <= H.MAX_PAUSE
    assert fetcher.stats["slow_downs"] == 30
    assert fetcher.stats["requests"] == 90


def test_unexpected_exception_in_worker_becomes_fetch_error():
    class Broken:
        def get(self, url, headers=None, timeout=None):
            raise RuntimeError("bug")

    fetcher = Fetcher(session=Broken(), sleep=lambda s: None)
    out = fetcher.get_many(["https://example.invalid/a"])
    assert isinstance(out["https://example.invalid/a"], FetchError)


def test_default_session_is_per_thread_and_closed():
    fetcher = Fetcher(sleep=lambda s: None)
    s1 = fetcher._get_session()
    assert fetcher._get_session() is s1
    other = []
    t = threading.Thread(target=lambda: other.append(fetcher._get_session()))
    t.start()
    t.join()
    assert other[0] is not s1
    assert s1.headers["User-Agent"] == C.BROWSER_USER_AGENT
    with fetcher:
        pass
    assert fetcher._own_sessions == []


def test_get_many_reuses_worker_sessions_across_batches(monkeypatch):
    """A long running serve calls get_many many times: the sessions must not pile up."""
    made = []

    class CountingSession:
        def __init__(self):
            self.headers = {}
            self.closed = False
            made.append(self)

        def get(self, url, headers=None, timeout=None):
            return FakeResponse(200, f"<html>{url}</html>")

        def close(self):
            self.closed = True

    monkeypatch.setattr(H.requests, "Session", CountingSession)
    fetcher = Fetcher(max_workers=2, sleep=lambda s: None)
    urls = ["https://example.invalid/a", "https://example.invalid/b"]
    for _ in range(20):
        out = fetcher.get_many(urls)
        assert all(out[u] == f"<html>{u}</html>" for u in urls)
    assert len(fetcher._own_sessions) <= 3 and len(made) <= 3
    fetcher.close()
    assert fetcher._own_sessions == [] and fetcher._pool is None
    assert all(s.closed for s in made)
    # a closed fetcher still works: a fresh pool is started
    assert fetcher.get_many(urls[:1]) == {urls[0]: f"<html>{urls[0]}</html>"}
    fetcher.close()


def test_get_many_stops_early_when_the_site_keeps_failing():
    urls = [f"https://example.invalid/down{i}" for i in range(40)]
    session = FakeSession(default=503)
    fetcher, _ = make(session, pause=0.0, max_workers=1, max_retries=1)
    out = fetcher.get_many(urls)
    assert list(out) == urls
    assert all(isinstance(v, FetchError) and v.transient for v in out.values())
    # 10 pages failed after their retries, the other 30 were never requested
    assert len({u for u, _, _ in session.calls}) == H.STOP_AFTER_FAILURES
    assert len(session.calls) == H.STOP_AFTER_FAILURES * 2
    skipped = out[urls[-1]]
    assert "not tried" in str(skipped) and "-" not in str(skipped)
    # the next batch starts afresh (the site may be back)
    session.default = (200, "<html>back</html>")
    assert fetcher.get_many(urls[:3]) == {u: "<html>back</html>" for u in urls[:3]}


def test_get_many_does_not_stop_for_scattered_failures():
    urls = [f"https://example.invalid/s{i}" for i in range(40)]
    scripts = {u: [503] for i, u in enumerate(urls) if i % 3 == 0}  # every third page is down
    session = FakeSession(scripts)
    fetcher, _ = make(session, pause=0.0, max_workers=1, max_retries=0)
    out = fetcher.get_many(urls)
    assert sum(isinstance(v, FetchError) for v in out.values()) == 14
    assert len({u for u, _, _ in session.calls}) == 40


def test_not_found_is_not_a_transient_failure():
    session = FakeSession({URL: [404]})
    fetcher, _ = make(session, pause=0.0)
    with pytest.raises(FetchError) as info:
        fetcher.get(URL)
    assert info.value.transient is False
