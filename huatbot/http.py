"""Polite HTTP fetching for the Singapore Pools pages.

``Fetcher`` sends a browser User Agent, keeps at most ``max_workers`` requests in
flight, pauses after every request, retries transient failures with exponential
backoff and slows the whole crawl down when the site pushes back (429 or 5xx),
easing back to the normal pace once requests succeed again.

The session and the sleep function are injectable so tests never touch the
network and never sleep for real.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests

from . import constants as C

log = logging.getLogger(__name__)

BACKOFF_BASE = 2.0  # first retry waits 2 s, then 4, 8, 16 ...
BACKOFF_CAP = 60.0  # never wait longer than this between attempts
MAX_PAUSE = 5.0  # the shared polite pause never grows beyond this
SLOW_DOWN_FLOOR = 0.5  # a slow down always pauses at least this long, even if the base pause is 0
EASE_AFTER = 3  # halve the pause after this many successes in a row

DEFAULT_HEADERS = {
    "User-Agent": C.BROWSER_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-SG,en;q=0.9",
}


class FetchError(Exception):
    """A page could not be fetched. The message is short plain English (no URL, no stack noise)."""

    def __init__(self, message: str, url: str | None = None, status: int | None = None,
                 attempts: int = 0, detail: str | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.attempts = attempts
        self.detail = detail  # raw exception text, for the log only

    @property
    def reason(self) -> str:
        return str(self)


def _response_text(resp: Any) -> str:
    """Body of a response as text.

    requests falls back to Latin 1 for ``text/html`` without a charset, which garbles
    UTF 8 pages, so prefer a UTF 8 decode of the raw bytes in that case.
    """
    content = getattr(resp, "content", None)
    encoding = getattr(resp, "encoding", None)
    if isinstance(content, (bytes, bytearray)) and (encoding is None or str(encoding).lower() in ("iso-8859-1", "latin-1")):
        try:
            return bytes(content).decode("utf-8")
        except UnicodeDecodeError:
            pass
    text = getattr(resp, "text", None)
    return text if isinstance(text, str) else ""


def _retry_after_seconds(resp: Any) -> float | None:
    """Seconds from a numeric Retry After header, if the server sent one."""
    headers = getattr(resp, "headers", None) or {}
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except ValueError:
        return None  # HTTP date form: ignore, the normal backoff applies


class Fetcher:
    """Thread safe polite fetcher. See the module docstring."""

    def __init__(
        self,
        max_workers: int = 4,
        pause: float = 0.4,
        max_retries: int = 5,
        timeout: float = 30,
        session: Any = None,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.base_pause = max(0.0, float(pause))
        self.max_retries = max(0, int(max_retries))  # retries after the first attempt
        self.timeout = timeout
        self.headers = dict(DEFAULT_HEADERS)
        self._session = session  # injected session shared by all workers (tests)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._local = threading.local()  # one requests.Session per worker thread otherwise
        self._own_sessions: list[requests.Session] = []
        self._pause = self.base_pause
        self._success_streak = 0
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "slow_downs": 0}

    # shared pace

    @property
    def pause(self) -> float:
        """Current polite pause in seconds (grows while the site struggles)."""
        with self._lock:
            return self._pause

    def _slow_down(self) -> None:
        with self._lock:
            old = self._pause
            self._pause = min(MAX_PAUSE, max(self._pause * 2, SLOW_DOWN_FLOOR))
            self._success_streak = 0
            self.stats["slow_downs"] += 1
        if self._pause != old:
            log.info("Site is struggling, pause raised from %.2f s to %.2f s", old, self._pause)

    def _ease(self) -> None:
        with self._lock:
            self._success_streak += 1
            if self._pause > self.base_pause and self._success_streak >= EASE_AFTER:
                self._pause = max(self.base_pause, self._pause / 2)
                self._success_streak = 0
                log.info("Site is responding again, pause eased to %.2f s", self._pause)

    def _polite_pause(self) -> None:
        delay = self.pause
        if delay > 0:
            self._sleep(delay)

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    # sessions

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update(self.headers)
            self._local.session = sess
            with self._lock:
                self._own_sessions.append(sess)
        return sess

    def close(self) -> None:
        """Close the sessions this fetcher created (an injected session is left alone)."""
        with self._lock:
            sessions, self._own_sessions = self._own_sessions, []
        for sess in sessions:
            try:
                sess.close()
            except Exception:  # pragma: no cover, closing must never fail a run
                pass

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # fetching

    def get(self, url: str) -> str:
        """Fetch one page and return its text, or raise FetchError."""
        session = self._get_session()
        attempts_allowed = self.max_retries + 1
        reason = "unknown error"
        detail = None
        status = None
        for attempt in range(1, attempts_allowed + 1):
            retry_after = None
            self._count("requests")
            try:
                resp = session.get(url, headers=self.headers, timeout=self.timeout)
            except requests.Timeout as exc:
                reason, detail, status = "the site took too long to answer", str(exc), None
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                reason, detail, status = "could not connect to the site", str(exc), None
            except requests.RequestException as exc:
                # Bad URL, too many redirects and the like: retrying will not help.
                self._polite_pause()
                self._count("failures")
                raise FetchError("request could not be made", url=url, attempts=attempt, detail=str(exc)) from exc
            else:
                status = int(getattr(resp, "status_code", 0) or 0)
                if 200 <= status < 300:
                    text = _response_text(resp)
                    if text.strip():
                        self._ease()
                        self._polite_pause()
                        return text
                    reason, detail = "the site returned an empty page", None
                elif status == 404:
                    self._polite_pause()
                    self._count("failures")
                    raise FetchError("page not found (HTTP 404)", url=url, status=404, attempts=attempt)
                elif status in (408, 429) or status >= 500:
                    reason = f"the site returned HTTP {status}"
                    retry_after = _retry_after_seconds(resp) if status == 429 else None
                else:
                    self._polite_pause()
                    self._count("failures")
                    raise FetchError(f"the site refused the request (HTTP {status})", url=url,
                                     status=status, attempts=attempt)

            # Transient failure: slow the whole crawl down, pause, back off and try again.
            self._slow_down()
            self._polite_pause()
            if attempt >= attempts_allowed:
                break
            delay = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
            if retry_after is not None:
                delay = min(BACKOFF_CAP, max(delay, retry_after))
            self._count("retries")
            log.warning("Fetch %s failed (%s), attempt %d of %d, retrying in %.0f s",
                        url, reason, attempt, attempts_allowed, delay)
            self._sleep(delay)

        self._count("failures")
        plural = "attempt" if attempts_allowed == 1 else "attempts"
        raise FetchError(f"{reason} after {attempts_allowed} {plural}", url=url, status=status,
                         attempts=attempts_allowed, detail=detail)

    def _safe_get(self, url: str) -> str | FetchError:
        try:
            return self.get(url)
        except FetchError as exc:
            return exc
        except Exception as exc:  # never let one worker crash the batch
            log.exception("Unexpected error fetching %s", url)
            return FetchError("unexpected error while fetching", url=url, detail=repr(exc))

    def get_many(self, urls: list[str]) -> dict[str, str | FetchError]:
        """Fetch many pages with at most ``max_workers`` in flight.

        Returns url -> page text, or url -> FetchError for pages that failed. Duplicate
        URLs are fetched once. The result keeps the order of ``urls``.
        """
        unique = list(dict.fromkeys(urls))
        if not unique:
            return {}
        results: dict[str, str | FetchError] = {}
        workers = min(self.max_workers, len(unique))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="huatbot-fetch") as pool:
            futures = {pool.submit(self._safe_get, url): url for url in unique}
            done = 0
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if done % 100 == 0:
                    log.info("Fetched %d of %d pages", done, len(unique))
        return {url: results[url] for url in unique}
