"""Bring toto.csv up to date from the Singapore Pools site.

Every function takes a fetcher (``http.Fetcher`` or a test double with the same
``get`` / ``get_many`` interface), so nothing here talks to the network directly.
Updates are incremental: only draws missing from the frame are fetched, each page is
checked to be the draw that was asked for, and the newest row is compared with the
latest draw on the site. One bad page never stops an update; it is reported instead.

A result page that is only partly published (no winning shares table yet) is not
stored: it counts as not out yet, so the next run reads it again. Stored rows among the
newest draws that are incomplete (saved before this check existed) are read again too,
and a draw list whose latest draw number is far above anything plausible is refused
instead of crawled.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from . import constants as C
from .http import FetchError
from .models import NextToto, UpdateResult
from .parse import (
    ParseError,
    parse_draw_list,
    parse_toto_next_draw,
    parse_toto_prize_structure,
    parse_toto_result,
    sppl,
)
from .prize_rules import pct_text
from .store import empty_toto, no_winner_streak, normalise_toto
from .textfmt import DASH_CHARS, fmt_date, fmt_datetime, money, plural

log = logging.getLogger(__name__)
SG = ZoneInfo(C.SG_TZ_NAME)

MAX_LISTED_FAILURES = 10  # failures named one by one in the messages; the rest are summarised
RECHECK_NEWEST = 3  # stored draws among the newest this many that are incomplete are fetched again
LIST_SLACK = 10  # draw numbers allowed above "one draw a day" before a draw list counts as wrong
RAW_TEXT_LIMIT = 500  # characters of next draw page text kept for the notes

# The official lists of special draw types: key -> (name in messages, URL).
DRAW_TYPE_LISTS = {
    "cascade": ("cascade", C.TOTO_CASCADE_LIST_URL),
    "hongbao": ("Hongbao", C.TOTO_HONGBAO_LIST_URL),
    "special": ("special", C.TOTO_SPECIAL_LIST_URL),
}


def toto_result_url(draw_number: int) -> str:
    return C.TOTO_RESULT_URL.format(sppl=sppl(draw_number))


# small helpers (no dashes anywhere in user facing text)

_DASHES = re.compile("[" + re.escape(DASH_CHARS) + "]")


def _plain(text: Any) -> str:
    """Text for a user facing message: every dash becomes a space, spaces are squeezed."""
    return " ".join(_DASHES.sub(" ", str(text)).split())


def _on(day: date | None) -> str:
    """" on Thu 1 Oct 2026", or "" when the date is not known."""
    return f" on {fmt_date(day)}" if day is not None else ""


def _join(items: Iterable[Any]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _draw_span(numbers: list[int]) -> str:
    if len(numbers) <= 5:
        return _join(numbers)
    return f"{numbers[0]} to {numbers[-1]}"


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(SG)
    return now if now.tzinfo else now.replace(tzinfo=SG)


def _as_date(value: Any) -> date | None:
    """The date of a date, datetime (aware ones in Singapore time), Timestamp or ISO string."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return (ts.tz_convert(SG) if ts.tzinfo is not None else ts).date()


# draw types


def tag_draw_types(df: pd.DataFrame, cascade: Iterable[int] | None, hongbao: Iterable[int] | None,
                   special: Iterable[int] | None) -> pd.DataFrame:
    """Set ``draw_type`` on every TOTO draw.

    Order of evidence: the official lists (Hongbao, then cascade, then special); an
    existing Hongbao or special tag; an inferred cascade (the 3 draws just before it,
    with consecutive draw numbers, had no Group 1 winner and none of them was a cascade
    or Hongbao draw, so the jackpot had snowballed for the 4th time); an existing
    cascade tag; otherwise "normal".
    """
    out = df.copy()
    if out.empty:
        if "draw_type" not in out.columns:
            out["draw_type"] = pd.Series(dtype=object)
        return out
    if "draw_type" not in out.columns:
        out["draw_type"] = "normal"
    cascade_set = {int(n) for n in (cascade or ())}
    hongbao_set = {int(n) for n in (hongbao or ())}
    special_set = {int(n) for n in (special or ())}

    order = out.sort_values("draw_number").index
    numbers = [int(n) for n in out.loc[order, "draw_number"]]
    winners = pd.to_numeric(out.loc[order, "g1_winners"], errors="coerce").fillna(0).astype(int).tolist()
    existing = [str(t).strip().lower() if isinstance(t, str) else "normal" for t in out.loc[order, "draw_type"]]
    existing = [t if t in C.TOTO_DRAW_TYPES else "normal" for t in existing]

    lag = C.TOTO_SNOWBALL_LIMIT - 1
    types: list[str] = []
    for i, n in enumerate(numbers):
        if n in hongbao_set:
            t = "hongbao"
        elif n in cascade_set:
            t = "cascade"
        elif n in special_set:
            t = "special"
        elif existing[i] in ("hongbao", "special"):
            t = existing[i]
        elif i >= lag and all(
            numbers[j] == n - (i - j) and winners[j] == 0 and types[j] not in ("cascade", "hongbao")
            for j in range(i - lag, i)
        ):
            t = "cascade"
        elif existing[i] == "cascade":
            t = "cascade"
        else:
            t = "normal"
        types.append(t)
    out.loc[order, "draw_type"] = types
    return out


def _behind(toto_df: pd.DataFrame | None, next_day: date | None) -> bool:
    """True when a regular TOTO draw day falls between the newest stored draw and ``next_day``:
    a draw was held that is not stored, so the stored no winner streak may be out of date."""
    if toto_df is None or toto_df.empty or next_day is None:
        return False
    last = _as_date(toto_df.sort_values("draw_number").iloc[-1]["draw_date"])
    if last is None:
        return False
    days = (last + timedelta(days=i) for i in range(1, (next_day - last).days))
    return any(d.weekday() in C.TOTO_WEEKDAYS for d in days)


def _predicted_draw_type(toto_df: pd.DataFrame | None) -> str:
    """"cascade" when the next draw will be the 4th in a row with a snowballed jackpot."""
    if toto_df is None or toto_df.empty:
        return "normal"
    streak = no_winner_streak(toto_df, reset_on_cascade=True)
    return "cascade" if streak == C.TOTO_SNOWBALL_LIMIT - 1 else "normal"


# draw list and latest result


def _read_draw_list(fetcher) -> list[tuple[int, date | None]]:
    """Draw list newest first. Raises FetchError if it cannot be fetched or read."""
    url = C.TOTO_DRAW_LIST_URL
    html = fetcher.get(url)  # FetchError propagates: the runner handles a site that is down
    try:
        draws = parse_draw_list(html)
    except Exception as exc:  # parse_draw_list is lenient, this is a safety net
        raise FetchError("the TOTO draw list could not be read", url=url, detail=repr(exc)) from exc
    if not draws:
        raise FetchError("the TOTO draw list had no draws in it", url=url)
    return draws


def latest_on_site(fetcher, *, strict: bool = False) -> tuple[int | None, date | None]:
    """Latest TOTO draw number and date on the site; (None, None) if unknown.

    With ``strict`` a network failure (FetchError) is raised instead of returning (None, None),
    so the scheduler can tell "the site could not be reached" from "the result is not out yet".
    """
    try:
        draws = parse_draw_list(fetcher.get(C.TOTO_DRAW_LIST_URL))
    except FetchError as exc:
        if strict:
            raise
        log.warning("Could not reach the TOTO draw list: %s", exc)
        return None, None
    except Exception as exc:
        log.warning("Could not read the TOTO draw list: %s", exc)
        return None, None
    return draws[0] if draws else (None, None)


def incomplete_reason(row: Any) -> str | None:
    """Why a parsed result (or stored row) is not the full published result, or None.

    Group 7 always has thousands of winners, so 0 winners means the winning shares table
    was not on the page.
    """
    try:
        winners = int(row["g7_winners"])
    except (KeyError, TypeError, ValueError):
        winners = 0
    return None if winners > 0 else "the winning shares table is not on the page yet"


def latest_complete_date(fetcher, *, strict: bool = False) -> date | None:
    """Date of the latest TOTO draw on the site, but only once its result page is complete.

    Reads the draw list and the latest result page. Returns None while the page is missing,
    unreadable or only partly published (see ``incomplete_reason``), so a caller waiting for
    the result keeps checking instead of storing half a result. ``strict`` is passed on to
    ``latest_on_site``.
    """
    number, day = latest_on_site(fetcher, strict=strict)
    if number is None:
        return None
    try:
        parsed = parse_toto_result(fetcher.get(toto_result_url(number)))
    except Exception as exc:
        log.info("TOTO draw %s is listed but its result page could not be read yet: %s", number, exc)
        return None
    if int(parsed["draw_number"]) != int(number):
        log.info("TOTO draw %s is listed but its result page shows draw %s", number, parsed["draw_number"])
        return None
    reason = incomplete_reason(parsed)
    if reason:
        log.info("TOTO draw %s is listed but %s", number, reason)
        return None
    return day or _as_date(parsed.get("draw_date"))


# update


def _fetch_results(fetcher, numbers: list[int], fetched_at: str) -> tuple[list[dict], list[tuple[int, str]]]:
    """Fetch and parse TOTO result pages. Returns (parsed rows, [(draw number, plain reason)]).
    A page that is not complete yet is a failure (not out yet), not a row."""
    if not numbers:
        return [], []
    urls = {toto_result_url(n): n for n in numbers}
    try:
        pages = fetcher.get_many(list(urls))
    except Exception as exc:  # a fetcher should not raise here, but never lose the whole update
        log.exception("Fetching %d result pages failed", len(urls))
        return [], [(n, f"pages could not be fetched ({_plain(exc)})") for n in numbers]
    rows: list[dict] = []
    failures: list[tuple[int, str]] = []
    for url, n in urls.items():
        page = pages.get(url)
        if page is None:
            failures.append((n, "page was not fetched"))
            continue
        if isinstance(page, BaseException):
            failures.append((n, f"page could not be fetched, {_plain(page)}"))
            continue
        try:
            parsed = parse_toto_result(page)
        except ParseError as exc:
            failures.append((n, f"page could not be read, {_plain(exc)}"))
            continue
        except Exception as exc:
            log.exception("Unexpected error parsing TOTO draw %s", n)
            failures.append((n, f"page could not be read ({type(exc).__name__})"))
            continue
        if int(parsed["draw_number"]) != int(n):
            failures.append((n, f"page showed draw {parsed['draw_number']} instead"))
            continue
        reason = incomplete_reason(parsed)
        if reason:
            failures.append((n, reason))
            continue
        parsed["fetched_at"] = fetched_at
        rows.append(parsed)
    return rows, failures


def _check_latest(draws: list[tuple[int, date | None]], df: pd.DataFrame, now: datetime) -> None:
    """Refuse a draw list whose latest draw number cannot be right (one stray or garbled
    option would otherwise make the bot request thousands of pages that do not exist).

    There is at most one draw a day, so the latest number may be at most one per day above
    the newest stored draw (or the next entry of the list), plus ``LIST_SLACK``.
    """
    latest, latest_date = draws[0]
    if not df.empty:
        newest = df.sort_values("draw_number").iloc[-1]
        base, base_date = int(newest["draw_number"]), _as_date(newest["draw_date"])
        what = f"the newest stored draw is {base}"
    elif len(draws) > 1:
        base, base_date = draws[1]
        what = f"the next draw in the list is {base}"
    else:
        return
    if latest <= base:
        return
    end = latest_date or now.date()
    days = max((end - base_date).days, 0) if base_date is not None else 0
    if latest > base + days + LIST_SLACK:
        raise FetchError(f"the TOTO draw list looks wrong: it lists draw {latest} but {what}, so nothing "
                         "was fetched")


def _report_fetch(result: UpdateResult, rows: list[dict], failures: list[tuple[int, str]],
                  missing: list[int], recheck: list[int], skipped: list[int]) -> None:
    """Fill new_draws, failed_draws and repaired_draws and say what happened in plain words."""
    got = sorted(int(r["draw_number"]) for r in rows)
    still = sorted(n for n, _ in failures if n in recheck)
    failures = sorted((n, reason) for n, reason in failures if n not in recheck)
    result.new_draws = [n for n in got if n not in recheck]
    result.repaired_draws = [n for n in got if n in recheck]
    result.failed_draws = [n for n, _ in failures]

    for n in result.repaired_draws:
        result.messages.append(f"TOTO draw {n} was updated with its winning shares.")
    for n in still:
        result.messages.append(f"TOTO draw {n} does not have all its winning shares yet, it will be read again "
                               "on the next run.")
    if result.new_draws:
        result.messages.append(f"TOTO: {plural(len(result.new_draws), 'new draw')} added "
                               f"({_draw_span(result.new_draws)}).")
    elif not missing and not recheck:
        result.messages.append("TOTO: no missing draws, nothing new to fetch.")
    if skipped:
        verb = "it is" if len(skipped) == 1 else "they are"
        result.messages.append(f"TOTO: {plural(len(skipped), 'draw')} left out because {verb} on the skip list "
                               f"({_draw_span(skipped)}).")
    for n, reason in failures[:MAX_LISTED_FAILURES]:
        result.messages.append(f"TOTO draw {n} was not added: {reason}.")
    extra = result.failed_draws[MAX_LISTED_FAILURES:]
    if extra:
        result.messages.append(f"TOTO: {plural(len(extra), 'more draw')} could not be added ({_draw_span(extra)}).")
    if failures:
        log.warning("TOTO: %d draws failed: %s", len(failures), result.failed_draws[:50])


def _draw_type_lists(fetcher, result: UpdateResult) -> dict[str, set[int]]:
    """Fetch the cascade, Hongbao and special draw lists. A failure is a warning only."""
    try:
        pages = fetcher.get_many([url for _, url in DRAW_TYPE_LISTS.values()])
    except Exception as exc:
        log.warning("Draw type lists could not be fetched: %s", exc)
        pages = {}
    found: dict[str, set[int]] = {}
    for key, (name, url) in DRAW_TYPE_LISTS.items():
        page = pages.get(url)
        if isinstance(page, str):
            try:
                found[key] = {n for n, _ in parse_draw_list(page)}
                continue
            except Exception as exc:  # pragma: no cover, parse_draw_list is lenient
                reason = f"it could not be read ({type(exc).__name__})"
        elif isinstance(page, BaseException):
            reason = _plain(page)
        else:
            reason = "it was not fetched"
        result.messages.append(f"TOTO: the {name} draw list could not be used ({reason}), "
                               f"so {name} draws were kept from earlier runs or inferred.")
    return found


def _verify(df: pd.DataFrame, result: UpdateResult, latest_date: date | None) -> None:
    """Set result.verified and latest_in_csv, and say plainly whether the data is current."""
    result.verified = False
    if df.empty:
        result.latest_in_csv = None
        result.messages.append("TOTO: there are no draws in the data yet.")
        return
    newest = df.sort_values("draw_number").iloc[-1]
    result.latest_in_csv = int(newest["draw_number"])
    newest_date = _as_date(newest["draw_date"])
    latest = result.latest_on_site
    if result.latest_in_csv != latest:
        result.messages.append(f"TOTO: the newest draw in the data is {result.latest_in_csv} but the site shows "
                               f"{latest}, so the data is not fully up to date.")
    elif latest_date is not None and newest_date is not None and newest_date != latest_date:
        result.messages.append(f"TOTO: draw {latest} is dated {fmt_date(newest_date)} in the data but "
                               f"{fmt_date(latest_date)} on the site, please check it.")
    elif incomplete_reason(newest):
        result.messages.append(f"TOTO: draw {latest} is on the site but its winning shares are not complete yet.")
    else:
        result.verified = True
        result.messages.append(f"TOTO: the newest draw in the data ({latest}{_on(newest_date)}) matches the site.")


def update_toto(fetcher, df: pd.DataFrame, start_draw: int, skip: set[int] | None = None,
                now: datetime | None = None) -> tuple[pd.DataFrame, UpdateResult]:
    """Fetch every TOTO draw from ``start_draw`` to the latest that is not in ``df`` or ``skip``.

    The newest stored draws that are incomplete are read again (``repaired_draws``), every
    draw gets its draw type, and the newest row is checked against the site. Raises
    FetchError only when the draw list itself cannot be fetched or looks wrong. Does not save.
    """
    now = _now(now)
    df = empty_toto() if df is None else normalise_toto(df)
    result = UpdateResult()

    draws = _read_draw_list(fetcher)
    _check_latest(draws, df, now)
    latest, latest_date = draws[0]
    result.latest_on_site = latest
    result.messages.append(f"TOTO: the latest draw on the site is {latest}{_on(latest_date)}.")

    window = range(int(start_draw), latest + 1)
    have = set(int(n) for n in df["draw_number"])
    skip_set = {int(s) for s in (skip or ())}
    skipped = [n for n in window if n in skip_set and n not in have]
    missing = [n for n in window if n not in have and n not in skip_set]
    # Stored newest draws saved while their page was only partly published are read again.
    stored = df.set_index("draw_number")
    recheck = [n for n in window[-RECHECK_NEWEST:]
               if n in have and n not in skip_set and incomplete_reason(stored.loc[n])]

    rows, failures = _fetch_results(fetcher, missing + recheck, now.isoformat(timespec="seconds"))
    if rows:
        new = normalise_toto(pd.DataFrame(rows).assign(draw_type="normal"))  # tagged below
        df = new if df.empty else normalise_toto(pd.concat([df, new], ignore_index=True))
    _report_fetch(result, rows, failures, missing, recheck, skipped)

    lists = _draw_type_lists(fetcher, result)
    if not df.empty:
        df = normalise_toto(tag_draw_types(df, lists.get("cascade"), lists.get("hongbao"), lists.get("special")))
    _verify(df, result, latest_date)
    return df, result


# next draw


def fetch_next_draw(fetcher, toto_df: pd.DataFrame | None) -> NextToto | None:
    """The next TOTO draw from the next draw page, or None when the page failed.

    The draw type is the hint on the page if there is one, else a cascade predicted from the
    stored history (only when that history is not behind the next draw).
    """
    try:
        page = fetcher.get(C.TOTO_NEXT_DRAW_URL)
        info = parse_toto_next_draw(page)
    except FetchError as exc:
        log.warning("TOTO next draw page could not be fetched: %s", exc)
        return None
    except Exception:
        log.exception("TOTO next draw page could not be read")
        return None
    when = info["draw_datetime"]
    if when is None and info["jackpot_estimate"] is None:
        log.warning("TOTO next draw page had neither a date nor a jackpot")
        return None
    hint = info["draw_type_hint"]
    # Stored results that are behind say nothing about the next draw's type: "normal"
    # (not None, which would make buysignal predict again from the same frame).
    predicted = "normal" if _behind(toto_df, _as_date(when)) else _predicted_draw_type(toto_df)
    return NextToto(
        draw_datetime=when,
        jackpot_estimate=info["jackpot_estimate"],
        draw_type=hint or predicted,
        draw_type_hint=hint,
        raw_text=info["raw_text"][:RAW_TEXT_LIMIT],
    )


# site check


class _Unavailable(Exception):
    """A page the check needs could not be fetched (the message is the plain reason)."""


class _SiteCheck:
    """Collects PASS / FAIL lines for check_site. Every item runs under a guard."""

    def __init__(self, fetcher, out: Callable[[str], Any]) -> None:
        self.fetcher = fetcher
        self.out = out
        self.critical: list[tuple[str, bool]] = []
        self.optional: list[tuple[str, bool]] = []
        self.pages: dict[str, Any] = {}

    def run(self, name: str, fn: Callable[[], tuple[bool, str]], critical: bool = True) -> bool:
        try:
            ok, detail = fn()
        except _Unavailable as exc:
            ok, detail = False, str(exc)
        except Exception as exc:  # a parser bug must show up as a FAIL line, not a crash
            log.exception("Site check %s failed unexpectedly", name)
            ok, detail = False, f"unexpected error ({type(exc).__name__})"
        tag = "PASS" if ok else "FAIL"
        suffix = "" if critical else " (not critical)"
        self.out(f"{tag}  {name}{suffix}: {_plain(detail)}")
        (self.critical if critical else self.optional).append((name, ok))
        return ok

    def fetch(self, urls: list[str]) -> None:
        try:
            self.pages.update(self.fetcher.get_many(urls))
        except Exception as exc:
            for url in urls:
                self.pages[url] = FetchError(f"fetch failed ({type(exc).__name__})", url=url)

    def page(self, url: str) -> str:
        """The page text, or raise _Unavailable with a plain reason."""
        page = self.pages.get(url)
        if isinstance(page, str):
            return page
        raise _Unavailable(f"could not fetch the page, {page if page is not None else 'no response'}")

    def summary(self) -> str:
        failed = [name for name, ok in self.critical if not ok]
        optional_failed = [name for name, ok in self.optional if not ok]
        if not failed:
            text = f"Summary: all {len(self.critical)} critical checks passed"
        else:
            text = f"Summary: {len(failed)} of {len(self.critical)} critical checks failed ({_join(failed)})"
        if optional_failed:
            text += f", {len(optional_failed)} of {len(self.optional)} non critical checks failed"
        return _plain(text) + "."


def check_site(fetcher, out: Callable[[str], Any] = print) -> bool:
    """Fetch every TOTO page the bot uses, run its parser and print one PASS or FAIL line per item.

    Returns True only when every critical item passes. The online2 prize structure page is
    not critical (built in prize values are used when it cannot be read).
    """
    chk = _SiteCheck(fetcher, out)
    chk.fetch([C.TOTO_DRAW_LIST_URL, C.TOTO_NEXT_DRAW_URL, *(url for _, url in DRAW_TYPE_LISTS.values()),
               C.TOTO_PRIZE_RULES_URL])
    latest: list[tuple[int, date | None]] = []  # the newest draw of the draw list, once read

    def draw_list() -> tuple[bool, str]:
        draws = parse_draw_list(chk.page(C.TOTO_DRAW_LIST_URL))
        if not draws:
            return False, "the page has no draws in it, the layout may have changed"
        latest.append(draws[0])
        n, d = draws[0]
        return True, f"{len(draws)} draws listed, the latest is draw {n}{_on(d)}"

    def result_page() -> tuple[bool, str]:
        """The latest result page, read through its sppl link."""
        if not latest:
            return False, "skipped because the draw list could not be read"
        wanted, wanted_date = latest[0]
        url = toto_result_url(wanted)
        chk.fetch([url])
        try:
            html = chk.page(url)
        except _Unavailable as exc:
            return False, f"draw {wanted}: {exc}"
        try:
            r = parse_toto_result(html)
        except ParseError as exc:
            return False, f"draw {wanted} could not be read, {exc}"
        if r["draw_number"] != wanted:
            return False, f"asked for draw {wanted} but the page shows draw {r['draw_number']}"
        if wanted_date is not None and r["draw_date"] != wanted_date:
            return False, (f"draw {wanted} is dated {fmt_date(r['draw_date'])} on the page but "
                           f"{fmt_date(wanted_date)} on the draw list")
        nums = " ".join(str(r[f"n{i}"]) for i in range(1, 7))
        groups = sum(1 for g in range(1, 8) if r[f"g{g}_winners"] > 0 or not pd.isna(r[f"g{g}_share"]))
        return True, (f"draw {wanted}{_on(r['draw_date'])}, numbers {nums}, additional {r['additional']}, "
                      f"Group 1 prize {money(r['jackpot'])}, {groups} of 7 prize groups with winners")

    def next_draw() -> tuple[bool, str]:
        info = parse_toto_next_draw(chk.page(C.TOTO_NEXT_DRAW_URL))
        when, jackpot = info["draw_datetime"], info["jackpot_estimate"]
        hint = f", draw type {info['draw_type_hint']}" if info["draw_type_hint"] else ""
        return when is not None and jackpot is not None, (
            f"next draw {fmt_datetime(when) if when else 'date not found'}, estimated jackpot "
            f"{money(jackpot) if jackpot is not None else 'not found'}{hint}")

    def type_list(url: str) -> tuple[bool, str]:
        """An empty list is fine: there may be no such draws in the window."""
        html = chk.page(url)
        draws = parse_draw_list(html)
        if draws:
            return True, f"{len(draws)} draws listed, the latest is draw {draws[0][0]}"
        if "<select" in html.lower() or "<option" in html.lower():
            return True, "the list is empty"
        return False, "no draw list found in the page, the layout may have changed"

    def prize_page() -> tuple[bool, str]:
        try:
            rules = parse_toto_prize_structure(chk.page(C.TOTO_PRIZE_RULES_URL))
        except _Unavailable as exc:
            return False, f"{exc}, built in values will be used"
        if rules is None:
            return False, ("the prize figures are not in the page text (the page is probably drawn by "
                           "JavaScript), built in values will be used")
        parts = [f"Group {g} {pct_text(v)}" for g, v in sorted(rules["group_pool_pct"].items())]
        parts += [f"Group {g} {money(v)}" for g, v in sorted(rules["fixed_prizes"].items())]
        if rules["pool_share_of_sales"] is not None:
            parts.append(f"prize pool {pct_text(rules['pool_share_of_sales'])} of sales")
        return True, ", ".join(parts)

    chk.run("TOTO draw list", draw_list)
    chk.run("TOTO latest result page", result_page)
    chk.run("TOTO next draw page", next_draw)
    for name, url in DRAW_TYPE_LISTS.values():
        chk.run(f"TOTO {name} draw list", lambda url=url: type_list(url))
    chk.run("TOTO prize structure page", prize_page, critical=False)

    out(chk.summary())
    return all(ok for _, ok in chk.critical)
