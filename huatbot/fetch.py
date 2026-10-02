"""Bring toto.csv and fourd.csv up to date from the Singapore Pools site.

Every function takes a fetcher (``http.Fetcher`` or a test double with the same
``get`` / ``get_many`` interface), so nothing here talks to the network directly.
Updates are incremental: only draws missing from the frame are fetched, each page is
checked to be the draw that was asked for, and the newest row is compared with the
latest draw on the site. One bad page never stops an update; it is reported instead.

A result page that is only partly published (a TOTO page without its winning shares
table, a 4D page with blank numbers) is not stored: it counts as not out yet, so the next
run reads it again. Stored rows among the newest draws that are incomplete (saved before
this check existed) are read again too, and a draw list whose latest draw number is far
above anything plausible is refused instead of crawled.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from . import constants as C
from .http import FetchError
from .models import FOURD_NUMBER_COLUMNS, NextFourD, NextToto, UpdateResult
from .parse import (
    ParseError,
    parse_draw_list,
    parse_fourd_next_draw,
    parse_fourd_prize_structure,
    parse_fourd_result,
    parse_toto_next_draw,
    parse_toto_prize_structure,
    parse_toto_result,
    sppl,
)
from .store import empty_fourd, empty_toto, no_winner_streak, normalise_fourd, normalise_toto

log = logging.getLogger(__name__)
SG = ZoneInfo(C.SG_TZ_NAME)

MAX_LISTED_FAILURES = 10  # failures named one by one in the messages; the rest are summarised
RECHECK_NEWEST = 3  # stored draws among the newest this many that are incomplete are fetched again
LIST_SLACK = 10  # draw numbers allowed above "one draw a day" before a draw list counts as wrong
RAW_TEXT_LIMIT = 500  # characters of next draw page text kept for the notes

GAME_LABELS = {"toto": "TOTO", "4d": "4D"}


# URLs


def toto_result_url(draw_number: int) -> str:
    return C.TOTO_RESULT_URL.format(sppl=sppl(draw_number))


def fourd_result_url(draw_number: int) -> str:
    return C.FOURD_RESULT_URL.format(sppl=sppl(draw_number))


# small plain English helpers (no dashes anywhere in user facing text)


def _plain(text: Any) -> str:
    """Remove dashes and squeeze spaces, for text that ends up in user facing messages."""
    s = str(text)
    for dash in ("—", "–", "-"):
        s = s.replace(dash, " ")
    return " ".join(s.split())


def _fmt_date(d: date | datetime | pd.Timestamp | None) -> str:
    if d is None or (isinstance(d, pd.Timestamp) and pd.isna(d)):
        return "unknown date"
    return f"{d:%a} {d.day} {d:%b %Y}"


def _fmt_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    return f"{hour}.{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"


def _fmt_datetime(dt: datetime | None) -> str:
    if dt is None:
        return "unknown date"
    dt = dt.astimezone(SG) if dt.tzinfo else dt
    return f"{_fmt_date(dt)}, {_fmt_time(dt)}"


def _fmt_money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _join(items: list) -> str:
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
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts.date()


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


def _regular_draw_day_between(after: date, before: date, weekdays: Iterable[int]) -> bool:
    """True when a day strictly between ``after`` and ``before`` falls on one of ``weekdays``."""
    days = set(weekdays)
    return any((after + timedelta(days=i)).weekday() in days for i in range(1, (before - after).days))


def _behind(toto_df: pd.DataFrame | None, next_day: date | None) -> bool:
    """True when a regular TOTO draw day falls between the newest stored draw and ``next_day``:
    a draw was held that is not stored, so the stored no winner streak may be out of date."""
    if toto_df is None or toto_df.empty or next_day is None:
        return False
    last = _as_date(toto_df.sort_values("draw_number").iloc[-1]["draw_date"])
    return last is not None and _regular_draw_day_between(last, next_day, C.TOTO_WEEKDAYS)


def _predicted_draw_type(toto_df: pd.DataFrame | None) -> str:
    """"cascade" when the next draw will be the 4th in a row with a snowballed jackpot."""
    if toto_df is None or toto_df.empty:
        return "normal"
    streak = no_winner_streak(toto_df, reset_on_cascade=True)
    return "cascade" if streak == C.TOTO_SNOWBALL_LIMIT - 1 else "normal"


# draw lists


def _read_draw_list(fetcher, url: str, label: str) -> list[tuple[int, date | None]]:
    """Draw list newest first. Raises FetchError if it cannot be fetched or read."""
    html = fetcher.get(url)  # FetchError propagates: the runner handles a site that is down
    try:
        draws = parse_draw_list(html)
    except Exception as exc:  # parse_draw_list is lenient, this is a safety net
        raise FetchError(f"the {label} draw list could not be read", url=url, detail=repr(exc)) from exc
    if not draws:
        raise FetchError(f"the {label} draw list had no draws in it", url=url)
    return draws


def latest_on_site(fetcher, game: str, strict: bool = False) -> tuple[int | None, date | None]:
    """Latest draw number and date on the site for "toto" or "4d"; (None, None) if unknown.

    With ``strict`` a network failure (FetchError) is raised instead of returning (None, None),
    so the scheduler can tell "the site could not be reached" from "the result is not out yet".
    """
    key = str(game).lower()
    url = C.TOTO_DRAW_LIST_URL if key == "toto" else C.FOURD_DRAW_LIST_URL
    try:
        draws = parse_draw_list(fetcher.get(url))
    except FetchError:
        if strict:
            raise
        log.warning("Could not reach the %s draw list", GAME_LABELS.get(key, key))
        return None, None
    except Exception as exc:
        log.warning("Could not read the %s draw list: %s", GAME_LABELS.get(key, key), exc)
        return None, None
    return draws[0] if draws else (None, None)


# result pages

def incomplete_reason(game: str, row: Any) -> str | None:
    """Why a parsed result (or stored row) is not the full published result, or None.

    TOTO: Group 7 always has thousands of winners, so 0 winners means the winning shares
    table was not on the page. 4D: every one of the 23 winning numbers must be there.
    """
    if game == "toto":
        try:
            winners = int(row["g7_winners"])
        except (KeyError, TypeError, ValueError):
            winners = 0
        return None if winners > 0 else "the winning shares table is not on the page yet"
    for col in FOURD_NUMBER_COLUMNS:
        try:
            value = row[col]
        except KeyError:
            value = None
        if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
            return "not all 23 winning numbers are on the page yet"
    return None


def latest_complete_date(fetcher, game: str, strict: bool = False) -> date | None:
    """Date of the latest draw on the site, but only once its result page is complete.

    Reads the draw list and the latest result page. Returns None while the page is missing,
    unreadable or only partly published (see ``incomplete_reason``), so a caller waiting for
    the result keeps checking instead of storing half a result.
    """
    key = str(game).lower()
    label = GAME_LABELS.get(key, key)
    number, day = latest_on_site(fetcher, key, strict=strict)  # strict: raises if unreachable
    if number is None:
        return None
    is_toto = key == "toto"
    try:
        parsed = (parse_toto_result if is_toto else parse_fourd_result)(
            fetcher.get((toto_result_url if is_toto else fourd_result_url)(number)))
    except Exception as exc:
        log.info("%s draw %s is listed but its result page could not be read yet: %s", label, number, exc)
        return None
    if int(parsed["draw_number"]) != int(number):
        log.info("%s draw %s is listed but its result page shows draw %s", label, number, parsed["draw_number"])
        return None
    reason = incomplete_reason(key, parsed)
    if reason:
        log.info("%s draw %s is listed but %s", label, number, reason)
        return None
    return day or _as_date(parsed.get("draw_date"))


def _fetch_results(fetcher, numbers: list[int], url_fn: Callable[[int], str],
                   parse_fn: Callable[[str], dict], fetched_at: str,
                   incomplete: Callable[[dict], str | None] | None = None,
                   ) -> tuple[list[dict], list[tuple[int, str]]]:
    """Fetch and parse result pages. Returns (parsed rows, [(draw number, plain reason)]).
    A page ``incomplete`` finds a reason against is a failure (not out yet), not a row."""
    if not numbers:
        return [], []
    urls = {url_fn(n): n for n in numbers}
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
            parsed = parse_fn(page)
        except ParseError as exc:
            failures.append((n, f"page could not be read, {_plain(exc)}"))
            continue
        except Exception as exc:
            log.exception("Unexpected error parsing draw %s", n)
            failures.append((n, f"page could not be read ({type(exc).__name__})"))
            continue
        if int(parsed["draw_number"]) != int(n):
            failures.append((n, f"page showed draw {parsed['draw_number']} instead"))
            continue
        reason = incomplete(parsed) if incomplete is not None else None
        if reason:
            failures.append((n, reason))
            continue
        parsed["fetched_at"] = fetched_at
        rows.append(parsed)
    return rows, failures


def _check_latest(label: str, draws: list[tuple[int, date | None]], df: pd.DataFrame, now: datetime) -> None:
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
        raise FetchError(f"the {label} draw list looks wrong: it lists draw {latest} but {what}, so nothing "
                         "was fetched")


def _update(game: str, fetcher, df: pd.DataFrame, wanted: Callable[[int], range], skip,
            now: datetime | None) -> tuple[pd.DataFrame, UpdateResult, list[tuple[int, date | None]]]:
    """Shared incremental update for both games (draw types are handled by update_toto)."""
    label = GAME_LABELS[game]
    is_toto = game == "toto"
    normalise = normalise_toto if is_toto else normalise_fourd
    now = _now(now)
    df = normalise(df) if df is not None else (empty_toto() if is_toto else empty_fourd())
    result = UpdateResult(game=game)

    draws = _read_draw_list(fetcher, C.TOTO_DRAW_LIST_URL if is_toto else C.FOURD_DRAW_LIST_URL, label)
    latest, latest_date = draws[0]
    _check_latest(label, draws, df, now)
    result.latest_on_site = latest
    when = f" on {_fmt_date(latest_date)}" if latest_date else ""
    result.messages.append(f"{label}: the latest draw on the site is {latest}{when}.")

    have = set(int(n) for n in df["draw_number"])
    skip_set = {int(s) for s in (skip or ())}
    window = list(wanted(latest))
    skipped = [n for n in window if n in skip_set and n not in have]
    missing = [n for n in window if n not in have and n not in skip_set]
    # Stored newest draws saved while their page was only partly published are read again.
    stored = df.set_index("draw_number") if have else None
    recheck = [n for n in window[-RECHECK_NEWEST:]
               if n in have and n not in skip_set and incomplete_reason(game, stored.loc[n])]

    rows, failures = _fetch_results(
        fetcher, missing + recheck,
        toto_result_url if is_toto else fourd_result_url,
        parse_toto_result if is_toto else parse_fourd_result,
        now.isoformat(timespec="seconds"),
        lambda parsed: incomplete_reason(game, parsed),
    )
    if rows:
        new = pd.DataFrame(rows)
        if is_toto:
            new["draw_type"] = "normal"  # tagged afterwards
        new = normalise(new)
        df = new if df.empty else normalise(pd.concat([df, new], ignore_index=True))

    repaired = sorted(int(r["draw_number"]) for r in rows if int(r["draw_number"]) in recheck)
    still = sorted(n for n, _ in failures if n in recheck)
    failures = [(n, reason) for n, reason in failures if n not in recheck]
    result.new_draws = sorted(int(r["draw_number"]) for r in rows if int(r["draw_number"]) not in recheck)
    result.failed_draws = sorted(n for n, _ in failures)
    result.repaired_draws = list(repaired)
    what = "winning shares" if is_toto else "winning numbers"
    for n in repaired:
        result.messages.append(f"{label} draw {n} was updated with its {what}.")
    for n in still:
        result.messages.append(f"{label} draw {n} does not have all its {what} yet, it will be read again on "
                               "the next run.")

    if result.new_draws:
        count = len(result.new_draws)
        result.messages.append(
            f"{label}: {count} new draw{'s' if count != 1 else ''} added ({_draw_span(result.new_draws)})."
        )
    elif not missing and not recheck:
        result.messages.append(f"{label}: no missing draws, nothing new to fetch.")
    if skipped:
        result.messages.append(
            f"{label}: {len(skipped)} draw{'s' if len(skipped) != 1 else ''} left out because "
            f"{'they are' if len(skipped) != 1 else 'it is'} on the skip list ({_draw_span(sorted(skipped))})."
        )
    for n, reason in sorted(failures)[:MAX_LISTED_FAILURES]:
        result.messages.append(f"{label} draw {n} was not added: {reason}.")
    if len(failures) > MAX_LISTED_FAILURES:
        extra = sorted(n for n, _ in failures)[MAX_LISTED_FAILURES:]
        result.messages.append(f"{label}: {len(extra)} more draws could not be added ({_draw_span(extra)}).")
    if failures:
        log.warning("%s: %d draws failed: %s", label, len(failures), [n for n, _ in failures][:50])
    return df, result, draws


def _verify(label: str, df: pd.DataFrame, result: UpdateResult, latest_date: date | None) -> None:
    """Set result.verified and latest_in_csv, and say plainly whether the data is current."""
    if df.empty:
        result.latest_in_csv = None
        result.verified = False
        result.messages.append(f"{label}: there are no draws in the data yet.")
        return
    newest = df.sort_values("draw_number").iloc[-1]
    result.latest_in_csv = int(newest["draw_number"])
    newest_date = _as_date(newest["draw_date"])
    latest = result.latest_on_site
    if result.latest_in_csv != latest:
        result.verified = False
        result.messages.append(
            f"{label}: the newest draw in the data is {result.latest_in_csv} but the site shows {latest}, "
            "so the data is not fully up to date."
        )
    elif latest_date is not None and newest_date is not None and newest_date != latest_date:
        result.verified = False
        result.messages.append(
            f"{label}: draw {latest} is dated {_fmt_date(newest_date)} in the data but "
            f"{_fmt_date(latest_date)} on the site, please check it."
        )
    elif incomplete_reason(result.game, newest):
        result.verified = False
        what = "winning shares" if result.game == "toto" else "winning numbers"
        result.messages.append(f"{label}: draw {latest} is on the site but its {what} are not complete yet.")
    else:
        result.verified = True
        result.messages.append(
            f"{label}: the newest draw in the data ({latest} on {_fmt_date(newest_date)}) matches the site."
        )


def _draw_type_lists(fetcher, result: UpdateResult) -> dict[str, set[int]]:
    """Fetch the cascade, Hongbao and special draw lists. A failure is a warning only."""
    urls = {
        "cascade": C.TOTO_CASCADE_LIST_URL,
        "hongbao": C.TOTO_HONGBAO_LIST_URL,
        "special": C.TOTO_SPECIAL_LIST_URL,
    }
    names = {"cascade": "cascade", "hongbao": "Hongbao", "special": "special"}
    try:
        pages = fetcher.get_many(list(urls.values()))
    except Exception as exc:
        log.warning("Draw type lists could not be fetched: %s", exc)
        pages = {}
    found: dict[str, set[int]] = {}
    for key, url in urls.items():
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
        result.messages.append(
            f"TOTO: the {names[key]} draw list could not be used ({reason}), "
            f"so {names[key]} draws were kept from earlier runs or inferred."
        )
    return found


def update_toto(fetcher, df: pd.DataFrame, start_draw: int, skip: set[int] | None = None,
                now: datetime | None = None) -> tuple[pd.DataFrame, UpdateResult]:
    """Fetch every TOTO draw from ``start_draw`` to the latest that is not in ``df`` or ``skip``.

    Raises FetchError only when the draw list itself cannot be fetched. Does not save.
    """
    start = int(start_draw)
    df, result, draws = _update("toto", fetcher, df, lambda latest: range(start, latest + 1), skip, now)
    lists = _draw_type_lists(fetcher, result)
    if not df.empty:
        df = normalise_toto(tag_draw_types(df, lists.get("cascade"), lists.get("hongbao"), lists.get("special")))
    _verify("TOTO", df, result, draws[0][1])
    return df, result


def update_fourd(fetcher, df: pd.DataFrame, history_draws: int = 1000, skip: set[int] | None = None,
                 now: datetime | None = None) -> tuple[pd.DataFrame, UpdateResult]:
    """Fill the latest ``history_draws`` 4D draws (existing rows are kept). Does not save."""
    count = max(1, int(history_draws))
    df, result, draws = _update(
        "4d", fetcher, df, lambda latest: range(max(1, latest - count + 1), latest + 1), skip, now
    )
    _verify("4D", df, result, draws[0][1])
    return df, result


# next draws


def fetch_next_draws(fetcher, toto_df: pd.DataFrame | None) -> tuple[NextToto | None, NextFourD | None]:
    """Next TOTO and 4D draw from the next draw pages; None for a game whose page failed."""
    urls = [C.TOTO_NEXT_DRAW_URL, C.FOURD_NEXT_DRAW_URL]
    try:
        pages = fetcher.get_many(urls)
    except Exception as exc:
        log.warning("Next draw pages could not be fetched: %s", exc)
        pages = {}

    next_toto = None
    page = pages.get(C.TOTO_NEXT_DRAW_URL)
    if isinstance(page, str):
        try:
            info = parse_toto_next_draw(page)
            if info["draw_datetime"] is not None or info["jackpot_estimate"] is not None:
                hint = info["draw_type_hint"]
                when = info["draw_datetime"]
                next_day = _as_date(when.astimezone(SG) if isinstance(when, datetime) and when.tzinfo else when)
                # Stored results that are behind say nothing about the next draw's type: "normal"
                # (not None, which would make buysignal predict again from the same frame).
                predicted = "normal" if _behind(toto_df, next_day) else _predicted_draw_type(toto_df)
                next_toto = NextToto(
                    draw_datetime=info["draw_datetime"],
                    jackpot_estimate=info["jackpot_estimate"],
                    draw_type=hint or predicted,
                    draw_type_hint=hint,
                    raw_text=info["raw_text"][:RAW_TEXT_LIMIT],
                )
            else:
                log.warning("TOTO next draw page had neither a date nor a jackpot")
        except Exception:
            log.exception("TOTO next draw page could not be read")
    else:
        log.warning("TOTO next draw page could not be fetched: %s", page)

    next_fourd = None
    page = pages.get(C.FOURD_NEXT_DRAW_URL)
    if isinstance(page, str):
        try:
            info = parse_fourd_next_draw(page)
            if info["draw_datetime"] is not None:
                next_fourd = NextFourD(draw_datetime=info["draw_datetime"], raw_text=info["raw_text"][:RAW_TEXT_LIMIT])
            else:
                log.warning("4D next draw page had no draw date")
        except Exception:
            log.exception("4D next draw page could not be read")
    else:
        log.warning("4D next draw page could not be fetched: %s", page)
    return next_toto, next_fourd


# site check


class _SiteCheck:
    """Collects PASS / FAIL lines for check_site. Every item runs under a guard."""

    def __init__(self, fetcher, out: Callable[[str], Any]) -> None:
        self.fetcher = fetcher
        self.out = out
        self.critical: list[tuple[str, bool]] = []
        self.optional: list[tuple[str, bool]] = []
        self.pages: dict[str, Any] = {}

    def report(self, name: str, ok: bool, detail: str, critical: bool = True) -> None:
        tag = "PASS" if ok else "FAIL"
        suffix = "" if critical else " (not critical)"
        self.out(f"{tag}  {name}{suffix}: {_plain(detail)}")
        (self.critical if critical else self.optional).append((name, ok))

    def run(self, name: str, fn: Callable[[], tuple[bool, str]], critical: bool = True) -> bool:
        try:
            ok, detail = fn()
        except Exception as exc:  # a parser bug must show up as a FAIL line, not a crash
            log.exception("Site check %s failed unexpectedly", name)
            ok, detail = False, f"unexpected error ({type(exc).__name__})"
        self.report(name, ok, detail, critical)
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


class _Unavailable(Exception):
    pass


def _guarded(fn: Callable[[], tuple[bool, str]]) -> Callable[[], tuple[bool, str]]:
    """Turn a missing page into a FAIL detail."""
    def inner() -> tuple[bool, str]:
        try:
            return fn()
        except _Unavailable as exc:
            return False, str(exc)
    return inner


def check_site(fetcher, out: Callable[[str], Any] = print) -> bool:
    """Fetch every page the bot uses, run its parser and print one PASS or FAIL line per item.

    Returns True only when every critical item passes. The online2 prize structure pages
    are not critical (built in prize values are used when they cannot be read).
    """
    chk = _SiteCheck(fetcher, out)
    chk.fetch([
        C.TOTO_DRAW_LIST_URL, C.FOURD_DRAW_LIST_URL, C.TOTO_NEXT_DRAW_URL, C.FOURD_NEXT_DRAW_URL,
        C.TOTO_CASCADE_LIST_URL, C.TOTO_HONGBAO_LIST_URL, C.TOTO_SPECIAL_LIST_URL,
        C.TOTO_PRIZE_RULES_URL, C.FOURD_PRIZE_RULES_URL,
    ])

    # Draw lists
    latest: dict[str, tuple[int, date | None]] = {}
    for game, url in (("toto", C.TOTO_DRAW_LIST_URL), ("4d", C.FOURD_DRAW_LIST_URL)):
        def draw_list(game=game, url=url) -> tuple[bool, str]:
            draws = parse_draw_list(chk.page(url))
            if not draws:
                return False, "the page has no draws in it, the layout may have changed"
            latest[game] = draws[0]
            n, d = draws[0]
            return True, f"{len(draws)} draws listed, the latest is draw {n} on {_fmt_date(d)}"
        chk.run(f"{GAME_LABELS[game]} draw list", _guarded(draw_list))

    # Latest result pages, read through the sppl link
    result_urls = {game: (toto_result_url if game == "toto" else fourd_result_url)(n)
                   for game, (n, _) in latest.items()}
    if result_urls:
        chk.fetch(list(result_urls.values()))

    def result_page(game: str) -> tuple[bool, str]:
        if game not in result_urls:
            return False, "skipped because the draw list could not be read"
        wanted, wanted_date = latest[game]
        try:
            html = chk.page(result_urls[game])
        except _Unavailable as exc:
            return False, f"draw {wanted}: {exc}"
        try:
            r = (parse_toto_result if game == "toto" else parse_fourd_result)(html)
        except ParseError as exc:
            return False, f"draw {wanted} could not be read, {exc}"
        if r["draw_number"] != wanted:
            return False, f"asked for draw {wanted} but the page shows draw {r['draw_number']}"
        if wanted_date is not None and r["draw_date"] != wanted_date:
            return False, (f"draw {wanted} is dated {_fmt_date(r['draw_date'])} on the page but "
                           f"{_fmt_date(wanted_date)} on the draw list")
        if game == "toto":
            nums = " ".join(str(r[f"n{i}"]) for i in range(1, 7))
            groups = sum(1 for g in range(1, 8) if r[f"g{g}_winners"] > 0 or not pd.isna(r[f"g{g}_share"]))
            return True, (f"draw {wanted} on {_fmt_date(r['draw_date'])}, numbers {nums}, additional "
                          f"{r['additional']}, Group 1 prize {_fmt_money(r['jackpot'])}, "
                          f"{groups} of 7 prize groups with winners")
        starters = sum(1 for i in range(1, 11) if r[f"starter_{i}"])
        consolations = sum(1 for i in range(1, 11) if r[f"consolation_{i}"])
        return True, (f"draw {wanted} on {_fmt_date(r['draw_date'])}, 1st {r['first'] or 'blank'}, "
                      f"2nd {r['second'] or 'blank'}, 3rd {r['third'] or 'blank'}, "
                      f"{starters} starter and {consolations} consolation numbers")

    chk.run("TOTO latest result page", lambda: result_page("toto"))
    chk.run("4D latest result page", lambda: result_page("4d"))

    # Next draw pages
    def toto_next() -> tuple[bool, str]:
        info = parse_toto_next_draw(chk.page(C.TOTO_NEXT_DRAW_URL))
        ok = info["draw_datetime"] is not None and info["jackpot_estimate"] is not None
        hint = f", draw type {info['draw_type_hint']}" if info["draw_type_hint"] else ""
        return ok, (f"next draw {_fmt_datetime(info['draw_datetime'])}, estimated jackpot "
                    f"{_fmt_money(info['jackpot_estimate'])}{hint}")

    def fourd_next() -> tuple[bool, str]:
        info = parse_fourd_next_draw(chk.page(C.FOURD_NEXT_DRAW_URL))
        return info["draw_datetime"] is not None, f"next draw {_fmt_datetime(info['draw_datetime'])}"

    chk.run("TOTO next draw page", _guarded(toto_next))
    chk.run("4D next draw page", _guarded(fourd_next))

    # Draw type lists (an empty list is fine: there may be no such draws in the window)
    for name, url in (("TOTO cascade draw list", C.TOTO_CASCADE_LIST_URL),
                      ("TOTO Hongbao draw list", C.TOTO_HONGBAO_LIST_URL),
                      ("TOTO special draw list", C.TOTO_SPECIAL_LIST_URL)):
        def type_list(url=url) -> tuple[bool, str]:
            html = chk.page(url)
            draws = parse_draw_list(html)
            if draws:
                return True, f"{len(draws)} draws listed, the latest is draw {draws[0][0]}"
            if "<select" in html.lower() or "<option" in html.lower():
                return True, "the list is empty"
            return False, "no draw list found in the page, the layout may have changed"
        chk.run(name, _guarded(type_list))

    # Prize structure pages (not critical)
    def toto_prizes() -> tuple[bool, str]:
        try:
            rules = parse_toto_prize_structure(chk.page(C.TOTO_PRIZE_RULES_URL))
        except _Unavailable as exc:
            return False, f"{exc}, built in values will be used"
        if rules is None:
            return False, ("the prize figures are not in the page text (the page is probably drawn by "
                           "JavaScript), built in values will be used")
        pct = rules["group_pool_pct"]
        parts = [f"Group {g} {_fmt_pct(pct[g])}" for g in sorted(pct)]
        parts += [f"Group {g} {_fmt_money(v)}" for g, v in sorted(rules["fixed_prizes"].items())]
        if rules["pool_share_of_sales"] is not None:
            parts.append(f"prize pool {_fmt_pct(rules['pool_share_of_sales'])} of sales")
        return True, ", ".join(parts)

    def fourd_prizes() -> tuple[bool, str]:
        try:
            rules = parse_fourd_prize_structure(chk.page(C.FOURD_PRIZE_RULES_URL))
        except _Unavailable as exc:
            return False, f"{exc}, built in values will be used"
        if rules is None:
            return False, ("the prize table is not in the page text (the page is probably drawn by "
                           "JavaScript), built in values will be used")
        big = ", ".join(_fmt_money(v) for v in rules["big"].values())
        small = ", ".join(_fmt_money(v) for v in rules["small"].values())
        ibet = " and iBet tables" if rules.get("ibet") else ""
        return True, f"Big per $1 {big}; Small per $1 {small}{ibet}"

    chk.run("TOTO prize structure page", toto_prizes, critical=False)
    chk.run("4D prize structure page", fourd_prizes, critical=False)

    failed = [name for name, ok in chk.critical if not ok]
    optional_failed = [name for name, ok in chk.optional if not ok]
    if not failed:
        summary = f"Summary: all {len(chk.critical)} critical checks passed"
    else:
        summary = f"Summary: {len(failed)} of {len(chk.critical)} critical checks failed ({_join(failed)})"
    if optional_failed:
        summary += f", {len(optional_failed)} of {len(chk.optional)} non critical checks failed"
    out(_plain(summary) + ".")
    return not failed
