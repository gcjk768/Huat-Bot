"""One Huat Bot run: fetch the TOTO results, work out the next draw and the next big prize, check
the tickets, write the vault, post.

The Obsidian vault on the NAS (``vault.Vault``) is the single storage. Everything the bot keeps
lives in it: ``Data/toto.csv``, ``Data/ledger.csv``, ``Data/state.json`` and
``Data/prize_rules.json``, plus the notes (dashboard, ledger, reports, one note per draw) and the
monthly activity log, which gets a row for every meaningful action (RUN, SETTINGS, FETCH,
NEW DRAW, TICKETS, LEDGER, SIGNAL, NOTE, POST, DRY RUN, ERROR; the scheduler adds SCHEDULE and
WAIT).

Message 2 carries a just for fun lucky set (``lucky``), but every draw is independent, so no pattern in past results makes a
set of numbers more likely. It reports the result, what the next draws are likely to do and where
the next big prize is (``outlook``), and how much each $1 returns on average (``buysignal``).

``run`` follows six steps:

1. vault layout (Settings.md and Tickets.md starter notes), settings, RUN row
2. prize rules (cached), incremental CSV update with the skip list from state.json
   (a site that cannot be reached is a warning: stored data is used; with nothing stored at
   all the run stops, logs an ERROR and sends a short Telegram notice)
3. next draw, buy signal, jackpot outlook and jackpot history (``build_context``)
4. tickets: Tickets.md is read, the ledger synced, settled and saved
5. optional commentary, the full report, every note, state.json
6. Telegram: the 2 messages are posted unless this is a dry run (no token configured also
   means a dry run)

state.json keys written here. Entries are kept per game (``{"toto": ...}``), the shape the
scheduler reads; what an older version kept for 4D is dropped when a run reads the state.

``next_draws``      {"toto": {"draw_datetime", "jackpot_estimate", "draw_type", ...},
                     "checked_at"} (also read by the scheduler)
``upcoming_draws``  every announced draw date, kept by ``Vault.save_state`` through
                    ``scheduler.remember_upcoming`` so a special draw day survives a restart
``last_posted``     {"toto": 4123}: newest draw already posted. A scheduled run
                    (``force_post=False``) posts only when there is a newer draw.
``posting``         {"draws": {"toto": 4123}, "sent": 1, "total": 2}: a set of messages that
                    was only partly posted; the next post of the same draw resumes after
                    message ``sent``. Removed once all of them went out.
``skip``            {"toto": [...]}: draw numbers never fetched again. A draw whose page fails
                    in 3 runs in a row is added automatically (never one of the newest 10
                    draws, and never for fetch trouble such as timeouts, 403, 429 or 5xx, only
                    for a page that is gone or cannot be read); the list can be edited by hand.
``fetch_failures``  failed runs per draw, feeding ``skip``
``unposted_settled`` ticket ids checked by a run that posts, saved before ledger.csv, until a
                    message 1 listing them went out: a failed post, or a run cut off after the
                    ledger was saved, still reports them in the next message 1
``last_logged``     short hash of the last SIGNAL row, so an unchanged signal is not logged
                    again on every run, and of the last "Next draw" FETCH row with its time, so
                    the scheduler's refresh and the run straight after it do not both log it
``last_run``        when, ok, new draws, dry run or demo

``demo=True`` uses synthetic history from ``huatbot.synth`` (no network, never posts) and writes
it into the given vault, so the whole pipeline can be tried without the site.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from . import (
    buysignal,
    commentary,
    listener,
    notes,
    outlook,
    prize_rules,
    report,
    sales,
    store,
    synth,
    telegram,
    tickets,
)
from . import constants as C
from . import fetch as site
from .http import Fetcher, FetchError
from .models import Context, NextToto, RunResult, Settings, UpdateResult
from .report import dollars
from .scheduler import UPCOMING_KEY
from .settings import SETTINGS_TEMPLATE, load_settings
from .textfmt import fmt_date, fmt_datetime, money, per_dollar, plural, remove_dashes, toto_nums
from .vault import SG, Vault

log = logging.getLogger(__name__)

GAME = "toto"  # state.json key of every per game entry
LABEL = "TOTO"
OLD_GAMES = ("4d", "4D", "fourd")  # state.json entries of earlier versions, dropped on read

TEMPLATES = {"Settings.md": SETTINGS_TEMPLATE, "Tickets.md": tickets.TICKETS_TEMPLATE}
TICKETS_NOTE = "Tickets.md"

# Activity log events written by the runner.
EV_RUN = "RUN"
EV_SETTINGS = "SETTINGS"
EV_FETCH = "FETCH"
EV_NEW_DRAW = "NEW DRAW"
EV_TICKETS = "TICKETS"
EV_LEDGER = "LEDGER"
EV_SIGNAL = "SIGNAL"
EV_NOTE = "NOTE"
EV_POST = "POST"
EV_DRY_RUN = "DRY RUN"
EV_ERROR = "ERROR"

NEW_DRAW_LOG_LIMIT = 10  # NEW DRAW rows; a big first fetch gets one summary row for the rest
LEDGER_LOG_LIMIT = 20  # settled tickets logged one by one
SKIP_AFTER_FAILED_RUNS = 3  # a draw page failing in this many runs goes on the skip list
SKIP_PROTECT_NEWEST = 10  # ... unless it is one of the newest draws on the site
PRIZE_RULES_MAX_AGE_DAYS = 7
NEXT_DRAW_LOG_KEY = "FETCH next draws"  # state["last_logged"] entry of the "Next draw" row
NEXT_DRAW_REPEAT_WINDOW = timedelta(minutes=5)  # the same row this soon after is not logged again
SIGNAL_LOG_KEY = "SIGNAL"

# Demo history: enough draws for the sales estimates and the jackpot history.
DEMO_TOTO_DRAWS = 600
# Anchor so synthetic draw numbers look like real ones (TOTO 4123 on Thu 1 Oct 2026).
DEMO_TOTO_ANCHOR = (date(2026, 10, 1), 4123)
DEMO_WARNING = ("Demo run: every draw here is synthetic, made up by huatbot.synth. These are not real "
                "Singapore Pools results.")


# Small helpers


def _aware(now: datetime | None) -> datetime:
    """Singapore time; None is the current time, a naive datetime is taken as Singapore time."""
    if now is None:
        return datetime.now(SG)
    return now.replace(tzinfo=SG) if now.tzinfo is None else now.astimezone(SG)


def _and(items: Iterable[str]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _span(numbers: list[int]) -> str:
    """"4122 and 4123" or "3974 to 4123" (draw numbers, ascending)."""
    numbers = sorted(numbers)
    if len(numbers) <= 3:
        return _and(str(n) for n in numbers)
    return f"{numbers[0]} to {numbers[-1]}"


def _plain(text: Any) -> str:
    """One line of dash free text for warnings, log rows and notices."""
    return " ".join(remove_dashes(str(text)).split())


def _error_text(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return _plain(text)[:300]


def _latest(df: pd.DataFrame | None) -> int | None:
    if df is None or df.empty:
        return None
    return int(df["draw_number"].max())


def _latest_date(df: pd.DataFrame | None) -> date | None:
    if df is None or df.empty:
        return None
    value = df.sort_values("draw_number")["draw_date"].iloc[-1]
    return None if pd.isna(value) else pd.Timestamp(value).date()


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt.replace(tzinfo=SG) if dt.tzinfo is None else dt.astimezone(SG)


def _link(rel: str) -> str:
    """[[wikilink]] of a vault note path."""
    return f"[[{rel.rsplit('/', 1)[-1].removesuffix('.md')}]]"


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def resolve_vault(vault: Vault | str | Path | None) -> Vault:
    """A Vault from an instance, a vault folder path (VAULT_FOLDER and DATA_DIR still apply) or
    None (VAULT_PATH from the environment)."""
    if isinstance(vault, Vault):
        return vault
    if vault is None:
        return Vault.from_env()
    return Vault.from_env({**os.environ, "VAULT_PATH": str(vault)})


class _Activity:
    """Writes activity log rows. ``when`` None stamps each row with the current time."""

    def __init__(self, vault: Vault, when: datetime | None, enabled: bool = True) -> None:
        self.vault = vault
        self.when = when
        self.enabled = enabled

    def __call__(self, event: str, message: str) -> None:
        if self.enabled:
            self.vault.log(event, message, when=self.when)


# Telegram notices


def notify(text: str, *, dry_run: bool = False, out: Callable[[str], Any] = print) -> bool:
    """Send a short plain text notice (no HTML) to the Telegram chat. Never raises.

    In a dry run, or with no token or chat id configured, the notice is printed instead.
    Returns True when it was sent.
    """
    text = _plain(text)
    token, chat_id = telegram.config_from_env()
    if dry_run or not token or not chat_id:
        out(f"Notice (not posted): {text}")
        return False
    try:
        telegram.send_message(token, chat_id, text, parse_mode=None)
        return True
    except Exception as exc:  # a notice must never crash the caller
        log.warning("Telegram notice could not be sent: %s", exc)
        return False


# State helpers


def load_state(vault: Vault) -> dict:
    """state.json without the entries an older version kept for 4D or for number suggestions."""
    state = vault.load_state()
    for key in ("next_draws", "skip", "fetch_failures", "last_posted", UPCOMING_KEY):
        entry = state.get(key)
        if isinstance(entry, dict):
            for old in OLD_GAMES:
                entry.pop(old, None)
    seen = state.get("last_logged")
    if isinstance(seen, dict):
        for key in [k for k in seen if str(k).startswith("SUGGEST")]:
            seen.pop(key)
    posting = state.get("posting")
    if isinstance(posting, dict) and isinstance(posting.get("draws"), dict) and set(posting["draws"]) - {GAME}:
        state.pop("posting")  # a half posted set of the old three messages is not resumed
    return state


def _skip_list(state: dict) -> set[int]:
    raw = (state.get("skip") or {}).get(GAME) or []
    out = set()
    for n in raw:
        try:
            out.add(int(n))
        except (TypeError, ValueError):
            log.warning("Ignoring %r in the TOTO skip list of state.json", n)
    return out


class _FailureRecorder:
    """Passes every call on to the real fetcher and remembers the URLs that failed only because
    of fetch trouble: no answer, timeouts, connection errors, 403, 408, 429 or 5xx. A page that
    is gone (HTTP 404 or 410), or one that came back but could not be read, is not recorded.
    The runner uses this so a site outage or rate limiting never puts draws on the skip list.
    """

    PERMANENT_STATUS = (404, 410)

    def __init__(self, inner) -> None:
        self.inner = inner
        self.transient: set[str] = set()

    def _note(self, url: str, page: Any) -> None:
        if page is None or (isinstance(page, BaseException)
                            and getattr(page, "status", None) not in self.PERMANENT_STATUS):
            self.transient.add(url)

    def get(self, url: str) -> str:
        try:
            return self.inner.get(url)
        except BaseException as exc:
            self._note(url, exc)
            raise

    def get_many(self, urls):
        urls = list(urls)
        try:
            pages = self.inner.get_many(urls)
        except BaseException:
            self.transient.update(urls)
            raise
        for url in urls:
            self._note(url, pages.get(url))
        return pages

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _track_failures(state: dict, result: UpdateResult, activity: _Activity,
                    transient: Iterable[int] = ()) -> list[int]:
    """Count the runs in a row in which each draw page failed; after SKIP_AFTER_FAILED_RUNS put the
    draw on the skip list (never one of the newest SKIP_PROTECT_NEWEST draws). A draw that did not
    fail in this run starts again from zero. A draw in ``transient`` failed only because of fetch
    trouble: its count is carried over unchanged and it never goes on the skip list for that run.
    Returns the draws put on the skip list in this run."""
    all_failures = state.get("fetch_failures") or {}
    previous = all_failures.get(GAME) or {}
    newest = int(result.latest_on_site or 0)
    skip = _skip_list(state)
    transient = {int(n) for n in transient}
    counts: dict[str, int] = {}
    added = []
    for n in result.failed_draws:
        if int(n) in transient:
            try:
                if str(n) in previous:
                    counts[str(n)] = int(previous[str(n)])
            except (TypeError, ValueError):
                pass
            continue
        try:
            count = int(previous.get(str(n), 0)) + 1
        except (TypeError, ValueError):
            count = 1
        if count >= SKIP_AFTER_FAILED_RUNS and n <= newest - SKIP_PROTECT_NEWEST:
            skip.add(int(n))
            added.append(int(n))
        else:
            counts[str(n)] = count
    if added:
        state.setdefault("skip", {})[GAME] = sorted(skip)
        activity(EV_FETCH, f"TOTO: {plural(len(added), 'draw')} ({_span(added)}) failed in "
                           f"{SKIP_AFTER_FAILED_RUNS} runs in a row and went on the skip list in state.json")
    if counts:
        all_failures[GAME] = counts
    else:
        all_failures.pop(GAME, None)
    if all_failures:
        state["fetch_failures"] = all_failures
    else:
        state.pop("fetch_failures", None)
    return added


# Next draw


def _next_toto_state(nt: NextToto) -> dict:
    return {
        "draw_datetime": nt.draw_datetime.isoformat() if nt.draw_datetime else None,
        "jackpot_estimate": nt.jackpot_estimate,
        "draw_type": nt.draw_type,
        "draw_type_hint": nt.draw_type_hint,
        "raw_text": (nt.raw_text or "")[:300],
    }


def _store_next_draw(state: dict, nt: NextToto | None, now: datetime) -> None:
    """Keep the next draw info in state (a page that failed keeps the old entry)."""
    if nt is None:
        return
    entry = dict(state.get("next_draws") or {})
    entry[GAME] = _next_toto_state(nt)
    entry["checked_at"] = now.isoformat(timespec="seconds")
    state["next_draws"] = entry


def next_draw_from_state(state: dict, toto: pd.DataFrame | None) -> NextToto | None:
    """The next draw remembered in state.json, unless it is not after the newest stored draw."""
    t = (state.get("next_draws") or {}).get(GAME)
    if not isinstance(t, dict):
        return None
    dt = _parse_dt(t.get("draw_datetime"))
    last = _latest_date(toto)
    if dt is None or (last is not None and dt.date() <= last):
        return None
    jackpot = t.get("jackpot_estimate")
    return NextToto(
        draw_datetime=dt,
        jackpot_estimate=float(jackpot) if isinstance(jackpot, (int, float)) else None,
        draw_type=str(t.get("draw_type") or "normal"),
        draw_type_hint=t.get("draw_type_hint"),
        raw_text=str(t.get("raw_text") or ""),
    )


def _next_draw_text(nt: NextToto | None) -> str:
    if nt is None:
        return "the next draw page could not be read"
    when = fmt_datetime(nt.draw_datetime) if nt.draw_datetime else "date not announced"
    jackpot = f", estimated jackpot {money(nt.jackpot_estimate)}" if nt.jackpot_estimate else ""
    kind = f", {report.draw_type_name(nt.draw_type)} draw" if nt.draw_type and nt.draw_type != "normal" else ""
    return f"TOTO {when}{jackpot}{kind}"


def _log_next_draw(log_fn: Callable[[str, str], Any], state: dict, nt: NextToto | None, now: datetime) -> None:
    """Log the "Next draw" FETCH row, unless the same row was logged less than
    NEXT_DRAW_REPEAT_WINDOW ago (the scheduler refreshes the next draw just before it runs, and
    the run reads it again)."""
    message = f"Next draw: {_next_draw_text(nt)}"
    digest = _digest(message)
    seen = state.get("last_logged") if isinstance(state.get("last_logged"), dict) else {}
    last_digest, _, last_at = str(seen.get(NEXT_DRAW_LOG_KEY) or "").partition(" ")
    last = _parse_dt(last_at)
    if last_digest == digest and last is not None and timedelta(0) <= now - last < NEXT_DRAW_REPEAT_WINDOW:
        return
    log_fn(EV_FETCH, message)
    state["last_logged"] = {**seen, NEXT_DRAW_LOG_KEY: f"{digest} {now.isoformat(timespec='seconds')}"}


def refresh_next_draws(vault: Vault | str | Path | None, fetcher, *, now: datetime | None = None,
                       log_activity: bool = True) -> dict:
    """Read the next draw page, store it in state.json and return the state.

    Used by the scheduler before each cycle (special draws on unusual days come from here).
    """
    vault = resolve_vault(vault)
    now = _aware(now)
    vault.ensure_layout()
    state = load_state(vault)
    nt = site.fetch_next_draw(fetcher, store.load_toto(vault.toto_csv))
    _store_next_draw(state, nt, now)
    if log_activity:
        _log_next_draw(lambda event, message: vault.log(event, message, when=now), state, nt, now)
    vault.save_state(state)
    return state


def announced_draws(state: dict, nt: NextToto | None) -> list[tuple[date, str]]:
    """(date, draw type) of every TOTO draw the next draw page has announced so far: the page's
    own draw type word for its date, "special" for a date off the regular Monday and Thursday."""
    out: dict[date, str] = {}
    for value in (state.get(UPCOMING_KEY) or {}).get(GAME) or []:
        try:
            d = date.fromisoformat(str(value)[:10])
        except ValueError:
            continue
        out[d] = "normal" if d.weekday() in C.TOTO_WEEKDAYS else "special"
    if nt is not None and nt.draw_datetime is not None and nt.draw_type_hint in outlook.GUARANTEED_TYPES:
        out[nt.draw_datetime.astimezone(SG).date()] = nt.draw_type_hint
    return sorted(out.items())


# Prize rules


def _cached_rules(vault: Vault):
    """Prize rules from the vault cache (whatever its age), else the built in values."""
    try:
        text = vault.prize_rules_path.read_text(encoding="utf-8")
        rules = prize_rules.rules_from_dict(json.loads(text))
        log.info("Using the prize rules cached at %s", rules.checked_at)
        return rules
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.warning("Prize rules cache could not be read: %s", exc)
    return prize_rules.load_prize_rules(None)


# prize_rules.load_prize_rules says this in source_note when it parsed the page in this run.
_RULES_CONFIRMED_NOW = "confirmed from the official page"


def _rules_text(rules, now: datetime) -> str | None:
    """FETCH row for the prize rules: the official page was checked this run, could not be read
    this run, or built in values. None when they were reused from the cache unchanged (nothing
    new to log)."""
    checked = _parse_dt(getattr(rules, "checked_at", None))
    note = _plain(getattr(rules, "source_note", "") or "")
    if checked is None:
        head = "Prize rules: built in values"
    elif rules.checked_at != now.isoformat(timespec="seconds"):
        return None  # reused from Data/prize_rules.json, checked again after PRIZE_RULES_MAX_AGE_DAYS
    elif _RULES_CONFIRMED_NOW in note:
        head = "Prize rules: the official prize page was checked this run"
    else:
        head = "Prize rules: the official prize page could not be read this run"
    return f"{head}. {note}".strip()


# Data


@dataclass
class _Data:
    toto: pd.DataFrame  # everything stored
    new_draws: list[int] = field(default_factory=list)
    repaired: list[int] = field(default_factory=list)  # stored draws completed this run
    problem: str | None = None  # why the site could not be used
    fetched: bool = False  # the site was read this run


def _rewrite_repaired_notes(vault: Vault, data: _Data, activity: _Activity) -> None:
    """Rewrite the draw note of a stored draw that was incomplete and got completed this run, so
    its note shows the winning shares. Only notes that exist are touched."""
    for n in data.repaired:
        rows = data.toto[data.toto["draw_number"] == int(n)]
        if rows.empty:
            continue
        try:
            rel, fm, body = notes.toto_draw_note(rows.iloc[-1])
            if vault.path(rel).exists() and vault.write_note(rel, body, fm):
                activity(EV_NOTE, f"Updated {_link(rel)} with the complete result")
        except Exception as exc:  # a note is never worth failing the run
            log.warning("Could not rewrite the note for TOTO draw %s: %s", n, exc)


def _save_toto(df: pd.DataFrame, old: pd.DataFrame, vault: Vault, warnings: list[str],
               activity: _Activity) -> None:
    """Save toto.csv when its content changed (so Obsidian sync sees no churn otherwise)."""
    if len(df) == len(old) and store.normalise_toto(df).equals(store.normalise_toto(old)):
        return
    try:
        store.save_toto(df, vault.toto_csv)
    except OSError as exc:
        msg = f"TOTO data could not be saved to Data/{vault.toto_csv.name} ({_error_text(exc)})"
        warnings.append(msg + ".")
        activity(EV_ERROR, msg)


def _log_new_draws(df: pd.DataFrame, numbers: list[int], activity: _Activity) -> None:
    """One NEW DRAW row per new draw (newest NEW_DRAW_LOG_LIMIT), one summary row for the rest."""
    if not numbers:
        return
    numbers = sorted(numbers)
    older, shown = numbers[:-NEW_DRAW_LOG_LIMIT], numbers[-NEW_DRAW_LOG_LIMIT:]
    if older:
        activity(EV_NEW_DRAW, f"TOTO: {plural(len(older), 'older draw')} added ({_span(older)})")
    rows = df.set_index("draw_number")
    for n in shown:
        if n not in rows.index:
            continue
        r = rows.loc[n]
        nums = toto_nums([r[f"n{i}"] for i in range(1, 7)])
        winners = int(r["g1_winners"])
        g1 = f"Group 1 {money(r['jackpot'])}, {plural(winners, 'winner') if winners else 'no winner'}"
        activity(EV_NEW_DRAW, f"TOTO draw {n} on {fmt_date(r['draw_date'])}: {nums}, "
                              f"additional {int(r['additional'])}, {g1}")


def _fetch_summary(result: UpdateResult) -> str:
    if result.new_draws:
        text = f"TOTO: {plural(len(result.new_draws), 'new draw')} ({_span(result.new_draws)})"
    else:
        text = "TOTO: nothing new"
    if result.latest_on_site:
        text += f", latest on the site is draw {result.latest_on_site}"
    if result.verified:
        text += ", newest stored draw matches the site"
    elif result.latest_in_csv is not None:
        text += f", newest stored draw is {result.latest_in_csv} (not confirmed against the site)"
    if result.failed_draws:
        text += f", {plural(len(result.failed_draws), 'draw')} failed ({_span(result.failed_draws)})"
    return text


# Fetch messages that need the user's attention (the rest are routine and summed up in the FETCH row).
_DRAW_TYPE_LIST_FAILED = "draw list could not be used"  # cascade, Hongbao or special list
_DATE_MISMATCH = "in the data but"  # "draw 4123 is dated ... in the data but ... on the site"
_NOT_COMPLETE = "not complete yet"  # "draw 4123 is on the site but its winning shares are not complete yet"


def _notable_messages(result: UpdateResult) -> list[str]:
    """The fetch messages worth a warning: a draw type list that could not be used (new draws may
    then be tagged normal), a date that differs from the site and a result still incomplete."""
    keys = (_DRAW_TYPE_LIST_FAILED, _DATE_MISMATCH, _NOT_COMPLETE)
    return [_plain(m) for m in result.messages or [] if any(k in str(m) for k in keys)]


def _update_from_site(vault: Vault, settings: Settings, state: dict, fetcher, now: datetime,
                      activity: _Activity, warnings: list[str]) -> _Data:
    """Step 2: bring toto.csv up to date."""
    old = store.load_toto(vault.toto_csv)
    data = _Data(toto=old)
    recorder = _FailureRecorder(fetcher)
    try:
        df, result = site.update_toto(recorder, old, settings.toto_start_draw, skip=_skip_list(state), now=now)
    except Exception as exc:  # FetchError when the draw list is unreachable, anything else is a bug
        if not isinstance(exc, FetchError):
            log.exception("TOTO update failed")
        reason = _error_text(exc) if not isinstance(exc, FetchError) else _plain(exc)
        data.problem = reason
        stored = f"using the {plural(len(old), 'stored draw')}" if len(old) else "and no draws are stored yet"
        msg = f"TOTO results could not be fetched from the Singapore Pools site ({reason}), {stored}"
        warnings.append(msg + ".")
        activity(EV_ERROR, msg)
        return data

    data.fetched = True
    activity(EV_FETCH, _fetch_summary(result))
    transient = [n for n in result.failed_draws if site.toto_result_url(n) in recorder.transient]
    skipped_now = _track_failures(state, result, activity, transient)
    retry = [n for n in result.failed_draws if n not in skipped_now]
    if retry:
        warnings.append(f"TOTO: {plural(len(retry), 'draw')} could not be added ({_span(retry)}); they will "
                        "be tried again on the next run.")
    if skipped_now:
        warnings.append(f"TOTO: {plural(len(skipped_now), 'draw')} ({_span(skipped_now)}) failed in "
                        f"{SKIP_AFTER_FAILED_RUNS} runs in a row and will not be fetched again (skip list "
                        "in Data/state.json).")
    notable = _notable_messages(result)
    for msg in notable:
        warnings.append(msg)
        # A result still being published is expected right after the draw, not an error.
        activity(EV_FETCH if _NOT_COMPLETE in msg else EV_ERROR, msg.rstrip("."))
    if result.latest_in_csv is not None and not result.verified:
        if result.latest_in_csv != result.latest_on_site:
            warnings.append(f"TOTO: the newest stored draw ({result.latest_in_csv}) does not match the latest "
                            f"draw on the site ({result.latest_on_site}).")
        elif not any(_DATE_MISMATCH in m or _NOT_COMPLETE in m for m in notable):
            warnings.append(f"TOTO: draw {result.latest_in_csv} has a different date in the stored data than "
                            "on the site, please check it.")
    if df.empty and not result.new_draws:
        data.problem = "none of the result pages could be read"
    _save_toto(df, old, vault, warnings, activity)
    _log_new_draws(df, result.new_draws, activity)
    data.toto = df
    data.new_draws = list(result.new_draws)
    data.repaired = list(result.repaired_draws or [])
    return data


# Demo


def _demo_last_day(now: datetime) -> date:
    """Newest draw day whose result would be out by ``now`` (results are in by the 7.30pm run)."""
    d = now.date()
    if now.time() < C.DEFAULT_RUN_TIME:
        d -= timedelta(days=1)
    while d.weekday() not in C.TOTO_WEEKDAYS:
        d -= timedelta(days=1)
    return d


def _demo_draw_number(day: date) -> int:
    """Draw number of ``day`` counted in draw days from the anchor draw."""
    anchor_day, n = DEMO_TOTO_ANCHOR
    step = 1 if day >= anchor_day else -1
    d = anchor_day
    while d != day:
        d += timedelta(days=step)
        if d.weekday() in C.TOTO_WEEKDAYS:
            n += step
    return n


def _demo_start(end: date, n: int) -> date:
    d, count = end, 1
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() in C.TOTO_WEEKDAYS:
            count += 1
    return d


def demo_history(now: datetime | None = None) -> pd.DataFrame:
    """Synthetic TOTO history ending on the latest draw day before ``now``."""
    end = _demo_last_day(_aware(now))
    last = _demo_draw_number(end)
    return synth.synth_toto(n_draws=DEMO_TOTO_DRAWS, start_draw=last - DEMO_TOTO_DRAWS + 1,
                            start_date=_demo_start(end, DEMO_TOTO_DRAWS))


def demo_tickets_note(toto: pd.DataFrame, next_toto: NextToto | None) -> str:
    """Tickets.md for a demo vault: the starter note plus a few tickets made from the synthetic
    results (a Group 7 winner, a ticket with no prize and one waiting for the next draw)."""
    last = toto.sort_values("draw_number").iloc[-1]
    win = store.toto_numbers(last)
    additional = int(last["additional"])
    others = [n for n in range(1, 50) if n not in win and n != additional]
    group7 = sorted(win[:3] + others[-3:])

    def day(value: Any) -> str:
        d = pd.Timestamp(value)
        return f"{d.day} {d:%b %Y}"

    next_day = (next_toto.draw_datetime.date() if next_toto is not None and next_toto.draw_datetime
                else pd.Timestamp(last["draw_date"]).date() + timedelta(days=4))
    rows = [
        f"| TOTO | {day(last['draw_date'])} | {' '.join(map(str, group7))} | Ordinary | $1 |",
        f"| TOTO | {day(last['draw_date'])} | {' '.join(map(str, others[:6]))} | Ordinary | $1 |",
        f"| TOTO | {day(next_day)} | {' '.join(map(str, others[6:12]))} | Ordinary | $1 |",
    ]
    rule = "| --- | --- | --- | --- | --- |"
    text = tickets.TICKETS_TEMPLATE
    head, sep, tail = text.partition(rule + "\n")
    if not sep:
        return text + "\n" + "\n".join(rows) + "\n"
    note = ("\nThe rows above are demo tickets made up from the synthetic results, so the ticket check "
            "has something to show.\n")
    return head + sep + "\n".join(rows) + "\n" + note + tail


def _demo_data(vault: Vault, now: datetime, activity: _Activity, warnings: list[str]) -> _Data:
    """Step 2 for a demo: synthetic history saved into the vault, the site is never contacted."""
    old = store.load_toto(vault.toto_csv)
    toto = demo_history(now)
    known = set(int(n) for n in old["draw_number"]) if len(old) else set()
    data = _Data(toto=toto, new_draws=sorted(int(n) for n in toto["draw_number"] if int(n) not in known))
    _save_toto(toto, old, vault, warnings, activity)
    activity(EV_FETCH, f"Demo run: synthetic history from huatbot.synth, the site was not contacted "
                       f"({plural(len(toto), 'TOTO draw')})")
    _log_new_draws(toto, data.new_draws, activity)
    return data


def _analysis_frame(toto: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The history the analysis uses: draws from settings.toto_start_draw (the CSV keeps every row)."""
    if toto.empty:
        return toto
    kept = toto[toto["draw_number"] >= int(settings.toto_start_draw)].reset_index(drop=True)
    return toto if kept.empty else kept  # a start draw above everything stored: use what there is


# Step 3: the next draw and the next big prize


def build_context(
    toto: pd.DataFrame,
    settings: Settings,
    rules,
    *,
    now: datetime | None = None,
    next_toto: NextToto | None = None,
    new_draws: Iterable[int] = (),
    warnings: list[str] | None = None,
    fetched: bool = False,
    announced: Iterable[tuple[date, str]] = (),
) -> Context:
    """Buy signal, jackpot outlook and jackpot history for the stored history.

    A part that fails is logged and becomes a plain warning; the rest still runs. Tickets are
    added by ``run`` (step 4).
    """
    ctx = Context(now=_aware(now), settings=settings, rules=rules, toto=toto, next_toto=next_toto,
                  fetched=fetched, new_draws=list(new_draws or []), warnings=list(warnings or []))
    if toto.empty:
        return ctx

    def safe(what: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:
            log.exception("%s failed", what)
            ctx.warnings.append(f"{what} could not be worked out in this run ({type(exc).__name__}).")
            return default

    table = safe("The TOTO sales estimate", lambda: sales.sales_table(toto, rules))
    # A next draw page that still shows the draw just held (or stored info from an earlier run)
    # carries that draw's jackpot: leave it out so the signal, its log row and the commentary
    # never present an old jackpot as the next one.
    current = next_toto if next_toto is None or report.next_info_is_current(ctx) else None
    ctx.outlook = safe("The jackpot outlook", lambda: outlook.jackpot_outlook(
        toto, rules, current, table=table, today=ctx.now.date(), announced=list(announced)))
    ctx.history = safe("The jackpot history", lambda: outlook.jackpot_history(toto))
    signal_next = current
    if (current is None or current.jackpot_estimate is None) and ctx.outlook is not None \
            and ctx.outlook.jackpot is not None:
        # No jackpot from the page: rate the draw on the jackpot worked out from the results,
        # the same figure the messages show (the outlook notes say where it comes from).
        signal_next = NextToto(draw_datetime=current.draw_datetime if current is not None else None,
                               jackpot_estimate=ctx.outlook.jackpot, draw_type=ctx.outlook.draw_type,
                               draw_type_hint=current.draw_type_hint if current is not None else None)
    ctx.buy_signal = safe("The buy signal", lambda: buysignal.buy_signal(signal_next, toto, settings, rules, table))
    return ctx


# Step 4: tickets


def check_tickets(ctx: Context, vault: Vault, toto_all: pd.DataFrame, *, save: bool = True,
                  activity: _Activity | None = None,
                  on_settled: Callable[[list[dict]], Any] | None = None) -> None:
    """Read Tickets.md, sync and settle the ledger against every stored draw, save ledger.csv.

    Sets ctx.ledger, ctx.ledger_totals, ctx.settled_this_run and ctx.bad_ticket_lines.
    ``on_settled(rows)`` is called with the rows settled in this run before ledger.csv is
    written (only when there are some), so the caller can record them first.
    """
    text = vault.read_text(TICKETS_NOTE) or ""
    parsed = tickets.parse_tickets(text)
    bad = [t for t in parsed if t.error]
    old = store.load_ledger(vault.ledger_csv)
    ledger = tickets.sync_ledger(old, parsed, ctx.now, note_read=tickets.has_ticket_table(text))
    added = len(ledger) - len(old)
    ledger, settled = tickets.settle_ledger(ledger, toto_all, ctx.rules, ctx.now)
    ctx.ledger = ledger
    ctx.ledger_totals = tickets.ledger_totals(ledger)
    ctx.settled_this_run = settled
    ctx.bad_ticket_lines = bad
    if on_settled is not None and settled:
        on_settled(settled)

    if save and not (len(ledger) == len(old) and store.normalise_ledger(ledger).equals(store.normalise_ledger(old))):
        try:
            store.save_ledger(ledger, vault.ledger_csv)
        except OSError as exc:
            ctx.warnings.append(f"The ticket ledger could not be saved ({_error_text(exc)}).")
            if activity:
                activity(EV_ERROR, f"Data/ledger.csv could not be saved ({_error_text(exc)})")

    if activity is None:
        return
    valid = len(parsed) - len(bad)
    msg = f"Read [[Tickets]]: {plural(valid, 'ticket')}"
    if bad:
        msg += f", {plural(len(bad), 'line')} could not be read (listed in [[Ledger]])"
    activity(EV_TICKETS, msg)
    for r in settled[:LEDGER_LOG_LIMIT]:
        won = float(r.get("winnings") or 0.0)
        result = r.get("result") or "No prize"
        activity(EV_LEDGER, f"TOTO {fmt_date(r.get('draw_date'))} {r.get('numbers')} {r.get('bet_type')} "
                            f"{dollars(r.get('cost'))}: {result}" + (f", won {dollars(won)}" if won else ""))
    if len(settled) > LEDGER_LOG_LIMIT:
        activity(EV_LEDGER, f"and {plural(len(settled) - LEDGER_LOG_LIMIT, 'more ticket')} checked")
    if added or settled:
        totals = ctx.ledger_totals
        activity(EV_LEDGER, f"{plural(max(added, 0), 'new ticket')}, {plural(len(settled), 'ticket')} checked; "
                            f"all tickets: spent {dollars(totals.get('spent'))}, won "
                            f"{dollars(totals.get('won'))}, net {dollars(totals.get('net'))}")


# Step 5 helpers


def _log_signal(ctx: Context, activity: _Activity, state: dict | None = None) -> None:
    """A SIGNAL row with the buy signal and the next big prize for the next draw. With ``state``,
    only when it changed since it was last logged (a new draw or a new jackpot changes it)."""
    sig = ctx.buy_signal
    nd = report.next_draw(ctx)
    if sig is None or nd.held:  # sales are closed for a draw already held
        return
    draw = f" {nd.number}" if nd.number else ""
    jackpot = money(sig.jackpot) if sig.jackpot is not None else "not known"
    ev = f", return per $1 about {per_dollar(sig.ev_per_dollar)}" if sig.ev_per_dollar is not None else ""
    message = f"Buy signal {sig.label} for TOTO draw{draw}: jackpot {jackpot}, {sig.draw_type} draw{ev}"
    out = ctx.outlook
    if out is not None and len(out.steps) > 1:
        big = out.biggest
        message += (f"; next big prize about {money(big.jackpot)} at the cascade draw on "
                    f"{fmt_date(big.draw_date)} if nobody wins it first")
    if state is not None:
        seen = state.get("last_logged") if isinstance(state.get("last_logged"), dict) else {}
        digest = _digest(message)
        if seen.get(SIGNAL_LOG_KEY) == digest:
            return
        state["last_logged"] = {**seen, SIGNAL_LOG_KEY: digest}
    activity(EV_SIGNAL, message)


def _commentary_figures(ctx: Context) -> dict:
    """The commentary figures without the parts about a draw already held (``NextDraw.held``):
    its sales are closed, so it has no jackpot or buy signal to write about."""
    figures = commentary.figures_from_context(ctx)
    if report.next_draw(ctx).held:
        for key in ("next_toto", "buy_signal"):
            figures.pop(key, None)
    return figures


def _commentary(ctx: Context, demo: bool, activity: _Activity | None = None) -> str | None:
    """The optional commentary, as the report and message 2 show it (``report.commentary_text``
    drops sentences that restate the odds). When COMMENTARY is turned on but nothing usable
    came back, an ERROR row says so (the container log has the reason)."""
    if demo:
        return None  # a demo never calls out to claude
    try:
        memory = activity.vault.recent_activity(ctx.now) if activity is not None else ""
        text = commentary.build_commentary(_commentary_figures(ctx), memory=memory)
        reason = "the container log says why"
    except Exception as exc:  # optional extra, never fatal
        log.warning("Commentary skipped: %s", exc)
        text, reason = None, _error_text(exc)
    if text:
        kept = report.commentary_text(text)
        if not kept:
            reason = "the reply only restated the odds, which the messages already say once"
        text = kept or None
    if text is None and activity is not None and commentary.is_enabled():
        activity(EV_ERROR, f"Commentary is turned on but none was added in this run ({reason})")
    return text


def _has_new(ctx: Context, state: dict) -> bool:
    """True when the newest stored draw is newer than the last one posted."""
    latest = _latest(ctx.toto)
    if latest is None:
        return False
    try:
        return latest > int((state.get("last_posted") or {}).get(GAME))
    except (TypeError, ValueError):
        return True


# Step 6


def _deliver(messages: list[str], ctx: Context, state: dict, *, dry_run: bool, post: bool, force_post: bool,
             demo: bool, out: Callable[[str], Any], activity: _Activity,
             vault: Vault | None = None) -> tuple[bool, bool]:
    """Post, print or skip the messages. Returns (posted, ok). A set that was only partly
    posted before (state["posting"], same draw) resumes after the last message that went out."""
    count = plural(len(messages), "message")
    latest = _latest(ctx.toto)
    if demo:
        telegram.post_messages(messages, None, None, dry_run=True, out=out)
        activity(EV_DRY_RUN, f"Demo run, {count} printed, nothing posted")
        return False, True
    if not post:
        activity(EV_POST, f"Posting is turned off for this run, {count} not sent")
        return False, True
    if dry_run:
        telegram.post_messages(messages, None, None, dry_run=True, out=out)
        activity(EV_DRY_RUN, f"{count} printed, nothing posted")
        return False, True
    if not force_post and not _has_new(ctx, state):
        activity(EV_POST, f"Nothing new since the last post (TOTO {latest}), not posted again")
        return False, True

    token, chat_id = telegram.config_from_env()
    # state["posting"] records how many messages of this set already went out, saved after each
    # one, so a set that failed half way is finished on the next run instead of posted again.
    draws = {GAME: latest}
    progress = state.get("posting") if isinstance(state.get("posting"), dict) else {}
    done = 0
    if progress.get("draws") == draws:
        try:
            done = max(0, min(int(progress.get("sent") or 0), len(messages)))
        except (TypeError, ValueError):
            done = 0

    def sent(i: int) -> None:
        state["posting"] = {"draws": draws, "sent": done + i, "total": len(messages)}
        if done == 0:
            # Message 1 of this run, which lists the tickets checked but not posted yet, went out.
            state.pop("unposted_settled", None)
        if vault is not None:
            try:
                vault.save_state(state)
            except Exception as exc:  # the final save in run() tries again
                log.warning("state.json could not be saved after a message was posted: %s", exc)

    try:
        if done < len(messages):
            telegram.post_messages(messages[done:], token, chat_id, dry_run=False, out=out, on_sent=sent,
                                   reply_markup=listener.BUTTONS)
    except Exception as exc:
        log.exception("Posting to Telegram failed")
        msg = f"Posting to Telegram failed: {_error_text(exc)}"
        if done:
            msg += f" ({plural(done, 'message')} had already gone out in an earlier run)"
        ctx.warnings.append(msg + ".")
        activity(EV_ERROR, msg)
        return False, False
    state.pop("posting", None)
    if latest is not None:
        state.setdefault("last_posted", {})[GAME] = latest
    state["last_posted_at"] = ctx.now.isoformat(timespec="seconds")
    if done:
        activity(EV_POST, f"Posted the remaining {plural(len(messages) - done, 'message')} to Telegram "
                          f"(TOTO {latest}); the first {plural(done, 'message')} went out in an earlier run, "
                          "so they were not posted again")
    else:
        activity(EV_POST, f"Posted {count} to Telegram (TOTO {latest})")
    return True, True


# The run


def _settings_text(s: Settings) -> str:
    special = "flagged" if s.alert_on_special_draws else "not flagged"
    return (f"Read [[Settings]]: jackpot alert {money(s.jackpot_alert)}, special draws {special}, "
            f"history from draw {s.toto_start_draw}")


def _run_flags(dry_run: bool, fetch: bool, post: bool, demo: bool, force_post: bool) -> str:
    if demo:
        return " (demo with synthetic data)"
    flags = []
    if dry_run:
        flags.append("dry run")
    if not fetch:
        flags.append("no fetch")
    if not post:
        flags.append("no posting")
    flags.append("manual" if force_post else "scheduled")
    return f" ({', '.join(flags)})"


def _telegram_missing_text() -> str:
    msg = ("Telegram is not set up (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing), so this is a dry "
           "run: the messages are printed, not posted.")
    env_file = (os.environ.get("ENV_FILE") or "").strip()
    if Path(env_file or ".env").is_file():
        # The CLI reads this file (cli.load_dotenv), so it gave no usable value.
        name = "The ENV_FILE settings file" if env_file else "The .env file in the working folder"
        msg += (f" {name} has no TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID value the bot could use: fill in "
                "both there (a value already set in the shell, even an empty one, wins over the file).")
    return msg


def run(*, dry_run: bool = False, fetch: bool = True, post: bool = True, demo: bool = False,
        vault: Vault | str | Path | None = None, fetcher=None, now: datetime | None = None,
        out: Callable[[str], Any] = print, force_post: bool = False) -> RunResult:
    """Do one full run (see the module docstring for the steps). Never raises for expected
    problems: a network failure is a warning, a fatal problem gives ``RunResult(ok=False)``.

    dry_run     print the 2 messages instead of posting them
    fetch       False uses only the stored data (no network at all)
    post        False skips Telegram entirely (nothing printed or sent)
    demo        synthetic history, no network, never posts
    vault       Vault, vault folder path, or None for VAULT_PATH
    fetcher     http.Fetcher or a test double; created (and closed) here when needed
    now         run time (Singapore time); defaults to the current time
    out         where printed messages go
    force_post  post even when nothing is new since the last post (manual runs)
    """
    started = time.monotonic()
    when = _aware(now) if now is not None else None
    now = _aware(now)
    if demo:
        if not isinstance(vault, Vault):
            vault = Vault(Path(vault) if vault else Path("demo-vault"),
                          (os.environ.get("VAULT_FOLDER") or "").strip() or "Huat Bot")
        fetch = False
    vault = resolve_vault(vault)
    activity = _Activity(vault, when)
    warnings: list[str] = []
    result = RunResult(ok=False)
    own_fetcher = None
    state: dict = {}

    try:
        # 1 vault layout, settings
        activity(EV_RUN, f"Run started{_run_flags(dry_run, fetch, post, demo, force_post)}")
        templates = {"Settings.md": SETTINGS_TEMPLATE} if demo else dict(TEMPLATES)
        for rel in vault.ensure_layout(templates):
            activity(EV_NOTE, f"Created {_link(rel)} (starter note)")
        settings = load_settings(vault)
        activity(EV_SETTINGS, _settings_text(settings))
        for w in settings.warnings:
            activity(EV_SETTINGS, w)
        state = load_state(vault)

        if not demo and post and not dry_run:
            token, chat_id = telegram.config_from_env()
            if not token or not chat_id:
                msg = _telegram_missing_text()
                warnings.append(msg)
                log.warning(msg)
                activity(EV_DRY_RUN, "Telegram is not set up, treating this run as a dry run")
                dry_run = True

        # 2 prize rules and data
        if demo:
            warnings.append(DEMO_WARNING)
            rules = prize_rules.load_prize_rules(None)
            data = _demo_data(vault, now, activity, warnings)
        elif fetch:
            if fetcher is None:
                fetcher = own_fetcher = Fetcher()
            rules = prize_rules.load_prize_rules(fetcher, cache_path=vault.prize_rules_path,
                                                 max_age_days=PRIZE_RULES_MAX_AGE_DAYS, now=now)
            rules_row = _rules_text(rules, now)
            if rules_row:
                activity(EV_FETCH, rules_row)
            data = _update_from_site(vault, settings, state, fetcher, now, activity, warnings)
        else:
            rules = _cached_rules(vault)
            data = _Data(toto=store.load_toto(vault.toto_csv))
            activity(EV_FETCH, f"Fetching is off for this run, using the stored data "
                               f"({plural(len(data.toto), 'TOTO draw')})")

        if data.toto.empty:
            reason = data.problem or ("" if fetch else "fetching was turned off")
            text = _plain("Huat Bot could not run: no TOTO results are stored yet"
                          + (f" and the Singapore Pools site could not be used ({reason})" if reason else "")
                          + ". It will try again on the next run.")
            warnings.append(text)
            activity(EV_ERROR, text)
            if post and not dry_run and not demo:
                notify(text, out=out)
            result.new_draws = data.new_draws
            return result

        # 3 next draw, buy signal, outlook
        if demo:
            nt = synth.synth_next_toto(data.toto)
            _store_next_draw(state, nt, now)
        else:
            nt = None
            if fetch:
                nt = site.fetch_next_draw(fetcher, data.toto)
                _store_next_draw(state, nt, now)
                _log_next_draw(activity, state, nt, now)
            nt = nt or next_draw_from_state(state, data.toto)
            last = _latest_date(data.toto)
            if nt is not None and nt.draw_datetime is not None and last and nt.draw_datetime.date() <= last:
                warnings.append(f"The next TOTO draw page has not been updated yet (it still shows the draw on "
                                f"{fmt_date(nt.draw_datetime)}), so the next jackpot is worked out from the "
                                "stored results.")

        known = len(warnings)
        ctx = build_context(_analysis_frame(data.toto, settings), settings, rules, now=now, next_toto=nt,
                            new_draws=data.new_draws, warnings=warnings, fetched=data.fetched,
                            announced=announced_draws(state, nt))
        warnings = ctx.warnings  # one list from here on, so later steps add to the report too
        for w in warnings[known:]:  # parts that failed (each is already a warning)
            activity(EV_ERROR, _plain(w).rstrip("."))
        _log_signal(ctx, activity, state)

        # 4 tickets
        if demo:
            for rel in vault.ensure_layout({TICKETS_NOTE: demo_tickets_note(data.toto, nt)}):
                activity(EV_NOTE, f"Created {_link(rel)} with demo tickets")
        # A run that posts keeps the tickets it checked in state["unposted_settled"] until a
        # message 1 listing them went out, saved before ledger.csv: after a failed post, or a run
        # cut off once the ledger was saved, the next message 1 still lists them.
        posting = post and not dry_run and not demo
        unposted = state.get("unposted_settled")
        carry = [str(x) for x in unposted] if posting and isinstance(unposted, list) else []

        def remember_settled(rows: list[dict]) -> None:
            ids = carry + [str(r["ticket_id"]) for r in rows if str(r.get("ticket_id")) not in carry]
            if ids == carry:
                return
            state["unposted_settled"] = ids
            try:
                vault.save_state(state)
            except Exception as exc:  # the final save in finally tries again
                log.warning("state.json could not be saved before the ledger: %s", exc)

        check_tickets(ctx, vault, data.toto, save=True, activity=activity,
                      on_settled=remember_settled if posting else None)
        if carry:
            have = {str(r.get("ticket_id")) for r in ctx.settled_this_run}
            earlier = [r for r in tickets.settled_rows(ctx.ledger, carry) if str(r["ticket_id"]) not in have]
            ctx.settled_this_run = earlier + list(ctx.settled_this_run)

        # 5 commentary, report, notes, state
        ctx.commentary = _commentary(ctx, demo, activity)
        if ctx.commentary:
            activity(EV_NOTE, "Added a short commentary written from the computed figures")
        report_md = report.full_report(ctx)
        notes.write_all(vault, ctx, report_md)
        _rewrite_repaired_notes(vault, data, activity)
        # A rerun whose report only differs in its time keeps the earlier report note.
        result.report_path = str(vault.path(notes.report_rel(vault, ctx, report_md)))
        messages = report.telegram_messages(ctx)
        result.messages = messages
        if not demo:
            listener.save_last_messages(vault, messages)

        # 6 telegram
        posted, ok = _deliver(messages, ctx, state, dry_run=dry_run, post=post, force_post=force_post,
                              demo=demo, out=out, activity=activity, vault=vault)
        result.posted = posted
        result.ok = ok
        result.new_draws = list(data.new_draws)
        return result
    except Exception as exc:  # last line of defence: log it in the vault and report failure
        log.exception("Huat Bot run failed")
        msg = f"Run failed: {_error_text(exc)}"
        warnings.append(msg + ".")
        activity(EV_ERROR, msg)
        result.ok = False
        return result
    finally:
        if own_fetcher is not None:
            own_fetcher.close()
        state["last_run"] = {
            "at": now.isoformat(timespec="seconds"),
            "ok": result.ok,
            "posted": result.posted,
            "dry_run": bool(dry_run),
            "demo": bool(demo),
            "new_draws": len(result.new_draws),
        }
        try:
            vault.save_state(state)
        except Exception as exc:
            log.warning("state.json could not be saved: %s", exc)
            activity(EV_ERROR, f"Data/state.json could not be saved ({_error_text(exc)})")
        result.warnings = list(dict.fromkeys(_plain(w) for w in warnings if w))
        report_link = f", report {_link(result.report_path)}" if result.report_path else ""
        activity(EV_RUN, f"Run {'finished' if result.ok else 'ended with a problem'} in "
                         f"{plural(round(time.monotonic() - started), 'second')}: "
                         f"{plural(len(result.new_draws), 'new draw')}"
                         f"{', posted' if result.posted else ''}{report_link}")


# Other entry points used by the CLI


def fetch_data(*, vault: Vault | str | Path | None = None, fetcher=None, now: datetime | None = None) -> RunResult:
    """Update toto.csv and the next draw info only (no analysis, notes or posting)."""
    when = _aware(now) if now is not None else None
    now = _aware(now)
    vault = resolve_vault(vault)
    activity = _Activity(vault, when)
    for rel in vault.ensure_layout(dict(TEMPLATES)):
        activity(EV_NOTE, f"Created {_link(rel)} (starter note)")
    settings = load_settings(vault)
    state = load_state(vault)
    warnings: list[str] = []
    own = None
    if fetcher is None:
        fetcher = own = Fetcher()
    try:
        activity(EV_RUN, "Fetch started")
        data = _update_from_site(vault, settings, state, fetcher, now, activity, warnings)
        _rewrite_repaired_notes(vault, data, activity)
        nt = site.fetch_next_draw(fetcher, data.toto)
        _store_next_draw(state, nt, now)
        activity(EV_FETCH, f"Next draw: {_next_draw_text(nt)}")
        vault.save_state(state)
    finally:
        if own is not None:
            own.close()
    return RunResult(ok=data.problem is None, new_draws=data.new_draws, warnings=[_plain(w) for w in warnings])


def build_report(vault: Vault | str | Path | None = None, *, now: datetime | None = None) -> str | None:
    """The full markdown report from the stored data only: no fetching, posting or note writing.
    None when no data is stored yet."""
    vault = resolve_vault(vault)
    now = _aware(now)
    settings = load_settings(vault)
    state = load_state(vault)
    toto = store.load_toto(vault.toto_csv)
    if toto.empty:
        return None
    nt = next_draw_from_state(state, toto)
    ctx = build_context(_analysis_frame(toto, settings), settings, _cached_rules(vault), now=now, next_toto=nt,
                        announced=announced_draws(state, nt))
    check_tickets(ctx, vault, toto, save=False, activity=None)
    return report.full_report(ctx)
