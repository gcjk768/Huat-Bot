"""One Huat Bot run: fetch, analyse, suggest, check tickets, write the vault, post.

The Obsidian vault on the NAS (``vault.Vault``) is the single storage. Everything the bot keeps
lives in it: ``Data/toto.csv``, ``Data/fourd.csv``, ``Data/ledger.csv``, ``Data/state.json``,
``Data/prize_rules.json`` and ``Data/backtest_cache.json``, plus the notes (dashboard, ledger,
reports, suggestions, one note per draw) and the monthly activity log, which gets a row for
every meaningful action (RUN, SETTINGS, FETCH, NEW DRAW, BACKTEST, TICKETS, LEDGER, SUGGEST,
SIGNAL, NOTE, POST, DRY RUN, ERROR; the scheduler adds SCHEDULE and WAIT).

``run`` follows the seven steps of the SPEC:

1. vault layout (Settings.md and Tickets.md starter notes), settings, RUN row
2. prize rules (cached), incremental CSV update with the skip lists from state.json
   (a site that cannot be reached is a warning: stored data is used; with nothing stored at
   all the run stops, logs an ERROR and sends a short Telegram notice)
3. next draws, analysis, crowd scores, picks (seeded with the next draw number), plans and
   the buy signal (``build_context``)
4. backtests, cached in backtest_cache.json by latest draw, settings and prize rules
5. tickets: Tickets.md is read, the ledger synced, settled and saved
6. optional commentary, the full report, every note, state.json
7. Telegram: posted unless this is a dry run (no token configured also means a dry run)

state.json keys written here:

``next_draws``      {"toto": {"draw_datetime", "jackpot_estimate", "draw_type", ...},
                     "4d": {"draw_datetime", ...}} (also read by the scheduler)
``last_posted``     {"toto": 4123, "4d": 5432}: newest draw already posted. A scheduled run
                    (``force_post=False``) posts only when a reported game has a newer draw.
``posting``         {"draws": {"toto": 4123, ...}, "sent": 1, "total": 3}: a set of messages
                    that was only partly posted; the next post of the same draws resumes
                    after message ``sent``. Removed once all of them went out.
``skip``            {"toto": [...], "4d": [...]}: draw numbers never fetched again. A draw
                    whose page fails in 3 runs in a row is added automatically (never one of
                    the newest 10 draws, and never for fetch trouble such as timeouts, 403,
                    429 or 5xx, only for a page that is gone or cannot be read); the lists
                    can also be edited by hand.
``fetch_failures``  failed runs per draw, feeding ``skip``
``last_run``        when, which games, ok, new draws, dry run or demo

``demo=True`` uses synthetic history from ``huatbot.synth`` (no network, never posts) and
writes it into the given vault, so the whole pipeline can be tried without the site.
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
    analysis_fourd,
    analysis_toto,
    backtest,
    buysignal,
    commentary,
    notes,
    prize_rules,
    report,
    store,
    strategies,
    suggest,
    synth,
    telegram,
    tickets,
)
from . import constants as C
from . import fetch as site
from .http import Fetcher, FetchError
from .models import Context, NextFourD, NextToto, RunResult, Settings, UpdateResult
from .report import dollars
from .settings import SETTINGS_TEMPLATE, load_settings
from .textfmt import (
    fmt_date,
    fmt_datetime,
    money,
    per_dollar,
    plural,
    remove_dashes,
    toto_nums,
)
from .vault import SG, Vault

log = logging.getLogger(__name__)

GAMES = ("toto", "4d")
LABELS = {"toto": "TOTO", "4d": "4D"}

TEMPLATES = {"Settings.md": SETTINGS_TEMPLATE, "Tickets.md": tickets.TICKETS_TEMPLATE}
TICKETS_NOTE = "Tickets.md"

# Activity log events written by the runner.
EV_RUN = "RUN"
EV_SETTINGS = "SETTINGS"
EV_FETCH = "FETCH"
EV_NEW_DRAW = "NEW DRAW"
EV_BACKTEST = "BACKTEST"
EV_TICKETS = "TICKETS"
EV_LEDGER = "LEDGER"
EV_SUGGEST = "SUGGEST"
EV_SIGNAL = "SIGNAL"
EV_NOTE = "NOTE"
EV_POST = "POST"
EV_DRY_RUN = "DRY RUN"
EV_ERROR = "ERROR"

NEW_DRAW_LOG_LIMIT = 10  # NEW DRAW rows per game; a big first fetch gets one summary row for the rest
LEDGER_LOG_LIMIT = 20  # settled tickets logged one by one
SKIP_AFTER_FAILED_RUNS = 3  # a draw page failing in this many runs goes on the skip list
SKIP_PROTECT_NEWEST = 10  # ... unless it is one of the newest draws on the site
PRIZE_RULES_MAX_AGE_DAYS = 7
PRIZE_RULES_CACHE_NAME = "prize_rules.json"
BACKTEST_CACHE_VERSION = 1

# Demo history: about as long as needed for a 300 draw backtest with 100 draws of warm up.
DEMO_TOTO_DRAWS = 600
DEMO_MIN_FOURD_DRAWS = 450
# Anchors so synthetic draw numbers look like real ones (TOTO 4123 on Thu 1 Oct 2026 ...).
DEMO_TOTO_ANCHOR = (date(2026, 10, 1), 4123)
DEMO_FOURD_ANCHOR = (date(2026, 9, 30), 5432)
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


def _games_text(games: Iterable[str]) -> str:
    return _and(LABELS[g] for g in games) or "no game"


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


def normalise_games(games: Any) -> tuple[str, ...]:
    """("toto", "4d") order from "both", "toto", "4D", "fourd" or a list of them."""
    if games is None:
        return GAMES
    if isinstance(games, str):
        games = [games]
    wanted: set[str] = set()
    for g in games:
        key = str(g).strip().lower()
        if key in ("", "both", "all"):
            wanted.update(GAMES)
            continue
        key = {"fourd": "4d", "4 d": "4d"}.get(key, key)
        if key not in GAMES:
            raise ValueError(f"Unknown game {g!r}: use toto, 4d or both")
        wanted.add(key)
    return tuple(g for g in GAMES if g in wanted) or GAMES


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


def _skip_list(state: dict, game: str) -> set[int]:
    raw = (state.get("skip") or {}).get(game) or []
    out = set()
    for n in raw:
        try:
            out.add(int(n))
        except (TypeError, ValueError):
            log.warning("Ignoring %r in the %s skip list of state.json", n, LABELS[game])
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


def _track_failures(state: dict, game: str, result: UpdateResult, activity: _Activity,
                    transient: Iterable[int] = ()) -> list[int]:
    """Count the runs in a row in which each draw page failed; after SKIP_AFTER_FAILED_RUNS put the
    draw on the skip list (never one of the newest SKIP_PROTECT_NEWEST draws). A draw that did not
    fail in this run starts again from zero. A draw in ``transient`` (or in the result's
    ``transient_draws``, when the fetch layer provides it) failed only because of fetch trouble:
    its count is carried over unchanged and it never goes on the skip list for that run.
    Returns the draws put on the skip list in this run."""
    all_failures = state.get("fetch_failures") or {}
    previous = all_failures.get(game) or {}
    newest = int(result.latest_on_site or 0)
    skip = _skip_list(state, game)
    transient = {int(n) for n in transient} | {int(n) for n in getattr(result, "transient_draws", None) or ()}
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
        state.setdefault("skip", {})[game] = sorted(skip)
        activity(EV_FETCH, f"{LABELS[game]}: {plural(len(added), 'draw')} ({_span(added)}) failed in "
                           f"{SKIP_AFTER_FAILED_RUNS} runs in a row and went on the skip list in state.json")
    if counts:
        all_failures[game] = counts
    else:
        all_failures.pop(game, None)
    if all_failures:
        state["fetch_failures"] = all_failures
    else:
        state.pop("fetch_failures", None)
    return added


def _next_toto_state(nt: NextToto) -> dict:
    return {
        "draw_datetime": nt.draw_datetime.isoformat() if nt.draw_datetime else None,
        "jackpot_estimate": nt.jackpot_estimate,
        "draw_type": nt.draw_type,
        "draw_type_hint": nt.draw_type_hint,
        "raw_text": (nt.raw_text or "")[:300],
    }


def _next_fourd_state(nf: NextFourD) -> dict:
    return {
        "draw_datetime": nf.draw_datetime.isoformat() if nf.draw_datetime else None,
        "raw_text": (nf.raw_text or "")[:300],
    }


def _store_next_draws(state: dict, nt: NextToto | None, nf: NextFourD | None, now: datetime) -> None:
    """Keep the next draw info in state (a game whose page failed keeps its old entry)."""
    entry = dict(state.get("next_draws") or {})
    if nt is not None:
        entry["toto"] = _next_toto_state(nt)
    if nf is not None:
        entry["4d"] = _next_fourd_state(nf)
    if nt is not None or nf is not None:
        entry["checked_at"] = now.isoformat(timespec="seconds")
    if entry:
        state["next_draws"] = entry


def next_draws_from_state(state: dict, toto: pd.DataFrame | None,
                          fourd: pd.DataFrame | None) -> tuple[NextToto | None, NextFourD | None]:
    """Next draws remembered in state.json, ignoring any that are not after the newest stored draw."""
    entry = state.get("next_draws") or {}
    nt = nf = None
    t = entry.get("toto")
    if isinstance(t, dict):
        dt = _parse_dt(t.get("draw_datetime"))
        last = _latest_date(toto)
        if dt is not None and (last is None or dt.date() > last):
            jackpot = t.get("jackpot_estimate")
            nt = NextToto(
                draw_datetime=dt,
                jackpot_estimate=float(jackpot) if isinstance(jackpot, (int, float)) else None,
                draw_type=str(t.get("draw_type") or "normal"),
                draw_type_hint=t.get("draw_type_hint"),
                raw_text=str(t.get("raw_text") or ""),
            )
    f = entry.get("4d")
    if isinstance(f, dict):
        dt = _parse_dt(f.get("draw_datetime"))
        last = _latest_date(fourd)
        if dt is not None and (last is None or dt.date() > last):
            nf = NextFourD(draw_datetime=dt, raw_text=str(f.get("raw_text") or ""))
    return nt, nf


def _next_draws_text(nt: NextToto | None, nf: NextFourD | None) -> str:
    parts = []
    if nt is not None:
        jackpot = f", estimated jackpot {money(nt.jackpot_estimate)}" if nt.jackpot_estimate else ""
        when = fmt_datetime(nt.draw_datetime) if nt.draw_datetime else "date not announced"
        parts.append(f"TOTO {when}{jackpot}")
    if nf is not None:
        parts.append(f"4D {fmt_datetime(nf.draw_datetime) if nf.draw_datetime else 'date not announced'}")
    return "; ".join(parts) if parts else "the next draw pages could not be read"


def refresh_next_draws(vault: Vault | str | Path | None, fetcher, *, now: datetime | None = None,
                       log_activity: bool = True) -> dict:
    """Read the next draw pages, store them in state.json and return the state.

    Used by the scheduler before each cycle (special draws on unusual days come from here).
    """
    vault = resolve_vault(vault)
    now = _aware(now)
    vault.ensure_layout()
    state = vault.load_state()
    toto = store.load_toto(vault.toto_csv)
    nt, nf = site.fetch_next_draws(fetcher, toto)
    _store_next_draws(state, nt, nf, now)
    vault.save_state(state)
    if log_activity:
        vault.log(EV_FETCH, f"Next draws: {_next_draws_text(nt, nf)}", when=now)
    return state


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


def _rules_text(rules, now: datetime) -> str:
    """FETCH row for the prize rules: checked this run, reused from the cache, or built in."""
    checked = _parse_dt(getattr(rules, "checked_at", None))
    note = _plain(getattr(rules, "source_note", "") or "")
    if checked is None:
        head = "Prize rules: built in values"
    elif rules.checked_at == now.isoformat(timespec="seconds"):
        head = "Prize rules: the official prize pages were checked this run"
    else:
        head = (f"Prize rules: reused Data/{PRIZE_RULES_CACHE_NAME} from {fmt_datetime(checked)} "
                f"(checked again after {plural(PRIZE_RULES_MAX_AGE_DAYS, 'day')})")
    return f"{head}. {note}".strip()


def _rules_fingerprint(rules) -> str:
    """Short hash of the prize amounts (not the confirmation notes), for the backtest cache key."""
    d = prize_rules.rules_to_dict(rules)
    keep = {k: d[k] for k in ("pool_share_of_sales", "group_pool_pct", "fixed_prizes", "min_group1",
                              "fourd_prizes", "ibet_prizes")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


# Data


@dataclass
class _Data:
    toto: pd.DataFrame  # everything stored
    fourd: pd.DataFrame
    new_draws: dict[str, list[int]] = field(default_factory=dict)
    problems: dict[str, str] = field(default_factory=dict)  # game -> why the site could not be used


def _save_frame(game: str, df: pd.DataFrame, old: pd.DataFrame, vault: Vault, warnings: list[str],
                activity: _Activity) -> None:
    """Save a CSV when its content changed (so Obsidian sync sees no churn otherwise)."""
    normalise = store.normalise_toto if game == "toto" else store.normalise_fourd
    if len(df) == len(old) and normalise(df).equals(normalise(old)):
        return
    path = vault.toto_csv if game == "toto" else vault.fourd_csv
    try:
        (store.save_toto if game == "toto" else store.save_fourd)(df, path)
    except OSError as exc:
        msg = f"{LABELS[game]} data could not be saved to Data/{path.name} ({_error_text(exc)})"
        warnings.append(msg + ".")
        activity(EV_ERROR, msg)


def _log_new_draws(game: str, df: pd.DataFrame, numbers: list[int], activity: _Activity) -> None:
    """One NEW DRAW row per new draw (newest NEW_DRAW_LOG_LIMIT), one summary row for the rest."""
    if not numbers:
        return
    numbers = sorted(numbers)
    older, shown = numbers[:-NEW_DRAW_LOG_LIMIT], numbers[-NEW_DRAW_LOG_LIMIT:]
    if older:
        activity(EV_NEW_DRAW, f"{LABELS[game]}: {plural(len(older), 'older draw')} added ({_span(older)})")
    rows = df.set_index("draw_number")
    for n in shown:
        if n not in rows.index:
            continue
        r = rows.loc[n]
        day = fmt_date(r["draw_date"])
        if game == "toto":
            nums = toto_nums([r[f"n{i}"] for i in range(1, 7)])
            winners = int(r["g1_winners"])
            g1 = f"Group 1 {money(r['jackpot'])}, {plural(winners, 'winner') if winners else 'no winner'}"
            activity(EV_NEW_DRAW, f"TOTO draw {n} on {day}: {nums}, additional {int(r['additional'])}, {g1}")
        else:
            activity(EV_NEW_DRAW, f"4D draw {n} on {day}: 1st {r['first'] or 'n/a'}, 2nd {r['second'] or 'n/a'}, "
                                  f"3rd {r['third'] or 'n/a'}")


def _fetch_summary(game: str, result: UpdateResult) -> str:
    label = LABELS[game]
    if result.new_draws:
        text = f"{label}: {plural(len(result.new_draws), 'new draw')} ({_span(result.new_draws)})"
    else:
        text = f"{label}: nothing new"
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


def _notable_messages(result: UpdateResult) -> list[str]:
    """The fetch messages worth a warning and an ERROR row: a draw type list that could not be
    used (new draws may then be tagged normal) and a date that differs from the site."""
    keys = (_DRAW_TYPE_LIST_FAILED, _DATE_MISMATCH)
    return [_plain(m) for m in result.messages or [] if any(k in str(m) for k in keys)]


def _update_from_site(vault: Vault, settings: Settings, state: dict, games: tuple[str, ...], fetcher,
                      now: datetime, activity: _Activity, warnings: list[str]) -> _Data:
    """Step 2: bring the CSVs up to date for ``games`` (plus any game with nothing stored)."""
    data = _Data(toto=store.load_toto(vault.toto_csv), fourd=store.load_fourd(vault.fourd_csv))
    for game in GAMES:
        old = data.toto if game == "toto" else data.fourd
        if game not in games and not old.empty:
            continue
        label = LABELS[game]
        recorder = _FailureRecorder(fetcher)
        try:
            if game == "toto":
                df, result = site.update_toto(recorder, old, settings.toto_start_draw,
                                              skip=_skip_list(state, game), now=now)
            else:
                df, result = site.update_fourd(recorder, old, settings.fourd_history_draws,
                                               skip=_skip_list(state, game), now=now)
        except Exception as exc:  # FetchError when the draw list is unreachable, anything else is a bug
            if not isinstance(exc, FetchError):
                log.exception("%s update failed", label)
            reason = _error_text(exc) if not isinstance(exc, FetchError) else _plain(exc)
            data.problems[game] = reason
            stored = f"using the {plural(len(old), 'stored draw')}" if len(old) else "and no draws are stored yet"
            msg = f"{label} results could not be fetched from the Singapore Pools site ({reason}), {stored}"
            warnings.append(msg + ".")
            activity(EV_ERROR, msg)
            continue

        activity(EV_FETCH, _fetch_summary(game, result))
        url_fn = site.toto_result_url if game == "toto" else site.fourd_result_url
        transient = [n for n in result.failed_draws if url_fn(n) in recorder.transient]
        skipped_now = _track_failures(state, game, result, activity, transient)
        retry = [n for n in result.failed_draws if n not in skipped_now]
        if retry:
            warnings.append(f"{label}: {plural(len(retry), 'draw')} could not be added "
                            f"({_span(retry)}); they will be tried again on the next run.")
        if skipped_now:
            warnings.append(f"{label}: {plural(len(skipped_now), 'draw')} ({_span(skipped_now)}) failed in "
                            f"{SKIP_AFTER_FAILED_RUNS} runs in a row and will not be fetched again (skip "
                            "list in Data/state.json).")
        notable = _notable_messages(result)
        for msg in notable:
            warnings.append(msg)
            activity(EV_ERROR, msg.rstrip("."))
        if result.latest_in_csv is not None and not result.verified:
            if result.latest_in_csv != result.latest_on_site:
                warnings.append(f"{label}: the newest stored draw ({result.latest_in_csv}) does not match the "
                                f"latest draw on the site ({result.latest_on_site}).")
            elif not any(_DATE_MISMATCH in m for m in notable):
                warnings.append(f"{label}: draw {result.latest_in_csv} has a different date in the stored data "
                                "than on the site, please check it.")
        if df.empty and not result.new_draws:
            data.problems[game] = "none of the result pages could be read"
        _save_frame(game, df, old, vault, warnings, activity)
        _log_new_draws(game, df, result.new_draws, activity)
        data.new_draws[game] = list(result.new_draws)
        if game == "toto":
            data.toto = df
        else:
            data.fourd = df
    return data


def _demo_last_day(now: datetime, weekdays: tuple[int, ...]) -> date:
    """Newest draw day whose result would be out by ``now`` (results are in by the 7.30pm run)."""
    d = now.date()
    if now.time() < C.DEFAULT_RUN_TIME:
        d -= timedelta(days=1)
    while d.weekday() not in weekdays:
        d -= timedelta(days=1)
    return d


def _demo_draw_number(day: date, weekdays: tuple[int, ...], anchor: tuple[date, int]) -> int:
    """Draw number of ``day`` counted in draw days from the anchor draw."""
    anchor_day, anchor_no = anchor
    step = 1 if day >= anchor_day else -1
    d, n = anchor_day, anchor_no
    while d != day:
        d += timedelta(days=step)
        if d.weekday() in weekdays:
            n += step
    return n


def _demo_start(end: date, weekdays: tuple[int, ...], n: int) -> date:
    d, count = end, 1
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() in weekdays:
            count += 1
    return d


def demo_history(now: datetime | None = None, fourd_draws: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic TOTO and 4D history ending on the latest draw days before ``now``."""
    now = _aware(now)
    t_end = _demo_last_day(now, C.TOTO_WEEKDAYS)
    f_end = _demo_last_day(now, C.FOURD_WEEKDAYS)
    t_last = _demo_draw_number(t_end, C.TOTO_WEEKDAYS, DEMO_TOTO_ANCHOR)
    f_last = _demo_draw_number(f_end, C.FOURD_WEEKDAYS, DEMO_FOURD_ANCHOR)
    n_fourd = max(DEMO_MIN_FOURD_DRAWS, int(fourd_draws))
    toto = synth.synth_toto(n_draws=DEMO_TOTO_DRAWS, start_draw=t_last - DEMO_TOTO_DRAWS + 1,
                            start_date=_demo_start(t_end, C.TOTO_WEEKDAYS, DEMO_TOTO_DRAWS))
    fourd = synth.synth_fourd(n_draws=n_fourd, start_draw=f_last - n_fourd + 1,
                              start_date=_demo_start(f_end, C.FOURD_WEEKDAYS, n_fourd))
    return toto, fourd


def demo_tickets_note(toto: pd.DataFrame, fourd: pd.DataFrame, next_toto: NextToto | None) -> str:
    """Tickets.md for a demo vault: the starter note plus a few tickets made from the synthetic
    results (a TOTO Group 7 winner, a 4D Big starter winner, a loser and a pending ticket)."""
    t_last, f_last = toto.iloc[-1], fourd.iloc[-1]
    win = store.toto_numbers(t_last)
    additional = int(t_last["additional"])
    others = [n for n in range(1, 50) if n not in win and n != additional]
    group7 = sorted(win[:3] + others[-3:])
    drawn = {n for tier in store.fourd_numbers(f_last).values() for n in tier}
    loser = next(f"{n % 10000:04d}" for n in range(1234, 11234) if f"{n % 10000:04d}" not in drawn)
    starter = str(f_last["starter_1"]) or str(f_last["first"])

    def day(value: Any) -> str:
        d = pd.Timestamp(value)
        return f"{d.day} {d:%b %Y}"

    next_day = (next_toto.draw_datetime.date() if next_toto is not None and next_toto.draw_datetime
                else pd.Timestamp(t_last["draw_date"]).date() + timedelta(days=4))
    rows = [
        f"| TOTO | {day(t_last['draw_date'])} | {' '.join(map(str, group7))} | Ordinary | $1 |",
        f"| 4D | {day(f_last['draw_date'])} | {starter} | Big | $1 |",
        f"| 4D | {day(f_last['draw_date'])} | {loser} | Small | $1 |",
        f"| TOTO | {day(next_day)} | {' '.join(map(str, others[:6]))} | Ordinary | $1 |",
    ]
    rule = "| --- | --- | --- | --- | --- |"
    text = tickets.TICKETS_TEMPLATE
    head, sep, tail = text.partition(rule + "\n")
    if not sep:
        return text + "\n" + "\n".join(rows) + "\n"
    note = ("\nThe rows above are demo tickets made up from the synthetic results, so the ticket check "
            "has something to show.\n")
    return head + sep + "\n".join(rows) + "\n" + note + tail


def _demo_data(vault: Vault, settings: Settings, now: datetime, activity: _Activity,
               warnings: list[str]) -> _Data:
    """Step 2 for a demo: synthetic history saved into the vault, the site is never contacted."""
    old_toto, old_fourd = store.load_toto(vault.toto_csv), store.load_fourd(vault.fourd_csv)
    toto, fourd = demo_history(now, settings.fourd_history_draws)
    data = _Data(toto=toto, fourd=fourd)
    for game, df, old in (("toto", toto, old_toto), ("4d", fourd, old_fourd)):
        known = set(int(n) for n in old["draw_number"]) if len(old) else set()
        data.new_draws[game] = sorted(int(n) for n in df["draw_number"] if int(n) not in known)
        _save_frame(game, df, old, vault, warnings, activity)
    activity(EV_FETCH, f"Demo run: synthetic history from huatbot.synth, the site was not contacted "
                       f"({plural(len(toto), 'TOTO draw')}, {plural(len(fourd), '4D draw')})")
    for game, df in (("toto", toto), ("4d", fourd)):
        _log_new_draws(game, df, data.new_draws[game], activity)
    return data


def _analysis_frames(data: _Data, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The history the analysis uses: TOTO from settings.toto_start_draw, the latest
    settings.fourd_history_draws 4D draws. The CSVs themselves keep every row."""
    toto = data.toto
    if len(toto):
        toto = toto[toto["draw_number"] >= int(settings.toto_start_draw)].reset_index(drop=True)
        if toto.empty:  # a start draw above everything stored: use what there is
            toto = data.toto
    fourd = data.fourd.tail(max(1, int(settings.fourd_history_draws))).reset_index(drop=True)
    return toto, fourd


# Step 3: analysis


def build_context(
    toto: pd.DataFrame,
    fourd: pd.DataFrame,
    settings: Settings,
    rules,
    *,
    now: datetime | None = None,
    next_toto: NextToto | None = None,
    next_fourd: NextFourD | None = None,
    games_drawn: Iterable[str] = GAMES,
    new_draws: dict[str, list[int]] | None = None,
    warnings: list[str] | None = None,
) -> Context:
    """Analysis, crowd scores, picks, plans and the buy signal for the stored history.

    Picks are seeded with the next draw number, so a rerun for the same draw suggests the same
    numbers. A part that fails is logged and becomes a plain warning; the rest still runs.
    Backtests and tickets are added by ``run`` (steps 4 and 5).
    """
    ctx = Context(
        now=_aware(now), settings=settings, rules=rules, toto=toto, fourd=fourd,
        next_toto=next_toto, next_fourd=next_fourd, games_drawn=tuple(games_drawn),
        new_draws={k: list(v) for k, v in (new_draws or {}).items()}, warnings=list(warnings or []),
    )

    def safe(what: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:
            log.exception("%s failed", what)
            ctx.warnings.append(f"{what} could not be worked out in this run ({type(exc).__name__}).")
            return default

    if len(toto):
        table = safe("The TOTO crowd table", lambda: analysis_toto.crowd_table(toto, rules))
        ctx.toto_frequency = safe("TOTO number frequency", lambda: analysis_toto.frequency_table(toto))
        ctx.toto_overdue = safe("TOTO overdue numbers", lambda: analysis_toto.overdue(toto))
        ctx.toto_pairs = safe("TOTO pairs", lambda: analysis_toto.top_pairs(toto), [])
        ctx.toto_shape = safe("TOTO set shape", lambda: analysis_toto.shape_stats(toto), {})
        ctx.toto_chi = safe("The TOTO fairness test", lambda: analysis_toto.chi_square_numbers(toto), {})
        if table is not None:
            scores = safe("TOTO crowd scores", lambda: analysis_toto.crowd_scores(toto, rules, table=table))
            if scores is not None:
                ctx.crowd_scores, ctx.crowd_diag = scores
        seed = int(toto["draw_number"].max()) + 1
        ctx.toto_picks = safe("TOTO suggestions",
                              lambda: strategies.toto_picks(toto, ctx.crowd_scores, seed=seed), [])
        last_draw = store.toto_numbers(toto.sort_values("draw_number").iloc[-1])
        if ctx.toto_picks:
            ctx.toto_plan = safe("The TOTO plan", lambda: suggest.toto_plan(
                ctx.toto_picks, settings.toto_budget, ctx.crowd_scores, settings.offer_system7, last_draw))
        ctx.buy_signal = safe("The buy signal",
                              lambda: buysignal.buy_signal(next_toto, toto, settings, rules, table))

    ctx.fourd_bet_values = safe("4D bet type values", lambda: analysis_fourd.bet_type_value(rules), {})
    if len(fourd):
        ctx.fourd_position_freq = safe("4D digit frequency", lambda: analysis_fourd.position_digit_freq(fourd))
        ctx.fourd_repeats = safe("4D repeat winners", lambda: analysis_fourd.repeat_winners(fourd), [])
        ctx.fourd_digit_sets = safe("4D digit sets", lambda: analysis_fourd.digit_set_freq(fourd), [])
        ctx.fourd_chi = safe("The 4D fairness test", lambda: analysis_fourd.chi_square_digits(fourd), {})
        seed = int(fourd["draw_number"].max()) + 1
        ctx.fourd_picks = safe("4D suggestions", lambda: strategies.fourd_picks(fourd, seed=seed), [])
        if ctx.fourd_picks:
            ctx.fourd_plan = safe("The 4D plan", lambda: suggest.fourd_plan(
                ctx.fourd_picks, settings.fourd_budget, ctx.fourd_bet_values or None))
    return ctx


# Step 4: backtests


def _backtest_key(game: str, df: pd.DataFrame, settings: Settings, rules) -> dict:
    return {
        "version": BACKTEST_CACHE_VERSION,
        "format": backtest.RESULT_FORMAT,
        "game": game,
        "latest_draw": _latest(df),
        "first_draw": int(df["draw_number"].min()) if len(df) else None,
        "draws_stored": int(len(df)),
        "backtest_draws": int(settings.backtest_draws),
        "random_sets_per_draw": int(settings.random_sets_per_draw),
        "rules": _rules_fingerprint(rules),
    }


def _read_backtest_cache(vault: Vault) -> dict:
    try:
        data = json.loads(vault.backtest_cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Backtest cache could not be read, it will be rebuilt: %s", exc)
        return {}


def run_backtests(ctx: Context, vault: Vault, activity: _Activity | None = None) -> None:
    """Set ctx.toto_backtest / ctx.fourd_backtest, reusing Data/backtest_cache.json when the
    latest draw, the backtest settings and the prize rules are unchanged."""
    cache = _read_backtest_cache(vault)
    changed = False
    jobs = (("toto", ctx.toto, backtest.backtest_toto), ("4d", ctx.fourd, backtest.backtest_fourd))
    for game, df, fn in jobs:
        if df is None or df.empty:
            continue
        key = _backtest_key(game, df, ctx.settings, ctx.rules)
        entry = cache.get(game)
        result = None
        if isinstance(entry, dict) and entry.get("key") == key:
            try:
                result = backtest.result_from_dict(entry["result"])
                if activity:
                    activity(EV_BACKTEST, f"{LABELS[game]} backtest reused from the cache (draw {key['latest_draw']})")
            except Exception as exc:
                log.warning("Cached %s backtest could not be read: %s", LABELS[game], exc)
                result = None
        if result is None:
            started = time.monotonic()
            try:
                result = fn(df, ctx.rules, n_draws=int(ctx.settings.backtest_draws),
                            n_random=int(ctx.settings.random_sets_per_draw))
            except Exception as exc:
                log.exception("%s backtest failed", LABELS[game])
                ctx.warnings.append(f"The {LABELS[game]} backtest could not be run ({type(exc).__name__}).")
                if activity:
                    activity(EV_ERROR, f"{LABELS[game]} backtest failed: {_error_text(exc)}")
                continue
            cache[game] = {"key": key, "result": backtest.result_to_dict(result),
                           "computed_at": ctx.now.isoformat(timespec="seconds")}
            changed = True
            if activity:
                activity(EV_BACKTEST, f"{LABELS[game]} backtest over {plural(result.draws_tested, 'draw')} with "
                                      f"{plural(result.random_sets_per_draw, 'random player')} took "
                                      f"{plural(round(time.monotonic() - started), 'second')}")
        if game == "toto":
            ctx.toto_backtest = result
        else:
            ctx.fourd_backtest = result
    if changed:
        try:
            store.atomic_write_text(vault.backtest_cache_path, json.dumps(cache, indent=1, sort_keys=True) + "\n")
        except OSError as exc:
            log.warning("Backtest cache could not be written: %s", exc)


# Step 5: tickets


def check_tickets(ctx: Context, vault: Vault, toto_all: pd.DataFrame, fourd_all: pd.DataFrame, *,
                  save: bool = True, activity: _Activity | None = None) -> None:
    """Read Tickets.md, sync and settle the ledger against every stored draw, save ledger.csv.

    Sets ctx.ledger, ctx.ledger_totals, ctx.settled_this_run and ctx.bad_ticket_lines.
    """
    text = vault.read_text(TICKETS_NOTE) or ""
    parsed = tickets.parse_tickets(text)
    bad = [t for t in parsed if t.error]
    old = store.load_ledger(vault.ledger_csv)
    ledger = tickets.sync_ledger(old, parsed, ctx.now)
    added = len(ledger) - len(old)
    ledger, settled = tickets.settle_ledger(ledger, toto_all, fourd_all, ctx.rules, ctx.now)
    ctx.ledger = ledger
    ctx.ledger_totals = tickets.ledger_totals(ledger)
    ctx.settled_this_run = settled
    ctx.bad_ticket_lines = bad

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
        activity(EV_LEDGER, f"{r.get('game')} {fmt_date(r.get('draw_date'))} {r.get('numbers')} {r.get('bet_type')} "
                            f"{dollars(r.get('cost'))}: {result}" + (f", won {dollars(won)}" if won else ""))
    if len(settled) > LEDGER_LOG_LIMIT:
        activity(EV_LEDGER, f"and {plural(len(settled) - LEDGER_LOG_LIMIT, 'more ticket')} checked")
    if added or settled:
        totals = ctx.ledger_totals
        activity(EV_LEDGER, f"{plural(max(added, 0), 'new ticket')}, {plural(len(settled), 'ticket')} checked; "
                            f"all tickets: spent {dollars(totals.get('spent'))}, won "
                            f"{dollars(totals.get('won'))}, net {dollars(totals.get('net'))}")


# Step 6 helpers


def _log_suggestions(ctx: Context, activity: _Activity) -> None:
    for plan, game in ((ctx.toto_plan, "toto"), (ctx.fourd_plan, "4d")):
        if plan is None:
            continue
        nd = report.next_draw(ctx, game)
        draw = f" draw {nd.number}" if nd.number else ""
        if plan.lines:
            picks = ", ".join(f"{ln.label} {ln.numbers}" for ln in plan.lines)
            activity(EV_SUGGEST, f"{LABELS[game]}{draw}: {picks}; total {money(plan.total)} of the "
                                 f"{money(plan.budget)} budget")
        else:
            activity(EV_SUGGEST, f"{LABELS[game]}{draw}: nothing to buy ({_plain(' '.join(plan.notes[:1]))})")
    sig = ctx.buy_signal
    if sig is not None:
        jackpot = money(sig.jackpot) if sig.jackpot is not None else "not known"
        ev = f", return per $1 about {per_dollar(sig.ev_per_dollar)}" if sig.ev_per_dollar is not None else ""
        activity(EV_SIGNAL, f"Buy signal {sig.label} for the next TOTO draw: jackpot {jackpot}, "
                            f"{sig.draw_type} draw{ev}")


def _commentary(ctx: Context, demo: bool, activity: _Activity | None = None) -> str | None:
    """The optional commentary. When COMMENTARY is turned on but nothing usable came back, an
    ERROR row says so (the container log has the reason)."""
    if demo:
        return None  # a demo never calls out to claude
    try:
        text = commentary.build_commentary(commentary.figures_from_context(ctx))
        reason = "the container log says why"
    except Exception as exc:  # optional extra, never fatal
        log.warning("Commentary skipped: %s", exc)
        text, reason = None, _error_text(exc)
    if text is None and activity is not None and commentary.is_enabled():
        activity(EV_ERROR, f"Commentary is turned on but none was added in this run ({reason})")
    return text


def _has_new(ctx: Context, state: dict) -> bool:
    """True when a reported game has a newer draw than the last one posted."""
    posted = state.get("last_posted") or {}
    for game in ctx.games_drawn:
        latest = _latest(ctx.toto if game == "toto" else ctx.fourd)
        if latest is None:
            continue
        try:
            done = int(posted.get(game))
        except (TypeError, ValueError):
            return True
        if latest > done:
            return True
    return False


def _posted_text(ctx: Context) -> str:
    return _and(f"{LABELS[g]} {_latest(ctx.toto if g == 'toto' else ctx.fourd)}" for g in ctx.games_drawn)


# Step 7


def _deliver(messages: list[str], ctx: Context, state: dict, *, dry_run: bool, post: bool, force_post: bool,
             demo: bool, out: Callable[[str], Any], activity: _Activity,
             vault: Vault | None = None) -> tuple[bool, bool]:
    """Post, print or skip the messages. Returns (posted, ok). A set that was only partly
    posted before (state["posting"], same draws) resumes after the last message that went out."""
    count = plural(len(messages), "message")
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
        activity(EV_POST, f"Nothing new since the last post ({_posted_text(ctx)}), not posted again")
        return False, True

    token, chat_id = telegram.config_from_env()
    # state["posting"] records how many messages of this set already went out, saved after each
    # one, so a set that failed half way is finished on the next run instead of posted again.
    draws = {g: _latest(ctx.toto if g == "toto" else ctx.fourd) for g in ctx.games_drawn}
    progress = state.get("posting") if isinstance(state.get("posting"), dict) else {}
    done = 0
    if progress.get("draws") == draws:
        try:
            done = max(0, min(int(progress.get("sent") or 0), len(messages)))
        except (TypeError, ValueError):
            done = 0

    def sent(i: int) -> None:
        state["posting"] = {"draws": draws, "sent": done + i, "total": len(messages)}
        if vault is not None:
            try:
                vault.save_state(state)
            except Exception as exc:  # the final save in run() tries again
                log.warning("state.json could not be saved after a message was posted: %s", exc)

    try:
        if done < len(messages):
            telegram.post_messages(messages[done:], token, chat_id, dry_run=False, out=out, on_sent=sent)
    except Exception as exc:
        log.exception("Posting to Telegram failed")
        msg = f"Posting to Telegram failed: {_error_text(exc)}"
        if done:
            msg += f" ({plural(done, 'message')} had already gone out in an earlier run)"
        ctx.warnings.append(msg + ".")
        activity(EV_ERROR, msg)
        return False, False
    state.pop("posting", None)
    posted = state.setdefault("last_posted", {})
    for game in ctx.games_drawn:
        latest = _latest(ctx.toto if game == "toto" else ctx.fourd)
        if latest is not None:
            posted[game] = latest
    state["last_posted_at"] = ctx.now.isoformat(timespec="seconds")
    if done:
        activity(EV_POST, f"Posted the remaining {plural(len(messages) - done, 'message')} to Telegram "
                          f"({_posted_text(ctx)}); the first {plural(done, 'message')} went out in an "
                          "earlier run, so they were not posted again")
    else:
        activity(EV_POST, f"Posted {count} to Telegram ({_posted_text(ctx)})")
    return True, True


# The run


def _settings_text(s: Settings) -> str:
    return (f"Read [[Settings]]: TOTO budget {dollars(s.toto_budget)}, "
            f"4D budget {dollars(s.fourd_budget)}, jackpot alert "
            f"{money(s.jackpot_alert)}, backtest over {plural(s.backtest_draws, 'draw')}")


def _run_flags(dry_run: bool, fetch: bool, post: bool, demo: bool, force_post: bool) -> str:
    flags = []
    if demo:
        flags.append("demo with synthetic data")
    else:
        if dry_run:
            flags.append("dry run")
        if not fetch:
            flags.append("no fetch")
        if not post:
            flags.append("no posting")
        if force_post:
            flags.append("manual")
        else:
            flags.append("scheduled")
    return f" ({', '.join(flags)})" if flags else ""


def run(games=GAMES, *, dry_run: bool = False, fetch: bool = True, post: bool = True, demo: bool = False,
        vault: Vault | str | Path | None = None, fetcher=None, now: datetime | None = None,
        out: Callable[[str], Any] = print, force_post: bool = False) -> RunResult:
    """Do one full run (see the module docstring for the steps). Never raises for expected
    problems: a network failure is a warning, a fatal problem gives ``RunResult(ok=False)``.

    games       games to fetch and report ("toto", "4d", "both" or a tuple)
    dry_run     print the 3 messages instead of posting them
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
    games = normalise_games(games)
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
        activity(EV_RUN, f"Run started for {_games_text(games)}{_run_flags(dry_run, fetch, post, demo, force_post)}")
        templates = {"Settings.md": SETTINGS_TEMPLATE} if demo else dict(TEMPLATES)
        for rel in vault.ensure_layout(templates):
            activity(EV_NOTE, f"Created {_link(rel)} (starter note)")
        settings = load_settings(vault)
        activity(EV_SETTINGS, _settings_text(settings))
        for w in settings.warnings:
            activity(EV_SETTINGS, w)
        state = vault.load_state()

        if not demo and post and not dry_run:
            token, chat_id = telegram.config_from_env()
            if not token or not chat_id:
                msg = ("Telegram is not set up (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing), so this is a "
                       "dry run: the messages are printed, not posted.")
                if Path(".env").is_file():
                    msg += (" A .env file is in the working folder, but only docker compose reads it: "
                            "outside Docker, export its values first (see the README).")
                warnings.append(msg)
                log.warning(msg)
                activity(EV_DRY_RUN, "Telegram is not set up, treating this run as a dry run")
                dry_run = True

        # 2 prize rules and data
        if demo:
            warnings.append(DEMO_WARNING)
            rules = prize_rules.load_prize_rules(None)
            data = _demo_data(vault, settings, now, activity, warnings)
        else:
            if fetch:
                if fetcher is None:
                    fetcher = own_fetcher = Fetcher()
                rules = prize_rules.load_prize_rules(fetcher, cache_path=vault.prize_rules_path,
                                                     max_age_days=PRIZE_RULES_MAX_AGE_DAYS, now=now)
                activity(EV_FETCH, _rules_text(rules, now))
                data = _update_from_site(vault, settings, state, games, fetcher, now, activity, warnings)
            else:
                rules = _cached_rules(vault)
                data = _Data(toto=store.load_toto(vault.toto_csv), fourd=store.load_fourd(vault.fourd_csv))
                activity(EV_FETCH, f"Fetching is off for this run, using the stored data "
                                   f"({plural(len(data.toto), 'TOTO draw')}, {plural(len(data.fourd), '4D draw')})")

        empty = [g for g in GAMES if (data.toto if g == "toto" else data.fourd).empty]
        if empty:
            reasons = "; ".join(data.problems[g] for g in empty if g in data.problems)
            if not fetch:
                reasons = reasons or "fetching was turned off"
            text = (f"Huat Bot could not run: no {_games_text(empty)} results are stored yet"
                    + (f" and the Singapore Pools site could not be used ({reasons})" if reasons else "")
                    + ". It will try again on the next run.")
            text = _plain(text)
            warnings.append(text)
            activity(EV_ERROR, text)
            if post and not dry_run and not demo:
                notify(text, out=out)
            result.new_draws = data.new_draws
            return result

        # 3 next draws and analysis
        if demo:
            nt, nf = synth.synth_next_toto(data.toto), synth.synth_next_fourd(data.fourd)
        else:
            nt = nf = None
            if fetch:
                nt, nf = site.fetch_next_draws(fetcher, data.toto)
                _store_next_draws(state, nt, nf, now)
                activity(EV_FETCH, f"Next draws: {_next_draws_text(nt, nf)}")
            stored_nt, stored_nf = next_draws_from_state(state, data.toto, data.fourd)
            nt, nf = nt or stored_nt, nf or stored_nf
            last_toto = _latest_date(data.toto)
            if nt is not None and nt.draw_datetime is not None and last_toto and nt.draw_datetime.date() <= last_toto:
                warnings.append(f"The next TOTO draw page still shows the draw on {fmt_date(nt.draw_datetime)}, "
                                "so the jackpot estimate may be out of date.")
        if demo:
            _store_next_draws(state, nt, nf, now)

        toto, fourd = _analysis_frames(data, settings)
        drawn = tuple(g for g in games if not (toto if g == "toto" else fourd).empty)
        known = len(warnings)
        ctx = build_context(toto, fourd, settings, rules, now=now, next_toto=nt, next_fourd=nf,
                            games_drawn=drawn, new_draws=data.new_draws, warnings=warnings)
        warnings = ctx.warnings  # one list from here on, so later steps add to the report too
        for w in warnings[known:]:  # analysis parts that failed (each is already a warning)
            activity(EV_ERROR, _plain(w).rstrip("."))
        _log_suggestions(ctx, activity)

        # 4 backtests
        run_backtests(ctx, vault, activity)

        # 5 tickets
        if demo:
            for rel in vault.ensure_layout({TICKETS_NOTE: demo_tickets_note(data.toto, data.fourd, nt)}):
                activity(EV_NOTE, f"Created {_link(rel)} with demo tickets")
        check_tickets(ctx, vault, data.toto, data.fourd, save=True, activity=activity)

        # 6 commentary, report, notes, state
        ctx.commentary = _commentary(ctx, demo, activity)
        if ctx.commentary:
            activity(EV_NOTE, "Added a short commentary written from the computed figures")
        report_md = report.full_report(ctx)
        notes.write_all(vault, ctx, report_md)
        result.report_path = str(vault.path(notes.report_note_path(ctx.now)))
        messages = report.telegram_messages(ctx)
        result.messages = messages

        # 7 telegram
        posted, ok = _deliver(messages, ctx, state, dry_run=dry_run, post=post, force_post=force_post,
                              demo=demo, out=out, activity=activity, vault=vault)
        result.posted = posted
        result.ok = ok
        result.new_draws = {k: list(v) for k, v in data.new_draws.items()}
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
            "games": list(games),
            "ok": result.ok,
            "posted": result.posted,
            "dry_run": bool(dry_run),
            "demo": bool(demo),
            "new_draws": {k: len(v) for k, v in result.new_draws.items()},
        }
        try:
            vault.save_state(state)
        except Exception as exc:
            log.warning("state.json could not be saved: %s", exc)
            activity(EV_ERROR, f"Data/state.json could not be saved ({_error_text(exc)})")
        result.warnings = list(dict.fromkeys(_plain(w) for w in warnings if w))
        new_total = sum(len(v) for v in result.new_draws.values())
        report_link = f", report {_link(result.report_path)}" if result.report_path else ""
        activity(EV_RUN, f"Run {'finished' if result.ok else 'ended with a problem'} in "
                         f"{plural(round(time.monotonic() - started), 'second')}: {plural(new_total, 'new draw')}"
                         f"{', posted' if result.posted else ''}{report_link}")


# Other entry points used by the CLI


def fetch_data(games=GAMES, *, vault: Vault | str | Path | None = None, fetcher=None,
               now: datetime | None = None) -> RunResult:
    """Update the CSVs and the next draw info only (no analysis, notes or posting)."""
    games = normalise_games(games)
    when = _aware(now) if now is not None else None
    now = _aware(now)
    vault = resolve_vault(vault)
    activity = _Activity(vault, when)
    for rel in vault.ensure_layout(dict(TEMPLATES)):
        activity(EV_NOTE, f"Created {_link(rel)} (starter note)")
    settings = load_settings(vault)
    state = vault.load_state()
    warnings: list[str] = []
    own = None
    if fetcher is None:
        fetcher = own = Fetcher()
    try:
        activity(EV_RUN, f"Fetch started for {_games_text(games)}")
        data = _update_from_site(vault, settings, state, games, fetcher, now, activity, warnings)
        nt, nf = site.fetch_next_draws(fetcher, data.toto)
        _store_next_draws(state, nt, nf, now)
        activity(EV_FETCH, f"Next draws: {_next_draws_text(nt, nf)}")
        vault.save_state(state)
    finally:
        if own is not None:
            own.close()
    failed = [g for g in games if g in data.problems]
    return RunResult(ok=not failed, new_draws=data.new_draws, warnings=[_plain(w) for w in warnings])


def build_report(vault: Vault | str | Path | None = None, *, now: datetime | None = None,
                 games=GAMES) -> str | None:
    """The full markdown report from the stored data only: no fetching, posting or note writing
    (the backtest cache may be filled). None when no data is stored yet."""
    vault = resolve_vault(vault)
    now = _aware(now)
    settings = load_settings(vault)
    state = vault.load_state()
    data = _Data(toto=store.load_toto(vault.toto_csv), fourd=store.load_fourd(vault.fourd_csv))
    if data.toto.empty or data.fourd.empty:
        return None
    rules = _cached_rules(vault)
    nt, nf = next_draws_from_state(state, data.toto, data.fourd)
    toto, fourd = _analysis_frames(data, settings)
    ctx = build_context(toto, fourd, settings, rules, now=now, next_toto=nt, next_fourd=nf,
                        games_drawn=normalise_games(games), warnings=[])
    run_backtests(ctx, vault, None)
    check_tickets(ctx, vault, data.toto, data.fourd, save=False, activity=None)
    return report.full_report(ctx)
