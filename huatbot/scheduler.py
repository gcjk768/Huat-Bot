"""When to run: draw days, the 7.30pm run, and waiting for results to be published.

TOTO draws on Monday and Thursday, 4D on Wednesday, Saturday and Sunday, at 6.30pm
Singapore time, plus special draws on other days (read from the next draw pages and kept
in the vault's state.json). The bot runs at 7.30pm on a draw day; if the new result is
not on the site yet it checks again every 10 minutes for up to 2 hours.

Every function takes its clock (``now_fn``) and sleep (``sleep_fn``) as arguments, so the
tests drive whole days with a fake clock and never sleep for real.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from . import constants as C

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)

GAMES = ("toto", "4d")
GAME_WEEKDAYS = {"toto": tuple(C.TOTO_WEEKDAYS), "4d": tuple(C.FOURD_WEEKDAYS)}
GAME_LABELS = {"toto": "TOTO", "4d": "4D"}
# Other spellings of the game keys that may appear in state.json.
_STATE_KEYS = {"toto": ("toto", "TOTO"), "4d": ("4d", "4D", "fourd")}

MAX_SLEEP_CHUNK = 300.0  # sleep at most 5 minutes at a time, so a clock jump or NAS sleep is noticed
MAX_RETRY_HOURS = 12.0  # keeps one cycle (19:30 plus the window) clear of the next day's run

_DAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# Time helpers


def _to_sg(dt: datetime) -> datetime:
    """Singapore time; a naive datetime is taken to already be Singapore time."""
    return dt.replace(tzinfo=SG) if dt.tzinfo is None else dt.astimezone(SG)


def _fmt_day(d: date) -> str:
    """Mon 5 Oct 2026"""
    return f"{_DAY_ABBR[d.weekday()]} {d.day} {_MONTH_ABBR[d.month - 1]} {d.year}"


def _fmt_clock(t: time) -> str:
    """7.30pm"""
    hour12 = t.hour % 12 or 12
    return f"{hour12}.{t.minute:02d}{'am' if t.hour < 12 else 'pm'}"


def _fmt_count(x: float, unit: str) -> str:
    """'10 minutes', '1 hour', '1.5 hours'."""
    text = f"{int(x)}" if float(x).is_integer() else f"{x:g}"
    return f"{text} {unit}" if x == 1 else f"{text} {unit}s"


def _fmt_window(retry_minutes: float, retry_hours: float) -> str:
    """Plain description of the retry window, e.g. 'checking every 10 minutes for 2 hours'."""
    if retry_hours <= 0 or retry_minutes <= 0:
        return "one check"
    if retry_hours < 1:
        span = _fmt_count(round(retry_hours * 60, 1), "minute")
    else:
        span = _fmt_count(retry_hours, "hour")
    return f"checking every {_fmt_count(retry_minutes, 'minute')} for {span}"


def _as_date(value: Any) -> date | None:
    """A date from a date, datetime, pandas Timestamp or ISO string. Aware datetimes are read in
    Singapore time. Anything unreadable gives None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value != value:  # pandas NaT
            return None
        return _to_sg(value).date() if value.tzinfo is not None else value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _as_date(datetime.fromisoformat(text))
        except ValueError:
            log.debug("Not an ISO date in state: %r", value)
            return None
    return None


def _dates_in(value: Any) -> set[date]:
    """Every date found in a state entry: one value, a list of values, or a dict with a
    draw_datetime / datetime / date / draw_date key."""
    if value is None:
        return set()
    if isinstance(value, dict):
        out: set[date] = set()
        for key in ("draw_datetime", "datetime", "draw_date", "date"):
            if key in value:
                out |= _dates_in(value[key])
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        out = set()
        for item in value:
            out |= _dates_in(item)
        return out
    d = _as_date(value)
    return {d} if d is not None else set()


def _next_draw_dates(state: Mapping | None, game: str) -> set[date]:
    """Next draw date(s) for a game from ``state["next_draws"]``."""
    if not isinstance(state, Mapping):
        return set()
    next_draws = state.get("next_draws")
    if not isinstance(next_draws, Mapping):
        return set()
    out: set[date] = set()
    for key in _STATE_KEYS[game]:
        out |= _dates_in(next_draws.get(key))
    return out


# Public functions


def games_on(d: date, state: dict | None) -> tuple[str, ...]:
    """Games drawn on day ``d``, in the order ("toto", "4d").

    A game counts when ``d`` is one of its regular weekdays, or when ``d`` is the next draw
    date recorded for it in ``state["next_draws"]`` (special draws on other days). State
    values are ISO strings such as ``"2026-10-09T18:30:00+08:00"`` (dates, lists of them, or
    dicts holding them also work).
    """
    if isinstance(d, datetime):
        d = _to_sg(d).date()
    return tuple(
        game for game in GAMES
        if d.weekday() in GAME_WEEKDAYS[game] or d in _next_draw_dates(state, game)
    )


def next_run_at(now: datetime, run_time: time) -> datetime:
    """Today at ``run_time`` (Singapore time) if that is still ahead of ``now``, else tomorrow."""
    local = _to_sg(now)
    wall = run_time.replace(tzinfo=None)
    today = datetime.combine(local.date(), wall).replace(tzinfo=SG)
    if today > local:
        return today
    return datetime.combine(local.date() + timedelta(days=1), wall).replace(tzinfo=SG)


def sleep_until(target: datetime, now_fn: Callable[[], datetime], sleep_fn: Callable[[float], Any],
                chunk: float = MAX_SLEEP_CHUNK) -> None:
    """Sleep in chunks of at most ``chunk`` seconds until ``now_fn()`` reaches ``target``.

    Re-reading the clock after every chunk keeps the schedule right after a NAS sleep or a
    clock change, instead of trusting one long sleep.
    """
    target = _to_sg(target)
    while True:
        remaining = (target - _to_sg(now_fn())).total_seconds()
        if remaining <= 0:
            return
        sleep_fn(min(remaining, chunk))


def _result_ready(check: Callable[[str], Any], game: str, d: date) -> bool:
    """Ask ``check`` for the latest draw date on the site. A failing check counts as not ready."""
    try:
        latest = _as_date(check(game))
    except Exception as exc:  # network trouble: just try again at the next check
        log.warning("Could not check the latest %s draw: %s", GAME_LABELS.get(game, game), exc)
        return False
    # The draw of day d is out when the site's latest draw is d (a later date also means it is out).
    return latest is not None and latest >= d


def wait_for_results(games: Iterable[str], d: date, check: Callable[[str], Any],
                     now_fn: Callable[[], datetime], sleep_fn: Callable[[float], Any],
                     retry_minutes: float = C.DEFAULT_RETRY_MINUTES,
                     retry_hours: float = C.DEFAULT_RETRY_HOURS) -> dict[str, bool]:
    """Wait until each game's result for day ``d`` is on the site, or the window runs out.

    The first check is immediate. Checks repeat every ``retry_minutes`` (timed from the first
    check, so slow checks do not make the spacing drift) while they fall within
    ``retry_hours`` of the first check; with the defaults that is 13 checks at 0, 10, ... 120
    minutes. A game that is ready is not checked again. Returns ``{game: ready}``.
    """
    games = tuple(games)
    status = {g: False for g in games}
    if not games:
        return status
    start = _to_sg(now_fn())
    deadline = start + timedelta(hours=max(float(retry_hours), 0.0))
    interval = timedelta(minutes=float(retry_minutes)) if retry_minutes and retry_minutes > 0 else None

    attempt = 0
    while True:
        attempt += 1
        for game in games:
            if not status[game]:
                status[game] = _result_ready(check, game, d)
        waiting = [GAME_LABELS.get(g, g) for g in games if not status[g]]
        if not waiting:
            log.info("Results for %s are out (check %d)", _fmt_day(d), attempt)
            break
        if interval is None:
            break
        next_at = start + interval * attempt
        if next_at > deadline:
            log.info("Gave up waiting for %s after %d checks", " and ".join(waiting), attempt)
            break
        log.info("%s result for %s not out yet (check %d), next check at %s",
                 " and ".join(waiting), _fmt_day(d), attempt, _fmt_clock(next_at.time()))
        sleep_until(next_at, now_fn, sleep_fn)
    return status


# Configuration


def parse_run_time(text: str) -> time:
    """Read a run time such as "19:30", "19.30", "1930", "7.30pm", "7:30 PM" or "7pm".
    Raises ValueError when it is not a valid time of day."""
    s = str(text).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?:[:.h]?(\d{2}))?(am|pm)?", s)
    if not m:
        raise ValueError(f"not a time of day: {text!r}")
    hour, minute, suffix = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if suffix:
        if not 1 <= hour <= 12:
            raise ValueError(f"not a time of day: {text!r}")
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"not a time of day: {text!r}")
    return time(hour, minute)


def _env_number(env: Mapping[str, str], name: str, default: float, low: float, high: float,
                warnings: list[str]) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        warnings.append(f"{name} is not a number, so the default {default:g} is used.")
        return default
    if value != value or value < low or value > high:
        clamped = min(max(value, low), high) if value == value else default
        warnings.append(f"{name} must be between {low:g} and {high:g}, so {clamped:g} is used.")
        return clamped
    return value


@dataclass
class SchedulerConfig:
    """When and how long to wait. ``catch_up``: if the scheduler starts during today's retry
    window (for example the NAS rebooted at 8pm on a draw day) it runs at once instead of
    waiting for tomorrow; the runner does not repost a draw it already posted."""

    run_time: time = C.DEFAULT_RUN_TIME
    retry_minutes: float = C.DEFAULT_RETRY_MINUTES
    retry_hours: float = C.DEFAULT_RETRY_HOURS
    catch_up: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SchedulerConfig":
        """Read RUN_AT ("19:30"), RETRY_MINUTES (10) and RETRY_HOURS (2). Bad values fall back to
        the defaults (or are clamped) with a plain warning in ``warnings``."""
        env = os.environ if env is None else env
        warnings: list[str] = []
        run_time = C.DEFAULT_RUN_TIME
        raw = (env.get("RUN_AT") or "").strip()
        if raw:
            try:
                run_time = parse_run_time(raw)
            except ValueError:
                warnings.append(f"RUN_AT is not a time like 19:30, so {_fmt_clock(run_time)} is used.")
        retry_minutes = _env_number(env, "RETRY_MINUTES", float(C.DEFAULT_RETRY_MINUTES), 1, 240, warnings)
        retry_hours = _env_number(env, "RETRY_HOURS", float(C.DEFAULT_RETRY_HOURS), 0, MAX_RETRY_HOURS, warnings)
        for w in warnings:
            log.warning("Scheduler: %s", w)
        return cls(run_time=run_time, retry_minutes=retry_minutes, retry_hours=retry_hours, warnings=warnings)

    def describe(self) -> str:
        """Plain summary for the activity log."""
        return (f"Runs at {_fmt_clock(self.run_time)} Singapore time on draw days, "
                f"{_fmt_window(self.retry_minutes, self.retry_hours)} until results are out")


# The loop


def _first_run_at(now: datetime, config: SchedulerConfig) -> datetime:
    """First cycle: now, when starting inside today's retry window (catch up), else the next run."""
    local = _to_sg(now)
    if config.catch_up:
        today_run = datetime.combine(local.date(), config.run_time.replace(tzinfo=None)).replace(tzinfo=SG)
        if today_run <= local < today_run + timedelta(hours=max(config.retry_hours, 0.0)):
            return local
    return next_run_at(local, config.run_time)


def _missing_message(missing: tuple[str, ...], day: date, config: SchedulerConfig) -> str:
    """Plain, dash free notice that a result never appeared within the retry window."""
    names = " and ".join(GAME_LABELS.get(g, g) for g in missing)
    one = len(missing) == 1
    return (
        f"{names} {'result' if one else 'results'} for {_fmt_day(day)} "
        f"{'is' if one else 'are'} still not on the Singapore Pools site after "
        f"{_fmt_window(config.retry_minutes, config.retry_hours)}, so "
        f"{'it was' if one else 'they were'} not posted. The next run will pick "
        f"{'it' if one else 'them'} up."
    )


def _error_text(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:300]


def _safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Call a log or notify hook without letting it break the loop."""
    try:
        fn(*args, **kwargs)
    except Exception:
        log.exception("Scheduler hook %s failed", getattr(fn, "__name__", fn))


def _remember_dates(known: dict[str, set[date]], state: Mapping | None, today: date) -> None:
    """Keep every upcoming draw date seen so far, so a special draw is not forgotten when the
    next draw page has already moved on to the following draw."""
    for game in GAMES:
        dates = known.setdefault(game, set())
        dates |= _next_draw_dates(state, game)
        dates -= {d for d in dates if d < today}


def _run_cycle(day: date, known: dict[str, set[date]], run_fn, check_fn, refresh_fn, notify_fn, log_fn,
               now_fn, sleep_fn, config: SchedulerConfig) -> None:
    """One scheduled run: refresh next draw dates, decide the games, wait for results, run."""
    try:
        state = refresh_fn() or {}
    except Exception as exc:
        log.exception("Could not refresh the next draw dates")
        _safe_call(log_fn, "ERROR",
                   f"Could not refresh the next draw dates ({_error_text(exc)}), using the regular draw days")
        state = {}
    _remember_dates(known, state, day)
    games = games_on(day, {"next_draws": {g: sorted(ds) for g, ds in known.items()}})

    if not games:
        _safe_call(log_fn, "SCHEDULE", "No draw today")
        return

    labels = " and ".join(GAME_LABELS[g] for g in games)
    _safe_call(log_fn, "SCHEDULE", f"{labels} draw on {_fmt_day(day)}, checking for the result")
    status = wait_for_results(games, day, check_fn, now_fn, sleep_fn, config.retry_minutes, config.retry_hours)
    ready = tuple(g for g in games if status.get(g))
    missing = tuple(g for g in games if not status.get(g))

    if ready:
        try:
            run_fn(games=ready)
        except Exception as exc:
            log.exception("Scheduled run failed")
            _safe_call(log_fn, "ERROR", f"Run for {' and '.join(GAME_LABELS[g] for g in ready)} failed: {_error_text(exc)}")

    if missing:
        message = _missing_message(missing, day, config)
        _safe_call(log_fn, "WAIT", message)
        _safe_call(notify_fn, message)


def serve(run_fn: Callable[..., Any], check_fn: Callable[[str], Any], refresh_fn: Callable[[], dict],
          notify_fn: Callable[[str], Any], log_fn: Callable[[str, str], Any],
          now_fn: Callable[[], datetime], sleep_fn: Callable[[float], Any],
          config: SchedulerConfig | None = None, max_cycles: int | None = None) -> None:
    """Run forever (or ``max_cycles`` times): sleep until the run time, then do one cycle.

    Each cycle: ``refresh_fn()`` updates and returns the state (next draw dates); the games of
    the day come from ``games_on``; with no draw it logs "No draw today". Otherwise it waits for
    the results with ``wait_for_results``, calls ``run_fn(games=ready)`` for the games whose
    result is out, and for any game still missing logs a WAIT row and sends ``notify_fn`` a plain
    message. An exception anywhere in a cycle is logged and the loop carries on.

    Hooks: ``check_fn(game) -> date | None`` (latest draw date on the site), ``log_fn(event,
    message)`` (the vault activity log), ``notify_fn(text)`` (Telegram).
    """
    config = config or SchedulerConfig()
    known: dict[str, set[date]] = {}
    _safe_call(log_fn, "SCHEDULE", f"Scheduler started. {config.describe()}")
    cycles = 0
    first = True
    while max_cycles is None or cycles < max_cycles:
        now = _to_sg(now_fn())
        target = _first_run_at(now, config) if first else next_run_at(now, config.run_time)
        first = False
        sleep_until(target, now_fn, sleep_fn)
        try:
            _run_cycle(target.date(), known, run_fn, check_fn, refresh_fn, notify_fn, log_fn,
                       now_fn, sleep_fn, config)
        except Exception as exc:  # never let one bad day stop the scheduler
            log.exception("Scheduled cycle failed")
            _safe_call(log_fn, "ERROR", f"Scheduled run failed: {_error_text(exc)}")
        cycles += 1
