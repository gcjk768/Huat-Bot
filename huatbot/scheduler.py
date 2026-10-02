"""When to run: draw days, the 7.30pm run, and waiting for results to be published.

TOTO draws on Monday and Thursday, 4D on Wednesday, Saturday and Sunday, at 6.30pm
Singapore time, plus special draws on other days (read from the next draw pages and kept
in the vault's state.json under ``upcoming_draws``, see ``remember_upcoming``). The bot runs
at 7.30pm on a draw day (or an hour after a draw announced for later that evening); if the
new result is not on the site yet it checks again every 10 minutes for up to 2 hours. Each
game is run as soon as its result is out, and run again at the next check while the run
did not store or post it.

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
# state.json key holding every announced draw date that is not past yet (see remember_upcoming).
UPCOMING_KEY = "upcoming_draws"

MAX_SLEEP_CHUNK = 300.0  # sleep at most 5 minutes at a time, so a clock jump or NAS sleep is noticed
MAX_RETRY_HOURS = 12.0  # keeps one cycle (19:30 plus the window) clear of the next day's run
RESULT_DELAY = timedelta(hours=1)  # results are on the site about an hour after the draw (6.30pm, 7.30pm)

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


def _game_dates(state: Mapping | None, section: str, game: str) -> set[date]:
    """Dates for a game in ``state[section]`` (any of the game's key spellings)."""
    if not isinstance(state, Mapping):
        return set()
    entries = state.get(section)
    if not isinstance(entries, Mapping):
        return set()
    out: set[date] = set()
    for key in _STATE_KEYS[game]:
        out |= _dates_in(entries.get(key))
    return out


def _next_draw_dates(state: Mapping | None, game: str) -> set[date]:
    """Draw date(s) known for a game: ``state["next_draws"]`` (what the next draw page showed
    last) plus ``state["upcoming_draws"]`` (every announced date not yet past)."""
    return _game_dates(state, "next_draws", game) | _game_dates(state, UPCOMING_KEY, game)


def _checked_on(state: Mapping) -> date | None:
    """Day of the newest check recorded in the state (next draw pages read, or a run)."""
    stamps = []
    next_draws = state.get("next_draws")
    if isinstance(next_draws, Mapping):
        stamps.append(next_draws.get("checked_at"))
    last_run = state.get("last_run")
    if isinstance(last_run, Mapping):
        stamps.append(last_run.get("at"))
    days = [d for d in (_as_date(s) for s in stamps) if d is not None]
    return max(days) if days else None


def remember_upcoming(state: dict, previous: Mapping | None = None) -> dict:
    """Keep every announced draw date in ``state["upcoming_draws"]`` (``{"toto": [ISO dates],
    "4d": [...]}``) and return ``state``.

    The next draw page only ever shows one date per game and every refresh replaces it, so a
    special draw announced on Thursday for Friday is gone from ``next_draws`` once the page
    moves on to Monday (Friday's draw is at 6.30pm, before the 7.30pm run). The dates in
    ``previous`` (the state being replaced) and in ``state`` are merged with the ones already
    kept; dates before the day of the newest check in ``state`` are dropped.
    """
    if not isinstance(state, dict):
        return state
    today = _checked_on(state)
    upcoming: dict[str, list[str]] = {}
    for game in GAMES:
        dates = _next_draw_dates(previous, game) | _next_draw_dates(state, game)
        if today is not None:
            dates = {d for d in dates if d >= today}
        if dates:
            upcoming[game] = [d.isoformat() for d in sorted(dates)]
    if upcoming or UPCOMING_KEY in state:
        state[UPCOMING_KEY] = upcoming
    return state


# Public functions


def games_on(d: date, state: dict | None) -> tuple[str, ...]:
    """Games drawn on day ``d``, in the order ("toto", "4d").

    A game counts when ``d`` is one of its regular weekdays, or when ``d`` is a draw date
    recorded for it in ``state["next_draws"]`` or ``state["upcoming_draws"]`` (special draws on
    other days). State values are ISO strings such as ``"2026-10-09T18:30:00+08:00"`` (dates,
    lists of them, or dicts holding them also work).
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


def _result_ready(check: Callable[[str], Any], game: str, d: date,
                  reached: dict[str, bool] | None = None) -> bool:
    """Ask ``check`` for the latest draw date on the site. A failing check counts as not ready.
    A check that did not raise reached the site: ``reached[game]`` is then set to True."""
    try:
        latest = _as_date(check(game))
    except Exception as exc:  # network trouble: just try again at the next check
        log.warning("Could not check the latest %s draw: %s", GAME_LABELS.get(game, game), exc)
        return False
    if reached is not None:
        reached[game] = True
    # The draw of day d is out when the site's latest draw is d (a later date also means it is out).
    return latest is not None and latest >= d


def wait_for_results(games: Iterable[str], d: date, check: Callable[[str], Any],
                     now_fn: Callable[[], datetime], sleep_fn: Callable[[float], Any],
                     retry_minutes: float = C.DEFAULT_RETRY_MINUTES,
                     retry_hours: float = C.DEFAULT_RETRY_HOURS, *,
                     on_ready: Callable[[tuple[str, ...], dict[str, datetime | None]], Iterable[str]] | None = None,
                     start_at: Mapping[str, datetime] | None = None,
                     reached: dict[str, bool] | None = None) -> dict[str, bool]:
    """Wait until each game's result for day ``d`` is on the site, or the window runs out.

    The first check is immediate. Checks repeat every ``retry_minutes`` (timed from the first
    check, so slow checks do not make the spacing drift) while they fall within
    ``retry_hours`` of the first check; with the defaults that is 13 checks at 0, 10, ... 120
    minutes. A game that is ready is not checked again. Returns ``{game: ready}``.

    ``start_at`` (optional, per game) delays a game's first check, for a draw held later in
    the evening; its window then runs from that time. ``on_ready(games, retry_at)`` (optional)
    is called as soon as a check finds results out, so a game is handled at once instead of
    waiting for the others. It returns the games it finished; a game it did not finish (the
    run could not store or post it) is passed to it again at that game's next check, while
    the window lasts. ``retry_at`` gives each game's next check time, or None after the last.
    ``reached`` (optional) is filled with ``{game: True}`` once a check of that game got an
    answer from the site (``check`` did not raise), else False.
    """
    games = tuple(games)
    status = {g: False for g in games}
    if reached is not None:
        for g in games:
            reached.setdefault(g, False)
    if not games:
        return status
    first = _to_sg(now_fn())
    window = timedelta(hours=max(float(retry_hours), 0.0))
    interval = timedelta(minutes=float(retry_minutes)) if retry_minutes and retry_minutes > 0 else None
    start = {g: max(first, _to_sg(start_at[g])) if start_at and start_at.get(g) else first for g in games}
    next_at = dict(start)
    checks = {g: 0 for g in games}
    active = set(games)

    def following(g: str) -> datetime | None:
        """Time of the check after the current one, or None when the window is over."""
        if interval is None:
            return None
        nxt = start[g] + interval * (checks[g] + 1)
        return nxt if nxt <= start[g] + window else None

    while active:
        sleep_until(min(next_at[g] for g in active), now_fn, sleep_fn)
        now = _to_sg(now_fn())
        due = [g for g in games if g in active and next_at[g] <= now]
        for g in due:
            if not status[g]:
                status[g] = _result_ready(check, g, d, reached)
        out = tuple(g for g in due if status[g])
        finished = set(out)
        if out and on_ready is not None:
            try:
                finished = set(on_ready(out, {g: following(g) for g in out}) or ()) & finished
            except Exception:  # a broken hook must not end the wait: try again at the next check
                log.exception("Handling the %s result failed", " and ".join(GAME_LABELS.get(g, g) for g in out))
                finished = set()
        for g in due:
            nxt = following(g)
            checks[g] += 1
            label = GAME_LABELS.get(g, g)
            if g in finished:
                log.info("%s result for %s is out (check %d)", label, _fmt_day(d), checks[g])
                active.discard(g)
            elif nxt is None:
                log.info("Gave up on the %s result for %s after %d checks", label, _fmt_day(d), checks[g])
                active.discard(g)
            else:
                next_at[g] = nxt
                if not status[g]:
                    log.info("%s result for %s not out yet (check %d), next check at %s",
                             label, _fmt_day(d), checks[g], _fmt_clock(nxt.time()))
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


def _names(games: Iterable[str]) -> str:
    return " and ".join(GAME_LABELS.get(g, g) for g in games)


def _missing_message(missing: tuple[str, ...], day: date, config: SchedulerConfig) -> str:
    """Plain, dash free notice that a result never appeared within the retry window. The
    next run of that game stores the draw, but message 1 only shows the newest draw, so the
    notice does not promise a post."""
    one = len(missing) == 1
    return (
        f"{_names(missing)} {'result' if one else 'results'} for {_fmt_day(day)} "
        f"{'is' if one else 'are'} still not on the Singapore Pools site after "
        f"{_fmt_window(config.retry_minutes, config.retry_hours)}, so "
        f"{'it was' if one else 'they were'} not posted. {'It' if one else 'They'} will be stored "
        f"on the next {_names(missing)} {'run' if one else 'runs'} but not posted."
    )


def _unreachable_message(games: tuple[str, ...], day: date, config: SchedulerConfig) -> str:
    """Plain, dash free notice that no check within the retry window reached the site, so it is
    not known whether the result is out."""
    one = len(games) == 1
    return (
        f"{_names(games)} {'result' if one else 'results'} for {_fmt_day(day)} could not be checked: the "
        f"Singapore Pools site could not be reached after {_fmt_window(config.retry_minutes, config.retry_hours)}, "
        f"so {'it was' if one else 'they were'} not posted. {'It' if one else 'They'} will be stored on the "
        f"next {_names(games)} {'run' if one else 'runs'} but not posted."
    )


def _failed_message(failed: tuple[str, ...], day: date, config: SchedulerConfig, posting: bool = False) -> str:
    """Plain, dash free notice that a result is out but no run managed to store or post it.
    ``posting``: the last run built the messages and only posting them failed."""
    one = len(failed) == 1
    what = "could not be posted to Telegram" if posting else "could not be fetched or posted"
    return (
        f"{_names(failed)} {'result' if one else 'results'} for {_fmt_day(day)} "
        f"{'is' if one else 'are'} on the Singapore Pools site but {what} "
        f"after {_fmt_window(config.retry_minutes, config.retry_hours)}. See the activity log."
    )


def _post_failed(result: Any) -> bool:
    """True when a run got as far as building the messages (so the draw data was used) and
    then failed: only posting them went wrong."""
    return (result is not _FAILED and not bool(getattr(result, "ok", True))
            and bool(getattr(result, "messages", None)))


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


def _times_in(value: Any) -> list[datetime]:
    """Draw times (Singapore time) in a state entry. Values without a time of day are skipped."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [t for key in ("draw_datetime", "datetime") if key in value for t in _times_in(value[key])]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [t for item in value for t in _times_in(item)]
    if isinstance(value, datetime):
        return [] if value != value else [_to_sg(value)]
    if isinstance(value, str) and len(value.strip()) > 10:
        try:
            return [_to_sg(datetime.fromisoformat(value.strip()))]
        except ValueError:
            return []
    return []


def _draw_times(state: Mapping | None, day: date) -> dict[str, datetime]:
    """Announced draw time on ``day`` for each game, from ``state["next_draws"]``."""
    next_draws = state.get("next_draws") if isinstance(state, Mapping) else None
    out: dict[str, datetime] = {}
    if not isinstance(next_draws, Mapping):
        return out
    for game in GAMES:
        for key in _STATE_KEYS[game]:
            for t in _times_in(next_draws.get(key)):
                if t.date() == day:
                    out[game] = t
    return out


_FAILED = object()  # run_fn raised


def _is_done(game: str, day: date, result: Any, done_fn: Callable[[str, date], Any] | None) -> bool:
    """Did the run store (and, when posting is on, post) ``game``'s draw of ``day``?"""
    if done_fn is not None:
        try:
            return bool(done_fn(game, day))
        except Exception:
            log.exception("Could not tell whether the %s run is done", GAME_LABELS.get(game, game))
            return False
    return result is not _FAILED and bool(getattr(result, "ok", True))


def _run_cycle(day: date, known: dict[str, set[date]], run_fn, check_fn, refresh_fn, notify_fn, log_fn,
               now_fn, sleep_fn, config: SchedulerConfig, done_fn=None, only: tuple[str, ...] | None = None) -> None:
    """One scheduled run: refresh next draw dates, decide the games, then run each game as soon
    as its result is out, again every ``retry_minutes`` while the run did not finish it."""
    cycle_start = _to_sg(now_fn())
    try:
        state = refresh_fn() or {}
    except Exception as exc:
        log.exception("Could not refresh the next draw dates")
        _safe_call(log_fn, "ERROR",
                   f"Could not refresh the next draw dates ({_error_text(exc)}), using the regular draw days")
        state = {}
    _remember_dates(known, state, day)
    games = games_on(day, {"next_draws": {g: sorted(ds) for g, ds in known.items()}})
    if only is not None:
        games = tuple(g for g in games if g in only)

    if not games:
        followed = "" if only is None or set(GAMES) <= set(only) else f"{_names(only)} "
        _safe_call(log_fn, "SCHEDULE", f"No {followed}draw today")
        return

    _safe_call(log_fn, "SCHEDULE", f"{_names(games)} draw on {_fmt_day(day)}, checking for the result")
    # A draw announced for later than the usual 6.30pm (some special draws) is checked from an
    # hour after it; draws at the usual time keep the configured run time.
    start_at: dict[str, datetime] = {}
    for game, when in _draw_times(state, day).items():
        if game in games and when.time() > C.DRAW_TIME and when + RESULT_DELAY > cycle_start:
            start_at[game] = when + RESULT_DELAY
            _safe_call(log_fn, "SCHEDULE", f"{GAME_LABELS[game]} draw on {_fmt_day(day)} is at "
                                           f"{_fmt_clock(when.time())}, checking for its result from "
                                           f"{_fmt_clock(start_at[game].time())}")

    finished: set[str] = set()
    post_failed: dict[str, bool] = {}  # game -> its last run failed only at posting

    def on_ready(ready: tuple[str, ...], retry_at: dict[str, datetime | None]) -> set[str]:
        # A draw an earlier run already stored (and posted) is not run again, for example after
        # a restart inside the retry window.
        already = {g for g in ready if done_fn is not None and _is_done(g, day, None, done_fn)}
        if already:
            one = len(already) == 1
            _safe_call(log_fn, "SCHEDULE", f"{_names(g for g in ready if g in already)} "
                                           f"{'result' if one else 'results'} for {_fmt_day(day)} "
                                           f"{'was' if one else 'were'} already handled by an earlier run, so "
                                           f"{'it is' if one else 'they are'} not run again")
            finished.update(already)
        ready = tuple(g for g in ready if g not in already)
        if not ready:
            return already
        result: Any = _FAILED
        try:
            result = run_fn(games=ready)
        except Exception as exc:
            log.exception("Scheduled run failed")
            _safe_call(log_fn, "ERROR", f"Run for {_names(ready)} failed: {_error_text(exc)}")
        done = {g for g in ready if _is_done(g, day, result, done_fn)}
        finished.update(done)
        for g in ready:
            post_failed[g] = _post_failed(result)
        left = [g for g in ready if g not in done and retry_at.get(g) is not None]
        if left:
            one = len(left) == 1
            when = min(retry_at[g] for g in left)
            _safe_call(log_fn, "WAIT", f"{_names(left)} {'result' if one else 'results'} for {_fmt_day(day)} "
                                       f"{'is' if one else 'are'} out but {'was' if one else 'were'} not stored "
                                       f"or posted yet, trying again at {_fmt_clock(when.time())}")
        return done | already

    reached: dict[str, bool] = {}
    status = wait_for_results(games, day, check_fn, now_fn, sleep_fn, config.retry_minutes, config.retry_hours,
                              on_ready=on_ready, start_at=start_at, reached=reached)
    # A game no check could ask the site about is not known to be missing: the site was down.
    unreachable = tuple(g for g in games if not status.get(g) and not reached.get(g, True))
    missing = tuple(g for g in games if not status.get(g) and g not in unreachable)
    failed = tuple(g for g in games if status.get(g) and g not in finished)

    if missing:
        message = _missing_message(missing, day, config)
        _safe_call(log_fn, "WAIT", message)
        _safe_call(notify_fn, message)
    if unreachable:
        message = _unreachable_message(unreachable, day, config)
        _safe_call(log_fn, "ERROR", message)
        _safe_call(notify_fn, message)
    for posting in (False, True):
        group = tuple(g for g in failed if post_failed.get(g, False) == posting)
        if group:
            message = _failed_message(group, day, config, posting)
            _safe_call(log_fn, "ERROR", message)
            _safe_call(notify_fn, message)


def serve(run_fn: Callable[..., Any], check_fn: Callable[[str], Any], refresh_fn: Callable[[], dict],
          notify_fn: Callable[[str], Any], log_fn: Callable[[str, str], Any],
          now_fn: Callable[[], datetime], sleep_fn: Callable[[float], Any],
          config: SchedulerConfig | None = None, max_cycles: int | None = None, *,
          done_fn: Callable[[str, date], bool] | None = None, games: Iterable[str] | None = None) -> None:
    """Run forever (or ``max_cycles`` times): sleep until the run time, then do one cycle.

    Each cycle: ``refresh_fn()`` updates and returns the state (next draw dates); the games of
    the day come from ``games_on`` (only those in ``games`` when it is given); with no draw it
    logs "No draw today". Otherwise ``wait_for_results`` checks the site and calls
    ``run_fn(games=...)`` as soon as a game's result is out. A game the run did not finish
    (``done_fn(game, day)`` is False, or without ``done_fn`` the run raised or returned
    ``ok=False``) is run again at each later check of the window, with a WAIT row. A game
    still missing at the end gets a WAIT row and a plain ``notify_fn`` message (an ERROR row
    saying the site could not be reached when every check of it raised); a game that was out
    but never finished gets an ERROR row and a message. An exception anywhere in a cycle is
    logged and the loop carries on.

    Hooks: ``check_fn(game) -> date | None`` (latest draw date on the site, None while it is
    not out; it should raise when the site cannot be reached), ``log_fn(event,
    message)`` (the vault activity log), ``notify_fn(text)`` (Telegram), ``done_fn(game, day)``
    (the draw of ``day`` is stored, and posted when posting is on).
    """
    config = config or SchedulerConfig()
    only = tuple(g for g in GAMES if g in set(games)) if games is not None else None
    known: dict[str, set[date]] = {}
    followed = f". Only {_names(only)} is followed" if only is not None and set(only) != set(GAMES) else ""
    _safe_call(log_fn, "SCHEDULE", f"Scheduler started. {config.describe()}{followed}")
    cycles = 0
    first = True
    while max_cycles is None or cycles < max_cycles:
        now = _to_sg(now_fn())
        target = _first_run_at(now, config) if first else next_run_at(now, config.run_time)
        first = False
        sleep_until(target, now_fn, sleep_fn)
        try:
            _run_cycle(target.date(), known, run_fn, check_fn, refresh_fn, notify_fn, log_fn,
                       now_fn, sleep_fn, config, done_fn, only)
        except Exception as exc:  # never let one bad day stop the scheduler
            log.exception("Scheduled cycle failed")
            _safe_call(log_fn, "ERROR", f"Scheduled run failed: {_error_text(exc)}")
        cycles += 1
