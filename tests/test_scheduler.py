"""Scheduler: draw days, run time, waiting for results and the serve loop, all on a fake clock."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from huatbot import constants as C
from huatbot.scheduler import (
    GAMES,
    SG,
    UPCOMING_KEY,
    SchedulerConfig,
    games_on,
    next_run_at,
    parse_run_time,
    remember_upcoming,
    serve,
    sleep_until,
    wait_for_results,
)

DASHES = ("-", "–", "—")


class FakeClock:
    """A clock that only moves when something sleeps on it."""

    def __init__(self, start: datetime):
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.sleeps.append(seconds)
        self.t += timedelta(seconds=seconds)


def sg(y, m, d, hh=0, mm=0, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=SG)


# games_on


def test_only_toto_is_followed():
    assert GAMES == ("toto",)


@pytest.mark.parametrize("day,expected", [
    (date(2026, 10, 5), ("toto",)),   # Mon
    (date(2026, 10, 6), ()),          # Tue
    (date(2026, 10, 7), ()),          # Wed (was a 4D day)
    (date(2026, 10, 8), ("toto",)),   # Thu
    (date(2026, 10, 9), ()),          # Fri
    (date(2026, 10, 10), ()),         # Sat (was a 4D day)
    (date(2026, 10, 11), ()),         # Sun (was a 4D day)
])
def test_games_on_regular_weekdays(day, expected):
    assert games_on(day, {}) == expected
    assert games_on(day, None) == expected
    assert games_on(datetime.combine(day, time(19, 30), tzinfo=SG), None) == expected


def test_games_on_special_draw_dates_from_state():
    state = {"next_draws": {"toto": "2026-10-09T18:30:00+08:00"}}
    assert games_on(date(2026, 10, 9), state) == ("toto",)   # special TOTO on a Friday
    assert games_on(date(2026, 10, 8), state) == ("toto",)   # regular days still count
    assert games_on(date(2026, 10, 10), state) == ()
    assert games_on(date(2026, 10, 2), state) == ()


def test_games_on_reads_every_remembered_special_draw():
    state = {"next_draws": {"toto": "2026-10-12T18:30:00+08:00"},
             UPCOMING_KEY: {"toto": ["2026-10-09", "2026-10-13"]}}
    assert games_on(date(2026, 10, 9), state) == ("toto",)   # Fri, remembered after the page moved on
    assert games_on(date(2026, 10, 13), state) == ("toto",)  # Tue, announced further ahead
    assert games_on(date(2026, 10, 10), state) == ()


def test_games_on_special_draw_on_a_regular_draw_day_counts_once():
    state = {"next_draws": {"toto": "2026-10-08T18:30:00+08:00"}, UPCOMING_KEY: {"toto": ["2026-10-08"]}}
    assert games_on(date(2026, 10, 8), state) == ("toto",)


@pytest.mark.parametrize("value", [
    "2026-10-09", "2026-10-09T18:30:00+08:00", "2026-10-09T10:30:00Z", date(2026, 10, 9),
    sg(2026, 10, 9, 18, 30), ["2026-10-09T18:30:00+08:00"], {"draw_datetime": "2026-10-09T18:30:00+08:00"},
])
def test_games_on_accepts_several_state_shapes(value):
    assert games_on(date(2026, 10, 9), {"next_draws": {"toto": value}}) == ("toto",)


def test_games_on_reads_utc_times_in_singapore_time():
    # 17:00 UTC on 8 Oct is 1am on 9 Oct in Singapore.
    state = {"next_draws": {"toto": "2026-10-08T17:00:00+00:00"}}
    assert games_on(date(2026, 10, 9), state) == ("toto",)


@pytest.mark.parametrize("state", [
    {"next_draws": {"toto": None}}, {"next_draws": {"toto": ""}}, {"next_draws": {"toto": "soon"}},
    {"next_draws": "x"}, {"other": 1}, {"next_draws": {"toto": 12345}}, {UPCOMING_KEY: ["2026-10-09"]},
])
def test_games_on_ignores_unreadable_state(state):
    assert games_on(date(2026, 10, 9), state) == ()


def test_games_on_accepts_the_upper_case_toto_key():
    assert games_on(date(2026, 10, 9), {"next_draws": {"TOTO": "2026-10-09"}}) == ("toto",)


def test_games_on_ignores_4d_dates_left_in_an_old_state():
    state = {"next_draws": {"4d": "2026-10-06", "4D": "2026-10-06", "fourd": "2026-10-06"},
             UPCOMING_KEY: {"4d": ["2026-10-06", "2026-10-07"]}}
    assert games_on(date(2026, 10, 6), state) == ()
    assert games_on(date(2026, 10, 7), state) == ()


# remember_upcoming


def test_remember_upcoming_merges_the_previous_state_and_drops_past_dates():
    # Friday 7.30pm: the next draw page moved on to Monday, the saved state still has Friday's
    # special draw, and an old Tuesday date is past.
    previous = {"next_draws": {"toto": {"draw_datetime": "2026-10-09T18:30:00+08:00"}},
                UPCOMING_KEY: {"toto": ["2026-10-06", "2026-10-09"]}}
    state = {"next_draws": {"toto": {"draw_datetime": "2026-10-12T18:30:00+08:00"},
                            "checked_at": "2026-10-09T19:30:00+08:00"}}
    assert remember_upcoming(state, previous) is state
    assert state[UPCOMING_KEY] == {"toto": ["2026-10-09", "2026-10-12"]}


def test_remember_upcoming_uses_the_last_run_time_and_drops_old_4d_dates():
    state = {"last_run": {"at": "2026-10-10T19:30:00+08:00"},
             UPCOMING_KEY: {"toto": ["2026-10-09", "2026-10-13"], "4d": ["2026-10-11"]}}
    remember_upcoming(state)
    assert state[UPCOMING_KEY] == {"toto": ["2026-10-13"]}
    # Everything past: the key stays, empty.
    state["last_run"]["at"] = "2026-10-14T19:30:00+08:00"
    remember_upcoming(state)
    assert state[UPCOMING_KEY] == {}


def test_remember_upcoming_adds_nothing_without_dates():
    state = {"last_posted": {"toto": 4123}}
    assert remember_upcoming(state, {}) == {"last_posted": {"toto": 4123}}
    assert remember_upcoming(["not", "a", "dict"]) == ["not", "a", "dict"]  # type: ignore[arg-type]


# next_run_at


def test_next_run_at_before_run_time_is_today():
    assert next_run_at(sg(2026, 10, 5, 8, 0), time(19, 30)) == sg(2026, 10, 5, 19, 30)
    assert next_run_at(sg(2026, 10, 5, 19, 29, 59), time(19, 30)) == sg(2026, 10, 5, 19, 30)


def test_next_run_at_after_run_time_is_tomorrow():
    assert next_run_at(sg(2026, 10, 5, 19, 30), time(19, 30)) == sg(2026, 10, 6, 19, 30)
    assert next_run_at(sg(2026, 10, 5, 21, 0), time(19, 30)) == sg(2026, 10, 6, 19, 30)
    assert next_run_at(sg(2026, 10, 31, 23, 59), time(19, 30)) == sg(2026, 11, 1, 19, 30)


def test_next_run_at_converts_other_time_zones():
    # 12:00 UTC is 8pm in Singapore, after the run time, so the next run is tomorrow.
    got = next_run_at(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc), time(19, 30))
    assert got == sg(2026, 10, 6, 19, 30) and got.tzinfo == SG
    # A naive time is Singapore time.
    assert next_run_at(datetime(2026, 10, 5, 8, 0), time(19, 30)) == sg(2026, 10, 5, 19, 30)


# sleep_until


def test_sleep_until_uses_chunks_of_at_most_300_seconds():
    clock = FakeClock(sg(2026, 10, 5, 19, 0))
    sleep_until(sg(2026, 10, 5, 19, 30), clock.now, clock.sleep)
    assert clock.t == sg(2026, 10, 5, 19, 30)
    assert max(clock.sleeps) <= 300 and sum(clock.sleeps) == 1800
    clock.sleeps.clear()
    sleep_until(sg(2026, 10, 5, 19, 0), clock.now, clock.sleep)  # in the past
    assert clock.sleeps == []


# wait_for_results


def test_wait_succeeds_on_third_check():
    clock = FakeClock(sg(2026, 10, 5, 19, 30))
    d = date(2026, 10, 5)
    answers = iter([None, date(2026, 10, 1), d])
    calls = []

    def check(game):
        calls.append((game, clock.now()))
        return next(answers)

    status = wait_for_results(("toto",), d, check, clock.now, clock.sleep)
    assert status == {"toto": True}
    assert [t for _, t in calls] == [sg(2026, 10, 5, 19, 30), sg(2026, 10, 5, 19, 40), sg(2026, 10, 5, 19, 50)]
    assert clock.t == sg(2026, 10, 5, 19, 50)


def test_wait_gives_up_after_two_hours_with_13_checks():
    clock = FakeClock(sg(2026, 10, 5, 19, 30))
    times = []

    def check(game):
        times.append(clock.now())
        return date(2026, 10, 1)  # still the previous draw

    status = wait_for_results(["toto"], date(2026, 10, 5), check, clock.now, clock.sleep,
                              retry_minutes=10, retry_hours=2.0)
    assert status == {"toto": False}
    assert len(times) == 13
    assert all(b - a == timedelta(minutes=10) for a, b in zip(times, times[1:]))
    assert times[0] == sg(2026, 10, 5, 19, 30) and times[-1] == sg(2026, 10, 5, 21, 30)
    assert clock.t == sg(2026, 10, 5, 21, 30)  # no pointless sleep after the last check


def test_wait_spacing_does_not_drift_when_checks_are_slow():
    clock = FakeClock(sg(2026, 10, 5, 19, 30))
    times = []

    def slow_check(game):
        times.append(clock.now())
        clock.t += timedelta(seconds=45)  # a slow site
        return None

    wait_for_results(["toto"], date(2026, 10, 5), slow_check, clock.now, clock.sleep, retry_minutes=10, retry_hours=1)
    assert [t.minute for t in times] == [30, 40, 50, 0, 10, 20, 30]


def test_wait_hands_the_result_to_on_ready_until_it_is_finished():
    # Thu 8 Oct: TOTO is out at the second check; the run finishes it only at its third try.
    clock = FakeClock(sg(2026, 10, 8, 19, 30))
    d = date(2026, 10, 8)
    checks = []
    calls = []

    def check(game):
        checks.append(clock.now())
        return d if len(checks) >= 2 else date(2026, 10, 5)

    def on_ready(games, retry_at):
        calls.append((clock.now(), games, dict(retry_at)))
        return games if len(calls) >= 3 else ()

    status = wait_for_results(("toto",), d, check, clock.now, clock.sleep, retry_minutes=10, retry_hours=2.0,
                              on_ready=on_ready)
    assert status == {"toto": True}
    assert checks == [sg(2026, 10, 8, 19, 30), sg(2026, 10, 8, 19, 40)]  # the site is not asked again once out
    assert calls == [
        (sg(2026, 10, 8, 19, 40), ("toto",), {"toto": sg(2026, 10, 8, 19, 50)}),
        (sg(2026, 10, 8, 19, 50), ("toto",), {"toto": sg(2026, 10, 8, 20, 0)}),
        (sg(2026, 10, 8, 20, 0), ("toto",), {"toto": sg(2026, 10, 8, 20, 10)}),
    ]
    assert clock.t == sg(2026, 10, 8, 20, 0)


def test_wait_on_ready_gets_no_retry_time_at_the_last_check_and_survives_errors():
    clock = FakeClock(sg(2026, 10, 8, 19, 30))
    d = date(2026, 10, 8)
    calls = []

    def on_ready(games, retry_at):
        calls.append(dict(retry_at))
        if len(calls) == 1:
            raise RuntimeError("hook blew up")  # tried again at the next check
        return ()

    status = wait_for_results(("toto",), d, lambda g: d, clock.now, clock.sleep, retry_minutes=10, retry_hours=0.5,
                              on_ready=on_ready)
    assert status == {"toto": True}
    assert calls == [{"toto": sg(2026, 10, 8, 19, 40)}, {"toto": sg(2026, 10, 8, 19, 50)},
                     {"toto": sg(2026, 10, 8, 20, 0)}, {"toto": None}]
    assert clock.t == sg(2026, 10, 8, 20, 0)


def test_wait_reports_whether_any_check_reached_the_site():
    clock = FakeClock(sg(2026, 10, 8, 19, 30))
    d = date(2026, 10, 8)

    def down(game):
        raise ConnectionError("HTTP 503")

    reached: dict[str, bool] = {}
    assert wait_for_results(("toto",), d, down, clock.now, clock.sleep, retry_minutes=10, retry_hours=0.5,
                            reached=reached) == {"toto": False}
    assert reached == {"toto": False}
    reached = {}
    assert wait_for_results(("toto",), d, lambda g: None, clock.now, clock.sleep, retry_minutes=10,
                            retry_hours=0, reached=reached) == {"toto": False}
    assert reached == {"toto": True}


def test_wait_start_at_delays_the_first_check_and_the_window():
    clock = FakeClock(sg(2026, 10, 9, 19, 30))
    d = date(2026, 10, 9)
    times = []

    def check(game):
        times.append(clock.now())
        return None

    wait_for_results(("toto",), d, check, clock.now, clock.sleep, retry_minutes=10, retry_hours=0.5,
                     start_at={"toto": sg(2026, 10, 9, 22, 30)})
    assert times == [sg(2026, 10, 9, 22, 30), sg(2026, 10, 9, 22, 40), sg(2026, 10, 9, 22, 50),
                     sg(2026, 10, 9, 23, 0)]


def test_wait_treats_check_errors_as_not_ready():
    clock = FakeClock(sg(2026, 10, 5, 19, 30))
    d = date(2026, 10, 5)
    answers = iter([ConnectionError("down"), d])

    def check(game):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert wait_for_results(("toto",), d, check, clock.now, clock.sleep) == {"toto": True}


def test_wait_accepts_datetimes_and_later_dates():
    clock = FakeClock(sg(2026, 10, 5, 19, 30))
    d = date(2026, 10, 5)
    assert wait_for_results(("toto",), d, lambda g: sg(2026, 10, 5, 18, 30), clock.now, clock.sleep) == {"toto": True}
    assert wait_for_results(("toto",), d, lambda g: "2026-10-05", clock.now, clock.sleep) == {"toto": True}
    assert wait_for_results(("toto",), d, lambda g: date(2026, 10, 8), clock.now, clock.sleep) == {"toto": True}
    assert clock.sleeps == []


def test_wait_with_no_window_checks_once_and_no_games_checks_nothing():
    clock = FakeClock(sg(2026, 10, 5, 19, 30))
    calls = []
    status = wait_for_results(("toto",), date(2026, 10, 5), lambda g: calls.append(g), clock.now, clock.sleep,
                              retry_minutes=10, retry_hours=0)
    assert status == {"toto": False} and calls == ["toto"]
    assert wait_for_results((), date(2026, 10, 5), lambda g: 1 / 0, clock.now, clock.sleep) == {}


# SchedulerConfig


def test_config_defaults_from_constants():
    cfg = SchedulerConfig.from_env({})
    assert cfg.run_time == C.DEFAULT_RUN_TIME == time(19, 30)
    assert cfg.retry_minutes == C.DEFAULT_RETRY_MINUTES
    assert cfg.retry_hours == C.DEFAULT_RETRY_HOURS
    assert cfg.warnings == []


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("RUN_AT", "20:15")
    monkeypatch.setenv("RETRY_MINUTES", "5")
    monkeypatch.setenv("RETRY_HOURS", "1.5")
    cfg = SchedulerConfig.from_env()
    assert (cfg.run_time, cfg.retry_minutes, cfg.retry_hours) == (time(20, 15), 5.0, 1.5)
    assert cfg.warnings == []


def test_config_bad_values_fall_back_or_clamp():
    cfg = SchedulerConfig.from_env({"RUN_AT": "dinner", "RETRY_MINUTES": "often", "RETRY_HOURS": "99"})
    assert cfg.run_time == time(19, 30)
    assert cfg.retry_minutes == C.DEFAULT_RETRY_MINUTES
    assert cfg.retry_hours == 12.0
    assert len(cfg.warnings) == 3
    cfg = SchedulerConfig.from_env({"RETRY_MINUTES": "0", "RETRY_HOURS": "-1"})
    assert cfg.retry_minutes == 1.0 and cfg.retry_hours == 0.0


@pytest.mark.parametrize("text,expected", [
    ("19:30", time(19, 30)), ("19.30", time(19, 30)), ("1930", time(19, 30)), ("7.30pm", time(19, 30)),
    ("7:30 PM", time(19, 30)), ("7pm", time(19, 0)), ("12am", time(0, 0)), ("12.15pm", time(12, 15)),
    ("06:05", time(6, 5)), ("0", time(0, 0)),
])
def test_parse_run_time(text, expected):
    assert parse_run_time(text) == expected


@pytest.mark.parametrize("text", ["24:00", "19:60", "13pm", "0am", "half past seven", "", "7.3pm"])
def test_parse_run_time_rejects(text):
    with pytest.raises(ValueError):
        parse_run_time(text)


# serve


class Recorder:
    """Collects every hook call made by serve."""

    def __init__(self, clock: FakeClock, latest: dict | None = None, state: dict | None = None):
        self.clock = clock
        self.latest = latest or {}  # game -> callable(day) returning the latest date on the site
        self.state = state if state is not None else {}
        self.runs: list[tuple[datetime, tuple[str, ...]]] = []
        self.notices: list[str] = []
        self.logs: list[tuple[str, str]] = []
        self.checks: list[tuple[str, datetime]] = []
        self.fail_runs = 0
        self.refresh_error: Exception | None = None

    def run(self, games):
        self.runs.append((self.clock.now(), tuple(games)))
        if self.fail_runs:
            self.fail_runs -= 1
            raise RuntimeError("parser blew up")

    def check(self, game):
        self.checks.append((game, self.clock.now()))
        fn = self.latest.get(game)
        return fn(self.clock.now().date()) if fn else None

    def refresh(self):
        if self.refresh_error:
            raise self.refresh_error
        return self.state

    def notify(self, text):
        self.notices.append(text)

    def log(self, event, message):
        self.logs.append((event, message))

    def serve(self, cycles, config=None):
        serve(self.run, self.check, self.refresh, self.notify, self.log, self.clock.now, self.clock.sleep,
              config or SchedulerConfig(), max_cycles=cycles)

    def events(self, name):
        return [m for e, m in self.logs if e == name]


def published_today(day):
    return day


def previous_draw(day):
    return day - timedelta(days=3)


def test_serve_notifies_once_when_a_special_draw_result_never_appears():
    # Fri 9 Oct 2026: a special TOTO draw from the state, but the site still shows Tuesday's draw.
    clock = FakeClock(sg(2026, 10, 9, 9, 0))
    rec = Recorder(clock, latest={"toto": previous_draw},
                   state={"next_draws": {"toto": "2026-10-09T18:30:00+08:00"}})
    rec.serve(cycles=1)

    assert rec.runs == []
    assert len(rec.notices) == 1
    notice = rec.notices[0]
    assert "TOTO result for Fri 9 Oct 2026 is still not on the Singapore Pools site" in notice
    assert "every 10 minutes for 2 hours" in notice
    assert not any(ch in notice for ch in DASHES)
    assert rec.events("WAIT") == [notice] and rec.events("ERROR") == []
    assert rec.events("SCHEDULE")[-1] == "TOTO draw on Fri 9 Oct 2026, checking for the result"
    # Checking started at 7.30pm and TOTO was checked 13 times, every 10 minutes.
    times = [t for g, t in rec.checks if g == "toto"]
    assert len(times) == 13 and len(rec.checks) == 13
    assert times[0] == sg(2026, 10, 9, 19, 30) and times[-1] == sg(2026, 10, 9, 21, 30)
    assert all(b - a == timedelta(minutes=10) for a, b in zip(times, times[1:]))
    assert max(clock.sleeps) <= 300


def test_serve_runs_immediately_when_everything_is_out():
    clock = FakeClock(sg(2026, 10, 5, 10, 0))  # Monday, TOTO
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 5, 19, 30), ("toto",))]
    assert rec.checks == [("toto", sg(2026, 10, 5, 19, 30))]
    assert rec.notices == [] and rec.events("WAIT") == []
    assert rec.events("SCHEDULE")[-1] == "TOTO draw on Mon 5 Oct 2026, checking for the result"


def test_serve_no_draw_day_logs_and_does_nothing():
    clock = FakeClock(sg(2026, 10, 6, 10, 0))  # Tuesday
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1)
    assert rec.runs == [] and rec.checks == [] and rec.notices == []
    assert "No draw today" in rec.events("SCHEDULE")
    assert clock.t == sg(2026, 10, 6, 19, 30)


def test_serve_does_nothing_on_the_old_4d_days():
    clock = FakeClock(sg(2026, 10, 9, 22, 0))  # Fri night, then Sat and Sun (old 4D days)
    rec = Recorder(clock, latest={"toto": published_today, "4d": published_today})
    rec.serve(cycles=2)
    assert rec.runs == [] and rec.checks == [] and rec.notices == []
    assert rec.events("SCHEDULE").count("No draw today") == 2
    assert clock.t == sg(2026, 10, 11, 19, 30)


def test_serve_survives_an_exception_in_run_fn():
    # Thu 8 Oct, then a special TOTO draw on Fri 9 Oct.
    clock = FakeClock(sg(2026, 10, 8, 9, 0))
    rec = Recorder(clock, latest={"toto": published_today},
                   state={"next_draws": {"toto": "2026-10-09T18:30:00+08:00"}})
    rec.fail_runs = 1
    rec.serve(cycles=2)
    # The run that blew up is tried again at the next check, then Friday runs as usual.
    assert rec.runs == [(sg(2026, 10, 8, 19, 30), ("toto",)), (sg(2026, 10, 8, 19, 40), ("toto",)),
                        (sg(2026, 10, 9, 19, 30), ("toto",))]
    errors = rec.events("ERROR")
    assert len(errors) == 1 and "parser blew up" in errors[0]
    assert rec.notices == []


def test_serve_survives_failing_hooks():
    clock = FakeClock(sg(2026, 10, 5, 9, 0))  # Monday, then Tuesday
    rec = Recorder(clock, latest={"toto": previous_draw})

    def bad_notify(text):
        raise OSError("telegram down")

    def bad_log(event, message):
        raise OSError("vault gone")

    serve(rec.run, rec.check, rec.refresh, bad_notify, bad_log, clock.now, clock.sleep,
          SchedulerConfig(retry_minutes=10, retry_hours=0.5), max_cycles=2)
    assert len(rec.checks) == 4  # Monday's whole window was checked
    assert clock.t.date() == date(2026, 10, 6)  # it went on to the second day


def test_serve_falls_back_to_weekdays_when_refresh_fails():
    clock = FakeClock(sg(2026, 10, 8, 9, 0))  # Thursday, TOTO
    rec = Recorder(clock, latest={"toto": published_today})
    rec.refresh_error = ConnectionError("site down")
    rec.serve(cycles=1)
    assert [g for _, g in rec.runs] == [("toto",)]
    assert rec.events("ERROR") == ["Could not refresh the next draw dates (site down), using the regular draw days"]


def test_serve_keeps_a_special_date_seen_earlier_when_refresh_fails():
    # Thursday's refresh sees a special TOTO draw on Friday; on Friday the site is down.
    clock = FakeClock(sg(2026, 10, 8, 9, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    refreshes = iter([{"next_draws": {"toto": "2026-10-09T18:30:00+08:00"}}, ConnectionError("site down")])

    def refresh():
        answer = next(refreshes)
        if isinstance(answer, Exception):
            raise answer
        return answer

    rec.refresh = refresh
    rec.serve(cycles=2)
    assert [(t.date(), g) for t, g in rec.runs] == [(date(2026, 10, 8), ("toto",)), (date(2026, 10, 9), ("toto",))]
    assert len(rec.events("ERROR")) == 1 and "site down" in rec.events("ERROR")[0]


def test_serve_remembers_a_special_date_after_the_next_draw_page_moves_on():
    # Thursday's refresh sees a special TOTO draw on Friday. By Friday evening the next draw
    # page already shows Monday, but Friday must still count as a TOTO day.
    clock = FakeClock(sg(2026, 10, 8, 9, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    states = iter([
        {"next_draws": {"toto": "2026-10-09T18:30:00+08:00"}},
        {"next_draws": {"toto": "2026-10-12T18:30:00+08:00"}},
    ])
    rec.refresh = lambda: next(states)
    rec.serve(cycles=2)
    assert [(t.date(), g) for t, g in rec.runs] == [(date(2026, 10, 8), ("toto",)), (date(2026, 10, 9), ("toto",))]


def test_serve_catches_up_when_started_inside_the_window():
    clock = FakeClock(sg(2026, 10, 5, 20, 0))  # Monday 8pm, NAS just rebooted
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 5, 20, 0), ("toto",))]


def test_serve_catches_up_on_a_special_draw_day_after_a_restart():
    # Fri 9 Oct 8pm: the NAS rebooted during the window of a special draw the state remembers.
    clock = FakeClock(sg(2026, 10, 9, 20, 0))
    rec = Recorder(clock, latest={"toto": published_today},
                   state={"next_draws": {"toto": "2026-10-12T18:30:00+08:00"}, UPCOMING_KEY: {"toto": ["2026-10-09"]}})
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 9, 20, 0), ("toto",))]


def test_serve_without_catch_up_waits_for_tomorrow():
    clock = FakeClock(sg(2026, 10, 5, 20, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1, config=SchedulerConfig(catch_up=False))
    assert rec.runs == []  # Tuesday has no draw
    assert clock.t == sg(2026, 10, 6, 19, 30)


def test_serve_after_window_waits_for_next_day():
    clock = FakeClock(sg(2026, 10, 5, 22, 0))  # Monday, after the 2 hour window
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=3)
    # Tuesday and Wednesday: no draw. Thursday: TOTO.
    assert [(t.date(), g) for t, g in rec.runs] == [(date(2026, 10, 8), ("toto",))]
    assert rec.events("SCHEDULE").count("No draw today") == 2


def test_serve_logs_start_and_respects_custom_run_time():
    clock = FakeClock(sg(2026, 10, 5, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1, config=SchedulerConfig(run_time=time(20, 15)))
    assert rec.runs[0][0] == sg(2026, 10, 5, 20, 15)
    started = rec.events("SCHEDULE")[0]
    assert started.startswith("Scheduler started. Runs at 8.15pm Singapore time")
    assert not any(ch in started for ch in DASHES)


def test_missing_message_wording():
    clock = FakeClock(sg(2026, 10, 10, 9, 0))  # Sat 10 Oct, a special TOTO draw
    rec = Recorder(clock, latest={}, state={"next_draws": {"toto": "2026-10-10"}})
    rec.serve(cycles=1, config=SchedulerConfig(retry_minutes=15, retry_hours=0.5))
    assert rec.runs == [] and len(rec.checks) == 3
    assert rec.notices == [
        "TOTO result for Sat 10 Oct 2026 is still not on the Singapore Pools site after checking "
        "every 15 minutes for 30 minutes, so it was not posted. It will be stored on the next TOTO "
        "run but not posted."
    ]


# serve: the run outcome, retries and notices


class Done:
    """done_fn double: answers from a list per call (the last answer repeats)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, date]] = []

    def __call__(self, game, day):
        self.calls.append((game, day))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_serve_runs_again_when_the_first_run_did_not_finish():
    clock = FakeClock(sg(2026, 10, 5, 10, 0))  # Monday, TOTO
    rec = Recorder(clock, latest={"toto": published_today})
    # Each check asks done_fn before the run (not done yet) and after it: the first run could not
    # fetch the result page, the second did.
    done = Done(False, False, False, True)
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=done)
    assert rec.runs == [(sg(2026, 10, 5, 19, 30), ("toto",)), (sg(2026, 10, 5, 19, 40), ("toto",))]
    assert done.calls == [("toto", date(2026, 10, 5))] * 4
    assert rec.notices == [] and rec.events("ERROR") == []
    assert rec.events("WAIT") == ["TOTO result for Mon 5 Oct 2026 is out but was not stored or posted yet, "
                                  "trying again at 7.40pm"]
    assert sum(1 for g, _ in rec.checks if g == "toto") == 1  # the site is not asked again once it is out


def test_serve_gives_up_with_one_notice_when_runs_never_finish():
    clock = FakeClock(sg(2026, 10, 5, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=Done(False))
    times = [t for t, _ in rec.runs]
    assert len(times) == 13 and times[0] == sg(2026, 10, 5, 19, 30) and times[-1] == sg(2026, 10, 5, 21, 30)
    assert all(b - a == timedelta(minutes=10) for a, b in zip(times, times[1:]))
    assert rec.notices == [
        "TOTO result for Mon 5 Oct 2026 is on the Singapore Pools site but could not be fetched or posted after "
        "checking every 10 minutes for 2 hours. See the activity log."
    ]
    assert rec.events("ERROR") == rec.notices
    assert len(rec.events("WAIT")) == 12  # no "trying again" after the last run
    assert not any(ch in rec.notices[0] for ch in DASHES)


def test_serve_without_done_fn_uses_the_run_result():
    class Result:
        def __init__(self, ok):
            self.ok = ok

    clock = FakeClock(sg(2026, 10, 8, 10, 0))  # Thursday, TOTO
    rec = Recorder(clock, latest={"toto": published_today})
    results = iter([Result(False), Result(True)])  # Telegram was down for the first run
    runs = []

    def run(games):
        runs.append((clock.now(), games))
        return next(results)

    serve(run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(), max_cycles=1)
    assert runs == [(sg(2026, 10, 8, 19, 30), ("toto",)), (sg(2026, 10, 8, 19, 40), ("toto",))]
    assert rec.notices == []
    assert rec.events("WAIT") == ["TOTO result for Thu 8 Oct 2026 is out but was not stored or posted yet, "
                                  "trying again at 7.40pm"]


def test_serve_quiet_run_finishes_at_once():
    # Posting off (dry run): the cycle is done as soon as the draw is stored.
    clock = FakeClock(sg(2026, 10, 5, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=Done(False, True))
    assert rec.runs == [(sg(2026, 10, 5, 19, 30), ("toto",))]
    assert rec.notices == [] and rec.events("WAIT") == []


def test_serve_does_not_rerun_a_draw_already_stored_and_posted_after_a_restart():
    # Thu 1 Oct: the TOTO draw was stored and posted at 7.30pm, the container was killed at 7.45pm
    # and restarted at 7.50pm, inside the retry window.
    clock = FakeClock(sg(2026, 10, 1, 19, 50))
    rec = Recorder(clock, latest={"toto": published_today})
    done = Done(True)
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=done)
    assert rec.runs == []
    assert done.calls == [("toto", date(2026, 10, 1))]
    assert rec.checks == [("toto", sg(2026, 10, 1, 19, 50))]
    assert "TOTO result for Thu 1 Oct 2026 was already handled by an earlier run, so it is not run again" in \
        rec.events("SCHEDULE")
    assert rec.notices == [] and rec.events("WAIT") == [] and rec.events("ERROR") == []


def test_serve_restarted_in_the_window_finishes_a_draw_the_earlier_run_did_not():
    # Mon 5 Oct: the 7.30pm run stored nothing before the container was killed; it restarts at 8.05pm.
    clock = FakeClock(sg(2026, 10, 5, 20, 5))
    rec = Recorder(clock, latest={"toto": published_today})
    done = Done(False, True)  # not done before the run, done after it
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=done)
    assert rec.runs == [(sg(2026, 10, 5, 20, 5), ("toto",))]
    assert done.calls == [("toto", date(2026, 10, 5))] * 2
    assert rec.notices == [] and rec.events("WAIT") == [] and rec.events("ERROR") == []


def test_serve_runs_as_soon_as_the_result_is_out():
    # Thu 8 Oct: the TOTO result only appears at 8pm, so it runs at the 8pm check, not later.
    clock = FakeClock(sg(2026, 10, 8, 9, 0))
    rec = Recorder(clock, latest={"toto": lambda day: day if clock.now() >= sg(2026, 10, 8, 20, 0) else previous_draw(day)})
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 8, 20, 0), ("toto",))]
    assert [t for _, t in rec.checks] == [sg(2026, 10, 8, 19, 30), sg(2026, 10, 8, 19, 40),
                                          sg(2026, 10, 8, 19, 50), sg(2026, 10, 8, 20, 0)]
    assert rec.notices == [] and rec.events("WAIT") == []
    assert clock.t == sg(2026, 10, 8, 20, 0)


def test_serve_waits_for_a_draw_held_later_in_the_evening():
    # A special TOTO draw at 9.30pm on Friday: checking starts at 10.30pm, not 7.30pm.
    clock = FakeClock(sg(2026, 10, 9, 10, 0))
    rec = Recorder(clock, latest={"toto": lambda day: day if clock.now() >= sg(2026, 10, 9, 22, 40) else previous_draw(day)},
                   state={"next_draws": {"toto": {"draw_datetime": "2026-10-09T21:30:00+08:00"}}})
    rec.serve(cycles=1)
    assert rec.checks[0] == ("toto", sg(2026, 10, 9, 22, 30))
    assert rec.runs == [(sg(2026, 10, 9, 22, 40), ("toto",))]
    assert rec.notices == []
    assert "TOTO draw on Fri 9 Oct 2026 is at 9.30pm, checking for its result from 10.30pm" in rec.events("SCHEDULE")


def test_serve_keeps_the_run_time_for_a_draw_at_the_usual_time():
    # RUN_AT 7pm and the next draw page still shows tonight's 6.30pm draw: checking starts at 7pm.
    clock = FakeClock(sg(2026, 10, 5, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today},
                   state={"next_draws": {"toto": {"draw_datetime": "2026-10-05T18:30:00+08:00"}}})
    rec.serve(cycles=1, config=SchedulerConfig(run_time=time(19, 0)))
    assert rec.runs == [(sg(2026, 10, 5, 19, 0), ("toto",))]


def test_serve_with_games_given_follows_toto_and_ignores_old_game_names():
    # An older caller still passes both games: 4D is dropped and TOTO is followed as usual.
    clock = FakeClock(sg(2026, 10, 5, 9, 0))  # Monday, Tuesday, then Wednesday (an old 4D day)
    rec = Recorder(clock, latest={"4d": published_today, "toto": published_today})
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=3, games=("toto", "4d"))
    assert rec.runs == [(sg(2026, 10, 5, 19, 30), ("toto",))]
    assert [g for g, _ in rec.checks] == ["toto"]
    started = rec.events("SCHEDULE")[0]
    assert started.endswith("until results are out") and "Only" not in started
    assert rec.events("SCHEDULE").count("No draw today") == 2


def test_serve_remembers_a_special_draw_after_a_restart(tmp_path):
    # Thursday's run stored the special TOTO draw on Friday. The container restarts on Friday
    # morning and by 7.30pm the next draw page already shows Monday's draw.
    from huatbot.vault import Vault

    vault = Vault(tmp_path / "vault")
    vault.save_state({"next_draws": {"toto": {"draw_datetime": "2026-10-09T18:30:00+08:00"},
                                     "checked_at": "2026-10-08T19:45:00+08:00"}})
    clock = FakeClock(sg(2026, 10, 9, 10, 0))

    def refresh():
        # What the refresh before each cycle does: read the next draw page, save the state.
        state = vault.load_state()
        state["next_draws"] = {"toto": {"draw_datetime": "2026-10-12T18:30:00+08:00"},
                               "checked_at": clock.now().isoformat()}
        vault.save_state(state)
        return state

    rec = Recorder(clock, latest={"toto": published_today})
    rec.refresh = refresh
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 9, 19, 30), ("toto",))]
    assert rec.events("SCHEDULE")[-1] == "TOTO draw on Fri 9 Oct 2026, checking for the result"
    state = vault.load_state()
    assert state["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-12")
    assert state["upcoming_draws"]["toto"] == ["2026-10-09", "2026-10-12"]


def test_serve_says_the_site_could_not_be_reached_when_every_check_failed():
    # Mon 26 Oct: every request fails from 7pm to 11pm, so no check learns whether TOTO is out.
    clock = FakeClock(sg(2026, 10, 26, 10, 0))
    rec = Recorder(clock)

    def down(game):
        rec.checks.append((game, clock.now()))
        raise ConnectionError("HTTP 503")

    serve(rec.run, down, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(), max_cycles=1)
    assert rec.runs == [] and len(rec.checks) == 13
    assert rec.notices == [
        "TOTO result for Mon 26 Oct 2026 could not be checked: the Singapore Pools site could not be reached after "
        "checking every 10 minutes for 2 hours, so it was not posted. It will be stored on the next TOTO run but "
        "not posted."
    ]
    assert rec.events("ERROR") == rec.notices
    assert rec.events("WAIT") == []
    assert "still not on the Singapore Pools site" not in rec.notices[0]
    assert not any(ch in rec.notices[0] for ch in DASHES)


def test_serve_blames_the_publisher_once_a_check_reached_the_site():
    # One check got through (the result was not out), the rest failed: the result is missing.
    clock = FakeClock(sg(2026, 10, 26, 10, 0))
    rec = Recorder(clock)
    calls = {"n": 0}

    def flaky(game):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("HTTP 503")
        return previous_draw(clock.now().date())

    serve(rec.run, flaky, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(), max_cycles=1)
    assert len(rec.notices) == 1 and "is still not on the Singapore Pools site" in rec.notices[0]
    assert rec.events("WAIT") == rec.notices and rec.events("ERROR") == []


def test_serve_notice_says_posting_failed_when_only_telegram_was_down():
    # Thu 22 Oct: the TOTO result is fetched and stored at 7.30pm, but Telegram is down all evening.
    class Result:
        ok = False
        messages = ["1", "2", "3"]

    clock = FakeClock(sg(2026, 10, 22, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    runs = []

    def run(games):
        runs.append(games)
        return Result()

    serve(run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep,
          SchedulerConfig(), max_cycles=1, done_fn=Done(False))
    assert len(runs) == 13 and set(runs) == {("toto",)}
    assert rec.notices == [
        "TOTO result for Thu 22 Oct 2026 is on the Singapore Pools site but could not be posted to Telegram after "
        "checking every 10 minutes for 2 hours. See the activity log."
    ]
    assert rec.events("ERROR") == rec.notices
    assert "fetched" not in rec.notices[0]
    assert not any(ch in rec.notices[0] for ch in DASHES)
