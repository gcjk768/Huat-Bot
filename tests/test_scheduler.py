"""Scheduler: draw days, run time, waiting for results and the serve loop, all on a fake clock."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from huatbot import constants as C
from huatbot.scheduler import (
    SG,
    SchedulerConfig,
    games_on,
    next_run_at,
    parse_run_time,
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


@pytest.mark.parametrize("day,expected", [
    (date(2026, 10, 5), ("toto",)),   # Mon
    (date(2026, 10, 6), ()),          # Tue
    (date(2026, 10, 7), ("4d",)),     # Wed
    (date(2026, 10, 8), ("toto",)),   # Thu
    (date(2026, 10, 9), ()),          # Fri
    (date(2026, 10, 10), ("4d",)),    # Sat
    (date(2026, 10, 11), ("4d",)),    # Sun
])
def test_games_on_regular_weekdays(day, expected):
    assert games_on(day, {}) == expected
    assert games_on(day, None) == expected


def test_games_on_special_draw_dates_from_state():
    state = {"next_draws": {"toto": "2026-10-09T18:30:00+08:00", "4d": "2026-10-06T18:30:00+08:00"}}
    assert games_on(date(2026, 10, 9), state) == ("toto",)   # special TOTO on a Friday
    assert games_on(date(2026, 10, 6), state) == ("4d",)     # 4D moved to a Tuesday
    assert games_on(date(2026, 10, 7), state) == ("4d",)     # regular days still count
    assert games_on(date(2026, 10, 2), state) == ()


def test_games_on_special_draw_on_a_draw_day_of_the_other_game():
    state = {"next_draws": {"toto": "2026-10-10T18:30:00+08:00"}}
    assert games_on(date(2026, 10, 10), state) == ("toto", "4d")


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
    {"next_draws": {"toto": None, "4d": ""}}, {"next_draws": {"toto": "soon"}}, {"next_draws": "x"},
    {"other": 1}, {"next_draws": {"toto": 12345}},
])
def test_games_on_ignores_unreadable_state(state):
    assert games_on(date(2026, 10, 9), state) == ()


def test_games_on_accepts_fourd_alias_key():
    assert games_on(date(2026, 10, 6), {"next_draws": {"fourd": "2026-10-06"}}) == ("4d",)


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

    wait_for_results(["4d"], date(2026, 10, 5), slow_check, clock.now, clock.sleep, retry_minutes=10, retry_hours=1)
    assert [t.minute for t in times] == [30, 40, 50, 0, 10, 20, 30]


def test_wait_tracks_each_game_separately():
    clock = FakeClock(sg(2026, 10, 10, 19, 30))
    d = date(2026, 10, 10)
    calls = {"toto": 0, "4d": 0}

    def check(game):
        calls[game] += 1
        if game == "4d":
            return d if calls["4d"] >= 2 else date(2026, 10, 7)
        return date(2026, 10, 8)

    status = wait_for_results(("toto", "4d"), d, check, clock.now, clock.sleep, retry_minutes=10, retry_hours=0.5)
    assert status == {"toto": False, "4d": True}
    assert calls == {"toto": 4, "4d": 2}  # 4D is not checked again once it is out


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


def test_serve_runs_ready_games_and_notifies_missing_ones():
    # Sat 10 Oct 2026: a regular 4D day plus a special TOTO draw from the state.
    clock = FakeClock(sg(2026, 10, 10, 9, 0))
    rec = Recorder(clock, latest={"4d": published_today, "toto": previous_draw},
                   state={"next_draws": {"toto": "2026-10-10T18:30:00+08:00"}})
    rec.serve(cycles=1)

    assert [g for _, g in rec.runs] == [("4d",)]
    assert len(rec.notices) == 1
    notice = rec.notices[0]
    assert "TOTO result for Sat 10 Oct 2026" in notice and "4D" not in notice
    assert "every 10 minutes for 2 hours" in notice
    assert not any(ch in notice for ch in DASHES)
    assert rec.events("WAIT") == [notice]
    assert rec.events("SCHEDULE")[-1] == "TOTO and 4D draw on Sat 10 Oct 2026, checking for the result"
    # The run started at 7.30pm and ran 4D at once (it does not wait for TOTO); TOTO was checked 13 times.
    assert rec.runs[0][0] == sg(2026, 10, 10, 19, 30)
    assert sum(1 for g, _ in rec.checks if g == "toto") == 13
    assert sum(1 for g, _ in rec.checks if g == "4d") == 1
    assert rec.checks[0][1] == sg(2026, 10, 10, 19, 30)
    assert max(clock.sleeps) <= 300


def test_serve_runs_immediately_when_everything_is_out():
    clock = FakeClock(sg(2026, 10, 5, 10, 0))  # Monday, TOTO
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 5, 19, 30), ("toto",))]
    assert rec.notices == [] and rec.events("WAIT") == []


def test_serve_no_draw_day_logs_and_does_nothing():
    clock = FakeClock(sg(2026, 10, 6, 10, 0))  # Tuesday
    rec = Recorder(clock, latest={"toto": published_today, "4d": published_today})
    rec.serve(cycles=1)
    assert rec.runs == [] and rec.checks == [] and rec.notices == []
    assert "No draw today" in rec.events("SCHEDULE")
    assert clock.t == sg(2026, 10, 6, 19, 30)


def test_serve_survives_an_exception_in_run_fn():
    clock = FakeClock(sg(2026, 10, 10, 9, 0))  # Sat, then Sun: both 4D days
    rec = Recorder(clock, latest={"4d": published_today})
    rec.fail_runs = 1
    rec.serve(cycles=2)
    # The run that blew up is tried again at the next check, then Sunday runs as usual.
    assert rec.runs == [(sg(2026, 10, 10, 19, 30), ("4d",)), (sg(2026, 10, 10, 19, 40), ("4d",)),
                        (sg(2026, 10, 11, 19, 30), ("4d",))]
    errors = rec.events("ERROR")
    assert len(errors) == 1 and "parser blew up" in errors[0]
    assert rec.notices == []


def test_serve_survives_failing_hooks():
    clock = FakeClock(sg(2026, 10, 10, 9, 0))
    rec = Recorder(clock, latest={"4d": previous_draw})

    def bad_notify(text):
        raise OSError("telegram down")

    def bad_log(event, message):
        raise OSError("vault gone")

    serve(rec.run, rec.check, rec.refresh, bad_notify, bad_log, clock.now, clock.sleep,
          SchedulerConfig(retry_minutes=10, retry_hours=0.5), max_cycles=2)
    assert clock.t.date() == date(2026, 10, 11)  # it went on to the second day


def test_serve_falls_back_to_weekdays_when_refresh_fails():
    clock = FakeClock(sg(2026, 10, 7, 9, 0))  # Wednesday, 4D
    rec = Recorder(clock, latest={"4d": published_today})
    rec.refresh_error = ConnectionError("site down")
    rec.serve(cycles=1)
    assert [g for _, g in rec.runs] == [("4d",)]
    assert any("site down" in m for m in rec.events("ERROR"))


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


def test_serve_without_catch_up_waits_for_tomorrow():
    clock = FakeClock(sg(2026, 10, 5, 20, 0))
    rec = Recorder(clock, latest={"toto": published_today, "4d": published_today})
    rec.serve(cycles=1, config=SchedulerConfig(catch_up=False))
    assert rec.runs == []  # Tuesday has no draw
    assert clock.t == sg(2026, 10, 6, 19, 30)


def test_serve_after_window_waits_for_next_day():
    clock = FakeClock(sg(2026, 10, 5, 22, 0))  # Monday, after the 2 hour window
    rec = Recorder(clock, latest={"4d": published_today})
    rec.serve(cycles=2)
    # Tuesday: no draw. Wednesday: 4D.
    assert [(t.date(), g) for t, g in rec.runs] == [(date(2026, 10, 7), ("4d",))]


def test_serve_logs_start_and_respects_custom_run_time():
    clock = FakeClock(sg(2026, 10, 5, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    rec.serve(cycles=1, config=SchedulerConfig(run_time=time(20, 15)))
    assert rec.runs[0][0] == sg(2026, 10, 5, 20, 15)
    started = rec.events("SCHEDULE")[0]
    assert started.startswith("Scheduler started. Runs at 8.15pm Singapore time")
    assert not any(ch in started for ch in DASHES)


def test_missing_message_for_two_games_is_plural():
    clock = FakeClock(sg(2026, 10, 10, 9, 0))
    rec = Recorder(clock, latest={}, state={"next_draws": {"toto": "2026-10-10"}})
    rec.serve(cycles=1, config=SchedulerConfig(retry_minutes=15, retry_hours=0.5))
    assert rec.runs == []
    assert rec.notices == [
        "TOTO and 4D results for Sat 10 Oct 2026 are still not on the Singapore Pools site after checking "
        "every 15 minutes for 30 minutes, so they were not posted. They will be stored on the next TOTO and 4D "
        "runs but not posted."
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

    clock = FakeClock(sg(2026, 10, 7, 10, 0))  # Wednesday, 4D
    rec = Recorder(clock, latest={"4d": published_today})
    results = iter([Result(False), Result(True)])  # Telegram was down for the first run
    runs = []

    def run(games):
        runs.append(clock.now())
        return next(results)

    serve(run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(), max_cycles=1)
    assert runs == [sg(2026, 10, 7, 19, 30), sg(2026, 10, 7, 19, 40)]
    assert rec.notices == []


def test_serve_quiet_run_finishes_at_once():
    # Posting off (dry run): the cycle is done as soon as the draw is stored.
    clock = FakeClock(sg(2026, 10, 5, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=Done(False, True))
    assert rec.runs == [(sg(2026, 10, 5, 19, 30), ("toto",))]
    assert rec.notices == [] and rec.events("WAIT") == []


def test_serve_does_not_rerun_a_draw_already_stored_and_posted_after_a_restart():
    # Sat 3 Oct: the 4D draw was stored and posted at 7.30pm, the container was killed at 7.45pm
    # and restarted at 7.50pm, inside the retry window.
    clock = FakeClock(sg(2026, 10, 3, 19, 50))
    rec = Recorder(clock, latest={"4d": published_today})
    done = Done(True)
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=1, done_fn=done)
    assert rec.runs == []
    assert done.calls == [("4d", date(2026, 10, 3))]
    assert "4D result for Sat 3 Oct 2026 was already handled by an earlier run, so it is not run again" in \
        rec.events("SCHEDULE")
    assert rec.notices == [] and rec.events("WAIT") == [] and rec.events("ERROR") == []


def test_serve_runs_each_game_as_soon_as_it_is_out():
    # Sat 10 Oct: 4D is out at 7.30pm, the special TOTO draw only at 8pm.
    clock = FakeClock(sg(2026, 10, 10, 9, 0))
    rec = Recorder(clock, latest={"4d": published_today,
                                  "toto": lambda day: day if clock.now() >= sg(2026, 10, 10, 20, 0) else previous_draw(day)},
                   state={"next_draws": {"toto": "2026-10-10"}})
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 10, 19, 30), ("4d",)), (sg(2026, 10, 10, 20, 0), ("toto",))]
    assert rec.notices == []


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


def test_serve_follows_only_the_chosen_games():
    clock = FakeClock(sg(2026, 10, 10, 9, 0))  # Saturday, a 4D day
    rec = Recorder(clock, latest={"4d": published_today, "toto": published_today})
    serve(rec.run, rec.check, rec.refresh, rec.notify, rec.log, clock.now, clock.sleep, SchedulerConfig(),
          max_cycles=2, games=("toto",))
    assert rec.runs == [] and rec.checks == []
    assert rec.events("SCHEDULE")[0].endswith("until results are out. Only TOTO is followed")
    assert rec.events("SCHEDULE").count("No TOTO draw today") == 2


def test_serve_remembers_a_special_draw_after_a_restart(tmp_path):
    # Thursday's run stored the special TOTO draw on Friday. The container restarts on Friday
    # morning and by 7.30pm the next draw page already shows Monday's draw.
    from huatbot import constants
    from huatbot.runner import refresh_next_draws
    from huatbot.vault import Vault
    from tests.htmlgen import FakeFetcher, fourd_next_draw_html, toto_next_draw_html

    vault = Vault(tmp_path / "vault")
    vault.save_state({"next_draws": {"toto": {"draw_datetime": "2026-10-09T18:30:00+08:00"},
                                     "checked_at": "2026-10-08T19:45:00+08:00"}})
    fetcher = FakeFetcher({
        constants.TOTO_NEXT_DRAW_URL: toto_next_draw_html(sg(2026, 10, 12, 18, 30), 1_000_000),
        constants.FOURD_NEXT_DRAW_URL: fourd_next_draw_html(sg(2026, 10, 10, 18, 30)),
    })
    clock = FakeClock(sg(2026, 10, 9, 10, 0))
    rec = Recorder(clock, latest={"toto": published_today})
    rec.refresh = lambda: refresh_next_draws(vault, fetcher, now=clock.now())
    rec.serve(cycles=1)
    assert rec.runs == [(sg(2026, 10, 9, 19, 30), ("toto",))]
    state = vault.load_state()
    assert state["next_draws"]["toto"]["draw_datetime"].startswith("2026-10-12")
    assert state["upcoming_draws"]["toto"] == ["2026-10-09", "2026-10-12"]
