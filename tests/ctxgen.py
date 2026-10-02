"""Build a realistic ``models.Context`` from synthetic data, offline, with the real modules.

``make_context`` runs the same pipeline the runner does (analysis, crowd scores, picks, plans,
buy signal, backtests, tickets ledger) on ``huatbot.synth`` history, so report, note and
integration tests see the shapes and values production code produces. Backtests use a small
window and few random players to stay fast.

Default timeline (Singapore time): the newest TOTO draw is Thu 1 Oct 2026 (draw 4123), the newest
4D draw Wed 30 Sep 2026 (draw 5432), the run happens Thu 1 Oct 2026 at 7.30pm, the next TOTO draw
is Mon 5 Oct 2026 and the next 4D draw Sat 3 Oct 2026.
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from huatbot import (
    analysis_fourd,
    analysis_toto,
    backtest,
    buysignal,
    strategies,
    suggest,
    tickets,
)
from huatbot import constants as C
from huatbot.models import Context, NextFourD, NextToto, PrizeRules, Settings
from huatbot.store import empty_ledger, fourd_numbers, toto_numbers
from huatbot.synth import synth_fourd, synth_next_fourd, synth_next_toto, synth_toto

SG = ZoneInfo(C.SG_TZ_NAME)

TOTO_LAST_DATE = date(2026, 10, 1)  # Thu
TOTO_LAST_DRAW = 4123
FOURD_LAST_DATE = date(2026, 9, 30)  # Wed
FOURD_LAST_DRAW = 5432
RUN_AT = datetime(2026, 10, 1, 19, 30, tzinfo=SG)

AUTO = "auto"  # sentinel: let make_context build this part synthetically


def start_date_for(end: date, weekdays: tuple[int, ...], n: int) -> date:
    """The first of ``n`` draw days (on ``weekdays``) that end exactly on ``end``."""
    if end.weekday() not in weekdays:
        raise ValueError(f"{end} is not a draw day")
    d, count = end, 1
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() in weekdays:
            count += 1
    return d


def toto_history(n_draws: int = 260, seed: int = 7) -> pd.DataFrame:
    """Synthetic TOTO draws ending on ``TOTO_LAST_DATE`` with draw number ``TOTO_LAST_DRAW``."""
    start = start_date_for(TOTO_LAST_DATE, C.TOTO_WEEKDAYS, n_draws)
    return synth_toto(n_draws=n_draws, start_draw=TOTO_LAST_DRAW - n_draws + 1, start_date=start, seed=seed)


def fourd_history(n_draws: int = 220, seed: int = 11) -> pd.DataFrame:
    """Synthetic 4D draws ending on ``FOURD_LAST_DATE`` with draw number ``FOURD_LAST_DRAW``."""
    start = start_date_for(FOURD_LAST_DATE, C.FOURD_WEEKDAYS, n_draws)
    return synth_fourd(n_draws=n_draws, start_draw=FOURD_LAST_DRAW - n_draws + 1, start_date=start, seed=seed)


def _day(value) -> date:
    return pd.Timestamp(value).date()


def _ticket_date(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def tickets_markdown(toto: pd.DataFrame, fourd: pd.DataFrame, next_toto_day: date, next_fourd_day: date,
                     winning: bool = True, extra: int = 0, bad_lines: bool = True, seed: int = 3) -> str:
    """A Tickets.md body with settled, pending and unreadable tickets.

    winning: one TOTO ticket that matches 3 numbers of the newest draw (Group 7) and one 4D Big
    ticket on the newest 1st Prize. extra: that many more random TOTO tickets on the newest draw.
    """
    rng = np.random.default_rng(seed)
    t_last, t_prev = toto.iloc[-1], toto.iloc[-2]
    f_last = fourd.iloc[-1]
    win_nums = toto_numbers(t_last)
    others = [n for n in range(1, 50) if n not in win_nums and n != int(t_last["additional"])]
    losing_toto = others[:6]
    drawn_4d = {n for tier in fourd_numbers(f_last).values() for n in tier}
    losing_4d = next(f"{n:04d}" for n in range(10_000) if f"{n:04d}" not in drawn_4d)

    rows = [
        ("TOTO", _day(t_prev["draw_date"]), " ".join(map(str, losing_toto)), "Ordinary", "$1"),
        ("TOTO", next_toto_day, "3 11 19 27 38 45", "Ordinary", "$1"),
        ("TOTO", next_toto_day, "3 11 19 27 38 45 49", "System 7", "$7"),
        ("4D", _day(f_last["draw_date"]), losing_4d, "Small", "$1"),
        ("4D", next_fourd_day, "0042", "iBet Big", "$1"),
    ]
    if winning:
        group7 = sorted(win_nums[:3] + others[6:9])
        rows.insert(0, ("TOTO", _day(t_last["draw_date"]), " ".join(map(str, group7)), "Ordinary", "$1"))
        rows.insert(1, ("4D", _day(f_last["draw_date"]), str(f_last["first"]), "Big", "$2"))
    for _ in range(extra):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 50), size=6, replace=False))
        rows.append(("TOTO", _day(t_last["draw_date"]), " ".join(map(str, nums)), "Ordinary", "$1"))

    lines = ["# My tickets", "", "| Game | Draw date | Numbers | Bet type | Cost |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {g} | {_ticket_date(d)} | {n} | {b} | {c} |" for g, d, n, b, c in rows]
    if bad_lines:
        lines += ["| TOTO | 31 Feb 2026 | 1 2 3 | Ordinary | $1 |",
                  "| 4D | 2026-10-03 | 12345 | Big | $1 |"]
    return "\n".join(lines) + "\n"


def make_context(
    tmp_settings: Settings | None = None,
    with_tickets: bool = True,
    *,
    toto_draws: int = 260,
    fourd_draws: int = 220,
    toto: pd.DataFrame | None = None,
    fourd: pd.DataFrame | None = None,
    rules: PrizeRules | None = None,
    next_toto: NextToto | None | str = AUTO,
    next_fourd: NextFourD | None | str = AUTO,
    games_drawn: tuple[str, ...] = ("toto", "4d"),
    new_draws: int | dict[str, list[int]] = 2,
    with_backtest: bool = True,
    backtest_draws: int = 20,
    backtest_random: int = 50,
    winning_ticket: bool = True,
    extra_tickets: int = 0,
    bad_ticket_lines: bool = True,
    commentary: str | None = None,
    warnings: list[str] | None = None,
    now: datetime | None = None,
) -> Context:
    """A complete Context built with the real analysis, strategy, plan, signal, backtest and
    ticket modules on synthetic history.

    tmp_settings: Settings to use (defaults when None). with_tickets: run a Tickets.md through
    the ledger (settled winners and losers, pending tickets, unreadable lines); False gives an
    empty ledger. next_toto / next_fourd: "auto" for the synthetic next draw, None for "not
    known", or an object to use as is. new_draws: how many of the newest draws per drawn game
    count as new (or the dict itself). with_backtest False leaves both backtests None.
    """
    settings = tmp_settings if tmp_settings is not None else Settings()
    rules = rules if rules is not None else PrizeRules()
    toto = toto if toto is not None else toto_history(toto_draws)
    fourd = fourd if fourd is not None else fourd_history(fourd_draws)
    now = now or RUN_AT

    nt = synth_next_toto(toto) if isinstance(next_toto, str) else next_toto
    nf = synth_next_fourd(fourd) if isinstance(next_fourd, str) else next_fourd

    # TOTO analysis
    table = analysis_toto.crowd_table(toto, rules)
    crowd, diag = analysis_toto.crowd_scores(toto, rules, table=table)
    last_draw = toto_numbers(toto.iloc[-1])
    toto_seed = int(toto["draw_number"].iloc[-1]) + 1
    toto_picks = strategies.toto_picks(toto, crowd, seed=toto_seed)

    # 4D analysis
    bet_values = analysis_fourd.bet_type_value(rules)
    fourd_picks = strategies.fourd_picks(fourd, seed=int(fourd["draw_number"].iloc[-1]) + 1)

    ctx = Context(
        now=now,
        settings=settings,
        rules=rules,
        toto=toto,
        fourd=fourd,
        next_toto=nt,
        next_fourd=nf,
        buy_signal=buysignal.buy_signal(nt, toto, settings, rules, table),
        toto_frequency=analysis_toto.frequency_table(toto),
        toto_overdue=analysis_toto.overdue(toto),
        toto_pairs=analysis_toto.top_pairs(toto),
        toto_shape=analysis_toto.shape_stats(toto),
        toto_chi=analysis_toto.chi_square_numbers(toto),
        crowd_scores=crowd,
        crowd_diag=diag,
        fourd_position_freq=analysis_fourd.position_digit_freq(fourd),
        fourd_repeats=analysis_fourd.repeat_winners(fourd),
        fourd_digit_sets=analysis_fourd.digit_set_freq(fourd),
        fourd_chi=analysis_fourd.chi_square_digits(fourd),
        fourd_bet_values=bet_values,
        toto_picks=toto_picks,
        fourd_picks=fourd_picks,
        toto_plan=suggest.toto_plan(toto_picks, settings.toto_budget, crowd, settings.offer_system7, last_draw),
        fourd_plan=suggest.fourd_plan(fourd_picks, settings.fourd_budget, bet_values),
        games_drawn=tuple(games_drawn),
        warnings=list(warnings or []),
        commentary=commentary,
    )

    if with_backtest:
        ctx.toto_backtest = backtest.backtest_toto(toto, rules, n_draws=backtest_draws, n_random=backtest_random)
        ctx.fourd_backtest = backtest.backtest_fourd(fourd, rules, n_draws=backtest_draws, n_random=backtest_random)

    if isinstance(new_draws, dict):
        ctx.new_draws = {k: list(v) for k, v in new_draws.items()}
    else:
        ctx.new_draws = {
            game: [int(n) for n in (toto if game == "toto" else fourd)["draw_number"].iloc[-new_draws:]]
            if new_draws else []
            for game in games_drawn
        }

    ledger = empty_ledger()
    if with_tickets:
        next_toto_day = (nt.draw_datetime.date() if nt is not None and nt.draw_datetime
                         else _day(toto["draw_date"].iloc[-1]) + timedelta(days=4))
        next_fourd_day = (nf.draw_datetime.date() if nf is not None and nf.draw_datetime
                          else _day(fourd["draw_date"].iloc[-1]) + timedelta(days=3))
        text = tickets_markdown(toto, fourd, next_toto_day, next_fourd_day, winning=winning_ticket,
                                extra=extra_tickets, bad_lines=bad_ticket_lines)
        parsed = tickets.parse_tickets(text)
        ctx.bad_ticket_lines = [t for t in parsed if t.error]
        ledger = tickets.sync_ledger(ledger, parsed, now)
        ledger, ctx.settled_this_run = tickets.settle_ledger(ledger, toto, fourd, rules, now)
    ctx.ledger = ledger
    ctx.ledger_totals = tickets.ledger_totals(ledger)
    return ctx


def variant(ctx: Context, **changes) -> Context:
    """A shallow copy of ``ctx`` with some fields replaced (cheap, for test variations)."""
    return dataclasses.replace(ctx, **changes)


def cascade_next_toto(ctx: Context, jackpot: float | None = 4_500_000.0) -> Context:
    """``ctx`` with the next TOTO draw announced as a cascade draw (and the buy signal redone)."""
    base = ctx.next_toto
    when = base.draw_datetime if base is not None else datetime.combine(TOTO_LAST_DATE + timedelta(days=4),
                                                                       time(18, 30), tzinfo=SG)
    nt = NextToto(draw_datetime=when, jackpot_estimate=jackpot, draw_type="cascade", draw_type_hint="cascade",
                  raw_text="Cascade draw")
    signal = buysignal.buy_signal(nt, ctx.toto, ctx.settings, ctx.rules)
    return variant(ctx, next_toto=nt, buy_signal=signal)


def unknown_jackpot(ctx: Context) -> Context:
    """``ctx`` with no jackpot estimate for the next TOTO draw (and the buy signal redone)."""
    base = ctx.next_toto
    nt = NextToto(draw_datetime=base.draw_datetime if base is not None else None, jackpot_estimate=None,
                  draw_type="normal", draw_type_hint=None, raw_text="")
    signal = buysignal.buy_signal(nt, ctx.toto, ctx.settings, ctx.rules)
    return variant(ctx, next_toto=nt, buy_signal=signal)
