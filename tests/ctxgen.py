"""Build a realistic ``models.Context`` from synthetic data, offline, with the real modules.

``make_context`` runs the same pipeline the runner does (``runner.build_context``: sales
estimate, buy signal, jackpot outlook and jackpot history, then the tickets ledger) on
``huatbot.synth`` history, so report, note and integration tests see the shapes and values
production code produces.

Default timeline (Singapore time): the newest TOTO draw is Thu 1 Oct 2026 (draw 4123), the run
happens Thu 1 Oct 2026 at 7.30pm and the next TOTO draw is Mon 5 Oct 2026.
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from huatbot import constants as C
from huatbot import runner, tickets
from huatbot.models import Context, NextToto, PrizeRules, Settings
from huatbot.store import empty_ledger, toto_numbers
from huatbot.synth import synth_next_toto, synth_toto

SG = ZoneInfo(C.SG_TZ_NAME)

TOTO_LAST_DATE = date(2026, 10, 1)  # Thu
TOTO_LAST_DRAW = 4123
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


def _day(value) -> date:
    return pd.Timestamp(value).date()


def _ticket_date(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def tickets_markdown(toto: pd.DataFrame, next_toto_day: date, winning: bool = True, extra: int = 0,
                     bad_lines: bool = True, seed: int = 3) -> str:
    """A Tickets.md body with settled, pending and unreadable tickets.

    winning: one ticket that matches 3 numbers of the newest draw (Group 7). extra: that many
    more random tickets on the newest draw. bad_lines: a line with an impossible date and a 4D
    line (4D is not tracked any more).
    """
    rng = np.random.default_rng(seed)
    t_last, t_prev = toto.iloc[-1], toto.iloc[-2]
    win_nums = toto_numbers(t_last)
    others = [n for n in range(1, 50) if n not in win_nums and n != int(t_last["additional"])]

    rows = [
        (_day(t_prev["draw_date"]), " ".join(map(str, others[:6])), "Ordinary", "$1"),
        (next_toto_day, "3 11 19 27 38 45", "Ordinary", "$1"),
        (next_toto_day, "3 11 19 27 38 45 49", "System 7", "$7"),
    ]
    if winning:
        group7 = sorted(win_nums[:3] + others[6:9])
        rows.insert(0, (_day(t_last["draw_date"]), " ".join(map(str, group7)), "Ordinary", "$1"))
    for _ in range(extra):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 50), size=6, replace=False))
        rows.append((_day(t_last["draw_date"]), " ".join(map(str, nums)), "Ordinary", "$1"))

    lines = ["# My tickets", "", "| Game | Draw date | Numbers | Bet type | Cost |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| TOTO | {_ticket_date(d)} | {n} | {b} | {c} |" for d, n, b, c in rows]
    if bad_lines:
        lines += ["| TOTO | 31 Feb 2026 | 1 2 3 | Ordinary | $1 |",
                  "| 4D | 3 Oct 2026 | 1234 | Big | $1 |"]
    return "\n".join(lines) + "\n"


def make_context(
    tmp_settings: Settings | None = None,
    with_tickets: bool = True,
    *,
    toto_draws: int = 260,
    toto: pd.DataFrame | None = None,
    rules: PrizeRules | None = None,
    next_toto: NextToto | None | str = AUTO,
    new_draws: int | list[int] = 2,
    fetched: bool = True,
    announced: list[tuple[date, str]] | tuple = (),
    winning_ticket: bool = True,
    extra_tickets: int = 0,
    bad_ticket_lines: bool = True,
    commentary: str | None = None,
    warnings: list[str] | None = None,
    now: datetime | None = None,
) -> Context:
    """A complete Context built with ``runner.build_context`` and the real ticket modules on
    synthetic history.

    tmp_settings: Settings to use (defaults when None). with_tickets: run a Tickets.md through
    the ledger (settled winners and losers, pending tickets, unreadable lines); False gives an
    empty ledger. next_toto: "auto" for the synthetic next draw, None for "not known", or an
    object to use as is. new_draws: how many of the newest draws count as new (or the list).
    """
    settings = tmp_settings if tmp_settings is not None else Settings()
    rules = rules if rules is not None else PrizeRules()
    toto = toto if toto is not None else toto_history(toto_draws)
    now = now or RUN_AT

    nt = synth_next_toto(toto) if isinstance(next_toto, str) else next_toto
    if isinstance(new_draws, int):
        new = [int(n) for n in toto["draw_number"].iloc[-new_draws:]] if new_draws else []
    else:
        new = list(new_draws)

    ctx = runner.build_context(toto, settings, rules, now=now, next_toto=nt, new_draws=new,
                               warnings=warnings, fetched=fetched, announced=announced)
    ctx.commentary = commentary

    ledger = empty_ledger()
    if with_tickets:
        next_day = (nt.draw_datetime.date() if nt is not None and nt.draw_datetime
                    else _day(toto["draw_date"].iloc[-1]) + timedelta(days=4))
        text = tickets_markdown(toto, next_day, winning=winning_ticket, extra=extra_tickets,
                                bad_lines=bad_ticket_lines)
        parsed = tickets.parse_tickets(text)
        ctx.bad_ticket_lines = [t for t in parsed if t.error]
        ledger = tickets.sync_ledger(ledger, parsed, now)
        ledger, ctx.settled_this_run = tickets.settle_ledger(ledger, toto, rules, now)
    ctx.ledger = ledger
    ctx.ledger_totals = tickets.ledger_totals(ledger)
    return ctx


def variant(ctx: Context, **changes) -> Context:
    """A shallow copy of ``ctx`` with some fields replaced (cheap, for test variations)."""
    return dataclasses.replace(ctx, **changes)


def rebuilt(ctx: Context, next_toto: NextToto | None) -> Context:
    """``ctx`` with another next draw: buy signal, outlook and history worked out again."""
    fresh = runner.build_context(ctx.toto, ctx.settings, ctx.rules, now=ctx.now, next_toto=next_toto,
                                 new_draws=ctx.new_draws, warnings=[], fetched=ctx.fetched)
    return variant(ctx, next_toto=next_toto, buy_signal=fresh.buy_signal, outlook=fresh.outlook,
                   history=fresh.history)


def cascade_next_toto(ctx: Context, jackpot: float | None = 4_500_000.0) -> Context:
    """``ctx`` with the next TOTO draw announced as a cascade draw (signal and outlook redone)."""
    base = ctx.next_toto
    when = base.draw_datetime if base is not None else datetime.combine(TOTO_LAST_DATE + timedelta(days=4),
                                                                       time(18, 30), tzinfo=SG)
    nt = NextToto(draw_datetime=when, jackpot_estimate=jackpot, draw_type="cascade", draw_type_hint="cascade",
                  raw_text="Cascade draw")
    return rebuilt(ctx, nt)


def unknown_jackpot(ctx: Context) -> Context:
    """``ctx`` with no jackpot estimate on the next draw page (signal and outlook redone, so the
    jackpot is worked out from the stored results)."""
    base = ctx.next_toto
    nt = NextToto(draw_datetime=base.draw_datetime if base is not None else None, jackpot_estimate=None,
                  draw_type="normal", draw_type_hint=None, raw_text="")
    return rebuilt(ctx, nt)
