"""Shared data structures (TOTO only). Every module talks through these, so keep them stable."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from . import constants as C

# CSV schemas

TOTO_GROUP_COLUMNS = [f"g{g}_{k}" for g in range(1, 8) for k in ("share", "winners")]
TOTO_COLUMNS = (
    ["draw_number", "draw_date"]
    + [f"n{i}" for i in range(1, 7)]
    + ["additional", "jackpot"]
    + TOTO_GROUP_COLUMNS
    + ["draw_type", "fetched_at"]
)
"""toto.csv, one row per draw, sorted by draw_number ascending.

draw_number  int
draw_date    pandas Timestamp in memory, ISO YYYY-MM-DD on disk
n1..n6       int, ascending
additional   int
jackpot      float, the Group 1 prize shown on the result page (td.jackpotPrize)
gK_share     float share amount for group K, NaN when the page shows no amount
gK_winners   int number of winning shares for group K, 0 when the page shows none
draw_type    one of constants.TOTO_DRAW_TYPES
fetched_at   ISO timestamp string
"""

LEDGER_COLUMNS = [
    "ticket_id", "game", "draw_date", "draw_number", "numbers", "bet_type",
    "cost", "units", "status", "result", "winnings", "added_at", "checked_at", "source",
]
"""ledger.csv, one row per ticket.

ticket_id    stable hash of the ticket line (see tickets.ticket_id)
game         "TOTO" (rows for other games from older versions are kept as "invalid")
draw_date    ISO YYYY-MM-DD
draw_number  int or empty until the draw is known
numbers      ascending numbers separated by single spaces
bet_type     "Ordinary" or "System 7".."System 12"
cost         float dollars paid
units        float stake multiplier (cost / boards)
status       "pending", "settled", "invalid" or "no_draw"
result       short plain text such as "Group 7 x1" or "No prize"
winnings     float dollars, 0 until settled
added_at, checked_at  ISO timestamps
source       the raw line from Tickets.md
"""

TICKET_STATUSES = ("pending", "settled", "invalid", "no_draw")


@dataclass
class PrizeRules:
    """TOTO prize structure in force. ``confirmed`` is True only if read from the official page."""

    pool_share_of_sales: float = C.TOTO_POOL_SHARE_OF_SALES
    group_pool_pct: dict[int, float] = field(default_factory=lambda: dict(C.TOTO_GROUP_POOL_PCT))
    fixed_prizes: dict[int, float] = field(default_factory=lambda: dict(C.TOTO_FIXED_PRIZES))
    min_group1: float = C.TOTO_MIN_GROUP1
    toto_confirmed: bool = False
    source_note: str = "built in values"
    checked_at: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.toto_confirmed


@dataclass
class Settings:
    """User settings, read from Settings.md frontmatter in the vault."""

    jackpot_alert: float = 3_000_000.0
    alert_on_special_draws: bool = True  # cascade, Hongbao or special draws count as alerts
    toto_start_draw: int = C.TOTO_FIRST_CURRENT_FORMAT_DRAW
    draw_notes_backfill: int = 50  # how many recent draws get their own vault note on first run
    warnings: list[str] = field(default_factory=list)  # problems found while reading settings


@dataclass
class NextToto:
    draw_datetime: datetime | None  # timezone aware, Asia/Singapore
    jackpot_estimate: float | None
    draw_type: str  # one of constants.TOTO_DRAW_TYPES, decided by fetch.fetch_next_draw
    draw_type_hint: str | None  # word found on the page ("cascade", "hongbao", "special") or None
    raw_text: str = ""


@dataclass
class UpdateResult:
    game: str = "toto"
    new_draws: list[int] = field(default_factory=list)
    latest_on_site: int | None = None
    latest_in_csv: int | None = None
    failed_draws: list[int] = field(default_factory=list)
    repaired_draws: list[int] = field(default_factory=list)  # stored draws re-read because they were incomplete
    verified: bool = False  # newest CSV row matches the latest draw on the site
    messages: list[str] = field(default_factory=list)


@dataclass
class PrizeResult:
    amount: float = 0.0
    groups: dict[int, int] = field(default_factory=dict)  # group -> boards won
    detail: str = "No prize"


@dataclass
class BuySignal:
    label: str  # "HIGH", "MEDIUM" or "LOW"
    jackpot: float | None
    draw_type: str
    no_winner_streak: int  # draws in a row, newest first, with no Group 1 winner
    boards_estimate: float | None
    boards_method: str
    ev_per_dollar: float | None
    ev_breakdown: dict[str, float] = field(default_factory=dict)  # g1, g2, g3, g4, fixed, cascade, total
    reason: str = ""


@dataclass
class OutlookStep:
    """One upcoming draw in the jackpot projection, assuming nobody wins Group 1 before it."""

    index: int  # 1 = the next draw
    draw_date: date | None  # regular draw day (Monday or Thursday), or the announced date
    jackpot: float  # estimated Group 1 prize at this draw
    boards: float | None  # estimated boards sold at this jackpot level
    chance_reached: float  # chance the jackpot is still unwon when this draw comes (1.0 for the next draw)
    chance_won: float  # chance at least one board wins Group 1 at this draw, given it is reached
    cascade: bool = False  # the jackpot cascades to Group 2 if nobody wins at this draw
    ev_per_dollar: float | None = None  # average return per $1 board at this draw


@dataclass
class JackpotOutlook:
    """What the next TOTO draws are likely to do and where the next big prize is."""

    jackpot: float | None  # next draw's estimated Group 1 prize
    draw_type: str
    snowball_draws: int  # draws in a row the current jackpot has rolled over
    draws_to_cascade: int | None  # 1 = the next draw is the cascade draw; None on a Hongbao or special draw
    steps: list[OutlookStep] = field(default_factory=list)
    special_draws: list[tuple[date, str]] = field(default_factory=list)  # announced upcoming draws (date, type)
    notes: list[str] = field(default_factory=list)

    @property
    def biggest(self) -> OutlookStep | None:
        """The projected draw with the largest jackpot (the next big prize if nobody wins first)."""
        return max(self.steps, key=lambda s: s.jackpot) if self.steps else None


@dataclass
class JackpotHistory:
    """How TOTO jackpots have behaved in the stored history."""

    draws: int = 0
    won_draws: int = 0  # draws where Group 1 had at least one winner
    cascades: int = 0  # cascade draws where nobody won Group 1
    average_run: float | None = None  # draws a jackpot lasts on average (1 = won at the first draw)
    typical_won: float | None = None  # median Group 1 prize on draws where it was won
    biggest: list[dict[str, Any]] = field(default_factory=list)  # draw_number, draw_date, jackpot, winners, draw_type
    last_won: dict[str, Any] | None = None  # newest draw with a Group 1 winner

    @property
    def won_share(self) -> float | None:
        return self.won_draws / self.draws if self.draws else None


@dataclass
class Ticket:
    game: str  # "TOTO"
    draw_date: date | None
    numbers: str  # normalised display string (see LEDGER_COLUMNS)
    bet_type: str  # normalised (see LEDGER_COLUMNS)
    cost: float
    line_no: int
    source: str
    error: str | None = None  # set when the line could not be read


@dataclass
class Context:
    """Everything a report, note or message needs. Built by runner.build_context."""

    now: datetime
    settings: Settings
    rules: PrizeRules
    toto: pd.DataFrame
    next_toto: NextToto | None = None
    buy_signal: BuySignal | None = None
    outlook: JackpotOutlook | None = None
    history: JackpotHistory | None = None
    # tickets
    ledger: pd.DataFrame | None = None
    ledger_totals: dict[str, float] = field(default_factory=dict)  # spent, won, net, pending_cost
    settled_this_run: list[dict[str, Any]] = field(default_factory=list)  # ledger rows settled this run
    bad_ticket_lines: list[Ticket] = field(default_factory=list)
    # run bookkeeping
    fetched: bool = False  # this run read the results from the site
    new_draws: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    commentary: str | None = None


@dataclass
class RunResult:
    ok: bool
    messages: list[str] = field(default_factory=list)  # Telegram messages built this run
    posted: bool = False
    new_draws: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    report_path: str | None = None
