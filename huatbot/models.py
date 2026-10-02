"""Shared data structures. Every module talks through these, so keep them stable."""
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

FOURD_STARTER_COLUMNS = [f"starter_{i}" for i in range(1, 11)]
FOURD_CONSOLATION_COLUMNS = [f"consolation_{i}" for i in range(1, 11)]
FOURD_NUMBER_COLUMNS = ["first", "second", "third"] + FOURD_STARTER_COLUMNS + FOURD_CONSOLATION_COLUMNS
FOURD_COLUMNS = ["draw_number", "draw_date"] + FOURD_NUMBER_COLUMNS + ["fetched_at"]
"""fourd.csv, one row per draw, sorted by draw_number ascending.

All 23 number columns are 4 character strings with leading zeros kept ("0042").
A missing number is the empty string.
"""

LEDGER_COLUMNS = [
    "ticket_id", "game", "draw_date", "draw_number", "numbers", "bet_type",
    "cost", "units", "status", "result", "winnings", "added_at", "checked_at", "source",
]
"""ledger.csv, one row per ticket.

ticket_id    stable hash of the ticket line (see tickets.ticket_id)
game         "TOTO" or "4D"
draw_date    ISO YYYY-MM-DD
draw_number  int or empty until the draw is known
numbers      TOTO: ascending numbers separated by single spaces. 4D: the 4 digit string
bet_type     TOTO: "Ordinary" or "System 7".."System 12". 4D: "Big", "Small", "iBet Big", "iBet Small"
cost         float dollars paid
units        float stake multiplier (TOTO: cost / boards, 4D: dollars staked)
status       "pending", "settled", "invalid" or "no_draw"
result       short plain text such as "Group 7 x1" or "No prize"
winnings     float dollars, 0 until settled
added_at, checked_at  ISO timestamps
source       the raw line from Tickets.md
"""

TICKET_STATUSES = ("pending", "settled", "invalid", "no_draw")


@dataclass
class PrizeRules:
    """Prize structure in force. ``confirmed`` is True only if read from the official pages."""

    pool_share_of_sales: float = C.TOTO_POOL_SHARE_OF_SALES
    group_pool_pct: dict[int, float] = field(default_factory=lambda: dict(C.TOTO_GROUP_POOL_PCT))
    fixed_prizes: dict[int, float] = field(default_factory=lambda: dict(C.TOTO_FIXED_PRIZES))
    min_group1: float = C.TOTO_MIN_GROUP1
    fourd_prizes: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in C.FOURD_PRIZES.items()}
    )
    # Published iBet prize per $1, keyed by bet ("big"/"small"), then permutations (4, 6, 12, 24),
    # then tier. Empty means "not read from the site": callers then divide the straight prize by
    # the permutation count and round down to a whole dollar.
    ibet_prizes: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    toto_confirmed: bool = False
    fourd_confirmed: bool = False
    source_note: str = "built in values"
    checked_at: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.toto_confirmed and self.fourd_confirmed


@dataclass
class Settings:
    """User settings, read from Settings.md frontmatter in the vault."""

    toto_budget: float = 10.0
    fourd_budget: float = 5.0
    jackpot_alert: float = 3_000_000.0
    alert_on_special_draws: bool = True  # cascade, Hongbao or special draws count as alerts
    toto_start_draw: int = C.TOTO_FIRST_CURRENT_FORMAT_DRAW
    fourd_history_draws: int = 1000
    backtest_draws: int = 300
    random_sets_per_draw: int = 1000
    offer_system7: bool = True
    draw_notes_backfill: int = 50  # how many recent draws get their own vault note on first run
    warnings: list[str] = field(default_factory=list)  # problems found while reading settings


@dataclass
class NextToto:
    draw_datetime: datetime | None  # timezone aware, Asia/Singapore
    jackpot_estimate: float | None
    draw_type: str  # one of constants.TOTO_DRAW_TYPES, decided by fetch.next_draws
    draw_type_hint: str | None  # word found on the page ("cascade", "hongbao", "special") or None
    raw_text: str = ""


@dataclass
class NextFourD:
    draw_datetime: datetime | None
    raw_text: str = ""


@dataclass
class UpdateResult:
    game: str  # "toto" or "4d"
    new_draws: list[int] = field(default_factory=list)
    latest_on_site: int | None = None
    latest_in_csv: int | None = None
    failed_draws: list[int] = field(default_factory=list)
    verified: bool = False  # newest CSV row matches the latest draw on the site
    messages: list[str] = field(default_factory=list)


@dataclass
class PrizeResult:
    amount: float = 0.0
    groups: dict[Any, int] = field(default_factory=dict)  # TOTO: group -> boards won. 4D: tier -> hits
    detail: str = "No prize"


@dataclass
class TotoPick:
    name: str  # "Hot", "Overdue", "Balanced", "Low Crowd"
    numbers: list[int]  # 6 ascending numbers
    reason: str


@dataclass
class FourDPick:
    name: str  # "Hot Digits", "Repeat Winner", "Digit Set", "Cold Digits", "Random"
    number: str  # 4 digit string
    bet_type: str  # "Big", "Small", "iBet Big" or "iBet Small"
    reason: str


@dataclass
class PlanLine:
    game: str  # "TOTO" or "4D"
    label: str  # strategy name, or "System 7"
    numbers: str  # display string, e.g. "3 11 19 27 38 45" or "1234"
    bet_type: str
    cost: float
    reason: str


@dataclass
class Plan:
    game: str
    budget: float
    lines: list[PlanLine] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    alternative: "Plan | None" = None  # e.g. a System 7 option that fits the budget instead

    @property
    def total(self) -> float:
        return float(sum(line.cost for line in self.lines))


@dataclass
class BuySignal:
    label: str  # "HIGH", "MEDIUM" or "LOW"
    jackpot: float | None
    draw_type: str
    no_winner_streak: int  # draws in a row, newest first, with no Group 1 winner
    boards_estimate: float | None
    boards_method: str
    ev_per_dollar: float | None
    ev_breakdown: dict[str, float] = field(default_factory=dict)  # g1, g2, g3, g4, fixed, total
    reason: str = ""


@dataclass
class StrategyScore:
    name: str
    draws: int
    cost: float
    winnings: float
    wins: int  # draws with any prize
    best_prize: float
    percentile_vs_random: float | None  # 0 to 100, where this strategy's total sits among random players
    verdict: str

    @property
    def return_per_dollar(self) -> float:
        return self.winnings / self.cost if self.cost else 0.0


@dataclass
class BacktestResult:
    game: str  # "toto" or "4d"
    draws_tested: int
    first_draw: int | None
    last_draw: int | None
    random_sets_per_draw: int
    scores: list[StrategyScore] = field(default_factory=list)  # includes a "Random" row
    notes: list[str] = field(default_factory=list)


@dataclass
class Ticket:
    game: str  # "TOTO" or "4D"
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
    fourd: pd.DataFrame
    next_toto: NextToto | None = None
    next_fourd: NextFourD | None = None
    buy_signal: BuySignal | None = None
    # analysis_toto outputs
    toto_frequency: pd.DataFrame | None = None  # index 1..49, columns all, last100, last50
    toto_overdue: pd.Series | None = None
    toto_pairs: list[tuple[tuple[int, int], int]] = field(default_factory=list)
    toto_shape: dict[str, Any] = field(default_factory=dict)
    toto_chi: dict[str, Any] = field(default_factory=dict)
    crowd_scores: pd.Series | None = None
    crowd_diag: dict[str, Any] = field(default_factory=dict)
    # analysis_fourd outputs
    fourd_position_freq: pd.DataFrame | None = None
    fourd_repeats: list[tuple[str, int, Any]] = field(default_factory=list)
    fourd_digit_sets: list[tuple[str, int]] = field(default_factory=list)
    fourd_chi: dict[str, Any] = field(default_factory=dict)
    fourd_bet_values: dict[str, Any] = field(default_factory=dict)
    # suggestions
    toto_picks: list[TotoPick] = field(default_factory=list)
    fourd_picks: list[FourDPick] = field(default_factory=list)
    toto_plan: Plan | None = None
    fourd_plan: Plan | None = None
    # backtests
    toto_backtest: BacktestResult | None = None
    fourd_backtest: BacktestResult | None = None
    # tickets
    ledger: pd.DataFrame | None = None
    ledger_totals: dict[str, float] = field(default_factory=dict)  # spent, won, net, pending_cost
    settled_this_run: list[dict[str, Any]] = field(default_factory=list)  # ledger rows settled this run
    bad_ticket_lines: list[Ticket] = field(default_factory=list)
    # run bookkeeping
    games_drawn: tuple[str, ...] = ("toto", "4d")  # games whose latest result this run reports
    new_draws: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    commentary: str | None = None


@dataclass
class RunResult:
    ok: bool
    messages: list[str] = field(default_factory=list)  # Telegram messages built this run
    posted: bool = False
    new_draws: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    report_path: str | None = None
