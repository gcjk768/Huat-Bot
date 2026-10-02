"""Tests for huatbot.commentary. claude is never run: a fake runner returns canned replies."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from huatbot.commentary import (
    MAX_CHARS,
    allowed_numbers,
    build_commentary,
    build_prompt,
    figures_from_context,
    number_tokens,
    validate_commentary,
)
from huatbot.models import (
    BacktestResult,
    BuySignal,
    Context,
    FourDPick,
    NextFourD,
    NextToto,
    Plan,
    PlanLine,
    StrategyScore,
    TotoPick,
)
from huatbot.store import empty_fourd, empty_toto
from huatbot.textfmt import DASH_CHARS

SG = ZoneInfo("Asia/Singapore")
NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)

FIGURES = {
    "next_toto": {"draw_time": "Mon 5 Oct 2026, 6.30pm", "estimated_jackpot": "$3,500,000", "draw_type": "cascade"},
    "buy_signal": {"label": "HIGH", "return_per_dollar": "$0.72", "draws_with_no_group_1_winner": 3},
    "my_tickets": {"tickets": 4, "spent": "$40", "won": "$20", "net": "minus $20"},
}


class FakeRunner:
    """Stands in for subprocess.run: records the call and returns (or raises) a canned result."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "", exc: BaseException | None = None):
        self.stdout, self.returncode, self.stderr, self.exc = stdout, returncode, stderr, exc
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout, stderr=self.stderr)


def no_dash(s: str) -> bool:
    return not any(ch in s for ch in DASH_CHARS)


# figures_from_context


@pytest.fixture
def full_ctx(settings, rules, toto_df, fourd_df, next_toto, next_fourd):
    toto_plan = Plan(game="TOTO", budget=10.0, lines=[
        PlanLine("TOTO", "Low Crowd", "3 11 19 27 38 45", "Ordinary", 1.0, "least crowded numbers"),
        PlanLine("TOTO", "Balanced", "2 13 20 31 40 44", "Ordinary", 1.0, "usual shape"),
        PlanLine("TOTO", "System 7", "3 11 19 27 38 41 45", "System 7", 7.0, "seven numbers"),
    ])
    fourd_plan = Plan(game="4D", budget=5.0, lines=[
        PlanLine("4D", "Hot Digits", "1234", "Big", 1.0, "most frequent digits"),
        PlanLine("4D", "Digit Set", "1347", "iBet Big", 1.0, "common digit set"),
    ])
    backtest = BacktestResult(game="toto", draws_tested=300, first_draw=4100, last_draw=4399,
                              random_sets_per_draw=1000, scores=[
        StrategyScore("Hot", 300, 300.0, 120.0, 20, 50.0, 43.0, "No better than random (beat 43% of random players)"),
        StrategyScore("Random", 300, 300.0, 130.0, 22, 50.0, None, ""),
    ])
    signal = BuySignal(label="HIGH", jackpot=3_500_000.0, draw_type="cascade", no_winner_streak=3,
                       boards_estimate=4_000_000.0, boards_method="median", ev_per_dollar=0.7234,
                       ev_breakdown={"total": 0.7234}, reason="cascade draw")
    return Context(
        now=NOW, settings=settings, rules=rules, toto=toto_df, fourd=fourd_df,
        next_toto=next_toto, next_fourd=next_fourd, buy_signal=signal,
        toto_plan=toto_plan, fourd_plan=fourd_plan, toto_backtest=backtest,
        toto_picks=[TotoPick("Hot", [1, 2, 3, 4, 5, 6], "hot")],
        fourd_picks=[FourDPick("Random", "0042", "Big", "random")],
        ledger_totals={"spent": 40.0, "won": 20.0, "net": -20.0, "pending_cost": 0.0, "tickets": 4, "settled": 4},
    )


def test_figures_from_full_context(full_ctx, toto_df, fourd_df, next_toto):
    figs = figures_from_context(full_ctx)
    json.dumps(figs)  # JSON safe
    assert set(figs) >= {"next_toto", "buy_signal", "next_4d", "latest_toto", "latest_4d",
                         "toto_suggestions", "4d_suggestions", "toto_backtest", "my_tickets"}

    assert figs["next_toto"]["estimated_jackpot"] == f"${next_toto.jackpot_estimate:,.0f}"
    assert figs["next_toto"]["draw_time"].endswith(", 6.30pm")
    assert figs["buy_signal"] == {"label": "HIGH", "return_per_dollar": "$0.72", "draws_with_no_group_1_winner": 3}

    last = toto_df.iloc[-1]
    latest = figs["latest_toto"]
    assert latest["draw"] == int(last["draw_number"])
    assert latest["numbers"] == " ".join(str(int(last[f"n{i}"])) for i in range(1, 7))
    assert latest["additional"] == int(last["additional"])
    assert latest["group_1_prize"].startswith("$")
    assert isinstance(latest["group_1_winners"], int)
    assert figs["latest_4d"]["first"] == fourd_df.iloc[-1]["first"]

    assert figs["toto_suggestions"] == {
        "buy": ["Low Crowd (Ordinary)", "Balanced (Ordinary)", "System 7"],
        "total_cost": "$9", "budget": "$10",
    }
    assert figs["4d_suggestions"]["buy"] == ["Hot Digits (Big)", "Digit Set (iBet Big)"]
    bt = figs["toto_backtest"]
    assert bt["draws_tested"] == 300
    assert bt["strategies"]["Hot"] == {"verdict": "No better than random (beat 43% of random players)",
                                       "return_per_dollar": "$0.40"}
    assert bt["strategies"]["Random"] == {"return_per_dollar": "$0.43"}
    assert figs["my_tickets"] == {"tickets": 4, "spent": "$40", "won": "$20", "net": "minus $20"}
    assert no_dash(json.dumps(figs, ensure_ascii=False))


def test_figures_respect_games_drawn(full_ctx):
    full_ctx.games_drawn = ("4d",)
    figs = figures_from_context(full_ctx)
    assert "latest_toto" not in figs and "latest_4d" in figs


def test_figures_from_bare_context(settings, rules):
    ctx = Context(now=NOW, settings=settings, rules=rules, toto=empty_toto(), fourd=empty_fourd())
    assert figures_from_context(ctx) == {}


def test_figures_tolerate_missing_fields(settings, rules, toto_df):
    ctx = Context(
        now=NOW, settings=settings, rules=rules, toto=toto_df, fourd=empty_fourd(),
        next_toto=NextToto(draw_datetime=None, jackpot_estimate=None, draw_type="normal", draw_type_hint=None),
        next_fourd=NextFourD(draw_datetime=None),
        buy_signal=BuySignal(label="MEDIUM", jackpot=None, draw_type="normal", no_winner_streak=0,
                             boards_estimate=None, boards_method="", ev_per_dollar=None,
                             reason="jackpot estimate not available"),
        toto_picks=[TotoPick("Hot", [1, 2, 3, 4, 5, 6], "hot"), TotoPick("Overdue", [7, 8, 9, 10, 11, 12], "gap")],
        ledger_totals={"spent": 0.0, "won": 0.0, "net": 0.0, "tickets": 0},
    )
    figs = figures_from_context(ctx)
    json.dumps(figs)
    assert figs["next_toto"] == {"draw_type": "normal"}
    assert figs["buy_signal"] == {"label": "MEDIUM", "draws_with_no_group_1_winner": 0}
    assert "next_4d" not in figs and "latest_4d" not in figs and "my_tickets" not in figs
    assert figs["toto_suggestions"] == {"strategies": ["Hot", "Overdue"]}
    assert "latest_toto" in figs


def test_figures_skip_a_broken_part(settings, rules):
    ctx = Context(now=NOW, settings=settings, rules=rules, toto=empty_toto(), fourd=empty_fourd(),
                  buy_signal=BuySignal("LOW", 1e6, "normal", 1, None, "", 0.5))
    ctx.toto_backtest = SimpleNamespace(scores=[object()], draws_tested=10)  # malformed score
    figs = figures_from_context(ctx)
    assert "toto_backtest" not in figs
    assert figs["buy_signal"]["label"] == "LOW"


# number checking


def test_number_tokens():
    toks = number_tokens("Jackpot $3,500,000 on Mon 5 Oct 2026, 6.30pm; 4D first 0042; $0.72 per $1, 5.4%")
    assert Decimal(3500000) in toks and Decimal(5) in toks and Decimal(2026) in toks
    assert "6.30pm" in toks and Decimal(42) in toks and Decimal("0.72") in toks and Decimal("5.4") in toks
    assert Decimal(4) not in toks  # "4D" is the game name
    assert number_tokens("about $3.5 million or $2m") == [Decimal(3500000), Decimal(2000000)]
    assert number_tokens("at 6:30 PM") == ["6.30pm"]
    assert number_tokens("numbers 3,11,19") == [Decimal(3), Decimal(11), Decimal(19)]


def test_allowed_numbers_includes_figures_and_cents():
    allowed = allowed_numbers(FIGURES)
    for value in ("3500000", "0.72", "72", "3", "4", "40", "20", "5", "2026", "1"):
        assert Decimal(value) in allowed, value
    assert "6.30pm" in allowed
    raw = allowed_numbers({"ev": 0.72345})
    assert Decimal("0.72") in raw and Decimal("0.7") in raw


# validate_commentary


def test_validate_accepts_figures_only_text():
    text = ("The Mon 5 Oct 2026, 6.30pm TOTO draw is a cascade with an estimated jackpot of $3,500,000, "
            "after 3 draws with no Group 1 winner. The buy signal is HIGH at $0.72 back per $1. "
            "Your 4 tickets cost $40 and won $20, so you are minus $20.")
    assert validate_commentary(text, FIGURES) == text


def test_validate_accepts_scaled_and_cent_forms():
    assert validate_commentary("Jackpot about $3.5 million, 72 cents back per dollar.", FIGURES)


def test_whole_number_counts_are_not_scaled_by_100():
    """A streak of 5 draws or 4 tickets must not let an invented $500 or $400 through."""
    figures = {"buy_signal": {"draws_with_no_group_1_winner": 5}, "my_tickets": {"tickets": 4},
               "next_toto": {"draw_time": "Mon 5 Oct 2026, 6.30pm"}, "rpd": "$0.66"}
    allowed = allowed_numbers(figures)
    assert Decimal(500) not in allowed and Decimal(400) not in allowed and Decimal(100) not in allowed
    assert Decimal(66) in allowed  # "$0.66" may still be written as 66 cents
    assert validate_commentary("After 5 draws with no winner, a $500 prize is waiting.", figures) is None
    assert validate_commentary("Your 4 tickets won $400.", figures) is None
    assert validate_commentary("After 5 draws with no winner, 66 cents back per $1.", figures)


def test_validate_rejects_invented_numbers():
    assert validate_commentary("The jackpot is $3,600,000 tonight.", FIGURES) is None
    assert validate_commentary("You have a 1 in 54 chance, about 2% better.", FIGURES) is None
    assert validate_commentary("Return is $0.75 per $1.", FIGURES) is None


def test_validate_removes_dashes():
    out = validate_commentary("A cascade draw — jackpot $3,500,000 – signal HIGH - net -$20.", FIGURES)
    assert out is not None and no_dash(out)
    assert "minus $20" in out and "$3,500,000" in out


def test_validate_cleans_markdown_and_whitespace():
    out = validate_commentary("```\n## Note\n**Signal is HIGH.**\n\n  Jackpot   `$3,500,000`.\n```", FIGURES)
    assert out == "Note Signal is HIGH. Jackpot $3,500,000."
    assert validate_commentary('"Signal is HIGH."', FIGURES) == "Signal is HIGH."


def test_validate_caps_length():
    sentence = "The buy signal is HIGH and the draw is a cascade. "
    out = validate_commentary(sentence * 20, FIGURES)
    assert out is not None and len(out) <= MAX_CHARS
    assert out.endswith(".")
    words = validate_commentary("word " * 200, FIGURES)
    assert words is not None and len(words) <= MAX_CHARS


@pytest.mark.parametrize("reply", ["", "   ", "— —", "3 3 3", None])
def test_validate_rejects_empty(reply):
    assert validate_commentary(reply, FIGURES) is None


# build_commentary


def test_build_commentary_off_by_default(monkeypatch):
    monkeypatch.delenv("COMMENTARY", raising=False)
    runner = FakeRunner("text")
    assert build_commentary(FIGURES, runner=runner) is None
    assert build_commentary(FIGURES, mode="off", runner=runner) is None
    assert runner.calls == []


def test_build_commentary_unknown_mode(monkeypatch):
    runner = FakeRunner("text")
    assert build_commentary(FIGURES, mode="gpt", runner=runner) is None
    assert runner.calls == []


def test_build_commentary_runs_claude(monkeypatch):
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    runner = FakeRunner("  The buy signal is HIGH with a $3,500,000 jackpot.\n")
    out = build_commentary(FIGURES, mode="claude", runner=runner, timeout=33)
    assert out == "The buy signal is HIGH with a $3,500,000 jackpot."
    assert len(runner.calls) == 1
    args, kwargs = runner.calls[0]
    assert args[0] == "claude" and args[1] == "-p" and len(args) == 3
    assert kwargs == {"capture_output": True, "text": True, "timeout": 33}
    prompt = args[2]
    assert prompt == build_prompt(FIGURES)
    assert "60 words" in prompt and '"$3,500,000"' in prompt
    assert no_dash(prompt)


def test_build_commentary_reads_env(monkeypatch):
    monkeypatch.setenv("COMMENTARY", " Claude ")
    monkeypatch.setenv("CLAUDE_BIN", "/usr/local/bin/claude")
    runner = FakeRunner("Signal HIGH.")
    assert build_commentary(FIGURES, runner=runner) == "Signal HIGH."
    assert runner.calls[0][0][0] == "/usr/local/bin/claude"


def test_build_commentary_rejects_invented_number():
    runner = FakeRunner("The jackpot will hit $9,999,999 and you should buy 50 tickets.")
    assert build_commentary(FIGURES, mode="claude", runner=runner) is None


def test_build_commentary_strips_dashes():
    runner = FakeRunner("Big night — a cascade draw with $3,500,000 up for grabs - signal HIGH.")
    out = build_commentary(FIGURES, mode="claude", runner=runner)
    assert out is not None and no_dash(out)
    assert out.startswith("Big night, a cascade draw")


@pytest.mark.parametrize(
    "runner",
    [
        FakeRunner(exc=subprocess.TimeoutExpired(["claude"], 120)),
        FakeRunner(exc=FileNotFoundError("claude")),
        FakeRunner(exc=PermissionError("denied")),
        FakeRunner("Signal HIGH.", returncode=1, stderr="error: not logged in"),
        FakeRunner(""),
    ],
)
def test_build_commentary_failures_return_none(runner):
    assert build_commentary(FIGURES, mode="claude", runner=runner) is None


def test_build_commentary_needs_figures():
    runner = FakeRunner("Signal HIGH.")
    assert build_commentary({}, mode="claude", runner=runner) is None
    assert runner.calls == []


def test_end_to_end_with_context(full_ctx):
    figs = figures_from_context(full_ctx)
    latest = figs["latest_toto"]
    reply = (f"Draw {latest['draw']} came out {latest['numbers']}. Next is a cascade, the signal is HIGH "
             f"at $0.72 per $1, and the Hot strategy did no better than random.")
    out = build_commentary(figs, mode="claude", runner=FakeRunner(reply))
    assert out == reply
