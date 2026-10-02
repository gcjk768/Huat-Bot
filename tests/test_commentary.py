"""Tests for huatbot.commentary (TOTO only). claude is never run: a fake runner returns canned replies."""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from huatbot import report
from huatbot.commentary import (
    MAX_CHARS,
    allowed_numbers,
    build_commentary,
    build_prompt,
    figures_from_context,
    is_enabled,
    number_tokens,
    validate_commentary,
)
from huatbot.models import (
    BuySignal,
    Context,
    JackpotHistory,
    JackpotOutlook,
    NextToto,
    OutlookStep,
    PrizeRules,
    Settings,
)
from huatbot.store import empty_toto, toto_numbers
from huatbot.textfmt import DASH_CHARS
from tests.ctxgen import make_context, toto_history, unknown_jackpot, variant

SG = ZoneInfo("Asia/Singapore")
NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)  # Fri, the day after draw 4123
SECTIONS = ["next_toto", "buy_signal", "jackpot_outlook", "latest_toto", "jackpot_history", "my_tickets"]

# No 4 anywhere in these figures, so a reply about "4D" is caught as an invented number.
FIGURES = {
    "next_toto": {"draw_time": "Mon 5 Oct 2026, 6.30pm", "estimated_jackpot": "$3,500,000", "draw_type": "cascade"},
    "buy_signal": {"label": "HIGH", "return_per_dollar": "$0.72"},
    "jackpot_outlook": {"jackpot_rollovers_so_far": 3, "chance_somebody_wins_next_draw": "25%",
                        "biggest_projected_jackpot": "$3,500,000", "biggest_projected_on": "Mon 5 Oct 2026",
                        "chance_jackpot_gets_that_far": "100%", "draws_until_cascade": 1},
    "jackpot_history": {"draws_stored": 260, "share_of_draws_with_a_group_1_winner": "23%",
                        "average_draws_a_jackpot_lasts": 3.0},
    "my_tickets": {"tickets": 6, "spent": "$30", "won": "$10", "net": "minus $20"},
}


class FakeRunner:
    """Stands in for subprocess.run: records the call and returns (or raises) a canned result."""

    def __init__(self, stdout="", returncode: int = 0, stderr: str = "", exc: BaseException | None = None):
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


@pytest.fixture(scope="module")
def toto():
    return toto_history(60)  # newest draw 4123 on Thu 1 Oct 2026


def _outlook() -> JackpotOutlook:
    return JackpotOutlook(
        jackpot=2_100_000.0, draw_type="normal", snowball_draws=1, draws_to_cascade=3,
        steps=[
            OutlookStep(1, date(2026, 10, 5), 2_100_000.0, 3_950_000.0, 1.0, 0.2462, False, 0.4531),
            OutlookStep(2, date(2026, 10, 8), 3_007_000.0, 4_490_000.0, 0.7538, 0.2747, False, 0.5079),
            OutlookStep(3, date(2026, 10, 12), 3_996_000.0, 4_820_000.0, 0.5468, 0.2915, True, 1.0798),
        ])


@pytest.fixture
def full_ctx(toto):
    return Context(
        now=NOW, settings=Settings(), rules=PrizeRules(), toto=toto,
        next_toto=NextToto(draw_datetime=datetime(2026, 10, 5, 18, 30, tzinfo=SG), jackpot_estimate=2_100_000.0,
                           draw_type="normal", draw_type_hint=None),
        buy_signal=BuySignal(label="MEDIUM", jackpot=2_100_000.0, draw_type="normal", no_winner_streak=1,
                             boards_estimate=3_950_000.0, boards_method="median of 85 past draws",
                             ev_per_dollar=0.4531, ev_breakdown={"total": 0.4531}, reason="below your alert"),
        outlook=_outlook(),
        history=JackpotHistory(draws=260, won_draws=61, cascades=26, average_run=2.977, typical_won=1_875_282.0),
        ledger_totals={"spent": 40.0, "won": 20.0, "net": -20.0, "pending_cost": 0.0, "tickets": 4, "settled": 4},
        new_draws=[4122, 4123],
    )


def test_figures_from_full_context(full_ctx, toto):
    figs = figures_from_context(full_ctx)
    json.dumps(figs)  # JSON safe
    assert list(figs) == SECTIONS
    last = toto.iloc[-1]
    assert figs == {
        "next_toto": {"draw_time": "Mon 5 Oct 2026, 6.30pm", "estimated_jackpot": "$2,100,000",
                      "draw_type": "normal"},
        "buy_signal": {"label": "MEDIUM", "return_per_dollar": "$0.45"},
        "jackpot_outlook": {
            "jackpot_rollovers_so_far": 1,
            "chance_somebody_wins_next_draw": "25%",
            "biggest_projected_jackpot": "$3,996,000",  # the cascade draw, not the next one
            "biggest_projected_on": "Mon 12 Oct 2026",
            "chance_jackpot_gets_that_far": "55%",
            "draws_until_cascade": 3,
        },
        "latest_toto": {
            "draw": 4123,
            "date": "Thu 1 Oct 2026",
            "numbers": " ".join(map(str, toto_numbers(last))),
            "additional": int(last["additional"]),
            "group_1_prize": f"${last['jackpot']:,.0f}",
            "group_1_winners": int(last["g1_winners"]),
        },
        "jackpot_history": {"draws_stored": 260, "share_of_draws_with_a_group_1_winner": "23%",
                            "average_draws_a_jackpot_lasts": 3.0},
        "my_tickets": {"tickets": 4, "spent": "$40", "won": "$20", "net": "minus $20"},
    }
    assert no_dash(json.dumps(figs, ensure_ascii=False))


def test_latest_result_only_when_there_are_new_draws(full_ctx):
    figs = figures_from_context(variant(full_ctx, new_draws=[]))
    assert "latest_toto" not in figs
    assert list(figs) == [s for s in SECTIONS if s != "latest_toto"]


def test_outlook_figures_for_a_hongbao_draw_and_no_rollovers(full_ctx):
    out = JackpotOutlook(jackpot=12_000_000.0, draw_type="hongbao", snowball_draws=0, draws_to_cascade=None,
                         steps=[OutlookStep(1, date(2026, 10, 5), 12_000_000.0, 6e6, 1.0, 0.349, False, 0.9)])
    figs = figures_from_context(variant(full_ctx, outlook=out))
    # 0 rollovers is a real figure and stays; a Hongbao draw never cascades, so no countdown.
    assert figs["jackpot_outlook"] == {
        "jackpot_rollovers_so_far": 0,
        "chance_somebody_wins_next_draw": "35%",
        "biggest_projected_jackpot": "$12,000,000",
        "biggest_projected_on": "Mon 5 Oct 2026",
        "chance_jackpot_gets_that_far": "100%",
    }


def test_outlook_with_cents_and_without_steps(full_ctx):
    out = dataclasses.replace(_outlook(), steps=[dataclasses.replace(_outlook().steps[0], jackpot=1_234_567.5)])
    assert figures_from_context(variant(full_ctx, outlook=out))["jackpot_outlook"]["biggest_projected_jackpot"] \
        == "$1,234,567.50"
    empty = JackpotOutlook(jackpot=None, draw_type="normal", snowball_draws=2, draws_to_cascade=2, steps=[])
    assert "jackpot_outlook" not in figures_from_context(variant(full_ctx, outlook=empty))
    assert "jackpot_outlook" not in figures_from_context(variant(full_ctx, outlook=None))


def test_ledger_figures(full_ctx):
    figs = figures_from_context(variant(full_ctx, ledger_totals={"spent": 12.5, "won": 0.0, "net": -12.5,
                                                                 "tickets": 3}))
    assert figs["my_tickets"] == {"tickets": 3, "spent": "$12.50", "won": "$0", "net": "minus $12.50"}
    # No tickets yet: nothing worth commenting on.
    for totals in ({}, {"spent": 0.0, "won": 0.0, "net": 0.0, "tickets": 0}):
        assert "my_tickets" not in figures_from_context(variant(full_ctx, ledger_totals=totals))


def test_figures_from_bare_context():
    ctx = Context(now=NOW, settings=Settings(), rules=PrizeRules(), toto=empty_toto(), new_draws=[4123])
    assert figures_from_context(ctx) == {}


def test_figures_tolerate_missing_fields(toto):
    ctx = Context(
        now=NOW, settings=Settings(), rules=PrizeRules(), toto=toto,
        next_toto=NextToto(draw_datetime=None, jackpot_estimate=None, draw_type="normal", draw_type_hint=None),
        buy_signal=BuySignal(label="MEDIUM", jackpot=None, draw_type="normal", no_winner_streak=0,
                             boards_estimate=None, boards_method="", ev_per_dollar=None,
                             reason="jackpot estimate not available"),
        history=JackpotHistory(draws=1, won_draws=0),
        ledger_totals={"spent": 0.0, "won": 0.0, "net": 0.0, "tickets": 0},
    )
    figs = figures_from_context(ctx)
    json.dumps(figs)
    assert figs == {
        "next_toto": {"draw_type": "normal"},
        "buy_signal": {"label": "MEDIUM"},
        "jackpot_history": {"draws_stored": 1, "share_of_draws_with_a_group_1_winner": "0%"},
    }


def test_figures_skip_a_broken_part(full_ctx, caplog):
    broken = SimpleNamespace(steps=[object()], biggest=object(), snowball_draws=1, draws_to_cascade=3)
    with caplog.at_level(logging.WARNING, logger="huatbot.commentary"):
        figs = figures_from_context(variant(full_ctx, outlook=broken))
    assert "jackpot_outlook" not in figs
    assert list(figs) == [s for s in SECTIONS if s != "jackpot_outlook"]
    assert "Commentary figures: skipped jackpot_outlook (AttributeError)" in caplog.text


def test_figures_skip_next_draw_info_about_a_draw_already_held(full_ctx):
    # The next draw page still shows the draw already stored: its jackpot belongs to that draw.
    held = dataclasses.replace(full_ctx.next_toto, draw_datetime=datetime(2026, 10, 1, 18, 30, tzinfo=SG))
    assert "next_toto" not in figures_from_context(variant(full_ctx, next_toto=held))
    # Stored info from an earlier run about a day already past is not current either.
    past = make_context(with_tickets=False, now=datetime(2026, 10, 6, 9, 0, tzinfo=SG))
    assert "next_toto" not in figures_from_context(past)


def test_figures_agree_with_the_messages():
    """The real pipeline: every figure is the same text the user sees in the Telegram messages."""
    ctx = make_context()
    figs = figures_from_context(ctx)
    msg1, msg2 = report.telegram_messages(ctx)
    assert list(figs) == SECTIONS
    nt, bs, out = figs["next_toto"], figs["buy_signal"], figs["jackpot_outlook"]
    assert f"Next TOTO draw</b> (draw 4124): {nt['draw_time']}" in msg2
    assert f"Estimated jackpot: <b>{nt['estimated_jackpot']}</b>" in msg2
    assert f"Buy signal: <b>{bs['label']}</b>" in msg2
    assert f"Return per $1: <b>{bs['return_per_dollar']}</b>" in msg2
    assert out["jackpot_rollovers_so_far"] == 1 and out["draws_until_cascade"] == 3
    assert "Jackpot rollovers so far: 1 of 3, then it cascades" in msg2
    assert f"Chance somebody wins Group 1 at this draw: <b>{out['chance_somebody_wins_next_draw']}</b>" in msg2
    assert (f"snowballs to about {out['biggest_projected_jackpot']} at the cascade draw on "
            f"{out['biggest_projected_on']} ({out['chance_jackpot_gets_that_far']} chance") in msg2
    assert f"Group 1 was won in {figs['jackpot_history']['share_of_draws_with_a_group_1_winner']} of draws" in msg2
    assert f"Winning numbers: <b>{figs['latest_toto']['numbers']}</b>" in msg1
    assert figs["my_tickets"] == {"tickets": 4, "spent": "$10", "won": "$10", "net": "$0"}
    assert "spent $10, won $10, net <b>$0</b>" in msg1


def test_figures_with_the_jackpot_worked_out_from_past_results():
    ctx = unknown_jackpot(make_context(with_tickets=False))
    figs = figures_from_context(ctx)
    # The page gave no jackpot: the next draw section has none, the outlook carries the estimate.
    assert "estimated_jackpot" not in figs["next_toto"]
    assert figs["jackpot_outlook"]["chance_somebody_wins_next_draw"] == f"{round(ctx.outlook.steps[0].chance_won * 100)}%"
    assert figs["buy_signal"]["label"] == ctx.buy_signal.label


_REMOVED = re.compile(r"4d|fourd|suggest|pick|plan|budget|backtest|strateg|\bhot\b|\bcold\b|overdue|crowd|system",
                      re.I)


def test_figures_hold_nothing_about_removed_features(full_ctx):
    for ctx in (full_ctx, make_context(), make_context(with_tickets=False, new_draws=0)):
        figs = figures_from_context(ctx)
        assert set(figs) <= set(SECTIONS)
        text = json.dumps(figs, ensure_ascii=False)
        assert not _REMOVED.search(text), text


# number checking


def test_number_tokens():
    toks = number_tokens("Jackpot $3,500,000 on Mon 5 Oct 2026, 6.30pm; draw 4123; $0.72 per $1, 5.4%")
    assert toks == [Decimal(3500000), Decimal(5), Decimal(2026), "6.30pm", Decimal(4123), Decimal("0.72"),
                    Decimal(1), Decimal("5.4")]
    assert number_tokens("about $3.5 million or $2m") == [Decimal(3500000), Decimal(2000000)]
    assert number_tokens("at 6:30 PM") == ["6.30pm"]
    assert number_tokens("numbers 3,11,19") == [Decimal(3), Decimal(11), Decimal(19)]
    assert number_tokens("the 1st draw") == [Decimal(1)]
    # 4D is no longer a game the bot knows: its 4 is just a number like any other.
    assert number_tokens("4D") == [Decimal(4)]


def test_allowed_numbers_includes_figures_and_cents():
    allowed = allowed_numbers(FIGURES)
    for value in ("3500000", "0.72", "72", "3", "25", "100", "260", "23", "6", "30", "10", "20", "5", "2026", "1"):
        assert Decimal(value) in allowed, value
    assert "6.30pm" in allowed
    assert Decimal(4) not in allowed
    raw = allowed_numbers({"ev": 0.72345})
    assert Decimal("0.72") in raw and Decimal("0.7") in raw and Decimal(1) in raw


def test_whole_number_counts_are_not_scaled_by_100():
    """5 rollovers or 6 tickets must not let an invented $500 or $600 through."""
    figures = {"jackpot_outlook": {"jackpot_rollovers_so_far": 5}, "my_tickets": {"tickets": 6},
               "next_toto": {"draw_time": "Mon 5 Oct 2026, 6.30pm"}, "buy_signal": {"return_per_dollar": "$0.66"}}
    allowed = allowed_numbers(figures)
    assert Decimal(500) not in allowed and Decimal(600) not in allowed and Decimal(100) not in allowed
    assert Decimal(66) in allowed  # "$0.66" may still be written as 66 cents
    assert validate_commentary("After 5 rollovers, a $500 prize is waiting.", figures) is None
    assert validate_commentary("Your 6 tickets won $600.", figures) is None
    assert validate_commentary("After 5 rollovers, 66 cents back per $1.", figures)


# validate_commentary


def test_validate_accepts_figures_only_text():
    text = ("The Mon 5 Oct 2026, 6.30pm draw is a cascade draw with an estimated jackpot of $3,500,000 after 3 "
            "rollovers, and somebody wins Group 1 with a 25% chance. The buy signal is HIGH at $0.72 back per $1. "
            "Your 6 tickets cost $30 and won $10, so you are minus $20.")
    assert validate_commentary(text, FIGURES) == text


def test_validate_accepts_scaled_and_cent_forms():
    assert validate_commentary("Jackpot about $3.5 million, 72 cents back per dollar.", FIGURES)


def test_validate_rejects_invented_numbers():
    assert validate_commentary("The jackpot is $3,600,000 tonight.", FIGURES) is None
    assert validate_commentary("You have a 1 in 54 chance, about 2% better.", FIGURES) is None
    assert validate_commentary("Return is $0.75 per $1.", FIGURES) is None


def test_validate_rejects_numbers_to_pick_and_4d():
    # Suggested numbers are numbers that are not in the figures, so they never get through.
    assert validate_commentary("Try 7 18 28 33 41 44 next time.", FIGURES) is None
    assert validate_commentary("Lucky numbers tonight: 8 and 9.", FIGURES) is None
    assert validate_commentary("Also try 4D tonight.", FIGURES) is None


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
    assert words is not None and len(words) <= MAX_CHARS and words.endswith("…")


@pytest.mark.parametrize("reply", ["", "   ", "— —", "3 3 3", None, 42])
def test_validate_rejects_empty(reply):
    assert validate_commentary(reply, FIGURES) is None


# The prompt


def test_prompt_rules():
    prompt = build_prompt(FIGURES)
    assert prompt.startswith("You write a short commentary for a Singapore Pools TOTO update posted to Telegram.")
    for rule in ("At most 60 words in one short paragraph",
                 "Use only the figures in the JSON below, written exactly as they appear there.",
                 "Do not add, round, convert or calculate any number.",
                 "Never use dashes or hyphens of any kind.",
                 "Every draw is independent, so past results do not change the odds.",
                 "Never suggest numbers to pick",
                 "never suggest any way to beat the odds",
                 "never urge anyone to buy more",
                 "Reply with the commentary text only."):
        assert rule in prompt, rule
    assert json.loads(prompt.split("Figures (JSON):\n", 1)[1]) == FIGURES
    assert no_dash(prompt)
    for gone in ("4D", "budget", "strategy", "backtest", "System 7"):
        assert gone not in prompt, gone


def test_is_enabled(monkeypatch):
    monkeypatch.delenv("COMMENTARY", raising=False)
    assert not is_enabled() and not is_enabled("off") and not is_enabled("  ") and not is_enabled("False")
    assert is_enabled("claude") and is_enabled(" Claude ")
    monkeypatch.setenv("COMMENTARY", "claude")
    assert is_enabled()


# build_commentary


def test_build_commentary_off_by_default(monkeypatch):
    monkeypatch.delenv("COMMENTARY", raising=False)
    runner = FakeRunner("text")
    assert build_commentary(FIGURES, runner=runner) is None
    assert build_commentary(FIGURES, mode="off", runner=runner) is None
    assert runner.calls == []


def test_build_commentary_unknown_mode():
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


def test_build_commentary_decodes_bytes():
    assert build_commentary(FIGURES, mode="claude", runner=FakeRunner(b"Signal HIGH.\n")) == "Signal HIGH."


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
    assert len(runner.calls) == 1


def test_build_commentary_logs_why_it_failed(caplog):
    with caplog.at_level(logging.WARNING, logger="huatbot.commentary"):
        build_commentary(FIGURES, mode="claude",
                         runner=FakeRunner("x", returncode=2, stderr="starting\nerror: not logged in"))
    assert "exited with code 2 error: not logged in" in caplog.text


def test_build_commentary_needs_figures():
    runner = FakeRunner("Signal HIGH.")
    assert build_commentary({}, mode="claude", runner=runner) is None
    assert runner.calls == []


def test_end_to_end_with_context(full_ctx):
    figs = figures_from_context(full_ctx)
    latest, out = figs["latest_toto"], figs["jackpot_outlook"]
    reply = (f"Draw {latest['draw']} came out {latest['numbers']}. If nobody wins first, the jackpot could reach "
             f"{out['biggest_projected_jackpot']} on {out['biggest_projected_on']}, and somebody wins Group 1 at the "
             f"next draw with a {out['chance_somebody_wins_next_draw']} chance. The signal is MEDIUM at $0.45 per $1.")
    runner = FakeRunner(reply)
    assert build_commentary(figs, mode="claude", runner=runner) == reply
    assert json.loads(runner.calls[0][0][2].split("Figures (JSON):\n", 1)[1]) == figs
    # A reply that suggests numbers to play is dropped.
    assert build_commentary(figs, mode="claude", runner=FakeRunner("Play 2 14 22 31 37 48 next.")) is None
