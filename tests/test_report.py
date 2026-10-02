"""Tests for huatbot.report: the full markdown report and the three Telegram messages."""
from __future__ import annotations

import re
from datetime import date, datetime
from html.parser import HTMLParser

import pandas as pd
import pytest

from huatbot import buysignal, report, suggest
from huatbot import constants as C
from huatbot.models import NextToto, PrizeRules
from huatbot.store import empty_ledger, fourd_numbers, toto_numbers
from huatbot.textfmt import contains_dash, has_prose_dashes
from huatbot.tickets import ledger_totals
from tests.ctxgen import cascade_next_toto, make_context, unknown_jackpot, variant

ALLOWED_TAGS = {"b", "i", "pre", "code"}


# Fixtures


@pytest.fixture(scope="module")
def ctx():
    return make_context()


@pytest.fixture(scope="module")
def many_tickets_ctx():
    return make_context(extra_tickets=400, with_backtest=False)


def _no_tickets(ctx):
    empty = empty_ledger()
    return variant(ctx, ledger=empty, ledger_totals=ledger_totals(empty), settled_this_run=[],
                   bad_ticket_lines=[])


def _scenarios(ctx, many_tickets_ctx):
    return {
        "normal": ctx,
        "cascade_high": cascade_next_toto(ctx),
        "jackpot_unknown": unknown_jackpot(ctx),
        "no_tickets": _no_tickets(ctx),
        "many_tickets": many_tickets_ctx,
        "toto_only": variant(ctx, games_drawn=("toto",)),
        "fourd_only": variant(ctx, games_drawn=("4d",)),
        "nothing_drawn": variant(ctx, games_drawn=()),
        "no_backtest": variant(ctx, toto_backtest=None, fourd_backtest=None),
        "no_next_draws": variant(ctx, next_toto=None, next_fourd=None, buy_signal=None),
        "commentary": variant(ctx, commentary="A quiet draw <b>tonight</b> & a modest jackpot; stay within budget."),
        "no_plans": variant(ctx, toto_plan=None, fourd_plan=None),
        "tiny_budget": _with_budgets(ctx, 0.5, 2),
        "system7_budget": _with_budgets(ctx, 12, 10),
    }


def _with_budgets(ctx, toto_budget, fourd_budget):
    """ctx with plans rebuilt for other budgets (the real suggest module, no rebuild of the rest)."""
    last = toto_numbers(report.latest_row(ctx.toto))
    return variant(
        ctx,
        settings=variant(ctx.settings, toto_budget=toto_budget, fourd_budget=fourd_budget),
        toto_plan=suggest.toto_plan(ctx.toto_picks, toto_budget, ctx.crowd_scores, True, last),
        fourd_plan=suggest.fourd_plan(ctx.fourd_picks, fourd_budget, ctx.fourd_bet_values),
    )


SCENARIOS = ["normal", "cascade_high", "jackpot_unknown", "no_tickets", "many_tickets", "toto_only",
             "fourd_only", "nothing_drawn", "no_backtest", "no_next_draws", "commentary", "no_plans",
             "tiny_budget", "system7_budget"]


# Telegram HTML checks


class _TagChecker(HTMLParser):
    """Collects tag problems: unknown tags, attributes, bad nesting, unclosed tags."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED_TAGS:
            self.problems.append(f"tag <{tag}> not allowed")
        if attrs:
            self.problems.append(f"attributes on <{tag}>")
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.problems.append(f"self closing <{tag}/>")

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"unexpected </{tag}>")
        else:
            self.stack.pop()


def html_problems(text: str) -> list[str]:
    checker = _TagChecker()
    checker.feed(text)
    checker.close()
    problems = list(checker.problems)
    if checker.stack:
        problems.append(f"unclosed tags {checker.stack}")
    plain = re.sub(r"</?(?:b|i|pre|code)>", "", text)
    if "<" in plain or ">" in plain:
        problems.append("unescaped < or > in text")
    if re.search(r"&(?!amp;|lt;|gt;|quot;)", plain):
        problems.append("unescaped & in text")
    return problems


def assert_valid_message(msg: str) -> None:
    assert isinstance(msg, str) and msg.strip()
    assert len(msg) <= C.TELEGRAM_MAX_CHARS, len(msg)
    for ch in "-–—":
        assert ch not in msg, f"dash {ch!r} in message"
    assert not contains_dash(msg)
    assert html_problems(msg) == []


# Telegram messages


@pytest.mark.parametrize("name", SCENARIOS)
def test_telegram_messages_are_three_valid_html_messages(name, ctx, many_tickets_ctx):
    messages = report.telegram_messages(_scenarios(ctx, many_tickets_ctx)[name])
    assert len(messages) == 3
    for msg in messages:
        assert_valid_message(msg)


def test_message1_headline_is_obvious_when_a_ticket_won(ctx):
    won = sum(r["winnings"] for r in ctx.settled_this_run)
    assert won > 0
    first_line = report.telegram_messages(ctx)[0].split("\n", 1)[0]
    assert first_line.startswith("<b>WINNER!")
    assert report.dollars(won) in first_line


def test_message1_headline_plain_when_nothing_won(ctx):
    settled = [dict(r, winnings=0.0, result="No prize") for r in ctx.settled_this_run]
    msg = report.telegram_messages(variant(ctx, settled_this_run=settled))[0]
    assert "WINNER" not in msg
    assert msg.startswith("<b>Huat Bot results</b>")
    assert "No prize" in msg


def test_message1_has_full_toto_result_and_all_23_fourd_numbers(ctx):
    msg = report.telegram_messages(ctx)[0]
    toto = report.latest_row(ctx.toto)
    assert f"TOTO draw {int(toto['draw_number'])}" in msg
    assert "Thu 1 Oct 2026" in msg
    assert " ".join(str(int(toto[f"n{i}"])) for i in range(1, 7)) in msg
    assert f"Additional number: <b>{int(toto['additional'])}</b>" in msg
    assert "Group 1 prize" in msg
    for g in range(1, 8):
        assert f"Group {g}" in msg
    fourd = report.latest_row(ctx.fourd)
    numbers = [n for tier in fourd_numbers(fourd).values() for n in tier]
    assert len(numbers) == 23
    for n in numbers:
        assert n in msg


def test_message1_ticket_check_and_totals(ctx):
    msg = report.telegram_messages(ctx)[0]
    assert "My tickets" in msg
    assert "1st Prize x1" in msg and "Group 7 x1" in msg
    totals = ctx.ledger_totals
    assert f"spent {report.dollars(totals['spent'])}" in msg
    assert f"net <b>{report.dollars(totals['net'])}</b>" in msg
    assert "could not be read" in msg  # the two unreadable lines


def test_message1_follows_games_drawn(ctx):
    toto_only = report.telegram_messages(variant(ctx, games_drawn=("toto",)))[0]
    assert "TOTO draw 4123" in toto_only and "4D draw 5432" not in toto_only
    fourd_only = report.telegram_messages(variant(ctx, games_drawn=("4d",)))[0]
    assert "4D draw 5432" in fourd_only and "TOTO draw 4123" not in fourd_only
    nothing = report.telegram_messages(variant(ctx, games_drawn=()))[0]
    assert "No new draw results" in nothing


def test_message1_without_tickets(ctx):
    msg = report.telegram_messages(_no_tickets(ctx))[0]
    assert "No tickets were checked" in msg
    assert "Tickets.md" in msg


def test_message1_many_tickets_are_summarised(many_tickets_ctx):
    settled = many_tickets_ctx.settled_this_run
    assert len(settled) > 400
    msg = report.telegram_messages(many_tickets_ctx)[0]
    assert_valid_message(msg)
    assert "WINNER!" in msg  # the 4D first prize ticket is still headlined
    assert "won nothing" in msg or "tickets checked" in msg


def test_message2_next_draws_and_buy_signal(ctx):
    msg = report.telegram_messages(ctx)[1]
    assert "Next TOTO draw" in msg and "Mon 5 Oct 2026, 6.30pm" in msg
    assert "$2,100,000" in msg
    assert f"Buy signal: <b>{ctx.buy_signal.label}</b>" in msg
    assert f"Draws in a row with no Group 1 winner: {ctx.buy_signal.no_winner_streak}" in msg
    assert report.per_dollar(ctx.buy_signal.ev_per_dollar) in msg
    assert "Groups 5 to 7" in msg  # breakdown table
    assert "Next 4D draw" in msg and "Sat 3 Oct 2026, 6.30pm" in msg
    assert "$2,000" in msg and "$3,000" in msg  # Big and Small 1st Prize per $1
    assert "Big <b>$0.66</b>" in msg


def test_message2_cascade_draw_with_high_label(ctx):
    cascade = cascade_next_toto(ctx)
    assert cascade.buy_signal.label == "HIGH"
    msg = report.telegram_messages(cascade)[1]
    assert "Buy signal: <b>HIGH</b>" in msg
    assert "<b>Cascade draw</b>" in msg
    assert "$4,500,000" in msg


def test_message2_jackpot_unknown(ctx):
    msg = report.telegram_messages(unknown_jackpot(ctx))[1]
    assert "Estimated jackpot: not available yet" in msg
    assert "Return per $1: not available" in msg
    assert "Buy signal: <b>MEDIUM</b>" in msg


def test_message2_without_next_draw_info_uses_schedule(ctx):
    msg = report.telegram_messages(variant(ctx, next_toto=None, next_fourd=None, buy_signal=None))[1]
    assert "Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)" in msg
    assert "Sat 3 Oct 2026" in msg


def test_message3_suggestions_costs_and_extras(ctx):
    msg = report.telegram_messages(ctx)[2]
    for line in ctx.toto_plan.lines + ctx.fourd_plan.lines:
        assert line.numbers in msg
    assert f"TOTO total: <b>{report.dollars(ctx.toto_plan.total)}</b> of your $10 budget" in msg
    assert f"4D total: <b>{report.dollars(ctx.fourd_plan.total)}</b> of your $5 budget" in msg
    assert "System 7 option" in msg
    assert "<b>Backtest</b>" in msg and "random" in msg
    assert "1 in 13,983,816" in msg and "about 1 in 54" in msg and "23 times in 10,000" in msg
    assert ctx.toto_picks[0].reason.split(":")[0][:20] in msg  # reasons shown at full detail


def test_message3_budget_edges(ctx, many_tickets_ctx):
    scenarios = _scenarios(ctx, many_tickets_ctx)
    tiny = report.telegram_messages(scenarios["tiny_budget"])[2]
    assert "Budget below $1, nothing to buy." in tiny
    assert "4D total: <b>$2</b> of your $2 budget" in tiny
    big = scenarios["system7_budget"]
    assert any(ln.label == "System 7" for ln in big.toto_plan.lines)
    msg = report.telegram_messages(big)[2]
    sys7 = next(ln for ln in big.toto_plan.lines if ln.label == "System 7")
    assert sys7.numbers in msg
    assert f"TOTO total: <b>{report.dollars(big.toto_plan.total)}</b> of your $12 budget" in msg
    assert big.toto_plan.total <= 12


def test_message3_commentary_is_escaped(ctx):
    msg = report.telegram_messages(variant(ctx, commentary="Quiet draw <b>tonight</b> & a modest jackpot — enjoy"))[2]
    assert "&lt;b&gt;tonight&lt;/b&gt; &amp; a modest jackpot" in msg
    assert_valid_message(msg)


def test_message3_without_backtest(ctx):
    msg = report.telegram_messages(variant(ctx, toto_backtest=None, fourd_backtest=None))[2]
    assert "<b>Backtest</b>" not in msg
    assert "Suggested numbers" in msg


def test_fit_drops_whole_blocks_when_too_long():
    text = report._fit(lambda level: ["<b>Header</b>", "x" * 3000, "<pre>" + "y" * 3000 + "</pre>"])
    assert len(text) <= C.TELEGRAM_MAX_CHARS
    assert text.startswith("<b>Header</b>")
    assert "More detail is in the report note" in text
    assert html_problems(text) == []


def test_fit_removes_stray_dashes():
    text = report._fit(lambda level: ["<b>A</b> 1 - 2"])
    assert not contains_dash(text)


# Full report


def test_full_report_headings_in_order(ctx):
    text = report.full_report(ctx)
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(report.SECTION_HEADINGS)
    assert text.startswith("# Huat Bot report, Thu 1 Oct 2026, 7.30pm")


@pytest.mark.parametrize("name", SCENARIOS)
def test_full_report_has_no_prose_dashes(name, ctx, many_tickets_ctx):
    text = report.full_report(_scenarios(ctx, many_tickets_ctx)[name])
    assert not has_prose_dashes(text)
    assert [ln for ln in text.splitlines() if ln.startswith("## ")] == list(report.SECTION_HEADINGS)


def test_full_report_uses_markdown_tables(ctx):
    text = report.full_report(ctx)
    assert "| Group   | Share amount | Winning shares |" in text
    assert text.count("| ---") >= 10


def test_full_report_chi_square_verdicts(ctx):
    text = report.full_report(ctx)
    assert ctx.toto_chi["verdict"] in text
    assert ctx.fourd_chi["verdict"] in text
    assert "Fairness tests (chi square)" in text


def test_full_report_crowd_diagnostics(ctx):
    text = report.full_report(ctx)
    assert "### Crowd score" in text
    assert "Average crowd ratio" in text
    assert "close to 1 as expected" in text
    assert "Most crowded:" in text and "Least crowded:" in text


def test_full_report_warns_when_crowd_ratio_is_off(ctx):
    diag = dict(ctx.crowd_diag, mean_ratio=0.86, ok=False, warning=None)
    text = report.full_report(variant(ctx, crowd_diag=diag))
    assert "**Warning:** Average crowd ratio is 0.86, not close to 1" in text
    diag2 = dict(ctx.crowd_diag, mean_ratio=1.3, ok=False,
                 warning="Average crowd ratio is 1.30, not close to 1, so the 54% prize pool assumption may be off.")
    text2 = report.full_report(variant(ctx, crowd_diag=diag2))
    assert "Average crowd ratio is 1.30, not close to 1" in text2
    assert "> * Average crowd ratio is 1.30" in text2  # also listed with the warnings
    assert not has_prose_dashes(text2)


def test_full_report_bet_type_values_name_the_best(ctx):
    text = report.full_report(ctx)
    assert "### 4D bet type value" in text
    assert f"Best bet type: **{ctx.fourd_bet_values['best']}**" in text
    assert "| Big " in text and "| Small " in text and "iBet Big, 24 permutations" in text


def test_full_report_backtest_scoreboard(ctx):
    text = report.full_report(ctx)
    section = text.split(report.SECTION_HEADINGS[3])[1].split(report.SECTION_HEADINGS[4])[0]
    for result in (ctx.toto_backtest, ctx.fourd_backtest):
        for score in result.scores:
            assert f"| {score.name} " in section
            if score.name != "Random":
                assert score.verdict in section
    assert "Return per $1" in section and "Cost" in section and "Winnings" in section
    assert "Baseline, the average random player" in section


def test_full_report_missing_backtest(ctx):
    text = report.full_report(variant(ctx, toto_backtest=None, fourd_backtest=None))
    assert "The TOTO backtest is not available in this run." in text
    assert "The 4D backtest is not available in this run." in text


def test_full_report_prize_rules_status(ctx):
    text = report.full_report(ctx)
    assert "not confirmed, built in values used" in text
    assert "prize rules could not be confirmed" in text
    confirmed = PrizeRules(toto_confirmed=True, fourd_confirmed=True,
                           source_note="TOTO and 4D prize rules confirmed on the official pages",
                           checked_at="2026-10-01T19:00:00+08:00")
    text2 = report.full_report(variant(ctx, rules=confirmed))
    assert text2.count("confirmed on the official prize page") == 2
    assert "Last checked: Thu 1 Oct 2026, 7.00pm." in text2
    assert "could not be confirmed" not in text2


def test_full_report_lists_warnings(ctx):
    settings_warned = variant(ctx.settings, warnings=["toto_budget was minus 5, using $0 instead"])
    text = report.full_report(variant(ctx, warnings=["Could not reach the site; used the stored data"],
                                      settings=settings_warned))
    assert "> [!warning] Warnings" in text
    assert "Could not reach the site; used the stored data" in text
    assert "toto_budget was minus 5, using $0 instead" in text
    assert not has_prose_dashes(text)


def test_full_report_states_the_odds_note_once(ctx):
    text = report.full_report(ctx)
    note = report.odds_note(ctx.rules)
    assert text.count(note) == 1
    assert text.split(report.SECTION_HEADINGS[5])[1].strip().startswith(note)


def test_full_report_ticket_check(ctx):
    text = report.full_report(ctx)
    section = text.split(report.SECTION_HEADINGS[1])[1].split(report.SECTION_HEADINGS[2])[0]
    assert "**You won $4,010**" in section
    assert "1st Prize x1" in section and "Group 7 x1" in section
    assert "Totals so far" in section
    assert "2 lines in [[Tickets]] could not be read" in section
    no_tickets = report.full_report(_no_tickets(ctx))
    assert "No tickets were checked in this run." in no_tickets
    assert "No tickets in the ledger yet" in no_tickets


def test_full_report_suggestions_within_budget(ctx):
    text = report.full_report(ctx)
    section = text.split(report.SECTION_HEADINGS[2])[1].split(report.SECTION_HEADINGS[3])[0]
    total = ctx.toto_plan.total + ctx.fourd_plan.total
    budget = ctx.settings.toto_budget + ctx.settings.fourd_budget
    assert total <= budget
    assert f"**Total for both games: {report.dollars(total)} of the {report.dollars(budget)} budget.**" in section
    for line in ctx.toto_plan.lines + ctx.fourd_plan.lines:
        assert line.numbers in section
    assert "Alternative for the same budget" in section  # the System 7 option


def test_full_report_includes_commentary_when_present(ctx):
    text = report.full_report(variant(ctx, commentary="A calm week — nothing special."))
    assert "*Commentary:* A calm week, nothing special." in text
    assert not has_prose_dashes(text)


def test_full_report_key_stats(ctx):
    text = report.full_report(ctx)
    assert "### TOTO numbers" in text and "Draws since seen" in text
    assert "### Most common TOTO pairs" in text
    assert "### Usual shape of a winning TOTO set" in text
    assert "### 4D digit frequency by position" in text
    assert "### 4D numbers that won more than once" in text


# Odds note and helpers


def test_odds_note_content():
    note = report.odds_note(PrizeRules())
    assert note.count("independent") == 1
    assert "1 in 13,983,816 per board" in note
    assert "about 1 in 54" in note
    assert "23 times in 10,000" in note
    assert not contains_dash(note)


def test_next_draw_falls_back_to_schedule(ctx):
    nd = report.next_draw(variant(ctx, next_toto=None), "toto")
    assert nd.number == 4124 and str(nd.day) == "2026-10-05" and nd.from_schedule
    stale = report.next_draw(ctx, "4d")
    assert stale.number == 5433 and str(stale.day) == "2026-10-03" and not stale.from_schedule


def test_backtest_summary_groups_verdicts(ctx):
    line = report.backtest_summary(ctx.toto_backtest, "toto")
    assert line.startswith("TOTO, last 20 draws:")
    assert report.backtest_summary(None, "toto") is None


# Stale next draw info and stale results


def _sg(y, m, d, hh=19, mm=30):
    return datetime(y, m, d, hh, mm, tzinfo=report.SG)


def test_next_draw_later_today_is_still_the_next_draw(ctx):
    # Mon 5 Oct at 7.30pm, Monday's result not stored yet: Monday is still the next draw.
    monday = variant(ctx, now=_sg(2026, 10, 5))
    nd = report.next_draw(monday, "toto")
    assert str(nd.day) == "2026-10-05" and nd.number == 4124 and not nd.stale and not nd.from_schedule


def test_next_draw_page_showing_the_draw_just_held_is_not_used(ctx):
    # The 7.30pm run on Thu 1 Oct: the next draw page still shows tonight's draw (already stored).
    held = NextToto(draw_datetime=_sg(2026, 10, 1, 18, 30), jackpot_estimate=5_000_000.0, draw_type="normal",
                    draw_type_hint=None)
    signal = buysignal.buy_signal(held, ctx.toto, ctx.settings, ctx.rules)
    assert signal.jackpot == 5_000_000.0  # what the runner computed from the stale page
    stale = variant(ctx, next_toto=held, buy_signal=signal)
    assert not report.next_info_is_current(stale, "toto")
    msg = report.telegram_messages(stale)[1]
    assert "$5,000,000" not in msg
    assert "Estimated jackpot: not available yet" in msg and "Return per $1: not available" in msg
    assert "Buy signal: <b>MEDIUM</b>" in msg
    assert "Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)" in msg
    assert "$5,000,000" not in report.full_report(stale)


def test_stored_results_a_week_old_are_marked_not_up_to_date(ctx):
    # The site could not be read for a week: stored results end Thu 1 Oct, it is now Fri 9 Oct,
    # and the stored next draws (Mon 5 Oct, Sat 3 Oct) are past.
    week = variant(ctx, now=_sg(2026, 10, 9))
    today = date(2026, 10, 9)
    for game in ("toto", "4d"):
        nd = report.next_draw(week, game)
        assert nd.stale and nd.number is None and nd.day >= today
        assert "results are not up to date" in nd.when_text
    assert str(report.next_draw(week, "toto").day) == "2026-10-12"  # Monday
    assert str(report.next_draw(week, "4d").day) == "2026-10-10"  # Saturday
    msgs = report.telegram_messages(week)
    assert "Results not up to date" in msgs[0] and "TOTO draw 4123 and 4D draw 5432" in msgs[0]
    assert "Mon 5 Oct" not in msgs[1] and "Sat 3 Oct" not in msgs[1]
    assert report.money(ctx.next_toto.jackpot_estimate) not in msgs[1]
    assert "Results not up to date" in report.full_report(week)
    for msg in msgs:
        assert_valid_message(msg)
    assert "Results not up to date" not in report.telegram_messages(ctx)[0]


def test_message3_does_not_suggest_buying_beyond_the_plan(ctx):
    msg = report.telegram_messages(ctx)[2]
    assert "if the budget allows" not in msg
    bought = {ln.label for ln in ctx.toto_plan.lines}
    if any(p.name not in bought for p in ctx.toto_picks):
        assert "Not in the plan (details in the report)" in msg


def test_message1_mentions_tickets_dated_on_a_day_without_a_draw(ctx):
    ledger = ctx.ledger.copy()
    extra = ledger.iloc[[0]].copy()
    extra["status"] = "no_draw"
    extra["cost"] = 3.0
    extra["ticket_id"] = "no-draw-ticket"
    ledger = pd.concat([ledger, extra], ignore_index=True)
    msg = report.telegram_messages(variant(ctx, ledger=ledger, ledger_totals=ledger_totals(ledger)))[0]
    assert "1 ticket ($3) has no draw on its date and still counts as spent. Check the date in Tickets.md." in msg
    assert "no draw on" not in report.telegram_messages(ctx)[0]
