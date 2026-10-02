"""Tests for huatbot.report: the full markdown report and the two Telegram messages (TOTO only)."""
from __future__ import annotations

import re
from datetime import date, datetime
from html.parser import HTMLParser

import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import notes, report, tickets
from huatbot.models import (
    LEDGER_COLUMNS,
    JackpotHistory,
    JackpotOutlook,
    NextToto,
    OutlookStep,
    PrizeRules,
)
from huatbot.outlook import chance_won_by_cascade
from huatbot.store import empty_ledger, empty_toto, toto_numbers
from huatbot.textfmt import contains_dash, fmt_date, fmt_num, has_prose_dashes, money, pct, per_dollar
from huatbot.tickets import ledger_totals
from huatbot.vault import Vault
from tests.ctxgen import (
    TOTO_LAST_DRAW,
    cascade_next_toto,
    make_context,
    rebuilt,
    unknown_jackpot,
    variant,
)

ALLOWED_TAGS = {"b", "i", "pre", "code", "blockquote"}
SG = report.SG


def _sg(y, m, d, hh=19, mm=30):
    return datetime(y, m, d, hh, mm, tzinfo=SG)


HONGBAO_FRI = NextToto(draw_datetime=_sg(2026, 10, 9, 21, 30), jackpot_estimate=4_000_000.0, draw_type="hongbao",
                       draw_type_hint="hongbao")


# Fixtures


@pytest.fixture(scope="module")
def ctx():
    return make_context()


@pytest.fixture(scope="module")
def many_tickets_ctx():
    return make_context(extra_tickets=400)


@pytest.fixture(scope="module")
def specials_ctx():
    return make_context(announced=[(date(2026, 10, 9), "hongbao"), (date(2026, 10, 31), "special"),
                                   (date(2026, 10, 5), "normal"), (date(2026, 9, 28), "special")])


def _no_tickets(ctx):
    empty = empty_ledger()
    return variant(ctx, ledger=empty, ledger_totals=ledger_totals(empty), settled_this_run=[],
                   bad_ticket_lines=[])


def _nothing_won(ctx):
    return variant(ctx, settled_this_run=[dict(r, winnings=0.0, result="No prize") for r in ctx.settled_this_run])


def _scenarios(ctx, many_tickets_ctx, specials_ctx):
    return {
        "normal": ctx,
        "cascade": cascade_next_toto(ctx),
        "jackpot_unknown": unknown_jackpot(ctx),
        "no_next_draw_page": rebuilt(ctx, None),
        "hongbao": rebuilt(ctx, HONGBAO_FRI),
        "specials": specials_ctx,
        "no_tickets": _no_tickets(ctx),
        "nothing_won": _nothing_won(ctx),
        "many_tickets": many_tickets_ctx,
        "no_new_draws": variant(ctx, new_draws=[]),
        "no_outlook": variant(ctx, outlook=None, history=None, buy_signal=None),
        "empty_history": variant(ctx, toto=empty_toto(), outlook=None, history=None, buy_signal=None,
                                 next_toto=None, new_draws=[]),
        "commentary": variant(ctx, commentary="A quiet draw <b>tonight</b> & a modest jackpot; enjoy it."),
        "held": variant(ctx, now=_sg(2026, 10, 5, 21, 0)),
        "stale_week": variant(ctx, now=_sg(2026, 10, 9)),
    }


SCENARIOS = ["normal", "cascade", "jackpot_unknown", "no_next_draw_page", "hongbao", "specials", "no_tickets",
             "nothing_won", "many_tickets", "no_new_draws", "no_outlook", "empty_history", "commentary", "held",
             "stale_week"]


@pytest.fixture(scope="module")
def scenarios(ctx, many_tickets_ctx, specials_ctx):
    return _scenarios(ctx, many_tickets_ctx, specials_ctx)


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
        if attrs and not (tag == "blockquote" and attrs == [("expandable", None)]):
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
    plain = re.sub(r"</?(?:b|i|pre|code|blockquote)(?: expandable)?>", "", text)
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


def md_cell(text: str, label: str) -> str | None:
    """Value cell of the two column markdown table row whose first cell is ``label``."""
    for line in text.splitlines():
        if line.startswith("| "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] == label:
                return cells[1]
    return None


def section(text: str, k: int) -> str:
    """Report section k (0 based) up to the next section heading."""
    start = text.index(report.SECTION_HEADINGS[k])
    end = text.index(report.SECTION_HEADINGS[k + 1]) if k + 1 < len(report.SECTION_HEADINGS) else len(text)
    return text[start:end]


# Telegram messages: shape


@pytest.mark.parametrize("name", SCENARIOS)
def test_telegram_messages_are_two_valid_html_messages(name, scenarios):
    messages = report.telegram_messages(scenarios[name])
    assert len(messages) == 2
    for msg in messages:
        assert_valid_message(msg)
    # Message 1 is the result, message 2 the next draw.
    assert "<b>NEXT TOTO DRAW</b>" not in messages[0]
    assert messages[1].startswith("🔮 <b>NEXT TOTO DRAW</b>")


@pytest.mark.parametrize("name", SCENARIOS)
def test_message2_never_carries_a_set_of_six_numbers(name, scenarios):
    # The third message used to hold suggested sets; message 2 must hold none at all.
    msg = report.telegram_messages(scenarios[name])[1]
    assert not re.search(r"(?<![\d,.$])\d{1,2}(?: \d{1,2}){5}(?![\d,.])", msg)


# Message 1: headline, latest result, my tickets


def test_message1_headline_is_obvious_when_a_ticket_won(ctx):
    won = sum(r["winnings"] for r in ctx.settled_this_run)
    assert won == 10.0
    head = report.telegram_messages(ctx)[0].split("\n\n", 1)[0]
    assert head == "🎉 <b>WINNER</b> · your tickets won $10\n1 winning ticket, details below."


def test_message1_headline_plain_when_nothing_won(ctx):
    msg = report.telegram_messages(_nothing_won(ctx))[0]
    assert "WINNER" not in msg
    assert msg.startswith("🎱 <b>TOTO RESULT</b> · Draw 4123, Thu 1 Oct 2026\n\n")
    assert "No prize" in msg


def test_message1_has_the_full_latest_result(ctx):
    msg = report.telegram_messages(ctx)[0]
    row = report.latest_row(ctx.toto)
    assert int(row["draw_number"]) == TOTO_LAST_DRAW
    assert "<b>TOTO RESULT</b> · Draw 4123, Thu 1 Oct 2026\n" in msg
    assert "(Normal draw)" not in msg  # only a special kind of draw is named
    assert f"<b>Winning numbers</b> · <code>{' '.join(str(n) for n in toto_numbers(row))}</code>" in msg
    assert f"Additional number · <code>{int(row['additional'])}</code>" in msg
    assert "Group 1 · <b>$1,000,000</b>, no winner" in msg
    assert "<pre>Group       Share  Winners\n" in msg
    for g in range(1, 8):
        assert f"Group {g}" in msg
        assert fmt_num(int(row[f"g{g}_winners"])) in msg
    assert money(row["g2_share"]) in msg
    assert "No new draw in this run" not in msg


def test_message1_names_a_cascade_result_and_its_winners(ctx):
    toto = ctx.toto.copy()
    i = toto.index[-1]
    toto.loc[i, "draw_type"] = "cascade"
    toto.loc[i, "g1_winners"] = 2
    toto.loc[i, "g1_share"] = 2_000_000.0
    toto.loc[i, "jackpot"] = 4_000_000.0
    msg = report.telegram_messages(variant(ctx, toto=toto))[0]
    assert "<b>TOTO RESULT</b> · Draw 4123, Thu 1 Oct 2026, Cascade draw" in msg
    assert "Group 1 · <b>$4,000,000</b>, 2 winning shares of $2,000,000 each" in msg


def test_message1_without_new_draws_says_so(ctx):
    msg = report.telegram_messages(variant(ctx, new_draws=[]))[0]
    assert "<i>No new draw in this run, this is the newest stored result.</i>" in msg


def test_message1_ticket_check_and_totals(ctx):
    msg = report.telegram_messages(ctx)[0]
    winner = next(r for r in ctx.settled_this_run if r["winnings"] > 0)
    assert "<b>MY TICKETS</b>" in msg
    assert f"🟢 Thu 1 Oct 2026, {winner['numbers']}, Ordinary $1: Group 7 x1, won <b>$10</b>" in msg
    assert "⚪ Mon 28 Sep 2026, 1 2 3 4 5 6, Ordinary $1: No prize" in msg
    assert "All tickets so far: spent $10, won $10, net <b>$0</b>." in msg
    assert "2 tickets ($8) wait for the draw." in msg
    assert "2 lines in Tickets.md could not be read, see Ledger.md." in msg
    # The winner is listed before the losing ticket.
    assert msg.index("Group 7 x1") < msg.index("No prize")


def test_message1_without_tickets(ctx):
    msg = report.telegram_messages(_no_tickets(ctx))[0]
    assert "No tickets were checked in this run." in msg
    assert "Add the tickets you buy to Tickets.md in the vault and the bot will check them." in msg
    assert "All tickets so far" not in msg


def test_message1_many_tickets_drop_the_losing_lines_first(many_tickets_ctx):
    c = many_tickets_ctx
    settled = c.settled_this_run
    winners = [r for r in settled if r["winnings"] > 0]
    assert len(settled) == 402 and len(winners) == 13
    full = report._tg_join(report._tg_message1(c, 0))
    assert len(full) > C.TELEGRAM_MAX_CHARS
    msg = report.telegram_messages(c)[0]
    assert msg == report._tg_join(report._tg_message1(c, 1))  # the most detailed level that fits
    assert_valid_message(msg)
    assert msg.startswith("🎉 <b>WINNER</b> · your tickets won $130\n13 winning tickets, details below.")
    assert msg.count("won <b>$10</b>") == 13
    assert "389 other tickets checked won nothing." in msg
    assert "No prize" not in msg
    assert "All tickets so far: spent $410, won $130, net <b>minus $280</b>." in msg


def test_ticket_levels_cap_and_summarise_the_winners(many_tickets_ctx):
    c = many_tickets_ctx
    level2 = report._tg_tickets(c, 2)
    assert level2.count("won <b>$10</b>") == 10
    assert "🟢 and 3 more winning tickets" in level2
    level3 = report._tg_tickets(c, 3)
    assert "won <b>" not in level3
    assert "402 tickets checked, 13 won, $130 in total." in level3


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
    second = extra.copy()
    second["ticket_id"] = "no-draw-ticket-2"
    two = pd.concat([ledger, second], ignore_index=True)
    msg2 = report.telegram_messages(variant(ctx, ledger=two, ledger_totals=ledger_totals(two)))[0]
    assert "2 tickets ($6) have no draw on their date and still count as spent." in msg2


def _with_old_fourd_rows(ctx):
    """ctx whose ledger also holds rows an older TOTO and 4D version wrote: a checked 4D ticket
    that won $2,000 (real history) and an unchecked one (retired as not tracked)."""
    base = {c: "" for c in LEDGER_COLUMNS}
    won = dict(base, ticket_id="old4d1", game="4D", draw_date="2026-09-26", draw_number=5430, numbers="1234",
               bet_type="Big", cost=1.0, units=1.0, status="settled", result="1st Prize x1", winnings=2000.0,
               source="| 4D | 26 Sep 2026 | 1234 | Big | $1 |")
    pending = dict(base, ticket_id="old4d2", game="4D", draw_date="2026-10-03", numbers="5678", bet_type="Small",
                   cost=2.0, units=2.0, status="pending", result="", winnings=0.0,
                   source="| 4D | 3 Oct 2026 | 5678 | Small | $2 |")
    ledger = pd.concat([ctx.ledger, pd.DataFrame([won, pending])], ignore_index=True)
    ledger, _ = tickets.settle_ledger(ledger, ctx.toto, ctx.rules, ctx.now)
    return variant(ctx, ledger=ledger, ledger_totals=ledger_totals(ledger))


def test_old_fourd_ledger_rows_count_in_the_totals_but_are_never_named(ctx):
    c = _with_old_fourd_rows(ctx)
    status = dict(zip(c.ledger["ticket_id"], c.ledger["status"]))
    assert status["old4d1"] == "settled" and status["old4d2"] == "invalid"
    assert c.ledger_totals["spent"] == 11.0 and c.ledger_totals["won"] == 2010.0
    msgs = report.telegram_messages(c)
    assert "All tickets so far: spent $11, won $2,010, net <b>$1,999</b>." in msgs[0]
    assert "<b>WINNER</b> · your tickets won $10" in msgs[0]  # the old win is not new
    text = report.full_report(c)
    for out in msgs + [text]:
        assert "4D" not in out
    assert "|       5 |   $11 | $2,010 | $1,999 |" in section(text, 1)


# Message 2: next draw


def test_message2_next_draw_block(ctx):
    msg = report.telegram_messages(ctx)[1]
    bs, out = ctx.buy_signal, ctx.outlook
    lines = msg.split("\n")
    assert lines[0] == "🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct 2026, 6.30pm, draw 4124"
    assert lines[1] == ""
    # read from the page: no worked out mark; up on the last draw's $1,000,000 Group 1
    assert lines[2] == "💰 <b>Jackpot $2,100,000</b> 🟢 <i>UP ▲$1,100,000</i>"
    assert lines[3] == "🗓 Normal draw · rollovers 1 of 3, then it cascades"
    assert lines[4] == f"🎯 Somebody wins Group 1: <b>{pct(out.steps[0].chance_won, 0)}</b>"
    assert lines[5] == f"🟡 Buy signal <b>MEDIUM</b> · <b>{per_dollar(bs.ev_per_dollar)}</b> back per $1 on average"
    assert lines[6] == f"<i>{bs.reason}</i>"
    assert "<pre>Part           Per $1\n" in msg
    assert f"Groups 5 to 7   {per_dollar(bs.ev_breakdown['fixed'])}" in msg
    assert f"Total           {per_dollar(bs.ev_breakdown['total'])}</pre>" in msg
    assert "Cascaded jackpot" not in msg  # zero on a normal draw
    assert (f"Sales estimate: about {report.boards_text(bs.boards_estimate)} boards "
            f"({bs.boards_method}).") in msg
    assert report.JACKPOT_WORKED_OUT not in msg


def test_message2_order_of_blocks(ctx):
    msg = report.telegram_messages(ctx)[1]
    order = ["<b>NEXT TOTO DRAW</b>", "<b>NEXT BIG PRIZE</b>", "Over 260 stored draws", "⚖️ Every draw is independent."]
    positions = [msg.index(s) for s in order]
    assert positions == sorted(positions)
    assert msg.count("independent") == 1


def test_message2_cascade_draw_with_high_label(ctx):
    cascade = cascade_next_toto(ctx)
    assert cascade.buy_signal.label == "HIGH"
    msg = report.telegram_messages(cascade)[1]
    assert "Buy signal <b>HIGH</b>" in msg
    assert "<b>Cascade draw</b>" in msg
    assert "<b>Jackpot $4,500,000</b>" in msg
    assert "Cascaded jackpot" in msg
    assert ("The next draw is the cascade draw: about $4,500,000. If nobody wins it, the jackpot goes to the "
            "Group 2 winners.") in msg
    assert "Unwon" not in msg  # one draw left: no projection table


def test_message2_hongbao_draw(ctx):
    c = rebuilt(ctx, HONGBAO_FRI)
    msg = report.telegram_messages(c)[1]
    assert msg.startswith("🔮 <b>NEXT TOTO DRAW</b> · Fri 9 Oct 2026, 9.30pm, draw 4124\n")
    assert "<b>Hongbao draw</b>" in msg
    assert "rollovers 1\n" in msg  # a Hongbao draw has its own jackpot, it does not cascade on
    assert "then it cascades" not in msg
    assert "The next draw is a Hongbao draw with a jackpot of about $4,000,000." in msg
    assert "Announced special draws: Fri 9 Oct 2026 (Hongbao)." in msg


def test_message2_jackpot_worked_out_from_past_results(ctx):
    c = unknown_jackpot(ctx)
    sig = report.toto_signal(c)
    assert sig["jackpot_worked_out"] is True
    assert sig["jackpot"] == c.outlook.jackpot == c.buy_signal.jackpot
    amount = money(c.outlook.jackpot)
    assert amount != "$2,100,000"
    msg = report.telegram_messages(c)[1]
    assert f"<b>Jackpot {amount}</b> <i>(worked out from past results)</i>" in msg
    assert "Buy signal <b>MEDIUM</b>" in msg
    assert f"<b>{per_dollar(c.buy_signal.ev_per_dollar)}</b> back per $1" in msg
    assert report.jackpot_text(sig) == f"{amount} (worked out from past results)"
    text = report.full_report(c)
    assert md_cell(text, "Estimated jackpot") == f"{amount} (worked out from past results)"
    assert ("The next draw page gave no usable jackpot, so the next jackpot is worked out from the stored "
            "results.") in text
    # A page jackpot is shown unchanged.
    plain = report.toto_signal(ctx)
    assert plain["jackpot_worked_out"] is False
    assert report.jackpot_text(plain) == "$2,100,000"
    assert md_cell(report.full_report(ctx), "Estimated jackpot") == "$2,100,000"


def test_jackpot_text_when_nothing_is_known():
    sig = {"jackpot": None, "jackpot_worked_out": False}
    assert report.jackpot_text(sig) == "not available yet"
    assert report.jackpot_text(sig, missing="n/a") == "n/a"
    assert report.jackpot_text({"jackpot": 1_500_000.0}) == "$1,500,000"


def test_message2_without_next_draw_page_uses_schedule(ctx):
    c = rebuilt(ctx, None)
    msg = report.telegram_messages(c)[1]
    assert msg.startswith("🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)\n")
    assert "(draw 4124)" not in msg  # a special draw before it would take that number
    assert f"<b>Jackpot {money(c.outlook.jackpot)}</b> <i>(worked out from past results)</i>" in msg


def test_message2_without_outlook_or_signal(ctx):
    c = variant(ctx, outlook=None, history=None, buy_signal=None)
    sig = report.toto_signal(c)
    assert sig["jackpot"] == 2_100_000.0 and not sig["jackpot_worked_out"]
    assert sig["rollovers"] == 1 and sig["label"] is None and sig["ev"] is None
    msg = report.telegram_messages(c)[1]
    assert "<b>Jackpot $2,100,000</b>" in msg
    assert "Buy signal <b>not available</b>" in msg
    assert "Next big prize" not in msg and "Somebody wins Group 1" not in msg
    assert "stored draws" not in msg
    assert "back per $1" not in msg


def test_message2_with_nothing_stored(scenarios):
    msg = report.telegram_messages(scenarios["empty_history"])[1]
    assert msg.startswith("🔮 <b>NEXT TOTO DRAW</b> · not announced yet\n\n💰 Jackpot not available yet\n")
    assert "No TOTO result is stored yet." in report.telegram_messages(scenarios["empty_history"])[0]


def test_toto_signal_keys(ctx):
    sig = report.toto_signal(ctx)
    assert set(sig) == {"jackpot", "jackpot_worked_out", "draw_type", "rollovers", "label", "reason", "ev",
                        "breakdown", "boards", "boards_method"}
    assert sig["draw_type"] == "normal" and sig["rollovers"] == 1 and sig["label"] == "MEDIUM"
    assert sig["breakdown"]["total"] == pytest.approx(sig["ev"])


# Message 2: the next big prize


def test_message2_next_big_prize_block(ctx):
    out = ctx.outlook
    assert [s.draw_date for s in out.steps] == [date(2026, 10, 5), date(2026, 10, 8), date(2026, 10, 12)]
    assert [s.cascade for s in out.steps] == [False, False, True]
    # Each step's "still unwon" chance is the product of nobody winning at the draws before it.
    reached = 1.0
    for s in out.steps:
        assert s.chance_reached == pytest.approx(reached)
        reached *= 1 - s.chance_won
    assert chance_won_by_cascade(out) == pytest.approx(1 - reached)
    big = out.biggest
    assert big is out.steps[-1]
    msg = report.telegram_messages(ctx)[1]
    block = msg[msg.index("<b>NEXT BIG PRIZE</b>"):]
    assert block.startswith(
        f"<b>NEXT BIG PRIZE</b> · about {report.short_money(big.jackpot)} on Mon 12 Oct\nIf nobody wins Group 1 first, the jackpot snowballs to about {money(big.jackpot)} "
        f"at the cascade draw on Mon 12 Oct 2026 ({pct(big.chance_reached, 0)} chance it gets that far). The "
        f"chance somebody wins it before then is about {pct(1 - reached, 0)}.\n<pre>Draw        Jackpot  Unwon  Won\n")
    assert f"Mon 5 Oct    {report.short_money(2_100_000)}   100%  {pct(out.steps[0].chance_won, 0)}\n" in block
    assert "Thu 8 Oct" in block and "Mon 12 Oct" in block
    assert ("<i>Unwon: chance nobody has won it by then. Won: chance somebody wins at that draw.</i>") in block
    assert "Announced special draws" not in block


def test_message2_lists_announced_special_draws(specials_ctx):
    out = specials_ctx.outlook
    # Past and normal draws are left out.
    assert out.special_draws == [(date(2026, 10, 9), "hongbao"), (date(2026, 10, 31), "special")]
    msg = report.telegram_messages(specials_ctx)[1]
    assert "Announced special draws: Fri 9 Oct 2026 (Hongbao) and Sat 31 Oct 2026 (Special)." in msg
    assert "Announced special draws" in report.full_report(specials_ctx)


def test_message2_history_line(ctx):
    h = ctx.history
    top = h.biggest[0]
    line = (f"Over 260 stored draws Group 1 was won in {pct(h.won_share, 0)} of draws, a jackpot lasted "
            f"{fmt_num(h.average_run, 1)} draws on average and the biggest was {money(top['jackpot'])} on "
            f"{fmt_date(top['draw_date'])}.")
    assert report.history_line(h) == line
    assert f"<blockquote expandable>📜 {line}\n\n⚖️ " in report.telegram_messages(ctx)[1]


def test_message2_commentary_is_escaped(ctx):
    msg = report.telegram_messages(variant(ctx, commentary="Quiet draw <b>tonight</b> & a modest jackpot — enjoy"))[1]
    assert "&lt;b&gt;tonight&lt;/b&gt; &amp; a modest jackpot, enjoy" in msg
    assert_valid_message(msg)


def test_commentary_does_not_repeat_the_odds_statement(ctx):
    c = variant(ctx, commentary="Every draw is independent, so past results do not change the odds. "
                                "The jackpot is bigger than last week.")
    msg = report.telegram_messages(c)[1]
    assert msg.count("independent") == 1
    assert "💬 The jackpot is bigger than last week.</blockquote>" in msg
    text = report.full_report(c)
    assert text.count("independent") == 1
    assert "*Commentary:* The jackpot is bigger than last week." in text
    only = variant(ctx, commentary="Remember that every draw is independent.")
    assert "Commentary" not in report.full_report(only)
    assert report.telegram_messages(only)[1].count("independent") == 1
    assert report.commentary_text(None) == ""


# Detail levels and the Telegram limit


def _long_commentary(n: int) -> str:
    sentence = "The jackpot keeps growing while nobody matches all six numbers. "
    return (sentence * (n // len(sentence) + 1))[:n].rsplit(" ", 1)[0] + "."


@pytest.mark.parametrize("length,level", [(2000, 0), (2700, 2), (2850, 3), (3200, 4)])
def test_message2_drops_detail_level_by_level(ctx, length, level):
    c = variant(ctx, commentary=_long_commentary(length))
    msg = report.telegram_messages(c)[1]
    assert_valid_message(msg)
    assert msg == report._tg_join(report._tg_message2(c, level))
    for lower in range(level):
        assert len(report._tg_join(report._tg_message2(c, lower))) > C.TELEGRAM_MAX_CHARS
    assert ("Sales estimate" in msg) == (level < 1)
    assert ("stored draws Group 1 was won" in msg) == (level < 2)
    assert (ctx.buy_signal.reason in msg) == (level < 3)
    assert ("<pre>Part" in msg) == (level < 4)
    assert ("Unwon" in msg) == (level < 4)
    assert ("The jackpot keeps growing" in msg) == (level < 4)
    # Always kept: the headline figures, the big prize sentence and the odds.
    assert "<b>Jackpot $2,100,000</b>" in msg and "Buy signal <b>MEDIUM</b>" in msg
    assert "the jackpot snowballs to about" in msg
    assert "⚖️ Every draw is independent." in msg


def test_fit_drops_whole_blocks_when_too_long():
    text = report._fit(lambda level: ["<b>Header</b>", "x" * 3000, "<pre>" + "y" * 3000 + "</pre>"])
    assert len(text) <= C.TELEGRAM_MAX_CHARS
    assert text == "<b>Header</b>\n\n" + "x" * 3000 + "\n\n" + report._TG_MORE
    assert html_problems(text) == []


def test_fit_uses_the_most_detailed_level_that_fits():
    seen = []

    def build(level):
        seen.append(level)
        return ["<b>Head</b>", "<i>" + "z" * (4500 - 1000 * level) + "</i>"]
    text = report._fit(build)
    assert seen == [0, 1]
    assert text == "<b>Head</b>\n\n<i>" + "z" * 3500 + "</i>"


def test_fit_never_cuts_a_tag_even_when_one_block_is_too_long():
    text = report._fit(lambda level: ["<pre>" + "q" * 5000 + "</pre>"])
    assert text == report._TG_MORE
    assert html_problems(text) == []


def test_fit_removes_stray_dashes():
    text = report._fit(lambda level: ["<b>A</b> 1 - 2 – 3 — 4"])
    assert not contains_dash(text)
    assert text.startswith("<b>A</b> 1")


# Full report


def test_full_report_headings_in_order(ctx):
    text = report.full_report(ctx)
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(report.SECTION_HEADINGS)
    assert report.SECTION_HEADINGS == ("## 1. Next draw and the next big prize",
                                       "## 2. Latest result and my ticket check",
                                       "## 3. Jackpot history", "## 4. Odds note")
    assert text.startswith("# Huat Bot report, Thu 1 Oct 2026, 7.30pm\n\nNew in this run: 2 draws (4122 and 4123).")


def test_full_report_new_draws_line(ctx):
    assert "No new draws in this run." in report.full_report(variant(ctx, new_draws=[]))
    many = variant(ctx, new_draws=list(range(4116, 4124)))
    assert "New in this run: 8 draws (4119, 4120, 4121, 4122 and 4123 and earlier)." in report.full_report(many)


@pytest.mark.parametrize("name", SCENARIOS)
def test_full_report_has_no_prose_dashes(name, scenarios):
    text = report.full_report(scenarios[name])
    assert not has_prose_dashes(text)
    assert [ln for ln in text.splitlines() if ln.startswith("## ")] == list(report.SECTION_HEADINGS)


def test_full_report_section1_next_draw(ctx):
    text = section(report.full_report(ctx), 0)
    bs, out = ctx.buy_signal, ctx.outlook
    assert md_cell(text, "Draw") == "4124"
    assert md_cell(text, "Date and time") == "Mon 5 Oct 2026, 6.30pm"
    assert md_cell(text, "Estimated jackpot") == "$2,100,000"
    assert md_cell(text, "Draw type") == "Normal"
    assert md_cell(text, "Jackpot rollovers so far") == "1 of 3, then it cascades"
    assert md_cell(text, "Chance somebody wins Group 1 at this draw") == pct(out.steps[0].chance_won, 0)
    assert md_cell(text, "Buy signal") == "**MEDIUM**"
    assert md_cell(text, "Return per $1") == per_dollar(bs.ev_per_dollar)
    assert md_cell(text, "Sales estimate") == (f"about {report.boards_text(bs.boards_estimate)} boards "
                                               f"({bs.boards_method})")
    assert bs.reason in text
    assert md_cell(text, "Groups 5 to 7 (fixed prizes)") == per_dollar(bs.ev_breakdown["fixed"])
    assert md_cell(text, "Total") == per_dollar(bs.ev_breakdown["total"])
    assert "### The next big prize" in text
    assert "| Mon 12 Oct 2026 |" in text and "cascade draw |" in text
    assert report.big_prize_text(out) in text
    assert "it is not your chance" in text


def test_full_report_section2_result_and_tickets(ctx):
    text = section(report.full_report(ctx), 1)
    row = report.latest_row(ctx.toto)
    assert "### Latest draw 4123, Thu 1 Oct 2026" in text
    assert (f"Winning numbers: **{' '.join(str(n) for n in toto_numbers(row))}**, additional number "
            f"**{int(row['additional'])}**.") in text
    assert "Draw type: Normal. Group 1 prize: $1,000,000, no winner." in text
    assert "| Group   | Share amount | Winning shares |" in text
    assert "Draw note: [[Draws/TOTO/2026-10-01 TOTO 4123|2026-10-01 TOTO 4123]]." in text
    assert "### My ticket check" in text
    assert "2 tickets checked in this run. **You won $10** with 1 ticket in this run." in text
    assert "| Mon 28 Sep 2026 | 1 2 3 4 5 6    | Ordinary |   $1 | No prize   |  $0 |" in text
    assert "Totals so far:" in text and "|       4 |   $10 | $10 |  $0 |                2 |           $8 |" in text
    assert "2 lines in [[Tickets]] could not be read; [[Ledger]] lists them with the reason." in text
    assert "No new draw in this run" not in text


def test_full_report_ticket_check_edges(ctx):
    no_tickets = section(report.full_report(_no_tickets(ctx)), 1)
    assert "No tickets were checked in this run." in no_tickets
    assert "No tickets in the ledger yet. Add the tickets you buy to [[Tickets]]." in no_tickets
    lost = section(report.full_report(_nothing_won(ctx)), 1)
    assert "2 tickets checked in this run. None of the tickets checked in this run won a prize." in lost
    stored = section(report.full_report(variant(ctx, new_draws=[])), 1)
    assert "No new draw in this run; this is the newest stored result." in stored


def test_full_report_section3_jackpot_history(ctx):
    text = section(report.full_report(ctx), 2)
    h = ctx.history
    assert h.draws == 260
    assert md_cell(text, "Draws stored") == "260"
    assert md_cell(text, "Draws where Group 1 was won") == f"{h.won_draws} ({pct(h.won_share, 0)})"
    assert md_cell(text, "Cascades (nobody won it by the 4th draw)") == str(h.cascades)
    assert md_cell(text, "Average draws a jackpot lasts") == fmt_num(h.average_run, 1)
    assert md_cell(text, "Typical Group 1 prize when won (median)") == money(h.typical_won)
    assert md_cell(text, "Last won") == "draw 4122, Mon 28 Sep 2026, $1,000,000, 1 winning share"
    assert "Biggest Group 1 prizes stored:" in text
    for b in h.biggest:
        assert f"| {b['draw_number']} | {fmt_date(b['draw_date'])}" in text
    assert "### Prize rules used" in text
    assert report.history_md(None) == "No results are stored yet."
    assert report.history_md(JackpotHistory()) == "No results are stored yet."


def test_full_report_prize_rules_status(ctx):
    text = report.full_report(ctx)
    assert "TOTO prize rules: not confirmed, built in values used." in text
    assert "prize rules could not be confirmed" in text
    confirmed = PrizeRules(toto_confirmed=True, source_note="TOTO prize rules confirmed from the official page",
                           checked_at="2026-10-01T19:00:00+08:00")
    text2 = report.full_report(variant(ctx, rules=confirmed))
    assert text2.count("confirmed on the official prize page") == 1
    assert "TOTO prize rules confirmed from the official page." in text2
    assert "Last checked: Thu 1 Oct 2026, 7.00pm." in text2
    assert "could not be confirmed" not in text2
    assert "> [!warning]" not in text2


def test_full_report_lists_warnings(ctx):
    settings_warned = variant(ctx.settings, warnings=["jackpot_alert was minus 5, using $0 instead"])
    text = report.full_report(variant(ctx, warnings=["Could not reach the site - used the stored data",
                                                     "Could not reach the site - used the stored data"],
                                      settings=settings_warned))
    assert "> [!warning] Warnings" in text
    assert text.count("> * Could not reach the site, used the stored data") == 1
    assert "> * jackpot_alert was minus 5, using $0 instead" in text
    assert not has_prose_dashes(text)


def test_full_report_states_the_odds_note_once(ctx):
    text = report.full_report(ctx)
    note = report.odds_note(ctx.rules)
    assert text.count(note) == 1
    assert section(text, 3).split("\n\n", 1)[1].startswith(note)


def test_full_report_includes_commentary_when_present(ctx):
    text = report.full_report(variant(ctx, commentary="A calm week — nothing special."))
    assert text.rstrip().endswith("*Commentary:* A calm week, nothing special.")
    assert not has_prose_dashes(text)


def test_full_report_with_nothing_stored(scenarios):
    text = report.full_report(scenarios["empty_history"])
    assert "No TOTO results are stored yet." in text
    assert "There is no jackpot projection in this run (no jackpot estimate and no stored results)." in text
    assert md_cell(text, "Estimated jackpot") == "not available yet"
    assert md_cell(text, "Draw") == "n/a"


# Odds note and small helpers


def test_odds_note_content():
    note = report.odds_note(PrizeRules())
    assert note.count("independent") == 1
    assert "1 in 13,983,816 per board" in note
    assert "about 1 in 54" in note
    assert "54% of sales" in note
    assert not contains_dash(note)
    line = report.odds_line()
    assert line.count("independent") == 1 and "1 in 13,983,816" in line and "about 1 in 54" in line


def test_rollover_text():
    assert report.rollover_text({"rollovers": 2, "draw_type": "normal"}) == "2 of 3, then it cascades"
    assert report.rollover_text({"rollovers": 3, "draw_type": "cascade"}) == "3 of 3, this is the cascade draw"
    # a cascade draw announced by the page counts as the last rollover even if the history is short
    assert report.rollover_text({"rollovers": 1, "draw_type": "cascade"}) == "3 of 3, this is the cascade draw"
    assert report.rollover_text({"rollovers": 2, "draw_type": "hongbao"}) == "2"
    assert report.rollover_text({"rollovers": 0, "draw_type": "special"}) == "0"


def _step(i, day, jackpot, reached, won, cascade=False):
    return OutlookStep(index=i, draw_date=day, jackpot=jackpot, boards=4e6, chance_reached=reached,
                       chance_won=won, cascade=cascade, ev_per_dollar=0.5)


def test_big_prize_text():
    assert report.big_prize_text(None) is None
    assert report.big_prize_text(JackpotOutlook(jackpot=None, draw_type="normal", snowball_draws=0,
                                                draws_to_cascade=4)) is None
    special = JackpotOutlook(jackpot=8e6, draw_type="special", snowball_draws=0, draws_to_cascade=None,
                             steps=[_step(1, date(2026, 10, 9), 8e6, 1.0, 0.3)])
    assert report.big_prize_text(special) == "The next draw is a Special draw with a jackpot of about $8,000,000."
    cascade = JackpotOutlook(jackpot=4e6, draw_type="cascade", snowball_draws=3, draws_to_cascade=1,
                             steps=[_step(1, date(2026, 10, 5), 4e6, 1.0, 0.3, True)])
    assert report.big_prize_text(cascade) == ("The next draw is the cascade draw: about $4,000,000. If nobody "
                                              "wins it, the jackpot goes to the Group 2 winners.")
    two = JackpotOutlook(jackpot=3e6, draw_type="normal", snowball_draws=2, draws_to_cascade=2,
                         steps=[_step(1, date(2026, 10, 5), 3e6, 1.0, 0.25),
                                _step(2, date(2026, 10, 8), 4e6, 0.75, 0.2, True)])
    assert report.big_prize_text(two) == ("If nobody wins Group 1 first, the jackpot snowballs to about $4,000,000 "
                                          "at the cascade draw on Thu 8 Oct 2026 (75% chance it gets that far). "
                                          "The chance somebody wins it before then is about 40%.")


def test_specials_and_history_text():
    out = JackpotOutlook(jackpot=None, draw_type="normal", snowball_draws=0, draws_to_cascade=4,
                         special_draws=[(date(2026, 10, 9), "hongbao")])
    assert report.specials_text(out) == "Announced special draws: Fri 9 Oct 2026 (Hongbao)."
    assert report.specials_text(variant(out, special_draws=[])) is None
    assert report.history_line(None) is None and report.history_line(JackpotHistory()) is None
    h = JackpotHistory(draws=1, won_draws=0)
    assert report.history_line(h) == "Over 1 stored draw Group 1 was won in 0% of draws."


def test_number_formatters():
    assert report.boards_text(3_951_784.5) == "3,952,000"
    assert report.boards_text(None) == "n/a"
    assert report.short_money(2_100_000) == "$2.10m"
    assert report.short_money(950_000) == "$950,000"
    assert report.short_money(None) == "n/a"
    assert report.short_date(date(2026, 10, 5)) == "Mon 5 Oct"
    assert report.dollars(2.5) == "$2.50" and report.dollars(10) == "$10" and report.dollars(None) == "n/a"


# Next draw: page, schedule, stale and held


def _with_draw(df, number, day):
    """``df`` with one more draw (a copy of the newest row) numbered ``number`` on ``day``."""
    row = df.iloc[[-1]].copy()
    row["draw_number"] = number
    row["draw_date"] = pd.Timestamp(day)
    return pd.concat([df, row], ignore_index=True)


def test_next_draw_from_the_page_and_from_the_schedule(ctx):
    nd = report.next_draw(ctx)
    assert nd.number == 4124 and str(nd.day) == "2026-10-05" and not nd.from_schedule
    assert not nd.stale and not nd.held
    assert report.stale_text(ctx) is None and report.held_text(ctx) is None
    sched = report.next_draw(variant(ctx, next_toto=None))
    # No number for a schedule date: a special draw before it would take 4124.
    assert sched.number is None and str(sched.day) == "2026-10-05" and sched.from_schedule and not sched.stale
    assert sched.when_text == "Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)"


def test_next_draw_later_today_is_still_the_next_draw(ctx):
    monday = variant(ctx, now=_sg(2026, 10, 5, 17, 0))
    nd = report.next_draw(monday)
    assert str(nd.day) == "2026-10-05" and nd.number == 4124 and not nd.stale and not nd.held
    assert "held" not in report.telegram_messages(monday)[1]
    assert report.next_info_is_current(monday)


def test_next_draw_page_showing_the_draw_just_held_is_not_used(ctx):
    # The 7.30pm run on Thu 1 Oct: the next draw page still shows tonight's draw (already stored).
    held = NextToto(draw_datetime=_sg(2026, 10, 1, 18, 30), jackpot_estimate=5_000_000.0, draw_type="normal",
                    draw_type_hint=None)
    stale = rebuilt(ctx, held)
    assert not report.next_info_is_current(stale)
    sig = report.toto_signal(stale)
    assert sig["jackpot"] == stale.outlook.jackpot != 5_000_000.0
    assert sig["jackpot_worked_out"]
    msg = report.telegram_messages(stale)[1]
    assert "$5,000,000" not in msg
    assert msg.startswith("🔮 <b>NEXT TOTO DRAW</b> · Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)")
    assert f"<b>Jackpot {money(sig['jackpot'])}</b> <i>(worked out from past results)</i>" in msg
    assert "Buy signal <b>MEDIUM</b>" in msg
    assert "$5,000,000" not in report.full_report(stale)
    # Even a hand made context whose signal still holds the old page jackpot shows the outlook's.
    assert "$5,000,000" not in report.telegram_messages(variant(ctx, next_toto=held))[1]


def test_stored_results_a_week_old_are_marked_not_up_to_date(ctx):
    # The site could not be read for a week: stored results end Thu 1 Oct, it is now Fri 9 Oct,
    # and the stored next draw (Mon 5 Oct) is past.
    week = rebuilt(variant(ctx, now=_sg(2026, 10, 9), fetched=False), ctx.next_toto)
    nd = report.next_draw(week)
    assert nd.stale and nd.number is None and nd.from_schedule and str(nd.day) == "2026-10-12"
    assert nd.when_text == ("Mon 12 Oct 2026, 6.30pm (worked out from the regular schedule, results are not up "
                            "to date)")
    assert [s.draw_date for s in week.outlook.steps][0] == date(2026, 10, 12)
    msgs = report.telegram_messages(week)
    assert ("<i>Results not up to date: the newest stored result is draw 4123, and newer draws have been held "
            "since.") in msgs[0]
    assert "Mon 5 Oct" not in msgs[1] and "Thu 8 Oct" not in msgs[1]
    assert "$2,100,000" not in msgs[1]
    assert "(worked out from past results)" in msgs[1]
    assert "Results not up to date" in report.full_report(week)
    for msg in msgs:
        assert_valid_message(msg)
    assert "Results not up to date" not in report.telegram_messages(ctx)[0]


def test_next_draw_after_a_missed_draw_has_no_number(ctx):
    # Wed 7 Oct: TOTO 4123 (Thu 1 Oct) is stored, Mon 5 Oct (4124) was missed because the site
    # could not be read. The next draw page shows Thu 8 Oct, which is draw 4125, not 4124.
    c = variant(ctx, now=_sg(2026, 10, 7), fetched=False, new_draws=[],
                next_toto=NextToto(_sg(2026, 10, 8, 18, 30), 1_500_000.0, "normal", None))
    nd = report.next_draw(c)
    assert nd.number is None and nd.stale and not nd.from_schedule and str(nd.day) == "2026-10-08"
    assert nd.when_text == "Thu 8 Oct 2026, 6.30pm (results are not up to date)"
    msgs = report.telegram_messages(c)
    assert "Results not up to date: the newest stored result is draw 4123" in msgs[0]
    assert msgs[1].startswith("🔮 <b>NEXT TOTO DRAW</b> · Thu 8 Oct 2026, 6.30pm (results are not up to date)")
    assert "4124" not in msgs[1] and "4125" not in msgs[1]
    for msg in msgs:
        assert_valid_message(msg)


def test_page_date_after_a_moved_draw_keeps_its_number_when_results_are_up_to_date(ctx):
    # Mon 5 Oct: this run read TOTO 4124 from the site, the newest draw there. The next draw page
    # says Fri 9 Oct 9.30pm (a Hongbao draw held in place of Thursday's): the regular Thursday in
    # between had no draw, so the results are up to date and the Friday draw is 4125.
    toto = _with_draw(ctx.toto, 4124, date(2026, 10, 5))
    c = variant(ctx, toto=toto, now=_sg(2026, 10, 5, 19, 45), new_draws=[4124], next_toto=HONGBAO_FRI)
    nd = report.next_draw(c)
    assert nd.number == 4125 and not nd.stale and not nd.from_schedule and str(nd.day) == "2026-10-09"
    assert report.stale_text(c) is None
    msgs = report.telegram_messages(c)
    assert "not up to date" not in msgs[0] and "not up to date" not in msgs[1]
    assert msgs[1].startswith("🔮 <b>NEXT TOTO DRAW</b> · Fri 9 Oct 2026, 9.30pm, draw 4125\n")
    # Still marked stale when the site has a newer draw than the stored ones, or when this run
    # did not read the site.
    behind = variant(c, warnings=["TOTO: the newest stored draw (4124) does not match the latest draw on "
                                  "the site (4125)."])
    not_read = variant(c, fetched=False)
    for other in (behind, not_read):
        nd = report.next_draw(other)
        assert nd.stale and nd.number is None and str(nd.day) == "2026-10-09"


def test_schedule_date_after_a_lagging_next_draw_page_has_no_number(ctx):
    # Mon 5 Oct 7.45pm: TOTO 4124 is stored and the next draw page still shows it. Thu 8 Oct is
    # the next regular draw day, but a special draw before it would be 4125: no number is given.
    toto = _with_draw(ctx.toto, 4124, date(2026, 10, 5))
    c = variant(ctx, toto=toto, now=_sg(2026, 10, 5, 19, 45), new_draws=[4124],
                next_toto=NextToto(_sg(2026, 10, 5, 18, 30), 1_000_000.0, "normal", None))
    nd = report.next_draw(c)
    assert nd.from_schedule and nd.number is None and not nd.stale and str(nd.day) == "2026-10-08"
    msgs = report.telegram_messages(c)
    assert msgs[1].startswith("🔮 <b>NEXT TOTO DRAW</b> · Thu 8 Oct 2026, 6.30pm (regular schedule, not announced "
                              "yet)\n")
    for msg in msgs:
        assert "draw 4125" not in msg and "4125)" not in msg
        assert_valid_message(msg)


def test_draw_held_earlier_today_gets_no_signal(ctx):
    # Mon 5 Oct at 9pm: the 6.30pm TOTO draw is held, its result is late (not stored yet).
    late = variant(ctx, now=_sg(2026, 10, 5, 21, 0))
    nd = report.next_draw(late)
    assert nd.held and nd.number == 4124 and str(nd.day) == "2026-10-05" and not nd.stale
    assert nd.when_text == "Mon 5 Oct 2026, 6.30pm (draw held, result not out yet)"
    sig = report.toto_signal(late)
    assert sig["label"] is None and sig["ev"] is None and sig["breakdown"] == {} and sig["boards"] is None
    assert sig["jackpot"] == 2_100_000.0 and not sig["jackpot_worked_out"]
    msgs = report.telegram_messages(late)
    assert "<i>TOTO draw 4124 was held at 6.30pm today, result not out yet.</i>" in msgs[0]
    assert "(draw held, result not out yet)" in msgs[1]
    assert "Its sales are closed, so there is no buy signal for it." in msgs[1]
    assert "Buy signal" not in msgs[1] and "back per $1" not in msgs[1] and "Sales estimate" not in msgs[1]
    for msg in msgs:
        assert_valid_message(msg)
    text = report.full_report(late)
    assert "**TOTO draw 4124 was held at 6.30pm today, result not out yet.**" in text
    assert ("This draw was held at 6.30pm today and its result is not out yet, so its sales are closed and there "
            "is no buy signal for it.") in text
    assert md_cell(text, "Buy signal") == "not available"
    assert not has_prose_dashes(text)


# A checked ticket corrected in Tickets.md


def _corrected(ctx, before: list[str], after: list[str]):
    """``ctx`` after two runs: ``before`` rows checked at 7pm on Thu 1 Oct, then the note changed
    to ``after`` and checked again in this run (ctx.now)."""
    head = "| Game | Draw date | Numbers | Bet type | Cost |\n| --- | --- | --- | --- | --- |\n"
    ledger = empty_ledger()
    settled: list[dict] = []
    for rows, when in ((before, _sg(2026, 10, 1, 19, 0)), (after, ctx.now)):
        text = head + "".join(r + "\n" for r in rows)
        ledger = tickets.sync_ledger(ledger, tickets.parse_tickets(text), when, note_read=True)
        ledger, settled = tickets.settle_ledger(ledger, ctx.toto, ctx.rules, when)
    return variant(ctx, ledger=ledger, ledger_totals=ledger_totals(ledger), settled_this_run=settled,
                   bad_ticket_lines=[])


def _numbers(ctx, matched: int) -> str:
    """A ticket on the newest draw that matches ``matched`` winning numbers (no additional)."""
    row = report.latest_row(ctx.toto)
    win = toto_numbers(row)
    others = [n for n in range(1, 50) if n not in win and n != int(row["additional"])]
    return " ".join(str(n) for n in sorted(win[:matched] + others[:6 - matched]))


def test_a_corrected_ticket_counts_only_what_it_won_above_the_old_check(ctx):
    g7 = _numbers(ctx, 3)
    c = _corrected(ctx, [f"| TOTO | 1 Oct 2026 | {g7} | Ordinary | $1 |"],
                   [f"| TOTO | 1 Oct 2026 | {g7} | Ordinary | $3 |"])
    assert len(c.settled_this_run) == 1
    assert report.run_winnings(c) == (pytest.approx(20.0), 1)
    msg = report.telegram_messages(c)[0]
    assert msg.startswith("🎉 <b>WINNER</b> · your tickets won $20")
    assert (f"🟢 Thu 1 Oct 2026, {g7}, Ordinary $3: Group 7 x1, corrected from {g7} Ordinary $1, was won $10, "
            "now won <b>$30</b>") in msg
    assert report.CORRECTION_HINT in msg
    assert "All tickets so far: spent $3, won $30, net <b>$27</b>." in msg
    assert_valid_message(msg)
    text = section(report.full_report(c), 1)
    assert "**You won $20** with 1 ticket in this run." in text
    assert f"Group 7 x1, corrected from {g7} Ordinary $1, was won $10" in text
    assert report.CORRECTION_HINT in text
    assert not has_prose_dashes(text)
    # Without a correction nothing changes.
    assert "corrected" not in report.telegram_messages(ctx)[0]
    assert report.run_winnings(ctx)[0] == pytest.approx(sum(r["winnings"] for r in ctx.settled_this_run))


def test_a_checked_winner_corrected_into_a_loser_is_announced(ctx):
    g7, loser = _numbers(ctx, 3), _numbers(ctx, 2)
    c = _corrected(ctx, [f"| TOTO | 1 Oct 2026 | {g7} | Ordinary | $1 |"],
                   [f"| TOTO | 1 Oct 2026 | {loser} | Ordinary | $1 |"])
    assert report.run_winnings(c) == (0.0, 0)
    line = f"Thu 1 Oct 2026, {loser}, Ordinary $1: No prize, corrected from {g7} Ordinary $1, was won $10"
    msg = report.telegram_messages(c)[0]
    assert "WINNER" not in msg and line in msg and report.CORRECTION_HINT in msg
    for level in (1, 2):  # a corrected ticket is listed even when losing tickets are not
        assert line in report._tg_tickets(c, level)
    compact = report._tg_tickets(c, 3)
    assert "1 ticket checked, 0 won, $0 in total." in compact
    assert "1 of them is a corrected row of a ticket already checked, see Ledger.md." in compact
    assert report.CORRECTION_HINT in compact


# Guard: the removed features never come back in user facing text

FORBIDDEN = {
    "4D": re.compile(r"\b4\s?D\b|four\s?d", re.I),
    "suggest": re.compile(r"suggest", re.I),
    "pick": re.compile(r"\bpick", re.I),
    "budget": re.compile(r"budget", re.I),
    "System 7 offer": re.compile(r"System 7 (?:offer|option|alternative)", re.I),
    "backtest": re.compile(r"back\s?test", re.I),
    "hot": re.compile(r"\bhot\b", re.I),
    "cold": re.compile(r"\bcold\b", re.I),
    "overdue": re.compile(r"overdue", re.I),
    "chi square": re.compile(r"chi\s?square", re.I),
    "crowd": re.compile(r"\bcrowd", re.I),
}


def forbidden_mentions(text: str, allowed: tuple[str, ...] = ()) -> list[str]:
    for phrase in allowed:
        text = text.replace(phrase, "")
    return [name for name, pattern in FORBIDDEN.items() if pattern.search(text)]


def _notes_text(ctx, tmp_path) -> dict[str, str]:
    vault = Vault(tmp_path / "vault", "Huat Bot")
    vault.ensure_layout()
    notes.write_all(vault, ctx, report.full_report(ctx))
    return {str(p.relative_to(vault.base)): p.read_text(encoding="utf-8") for p in vault.base.rglob("*.md")}


def test_guard_forbidden_patterns_catch_the_old_features():
    for text in ("4D draw 5432", "Suggested numbers", "Your picks", "of your $10 budget", "System 7 option",
                 "Backtest", "Hot numbers", "cold numbers", "Overdue", "chi square", "Crowd score"):
        assert forbidden_mentions(text), text
    assert forbidden_mentions("A System 7 ticket, the shortest draw, 4 Oct, picture") == []


@pytest.mark.parametrize("name", SCENARIOS)
def test_guard_no_removed_feature_in_messages_or_report(name, scenarios):
    c = scenarios[name]
    for text in report.telegram_messages(c) + [report.full_report(c)]:
        assert forbidden_mentions(text) == []


def test_guard_no_removed_feature_in_the_notes(ctx, tmp_path):
    written = _notes_text(ctx, tmp_path)
    assert {"Home.md", "Ledger.md", "Draws/TOTO/2026-10-01 TOTO 4123.md",
            "Reports/2026/10/2026-10-01 1930 Report.md"} <= set(written)
    # The only 4D mention allowed: the user's own 4D line in Tickets.md and why it is not counted.
    bad = next(t for t in ctx.bad_ticket_lines if t.game == "4D")
    allowed = (tickets.FOURD_NOT_TRACKED, bad.source)
    assert tickets.FOURD_NOT_TRACKED in written["Ledger.md"]
    for rel, text in written.items():
        assert forbidden_mentions(text, allowed) == [], rel
        # The activity log is a bullet list: its "- " markers are list syntax, not prose.
        assert not has_prose_dashes(re.sub(r"(?m)^- ", "", text)), rel
    clean = _notes_text(make_context(bad_ticket_lines=False), tmp_path / "clean")
    for rel, text in clean.items():
        assert forbidden_mentions(text) == [], rel


def test_guard_old_fourd_ledger_rows_leave_no_feature_text_in_the_notes(ctx, tmp_path):
    c = _with_old_fourd_rows(variant(ctx, bad_ticket_lines=[]))
    written = _notes_text(c, tmp_path)
    for rel, text in written.items():
        assert forbidden_mentions(text, (tickets.FOURD_NOT_TRACKED,)) == [], rel
