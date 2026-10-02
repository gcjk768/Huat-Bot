"""Tests for huatbot.notes: the Obsidian notes the bot writes into the vault (TOTO only).

Also holds the guard that no user facing text (Telegram messages, the report, the notes) still
offers the removed features: 4D, number suggestions or picks, budgets, the System 7 offer,
backtests, hot, cold or overdue numbers.
"""
from __future__ import annotations

import dataclasses
import re
from datetime import date, datetime, timezone

import pandas as pd
import pytest
import yaml

from huatbot import notes, report, tickets
from huatbot.models import LEDGER_COLUMNS, NextToto, PrizeRules, Settings, Ticket
from huatbot.outlook import chance_won_by_cascade
from huatbot.store import empty_ledger, normalise_ledger, toto_numbers
from huatbot.synth import synth_next_toto
from huatbot.textfmt import find_prose_dashes, has_prose_dashes, money, pct, per_dollar, toto_nums
from huatbot.tickets import ledger_totals
from huatbot.vault import Vault, parse_note, render_note
from tests.ctxgen import cascade_next_toto, make_context, rebuilt, unknown_jackpot, variant

SG = report.SG
REPORT_NAME = "2026-10-01 1930 Report"
LOG_NAME = "2026-10 Activity"
DRAW_4123 = "Draws/TOTO/2026-10-01 TOTO 4123.md"
DRAW_4122 = "Draws/TOTO/2026-09-28 TOTO 4122.md"
DRAW_4123_LINK = "[[Draws/TOTO/2026-10-01 TOTO 4123|2026-10-01 TOTO 4123]]"
DASHBOARD_HEADINGS = ["## Next draw", "## Buy signal", "## The next big prize", "## Latest result",
                      "## My tickets", "## Warnings"]


@pytest.fixture(scope="module")
def ctx():
    return make_context()


@pytest.fixture(scope="module")
def report_md(ctx):
    return report.full_report(ctx)


# Helpers


def _yaml_block(text: str) -> dict:
    """Parse the frontmatter block with plain yaml.safe_load (independent of the vault parser)."""
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    assert m, "note has no frontmatter block"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict)
    return data


def _check_note(spec):
    """Common checks for every note; returns (path, parsed frontmatter, body)."""
    rel, frontmatter, body = spec
    assert rel.endswith(".md")
    assert isinstance(frontmatter, dict) and "huatbot" in frontmatter["tags"]
    text = render_note(body, frontmatter)
    parsed, parsed_body = parse_note(text)
    assert parsed["tags"] == frontmatter["tags"]
    assert _yaml_block(text)["tags"] == frontmatter["tags"]
    assert not has_prose_dashes(text), find_prose_dashes(text)
    return rel, parsed, parsed_body


def _section(body: str, heading: str) -> str:
    """Text under "## heading" up to the next "## " heading."""
    start = body.index(f"## {heading}\n") + len(heading) + 4
    end = body.find("\n## ", start)
    return body[start:end if end >= 0 else None].strip("\n")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _table(text: str) -> list[list[str]]:
    """Rows (header first) of the markdown tables in ``text``, separator rows left out."""
    return [_cells(ln) for ln in text.splitlines()
            if ln.startswith("|") and not re.fullmatch(r"[|\s:\-]+", ln)]


def _kv(text: str) -> dict[str, str]:
    rows = _table(text)
    assert rows[0] == ["Item", "Value"]
    return {k: v for k, v in rows[1:]}


def _with_streak(ctx, k: int):
    """ctx whose jackpot has rolled over ``k`` times (the newest k draws had no Group 1 winner),
    with the next draw page and the signal, outlook and history worked out again."""
    t = ctx.toto.copy()
    n = len(t)
    t.loc[t.index[n - k:], "g1_winners"] = 0
    t.loc[t.index[n - k:], "draw_type"] = "normal"
    t.loc[t.index[n - k - 1], "g1_winners"] = 1
    t.loc[t.index[n - k - 1], "draw_type"] = "normal"
    return rebuilt(variant(ctx, toto=t), synth_next_toto(t))


# Draw notes


def test_toto_draw_note(ctx):
    row = report.latest_row(ctx.toto)
    assert int(row["g1_winners"]) == 0
    rel, fm, body = _check_note(notes.toto_draw_note(row))
    assert rel == DRAW_4123
    assert fm == {
        "tags": ["huatbot", "toto", "draw"],
        "draw": 4123,
        "date": "2026-10-01",
        "numbers": toto_numbers(row),
        "additional": int(row["additional"]),
        "draw_type": "normal",
        "group1_prize": float(row["jackpot"]),
        "group1_winners": 0,
    }
    assert body.startswith("# TOTO draw 4123, Thu 1 Oct 2026\n\nBack to [[Dashboard]].\n\n")
    assert (f"Winning numbers: **{toto_nums(toto_numbers(row))}**, additional number "
            f"**{int(row['additional'])}**.") in body
    assert f"Draw type: Normal. Group 1 prize: {money(row['jackpot'])}, no winner." in body
    groups = _table(body)
    assert groups[0] == ["Group", "Share amount", "Winning shares"]
    assert [r[0] for r in groups[1:]] == [f"Group {g}" for g in range(1, 8)]
    assert groups[1] == ["Group 1", "n/a", "0"]
    assert "Shape of this set" not in body  # the number analysis is gone


def test_toto_draw_note_with_winners_on_a_hongbao_draw(ctx):
    row = report.latest_row(ctx.toto).copy()
    row["draw_type"], row["jackpot"], row["g1_winners"], row["g1_share"] = "hongbao", 12_000_000.0, 2, 6_000_000.0
    rel, fm, body = _check_note(notes.toto_draw_note(row))
    assert rel == DRAW_4123
    assert fm["draw_type"] == "hongbao" and fm["group1_prize"] == 12_000_000.0 and fm["group1_winners"] == 2
    assert "Draw type: Hongbao. Group 1 prize: $12,000,000, 2 winning shares of $6,000,000 each." in body
    assert _table(body)[1] == ["Group 1", "$6,000,000", "2"]


def test_toto_draw_note_accepts_a_dict_row(ctx):
    row = report.latest_row(ctx.toto).to_dict()
    row["draw_date"] = "2026-10-01"
    rel, fm, _ = _check_note(notes.toto_draw_note(row))
    assert rel == DRAW_4123 and fm["date"] == "2026-10-01" and fm["draw"] == 4123


# Dashboard


def test_dashboard_links_and_headings(ctx):
    rel, _, body = _check_note(notes.dashboard_note(ctx))
    assert rel == "Dashboard.md"
    for name in (REPORT_NAME, "Ledger", "Tickets", "Settings", LOG_NAME):
        assert f"[[{name}]]" in body, name
    assert DRAW_4123_LINK in body
    assert [ln for ln in body.splitlines() if ln.startswith("## ")] == DASHBOARD_HEADINGS
    assert "Suggestions/" not in body and "Draws/4D" not in body
    # Another report name (an earlier report of the same day) is linked when given.
    assert "Newest report: [[2026-10-01 1900 Report]]." in notes.dashboard_note(ctx, "2026-10-01 1900 Report")[2]


def test_dashboard_frontmatter(ctx):
    _, fm, _ = _check_note(notes.dashboard_note(ctx))
    out, bs = ctx.outlook, ctx.buy_signal
    assert len(out.steps) == 3 and out.biggest is out.steps[-1] and out.biggest.cascade
    assert fm == {
        "tags": ["huatbot", "dashboard"],
        "updated": "2026-10-01T19:30:00+08:00",
        "next_draw": 4124,
        "next_date": "2026-10-05",
        "next_jackpot": 2_100_000.0,
        "next_draw_type": "normal",
        "chance_jackpot_won": round(out.steps[0].chance_won, 4),
        "next_big_prize": round(out.biggest.jackpot),
        "next_big_prize_date": "2026-10-12",
        "buy_signal": bs.label,
        "return_per_dollar": round(bs.ev_per_dollar, 4),
        "latest_draw": 4123,
        "spent": 10.0,
        "won": 10.0,
        "net": 0.0,
    }
    assert 0 < fm["chance_jackpot_won"] < 1 and fm["next_big_prize"] > fm["next_jackpot"]


def test_dashboard_next_draw_table(ctx):
    body = notes.dashboard_note(ctx)[2]
    assert _kv(_section(body, "Next draw")) == {
        "Draw": "4124",
        "Date and time": "Mon 5 Oct 2026, 6.30pm",
        "Estimated jackpot": "$2,100,000",
        "Draw type": "Normal",
        "Jackpot rollovers so far": "1 of 3, then it cascades",
        "Chance somebody wins Group 1": pct(ctx.outlook.steps[0].chance_won, 0),
    }


@pytest.mark.parametrize("streak, rollovers, steps, big_date", [
    (0, "0 of 3, then it cascades", 4, "2026-10-15"),
    (1, "1 of 3, then it cascades", 3, "2026-10-12"),
    (2, "2 of 3, then it cascades", 2, "2026-10-08"),
    (3, "3 of 3, this is the cascade draw", 1, "2026-10-05"),
])
def test_dashboard_rollovers_count_up_to_the_cascade(ctx, streak, rollovers, steps, big_date):
    c = _with_streak(ctx, streak)
    _, fm, body = _check_note(notes.dashboard_note(c))
    table = _kv(_section(body, "Next draw"))
    assert table["Jackpot rollovers so far"] == rollovers
    assert table["Draw type"] == ("Cascade" if streak == 3 else "Normal")
    assert len(c.outlook.steps) == steps
    assert fm["next_big_prize_date"] == big_date
    assert fm["next_big_prize"] == round(c.outlook.biggest.jackpot)
    projection = _table(_section(body, "The next big prize"))
    assert len(projection) == 1 + steps and projection[-1][-1] == "cascade draw"
    if streak == 3:
        assert (f"The next draw is the cascade draw: about {money(c.outlook.biggest.jackpot)}. If nobody wins "
                "it, the jackpot goes to the Group 2 winners.") in body


def test_dashboard_buy_signal(ctx):
    bs = ctx.buy_signal
    section = _section(notes.dashboard_note(ctx)[2], "Buy signal")
    assert section == (f"**{bs.label}**. {report.sentence(bs.reason)} Return per $1: about "
                       f"{per_dollar(bs.ev_per_dollar)}.")


def test_dashboard_next_big_prize(ctx):
    out = ctx.outlook
    section = _section(notes.dashboard_note(ctx)[2], "The next big prize")
    assert section == report.outlook_md(out)
    big = out.biggest
    assert (f"If nobody wins Group 1 first, the jackpot snowballs to about {money(big.jackpot)} at the cascade "
            f"draw on Mon 12 Oct 2026 ({pct(big.chance_reached, 0)} chance it gets that far). The chance "
            f"somebody wins it before then is about {pct(chance_won_by_cascade(out), 0)}.") in section
    rows = _table(section)
    assert [r[0] for r in rows[1:]] == ["Mon 5 Oct 2026", "Thu 8 Oct 2026", "Mon 12 Oct 2026"]
    assert rows[1][1] == "$2,100,000" and rows[1][3] == "100%"
    assert [r[-1] for r in rows[1:]] == ["", "", "cascade draw"]


def test_dashboard_latest_result_and_history(ctx):
    row = report.latest_row(ctx.toto)
    section = _section(notes.dashboard_note(ctx)[2], "Latest result")
    first, second = section.split("\n\n")
    assert first == (f"Draw 4123, Thu 1 Oct 2026: **{toto_nums(toto_numbers(row))}**, additional "
                     f"{int(row['additional'])}. Group 1 {money(row['jackpot'])}, no winner. {DRAW_4123_LINK}")
    assert second == report.history_line(ctx.history)
    assert second.startswith("Over 260 stored draws Group 1 was won in ")

    t = ctx.toto.copy()
    t.loc[t.index[-1], "g1_winners"] = 2
    won = _section(notes.dashboard_note(variant(ctx, toto=t))[2], "Latest result")
    assert f"Group 1 {money(row['jackpot'])}, 2 winners." in won


def test_dashboard_my_tickets(ctx):
    section = _section(notes.dashboard_note(ctx)[2], "My tickets")
    assert _table(section) == [["Tickets", "Spent", "Won", "Net", "Waiting", "Waiting cost"],
                               ["4", "$10", "$10", "$0", "2", "$8"]]
    assert "Checked in this run: 2 tickets, won $10. Details in [[Ledger]]." in section
    assert "2 lines in [[Tickets]] could not be read, see [[Ledger]]." in section


def test_dashboard_without_tickets():
    c = make_context(with_tickets=False)
    section = _section(notes.dashboard_note(c)[2], "My tickets")
    assert _table(section)[1] == ["0", "$0", "$0", "$0", "0", "$0"]
    assert "Checked in this run" not in section and "could not be read" not in section


def test_dashboard_warnings(ctx):
    body = notes.dashboard_note(ctx)[2]
    assert _section(body, "Warnings") == ("* The TOTO prize rules could not be confirmed on the official page, "
                                          "so built in values were used.")
    noisy = variant(ctx, warnings=["TOTO: the site was slow - retried twice", "TOTO: the site was slow - retried twice"],
                    settings=Settings(warnings=["Settings.md: jackpot_alert is not a number"]))
    _, _, body = _check_note(notes.dashboard_note(noisy))
    assert _section(body, "Warnings").splitlines() == [
        "* TOTO: the site was slow, retried twice",
        "* Settings.md: jackpot_alert is not a number",
        "* The TOTO prize rules could not be confirmed on the official page, so built in values were used.",
    ]
    calm = variant(ctx, rules=PrizeRules(toto_confirmed=True), warnings=[])
    assert "## Warnings" not in notes.dashboard_note(calm)[2]


def test_dashboard_without_next_draw_info(ctx):
    bare = variant(ctx, next_toto=None, buy_signal=None, outlook=None)
    _, fm, body = _check_note(notes.dashboard_note(bare))
    assert _kv(_section(body, "Next draw")) == {
        "Draw": "n/a",  # a date from the regular schedule has no number: a special draw could take it
        "Date and time": "Mon 5 Oct 2026, 6.30pm (regular schedule, not announced yet)",
        "Estimated jackpot": "not announced yet",
        "Draw type": "Normal",
        "Jackpot rollovers so far": "1 of 3, then it cascades",
        "Chance somebody wins Group 1": "n/a",
    }
    assert _section(body, "Buy signal") == "Not worked out in this run."
    assert _section(body, "The next big prize") == ("There is no jackpot projection in this run (no jackpot "
                                                    "estimate and no stored results).")
    for key in ("next_draw", "next_jackpot", "chance_jackpot_won", "next_big_prize", "next_big_prize_date",
                "buy_signal", "return_per_dollar"):
        assert fm[key] is None, key
    assert fm["next_date"] == "2026-10-05" and fm["next_draw_type"] == "normal"


def test_dashboard_jackpot_worked_out_from_past_results(ctx):
    for c in (unknown_jackpot(ctx), rebuilt(ctx, None)):
        assert report.toto_signal(c)["jackpot_worked_out"]
        _, fm, body = _check_note(notes.dashboard_note(c))
        table = _kv(_section(body, "Next draw"))
        assert table["Estimated jackpot"] == f"{money(c.outlook.jackpot)} (worked out from past results)"
        assert fm["next_jackpot"] == c.outlook.jackpot and fm["buy_signal"] == c.buy_signal.label
        assert ("The next draw page gave no usable jackpot, so the next jackpot is worked out from the stored "
                "results.") in _section(body, "The next big prize")
    # A page jackpot is shown as it is.
    assert not report.toto_signal(ctx)["jackpot_worked_out"]
    assert _kv(_section(notes.dashboard_note(ctx)[2], "Next draw"))["Estimated jackpot"] == "$2,100,000"
    # With no page at all the date comes from the regular schedule and has no draw number.
    no_page = rebuilt(ctx, None)
    assert notes.dashboard_note(no_page)[1]["next_draw"] is None


def test_dashboard_cascade_draw(ctx):
    c = cascade_next_toto(ctx)
    _, fm, body = _check_note(notes.dashboard_note(c))
    table = _kv(_section(body, "Next draw"))
    assert table["Draw type"] == "Cascade" and table["Estimated jackpot"] == "$4,500,000"
    assert ("The next draw is the cascade draw: about $4,500,000. If nobody wins it, the jackpot goes to the "
            "Group 2 winners.") in _section(body, "The next big prize")
    assert fm["next_draw_type"] == "cascade"
    assert fm["next_big_prize"] == 4_500_000 and fm["next_big_prize_date"] == "2026-10-05"


def test_dashboard_hongbao_draw(ctx):
    nt = NextToto(draw_datetime=datetime(2026, 10, 5, 18, 30, tzinfo=SG), jackpot_estimate=12_000_000.0,
                  draw_type="hongbao", draw_type_hint="hongbao", raw_text="Hongbao draw")
    c = rebuilt(ctx, nt)
    _, fm, body = _check_note(notes.dashboard_note(c))
    table = _kv(_section(body, "Next draw"))
    assert table["Draw type"] == "Hongbao" and table["Estimated jackpot"] == "$12,000,000"
    assert table["Jackpot rollovers so far"] == "1"  # its own jackpot: no cascade countdown
    prize = _section(body, "The next big prize")
    assert "The next draw is a Hongbao draw with a jackpot of about $12,000,000." in prize
    assert "Announced special draws: Mon 5 Oct 2026 (Hongbao)." in prize
    assert fm["next_big_prize"] == 12_000_000 and fm["next_draw_type"] == "hongbao"


def test_dashboard_lists_announced_special_draws():
    c = make_context(with_tickets=False, announced=[(date(2026, 10, 15), "special"), (date(2026, 9, 1), "special"),
                                                    (date(2026, 10, 8), "normal")])
    prize = _section(notes.dashboard_note(c)[2], "The next big prize")
    assert "Announced special draws: Thu 15 Oct 2026 (Special)." in prize  # past and normal draws left out


def test_held_draw_gets_no_buy_signal_in_the_dashboard(ctx):
    # Mon 5 Oct at 9pm: the 6.30pm draw 4124 is held, its result is not stored yet.
    late = variant(ctx, now=datetime(2026, 10, 5, 21, 0, tzinfo=SG))
    assert report.next_draw(late).held
    _, fm, body = _check_note(notes.dashboard_note(late))
    table = _kv(_section(body, "Next draw"))
    assert table["Draw"] == "4124"
    assert table["Date and time"] == "Mon 5 Oct 2026, 6.30pm (draw held, result not out yet)"
    assert table["Estimated jackpot"] == "sales closed"
    assert _section(body, "Buy signal") == "No buy signal: this draw was held and its sales are closed."
    assert fm["buy_signal"] is None and fm["return_per_dollar"] is None


# Ledger


def test_ledger_note_lists_every_ticket_and_bad_line(ctx):
    rel, fm, body = _check_note(notes.ledger_note(ctx))
    assert rel == "Ledger.md"
    assert fm == {"tags": ["huatbot", "ledger"], "updated": "2026-10-01T19:30:00+08:00", "tickets": 4,
                  "spent": 10.0, "won": 10.0, "net": 0.0, "pending_cost": 8.0}
    assert _table(_section(body, "Totals")) == [["Tickets", "Spent", "Won", "Net", "Waiting", "Waiting cost"],
                                                ["4", "$10", "$10", "$0", "2", "$8"]]
    rows = _table(_section(body, "Tickets"))
    assert rows[0] == ["Draw date", "Draw", "Numbers", "Bet type", "Cost", "Status", "Result", "Won"]
    assert [[r[0], r[1], r[3], r[4], r[5], r[6], r[7]] for r in rows[1:]] == [
        ["Mon 5 Oct 2026", "", "Ordinary", "$1", "Waiting for the draw", "", ""],
        ["Mon 5 Oct 2026", "", "System 7", "$7", "Waiting for the draw", "", ""],
        ["Thu 1 Oct 2026", "4123", "Ordinary", "$1", "Checked", "Group 7 x1", "$10"],
        ["Mon 28 Sep 2026", "4122", "Ordinary", "$1", "Checked", "No prize", "$0"],
    ]
    assert sorted(r[2] for r in rows[1:]) == sorted(ctx.ledger["numbers"])

    bad = _section(body, "Lines that could not be read").split("\n\n")
    assert len(ctx.bad_ticket_lines) == 2
    assert bad[0].splitlines() == [f"* Line {t.line_no}: `{t.source}` {report.sentence(t.error)}"
                                   for t in ctx.bad_ticket_lines]
    # An old 4D line gets a clear reason instead of being skipped.
    assert bad[0].splitlines()[1].endswith("` 4D is not tracked any more, this bot follows TOTO only.")
    assert bad[1] == "Fix these lines in [[Tickets]] and they will be added on the next run."
    assert "[[Dashboard]]" in body and "[[Tickets]]" in body


def test_ledger_note_keeps_raw_lines_as_code(ctx):
    odd = Ticket(game="TOTO", draw_date=None, numbers="", bet_type="", cost=0.0, line_no=12,
                 source="| TOTO | 2026-10-03 | 1-2-3 | `odd` |", error="numbers must be 1 to 49 - got 1-2-3")
    _, _, body = _check_note(notes.ledger_note(variant(ctx, bad_ticket_lines=[odd])))
    line = _section(body, "Lines that could not be read").splitlines()[0]
    assert line.startswith("* Line 12: `` | TOTO | 2026-10-03 | 1-2-3 | `odd` | `` Numbers must be 1 to 49, got ")


def test_ledger_note_without_tickets(ctx):
    empty = empty_ledger()
    bare = variant(ctx, ledger=empty, ledger_totals=ledger_totals(empty), settled_this_run=[], bad_ticket_lines=[])
    _, fm, body = _check_note(notes.ledger_note(bare))
    assert _section(body, "Tickets") == "No tickets yet. Add the tickets you buy to [[Tickets]]."
    assert _section(body, "Lines that could not be read") == "None, every line was read."
    assert fm["tickets"] == 0 and fm["spent"] == 0.0
    assert "## Tickets" in notes.ledger_note(variant(ctx, ledger=None))[2]  # no ledger frame at all


def test_ledger_note_keeps_old_4d_rows(ctx):
    """Rows of 4D tickets from an older version: a checked one is real history and still counts, an
    unchecked one is not counted and says why."""
    base = {c: "" for c in LEDGER_COLUMNS}
    rows = [
        {**base, "ticket_id": "old4dwin", "game": "4D", "draw_date": "2026-09-30", "draw_number": 5432,
         "numbers": "1234", "bet_type": "Big", "cost": 1.0, "units": 1.0, "status": "settled",
         "result": "Starter", "winnings": 250.0},
        {**base, "ticket_id": "old4dwait", "game": "4D", "draw_date": "2026-10-03", "numbers": "5678",
         "bet_type": "Small", "cost": 2.0, "units": 2.0, "status": "pending", "winnings": 0.0},
        {**base, "ticket_id": "totowait", "game": "TOTO", "draw_date": "2026-10-05", "numbers": "3 11 19 27 38 45",
         "bet_type": "Ordinary", "cost": 1.0, "units": 1.0, "status": "pending", "winnings": 0.0},
    ]
    ledger, settled = tickets.settle_ledger(normalise_ledger(pd.DataFrame(rows)), ctx.toto, ctx.rules, ctx.now)
    assert settled == []
    totals = ledger_totals(ledger)
    c = variant(ctx, ledger=ledger, ledger_totals=totals, settled_this_run=[], bad_ticket_lines=[])
    _, fm, body = _check_note(notes.ledger_note(c))
    assert fm["tickets"] == 2 and fm["spent"] == 2.0 and fm["won"] == 250.0 and fm["net"] == 248.0
    table = {r[2]: r for r in _table(_section(body, "Tickets"))[1:]}
    assert table["1234"][5:] == ["Checked", "Starter", "$250"]
    assert table["5678"][5:] == ["Not counted", tickets.FOURD_NOT_TRACKED, ""]
    assert table["3 11 19 27 38 45"][5] == "Waiting for the draw"


# Report note


def test_report_note(ctx, report_md):
    rel, fm, body = _check_note(notes.report_note(ctx, report_md))
    assert rel == f"Reports/{REPORT_NAME}.md"
    assert fm == {
        "tags": ["huatbot", "report"],
        "date": "2026-10-01",
        "created": "2026-10-01T19:30:00+08:00",
        "new_draws": 2,
        "latest_draw": 4123,
        "buy_signal": ctx.buy_signal.label,
        "next_jackpot": 2_100_000.0,
        "next_big_prize": round(ctx.outlook.biggest.jackpot),
    }
    lines = body.splitlines()
    assert lines[0] == "# Huat Bot report, Thu 1 Oct 2026, 7.30pm"
    assert lines[2] == f"Back to [[Dashboard]]. Activity log: [[{LOG_NAME}]]."
    assert "\n".join(lines[3:]).strip("\n") == "\n".join(report_md.strip("\n").split("\n")[1:]).strip("\n")
    assert [ln for ln in lines if ln.startswith("## ")] == list(report.SECTION_HEADINGS)


def test_report_note_without_a_title_and_its_path(ctx):
    _, fm, body = notes.report_note(variant(ctx, new_draws=[]), "Plain text.")
    assert body == f"Back to [[Dashboard]]. Activity log: [[{LOG_NAME}]].\n\nPlain text."
    assert fm["new_draws"] == 0
    # The note name is in Singapore time, whatever zone the run time is in.
    assert notes.report_note_path(datetime(2026, 10, 2, 11, 30, tzinfo=timezone.utc)) == \
        "Reports/2026-10-02 1930 Report.md"


# Writing to a vault


def _md_files(vault):
    return sorted(p for p in vault.base.rglob("*.md"))


def _log_rows(vault, event):
    text = vault.read_text(f"Logs/{LOG_NAME}.md") or ""
    return [ln for ln in text.splitlines() if f"| {event} |" in ln]


def test_write_all_creates_notes_and_logs(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    written = notes.write_all(vault, ctx, report_md)
    # Draw notes oldest first, then the report, the ledger and the dashboard.
    assert written == [DRAW_4122, DRAW_4123, f"Reports/{REPORT_NAME}.md", "Ledger.md", "Dashboard.md"]
    for rel in written:
        text = vault.read_text(rel)
        assert _yaml_block(text)["tags"][0] == "huatbot"
        assert vault.read_note(rel)[0]
    assert len(_log_rows(vault, "NOTE")) == len(written)
    assert any("Created [[Dashboard]]" in row for row in _log_rows(vault, "NOTE"))
    for path in _md_files(vault):  # notes and the activity log
        assert not has_prose_dashes(path.read_text(encoding="utf-8")), path
    # No suggestion notes, no 4D notes or data, nothing outside the bot folder.
    assert not vault.path("Suggestions").exists() and not vault.path("Draws/4D").exists()
    assert not (vault.data_dir / "fourd.csv").exists() and not (vault.data_dir / "backtest_cache.json").exists()
    assert {p.relative_to(vault.root).parts[0] for p in (tmp_path / "vault").rglob("*") if p.is_file()} == {"Huat Bot"}
    assert {p.relative_to(vault.base).parts[0] for p in _md_files(vault)} == {
        "Dashboard.md", "Ledger.md", "Reports", "Draws", "Logs"}


def test_write_all_twice_changes_nothing(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    first = notes.write_all(vault, ctx, report_md)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in _md_files(vault)}
    again = notes.write_all(vault, ctx, report_md)
    assert first and again == []
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in _md_files(vault)}
    assert after == before  # not even the activity log changed
    for rel in first:
        rel_path, fm, body = next(s for s in notes._all_notes(ctx, report_md) if s[0] == rel)
        assert vault.write_note(rel_path, body, fm) is False


def test_write_all_updates_changed_notes_only(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report_md)
    later = variant(ctx, commentary="Second run.", new_draws=[])
    written = notes.write_all(vault, later, report.full_report(later))
    assert written == [f"Reports/{REPORT_NAME}.md"]
    assert any("Updated [[Reports/2026-10-01 1930 Report\\|2026-10-01 1930 Report]]" in row
               for row in _log_rows(vault, "NOTE"))


def test_write_all_caps_draw_notes_at_backfill(tmp_path, ctx, report_md):
    new = [int(n) for n in ctx.toto["draw_number"].iloc[-10:]]
    capped = variant(ctx, settings=Settings(draw_notes_backfill=3), new_draws=new)
    vault = Vault(tmp_path / "vault")
    written = notes.write_all(vault, capped, report_md)
    assert [p.rsplit(" ", 1)[-1] for p in written if p.startswith("Draws/")] == ["4121.md", "4122.md", "4123.md"]
    assert len(list(vault.path("Draws/TOTO").glob("*.md"))) == 3


def test_write_all_without_backfill_still_writes_the_newest_draw_note(tmp_path, ctx, report_md):
    # The Dashboard and the report link to the newest draw note, so it is always written.
    vault = Vault(tmp_path / "vault")
    written = notes.write_all(vault, variant(ctx, settings=Settings(draw_notes_backfill=0)), report_md)
    assert [p for p in written if p.startswith("Draws/")] == [DRAW_4123]
    assert "Dashboard.md" in written
    assert DRAW_4123_LINK.replace("|", "\\|") not in vault.read_text("Dashboard.md")  # not in a table
    assert DRAW_4123_LINK in vault.read_text("Dashboard.md") and vault.exists(DRAW_4123)


def test_write_all_logs_an_error_and_carries_on(tmp_path, ctx, report_md, monkeypatch):
    vault = Vault(tmp_path / "vault")
    real = vault.write_note

    def flaky(rel, body, frontmatter=None):
        if rel == "Ledger.md":
            raise PermissionError("read only share")
        return real(rel, body, frontmatter)

    monkeypatch.setattr(vault, "write_note", flaky)
    written = notes.write_all(vault, ctx, report_md)
    assert "Ledger.md" not in written and "Dashboard.md" in written
    errors = _log_rows(vault, "ERROR")
    assert len(errors) == 1 and "Could not write [[Ledger]] (PermissionError)" in errors[0]


def test_write_all_recreates_a_missing_newest_draw_note(tmp_path, ctx, report_md):
    # A run stopped after the draw was saved but before its note was written: the next run
    # finds no new draw, yet the Dashboard links to that note, so it is written now.
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report_md)
    vault.path(DRAW_4123).unlink()
    vault.path(DRAW_4122).unlink()
    written = notes.write_all(vault, variant(ctx, new_draws=[]), report_md)
    assert DRAW_4123 in written and vault.exists(DRAW_4123)
    assert DRAW_4122 not in written and not vault.exists(DRAW_4122)  # only the newest one is linked


def test_write_all_skips_notes_whose_only_change_is_the_time_stamp(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    first = variant(ctx, new_draws=[])  # a rerun with no new draw: the same report apart from its time
    notes.write_all(vault, first, report.full_report(first))
    keep = {rel: vault.read_text(rel) for rel in ("Ledger.md", DRAW_4123)}
    later = variant(first, now=ctx.now + pd.Timedelta(hours=1))
    logged = len(_log_rows(vault, "NOTE"))
    written = notes.write_all(vault, later, report.full_report(later))
    # The report differs only in its time, so the 7.30pm report stays the newest one.
    assert written == ["Dashboard.md"]
    assert not vault.exists("Reports/2026-10-01 2030 Report.md")
    assert f"Newest report: [[{REPORT_NAME}]]" in vault.read_text("Dashboard.md")
    for rel, text in keep.items():
        assert vault.read_text(rel) == text, rel
    assert len(_log_rows(vault, "NOTE")) == logged + 1  # no "Updated [[Ledger]]" row

    # A real change is still written, with its new time stamp.
    changed = variant(later, ledger_totals={**ctx.ledger_totals, "spent": 99.0})
    assert "Ledger.md" in notes.write_all(vault, changed, report.full_report(changed))
    assert vault.read_note("Ledger.md")[0]["updated"].startswith("2026-10-01T20:30")


def test_rerun_with_the_same_report_keeps_the_earlier_report_note(tmp_path, ctx):
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report.full_report(ctx))
    retry = variant(ctx, now=ctx.now + pd.Timedelta(minutes=10), new_draws=[])
    retry_md = report.full_report(retry)
    # Differs from the first report: "No new draws in this run." So write it once at 7.40pm...
    notes.write_all(vault, retry, retry_md)
    # ...and a second retry at 7.50pm with the same content adds no report note.
    again = variant(retry, now=ctx.now + pd.Timedelta(minutes=20))
    again_md = report.full_report(again)
    written = notes.write_all(vault, again, again_md)
    assert not any(p.startswith("Reports/") for p in written)
    assert sorted(p.name for p in vault.path("Reports").glob("*.md")) == [
        f"{REPORT_NAME}.md", "2026-10-01 1940 Report.md"]
    assert notes.report_rel(vault, again, again_md) == "Reports/2026-10-01 1940 Report.md"
    assert "Newest report: [[2026-10-01 1940 Report]]" in vault.read_text("Dashboard.md")

    # A report with different content is a new note, and the Dashboard links to it.
    changed = variant(again, commentary="Something new.")
    changed_md = report.full_report(changed)
    assert "Reports/2026-10-01 1950 Report.md" in notes.write_all(vault, changed, changed_md)
    assert notes.report_rel(vault, changed, changed_md) == "Reports/2026-10-01 1950 Report.md"
    assert "Newest report: [[2026-10-01 1950 Report]]" in vault.read_text("Dashboard.md")

    # A report from another day is never reused.
    next_day = variant(again, now=ctx.now + pd.Timedelta(days=1))
    assert "Reports/2026-10-02 1930 Report.md" in notes.write_all(vault, next_day, report.full_report(next_day))


def test_write_all_leaves_old_4d_and_suggestion_files_alone(tmp_path, ctx, report_md):
    vault = Vault(tmp_path / "vault")
    old = {
        "Suggestions/2026-10-05 TOTO 4124.md": "---\ntags: [huatbot, suggestion, toto]\n---\n# Old suggestions\n",
        "Draws/4D/2026-09-30 4D 5432.md": "---\ntags: [huatbot, 4d, draw]\n---\n# 4D draw 5432\n",
        "Data/fourd.csv": "draw_number,draw_date\n5432,2026-09-30\n",
        "Data/backtest_cache.json": "{}\n",
    }
    for rel, text in old.items():
        path = vault.path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    written = notes.write_all(vault, ctx, report_md)
    assert not any(p.startswith(("Suggestions/", "Draws/4D/", "Data/")) for p in written)
    for rel, text in old.items():
        assert vault.path(rel).read_text(encoding="utf-8") == text, rel
    log_text = vault.read_text(f"Logs/{LOG_NAME}.md")
    assert "Suggestions" not in log_text and "4D" not in log_text
    assert "Suggestions/" not in vault.read_text("Dashboard.md")


WIKILINK = re.compile(r"\[\[([^\]|\\]+)(?:\\?\|[^\]]*)?\]\]")
USER_NOTES = {"Tickets", "Settings"}  # starter notes the runner creates for the user


def test_every_link_in_the_notes_leads_to_a_note(tmp_path, ctx):
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report.full_report(ctx))
    # Mon 5 Oct: draw 4124 is held and stored, its note is written and linked.
    held = ctx.toto.iloc[[-1]].copy()
    held["draw_number"] = 4124
    held["draw_date"] = pd.Timestamp("2026-10-05")
    later = rebuilt(variant(ctx, toto=pd.concat([ctx.toto, held], ignore_index=True), new_draws=[4124],
                            now=datetime(2026, 10, 5, 19, 30, tzinfo=SG)), None)
    written = notes.write_all(vault, later, report.full_report(later))
    assert "Draws/TOTO/2026-10-05 TOTO 4124.md" in written
    dashboard = vault.read_text("Dashboard.md")
    assert "[[Draws/TOTO/2026-10-05 TOTO 4124|2026-10-05 TOTO 4124]]" in dashboard
    assert vault.read_note("Dashboard.md")[0]["latest_draw"] == 4124

    stems = {p.stem for p in _md_files(vault)}
    checked = 0
    for path in _md_files(vault):
        for target in WIKILINK.findall(path.read_text(encoding="utf-8")):
            checked += 1
            assert not target.startswith(("Suggestions/", "Draws/4D")), (path.name, target)
            if "/" in target:
                assert vault.exists(f"{target}.md"), (path.name, target)
            else:
                assert target in stems or target in USER_NOTES, (path.name, target)
    assert checked > 10


# Guard: the removed features never come back in user facing text


FORBIDDEN = {
    "4D": re.compile(r"\b4\s?D\b|\bfour\s?d\b", re.I),
    "suggest": re.compile(r"suggest", re.I),
    "pick": re.compile(r"\bpick", re.I),
    "budget": re.compile(r"budget", re.I),
    "System 7 offer": re.compile(r"System 7 offer", re.I),
    "backtest": re.compile(r"back\s?test", re.I),
    "hot": re.compile(r"\bhot\b", re.I),
    "cold": re.compile(r"\bcold\b", re.I),
    "overdue": re.compile(r"overdue", re.I),
}
# The only allowed mention: the reason an old 4D ticket line or ledger row is not counted.
ALLOWED = [tickets.FOURD_NOT_TRACKED]
_CODE = re.compile(r"``.*?``|`[^`\n]*`")  # the user's own raw ticket lines


def forbidden_mentions(text: str) -> list[str]:
    text = _CODE.sub(" ", text)
    for allowed in ALLOWED:
        text = text.replace(allowed, " ")
    return [f"{name}: {m.group(0)!r} in ...{text[max(0, m.start() - 40):m.end() + 40]!r}..."
            for name, pattern in FORBIDDEN.items() for m in pattern.finditer(text)]


def _guard_contexts(ctx) -> dict:
    hongbao = NextToto(draw_datetime=datetime(2026, 10, 5, 18, 30, tzinfo=SG), jackpot_estimate=12_000_000.0,
                       draw_type="hongbao", draw_type_hint="hongbao", raw_text="Hongbao draw")
    return {
        "default": ctx,
        "no_tickets_no_new_draws": make_context(with_tickets=False, new_draws=0),
        "many_tickets": make_context(extra_tickets=60),
        "cascade": cascade_next_toto(ctx),
        "hongbao": rebuilt(ctx, hongbao),
        "worked_out_jackpot": unknown_jackpot(ctx),
        "no_next_draw": variant(ctx, next_toto=None, buy_signal=None, outlook=None),
        "held": variant(ctx, now=datetime(2026, 10, 5, 21, 0, tzinfo=SG)),
        "commentary": variant(ctx, commentary="The jackpot of $2,100,000 rolls on.",
                              warnings=["TOTO: the site was slow"]),
    }


def test_no_user_facing_text_mentions_removed_features(tmp_path, ctx):
    texts: dict[str, str] = {}
    for name, c in _guard_contexts(ctx).items():
        messages = report.telegram_messages(c)
        assert len(messages) == 2, name
        for i, msg in enumerate(messages, 1):
            texts[f"{name} message {i}"] = msg
        md = report.full_report(c)
        texts[f"{name} report"] = md
        for rel, fm, body in notes._all_notes(c, md):
            texts[f"{name} {rel}"] = render_note(body, fm)
    vault = Vault(tmp_path / "vault")
    notes.write_all(vault, ctx, report.full_report(ctx))
    for path in vault.base.rglob("*"):
        if path.is_file():
            texts[f"vault {path.relative_to(vault.base)}"] = path.read_text(encoding="utf-8")

    problems = {where: found for where, text in texts.items() if (found := forbidden_mentions(text))}
    assert problems == {}
    # The guard does see the one allowed mention, and would catch a real one.
    assert tickets.FOURD_NOT_TRACKED in texts["default Ledger.md"]
    assert forbidden_mentions("Suggested numbers for 4D: hot picks within your budget")
    assert forbidden_mentions("System 7 offer") and forbidden_mentions("Backtest: overdue and cold numbers")
    assert not forbidden_mentions("photo, shot, scold, 4th draw, `| 4D | 1234 |`, " + tickets.FOURD_NOT_TRACKED)


def test_note_properties_carry_no_removed_features(ctx, report_md):
    keys = {k for _, fm, _ in notes._all_notes(ctx, report_md) for k in fm}
    assert not any(forbidden_mentions(k.replace("_", " ")) for k in keys)
    assert {"next_big_prize", "chance_jackpot_won", "next_big_prize_date", "next_jackpot"} <= keys
    assert not keys & {"game", "games", "next_4d_draw", "budget", "sets", "total_cost"}


def test_dataclass_copies_keep_the_module_context_untouched(ctx):
    # variant() and rebuilt() never change the shared module context.
    before = dataclasses.asdict(ctx.outlook)
    rebuilt(ctx, None)
    cascade_next_toto(ctx)
    assert dataclasses.asdict(ctx.outlook) == before and ctx.new_draws == [4122, 4123]
