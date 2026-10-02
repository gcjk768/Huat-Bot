"""fetch.py against a fake site built from synthetic history and the static fixtures."""
from __future__ import annotations

import inspect
import math
import re
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import fetch as F
from huatbot.http import FetchError
from huatbot.models import NextToto
from huatbot.store import empty_toto
from huatbot.synth import SG, synth_toto
from tests import htmlgen as H
from tests.conftest import FIXTURES

NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)
TOTO_RESULT_COLS = (
    ["draw_number", "draw_date"] + [f"n{i}" for i in range(1, 7)] + ["additional", "jackpot"]
    + [f"g{g}_{k}" for g in range(1, 8) for k in ("share", "winners")]
)
DASHES = re.compile("[-–—]")


def assert_no_dashes(lines):
    for line in lines:
        assert not DASHES.search(line), line


def assert_same_results(got: pd.DataFrame, expected: pd.DataFrame, cols):
    got = got.set_index("draw_number")
    expected = expected.set_index("draw_number")
    assert list(got.index) == list(expected.index)
    for col in cols:
        if col == "draw_number":
            continue
        pd.testing.assert_series_equal(got[col], expected[col], check_names=False, check_dtype=False)


def result_urls(fetcher: H.FakeFetcher) -> list[str]:
    return [u for u in fetcher.requested if "toto_results.aspx" in u]


def static_site() -> H.FakeFetcher:
    """The handwritten fixtures mapped to their real URLs."""
    read = lambda name: (FIXTURES / name).read_text(encoding="utf-8")  # noqa: E731
    return H.FakeFetcher({
        C.TOTO_DRAW_LIST_URL: read("toto_draw_list.html"),
        F.toto_result_url(4123): read("toto_result_with_winner.html"),
        F.toto_result_url(4122): read("toto_result_no_winner.html"),
        C.TOTO_NEXT_DRAW_URL: read("toto_next_draw.html"),
        C.TOTO_CASCADE_LIST_URL: read("toto_cascade_list.html"),
        C.TOTO_HONGBAO_LIST_URL: H.draw_type_list_html([]),
        C.TOTO_SPECIAL_LIST_URL: H.draw_type_list_html([]),
        C.TOTO_PRIZE_RULES_URL: read("toto_prize_structure.html"),
    })


# URLs


def test_result_urls():
    assert F.toto_result_url(4123) == (
        "https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl=RHJhd051bWJlcj00MTIz")


# draw types


def test_tag_draw_types_reproduces_synthetic_history(toto_df):
    """With only the Hongbao list, inference recovers every cascade draw of the synthetic history."""
    assert (toto_df["draw_type"] == "cascade").sum() > 5
    plain = toto_df.assign(draw_type="normal")
    hongbao = set(toto_df.loc[toto_df["draw_type"] == "hongbao", "draw_number"])
    tagged = F.tag_draw_types(plain, set(), hongbao, set())
    assert list(tagged["draw_type"]) == list(toto_df["draw_type"])
    assert (plain["draw_type"] == "normal").all()  # input not modified


def _streak_df(winners: list[int], start: int = 100) -> pd.DataFrame:
    df = synth_toto(n_draws=len(winners), start_draw=start, hongbao_every=0)
    df["g1_winners"] = winners
    df["draw_type"] = "normal"
    return df


def test_tag_draw_types_inference_rules():
    # 6 draws without a winner after a winner: the 4th one (index 4) is the cascade draw,
    # and the count restarts after it
    df = _streak_df([1, 0, 0, 0, 0, 0, 0, 0, 0])
    types = list(F.tag_draw_types(df, None, None, None)["draw_type"])
    assert types == ["normal", "normal", "normal", "normal", "cascade", "normal", "normal", "normal", "cascade"]


def test_tag_draw_types_needs_consecutive_draws():
    df = _streak_df([1, 0, 0, 0, 0])
    df = df[df["draw_number"] != 102]  # a gap before draw 104
    out = F.tag_draw_types(df, None, None, None)
    types = dict(zip(out["draw_number"], out["draw_type"]))
    assert types[104] == "normal"


def test_tag_draw_types_lists_and_existing_tags():
    df = _streak_df([1, 1, 1, 1, 1, 1])
    df.loc[df["draw_number"] == 101, "draw_type"] = "hongbao"
    df.loc[df["draw_number"] == 102, "draw_type"] = "cascade"
    df.loc[df["draw_number"] == 103, "draw_type"] = "special"
    out = F.tag_draw_types(df, cascade={104, 105}, hongbao={105}, special={100})
    got = dict(zip(out["draw_number"], out["draw_type"]))
    assert got == {100: "special", 101: "hongbao", 102: "cascade", 103: "special", 104: "cascade", 105: "hongbao"}


def test_tag_draw_types_hongbao_resets_the_streak():
    df = _streak_df([1, 0, 0, 0, 0])
    out = F.tag_draw_types(df, None, {102}, None)
    assert list(out["draw_type"]) == ["normal", "normal", "hongbao", "normal", "normal"]


def test_tag_draw_types_empty_and_unknown_tags():
    assert F.tag_draw_types(empty_toto(), {1}, set(), set()).empty
    df = _streak_df([1, 1])
    df["draw_type"] = ["Weird", None]
    assert list(F.tag_draw_types(df, None, None, None)["draw_type"]) == ["normal", "normal"]


# update_toto


def test_update_toto_fetches_only_missing_draws(toto_df):
    site = H.fake_site(toto_df)
    local = toto_df.iloc[:590]
    df, res = F.update_toto(site, local, start_draw=int(toto_df["draw_number"].min()), now=NOW)

    expected_new = list(toto_df["draw_number"].iloc[590:])
    assert res.new_draws == expected_new
    assert sorted(result_urls(site)) == sorted(F.toto_result_url(n) for n in expected_new)
    assert res.failed_draws == []
    assert res.verified is True
    assert res.latest_on_site == res.latest_in_csv == int(toto_df["draw_number"].max())
    assert res.game == "toto"
    assert_same_results(df, toto_df, TOTO_RESULT_COLS + ["draw_type"])
    new_rows = df[df["draw_number"].isin(expected_new)]
    assert (new_rows["fetched_at"] == NOW.isoformat(timespec="seconds")).all()
    assert (df[~df["draw_number"].isin(expected_new)]["fetched_at"] == "synthetic").all()
    assert_no_dashes(res.messages)
    assert any("10 new draws added" in m for m in res.messages)
    assert any("matches the site" in m for m in res.messages)


def test_update_toto_from_empty_with_start_draw(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    df, res = F.update_toto(site, empty_toto(), start_draw=latest - 19, now=NOW)
    assert len(df) == 20 and res.new_draws == list(range(latest - 19, latest + 1))
    assert_same_results(df, toto_df[toto_df["draw_number"] >= latest - 19], TOTO_RESULT_COLS)
    assert res.verified


def test_update_toto_nothing_missing(toto_df):
    site = H.fake_site(toto_df)
    df, res = F.update_toto(site, toto_df, start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert res.new_draws == [] and result_urls(site) == []
    assert res.verified
    assert any("nothing new to fetch" in m for m in res.messages)
    assert len(df) == len(toto_df)


def test_update_toto_respects_skip(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    df, res = F.update_toto(site, toto_df.iloc[:-5], start_draw=int(toto_df["draw_number"].min()),
                            skip={latest - 2, latest - 3}, now=NOW)
    assert res.new_draws == [latest - 4, latest - 1, latest]
    assert F.toto_result_url(latest - 2) not in site.requested
    assert latest - 2 not in set(df["draw_number"])
    assert res.verified  # the newest draw is still there
    assert any("skip list" in m for m in res.messages)


def test_update_toto_rejects_page_for_the_wrong_draw(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    site.pages[F.toto_result_url(latest)] = H.toto_result_html(toto_df.iloc[-2])  # shows draw latest - 1
    df, res = F.update_toto(site, toto_df.iloc[:-3], start_draw=latest - 10, now=NOW)
    assert res.failed_draws == [latest]
    assert latest not in set(df["draw_number"])
    assert res.new_draws == [latest - 2, latest - 1]
    assert res.verified is False
    assert any(f"draw {latest} was not added: page showed draw {latest - 1} instead" in m for m in res.messages)
    assert any("not fully up to date" in m for m in res.messages)
    assert_no_dashes(res.messages)


def test_update_toto_missing_and_broken_pages_are_reported_not_raised(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    del site.pages[F.toto_result_url(latest - 1)]
    site.pages[F.toto_result_url(latest - 2)] = "<html><body>Service temporarily unavailable</body></html>"
    site.pages[F.toto_result_url(latest - 3)] = FetchError("the site returned HTTP 503 after 6 attempts")
    df, res = F.update_toto(site, toto_df.iloc[:-4], start_draw=latest - 10, now=NOW)
    assert res.failed_draws == [latest - 3, latest - 2, latest - 1]
    assert res.new_draws == [latest]
    assert res.verified  # the newest row is the latest draw on the site
    joined = " ".join(res.messages)
    assert "page not found (HTTP 404)" in joined
    assert "HTTP 503" in joined
    assert "could not be read" in joined
    assert_no_dashes(res.messages)


def test_update_toto_summarises_many_failures(toto_df):
    site = H.FakeFetcher({C.TOTO_DRAW_LIST_URL: H.draw_list_html(toto_df.tail(10))})
    latest = int(toto_df["draw_number"].max())
    df, res = F.update_toto(site, empty_toto(), start_draw=latest - 29, now=NOW)
    assert len(res.failed_draws) == 30 and df.empty
    failures = [m for m in res.messages if "was not added" in m]
    assert len(failures) == F.MAX_LISTED_FAILURES
    assert any("20 more draws could not be added" in m for m in res.messages)
    assert res.verified is False and res.latest_in_csv is None
    assert sum("draw list could not be used" in m for m in res.messages) == 3


def test_update_toto_messages_singular_and_plural(toto_df):
    latest = int(toto_df["draw_number"].max())
    site = H.fake_site(toto_df)
    _, res = F.update_toto(site, toto_df.iloc[:-2], start_draw=int(toto_df["draw_number"].min()),
                           skip={latest - 1}, now=NOW)
    assert "TOTO: 1 new draw added (%d)." % latest in res.messages
    assert "TOTO: 1 draw left out because it is on the skip list (%d)." % (latest - 1) in res.messages

    # 11 failures: 10 named, then "1 more draw" (not "1 more draws")
    site = H.FakeFetcher({C.TOTO_DRAW_LIST_URL: H.draw_list_html(toto_df.tail(3))})
    _, res = F.update_toto(site, empty_toto(), start_draw=latest - 10, now=NOW)
    assert f"TOTO: 1 more draw could not be added ({latest})." in res.messages
    assert_no_dashes(res.messages)


def test_update_toto_accepts_none_and_a_draw_list_without_dates(toto_df):
    latest = int(toto_df["draw_number"].max())
    site = H.fake_site(toto_df)
    site.pages[C.TOTO_DRAW_LIST_URL] = H.draw_list_html(list(toto_df["draw_number"].tail(5)))  # no dates
    df, res = F.update_toto(site, None, start_draw=latest - 2, now=NOW)
    assert res.new_draws == [latest - 2, latest - 1, latest] and res.verified
    assert f"TOTO: the latest draw on the site is {latest}." in res.messages
    assert res.game == "toto" and res.repaired_draws == []
    assert not any("4d" in u.lower() or "fourd" in u for u in site.requested)


def test_update_toto_draw_list_failure_raises(toto_df):
    with pytest.raises(FetchError):
        F.update_toto(H.FakeFetcher({}), toto_df, start_draw=1, now=NOW)
    broken = H.FakeFetcher({C.TOTO_DRAW_LIST_URL: "<html><body>Maintenance</body></html>"})
    with pytest.raises(FetchError):
        F.update_toto(broken, toto_df, start_draw=1, now=NOW)


def test_update_toto_type_lists_failing_is_only_a_warning(toto_df):
    site = H.fake_site(toto_df)
    for url in (C.TOTO_CASCADE_LIST_URL, C.TOTO_HONGBAO_LIST_URL, C.TOTO_SPECIAL_LIST_URL):
        del site.pages[url]
    df, res = F.update_toto(site, toto_df.iloc[:-2], start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert res.new_draws == list(toto_df["draw_number"].iloc[-2:])
    # existing Hongbao tags are kept and cascades are inferred, so the types still match
    assert list(df["draw_type"]) == list(toto_df["draw_type"])
    assert sum("draw list could not be used" in m for m in res.messages) == 3
    assert_no_dashes(res.messages)


def test_update_toto_date_mismatch_is_not_verified(toto_df):
    site = H.fake_site(toto_df)
    pairs = list(zip(toto_df["draw_number"].tail(5), toto_df["draw_date"].tail(5)))
    pairs[-1] = (pairs[-1][0], pairs[-1][1] + pd.Timedelta(days=1))
    site.pages[C.TOTO_DRAW_LIST_URL] = H.draw_list_html(pairs)
    _, res = F.update_toto(site, toto_df, start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert res.verified is False
    assert any("please check it" in m for m in res.messages)


def test_update_toto_with_static_fixtures():
    site = static_site()
    df, res = F.update_toto(site, empty_toto(), start_draw=4122, now=NOW)
    assert res.new_draws == [4122, 4123]
    assert res.verified and res.latest_on_site == 4123
    row = df.set_index("draw_number").loc[4123]
    assert [row[f"n{i}"] for i in range(1, 7)] == [3, 11, 19, 27, 38, 45]
    assert row["g1_winners"] == 2 and row["jackpot"] == 2371852.0
    assert row["draw_date"] == pd.Timestamp("2026-10-01")
    assert math.isnan(df.set_index("draw_number").loc[4122, "g1_share"])
    assert set(df["draw_type"]) == {"normal"}


# next draw


def test_fetch_next_draw_from_page(toto_df):
    dt = datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    nt = NextToto(draw_datetime=dt, jackpot_estimate=4_500_000.0, draw_type="normal", draw_type_hint="hongbao")
    site = H.fake_site(toto_df, next_toto=nt)
    t = F.fetch_next_draw(site, toto_df)
    assert isinstance(t, NextToto)
    assert t.draw_datetime == dt and t.jackpot_estimate == 4_500_000.0
    assert t.draw_type == "hongbao" and t.draw_type_hint == "hongbao"
    assert "Next Jackpot" in t.raw_text and len(t.raw_text) <= F.RAW_TEXT_LIMIT
    assert site.requested == [C.TOTO_NEXT_DRAW_URL]  # the TOTO page only


def _streak_df_ending(winners: list[int], end: date) -> pd.DataFrame:
    """_streak_df whose newest draw is on ``end`` (a TOTO day)."""
    df = _streak_df(winners)
    df["draw_date"] = pd.to_datetime(start_dates(end, len(winners)))
    return df


def start_dates(end: date, n: int) -> list[date]:
    """The last ``n`` regular TOTO draw days up to ``end``, oldest first."""
    days, d = [], end
    while len(days) < n:
        if d.weekday() in C.TOTO_WEEKDAYS:
            days.append(d)
        d -= timedelta(days=1)
    return days[::-1]


def test_fetch_next_draw_predicts_cascade_from_history():
    # The stored history ends Thu 1 Oct, the next draw page shows Mon 5 Oct.
    df = _streak_df_ending([1, 0, 0, 0], date(2026, 10, 1))  # 3 snowballs in a row: the next draw is the 4th
    site = static_site()
    t = F.fetch_next_draw(site, df)
    assert t.draw_type_hint is None and t.draw_type == "cascade"
    assert t.draw_datetime == datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    assert F.fetch_next_draw(site, _streak_df_ending([1, 0, 0], date(2026, 10, 1))).draw_type == "normal"
    assert F.fetch_next_draw(site, empty_toto()).draw_type == "normal"
    assert F.fetch_next_draw(site, None).draw_type == "normal"


def test_fetch_next_draw_does_not_predict_a_cascade_from_stale_history():
    # The stored history ends Mon 28 Sep with 3 snowballs, but Thu 1 Oct was held and is not
    # stored: the Mon 5 Oct draw may be anything, so no cascade (and no HIGH signal) is claimed.
    from huatbot import buysignal
    from huatbot.models import PrizeRules, Settings

    df = _streak_df_ending([1, 0, 0, 0], date(2026, 9, 28))
    t = F.fetch_next_draw(static_site(), df)
    assert t.draw_type == "normal" and t.draw_type_hint is None
    signal = buysignal.buy_signal(t, df, Settings(jackpot_alert=1e12), PrizeRules())
    assert signal.draw_type == "normal" and signal.label != "HIGH"
    assert not signal.ev_breakdown.get("cascade")


def test_fetch_next_draw_missing_or_unreadable_page():
    assert F.fetch_next_draw(H.FakeFetcher({}), None) is None
    assert F.fetch_next_draw(H.FakeFetcher({C.TOTO_NEXT_DRAW_URL: "<p>Coming soon</p>"}), None) is None
    assert F.fetch_next_draw(H.FakeFetcher({C.TOTO_NEXT_DRAW_URL: RuntimeError("weird")}), None) is None


def test_fetch_next_draw_with_only_a_date_or_only_a_jackpot():
    dt = datetime(2026, 10, 8, 18, 30, tzinfo=SG)
    only_date = H.FakeFetcher({C.TOTO_NEXT_DRAW_URL: H.toto_next_draw_html(dt, None)})
    t = F.fetch_next_draw(only_date, None)
    assert t.draw_datetime == dt and t.jackpot_estimate is None
    only_jackpot = H.FakeFetcher({C.TOTO_NEXT_DRAW_URL: H.toto_next_draw_html(None, 2_000_000)})
    t = F.fetch_next_draw(only_jackpot, None)
    assert t.draw_datetime is None and t.jackpot_estimate == 2_000_000


# latest_on_site


def test_latest_on_site():
    site = static_site()
    assert F.latest_on_site(site) == (4123, date(2026, 10, 1))
    assert site.requested == [C.TOTO_DRAW_LIST_URL]
    assert F.latest_on_site(H.FakeFetcher({})) == (None, None)
    empty = H.FakeFetcher({C.TOTO_DRAW_LIST_URL: "<html><body>Maintenance</body></html>"})
    assert F.latest_on_site(empty) == (None, None)
    assert F.latest_on_site(empty, strict=True) == (None, None)  # reachable, just nothing listed


# check_site


def test_check_site_all_pass_on_fake_site(toto_df):
    lines: list[str] = []
    site = H.fake_site(toto_df)
    assert F.check_site(site, out=lines.append) is True
    assert len(lines) == 8
    assert all(line.startswith("PASS  ") for line in lines[:-1]), lines
    assert [line.split(":")[0] for line in lines[:-1]] == [
        "PASS  TOTO draw list", "PASS  TOTO latest result page", "PASS  TOTO next draw page",
        "PASS  TOTO cascade draw list", "PASS  TOTO Hongbao draw list", "PASS  TOTO special draw list",
        "PASS  TOTO prize structure page (not critical)",
    ]
    assert lines[-1] == "Summary: all 6 critical checks passed."
    assert_no_dashes(lines)
    assert not any("fourd" in u or "4d" in u for u in site.requested)


def test_check_site_static_fixtures():
    lines: list[str] = []
    assert F.check_site(static_site(), out=lines.append) is True
    text = "\n".join(lines)
    assert "draw 4123 on Thu 1 Oct 2026, numbers 3 11 19 27 38 45, additional 7" in text
    assert "next draw Mon 5 Oct 2026, 6.30pm, estimated jackpot $1,000,000" in text
    assert "Group 1 38%, Group 2 8%, Group 3 5.5%, Group 4 3%, Group 5 $50, Group 6 $25, Group 7 $10, " \
           "prize pool 54% of sales" in text
    assert "4D" not in text
    assert_no_dashes(lines)


@pytest.mark.parametrize("prize_pages", ["js", None])
def test_check_site_prize_page_is_not_critical(toto_df, prize_pages):
    lines: list[str] = []
    site = H.fake_site(toto_df, prize_pages=prize_pages)
    assert F.check_site(site, out=lines.append) is True
    prize = [line for line in lines if "prize structure" in line]
    assert len(prize) == 1 and prize[0].startswith("FAIL") and "(not critical)" in prize[0]
    assert "built in values will be used" in prize[0]
    assert lines[-1] == "Summary: all 6 critical checks passed, 1 of 1 non critical checks failed."
    assert_no_dashes(lines)


def test_check_site_fails_when_a_result_page_is_missing(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    del site.pages[F.toto_result_url(latest)]
    lines: list[str] = []
    assert F.check_site(site, out=lines.append) is False
    assert any(line.startswith(f"FAIL  TOTO latest result page: draw {latest}: could not fetch the page")
               for line in lines)
    assert lines[-1] == "Summary: 1 of 6 critical checks failed (TOTO latest result page)."
    assert_no_dashes(lines)


def test_check_site_reports_a_wrong_draw_and_a_wrong_date(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    site.pages[F.toto_result_url(latest)] = H.toto_result_html(toto_df.iloc[-2])
    lines: list[str] = []
    assert F.check_site(site, out=lines.append) is False
    assert f"asked for draw {latest} but the page shows draw {latest - 1}" in "\n".join(lines)

    site = H.fake_site(toto_df)
    pairs = list(zip(toto_df["draw_number"].tail(5), toto_df["draw_date"].tail(5)))
    pairs[-1] = (pairs[-1][0], pairs[-1][1] + pd.Timedelta(days=1))
    site.pages[C.TOTO_DRAW_LIST_URL] = H.draw_list_html(pairs)
    lines = []
    assert F.check_site(site, out=lines.append) is False
    assert any("on the page but" in line and "on the draw list" in line for line in lines)
    assert_no_dashes(lines)


def test_check_site_with_nothing_reachable():
    lines: list[str] = []
    assert F.check_site(H.FakeFetcher({}), out=lines.append) is False
    assert sum(line.startswith("FAIL") for line in lines) == 7
    assert any("skipped because the draw list could not be read" in line for line in lines)
    assert lines[-1].startswith("Summary: 6 of 6 critical checks failed")
    assert_no_dashes(lines)


def test_check_site_survives_parser_bugs_and_fetcher_errors(monkeypatch):
    def broken(html):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(F, "parse_toto_next_draw", broken)
    lines: list[str] = []
    assert F.check_site(static_site(), out=lines.append) is False
    assert "FAIL  TOTO next draw page: unexpected error (RuntimeError)" in lines

    class Exploding:
        def get_many(self, urls):
            raise RuntimeError("boom")

    lines = []
    assert F.check_site(Exploding(), out=lines.append) is False
    assert lines[-1].startswith("Summary: 6 of 6 critical checks failed")
    assert_no_dashes(lines)


# partly published pages


def _without_shares(html: str) -> str:
    """The result page as it may look right after the draw: numbers out, shares table not yet."""
    out = re.sub(r"<table class='table table-striped tableWinningShares'>.*?</table>", "", html, flags=re.S)
    assert out != html
    return out


def test_toto_page_without_shares_table_is_not_stored_and_is_fetched_again(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    complete = site.pages[F.toto_result_url(latest)]
    site.pages[F.toto_result_url(latest)] = _without_shares(complete)
    df, res = F.update_toto(site, toto_df.iloc[:-1], start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert res.new_draws == [] and res.failed_draws == [latest]
    assert latest not in set(df["draw_number"])
    assert res.verified is False
    assert any(f"draw {latest} was not added" in m and "winning shares table" in m for m in res.messages)
    assert_no_dashes(res.messages)
    # once the shares are published the next update stores the draw
    site.pages[F.toto_result_url(latest)] = complete
    df2, res2 = F.update_toto(site, df, start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert res2.new_draws == [latest] and res2.verified
    assert int(df2.set_index("draw_number").loc[latest, "g7_winners"]) > 0


def test_incomplete_stored_newest_draw_is_read_again(toto_df):
    latest = int(toto_df["draw_number"].max())
    stored = toto_df.copy()
    idx = stored.index[stored["draw_number"] == latest][0]
    for g in range(1, 8):  # saved before the shares were published
        stored.loc[idx, f"g{g}_share"] = float("nan")
        stored.loc[idx, f"g{g}_winners"] = 0
    site = H.fake_site(toto_df)
    good = site.pages[F.toto_result_url(latest)]

    # still incomplete on the site: the old row is kept and reported, not counted as new or failed
    site.pages[F.toto_result_url(latest)] = _without_shares(good)
    df, res = F.update_toto(site, stored, start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert site.count(F.toto_result_url(latest)) == 1
    assert res.new_draws == [] and res.failed_draws == []
    assert res.verified is False
    assert any(f"TOTO draw {latest} does not have all its winning shares yet" in m for m in res.messages)
    assert any("not complete yet" in m for m in res.messages)
    assert int(df.set_index("draw_number").loc[latest, "g7_winners"]) == 0

    # published: the row is replaced with the full result
    site.pages[F.toto_result_url(latest)] = good
    df, res = F.update_toto(site, df, start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert res.new_draws == [] and res.verified
    assert any(f"TOTO draw {latest} was updated with its winning shares" in m for m in res.messages)
    assert res.repaired_draws == [latest]
    fixed = df.set_index("draw_number").loc[latest]
    expected = toto_df.set_index("draw_number").loc[latest]
    assert int(fixed["g7_winners"]) == int(expected["g7_winners"]) > 0
    assert len(df) == len(toto_df)
    assert_no_dashes(res.messages)


def test_only_the_newest_stored_draws_are_rechecked(toto_df):
    stored = toto_df.copy()
    old = int(stored["draw_number"].iloc[-10])
    stored.loc[stored["draw_number"] == old, "g7_winners"] = 0
    site = H.fake_site(toto_df)
    F.update_toto(site, stored, start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert result_urls(site) == []


def test_latest_complete_date_waits_for_the_shares_table(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    day = toto_df["draw_date"].iloc[-1].date()
    good = site.pages[F.toto_result_url(latest)]
    site.pages[F.toto_result_url(latest)] = _without_shares(good)
    assert F.latest_complete_date(site) is None
    site.pages[F.toto_result_url(latest)] = good
    assert F.latest_complete_date(site) == day
    assert F.latest_complete_date(site, strict=True) == day
    site.pages[F.toto_result_url(latest)] = H.toto_result_html(toto_df.iloc[-2])  # the wrong draw
    assert F.latest_complete_date(site) is None
    del site.pages[F.toto_result_url(latest)]
    assert F.latest_complete_date(site) is None
    assert F.latest_complete_date(H.FakeFetcher({})) is None


def test_incomplete_reason():
    assert F.incomplete_reason({"g7_winners": 84113}) is None
    assert F.incomplete_reason(pd.Series({"g7_winners": 12})) is None
    for row in ({"g7_winners": 0}, {}, {"g7_winners": None}, {"g7_winners": "x"}):
        assert F.incomplete_reason(row) == "the winning shares table is not on the page yet"


# draw list sanity


def test_draw_list_with_a_stray_huge_number_is_refused(toto_df):
    site = H.fake_site(toto_df)
    latest = int(toto_df["draw_number"].max())
    pairs = list(zip(toto_df["draw_number"].tail(5), toto_df["draw_date"].tail(5)))
    site.pages[C.TOTO_DRAW_LIST_URL] = H.draw_list_html(pairs + [(latest * 10, pairs[-1][1])])
    with pytest.raises(FetchError) as info:
        F.update_toto(site, toto_df.iloc[:-1], start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert "draw list looks wrong" in str(info.value) and not DASHES.search(str(info.value))
    assert result_urls(site) == []
    # the same on a first fill, judged against the rest of the list
    with pytest.raises(FetchError):
        F.update_toto(site, empty_toto(), start_draw=latest - 5, now=NOW)
    assert result_urls(site) == []


def test_draw_list_after_a_long_break_is_accepted(toto_df):
    # 50 draws behind, but months have passed since the newest stored draw: that is plausible
    site = H.fake_site(toto_df)
    df, res = F.update_toto(site, toto_df.iloc[:-50], start_draw=int(toto_df["draw_number"].min()), now=NOW)
    assert len(res.new_draws) == 50 and res.verified


def test_latest_on_site_strict_raises_when_the_site_cannot_be_reached():
    from huatbot.http import FetchError as FE

    class Down:
        def get(self, url):
            raise FE("site down")

    assert F.latest_on_site(Down()) == (None, None)
    with pytest.raises(FE):
        F.latest_on_site(Down(), strict=True)
    with pytest.raises(FE):
        F.latest_complete_date(Down(), strict=True)
    assert F.latest_complete_date(Down()) is None


# TOTO only API


def test_fetch_has_no_fourd_api():
    for name in ("fourd_result_url", "update_fourd", "fetch_next_draws", "GAME_LABELS"):
        assert not hasattr(F, name), name


def test_signatures_take_no_game_argument():
    assert list(inspect.signature(F.latest_on_site).parameters) == ["fetcher", "strict"]
    assert list(inspect.signature(F.latest_complete_date).parameters) == ["fetcher", "strict"]
    assert list(inspect.signature(F.incomplete_reason).parameters) == ["row"]
    assert list(inspect.signature(F.fetch_next_draw).parameters) == ["fetcher", "toto_df"]
    assert list(inspect.signature(F.update_toto).parameters) == ["fetcher", "df", "start_draw", "skip", "now"]
    # strict is keyword only, so an old call with a game name fails loudly instead of meaning strict
    with pytest.raises(TypeError):
        F.latest_on_site(static_site(), "toto")
    with pytest.raises(TypeError):
        F.latest_complete_date(static_site(), "toto", strict=True)
    with pytest.raises(TypeError):
        F.incomplete_reason("toto", {"g7_winners": 1})
