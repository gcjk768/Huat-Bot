"""fetch.py against a fake site built from synthetic history and the static fixtures."""
from __future__ import annotations

import math
import re
from datetime import date, datetime

import pandas as pd
import pytest

from huatbot import constants as C
from huatbot import fetch as F
from huatbot.http import FetchError
from huatbot.models import FOURD_NUMBER_COLUMNS, NextToto
from huatbot.store import empty_fourd, empty_toto, normalise_toto
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


def result_urls(fetcher: H.FakeFetcher, game: str = "toto") -> list[str]:
    marker = "toto_results.aspx" if game == "toto" else "4d_results.aspx"
    return [u for u in fetcher.requested if marker in u]


def static_site() -> H.FakeFetcher:
    """The handwritten fixtures mapped to their real URLs."""
    read = lambda name: (FIXTURES / name).read_text(encoding="utf-8")  # noqa: E731
    return H.FakeFetcher({
        C.TOTO_DRAW_LIST_URL: read("toto_draw_list.html"),
        C.FOURD_DRAW_LIST_URL: read("fourd_draw_list.html"),
        F.toto_result_url(4123): read("toto_result_with_winner.html"),
        F.toto_result_url(4122): read("toto_result_no_winner.html"),
        F.fourd_result_url(5432): read("fourd_result.html"),
        C.TOTO_NEXT_DRAW_URL: read("toto_next_draw.html"),
        C.FOURD_NEXT_DRAW_URL: read("fourd_next_draw.html"),
        C.TOTO_CASCADE_LIST_URL: read("toto_cascade_list.html"),
        C.TOTO_HONGBAO_LIST_URL: H.draw_type_list_html([]),
        C.TOTO_SPECIAL_LIST_URL: H.draw_type_list_html([]),
        C.TOTO_PRIZE_RULES_URL: read("toto_prize_structure.html"),
        C.FOURD_PRIZE_RULES_URL: read("fourd_prize_structure.html"),
    })


# URLs


def test_result_urls():
    assert F.toto_result_url(4123) == (
        "https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl=RHJhd051bWJlcj00MTIz")
    assert F.fourd_result_url(5432) == (
        "https://www.singaporepools.com.sg/en/product/Pages/4d_results.aspx?sppl=RHJhd051bWJlcj01NDMy")


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
    types = dict(zip(F.tag_draw_types(df, None, None, None)["draw_number"], F.tag_draw_types(df, None, None, None)["draw_type"]))
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


# update_fourd


def test_update_fourd_fills_the_window(fourd_df):
    site = H.fake_site(fourd_df=fourd_df)
    df, res = F.update_fourd(site, empty_fourd(), history_draws=30, now=NOW)
    latest = int(fourd_df["draw_number"].max())
    assert res.new_draws == list(range(latest - 29, latest + 1))
    assert res.verified and res.game == "4d"
    assert_same_results(df, fourd_df.tail(30).reset_index(drop=True), ["draw_date"] + FOURD_NUMBER_COLUMNS)
    assert_no_dashes(res.messages)


def test_update_fourd_keeps_existing_rows_and_fetches_only_missing(fourd_df):
    site = H.fake_site(fourd_df=fourd_df)
    old = fourd_df.iloc[:100]  # far outside the window, must be kept
    recent = fourd_df.iloc[-50:-3]
    local = pd.concat([old, recent], ignore_index=True)
    df, res = F.update_fourd(site, local, history_draws=50, now=NOW)
    assert res.new_draws == list(fourd_df["draw_number"].iloc[-3:])
    assert len(result_urls(site, "4d")) == 3
    assert len(df) == 100 + 50
    assert set(old["draw_number"]) <= set(df["draw_number"])
    assert res.verified


def test_update_fourd_skip_and_failures(fourd_df):
    site = H.fake_site(fourd_df=fourd_df)
    latest = int(fourd_df["draw_number"].max())
    del site.pages[F.fourd_result_url(latest)]
    df, res = F.update_fourd(site, fourd_df.iloc[:-4], history_draws=100, skip={latest - 1}, now=NOW)
    assert res.new_draws == [latest - 3, latest - 2]
    assert res.failed_draws == [latest]
    assert res.verified is False and res.latest_in_csv == latest - 2
    assert_no_dashes(res.messages)


def test_update_fourd_with_static_fixtures():
    site = static_site()
    df, res = F.update_fourd(site, empty_fourd(), history_draws=1, now=NOW)
    assert res.new_draws == [5432] and res.verified
    row = df.iloc[0]
    assert row["first"] == "0417" and row["starter_6"] == "0038"


# next draws


def test_fetch_next_draws_from_pages(toto_df, fourd_df):
    dt = datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    nt = NextToto(draw_datetime=dt, jackpot_estimate=4_500_000.0, draw_type="normal", draw_type_hint="hongbao")
    site = H.fake_site(toto_df, fourd_df, next_toto=nt)
    t, f = F.fetch_next_draws(site, toto_df)
    assert t.draw_datetime == dt and t.jackpot_estimate == 4_500_000.0
    assert t.draw_type == "hongbao" and t.draw_type_hint == "hongbao"
    assert "Next Jackpot" in t.raw_text and len(t.raw_text) <= F.RAW_TEXT_LIMIT
    assert f.draw_datetime is not None and f.draw_datetime.tzinfo is not None


def test_fetch_next_draws_predicts_cascade_from_history():
    df = _streak_df([1, 0, 0, 0])  # 3 snowballs in a row: the next draw is the 4th
    site = static_site()
    t, f = F.fetch_next_draws(site, df)
    assert t.draw_type_hint is None and t.draw_type == "cascade"
    assert t.draw_datetime == datetime(2026, 10, 5, 18, 30, tzinfo=SG)
    assert f.draw_datetime == datetime(2026, 10, 3, 18, 30, tzinfo=SG)
    t2, _ = F.fetch_next_draws(site, _streak_df([1, 0, 0]))
    assert t2.draw_type == "normal"
    t3, _ = F.fetch_next_draws(site, empty_toto())
    assert t3.draw_type == "normal"


def test_fetch_next_draws_missing_pages():
    assert F.fetch_next_draws(H.FakeFetcher({}), None) == (None, None)
    junk = H.FakeFetcher({C.TOTO_NEXT_DRAW_URL: "<p>Coming soon</p>", C.FOURD_NEXT_DRAW_URL: "<p>Coming soon</p>"})
    assert F.fetch_next_draws(junk, None) == (None, None)


# latest_on_site


def test_latest_on_site():
    site = static_site()
    assert F.latest_on_site(site, "toto") == (4123, date(2026, 10, 1))
    assert F.latest_on_site(site, "4d") == (5432, date(2026, 9, 30))
    assert F.latest_on_site(H.FakeFetcher({}), "toto") == (None, None)


# check_site


def test_check_site_all_pass_on_fake_site(toto_df, fourd_df):
    lines: list[str] = []
    assert F.check_site(H.fake_site(toto_df, fourd_df), out=lines.append) is True
    assert len(lines) == 12
    assert all(line.startswith(("PASS  ", "FAIL  ")) for line in lines[:-1])
    assert all(line.startswith("PASS") for line in lines[:-1]), lines
    assert lines[-1].startswith("Summary: all 9 critical checks passed")
    assert_no_dashes(lines)


def test_check_site_static_fixtures():
    lines: list[str] = []
    assert F.check_site(static_site(), out=lines.append) is True
    text = "\n".join(lines)
    assert "draw 4123 on Thu 1 Oct 2026, numbers 3 11 19 27 38 45, additional 7" in text
    assert "next draw Mon 5 Oct 2026, 6.30pm, estimated jackpot $1,000,000" in text
    assert "Group 3 5.5%" in text
    assert "and iBet tables" in text
    assert_no_dashes(lines)


def test_check_site_prize_pages_are_not_critical(toto_df, fourd_df):
    lines: list[str] = []
    site = H.fake_site(toto_df, fourd_df, prize_pages="js")
    assert F.check_site(site, out=lines.append) is True
    prize = [line for line in lines if "prize structure" in line]
    assert len(prize) == 2 and all(line.startswith("FAIL") and "(not critical)" in line for line in prize)
    assert "2 of 2 non critical checks failed" in lines[-1]
    assert_no_dashes(lines)


def test_check_site_fails_when_a_result_page_is_missing(toto_df, fourd_df):
    site = H.fake_site(toto_df, fourd_df)
    del site.pages[F.toto_result_url(int(toto_df["draw_number"].max()))]
    lines: list[str] = []
    assert F.check_site(site, out=lines.append) is False
    assert any(line.startswith("FAIL  TOTO latest result page") for line in lines)
    assert "1 of 9 critical checks failed (TOTO latest result page)" in lines[-1]
    assert_no_dashes(lines)


def test_check_site_with_nothing_reachable():
    lines: list[str] = []
    assert F.check_site(H.FakeFetcher({}), out=lines.append) is False
    assert sum(line.startswith("FAIL") for line in lines) == 11
    assert any("skipped because the draw list could not be read" in line for line in lines)
    assert_no_dashes(lines)
