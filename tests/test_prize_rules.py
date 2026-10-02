"""load_prize_rules: built in fallback, confirmation from the official page, caching, never raising."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest

from huatbot import constants as C
from huatbot.http import FetchError
from huatbot.models import PrizeRules
from huatbot.prize_rules import CACHE_VERSION, load_prize_rules, pct_text, rules_from_dict, rules_to_dict
from huatbot.synth import SG
from tests import htmlgen as H
from tests.conftest import FIXTURES

NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)
DASHES = re.compile("[-–—]")


def prize_site(toto: str | Exception | None = "fixture") -> H.FakeFetcher:
    """A fetcher serving only the TOTO prize structure page ("fixture", given HTML, an error or nothing)."""
    if toto == "fixture":
        toto = (FIXTURES / "toto_prize_structure.html").read_text(encoding="utf-8")
    return H.FakeFetcher({} if toto is None else {C.TOTO_PRIZE_RULES_URL: toto})


def test_no_fetcher_gives_built_in_values(tmp_path):
    rules = load_prize_rules(None, cache_path=tmp_path / "prize_rules.json", now=NOW)
    assert rules == PrizeRules(source_note=rules.source_note)
    assert rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT
    assert not rules.toto_confirmed and not rules.confirmed
    assert "built in" in rules.source_note.lower() and "4D" not in rules.source_note
    assert not (tmp_path / "prize_rules.json").exists()


def test_confirmed_from_official_page_and_cached(tmp_path):
    cache = tmp_path / "Data" / "prize_rules.json"
    site = prize_site()
    rules = load_prize_rules(site, cache_path=cache, now=NOW)
    assert site.requested == [C.TOTO_PRIZE_RULES_URL]  # the TOTO page only
    assert rules.toto_confirmed and rules.confirmed
    assert rules.group_pool_pct == {1: 0.38, 2: 0.08, 3: 0.055, 4: 0.03}
    assert rules.fixed_prizes == {5: 50.0, 6: 25.0, 7: 10.0}
    assert rules.pool_share_of_sales == 0.54 and rules.min_group1 == 1_000_000.0
    assert rules.checked_at == NOW.isoformat(timespec="seconds")
    assert rules.source_note == "TOTO prize percentages confirmed from the official page on Fri 2 Oct 2026."
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert data["toto_confirmed"] is True and data["group_pool_pct"]["3"] == 0.055
    assert data["version"] == CACHE_VERSION
    assert not any("fourd" in k or "ibet" in k for k in data)

    # A second call within the week uses the cache without fetching
    again = H.FakeFetcher({})
    cached = load_prize_rules(again, cache_path=cache, now=NOW + timedelta(days=3))
    assert again.requested == []
    assert cached == rules


def test_stale_cache_is_refreshed(tmp_path):
    cache = tmp_path / "prize_rules.json"
    load_prize_rules(prize_site(), cache_path=cache, now=NOW)
    site = prize_site()
    later = NOW + timedelta(days=8)
    rules = load_prize_rules(site, cache_path=cache, now=later)
    assert site.requested == [C.TOTO_PRIZE_RULES_URL]
    assert rules.checked_at == later.isoformat(timespec="seconds")
    assert json.loads(cache.read_text())["checked_at"] == later.isoformat(timespec="seconds")


def test_javascript_page_falls_back_to_built_in(tmp_path):
    cache = tmp_path / "prize_rules.json"
    rules = load_prize_rules(prize_site(H.toto_prize_structure_html(rendered=False)), cache_path=cache, now=NOW)
    assert not rules.toto_confirmed and not rules.confirmed
    assert rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT
    assert "JavaScript" in rules.source_note and "built in" in rules.source_note
    assert not DASHES.search(rules.source_note)
    assert cache.exists()  # the page was fetched, so no need to ask again for a week


def test_parsed_values_override_built_in():
    toto = H.toto_prize_structure_html(group_pool_pct={1: 0.38, 2: 0.08, 3: 0.06, 4: 0.03},
                                       fixed_prizes={5: 60.0, 6: 25.0, 7: 10.0}, pool_share=0.55,
                                       min_group1=1_500_000.0)
    rules = load_prize_rules(prize_site(toto), cache_path=None, now=NOW)
    assert rules.group_pool_pct[3] == 0.06 and rules.fixed_prizes[5] == 60.0
    assert rules.pool_share_of_sales == 0.55 and rules.min_group1 == 1_500_000.0
    note = rules.source_note
    assert ("Official values that differ from the built in ones: Group 3 is 6% (built in 5.5%); "
            "Group 5 is $60 (built in $50); prize pool is 55% of sales (built in 54%); "
            "Group 1 minimum is $1,500,000 (built in $1,000,000).") in note
    assert not DASHES.search(note)


def test_missing_figures_are_named_as_built_in():
    rules = load_prize_rules(prize_site(H.toto_prize_structure_html(pool_share=None)), cache_path=None, now=NOW)
    assert rules.toto_confirmed
    assert "the 54% share of sales is built in" in rules.source_note
    assert rules.pool_share_of_sales == C.TOTO_POOL_SHARE_OF_SALES


def test_missing_fixed_prizes_are_not_confirmed():
    html = ("<p>Group 1 gets 38% of the Prize Pool.</p><p>Group 2 gets 8%.</p><p>Group 3 gets 5.5%.</p>"
            "<p>Group 4 gets 3%.</p><p>Group 7 pays $10.</p>")
    rules = load_prize_rules(prize_site(html), cache_path=None, now=NOW)
    assert not rules.toto_confirmed
    assert rules.fixed_prizes == C.TOTO_FIXED_PRIZES
    assert ("the fixed prizes for Group 5 and 6, the 54% share of sales, the $1,000,000 Group 1 minimum "
            "are built in.") in rules.source_note


def test_network_failure_uses_built_in_and_does_not_cache(tmp_path):
    cache = tmp_path / "prize_rules.json"
    rules = load_prize_rules(H.FakeFetcher({}), cache_path=cache, now=NOW)
    assert not rules.toto_confirmed
    assert rules.source_note == "The official TOTO prize page could not be fetched, so the built in values are used."
    assert not cache.exists()  # retried on the next run


def test_network_failure_falls_back_to_stale_confirmed_cache(tmp_path):
    cache = tmp_path / "prize_rules.json"
    load_prize_rules(prize_site(H.toto_prize_structure_html(group_pool_pct={1: 0.37, 2: 0.09, 3: 0.055, 4: 0.03})),
                     cache_path=cache, now=NOW)
    later = NOW + timedelta(days=30)
    rules = load_prize_rules(H.FakeFetcher({}), cache_path=cache, now=later)
    assert rules.toto_confirmed and rules.group_pool_pct[1] == 0.37
    assert rules.checked_at == later.isoformat(timespec="seconds")
    assert rules.source_note == ("TOTO prize rules are from the last successful check on Fri 2 Oct 2026 "
                                 "(the official page could not be fetched today).")
    assert json.loads(cache.read_text())["checked_at"] == NOW.isoformat(timespec="seconds")  # not rewritten


class ExplodingFetcher:
    def get(self, url):
        raise RuntimeError("boom")

    def get_many(self, urls):
        raise RuntimeError("boom")


@pytest.mark.parametrize("fetcher", [ExplodingFetcher(), prize_site(FetchError("down")),
                                     prize_site(RuntimeError("weird"))])
def test_never_raises(tmp_path, fetcher):
    rules = load_prize_rules(fetcher, cache_path=tmp_path / "x.json", now=NOW)
    assert isinstance(rules, PrizeRules)
    assert rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT
    assert not (tmp_path / "x.json").exists()


def test_parser_bug_falls_back_to_built_in(tmp_path, monkeypatch):
    from huatbot import prize_rules

    def broken(html):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(prize_rules, "parse_toto_prize_structure", broken)
    rules = load_prize_rules(prize_site(), cache_path=None, now=NOW)
    assert not rules.toto_confirmed and rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT


def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "prize_rules.json"
    cache.write_text("{not json")
    rules = load_prize_rules(prize_site(), cache_path=cache, now=NOW)
    assert rules.confirmed
    assert json.loads(cache.read_text())["toto_confirmed"] is True


def test_unwritable_cache_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    rules = load_prize_rules(prize_site(), cache_path=blocker / "sub" / "prize_rules.json", now=NOW)
    assert rules.confirmed


def test_rules_dict_round_trip():
    rules = PrizeRules(
        group_pool_pct={1: 0.37, 2: 0.08, 3: 0.06, 4: 0.03}, min_group1=1_200_000.0,
        toto_confirmed=True, source_note="checked", checked_at=NOW.isoformat(),
    )
    data = rules_to_dict(rules)
    assert set(data) == {"version", "pool_share_of_sales", "group_pool_pct", "fixed_prizes", "min_group1",
                         "toto_confirmed", "source_note", "checked_at"}
    restored = rules_from_dict(json.loads(json.dumps(data)))
    assert restored == rules


def test_rules_from_partial_dict_uses_defaults():
    assert rules_from_dict({}) == PrizeRules()
    assert rules_from_dict(None) == PrizeRules()
    r = rules_from_dict({"fixed_prizes": {"5": 60}, "unknown": 1, "min_group1": None})
    assert r.fixed_prizes == {5: 60.0} and r.group_pool_pct == C.TOTO_GROUP_POOL_PCT
    assert r.min_group1 == C.TOTO_MIN_GROUP1


# a cache written by the TOTO and 4D version

OLD_NOTE = ("TOTO prize percentages confirmed from the official page on Fri 2 Oct 2026; the $1,000,000 Group 1 "
            "minimum is built in. 4D Big, Small and iBet prize tables confirmed from the official page on Fri 2 "
            "Oct 2026. Official values that differ from the built in ones: Group 3 is 6% (built in 5.5%); 4D Big "
            "first is $2,500 (built in $2,000).")


def old_cache(checked_at: str, note: str = OLD_NOTE) -> dict:
    return {
        "version": 1,
        "pool_share_of_sales": 0.54,
        "group_pool_pct": {"1": 0.38, "2": 0.08, "3": 0.06, "4": 0.03},
        "fixed_prizes": {"5": 50.0, "6": 25.0, "7": 10.0},
        "min_group1": 1000000.0,
        "fourd_prizes": {"big": {"first": 2500.0, "second": 1000.0, "third": 490.0, "starter": 250.0,
                                 "consolation": 60.0},
                         "small": {"first": 3000.0, "second": 2000.0, "third": 800.0}},
        "ibet_prizes": {"big": {"24": {"first": 104.0}}},
        "toto_confirmed": True,
        "fourd_confirmed": True,
        "source_note": note,
        "checked_at": checked_at,
    }


def test_old_cache_with_fourd_keys_loads(tmp_path):
    cache = tmp_path / "prize_rules.json"
    cache.write_text(json.dumps(old_cache(NOW.isoformat(timespec="seconds")), indent=2))
    site = prize_site()
    rules = load_prize_rules(site, cache_path=cache, now=NOW + timedelta(days=2))
    assert site.requested == []  # still fresh: used as it is
    assert rules.toto_confirmed and rules.confirmed and rules.group_pool_pct[3] == 0.06
    assert rules.source_note == ("TOTO prize percentages confirmed from the official page on Fri 2 Oct 2026; "
                                 "the $1,000,000 Group 1 minimum is built in. Official values that differ from "
                                 "the built in ones: Group 3 is 6% (built in 5.5%).")
    # stale: refreshed from the page and rewritten in the TOTO only format
    rules = load_prize_rules(site, cache_path=cache, now=NOW + timedelta(days=10))
    assert site.requested == [C.TOTO_PRIZE_RULES_URL]
    data = json.loads(cache.read_text())
    assert data["version"] == CACHE_VERSION and "fourd_prizes" not in data and "fourd_confirmed" not in data


def test_old_cache_is_the_fallback_when_the_page_cannot_be_fetched(tmp_path):
    cache = tmp_path / "prize_rules.json"
    cache.write_text(json.dumps(old_cache(NOW.isoformat(timespec="seconds"))))
    rules = load_prize_rules(H.FakeFetcher({}), cache_path=cache, now=NOW + timedelta(days=30))
    assert rules.toto_confirmed and rules.group_pool_pct[3] == 0.06
    assert "4D" not in rules.source_note and "last successful check on Fri 2 Oct 2026" in rules.source_note


@pytest.mark.parametrize("note, expected", [
    (OLD_NOTE, "TOTO prize percentages confirmed from the official page on Fri 2 Oct 2026; the $1,000,000 "
               "Group 1 minimum is built in. Official values that differ from the built in ones: Group 3 is 6% "
               "(built in 5.5%)."),
    ("TOTO prize figures are not in the official page text (it is probably drawn by JavaScript), so the built in "
     "values are used. The official 4D prize page could not be fetched, so the built in values are used.",
     "TOTO prize figures are not in the official page text (it is probably drawn by JavaScript), so the built in "
     "values are used."),
    ("TOTO prize percentages confirmed from the official page on Fri 2 Oct 2026. 4D Big and Small prize tables "
     "confirmed from the official page on Fri 2 Oct 2026. Official values that differ from the built in ones: "
     "4D Big first is $2,500 (built in $2,000); 4D Small first is $3,500 (built in $3,000).",
     "TOTO prize percentages confirmed from the official page on Fri 2 Oct 2026."),
    ("4D prize table is from the last successful check.", "built in values"),  # nothing left: the default note
    ("Built in prize values are used (the official prize pages were not checked this run).",
     "Built in prize values are used (the official prize pages were not checked this run)."),
])
def test_old_notes_lose_their_fourd_parts(note, expected):
    assert rules_from_dict({"source_note": note}).source_note == expected


@pytest.mark.parametrize("x, text", [(0.38, "38%"), (0.055, "5.5%"), (0.54, "54%"), (0.0525, "5.25%"),
                                     (1.0, "100%"), (-0.012, "minus 1.2%"), (None, "n/a")])
def test_pct_text(x, text):
    assert pct_text(x) == text
