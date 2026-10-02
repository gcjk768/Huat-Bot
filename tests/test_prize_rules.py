"""load_prize_rules: built in fallback, confirmation from the official pages, caching, never raising."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest

from huatbot import constants as C
from huatbot.http import FetchError
from huatbot.models import PrizeRules
from huatbot.prize_rules import load_prize_rules, rules_from_dict, rules_to_dict
from huatbot.synth import SG
from tests import htmlgen as H
from tests.conftest import FIXTURES

NOW = datetime(2026, 10, 2, 19, 30, tzinfo=SG)
DASHES = re.compile("[-–—]")


def prize_site(toto: str | Exception | None = "fixture", fourd: str | Exception | None = "fixture") -> H.FakeFetcher:
    pages = {}
    for url, value, name in ((C.TOTO_PRIZE_RULES_URL, toto, "toto_prize_structure.html"),
                             (C.FOURD_PRIZE_RULES_URL, fourd, "fourd_prize_structure.html")):
        if value == "fixture":
            pages[url] = (FIXTURES / name).read_text(encoding="utf-8")
        elif value is not None:
            pages[url] = value
    return H.FakeFetcher(pages)


def test_no_fetcher_gives_built_in_values(tmp_path):
    rules = load_prize_rules(None, cache_path=tmp_path / "prize_rules.json", now=NOW)
    assert rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT
    assert rules.fourd_prizes == C.FOURD_PRIZES
    assert not rules.toto_confirmed and not rules.fourd_confirmed and not rules.confirmed
    assert "built in" in rules.source_note.lower()
    assert not (tmp_path / "prize_rules.json").exists()


def test_confirmed_from_official_pages_and_cached(tmp_path):
    cache = tmp_path / "Data" / "prize_rules.json"
    site = prize_site()
    rules = load_prize_rules(site, cache_path=cache, now=NOW)
    assert rules.toto_confirmed and rules.fourd_confirmed and rules.confirmed
    assert rules.group_pool_pct == {1: 0.38, 2: 0.08, 3: 0.055, 4: 0.03}
    assert rules.fixed_prizes == {5: 50.0, 6: 25.0, 7: 10.0}
    assert rules.pool_share_of_sales == 0.54 and rules.min_group1 == 1_000_000.0
    assert rules.fourd_prizes == C.FOURD_PRIZES
    assert rules.ibet_prizes["big"][24]["first"] == 83.0
    assert rules.checked_at == NOW.isoformat(timespec="seconds")
    assert "confirmed" in rules.source_note and not DASHES.search(rules.source_note)
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert data["toto_confirmed"] is True and data["group_pool_pct"]["3"] == 0.055

    # A second call within the week uses the cache without fetching
    again = H.FakeFetcher({})
    cached = load_prize_rules(again, cache_path=cache, now=NOW + timedelta(days=3))
    assert again.requested == []
    assert cached.confirmed and cached.group_pool_pct == rules.group_pool_pct
    assert cached.ibet_prizes == rules.ibet_prizes


def test_stale_cache_is_refreshed(tmp_path):
    cache = tmp_path / "prize_rules.json"
    load_prize_rules(prize_site(), cache_path=cache, now=NOW)
    site = prize_site()
    later = NOW + timedelta(days=8)
    rules = load_prize_rules(site, cache_path=cache, now=later)
    assert len(site.requested) == 2
    assert rules.checked_at == later.isoformat(timespec="seconds")
    assert json.loads(cache.read_text())["checked_at"] == later.isoformat(timespec="seconds")


def test_javascript_pages_fall_back_to_built_in(tmp_path):
    cache = tmp_path / "prize_rules.json"
    site = prize_site(H.toto_prize_structure_html(rendered=False), H.fourd_prize_structure_html(rendered=False))
    rules = load_prize_rules(site, cache_path=cache, now=NOW)
    assert not rules.toto_confirmed and not rules.fourd_confirmed
    assert rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT and rules.fourd_prizes == C.FOURD_PRIZES
    assert "JavaScript" in rules.source_note and "built in" in rules.source_note
    assert not DASHES.search(rules.source_note)
    assert cache.exists()  # both pages were fetched, so no need to ask again for a week


def test_one_game_confirmed_the_other_built_in(tmp_path):
    site = prize_site("fixture", H.fourd_prize_structure_html(rendered=False))
    rules = load_prize_rules(site, cache_path=tmp_path / "c.json", now=NOW)
    assert rules.toto_confirmed and not rules.fourd_confirmed and not rules.confirmed
    assert "TOTO prize percentages confirmed" in rules.source_note
    assert "4D prize table is not in the official page text" in rules.source_note


def test_parsed_values_override_built_in(tmp_path):
    toto = H.toto_prize_structure_html(group_pool_pct={1: 0.38, 2: 0.08, 3: 0.06, 4: 0.03})
    big = {"first": 2500.0, "second": 1000.0, "third": 490.0, "starter": 250.0, "consolation": 60.0}
    fourd = H.fourd_prize_structure_html(big=big, ibet=False)
    rules = load_prize_rules(prize_site(toto, fourd), cache_path=None, now=NOW)
    assert rules.group_pool_pct[3] == 0.06
    assert rules.fourd_prizes["big"]["first"] == 2500.0
    assert rules.ibet_prizes == {}
    assert "Group 3 is 6% (built in 5.5%)" in rules.source_note
    assert "4D Big first is $2,500 (built in $2,000)" in rules.source_note
    assert not DASHES.search(rules.source_note)


def test_missing_figures_are_named_as_built_in(tmp_path):
    toto = H.toto_prize_structure_html(pool_share=None)
    rules = load_prize_rules(prize_site(toto, "fixture"), cache_path=None, now=NOW)
    assert rules.toto_confirmed
    assert "the 54% share of sales is built in" in rules.source_note
    assert rules.pool_share_of_sales == C.TOTO_POOL_SHARE_OF_SALES


def test_network_failure_uses_built_in_and_does_not_cache(tmp_path):
    cache = tmp_path / "prize_rules.json"
    rules = load_prize_rules(H.FakeFetcher({}), cache_path=cache, now=NOW)
    assert not rules.toto_confirmed and not rules.fourd_confirmed
    assert "could not be fetched" in rules.source_note
    assert not cache.exists()  # retried on the next run


def test_network_failure_falls_back_to_stale_confirmed_cache(tmp_path):
    cache = tmp_path / "prize_rules.json"
    load_prize_rules(prize_site(), cache_path=cache, now=NOW)
    rules = load_prize_rules(H.FakeFetcher({}), cache_path=cache, now=NOW + timedelta(days=30))
    assert rules.toto_confirmed and rules.fourd_confirmed
    assert rules.ibet_prizes["big"][4]["first"] == 500.0
    assert "last successful check on Fri 2 Oct 2026" in rules.source_note


class ExplodingFetcher:
    def get(self, url):
        raise RuntimeError("boom")

    def get_many(self, urls):
        raise RuntimeError("boom")


@pytest.mark.parametrize("fetcher", [ExplodingFetcher(), prize_site(FetchError("down"), RuntimeError("weird"))])
def test_never_raises(tmp_path, fetcher):
    rules = load_prize_rules(fetcher, cache_path=tmp_path / "x.json", now=NOW)
    assert isinstance(rules, PrizeRules)
    assert rules.group_pool_pct == C.TOTO_GROUP_POOL_PCT


def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "prize_rules.json"
    cache.write_text("{not json")
    rules = load_prize_rules(prize_site(), cache_path=cache, now=NOW)
    assert rules.confirmed
    assert json.loads(cache.read_text())["fourd_confirmed"] is True


def test_unwritable_cache_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    rules = load_prize_rules(prize_site(), cache_path=blocker / "sub" / "prize_rules.json", now=NOW)
    assert rules.confirmed


def test_rules_dict_round_trip():
    rules = PrizeRules(
        group_pool_pct={1: 0.37, 2: 0.08, 3: 0.06, 4: 0.03},
        ibet_prizes={"big": {24: {"first": 83.0}}, "small": {4: {"first": 750.0}}},
        toto_confirmed=True, source_note="checked", checked_at=NOW.isoformat(),
    )
    data = rules_to_dict(rules)
    restored = rules_from_dict(json.loads(json.dumps(data)))
    assert restored == rules
    assert restored.ibet_prizes["big"][24]["first"] == 83.0


def test_rules_from_partial_dict_uses_defaults():
    assert rules_from_dict({}) == PrizeRules()
    r = rules_from_dict({"fixed_prizes": {"5": 60}, "unknown": 1})
    assert r.fixed_prizes == {5: 60.0} and r.group_pool_pct == C.TOTO_GROUP_POOL_PCT
