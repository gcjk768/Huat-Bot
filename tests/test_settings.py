"""Settings: the Settings.md template, forgiving coercion, clamping and loading from the vault."""
from __future__ import annotations

import logging
from dataclasses import asdict, fields

import pytest

from huatbot import constants as C
from huatbot.models import Settings
from huatbot.settings import (
    FIELD_SPECS,
    SETTINGS_TEMPLATE,
    load_settings,
    parse_bool,
    parse_number,
    settings_from_mapping,
)
from huatbot.vault import Vault, parse_note

SETTING_NAMES = [f.name for f in fields(Settings) if f.name != "warnings"]
DASHES = ("-", "–", "—")


def _values(s: Settings) -> dict:
    d = asdict(s)
    d.pop("warnings")
    return d


def _no_dashes(text: str) -> bool:
    return not any(ch in text for ch in DASHES)


# Template


def test_every_settings_field_has_a_spec():
    assert list(FIELD_SPECS) == SETTING_NAMES


def test_template_frontmatter_has_every_field_with_defaults():
    fm, body = parse_note(SETTINGS_TEMPLATE)
    assert list(fm) == SETTING_NAMES
    defaults = Settings()
    for name in SETTING_NAMES:
        assert fm[name] == getattr(defaults, name), name
    s = settings_from_mapping(fm)
    assert s.warnings == []
    assert _values(s) == _values(defaults)


def test_template_frontmatter_is_block_style_yaml():
    assert SETTINGS_TEMPLATE.startswith("---\ntoto_budget: 10\nfourd_budget: 5\njackpot_alert: 3000000\n")
    assert "alert_on_special_draws: true\n" in SETTINGS_TEMPLATE


def test_template_help_is_plain_and_complete():
    _, body = parse_note(SETTINGS_TEMPLATE)
    assert body.startswith("# Huat Bot settings\n")
    assert "the properties at the top of this note" in body
    for name in SETTING_NAMES:
        assert f"**{name}**" in body, name
    assert _no_dashes(body), [line for line in body.splitlines() if not _no_dashes(line)]
    # Defaults and ranges in the help come from the code.
    assert "Default $10." in body and "Default $3,000,000." in body
    assert f"Default {C.TOTO_FIRST_CURRENT_FORMAT_DRAW}." in body
    assert "Allowed from 10 to 2,000." in body


# Coercion


@pytest.mark.parametrize("raw,expected", [
    (10, 10.0), (2.5, 2.5), ("10", 10.0), ("$10", 10.0), ("S$ 12.50", 12.5), ("SGD 7", 7.0),
    ("3,000,000", 3_000_000.0), ("$3,000,000", 3_000_000.0), ("3m", 3_000_000.0), ("2.5 million", 2_500_000.0),
    ("1.5k", 1500.0), ("minus 5", -5.0), ("-5", -5.0), ("$-5", -5.0), ("1e6", 1_000_000.0), ("10 dollars", 10.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, True, False, "", "abc", "nan", "inf", float("nan"), "10 apples", [1], "3x"])
def test_parse_number_rejects(raw):
    assert parse_number(raw) is None


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False), ("yes", True), ("No", False), ("TRUE", True), ("false", False),
    ("on", True), ("off", False), ("y", True), ("n", False), (1, True), (0, False), ("1", True),
])
def test_parse_bool(raw, expected):
    assert parse_bool(raw) is expected


@pytest.mark.parametrize("raw", ["maybe", 2, None, "", [True]])
def test_parse_bool_rejects(raw):
    assert parse_bool(raw) is None


def test_money_and_bools_are_coerced():
    s = settings_from_mapping({
        "toto_budget": "$20",
        "fourd_budget": 8,
        "jackpot_alert": "5,000,000",
        "alert_on_special_draws": "no",
        "offer_system7": "yes",
        "backtest_draws": "500",
        "random_sets_per_draw": "2k",
        "toto_start_draw": 3000.0,
    })
    assert s.warnings == []
    assert s.toto_budget == 20.0 and isinstance(s.toto_budget, float)
    assert s.fourd_budget == 8.0 and isinstance(s.fourd_budget, float)
    assert s.jackpot_alert == 5_000_000.0
    assert s.alert_on_special_draws is False and s.offer_system7 is True
    assert s.backtest_draws == 500 and isinstance(s.backtest_draws, int)
    assert s.random_sets_per_draw == 2000
    assert s.toto_start_draw == 3000 and isinstance(s.toto_start_draw, int)


def test_missing_keys_keep_defaults():
    s = settings_from_mapping({"toto_budget": 15})
    assert s.toto_budget == 15.0
    assert s.fourd_budget == Settings().fourd_budget
    assert s.warnings == []
    assert settings_from_mapping({}).warnings == []
    assert settings_from_mapping(None).warnings == []


@pytest.mark.parametrize("key,raw,expected,fragment", [
    ("toto_budget", -5, 0.0, "below the lowest allowed $0"),
    ("toto_budget", "-5", 0.0, "minus $5"),
    ("fourd_budget", 1_000_000, 100_000.0, "above the highest allowed $100,000"),
    ("backtest_draws", 5, 10, "below the lowest allowed 10"),
    ("backtest_draws", 5000, 2000, "above the highest allowed 2,000"),
    ("toto_start_draw", 100, C.TOTO_FIRST_CURRENT_FORMAT_DRAW, "lowest allowed 2995"),
    ("random_sets_per_draw", 0, 10, "lowest allowed 10"),
    ("draw_notes_backfill", -1, 0, "minus 1"),
    ("jackpot_alert", "minus 1m", 0.0, "minus $1,000,000"),
])
def test_nonsense_is_clamped_with_a_warning(key, raw, expected, fragment):
    s = settings_from_mapping({key: raw})
    assert getattr(s, key) == expected
    assert len(s.warnings) == 1 and fragment in s.warnings[0]
    assert _no_dashes(s.warnings[0])


def test_fractional_whole_number_settings_are_rounded():
    s = settings_from_mapping({"draw_notes_backfill": 12.6})
    assert s.draw_notes_backfill == 13
    assert "rounded to 13" in s.warnings[0]


@pytest.mark.parametrize("key,raw", [
    ("toto_budget", "lots"), ("backtest_draws", "many"), ("offer_system7", "maybe"),
    ("toto_budget", None), ("alert_on_special_draws", ""), ("jackpot_alert", [1, 2]),
])
def test_unreadable_values_fall_back_to_default(key, raw):
    s = settings_from_mapping({key: raw})
    assert getattr(s, key) == getattr(Settings(), key)
    assert len(s.warnings) == 1 and key in s.warnings[0] and "default" in s.warnings[0]
    assert _no_dashes(s.warnings[0])


def test_unknown_keys_warn_but_obsidian_keys_are_quiet():
    s = settings_from_mapping({"budget_for_fun": 1, "tags": ["huatbot"], "aliases": ["Config"], "cssclasses": []})
    assert s.warnings == ['Unknown setting "budget_for_fun" was ignored.']
    s = settings_from_mapping({"bad-key-name": 1})
    assert _no_dashes(s.warnings[0])


def test_keys_are_matched_loosely():
    s = settings_from_mapping({"TOTO budget": 12, "4D budget": 3, "Offer System 7": "no", "backtest-draws": 50})
    assert s.warnings == []
    assert (s.toto_budget, s.fourd_budget, s.offer_system7, s.backtest_draws) == (12.0, 3.0, False, 50)


def test_duplicate_keys_warn():
    s = settings_from_mapping({"toto_budget": 12, "TOTO budget": 14})
    assert s.toto_budget == 14.0
    assert any("more than once" in w for w in s.warnings)


def test_non_mapping_gives_defaults_with_warning():
    s = settings_from_mapping(["toto_budget", 5])  # type: ignore[arg-type]
    assert _values(s) == _values(Settings())
    assert len(s.warnings) == 1


def test_warnings_list_is_not_shared_between_instances():
    a = settings_from_mapping({"toto_budget": -1})
    b = settings_from_mapping({})
    assert a.warnings and b.warnings == []
    assert Settings().warnings == []


# Loading from the vault


def test_load_settings_from_template(tmp_path):
    v = Vault(tmp_path)
    v.ensure_layout({"Settings.md": SETTINGS_TEMPLATE})
    s = load_settings(v)
    assert s.warnings == []
    assert _values(s) == _values(Settings())


def test_load_settings_after_user_edits_in_obsidian(tmp_path):
    v = Vault(tmp_path)
    v.ensure_layout({"Settings.md": SETTINGS_TEMPLATE})
    text = v.read_text("Settings.md")
    text = text.replace("toto_budget: 10\n", "toto_budget: $25\n").replace("offer_system7: true\n", "offer_system7: no\n")
    text = text.replace("backtest_draws: 300\n", "backtest_draws: 99999\n")
    v.write_text("Settings.md", text)
    s = load_settings(v)
    assert s.toto_budget == 25.0 and s.offer_system7 is False and s.backtest_draws == 2000
    assert len(s.warnings) == 1 and "backtest_draws" in s.warnings[0]


def test_load_settings_missing_note(tmp_path):
    s = load_settings(Vault(tmp_path))
    assert _values(s) == _values(Settings())
    assert s.warnings == ["Settings.md was not found, so the default settings are used."]


def test_load_settings_bad_yaml(tmp_path, caplog):
    v = Vault(tmp_path)
    v.write_text("Settings.md", "---\ntoto_budget: [oops\n---\n# Settings\n")
    with caplog.at_level(logging.WARNING):
        s = load_settings(v)
    assert _values(s) == _values(Settings())
    assert s.warnings == ["The properties in Settings.md could not be read, so the default settings are used."]
    assert "Settings" in caplog.text


def test_load_settings_note_without_properties(tmp_path):
    v = Vault(tmp_path)
    v.write_text("Settings.md", "# Settings\n\nI deleted the properties.\n")
    s = load_settings(v)
    assert s.warnings == ["Settings.md has no properties, so the default settings are used."]


def test_load_settings_unreadable_file(tmp_path):
    class BrokenVault:
        def read_text(self, rel):
            raise PermissionError("denied")

    s = load_settings(BrokenVault())
    assert _values(s) == _values(Settings())
    assert s.warnings == ["Settings.md could not be read, so the default settings are used."]
