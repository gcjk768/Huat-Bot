"""Settings: the Settings.md template, forgiving coercion, clamping and loading from the vault."""
from __future__ import annotations

import logging
from dataclasses import asdict, fields

import pytest

from huatbot import constants as C
from huatbot.models import Settings
from huatbot.settings import (
    FIELD_SPECS,
    KEY_ALIASES,
    OBSIDIAN_KEYS,
    RETIRED_KEYS,
    SETTINGS_TEMPLATE,
    load_settings,
    normalise_key,
    parse_bool,
    parse_number,
    settings_from_mapping,
)
from huatbot.vault import Vault, parse_note

SETTING_NAMES = [f.name for f in fields(Settings) if f.name != "warnings"]
DASHES = ("-", "–", "—")
RETIRED_NOTE = "These settings are no longer used and can be deleted from Settings.md: "


def _values(s: Settings) -> dict:
    d = asdict(s)
    d.pop("warnings")
    return d


def _no_dashes(text: str) -> bool:
    return not any(ch in text for ch in DASHES)


# Template


def test_every_settings_field_has_a_spec():
    assert list(FIELD_SPECS) == SETTING_NAMES
    assert SETTING_NAMES == ["jackpot_alert", "alert_on_special_draws", "toto_start_draw", "draw_notes_backfill"]


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
    assert SETTINGS_TEMPLATE.startswith(
        "---\njackpot_alert: 3000000\nalert_on_special_draws: true\n"
        f"toto_start_draw: {C.TOTO_FIRST_CURRENT_FORMAT_DRAW}\ndraw_notes_backfill: 50\n---\n")


def test_template_help_is_plain_and_complete():
    _, body = parse_note(SETTINGS_TEMPLATE)
    assert body.startswith("# Huat Bot settings\n")
    assert "the properties at the top of this note" in body
    for name in SETTING_NAMES:
        assert f"**{name}**" in body, name
    assert _no_dashes(body), [line for line in body.splitlines() if not _no_dashes(line)]
    # Defaults and ranges in the help come from the code.
    assert "Default $3,000,000. Lowest allowed $0." in body
    assert "Default true." in body
    assert f"Default {C.TOTO_FIRST_CURRENT_FORMAT_DRAW}. Allowed from {C.TOTO_FIRST_CURRENT_FORMAT_DRAW} to 99999." in body
    assert "Default 50. Allowed from 0 to 5,000." in body


def test_template_mentions_no_removed_feature():
    lowered = SETTINGS_TEMPLATE.lower()
    for word in ("4d", "fourd", "budget", "backtest", "system 7", "system7", "suggest", "random"):
        assert word not in lowered, word
    for key in RETIRED_KEYS:
        assert key not in SETTINGS_TEMPLATE, key


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


def test_money_numbers_and_bools_are_coerced():
    s = settings_from_mapping({
        "jackpot_alert": "5,000,000",
        "alert_on_special_draws": "no",
        "toto_start_draw": 3000.0,
        "draw_notes_backfill": "1k",
    })
    assert s.warnings == []
    assert s.jackpot_alert == 5_000_000.0 and isinstance(s.jackpot_alert, float)
    assert s.alert_on_special_draws is False
    assert s.toto_start_draw == 3000 and isinstance(s.toto_start_draw, int)
    assert s.draw_notes_backfill == 1000 and isinstance(s.draw_notes_backfill, int)

    s = settings_from_mapping({"jackpot_alert": "$2.5 million", "alert_on_special_draws": "on"})
    assert s.warnings == []
    assert s.jackpot_alert == 2_500_000.0 and s.alert_on_special_draws is True


def test_missing_keys_keep_defaults():
    s = settings_from_mapping({"jackpot_alert": "4m"})
    assert s.jackpot_alert == 4_000_000.0
    defaults = Settings()
    assert (s.alert_on_special_draws, s.toto_start_draw, s.draw_notes_backfill) == (
        defaults.alert_on_special_draws, defaults.toto_start_draw, defaults.draw_notes_backfill)
    assert s.warnings == []
    assert settings_from_mapping({}).warnings == []
    assert settings_from_mapping(None).warnings == []


@pytest.mark.parametrize("key,raw,expected,fragment", [
    ("jackpot_alert", -5, 0.0, "below the lowest allowed $0"),
    ("jackpot_alert", "-5", 0.0, "minus $5"),
    ("jackpot_alert", "minus 1m", 0.0, "minus $1,000,000"),
    ("toto_start_draw", 100, C.TOTO_FIRST_CURRENT_FORMAT_DRAW, "lowest allowed 2995"),
    ("toto_start_draw", 123_456, 99_999, "toto_start_draw was 123456, above the highest allowed 99999"),
    ("draw_notes_backfill", -1, 0, "minus 1"),
    ("draw_notes_backfill", 9000, 5000, "above the highest allowed 5,000"),
])
def test_nonsense_is_clamped_with_a_warning(key, raw, expected, fragment):
    s = settings_from_mapping({key: raw})
    assert getattr(s, key) == expected
    assert type(getattr(s, key)) is type(expected)
    assert len(s.warnings) == 1 and fragment in s.warnings[0]
    assert _no_dashes(s.warnings[0])


def test_fractional_whole_number_settings_are_rounded():
    s = settings_from_mapping({"draw_notes_backfill": 12.6})
    assert s.draw_notes_backfill == 13
    assert "rounded to 13" in s.warnings[0]
    s = settings_from_mapping({"toto_start_draw": 3000.4})
    assert s.toto_start_draw == 3000
    assert s.warnings == ["toto_start_draw should be a whole number, so it was rounded to 3000."]


@pytest.mark.parametrize("key,raw", [
    ("jackpot_alert", "lots"), ("toto_start_draw", "many"), ("alert_on_special_draws", "maybe"),
    ("jackpot_alert", None), ("alert_on_special_draws", ""), ("jackpot_alert", [1, 2]),
    ("draw_notes_backfill", None),
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
    s = settings_from_mapping({"Jackpot Alert": "4m", "alert-on-special-draws": "off", "TOTO start draw": 3100,
                               "Draw notes backfill": 20})
    assert s.warnings == []
    assert (s.jackpot_alert, s.alert_on_special_draws, s.toto_start_draw, s.draw_notes_backfill) == (
        4_000_000.0, False, 3100, 20)
    # A friendly singular spelling still finds the real setting.
    s = settings_from_mapping({"Alert on special draw": "no"})
    assert s.warnings == [] and s.alert_on_special_draws is False


def test_duplicate_keys_warn():
    s = settings_from_mapping({"jackpot_alert": 1_000_000, "Jackpot alert": 2_000_000})
    assert s.jackpot_alert == 2_000_000.0
    assert any("more than once" in w for w in s.warnings)


def test_non_mapping_gives_defaults_with_warning():
    s = settings_from_mapping(["jackpot_alert", 5])  # type: ignore[arg-type]
    assert _values(s) == _values(Settings())
    assert len(s.warnings) == 1


def test_warnings_list_is_not_shared_between_instances():
    a = settings_from_mapping({"jackpot_alert": -1})
    b = settings_from_mapping({})
    assert a.warnings and b.warnings == []
    assert Settings().warnings == []


# Settings of earlier versions (budgets, backtests, System 7 and 4D are gone)


def test_retired_keys_can_match_and_do_not_clash_with_current_settings():
    for key in RETIRED_KEYS:
        assert normalise_key(key) == key, key  # written the way a property name is normalised
    assert RETIRED_KEYS.isdisjoint(SETTING_NAMES)
    assert RETIRED_KEYS.isdisjoint(KEY_ALIASES) and RETIRED_KEYS.isdisjoint(KEY_ALIASES.values())
    assert RETIRED_KEYS.isdisjoint(OBSIDIAN_KEYS)
    for old in ("toto_budget", "fourd_budget", "fourd_history_draws", "backtest_draws", "random_sets_per_draw",
                "offer_system7"):
        assert old in RETIRED_KEYS, old
        assert not hasattr(Settings(), old), old


@pytest.mark.parametrize("key", sorted(RETIRED_KEYS))
def test_each_retired_setting_gives_one_gentle_note(key):
    s = settings_from_mapping({key: 10})
    assert _values(s) == _values(Settings())
    assert s.warnings == [f"{RETIRED_NOTE}{key}."]
    assert _no_dashes(s.warnings[0])


def test_retired_settings_are_listed_once_in_one_warning():
    s = settings_from_mapping({"toto_budget": 10, "fourd_budget": 5, "jackpot_alert": 4_000_000,
                               "offer_system7": True, "TOTO budget": 12})
    assert s.jackpot_alert == 4_000_000.0
    assert s.warnings == [f"{RETIRED_NOTE}fourd_budget, offer_system7, toto_budget."]
    assert not any("Unknown setting" in w or "more than once" in w for w in s.warnings)


def test_retired_settings_are_matched_loosely_and_their_values_are_not_checked():
    s = settings_from_mapping({"TOTO budget": "$20", "4D budget": 3, "Offer System 7": "maybe",
                               "backtest-draws": "many", "Random sets per draw": None, "budget 4D": "lots"})
    assert _values(s) == _values(Settings())
    assert s.warnings == [
        f"{RETIRED_NOTE}4d_budget, backtest_draws, budget_4d, offer_system_7, random_sets_per_draw, toto_budget."]
    assert _no_dashes(s.warnings[0])


def test_retired_note_comes_after_the_other_warnings():
    s = settings_from_mapping({"toto_budget": 10, "budget_for_fun": 1, "jackpot_alert": "lots",
                               "backtest_draws": 300})
    assert s.warnings == [
        'Unknown setting "budget_for_fun" was ignored.',
        "jackpot_alert is not a number, so the default $3,000,000 is used.",
        f"{RETIRED_NOTE}backtest_draws, toto_budget.",
    ]


def test_settings_note_of_an_earlier_version_still_loads(tmp_path, caplog):
    # Settings.md as an earlier version wrote it, with the user's own changes to the kept settings.
    v = Vault(tmp_path)
    v.write_text("Settings.md", (
        "---\n"
        "toto_budget: 10\n"
        "fourd_budget: 5\n"
        "jackpot_alert: 5000000\n"
        "alert_on_special_draws: false\n"
        "toto_start_draw: 3000\n"
        "fourd_history_draws: 1000\n"
        "backtest_draws: 300\n"
        "random_sets_per_draw: 1000\n"
        "offer_system7: true\n"
        "draw_notes_backfill: 20\n"
        "---\n"
        "# Huat Bot settings\n"
    ))
    with caplog.at_level(logging.WARNING):
        s = load_settings(v)
    assert (s.jackpot_alert, s.alert_on_special_draws, s.toto_start_draw, s.draw_notes_backfill) == (
        5_000_000.0, False, 3000, 20)
    assert s.warnings == [f"{RETIRED_NOTE}backtest_draws, fourd_budget, fourd_history_draws, offer_system7, "
                          "random_sets_per_draw, toto_budget."]
    assert "no longer used" in caplog.text


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
    text = text.replace("jackpot_alert: 3000000\n", "jackpot_alert: $5m\n")
    text = text.replace("alert_on_special_draws: true\n", "alert_on_special_draws: no\n")
    text = text.replace("draw_notes_backfill: 50\n", "draw_notes_backfill: 99999\n")
    v.write_text("Settings.md", text)
    s = load_settings(v)
    assert s.jackpot_alert == 5_000_000.0 and s.alert_on_special_draws is False and s.draw_notes_backfill == 5000
    assert len(s.warnings) == 1 and "draw_notes_backfill" in s.warnings[0]


def test_load_settings_missing_note(tmp_path):
    s = load_settings(Vault(tmp_path))
    assert _values(s) == _values(Settings())
    assert s.warnings == ["Settings.md was not found, so the default settings are used."]


def test_load_settings_bad_yaml(tmp_path, caplog):
    v = Vault(tmp_path)
    v.write_text("Settings.md", "---\njackpot_alert: [oops\n---\n# Settings\n")
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
