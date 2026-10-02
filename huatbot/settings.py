"""User settings, kept as note properties in ``Settings.md`` inside the Obsidian vault.

The user edits the properties in Obsidian (on the NAS, a phone or a PC). Every run reads
them with ``load_settings``. Reading is forgiving: money may be written as ``10``, ``$10``,
``3,000,000`` or ``3m``; yes/no settings accept true, false, yes, no, on and off; property
names are matched ignoring case and spaces. Anything that cannot be used falls back to the
default, out of range values are clamped, and every such problem becomes a plain English
line in ``Settings.warnings`` (no dashes, so it can go straight into a report or message).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, fields
from typing import Any

from . import constants as C
from .models import Settings
from .vault import render_note

log = logging.getLogger(__name__)

SETTINGS_NOTE = "Settings.md"


@dataclass(frozen=True)
class FieldSpec:
    """How to read one setting and what range makes sense for it."""

    kind: str  # "money", "int" or "bool"
    low: float | None = None
    high: float | None = None
    help: str = ""
    commas: bool = True  # show 1,000 rather than 1000 (off for draw numbers)


# Every user facing Settings field, in the order they appear in Settings.md.
FIELD_SPECS: dict[str, FieldSpec] = {
    "toto_budget": FieldSpec(
        "money", 0, 100_000,
        "The most to spend on TOTO for each draw. The suggested sets never cost more than this."),
    "fourd_budget": FieldSpec(
        "money", 0, 100_000,
        "The most to spend on 4D for each draw. The suggested numbers never cost more than this."),
    "jackpot_alert": FieldSpec(
        "money", 0, None,
        "A TOTO jackpot estimate at or above this amount makes the buy signal HIGH."),
    "alert_on_special_draws": FieldSpec(
        "bool", help="When true, any cascade, Hongbao or special TOTO draw also makes the buy signal HIGH."),
    "toto_start_draw": FieldSpec(
        "int", C.TOTO_FIRST_CURRENT_FORMAT_DRAW, 99_999,
        "The first TOTO draw number kept in the history. The lowest allowed is the first draw of "
        "the current 6 from 49 format.", commas=False),
    "fourd_history_draws": FieldSpec(
        "int", 10, 10_000, "How many of the latest 4D draws to keep and analyse."),
    "backtest_draws": FieldSpec(
        "int", 10, 2_000, "How many recent draws each backtest replays."),
    "random_sets_per_draw": FieldSpec(
        "int", 10, 20_000,
        "How many random sets (TOTO) or random numbers (4D) each backtest compares against for every draw. "
        "More is steadier but slower."),
    "offer_system7": FieldSpec(
        "bool", help="When true, the TOTO suggestions include a System 7 option whenever the budget allows."),
    "draw_notes_backfill": FieldSpec(
        "int", 0, 5_000,
        "How many recent draws of each game get their own note the first time the bot fills this vault."),
}

# Properties Obsidian and its plugins commonly add. They are allowed and silently ignored.
OBSIDIAN_KEYS = frozenset({"tags", "tag", "aliases", "alias", "cssclasses", "cssclass", "publish", "permalink"})

# Friendly spellings people are likely to type, mapped to the real field name.
KEY_ALIASES = {
    "4d_budget": "fourd_budget",
    "4d_history_draws": "fourd_history_draws",
    "budget_toto": "toto_budget",
    "budget_4d": "fourd_budget",
    "system7": "offer_system7",
    "offer_system_7": "offer_system7",
    "alert_on_special_draw": "alert_on_special_draws",
}

_TRUE = frozenset({"true", "yes", "y", "on", "1"})
_FALSE = frozenset({"false", "no", "n", "off", "0"})
_MULTIPLIERS = {
    "": 1.0, "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mil": 1e6, "mln": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}
_NUMBER_RE = re.compile(r"^([+-]?)(\d+(?:\.\d*)?|\.\d+)([a-z]*)$")


# Formatting (kept local: plain, no dashes)


def _money(x: float) -> str:
    """$10, $2.50, $3,000,000; negatives as "minus $5"."""
    sign = "minus " if x < 0 else ""
    x = abs(x)
    text = f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"
    return f"{sign}${text}"


def _show(name: str, value: Any) -> str:
    """A setting value the way the user would write it."""
    spec = FIELD_SPECS.get(name)
    if isinstance(value, bool):
        return "true" if value else "false"
    if spec is not None and spec.kind == "money":
        return _money(float(value))
    if isinstance(value, (int, float)):
        sign = "minus " if value < 0 else ""
        v = abs(value)
        if spec is not None and not spec.commas:
            return f"{sign}{v:g}" if isinstance(v, float) else f"{sign}{v}"
        return f"{sign}{v:,}" if isinstance(v, int) else f"{sign}{v:,g}"
    return str(value)


# Coercion


def parse_number(value: Any) -> float | None:
    """A number from a property value: 10, 2.5, "$10", "S$ 2.50", "3,000,000", "3m", "1.5k",
    "minus 5". Returns None when it is not a number (booleans are not numbers)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    for word in ("sgd", "s$", "$", "dollars", "dollar"):
        s = s.replace(word, "")
    s = s.replace(",", "").replace("_", "").replace(" ", "").replace("−", "-")
    if s.startswith("minus"):
        s = "-" + s[len("minus"):]
    if not s:
        return None
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except ValueError:
        pass
    m = _NUMBER_RE.match(s)
    if not m or m.group(3) not in _MULTIPLIERS:
        return None
    x = float(m.group(2)) * _MULTIPLIERS[m.group(3)]
    return -x if m.group(1) == "-" else x


def parse_bool(value: Any) -> bool | None:
    """True/False from true, false, yes, no, on, off, y, n, 1, 0 (any case). None when unclear."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def normalise_key(key: Any) -> str:
    """Match property names loosely: "TOTO budget" and "toto-budget" both mean toto_budget."""
    k = re.sub(r"[\s\-]+", "_", str(key).strip().lower())
    return KEY_ALIASES.get(k, k)


def _spec_for(name: str, default: Any) -> FieldSpec:
    """Spec of a field; a Settings field added later without a spec is read by its default's type."""
    if name in FIELD_SPECS:
        return FIELD_SPECS[name]
    if isinstance(default, bool):
        return FieldSpec("bool")
    if isinstance(default, int):
        return FieldSpec("int")
    if isinstance(default, float):
        return FieldSpec("money")
    return FieldSpec("text")


def _coerce(name: str, raw: Any, default: Any, warnings: list[str]) -> Any:
    """Turn one raw property value into a usable setting, appending plain warnings."""
    spec = _spec_for(name, default)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        warnings.append(f"{name} is empty, so the default {_show(name, default)} is used.")
        return default

    if spec.kind == "bool":
        b = parse_bool(raw)
        if b is None:
            warnings.append(f"{name} should be true or false, so the default {_show(name, default)} is used.")
            return default
        return b

    if spec.kind == "text":
        return str(raw)

    x = parse_number(raw)
    if x is None:
        warnings.append(f"{name} is not a number, so the default {_show(name, default)} is used.")
        return default

    if spec.kind == "int" and not float(x).is_integer():
        rounded = int(round(x))
        warnings.append(f"{name} should be a whole number, so it was rounded to {_show(name, rounded)}.")
        x = rounded

    if spec.low is not None and x < spec.low:
        clamped = spec.low
        warnings.append(
            f"{name} was {_show(name, x)}, below the lowest allowed {_show(name, clamped)}, "
            f"so {_show(name, clamped)} is used.")
        x = clamped
    elif spec.high is not None and x > spec.high:
        clamped = spec.high
        warnings.append(
            f"{name} was {_show(name, x)}, above the highest allowed {_show(name, clamped)}, "
            f"so {_show(name, clamped)} is used.")
        x = clamped

    return int(x) if spec.kind == "int" else float(x)


def settings_from_mapping(data: dict | None) -> Settings:
    """Build Settings from a properties dict (Settings.md frontmatter).

    Unknown keys are ignored with a warning (Obsidian's own tags, aliases and cssclasses are
    ignored quietly). Bad values fall back to the default and out of range values are clamped,
    each with a warning in ``Settings.warnings``.
    """
    defaults = Settings()
    warnings: list[str] = []
    if data is None:
        data = {}
    if not isinstance(data, dict):
        warnings.append("The settings are not a list of name and value pairs, so the defaults are used.")
        data = {}

    names = [f.name for f in fields(Settings) if f.name != "warnings"]
    values: dict[str, Any] = {}
    for key, raw in data.items():
        name = normalise_key(key)
        if name in OBSIDIAN_KEYS:
            continue
        if name not in names:
            shown = re.sub(r"[\-–—]+", " ", str(key)).strip()
            warnings.append(f'Unknown setting "{shown}" was ignored.')
            continue
        if name in values:
            warnings.append(f"{name} is set more than once, so the last value is used.")
        values[name] = _coerce(name, raw, getattr(defaults, name), warnings)

    settings = Settings(**values)
    settings.warnings = warnings
    return settings


def load_settings(vault) -> Settings:
    """Settings from the vault's Settings.md properties. A missing or unreadable note gives the
    defaults plus a warning explaining why; every warning is also logged."""
    try:
        text = vault.read_text(SETTINGS_NOTE)
    except OSError as exc:
        log.warning("Could not read %s: %s", SETTINGS_NOTE, exc)
        s = Settings()
        s.warnings.append(f"{SETTINGS_NOTE} could not be read, so the default settings are used.")
        return s

    if text is None:
        s = Settings()
        s.warnings.append(f"{SETTINGS_NOTE} was not found, so the default settings are used.")
        return s

    frontmatter, _ = vault.read_note(SETTINGS_NOTE)
    s = settings_from_mapping(frontmatter)
    if not frontmatter:
        if text.lstrip("﻿").startswith("---"):
            s.warnings.insert(0, f"The properties in {SETTINGS_NOTE} could not be read, so the default settings are used.")
        else:
            s.warnings.insert(0, f"{SETTINGS_NOTE} has no properties, so the default settings are used.")
    for w in s.warnings:
        log.warning("Settings: %s", w)
    return s


# Settings.md starter note


def _template_value(name: str, value: Any) -> Any:
    """Defaults as the user would type them: whole dollar amounts as integers."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def build_settings_template() -> str:
    """Settings.md text: every setting as a property with its default, then plain help.

    The defaults and ranges in the help come from ``models.Settings`` and ``FIELD_SPECS``,
    so the note can never disagree with the code.
    """
    defaults = Settings()
    names = [f.name for f in fields(Settings) if f.name != "warnings"]
    frontmatter = {n: _template_value(n, getattr(defaults, n)) for n in names}

    lines = [
        "# Huat Bot settings",
        "",
        "Change the properties at the top of this note to tune the bot. It reads them at the start "
        "of every run, so a change applies from the next run. If a value cannot be used, the bot "
        "falls back to the default and lists the problem in its warnings.",
        "",
        "Money can be written as 10, $10 or 3,000,000. Yes or no settings take true or false "
        "(yes and no work too).",
        "",
        "## What each property does",
        "",
    ]
    for n in names:
        spec = _spec_for(n, getattr(defaults, n))
        line = f"* **{n}**: {spec.help}".rstrip() if spec.help else f"* **{n}**:"
        line += f" Default {_show(n, getattr(defaults, n))}."
        if spec.kind in ("int", "money") and spec.low is not None and spec.high is not None:
            line += f" Allowed from {_show(n, spec.low)} to {_show(n, spec.high)}."
        elif spec.kind in ("int", "money") and spec.low is not None:
            line += f" Lowest allowed {_show(n, spec.low)}."
        lines.append(line)
    lines += [
        "",
        "Your tickets go in [[Tickets]], not here. Every draw is independent, so no setting "
        "changes the odds of winning.",
    ]
    return render_note("\n".join(lines), frontmatter)


SETTINGS_TEMPLATE: str = build_settings_template()
