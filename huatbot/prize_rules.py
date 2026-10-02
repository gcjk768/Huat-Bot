"""The TOTO prize rules in force: built in values, confirmed against the official page when possible.

``load_prize_rules`` never raises. It returns the built in values from constants.py
unless the official online2 TOTO prize structure page can be fetched and read, in which
case the figures read from the page win. The page may be drawn by JavaScript, in which
case the figures are not in the HTML and the built in values are used; the
``source_note`` says plainly which figures were confirmed and which are built in.
Results are cached as JSON (Data/prize_rules.json in the vault) for ``max_age_days``.

A cache written by the older TOTO and 4D version still loads: its 4D keys are ignored and
the 4D parts of its note are dropped.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import constants as C
from .models import PrizeRules
from .parse import parse_toto_prize_structure
from .store import atomic_write_text
from .textfmt import fmt_date, money, pct

log = logging.getLogger(__name__)
SG = ZoneInfo(C.SG_TZ_NAME)

CACHE_VERSION = 2  # version 1 also held the 4D prize tables; those keys are ignored when read


def pct_text(x: float) -> str:
    """A prize percentage without trailing zeros: 0.055 -> "5.5%", 0.38 -> "38%"."""
    text = pct(x, digits=2)
    return text[:-1].rstrip("0").rstrip(".") + "%" if "." in text else text


# serialisation


def rules_to_dict(r: PrizeRules) -> dict:
    """JSON safe dict (integer keys become strings)."""
    return {
        "version": CACHE_VERSION,
        "pool_share_of_sales": float(r.pool_share_of_sales),
        "group_pool_pct": {str(k): float(v) for k, v in r.group_pool_pct.items()},
        "fixed_prizes": {str(k): float(v) for k, v in r.fixed_prizes.items()},
        "min_group1": float(r.min_group1),
        "toto_confirmed": bool(r.toto_confirmed),
        "source_note": str(r.source_note),
        "checked_at": r.checked_at,
    }


def _int_keys(d: Any) -> dict[int, float]:
    return {int(k): float(v) for k, v in dict(d).items()}


def _float(d: dict, key: str, default: float) -> float:
    value = d.get(key)
    return default if value is None else float(value)


_SENTENCE_BREAK = re.compile(r"(?<=\.)\s+")


def _without_fourd(note: str) -> str:
    """The note without the 4D sentences (and 4D items of the "values that differ" list) that a
    cache from the TOTO and 4D version holds. A TOTO only note comes back unchanged."""
    if "4D" not in note:
        return note
    kept = []
    for sentence in _SENTENCE_BREAK.split(note.strip()):
        if "4D" not in sentence:
            kept.append(sentence)
            continue
        head, sep, items = sentence.partition(": ")
        toto = [i for i in items.rstrip(".").split("; ") if "4D" not in i] if sep and "4D" not in head else []
        if toto:
            kept.append(f"{head}: {'; '.join(toto)}.")
    return " ".join(kept)


def rules_from_dict(d: dict) -> PrizeRules:
    """Inverse of rules_to_dict. Missing keys take the built in values; unknown keys (such as the
    4D tables of a version 1 cache) are ignored."""
    base = PrizeRules()
    d = d or {}
    return PrizeRules(
        pool_share_of_sales=_float(d, "pool_share_of_sales", base.pool_share_of_sales),
        group_pool_pct=_int_keys(d["group_pool_pct"]) if d.get("group_pool_pct") else base.group_pool_pct,
        fixed_prizes=_int_keys(d["fixed_prizes"]) if d.get("fixed_prizes") else base.fixed_prizes,
        min_group1=_float(d, "min_group1", base.min_group1),
        toto_confirmed=bool(d.get("toto_confirmed", False)),
        source_note=_without_fourd(str(d.get("source_note") or "")) or base.source_note,
        checked_at=d.get("checked_at"),
    )


# helpers


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(SG)
    return now if now.tzinfo else now.replace(tzinfo=SG)


def _parse_when(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        when = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=SG)


def _read_cache(path: Path | None) -> PrizeRules | None:
    if path is None:
        return None
    try:
        path = Path(path)
        if not path.exists():
            return None
        return rules_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        log.warning("Prize rules cache %s could not be read, ignoring it: %s", path, exc)
        return None


def _write_cache(path: Path | None, rules: PrizeRules) -> None:
    if path is None:
        return
    try:
        atomic_write_text(Path(path), json.dumps(rules_to_dict(rules), indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        log.warning("Prize rules cache %s could not be written: %s", path, exc)


def _is_fresh(rules: PrizeRules, now: datetime, max_age_days: int) -> bool:
    when = _parse_when(rules.checked_at)
    if when is None:
        return False
    age = now - when
    return timedelta(days=-1) <= age < timedelta(days=max(0, max_age_days))


def _differences(label: str, old: dict, new: dict, fmt) -> list[str]:
    return [f"{label} {k} is {fmt(new[k])} (built in {fmt(old[k])})"
            for k in new if k in old and abs(float(new[k]) - float(old[k])) > 1e-9]


def _apply(rules: PrizeRules, parsed: dict) -> tuple[list[str], list[str]]:
    """Copy the figures read from the official page into ``rules`` (built in values so far) and
    set ``toto_confirmed``. Returns (figures still built in, figures that differ from built in)."""
    built_in = PrizeRules()
    group_pct = {g: float(parsed["group_pool_pct"][g]) for g in (1, 2, 3, 4)}
    fixed = {int(g): float(v) for g, v in (parsed.get("fixed_prizes") or {}).items()}
    changes = _differences("Group", built_in.group_pool_pct, group_pct, pct_text)
    changes += _differences("Group", built_in.fixed_prizes, fixed, money)
    rules.group_pool_pct = group_pct
    rules.fixed_prizes = {**built_in.fixed_prizes, **fixed}

    unread: list[str] = []
    missing_fixed = [g for g in (5, 6, 7) if g not in fixed]
    if missing_fixed:
        unread.append("the fixed prizes for Group " + " and ".join(str(g) for g in missing_fixed))
    share = parsed.get("pool_share_of_sales")
    if share is None:
        unread.append(f"the {pct_text(built_in.pool_share_of_sales)} share of sales")
    else:
        if abs(share - built_in.pool_share_of_sales) > 1e-9:
            changes.append(f"prize pool is {pct_text(share)} of sales "
                           f"(built in {pct_text(built_in.pool_share_of_sales)})")
        rules.pool_share_of_sales = float(share)
    minimum = parsed.get("min_group1")
    if minimum is None:
        unread.append(f"the {money(built_in.min_group1)} Group 1 minimum")
    else:
        if abs(minimum - built_in.min_group1) > 1e-9:
            changes.append(f"Group 1 minimum is {money(minimum)} (built in {money(built_in.min_group1)})")
        rules.min_group1 = float(minimum)
    rules.toto_confirmed = not missing_fixed
    return unread, changes


def _built_in(reason: str) -> PrizeRules:
    return PrizeRules(source_note=f"Built in prize values are used ({reason}).")


# main entry point


def load_prize_rules(fetcher=None, cache_path: Path | None = None, max_age_days: int = 7,
                     now: datetime | None = None) -> PrizeRules:
    """TOTO prize rules for this run. Never raises (any failure falls back and is logged)."""
    try:
        return _load(fetcher, cache_path, max_age_days, _now(now))
    except Exception:  # last line of defence: a run must never die over prize rules
        log.exception("Prize rules could not be loaded, using the built in values")
        return _built_in("the official prize page could not be checked")


def _load(fetcher, cache_path: Path | None, max_age_days: int, now: datetime) -> PrizeRules:
    if fetcher is None:
        return _built_in("the official prize page was not checked this run")

    cached = _read_cache(cache_path)
    if cached is not None and _is_fresh(cached, now, max_age_days):
        log.info("Using cached prize rules checked at %s", cached.checked_at)
        return cached

    try:
        page = fetcher.get(C.TOTO_PRIZE_RULES_URL)
    except Exception as exc:
        log.warning("TOTO prize structure page could not be fetched: %s", exc)
        page = None
    fetched = isinstance(page, str)
    parsed = None
    if fetched:
        try:
            parsed = parse_toto_prize_structure(page)
        except Exception:
            log.exception("TOTO prize structure page could not be parsed")

    rules = PrizeRules(checked_at=now.isoformat(timespec="seconds"))
    changes: list[str] = []
    if parsed is not None:
        unread, changes = _apply(rules, parsed)
        note = f"TOTO prize percentages confirmed from the official page on {fmt_date(now)}"
        if unread:
            note += f"; {', '.join(unread)} {'is' if len(unread) == 1 else 'are'} built in"
        note += "."
    elif not fetched and cached is not None and cached.toto_confirmed:
        rules = replace(cached, checked_at=rules.checked_at)
        when = _parse_when(cached.checked_at)
        note = ("TOTO prize rules are from the last successful check" + (f" on {fmt_date(when)}" if when else "")
                + " (the official page could not be fetched today).")
    elif fetched:
        note = ("TOTO prize figures are not in the official page text (it is probably drawn by JavaScript), "
                "so the built in values are used.")
    else:
        note = "The official TOTO prize page could not be fetched, so the built in values are used."
    if changes:
        note += " Official values that differ from the built in ones: " + "; ".join(changes) + "."
    rules.source_note = note

    # Cache only a completed check, so a network hiccup is retried on the next run.
    if fetched:
        _write_cache(cache_path, rules)
    return rules
