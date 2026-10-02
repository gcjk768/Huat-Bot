"""The prize rules in force: built in values, confirmed against the official pages when possible.

``load_prize_rules`` never raises. It returns the built in values from constants.py
unless the official online2 prize structure pages can be fetched and read, in which
case the figures read from the pages win. The pages may be drawn by JavaScript, in
which case the figures are not in the HTML and the built in values are used; the
``source_note`` says plainly which figures were confirmed and which are built in.
Results are cached as JSON (Data/prize_rules.json in the vault) for ``max_age_days``.
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import constants as C
from .models import PrizeRules
from .parse import parse_fourd_prize_structure, parse_toto_prize_structure
from .store import atomic_write_text

log = logging.getLogger(__name__)
SG = ZoneInfo(C.SG_TZ_NAME)

CACHE_VERSION = 1
_TIERS = ("first", "second", "third", "starter", "consolation")


# serialisation


def rules_to_dict(r: PrizeRules) -> dict:
    """JSON safe dict (integer keys become strings)."""
    return {
        "version": CACHE_VERSION,
        "pool_share_of_sales": float(r.pool_share_of_sales),
        "group_pool_pct": {str(k): float(v) for k, v in r.group_pool_pct.items()},
        "fixed_prizes": {str(k): float(v) for k, v in r.fixed_prizes.items()},
        "min_group1": float(r.min_group1),
        "fourd_prizes": {bet: {t: float(v) for t, v in tiers.items()} for bet, tiers in r.fourd_prizes.items()},
        "ibet_prizes": {
            bet: {str(perm): {t: float(v) for t, v in tiers.items()} for perm, tiers in perms.items()}
            for bet, perms in r.ibet_prizes.items()
        },
        "toto_confirmed": bool(r.toto_confirmed),
        "fourd_confirmed": bool(r.fourd_confirmed),
        "source_note": str(r.source_note),
        "checked_at": r.checked_at,
    }


def _int_keys(d: Any) -> dict[int, float]:
    return {int(k): float(v) for k, v in dict(d).items()}


def rules_from_dict(d: dict) -> PrizeRules:
    """Inverse of rules_to_dict. Missing keys take the built in values; unknown keys are ignored."""
    base = PrizeRules()
    d = d or {}
    return PrizeRules(
        pool_share_of_sales=float(d.get("pool_share_of_sales", base.pool_share_of_sales)),
        group_pool_pct=_int_keys(d["group_pool_pct"]) if d.get("group_pool_pct") else base.group_pool_pct,
        fixed_prizes=_int_keys(d["fixed_prizes"]) if d.get("fixed_prizes") else base.fixed_prizes,
        min_group1=float(d.get("min_group1", base.min_group1)),
        fourd_prizes=(
            {bet: {t: float(v) for t, v in tiers.items()} for bet, tiers in d["fourd_prizes"].items()}
            if d.get("fourd_prizes") else base.fourd_prizes
        ),
        ibet_prizes={
            bet: {int(perm): {t: float(v) for t, v in tiers.items()} for perm, tiers in perms.items()}
            for bet, perms in (d.get("ibet_prizes") or {}).items()
        },
        toto_confirmed=bool(d.get("toto_confirmed", False)),
        fourd_confirmed=bool(d.get("fourd_confirmed", False)),
        source_note=str(d.get("source_note") or base.source_note),
        checked_at=d.get("checked_at"),
    )


# helpers


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(SG)
    return now if now.tzinfo else now.replace(tzinfo=SG)


def _fmt_date(d: datetime) -> str:
    return f"{d:%a} {d.day} {d:%b %Y}"


def _pct(x: float) -> str:
    return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _money(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


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


def _apply_toto(rules: PrizeRules, parsed: dict) -> tuple[bool, list[str], list[str]]:
    """Copy parsed TOTO figures into rules. Returns (confirmed, still built in parts, changes)."""
    built_in = PrizeRules()
    changes = _differences("Group", built_in.group_pool_pct, parsed["group_pool_pct"], _pct)
    rules.group_pool_pct = {g: float(parsed["group_pool_pct"][g]) for g in (1, 2, 3, 4)}
    fixed = dict(built_in.fixed_prizes)
    fixed.update({int(g): float(v) for g, v in parsed.get("fixed_prizes", {}).items()})
    changes += _differences("Group", built_in.fixed_prizes, parsed.get("fixed_prizes", {}), _money)
    rules.fixed_prizes = fixed
    unread: list[str] = []
    missing_fixed = [g for g in (5, 6, 7) if g not in parsed.get("fixed_prizes", {})]
    if missing_fixed:
        unread.append("the fixed prizes for Group " + " and ".join(str(g) for g in missing_fixed))
    if parsed.get("pool_share_of_sales") is not None:
        if abs(parsed["pool_share_of_sales"] - built_in.pool_share_of_sales) > 1e-9:
            changes.append(f"prize pool is {_pct(parsed['pool_share_of_sales'])} of sales "
                           f"(built in {_pct(built_in.pool_share_of_sales)})")
        rules.pool_share_of_sales = float(parsed["pool_share_of_sales"])
    else:
        unread.append(f"the {_pct(built_in.pool_share_of_sales)} share of sales")
    if parsed.get("min_group1") is not None:
        rules.min_group1 = float(parsed["min_group1"])
    else:
        unread.append(f"the {_money(built_in.min_group1)} Group 1 minimum")
    return not missing_fixed, unread, changes


def _apply_fourd(rules: PrizeRules, parsed: dict) -> list[str]:
    built_in = PrizeRules()
    changes = []
    for bet in ("big", "small"):
        changes += _differences(f"4D {bet.title()}", built_in.fourd_prizes[bet], parsed[bet], _money)
    rules.fourd_prizes = {bet: {t: float(v) for t, v in parsed[bet].items()} for bet in ("big", "small")}
    rules.ibet_prizes = {
        bet: {int(perm): {t: float(v) for t, v in tiers.items()} for perm, tiers in perms.items()}
        for bet, perms in (parsed.get("ibet") or {}).items()
    }
    return changes


def _copy_game(target: PrizeRules, source: PrizeRules, game: str) -> None:
    if game == "toto":
        target.pool_share_of_sales = source.pool_share_of_sales
        target.group_pool_pct = dict(source.group_pool_pct)
        target.fixed_prizes = dict(source.fixed_prizes)
        target.min_group1 = source.min_group1
        target.toto_confirmed = source.toto_confirmed
    else:
        target.fourd_prizes = copy.deepcopy(source.fourd_prizes)
        target.ibet_prizes = copy.deepcopy(source.ibet_prizes)
        target.fourd_confirmed = source.fourd_confirmed


# main entry point


def load_prize_rules(fetcher=None, cache_path: Path | None = None, max_age_days: int = 7,
                     now: datetime | None = None) -> PrizeRules:
    """Prize rules for this run. Never raises (any failure falls back and is logged)."""
    try:
        return _load(fetcher, cache_path, max_age_days, _now(now))
    except Exception:  # last line of defence: a run must never die over prize rules
        log.exception("Prize rules could not be loaded, using the built in values")
        rules = PrizeRules()
        rules.source_note = "Built in prize values are used (the official prize pages could not be checked)."
        return rules


def _load(fetcher, cache_path: Path | None, max_age_days: int, now: datetime) -> PrizeRules:
    if fetcher is None:
        rules = PrizeRules()
        rules.source_note = "Built in prize values are used (the official prize pages were not checked this run)."
        return rules

    cached = _read_cache(cache_path)
    if cached is not None and _is_fresh(cached, now, max_age_days):
        log.info("Using cached prize rules checked at %s", cached.checked_at)
        return cached

    urls = [C.TOTO_PRIZE_RULES_URL, C.FOURD_PRIZE_RULES_URL]
    try:
        pages = fetcher.get_many(urls)
    except Exception as exc:
        log.warning("Prize structure pages could not be fetched: %s", exc)
        pages = {}

    rules = PrizeRules()
    rules.checked_at = now.isoformat(timespec="seconds")
    today = _fmt_date(now)
    notes: list[str] = []
    changes: list[str] = []
    fetched_both = True

    # TOTO
    page = pages.get(C.TOTO_PRIZE_RULES_URL)
    parsed = None
    if isinstance(page, str):
        try:
            parsed = parse_toto_prize_structure(page)
        except Exception:
            log.exception("TOTO prize structure page could not be parsed")
    else:
        fetched_both = False
        log.warning("TOTO prize structure page could not be fetched: %s", page)
    if parsed is not None:
        confirmed, unread, toto_changes = _apply_toto(rules, parsed)
        changes += toto_changes
        rules.toto_confirmed = confirmed
        note = f"TOTO prize percentages confirmed from the official page on {today}"
        if unread:
            note += f"; {', '.join(unread)} {'is' if len(unread) == 1 else 'are'} built in"
        notes.append(note + ".")
    elif not isinstance(page, str) and cached is not None and cached.toto_confirmed:
        _copy_game(rules, cached, "toto")
        when = _parse_when(cached.checked_at)
        notes.append("TOTO prize rules are from the last successful check"
                     + (f" on {_fmt_date(when)}" if when else "") + " (the official page could not be fetched today).")
    elif isinstance(page, str):
        notes.append("TOTO prize figures are not in the official page text (it is probably drawn by JavaScript), "
                     "so the built in values are used.")
    else:
        notes.append("The official TOTO prize page could not be fetched, so the built in values are used.")

    # 4D
    page = pages.get(C.FOURD_PRIZE_RULES_URL)
    parsed = None
    if isinstance(page, str):
        try:
            parsed = parse_fourd_prize_structure(page)
        except Exception:
            log.exception("4D prize structure page could not be parsed")
    else:
        fetched_both = False
        log.warning("4D prize structure page could not be fetched: %s", page)
    if parsed is not None:
        changes += _apply_fourd(rules, parsed)
        rules.fourd_confirmed = True
        tables = "Big, Small and iBet" if rules.ibet_prizes else "Big and Small"
        notes.append(f"4D {tables} prize tables confirmed from the official page on {today}.")
    elif not isinstance(page, str) and cached is not None and cached.fourd_confirmed:
        _copy_game(rules, cached, "4d")
        when = _parse_when(cached.checked_at)
        notes.append("4D prize table is from the last successful check"
                     + (f" on {_fmt_date(when)}" if when else "") + " (the official page could not be fetched today).")
    elif isinstance(page, str):
        notes.append("4D prize table is not in the official page text (it is probably drawn by JavaScript), "
                     "so the built in values are used.")
    else:
        notes.append("The official 4D prize page could not be fetched, so the built in values are used.")

    if changes:
        notes.append("Official values that differ from the built in ones: " + "; ".join(changes) + ".")
    rules.source_note = " ".join(notes)

    # Cache only a complete check, so a network hiccup is retried on the next run.
    if fetched_both:
        _write_cache(cache_path, rules)
    return rules
