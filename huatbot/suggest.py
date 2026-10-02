"""Turn strategy picks into a purchase plan that never spends above the budget.

TOTO: $1 per ordinary set in priority order (Low Crowd, Balanced, Hot, Overdue), at most the four
suggested sets, plus a System 7 when the money left allows it. When a System 7 does not fit next
to the sets but does fit the budget, the plan carries an ``alternative`` plan built around it.

4D: an equal whole dollar stake on each pick, keeping each pick's own bet type.

Every amount in a plan comes from here; notes are plain sentences with no dashes.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from . import constants as C
from . import prizes
from .analysis_fourd import bet_type_value
from .models import FourDPick, Plan, PlanLine, PrizeRules, TotoPick
from .strategies import crowd_score_array, pattern_problems, system7_from
from .textfmt import fmt_num, money, per_dollar, plural, toto_nums

log = logging.getLogger(__name__)

TOTO_PRIORITY = ("Low Crowd", "Balanced", "Hot", "Overdue")
TOTO_MAX_SETS = 4
TOTO_SET_COST = C.TOTO_BOARD_COST  # $1 per ordinary set
SYSTEM7_SIZE = 7
SYSTEM7_COST = C.TOTO_SYSTEM_BOARDS[SYSTEM7_SIZE] * C.TOTO_BOARD_COST  # 7 boards at $1 = $7
FOURD_MIN_STAKE = 1  # whole dollars per 4D bet

NOTHING_TO_BUY = "Budget below $1, nothing to buy."


# Helpers


def _clean_budget(budget: Any) -> float:
    """A usable budget in dollars: missing, non numeric, negative or infinite -> 0."""
    try:
        value = float(budget)
    except (TypeError, ValueError):
        log.warning("budget %r is not a number, using $0", budget)
        return 0.0
    if not math.isfinite(value) or value < 0:
        log.warning("budget %r is not usable, using $0", budget)
        return 0.0
    return value


def _dollars(x: float) -> str:
    """ "$10" for whole dollars, "$2.50" otherwise."""
    return money(x, cents=not float(x).is_integer())


def _unspent(budget: float, total: float) -> float:
    left = budget - total
    return left if left >= 0.005 else 0.0


def _priority_order(picks: Sequence[TotoPick]) -> list[TotoPick]:
    """Picks in TOTO_PRIORITY order; any other names follow in their given order."""
    rank = {name: i for i, name in enumerate(TOTO_PRIORITY)}
    indexed = list(enumerate(picks))
    indexed.sort(key=lambda ip: (rank.get(ip[1].name, len(TOTO_PRIORITY)), ip[0]))
    return [p for _, p in indexed]


def _set_line(pick: TotoPick) -> PlanLine:
    return PlanLine(game="TOTO", label=pick.name, numbers=toto_nums(pick.numbers),
                    bet_type="Ordinary", cost=float(TOTO_SET_COST), reason=pick.reason)


def _system7_line(source: TotoPick, crowd_scores: Any, last_draw: Any) -> PlanLine:
    """System 7 line built from ``source``: its six numbers plus the least crowded safe extra."""
    numbers = system7_from(source, crowd_scores, last_draw)
    base = {int(x) for x in source.numbers}
    extra = next(n for n in numbers if n not in base)
    scores, available = crowd_score_array(crowd_scores)
    problems = pattern_problems(numbers, last_draw)
    if available:
        reason = (f"The {source.name} set plus {extra}, the least crowded number that keeps all 7 "
                  f"free of patterns (crowd score {fmt_num(scores[extra - 1], 2)})")
        if problems:
            reason = (f"The {source.name} set plus {extra}, the least crowded number with the fewest "
                      f"patterns (crowd score {fmt_num(scores[extra - 1], 2)}), but {problems[0]}")
    elif problems:
        reason = (f"The {source.name} set plus {extra} (crowd scores not available), but "
                  f"{problems[0]}")
    else:
        reason = (f"The {source.name} set plus {extra}, chosen to keep all 7 free of patterns "
                  "(crowd scores not available)")
    return PlanLine(game="TOTO", label="System 7", numbers=toto_nums(numbers),
                    bet_type="System 7", cost=float(SYSTEM7_COST), reason=reason)


def _system7_odds_note() -> str:
    boards = C.TOTO_SYSTEM_BOARDS[SYSTEM7_SIZE]
    return (f"A System 7 plays all {boards} boards inside its {SYSTEM7_SIZE} numbers, so it has "
            f"{boards} chances in {fmt_num(C.TOTO_COMBOS)} of Group 1 for {_dollars(SYSTEM7_COST)}, "
            f"the same odds as {boards} separate sets.")


# TOTO


def toto_plan(
    picks: Sequence[TotoPick],
    budget: float,
    crowd_scores: pd.Series | None = None,
    offer_system7: bool = True,
    last_draw: Any = None,
) -> Plan:
    """What to buy for one TOTO draw within ``budget``.

    $1 per set in TOTO_PRIORITY order while the budget allows (at most 4 sets). If
    ``offer_system7`` and at least $7 is left after the sets, a System 7 line is added (built
    with ``system7_from`` from the Low Crowd pick, or the top priority pick if there is no Low
    Crowd pick). Otherwise, if ``offer_system7`` and the budget is at least $7, ``alternative``
    is a plan with the System 7 plus as many other sets as fit; the set the System 7 was built
    from is left out there because its board is already one of the System 7's boards.
    ``plan.total`` (and ``alternative.total``) never exceed the budget.
    """
    budget = _clean_budget(budget)
    plan = Plan(game="TOTO", budget=budget)
    ordered = _priority_order(list(picks or []))
    if not ordered:
        plan.notes.append("No suggested sets are available, so there is nothing to buy.")
        return plan
    if budget < TOTO_SET_COST:
        plan.notes.append(NOTHING_TO_BUY)
        return plan

    n_sets = min(len(ordered), TOTO_MAX_SETS, math.floor(budget / TOTO_SET_COST))
    plan.lines = [_set_line(p) for p in ordered[:n_sets]]
    left = budget - plan.total
    source = next((p for p in ordered if p.name == "Low Crowd"), ordered[0])

    if offer_system7 and left >= SYSTEM7_COST:
        plan.lines.append(_system7_line(source, crowd_scores, last_draw))
        plan.notes.append(
            f"{plural(n_sets, 'set')} at $1 each plus a System 7 at {_dollars(SYSTEM7_COST)}: "
            f"{_dollars(plan.total)} of the {_dollars(budget)} budget."
        )
        if source in ordered[:n_sets]:
            plan.notes.append(f"The System 7 contains the {source.name} set as one of its boards, "
                              "so that board is played twice.")
        plan.notes.append(_system7_odds_note())
    else:
        plan.notes.append(f"{plural(n_sets, 'set')} at $1 each, in the order "
                          f"{', '.join(p.name for p in ordered[:n_sets])}: "
                          f"{_dollars(plan.total)} of the {_dollars(budget)} budget.")
        if offer_system7 and budget >= SYSTEM7_COST:
            plan.alternative = _toto_alternative(ordered, source, budget, crowd_scores, last_draw)
            n_other = len(plan.alternative.lines) - 1
            extra = f" plus {plural(n_other, 'other set')}" if n_other else ""
            plan.notes.append(
                f"A System 7 costs {_dollars(SYSTEM7_COST)} and does not fit next to the "
                f"{plural(n_sets, 'set')}, so the alternative is a System 7 built from the "
                f"{source.name} set{extra}, for {_dollars(plan.alternative.total)}."
            )
            plan.notes.append(_system7_odds_note())
        elif offer_system7:
            plan.notes.append(f"A System 7 costs {_dollars(SYSTEM7_COST)}, more than the "
                              f"{_dollars(budget)} budget, so it is not offered.")

    unspent = _unspent(budget, plan.total)
    if unspent:
        has_system7 = any(line.label == "System 7" for line in plan.lines)
        if plan.alternative is not None:
            why = "the alternative plan uses more of it"
        elif n_sets == min(len(ordered), TOTO_MAX_SETS) and unspent >= TOTO_SET_COST:
            why = ("every suggested set and the System 7 are already in the plan" if has_system7
                   else "every suggested set is already in the plan")
        else:
            why = "sets cost whole dollars"
        plan.notes.append(f"{_dollars(unspent)} of the budget is left unspent: {why}.")
    return plan


def _toto_alternative(ordered: list[TotoPick], source: TotoPick, budget: float,
                      crowd_scores: Any, last_draw: Any) -> Plan:
    """System 7 from ``source`` plus as many of the other priority sets as fit the budget."""
    alt = Plan(game="TOTO", budget=budget)
    alt.lines.append(_system7_line(source, crowd_scores, last_draw))
    others = [p for p in ordered if p is not source]
    n_sets = min(len(others), TOTO_MAX_SETS, math.floor((budget - SYSTEM7_COST) / TOTO_SET_COST))
    alt.lines.extend(_set_line(p) for p in others[:n_sets])
    if n_sets:
        alt.notes.append(
            f"System 7 built from the {source.name} set at {_dollars(SYSTEM7_COST)} plus "
            f"{plural(n_sets, 'set')} at $1 each ({', '.join(p.name for p in others[:n_sets])}): "
            f"{_dollars(alt.total)} of the {_dollars(budget)} budget."
        )
    else:
        alt.notes.append(f"System 7 built from the {source.name} set: {_dollars(alt.total)} of the "
                         f"{_dollars(budget)} budget.")
    alt.notes.append(f"The {source.name} set is not bought on its own here because it is already "
                     "one of the boards in the System 7.")
    unspent = _unspent(budget, alt.total)
    if unspent:
        alt.notes.append(f"{_dollars(unspent)} of the budget is left unspent.")
    return alt


# 4D


def _ibet_value(bet_values: dict, perms: int) -> float | None:
    """iBet Big return per $1 for ``perms`` permutations (keys may be ints or JSON strings)."""
    table = bet_values.get("iBet Big") or {}
    if not isinstance(table, dict):
        return None
    value = table.get(perms, table.get(str(perms)))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bet_notes(lines: list[PlanLine], bet_values: dict | None) -> list[str]:
    """Plain notes on the bet types, with the return per $1 from ``bet_values``.

    Without ``bet_values`` the figures come from the built in prize table (and say so).
    """
    source = ""
    if not bet_values:
        bet_values = bet_type_value(PrizeRules())
        source = " with the built in prize table"
    notes: list[str] = []
    big, small = bet_values.get("Big"), bet_values.get("Small")
    best = bet_values.get("best") or "Big"
    straight = [line for line in lines if not line.bet_type.lower().startswith("ibet")]
    if straight and big is not None:
        all_big = all(line.bet_type == "Big" for line in straight)
        if best == "Big":
            tail = ", so the straight picks are Big bets" if all_big else ""
            notes.append(f"Big returns the most per $1 ({per_dollar(big)} on average{source}, "
                         f"against {per_dollar(small)} for Small){tail}.")
        else:
            notes.append(f"{best} returns the most per $1{source}; Big returns {per_dollar(big)} and "
                         f"Small {per_dollar(small)} on average.")
    for line in lines:
        if not line.bet_type.lower().startswith("ibet"):
            continue
        perms = prizes.permutations_count(line.numbers)
        value = _ibet_value(bet_values, perms)
        figure = f", returning {per_dollar(value)} per $1 on average{source}" if value is not None else ""
        notes.append(f"The {line.label} pick {line.numbers} is an {line.bet_type} bet: the stake is "
                     f"spread over all {perms} orders of its digits{figure}.")
    return notes


def fourd_plan(picks: Sequence[FourDPick], budget: float, bet_values: dict | None = None) -> Plan:
    """What to buy for one 4D draw within ``budget``.

    Every pick gets the same whole dollar stake: floor(budget / number of picks), at least $1.
    When the budget is below one dollar per pick, the first floor(budget) picks get $1 each.
    Each line keeps the pick's bet type (Digit Set is usually iBet Big). ``plan.total`` never
    exceeds the budget. ``bet_values`` is ``analysis_fourd.bet_type_value(rules)``; its figures
    go into the notes.
    """
    budget = _clean_budget(budget)
    plan = Plan(game="4D", budget=budget)
    picks = list(picks or [])
    if not picks:
        plan.notes.append("No 4D picks are available, so there is nothing to buy.")
        return plan
    if budget < FOURD_MIN_STAKE:
        plan.notes.append(NOTHING_TO_BUY)
        return plan

    if budget >= FOURD_MIN_STAKE * len(picks):
        stake = math.floor(budget / len(picks))
        chosen = picks
    else:
        stake = FOURD_MIN_STAKE
        chosen = picks[: math.floor(budget / FOURD_MIN_STAKE)]
    plan.lines = [PlanLine(game="4D", label=p.name, numbers=p.number, bet_type=p.bet_type,
                           cost=float(stake), reason=p.reason) for p in chosen]

    plan.notes.append(f"{plural(len(chosen), 'number')} at {_dollars(stake)} each: "
                      f"{_dollars(plan.total)} of the {_dollars(budget)} budget.")
    if len(chosen) < len(picks):
        plan.notes.append(f"The {_dollars(budget)} budget covers the first {len(chosen)} of the "
                          f"{len(picks)} picks at the {_dollars(FOURD_MIN_STAKE)} minimum stake.")
    unspent = _unspent(budget, plan.total)
    if unspent:
        plan.notes.append(f"{_dollars(unspent)} is unspent, because every pick gets the same whole "
                          "dollar stake.")
    plan.notes.extend(_bet_notes(plan.lines, bet_values))
    return plan
