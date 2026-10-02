"""Optional short commentary written by ``claude -p`` from figures the Python code computed.

Claude never computes anything here. It gets a small JSON of figures that are already
formatted for the user, and its reply is only accepted if every number in it appears in those
figures (``validate_commentary``). Any failure (command missing, timeout, invented number,
empty reply) means no commentary, never a failed run.

Environment: COMMENTARY ("off" by default, or "claude") and CLAUDE_BIN (default "claude").
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

from .textfmt import fmt_date, fmt_datetime, money, per_dollar, remove_dashes, toto_nums

log = logging.getLogger(__name__)

DEFAULT_MODE = "off"
OFF_MODES = ("", "off", "none", "no", "false", "0", "disabled")
MAX_WORDS = 60
MAX_CHARS = 400
DEFAULT_TIMEOUT = 120

# Numbers a short commentary may always use without them being in the figures:
# "per $1" and "Group 1" are phrases, not computed results.
ALWAYS_ALLOWED = frozenset({Decimal(1)})

PROMPT_TEMPLATE = """You write a short commentary for a Singapore Pools TOTO and 4D update posted to Telegram.

Rules:
1. At most {max_words} words in one short paragraph of plain sentences. No markdown, no lists, no headings, no emojis.
2. Use only the figures in the JSON below, written exactly as they appear there. Do not add, round, convert or calculate any number.
3. Never use dashes or hyphens of any kind. Write negative amounts as "minus $20" and ranges with the word "to".
4. Every draw is independent, so past results do not change the odds. Never suggest a strategy beats the odds and never suggest spending above the budget.
5. Reply with the commentary text only.

Figures (JSON):
{figures}
"""


# Figures


def _s(value: Any) -> str:
    """A string figure, dash free so the model is not tempted to copy dashes."""
    return remove_dashes(str(value)).strip()


def _int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _money_auto(value: Any) -> str | None:
    """Whole dollars unless the amount has cents."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return money(f, cents=not f.is_integer())


def _compact(obj: Any) -> Any:
    """Drop None, empty strings, empty lists and empty dicts, recursively."""
    if isinstance(obj, dict):
        out = {k: _compact(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        return [x for x in (_compact(v) for v in obj) if x not in (None, "", [], {})]
    return obj


def _latest_row(df: Any) -> Any:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "draw_number" not in df.columns:
        return None
    return df.loc[df["draw_number"].astype("int64").idxmax()]


def _next_toto(ctx: Any) -> dict:
    nt = getattr(ctx, "next_toto", None)
    if nt is None:
        return {}
    dt = getattr(nt, "draw_datetime", None)
    jackpot = getattr(nt, "jackpot_estimate", None)
    return {
        "draw_time": fmt_datetime(dt) if dt is not None else None,
        "estimated_jackpot": money(jackpot) if jackpot is not None else None,
        "draw_type": _s(getattr(nt, "draw_type", "") or ""),
    }


def _buy_signal(ctx: Any) -> dict:
    bs = getattr(ctx, "buy_signal", None)
    if bs is None:
        return {}
    ev = getattr(bs, "ev_per_dollar", None)
    return {
        "label": _s(getattr(bs, "label", "") or ""),
        "return_per_dollar": per_dollar(ev) if ev is not None else None,
        "draws_with_no_group_1_winner": _int(getattr(bs, "no_winner_streak", None)),
    }


def _next_fourd(ctx: Any) -> dict:
    nf = getattr(ctx, "next_fourd", None)
    dt = getattr(nf, "draw_datetime", None) if nf is not None else None
    return {"draw_time": fmt_datetime(dt)} if dt is not None else {}


def _latest_toto(ctx: Any) -> dict:
    row = _latest_row(getattr(ctx, "toto", None))
    if row is None:
        return {}
    winners = _int(row.get("g1_winners"))
    return {
        "draw": _int(row.get("draw_number")),
        "date": fmt_date(row.get("draw_date")),
        "numbers": toto_nums([row.get(f"n{i}") for i in range(1, 7)]),
        "additional": _int(row.get("additional")),
        "group_1_prize": _money_auto(row.get("jackpot")),
        "group_1_winners": winners,
    }


def _latest_fourd(ctx: Any) -> dict:
    row = _latest_row(getattr(ctx, "fourd", None))
    if row is None:
        return {}
    return {
        "draw": _int(row.get("draw_number")),
        "date": fmt_date(row.get("draw_date")),
        "first": _s(row.get("first") or ""),
        "second": _s(row.get("second") or ""),
        "third": _s(row.get("third") or ""),
    }


def _line_label(line: Any) -> str:
    """ "Low Crowd (Ordinary)", "Hot Digits (Big)", or just "System 7" when the bet type repeats it."""
    label, bet = _s(getattr(line, "label", "")), _s(getattr(line, "bet_type", "") or "")
    return f"{label} ({bet})" if bet and bet != label else label


def _plan(plan: Any, picks: Iterable[Any]) -> dict:
    if plan is not None:
        lines = list(getattr(plan, "lines", []) or [])
        alternative = getattr(plan, "alternative", None)
        return {
            "buy": [_line_label(line) for line in lines],
            "total_cost": _money_auto(getattr(plan, "total", None)),
            "budget": _money_auto(getattr(plan, "budget", None)),
            "alternative_total_cost": _money_auto(alternative.total) if alternative is not None else None,
        }
    names = [_s(getattr(p, "name", "")) for p in picks or []]
    return {"strategies": names}


def _backtest(result: Any) -> dict:
    if result is None:
        return {}
    scores = {}
    for sc in getattr(result, "scores", []) or []:
        scores[_s(sc.name)] = _compact({
            "verdict": _s(sc.verdict or ""),
            "return_per_dollar": per_dollar(sc.return_per_dollar),
        })
    return {"draws_tested": _int(getattr(result, "draws_tested", None)), "strategies": scores}


def _ledger(ctx: Any) -> dict:
    totals = getattr(ctx, "ledger_totals", None) or {}
    tickets = _int(totals.get("tickets"))
    spent = totals.get("spent")
    if not totals or (not tickets and not spent):
        return {}  # no tickets yet: nothing worth commenting on
    return {
        "tickets": tickets,
        "spent": _money_auto(spent),
        "won": _money_auto(totals.get("won")),
        "net": _money_auto(totals.get("net")),
    }


def _json_safe(obj: Any) -> Any:
    """Plain JSON types only (numpy scalars and other odd values become int, float or str)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def figures_from_context(ctx: Any) -> dict:
    """A small JSON safe dict of computed figures for the commentary, formatted as the user sees them.

    Covers the next TOTO draw (jackpot, type), buy signal (label, return per $1, no winner
    streak), next 4D draw, latest results for the games reported this run, the suggested
    purchases, backtest verdicts and ledger totals. Missing parts of the context are skipped;
    a broken part is logged and skipped rather than raised.
    """
    games = tuple(getattr(ctx, "games_drawn", ("toto", "4d")) or ())
    sections: list[tuple[str, Callable[[], dict]]] = [
        ("next_toto", lambda: _next_toto(ctx)),
        ("buy_signal", lambda: _buy_signal(ctx)),
        ("next_4d", lambda: _next_fourd(ctx)),
        ("latest_toto", lambda: _latest_toto(ctx) if "toto" in games else {}),
        ("latest_4d", lambda: _latest_fourd(ctx) if "4d" in games else {}),
        ("toto_suggestions", lambda: _plan(getattr(ctx, "toto_plan", None), getattr(ctx, "toto_picks", []))),
        ("4d_suggestions", lambda: _plan(getattr(ctx, "fourd_plan", None), getattr(ctx, "fourd_picks", []))),
        ("toto_backtest", lambda: _backtest(getattr(ctx, "toto_backtest", None))),
        ("4d_backtest", lambda: _backtest(getattr(ctx, "fourd_backtest", None))),
        ("my_tickets", lambda: _ledger(ctx)),
    ]
    figures: dict[str, Any] = {}
    for key, build in sections:
        try:
            part = _compact(build())
        except Exception as exc:  # one bad part must not lose the rest
            log.warning("Commentary figures: skipped %s (%s)", key, type(exc).__name__)
            continue
        if part:
            figures[key] = part
    return _json_safe(figures)


# Number checking

# Times like "6.30pm", "6:30 pm" or "6 p.m." are normalised to "6.30pm" before tokenising.
_TIME = re.compile(r"(?<![\w.:])(\d{1,2})(?:[.:](\d{2}))?\s*([ap])\.?m\.?(?![a-z])", re.I)
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?([A-Za-z]*)")
_SCALE_WORD = re.compile(r"\s+(million|thousand|billion)\b", re.I)
_SCALES = {
    "k": 3, "thousand": 3,
    "m": 6, "mil": 6, "mn": 6, "million": 6,
    "b": 9, "bn": 9, "billion": 9,
}


def _normalise_times(text: str) -> str:
    return _TIME.sub(lambda m: f"{int(m.group(1))}.{m.group(2) or '00'}{m.group(3).lower()}m", text)


def number_tokens(text: str) -> list[Decimal | str]:
    """Every number in the text as a Decimal (thousands separators ignored), times as "6.30pm".

    "$3.5 million" counts as 3500000, "1st" as 1, and the "4" in "4D" is the game name, not a number.
    """
    tokens: list[Decimal | str] = []
    text = _normalise_times(text)
    for m in _NUMBER.finditer(text):
        whole, frac, suffix = m.group(1), m.group(2) or "", m.group(3).lower()
        if suffix in ("am", "pm"):
            tokens.append(f"{whole}{frac}{suffix}")
            continue
        if suffix == "d" and whole == "4" and not frac:
            continue  # the game "4D"
        try:
            value = Decimal(whole.replace(",", "") + frac)
        except InvalidOperation:
            continue
        scale = _SCALES.get(suffix)
        if scale is None and not suffix:
            word = _SCALE_WORD.match(text, m.end())
            scale = _SCALES[word.group(1).lower()] if word else None
        if scale is not None:
            value = value.scaleb(scale)
        tokens.append(value)
    return tokens


def _walk_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_values(v)
    else:
        yield obj


def allowed_numbers(figures: dict) -> set[Decimal | str]:
    """Every number (and time) the commentary may mention: those in the figure values, plus 1.

    For a figure below 10 the value times 100 is allowed too, so "$0.66" may be written as
    "66 cents" and a 0.054 ratio as "5.4%". Raw floats may also appear rounded to 0 to 2 places.
    """
    allowed: set[Decimal | str] = set(ALWAYS_ALLOWED)

    def add(value: Decimal) -> None:
        allowed.add(value)
        if 0 < abs(value) < 10:
            allowed.add(value * 100)

    for v in _walk_values(figures):
        if v is None or isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (int, np.integer)):
            add(Decimal(int(v)))
        elif isinstance(v, (float, np.floating)):
            if np.isfinite(v):
                exact = Decimal(repr(float(v)))
                add(exact)
                for places in (0, 1, 2):
                    add(round(exact, places))
        else:
            for tok in number_tokens(str(v)):
                if isinstance(tok, Decimal):
                    add(tok)
                else:
                    allowed.add(tok)
    return allowed


# Cleaning and validation


def _clean(text: str) -> str:
    """Plain single paragraph: no code fences, markdown emphasis, headings or wrapping quotes."""
    t = text.strip()
    t = re.sub(r"^```[^\n]*\n?|\n?```\s*$", "", t)  # a reply wrapped in a code fence
    t = re.sub(r"^\s{0,3}#{1,6}\s+", "", t, flags=re.M)  # headings
    t = re.sub(r"(\*+|__|`)", "", t)  # emphasis and code marks
    t = " ".join(t.split())
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def _cap(text: str, limit: int = MAX_CHARS) -> str:
    """At most ``limit`` characters, cut at a sentence end if one is near, else at a word."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if cut[-1] in ".!?" and (limit >= len(text) or text[limit] == " "):
        end = limit - 1
    if end >= limit // 2:
        return cut[: end + 1].strip()
    words = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return words + "…"


def validate_commentary(text: str, figures: dict) -> str | None:
    """Accept a model reply only if it is safe to show.

    Steps: strip (and drop markdown), remove dashes, reject (None) if it holds any number not
    present in the figures (thousands separators and $ ignored), then cap at 400 characters.
    Empty or wordless replies are rejected too.
    """
    if not isinstance(text, str):
        return None
    t = _clean(text)
    t = " ".join(remove_dashes(t).split())
    if not t or not re.search(r"[A-Za-z]{2,}", t):
        return None
    allowed = allowed_numbers(figures or {})
    invented = [tok for tok in number_tokens(t) if tok not in allowed]
    if invented:
        log.warning("Commentary rejected, it has numbers not in the figures: %s", ", ".join(map(str, invented[:5])))
        return None
    return _cap(t, MAX_CHARS)


# Running claude


def build_prompt(figures: dict) -> str:
    """The prompt sent to ``claude -p``: the rules plus the figures as JSON."""
    return PROMPT_TEMPLATE.format(
        max_words=MAX_WORDS,
        figures=json.dumps(figures, indent=2, ensure_ascii=False, default=str),
    )


def _mode(mode: str | None) -> str:
    if mode is None:
        mode = os.environ.get("COMMENTARY", DEFAULT_MODE)
    return (mode or "").strip().lower()


def build_commentary(
    figures: dict,
    mode: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    """Commentary text, or None when it is off or anything goes wrong (the reason is logged).

    mode None reads env COMMENTARY (default "off"). "claude" runs
    ``runner([CLAUDE_BIN or "claude", "-p", prompt], capture_output=True, text=True, timeout=timeout)``
    and passes the reply through ``validate_commentary``.
    """
    m = _mode(mode)
    if m in OFF_MODES:
        return None
    if m != "claude":
        log.warning("Unknown COMMENTARY mode %r (use off or claude), no commentary this run", m)
        return None
    if not figures:
        log.info("No figures to comment on, commentary skipped")
        return None

    claude_bin = (os.environ.get("CLAUDE_BIN") or "").strip() or "claude"
    prompt = build_prompt(figures)
    try:
        proc = runner([claude_bin, "-p", prompt], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("Commentary skipped: %s took longer than %s seconds", claude_bin, timeout)
        return None
    except FileNotFoundError:
        log.warning("Commentary skipped: command %r not found (set CLAUDE_BIN)", claude_bin)
        return None
    except Exception as exc:  # any other failure to run must not stop the bot
        log.warning("Commentary skipped: running %s failed (%s)", claude_bin, type(exc).__name__)
        return None

    code = getattr(proc, "returncode", 0)
    if code:
        err = (getattr(proc, "stderr", "") or "").strip().splitlines()
        log.warning("Commentary skipped: %s exited with code %s %s", claude_bin, code, err[-1][:200] if err else "")
        return None
    out = getattr(proc, "stdout", "") or ""
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    result = validate_commentary(out, figures)
    if result is None:
        log.warning("Commentary from %s was not usable and was dropped", claude_bin)
    return result
