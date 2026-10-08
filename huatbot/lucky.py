"""A just for fun lucky pick for the next draw, weighted by how often each number came up before.

Every draw is independent, so this has no edge over any other set of six. It is a toy.
"""
from __future__ import annotations

import random
from collections import Counter

import pandas as pd

from .store import toto_numbers

RECENT = 100  # draws that count double, so the pick leans a little on the recent past
POOL = range(1, 50)


def lucky_pick(df: pd.DataFrame, seed: int) -> tuple[list[int], list[int]] | None:
    """(six numbers ascending, the three hottest numbers) from the stored draws, or None without history.

    Weight = 1 + times drawn, recent draws counted twice. Sampling is seeded by the draw number,
    so the same draw always gets the same pick but each draw gets a new one.
    """
    if df is None or df.empty:
        return None
    rows = df.sort_values("draw_number")
    counts: Counter[int] = Counter()
    for i, (_, r) in enumerate(rows.iterrows()):
        counts.update(toto_numbers(r) * (2 if i >= len(rows) - RECENT else 1))
    weights = {n: 1 + counts[n] for n in POOL}
    rng, pick = random.Random(seed), []
    while len(pick) < 6:  # weighted draw without replacement
        n = rng.choices(list(weights), list(weights.values()))[0]
        pick.append(n)
        del weights[n]
    hot = [n for n, _ in counts.most_common(3)]
    return sorted(pick), sorted(hot)
