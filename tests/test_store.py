"""CSV store: round trips keep dtypes and NaN shares; writes are atomic and readable."""
from __future__ import annotations

import os
import stat

import numpy as np

from huatbot import store
from huatbot.synth import synth_toto


def test_toto_round_trip(tmp_path):
    df = synth_toto(n_draws=60)
    path = tmp_path / "toto.csv"
    store.save_toto(df, path)
    back = store.load_toto(path)
    assert list(back.columns) == list(df.columns)
    assert back["draw_number"].tolist() == df["draw_number"].tolist()
    assert (back["draw_date"] == df["draw_date"]).all()
    for col in ("g1_share", "g2_share"):
        assert np.array_equal(np.isnan(back[col]), np.isnan(df[col]))


def test_missing_file_gives_empty_frames(tmp_path):
    assert store.load_toto(tmp_path / "nope.csv").empty
    assert store.load_ledger(tmp_path / "nope.csv").empty


def test_atomic_write_is_readable_and_keeps_mode(tmp_path):
    path = tmp_path / "a.txt"
    store.atomic_write_text(path, "one")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    umask = os.umask(0)
    os.umask(umask)
    assert mode == 0o666 & ~umask  # not the 0600 that mkstemp would leave
    os.chmod(path, 0o640)
    store.atomic_write_text(path, "two")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    assert path.read_text() == "two"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_no_winner_streak():
    df = synth_toto(n_draws=80)
    df["g1_winners"] = 0
    df["draw_type"] = "normal"
    df.loc[df.index[-5], "g1_winners"] = 1
    assert store.no_winner_streak(df) == 4
    df.loc[df.index[-2], "draw_type"] = "cascade"
    assert store.no_winner_streak(df) == 4
    assert store.no_winner_streak(df, reset_on_cascade=True) == 1
