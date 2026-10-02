"""CSV storage for TOTO draws and the ticket ledger. All writes are atomic."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .models import LEDGER_COLUMNS, TOTO_COLUMNS

_TOTO_INT = ["draw_number"] + [f"n{i}" for i in range(1, 7)] + ["additional"] + [
    f"g{g}_winners" for g in range(1, 8)
]
_TOTO_FLOAT = ["jackpot"] + [f"g{g}_share" for g in range(1, 8)]


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Write via a temp file in the same folder, then rename, so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates 0600 files; keep the old file's mode, or use the normal umask for a new
    # one, so other NAS users, SMB clients and Obsidian sync can still read the vault.
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def empty_toto() -> pd.DataFrame:
    return normalise_toto(pd.DataFrame(columns=TOTO_COLUMNS))


def empty_ledger() -> pd.DataFrame:
    return normalise_ledger(pd.DataFrame(columns=LEDGER_COLUMNS))


def normalise_toto(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, add missing columns, drop duplicate draws, sort ascending."""
    df = df.copy()
    for col in TOTO_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[TOTO_COLUMNS]
    for col in _TOTO_INT:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    for col in _TOTO_FLOAT:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce").dt.normalize()
    df["draw_type"] = df["draw_type"].fillna("normal").astype(str).replace({"": "normal", "nan": "normal"})
    df["fetched_at"] = df["fetched_at"].fillna("").astype(str).replace({"nan": ""})
    df = df.drop_duplicates("draw_number", keep="last").sort_values("draw_number").reset_index(drop=True)
    return df


def normalise_ledger(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[LEDGER_COLUMNS]
    for col in ("cost", "units", "winnings"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
    df["draw_number"] = pd.to_numeric(df["draw_number"], errors="coerce").astype("Int64")
    for col in ("ticket_id", "game", "draw_date", "numbers", "bet_type", "status", "result",
                "added_at", "checked_at", "source"):
        df[col] = df[col].fillna("").astype(str).replace({"nan": ""}).astype(object)
    df["status"] = df["status"].replace({"": "pending"})
    df = df.drop_duplicates("ticket_id", keep="last").reset_index(drop=True)
    return df


def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, keep_default_na=True, **kw)


def load_toto(path: Path | str) -> pd.DataFrame:
    df = _read_csv(Path(path))
    return empty_toto() if df is None else normalise_toto(df)


def load_ledger(path: Path | str) -> pd.DataFrame:
    df = _read_csv(Path(path), dtype={c: str for c in LEDGER_COLUMNS if c not in ("cost", "units", "winnings")})
    return empty_ledger() if df is None else normalise_ledger(df)


def _to_csv_text(df: pd.DataFrame) -> str:
    out = df.copy()
    if "draw_date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["draw_date"]):
        out["draw_date"] = out["draw_date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False, lineterminator="\n")


def save_toto(df: pd.DataFrame, path: Path | str) -> None:
    atomic_write_text(path, _to_csv_text(normalise_toto(df)))


def save_ledger(df: pd.DataFrame, path: Path | str) -> None:
    atomic_write_text(path, _to_csv_text(normalise_ledger(df)))


def toto_numbers(row) -> list[int]:
    """The six winning numbers of a toto.csv row, ascending."""
    return sorted(int(row[f"n{i}"]) for i in range(1, 7))


def no_winner_streak(df: pd.DataFrame, reset_on_cascade: bool = False) -> int:
    """Newest first count of TOTO draws with no Group 1 winner.

    With ``reset_on_cascade`` the count stops at a cascade or Hongbao draw, because the
    jackpot was paid out there (use this to predict the next cascade draw). Without it the
    count is the plain "draws in a row with no Group 1 winner" shown to the user.
    """
    n = 0
    for _, r in df.sort_values("draw_number", ascending=False).iterrows():
        if int(r["g1_winners"]) > 0:
            break
        if reset_on_cascade and r["draw_type"] in ("cascade", "hongbao"):
            break
        n += 1
    return n
