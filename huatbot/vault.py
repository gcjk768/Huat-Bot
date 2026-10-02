"""The Obsidian vault on the NAS: the one place Huat Bot reads from and writes to.

Layout inside the vault (``root``), everything under one bot folder (``base``)::

    <root>/
      .obsidian/                 never touched
      Huat Bot/                  base (VAULT_FOLDER)
        Settings.md              read: budgets and options as note properties
        Tickets.md               read: the user's tickets
        Dashboard.md, Ledger.md  written every run
        Data/                    toto.csv, fourd.csv, ledger.csv, state.json, caches
        Draws/TOTO, Draws/4D     one note per draw
        Reports/, Suggestions/   one note per run / upcoming draw
        Logs/YYYY-MM Activity.md every fetch, note, post and error

Obsidian (or Obsidian Sync, Syncthing, SMB clients) may be watching the same folder
from phones and PCs, so this module is careful:

* every write is atomic (temp file plus rename) so a syncing client never sees half a note;
* a note whose content did not change is not rewritten, so sync tools see no churn;
* new files get normal permissions (not the 0600 a temp file is born with) and an
  existing file keeps its mode, so other NAS users and sync apps can still read it;
* YAML frontmatter is plain, valid Obsidian properties (block style keys, inline lists,
  dates as ISO strings);
* paths are confined to the bot folder and hidden folders such as ``.obsidian`` are refused;
* the activity log never raises, so a full disk or a locked file cannot crash a run.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from . import constants as C
from .store import atomic_write_text

log = logging.getLogger(__name__)

SG = ZoneInfo(C.SG_TZ_NAME)

DEFAULT_VAULT_PATH = "./vault"
DEFAULT_FOLDER = "Huat Bot"
DATA_FOLDER = "Data"
LOG_FOLDER = "Logs"
# Folders created inside the bot folder by ensure_layout (Data is created separately because
# DATA_DIR may point somewhere else).
LAYOUT_FOLDERS = ("Draws/TOTO", "Draws/4D", "Reports", "Suggestions", LOG_FOLDER)

LOG_TABLE_HEADER = "| Time | Event | Details |"
LOG_TABLE_RULE = "| --- | --- | --- |"

# Obsidian's own frontmatter fences. A closing "..." is valid YAML too, so accept it on read.
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)^(?:---|\.\.\.)[ \t]*$\n?", re.S | re.M)

# Locale independent names, so a container with an odd LANG still writes English dates.
_DAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# One lock for every Vault object in this process, so two writers never interleave log rows.
_LOG_LOCK = threading.Lock()
_UMASK_LOCK = threading.Lock()
_UMASK: int | None = None


# Small helpers (no textfmt dependency, so the vault works on its own)


def to_sg(dt: datetime) -> datetime:
    """Singapore time. A naive datetime is taken to already be Singapore time."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SG)
    return dt.astimezone(SG)


def fmt_log_time(when: datetime) -> str:
    """Activity log timestamp, e.g. ``Fri 2 Oct 2026 7.30pm`` (Singapore time)."""
    t = to_sg(when)
    hour12 = t.hour % 12 or 12
    suffix = "am" if t.hour < 12 else "pm"
    return f"{_DAY_ABBR[t.weekday()]} {t.day} {_MONTH_ABBR[t.month - 1]} {t.year} {hour12}.{t.minute:02d}{suffix}"


def log_note_path(when: datetime) -> str:
    """Relative path of the activity log for the month of ``when``: ``Logs/2026-10 Activity.md``."""
    t = to_sg(when)
    return f"{LOG_FOLDER}/{t.year:04d}-{t.month:02d} Activity.md"


def _log_title(when: datetime) -> str:
    t = to_sg(when)
    return f"# Huat Bot activity, {_MONTH_NAMES[t.month - 1]} {t.year}"


def _table_cell(value: Any) -> str:
    """Make text safe for one markdown table cell.

    Newlines would end the row and a bare ``|`` would start a new cell, so flatten the
    one and escape the other. Spaced dashes used as punctuation become commas, and long
    dashes become spaces (the log is prose and must have no dashes). Hyphens inside
    tokens such as ISO dates and file names are left alone.
    """
    text = " ".join(str(value).split())
    text = re.sub(r"\s+[\-–—]+\s+", ", ", text)
    text = re.sub(r"[–—]", " ", text)
    text = text.replace("\\|", "|").replace("|", "\\|")
    return text.strip()


# Frontmatter


class _FrontmatterDumper(yaml.SafeDumper):
    """SafeDumper that writes lists of plain values inline (``tags: [huatbot, toto]``).

    The top level mapping stays in block style, which is what Obsidian's properties panel
    reads and writes. ``yaml.safe_dump(default_flow_style=None)`` cannot do this: it would
    turn a mapping of plain values into one ``{...}`` line.
    """


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flat = all(x is None or isinstance(x, (str, int, float, bool)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flat and bool(data))


_FrontmatterDumper.add_representer(list, _represent_list)

# Strings that a YAML 1.2 reader (Obsidian) would turn into a number, such as the 4D number
# "0042" or "1e3". PyYAML follows YAML 1.1 and leaves "0042" unquoted because 1.1 reads it as
# octal only with digits 0 to 7, so force quotes for anything number shaped.
_NUMBER_LIKE = re.compile(r"^[+-]?(\d[\d_]*(\.\d*)?|\.\d+)([eE][+-]?\d+)?$|^0[xXoObB][0-9a-fA-F_]+$")


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    if _NUMBER_LIKE.match(data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_str(data)


_FrontmatterDumper.add_representer(str, _represent_str)


def _plain(value: Any) -> Any:
    """Convert a frontmatter value to plain YAML friendly types.

    Dates and datetimes become ISO strings, numpy scalars become Python numbers, NaN and
    NaT become null, tuples and sets become lists, anything unknown becomes its text.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", 0):  # an array
            return _plain(value.tolist())
        if hasattr(value, "item"):  # a scalar such as numpy.int64
            return _plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        if value != value:  # pandas NaT
            return None
        if hasattr(value, "to_pydatetime"):  # pandas Timestamp
            value = value.to_pydatetime()
            if value.tzinfo is None and value.time() == datetime.min.time():
                return value.date().isoformat()  # a normalised draw date
        return value.isoformat(timespec="seconds" if not value.microsecond else "auto")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_plain(v) for v in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):  # numpy array
        return _plain(value.tolist())
    return str(value)


def _iso_values(value: Any) -> Any:
    """Turn YAML dates that Obsidian wrote unquoted back into ISO strings."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _iso_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_iso_values(v) for v in value]
    return value


def dump_frontmatter(frontmatter: dict) -> str:
    """YAML text for the properties block (without the ``---`` fences), keys in the given order."""
    return yaml.dump(
        _plain(dict(frontmatter)),
        Dumper=_FrontmatterDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )


def render_note(body: str, frontmatter: dict | None = None) -> str:
    """Full note text: optional ``---`` properties block, then the body ending in one newline."""
    body = (body or "").replace("\r\n", "\n")
    body = body.rstrip("\n") + "\n" if body.strip() else ""
    if frontmatter:
        return f"---\n{dump_frontmatter(frontmatter)}---\n{body}"
    return body


def parse_note(text: str | None, source: str = "note") -> tuple[dict, str]:
    """Split note text into (frontmatter dict, body).

    No properties block gives ``({}, text)``. Properties that are not valid YAML, or not a
    mapping, give ``({}, body)`` and a logged warning, so one bad edit in Obsidian never
    stops a run.
    """
    if not text:
        return {}, ""
    text = text.lstrip("﻿").replace("\r\n", "\n")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    body = text[m.end():].lstrip("\n")
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        log.warning("Properties in %s are not valid YAML, ignoring them: %s", source, exc)
        return {}, body
    if data is None:
        return {}, body
    if not isinstance(data, dict):
        log.warning("Properties in %s are not a list of name: value pairs, ignoring them", source)
        return {}, body
    return _iso_values(data), body


def _json_default(obj: Any) -> Any:
    """json.dumps fallback for state.json: dates, numpy numbers, sets and paths."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(obj)
        except TypeError:
            return list(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return obj.as_posix()
    return str(obj)


def _clean_parts(text: str, what: str) -> list[str]:
    """Split a relative path into safe parts, refusing absolute paths, ``..`` and hidden names."""
    raw = str(text).replace("\\", "/")
    if raw.startswith("/") or PureWindowsPath(raw).drive:
        raise ValueError(f"{what} must be a path inside the bot folder, got an absolute path: {text!r}")
    parts = []
    for piece in raw.split("/"):
        piece = piece.strip()
        if piece in ("", "."):
            continue
        if piece == "..":
            raise ValueError(f"{what} may not leave the bot folder: {text!r}")
        if piece.startswith("."):
            # .obsidian, .trash, .git, .stfolder ... belong to Obsidian and sync tools.
            raise ValueError(f"{what} may not use hidden names such as .obsidian: {text!r}")
        parts.append(piece)
    return parts


def _default_file_mode() -> int:
    """0666 minus the process umask: the mode a normally created file would get."""
    global _UMASK
    with _UMASK_LOCK:
        if _UMASK is None:
            current = os.umask(0)
            os.umask(current)
            _UMASK = current
    return 0o666 & ~_UMASK


class Vault:
    """An Obsidian vault folder on the NAS, with the bot confined to ``root / folder``."""

    def __init__(self, root: Path | str, folder: str = DEFAULT_FOLDER, data_dir: Path | str | None = None) -> None:
        self.root = Path(root).expanduser()
        self.folder = "/".join(_clean_parts(folder or "", "VAULT_FOLDER"))
        self.base = self.root.joinpath(*self.folder.split("/")) if self.folder else self.root
        self.data_dir = Path(data_dir).expanduser() if data_dir else self.base / DATA_FOLDER

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Vault":
        """Build from VAULT_PATH (default ./vault), VAULT_FOLDER (default "Huat Bot") and DATA_DIR
        (default <vault>/<folder>/Data). Empty variables count as unset."""
        env = os.environ if env is None else env
        root = (env.get("VAULT_PATH") or "").strip() or DEFAULT_VAULT_PATH
        folder = env.get("VAULT_FOLDER")
        folder = DEFAULT_FOLDER if folder is None or not folder.strip() else folder.strip()
        data_dir = (env.get("DATA_DIR") or "").strip() or None
        return cls(Path(root), folder, Path(data_dir) if data_dir else None)

    def __repr__(self) -> str:
        return f"Vault(root={str(self.root)!r}, folder={self.folder!r}, data_dir={str(self.data_dir)!r})"

    # Paths

    @property
    def toto_csv(self) -> Path:
        return self.data_dir / "toto.csv"

    @property
    def fourd_csv(self) -> Path:
        return self.data_dir / "fourd.csv"

    @property
    def ledger_csv(self) -> Path:
        return self.data_dir / "ledger.csv"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def prize_rules_path(self) -> Path:
        return self.data_dir / "prize_rules.json"

    @property
    def backtest_cache_path(self) -> Path:
        return self.data_dir / "backtest_cache.json"

    def path(self, *parts: str | Path) -> Path:
        """Absolute path of a file inside the bot folder.

        ``vault.path("Draws/TOTO", "2026-10-01 TOTO 4123.md")``. Raises ValueError for absolute
        paths, ``..`` and hidden names, so nothing outside the bot folder (and nothing in
        ``.obsidian``) can ever be written through the vault.
        """
        pieces: list[str] = []
        for part in parts:
            pieces.extend(_clean_parts(str(part), "Note path"))
        return self.base.joinpath(*pieces)

    def rel(self, *parts: str | Path) -> str:
        """Normalised relative path (forward slashes) of a file inside the bot folder."""
        pieces: list[str] = []
        for part in parts:
            pieces.extend(_clean_parts(str(part), "Note path"))
        return "/".join(pieces)

    def exists(self, rel: str) -> bool:
        return self.path(rel).is_file()

    # Layout

    def ensure_layout(self, templates: dict[str, str] | None = None) -> list[str]:
        """Create the bot folder and its sub folders, and write each starter note that is missing.

        ``templates`` maps a relative path (e.g. ``"Settings.md"``) to the full note text. An
        existing file is never overwritten, so the user's edits are safe. Returns the relative
        paths of the notes created by this call (folders are not listed).
        """
        if not self.root.exists():
            log.info("Vault folder %s does not exist yet, creating it", self.root)
        self.base.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for sub in LAYOUT_FOLDERS:
            self.path(sub).mkdir(parents=True, exist_ok=True)
        created: list[str] = []
        for rel, content in (templates or {}).items():
            target = self.path(rel)
            if target.exists():
                continue
            self._write(target, content.replace("\r\n", "\n"))
            created.append(self.rel(rel))
            log.info("Created %s in the vault", self.rel(rel))
        return created

    # Reading

    @staticmethod
    def _read(path: Path) -> str | None:
        """Text of a file, or None if it does not exist. Strips a BOM, normalises line endings,
        and replaces undecodable bytes rather than failing on them."""
        try:
            return path.read_text(encoding="utf-8-sig", errors="replace")
        except (FileNotFoundError, NotADirectoryError):
            return None
        except IsADirectoryError:
            log.warning("%s is a folder, expected a note", path)
            return None

    def read_text(self, rel: str) -> str | None:
        """Text of a note in the bot folder, or None if it does not exist."""
        return self._read(self.path(rel))

    def read_note(self, rel: str) -> tuple[dict, str]:
        """(properties, body) of a note. Missing note: ``({}, "")``. Bad YAML: ``({}, body)``
        plus a logged warning. Dates in the properties come back as ISO strings."""
        return parse_note(self.read_text(rel), source=self.rel(rel))

    # Writing

    def _write(self, target: Path, text: str) -> None:
        """Atomic write that keeps an existing file's mode, or gives a new file the normal mode."""
        try:
            mode = target.stat().st_mode & 0o7777
        except OSError:
            mode = _default_file_mode()
        atomic_write_text(target, text)
        try:
            os.chmod(target, mode)
        except OSError as exc:  # some network file systems do not support chmod
            log.debug("Could not set permissions on %s: %s", target, exc)

    def write_text(self, rel: str, text: str) -> bool:
        """Atomically write raw text to a file in the bot folder.

        Returns False, and leaves the file untouched, when it already holds exactly this text.
        """
        target = self.path(rel)
        text = text.replace("\r\n", "\n")
        if self._read(target) == text:
            log.debug("%s unchanged, not rewritten", self.rel(rel))
            return False
        self._write(target, text)
        return True

    def write_note(self, rel: str, body: str, frontmatter: dict | None = None) -> bool:
        """Write a note (properties block plus markdown body) atomically.

        Returns False and skips the write when the note already has exactly this content,
        so Obsidian sync sees no change. Errors (for example a read only share) are raised.
        """
        return self.write_text(rel, render_note(body, frontmatter))

    # Activity log

    def log(self, event: str, message: str, when: datetime | None = None) -> None:
        """Append one row to ``Logs/YYYY-MM Activity.md``.

        Row example: ``| Fri 2 Oct 2026 7.30pm | FETCH | 2 new TOTO draws |``. The note is created
        with a title and table header on first use each month. Pipes are escaped, newlines
        flattened. Pass note paths as [[wikilinks]] so the row stays free of prose dashes.
        Never raises: a failure is logged and the run carries on.
        """
        try:
            when = to_sg(when) if when is not None else datetime.now(SG)
            row = f"| {fmt_log_time(when)} | {_table_cell(event).upper()} | {_table_cell(message)} |"
            log.info("%s: %s", str(event).upper(), message)
            with _LOG_LOCK:
                target = self.path(log_note_path(when))
                existing = self._read(target)
                if existing is None or not existing.strip():
                    text = f"{_log_title(when)}\n\n{LOG_TABLE_HEADER}\n{LOG_TABLE_RULE}\n{row}\n"
                else:
                    text = existing.replace("\r\n", "\n").rstrip("\n") + "\n"
                    last_line = text.rstrip("\n").rsplit("\n", 1)[-1].strip()
                    if not last_line.startswith("|"):
                        # The user wrote below the table (or removed it): start a fresh table so
                        # the new row still renders as part of one.
                        text += f"\n{LOG_TABLE_HEADER}\n{LOG_TABLE_RULE}\n"
                    text += row + "\n"
                self._write(target, text)
        except Exception as exc:  # the activity log must never stop a run
            log.warning("Could not write the activity log (%s: %s)", type(exc).__name__, exc)

    # Run state (next draw dates, skip lists, last posted ...)

    def load_state(self) -> dict:
        """Contents of Data/state.json, ``{}`` if missing or unreadable (a warning is logged)."""
        text = self._read(self.state_path)
        if text is None or not text.strip():
            return {}
        try:
            data = json.loads(text)
        except ValueError as exc:
            log.warning("state.json is not valid JSON, starting from an empty state: %s", exc)
            return {}
        if not isinstance(data, dict):
            log.warning("state.json does not hold an object, starting from an empty state")
            return {}
        return data

    def save_state(self, state: dict) -> None:
        """Atomically write Data/state.json (skipped when unchanged). Dates become ISO strings."""
        text = json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n"
        if self._read(self.state_path) == text:
            return
        self._write(self.state_path, text)
