"""Command line for Huat Bot: ``python -m huatbot <command> [options]`` (or the ``huatbot`` script).

Commands:

  run         fetch new draws, analyse, write the vault and post the 3 Telegram messages
              (a manual run posts even when the newest draw was posted before)
  serve       run on the schedule: 7.30pm Singapore time on draw days, retrying until the
              results are out (this is what the Docker container runs)
  fetch       update the CSVs and the next draw info only
  report      print the full report from the stored data (no fetching, posting or writing)
  check-site  fetch every page the bot reads and say plainly whether it can still read it
  demo        the whole pipeline on synthetic data in a demo vault (no network, never posts)
  init-vault  create the bot folder with Settings.md and Tickets.md in the vault

Options (after the command): --game toto|4d|both, --dry-run, --no-fetch, --no-post, --vault PATH.
Environment: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, VAULT_PATH, VAULT_FOLDER, DATA_DIR, RUN_AT,
RETRY_MINUTES, RETRY_HOURS, DRY_RUN (1 forces a dry run), COMMENTARY, CLAUDE_BIN, LOG_LEVEL
(default INFO for serve, WARNING otherwise), TZ.

Exit codes: 0 when the command worked, 1 when it did not.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable

from . import __version__, runner, scheduler
from . import fetch as site
from .http import Fetcher
from .models import RunResult
from .textfmt import plural
from .vault import SG, Vault

log = logging.getLogger(__name__)

DEMO_VAULT = "demo-vault"
TRUE_WORDS = ("1", "true", "yes", "on", "y")
GAME_CHOICES = ("toto", "4d", "both")


def make_fetcher() -> Fetcher:
    """The polite site fetcher. Tests replace this function with an offline fake."""
    return Fetcher()


def env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in TRUE_WORDS


def _setup_logging(command: str) -> None:
    """LOG_LEVEL (default INFO for serve, WARNING otherwise). Leaves existing handlers alone."""
    default = "INFO" if command == "serve" else "WARNING"
    name = (os.environ.get("LOG_LEVEL") or default).strip().upper()
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        level = logging.getLevelName(default)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--game", type=str.lower, choices=GAME_CHOICES, default="both",
                        help="which game to fetch and report (default both)")
    common.add_argument("--dry-run", action="store_true",
                        help="print the Telegram messages instead of posting them (DRY_RUN=1 does the same)")
    common.add_argument("--no-fetch", action="store_true", help="use only the stored data, do not contact the site")
    common.add_argument("--no-post", action="store_true", help="do not post or print the Telegram messages")
    common.add_argument("--vault", metavar="PATH",
                        help="the Obsidian vault folder (default VAULT_PATH, or ./demo-vault for demo)")

    parser = argparse.ArgumentParser(
        prog="huatbot",
        description="Huat Bot: Singapore Pools TOTO and 4D analyst. Everything is read from and written "
                    "to an Obsidian vault.",
    )
    parser.add_argument("--version", action="version", version=f"huatbot {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")
    helps = {
        "run": "fetch, analyse, write the vault and post the 3 messages now",
        "serve": "run on the schedule (7.30pm Singapore time on draw days)",
        "fetch": "update the CSVs and next draw info only",
        "report": "print the full report from the stored data",
        "check-site": "check that every Singapore Pools page can still be read",
        "demo": "run everything on synthetic data in a demo vault",
        "init-vault": "create the bot folder, Settings.md and Tickets.md",
    }
    for name, text in helps.items():
        sub.add_parser(name, parents=[common], help=text, description=text)
    return parser


def _vault(args: argparse.Namespace) -> Vault:
    return runner.resolve_vault(args.vault) if args.vault else Vault.from_env()


def _games(args: argparse.Namespace) -> tuple[str, ...]:
    return runner.normalise_games(args.game)


def _print_summary(result: RunResult, out: Callable[[str], Any] = print) -> None:
    for w in result.warnings:
        out(f"Warning: {w}")
    new = [f"{runner.LABELS[g]} {plural(len(v), 'draw')}" for g, v in result.new_draws.items() if v]
    out(f"New draws: {', '.join(new)}." if new else "No new draws.")
    if result.report_path:
        out(f"Report: {result.report_path}")
    if result.posted:
        out(f"Posted {plural(len(result.messages), 'message')} to Telegram.")
    else:
        out("Nothing was posted to Telegram.")
    out("Done." if result.ok else "Finished with a problem: see the warnings above and the activity log in the vault.")


# Commands


def cmd_run(args: argparse.Namespace, dry_run: bool) -> int:
    vault = _vault(args)
    fetcher = None if args.no_fetch else make_fetcher()
    try:
        result = runner.run(_games(args), dry_run=dry_run, fetch=not args.no_fetch, post=not args.no_post,
                            vault=vault, fetcher=fetcher, out=print, force_post=True)
    finally:
        if fetcher is not None and hasattr(fetcher, "close"):
            fetcher.close()
    _print_summary(result)
    return 0 if result.ok else 1


def cmd_serve(args: argparse.Namespace, dry_run: bool) -> int:
    if args.no_fetch:
        print("serve checks the Singapore Pools site for new results, so it cannot be used with --no-fetch.",
              file=sys.stderr)
        return 1
    vault = _vault(args)
    config = scheduler.SchedulerConfig.from_env()
    fetcher = make_fetcher()
    vault.ensure_layout(dict(runner.TEMPLATES))
    quiet = dry_run or args.no_post

    def run_fn(games: tuple[str, ...]) -> RunResult:
        return runner.run(games, dry_run=dry_run, fetch=True, post=not args.no_post, vault=vault,
                          fetcher=fetcher, out=print, force_post=False)

    def check_fn(game: str):
        return site.latest_on_site(fetcher, game)[1]

    def refresh_fn() -> dict:
        return runner.refresh_next_draws(vault, fetcher)

    def notify_fn(text: str) -> bool:
        return runner.notify(text, dry_run=quiet, out=print)

    for w in config.warnings:
        print(f"Warning: {w}")
    print(f"Huat Bot scheduler started. {config.describe()}. Vault folder: {vault.base}")
    try:
        scheduler.serve(run_fn, check_fn, refresh_fn, notify_fn, vault.log,
                        lambda: datetime.now(SG), time.sleep, config)
    except KeyboardInterrupt:
        print("Scheduler stopped.")
    finally:
        if hasattr(fetcher, "close"):
            fetcher.close()
    return 0


def cmd_fetch(args: argparse.Namespace, dry_run: bool) -> int:
    if args.no_fetch:
        print("Nothing to do: fetch with --no-fetch.", file=sys.stderr)
        return 1
    vault = _vault(args)
    fetcher = make_fetcher()
    try:
        result = runner.fetch_data(_games(args), vault=vault, fetcher=fetcher)
    finally:
        if hasattr(fetcher, "close"):
            fetcher.close()
    for w in result.warnings:
        print(f"Warning: {w}")
    new = [f"{runner.LABELS[g]} {plural(len(v), 'new draw')}" for g, v in result.new_draws.items()]
    print(", ".join(new) + "." if new else "No game was updated.")
    print(f"Data folder: {vault.data_dir}")
    return 0 if result.ok else 1


def cmd_report(args: argparse.Namespace, dry_run: bool) -> int:
    vault = _vault(args)
    text = runner.build_report(vault, games=_games(args))
    if text is None:
        print(f"No TOTO and 4D data is stored in {vault.data_dir} yet. Run the fetch command first.",
              file=sys.stderr)
        return 1
    print(text)
    return 0


def cmd_check_site(args: argparse.Namespace, dry_run: bool) -> int:
    fetcher = make_fetcher()
    try:
        ok = site.check_site(fetcher, out=print)
    finally:
        if hasattr(fetcher, "close"):
            fetcher.close()
    return 0 if ok else 1


def cmd_demo(args: argparse.Namespace, dry_run: bool) -> int:
    path = args.vault or DEMO_VAULT
    print(f"Demo run with synthetic data in the vault at {path}. Nothing is fetched and nothing is posted.")
    print()
    result = runner.run(_games(args), demo=True, vault=path, out=print)
    _print_summary(result)
    return 0 if result.ok else 1


def cmd_init_vault(args: argparse.Namespace, dry_run: bool) -> int:
    vault = _vault(args)
    created = vault.ensure_layout(dict(runner.TEMPLATES))
    for rel in created:
        vault.log(runner.EV_NOTE, f"Created [[{rel.rsplit('/', 1)[-1].removesuffix('.md')}]] (starter note)")
        print(f"Created {vault.path(rel)}")
    print(f"The bot folder is ready at {vault.base}" + ("" if created else " (nothing new was needed)") + ".")
    print("Edit Settings.md for your budgets and add your tickets to Tickets.md in Obsidian.")
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace, bool], int]] = {
    "run": cmd_run,
    "serve": cmd_serve,
    "fetch": cmd_fetch,
    "report": cmd_report,
    "check-site": cmd_check_site,
    "demo": cmd_demo,
    "init-vault": cmd_init_vault,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the exit code: 0 ok, 1 failure."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help and --version exit 0, usage errors 2: keep to 0 and 1
        return 0 if exc.code in (0, None) else 1
    if not args.command:
        parser.print_help()
        return 1
    _setup_logging(args.command)
    dry_run = bool(args.dry_run or env_flag("DRY_RUN"))
    try:
        return COMMANDS[args.command](args, dry_run)
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 1
    except Exception as exc:
        log.exception("%s failed", args.command)
        text = " ".join(str(exc).split()) or type(exc).__name__
        print(f"Error: {args.command} failed ({type(exc).__name__}: {text})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
