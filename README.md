<div align="center">

# Huat Bot

**Singapore Pools TOTO: the latest result, what the next draws will do and where the next big prize is**

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Telegram](https://img.shields.io/badge/Telegram-2%20messages-26A5E4?logo=telegram&logoColor=white)](#telegram-bot-and-channel)
[![Obsidian](https://img.shields.io/badge/Obsidian-vault-7C3AED?logo=obsidian&logoColor=white)](#the-vault)
[![draw.io](https://img.shields.io/badge/diagrams-draw.io-F08705?logo=diagramsdotnet&logoColor=white)](docs/diagrams)
[![Tests](https://github.com/gcjk768/Huat-Bot/actions/workflows/ci.yml/badge.svg)](https://github.com/gcjk768/Huat-Bot/actions/workflows/ci.yml)

</div>

It lives on your NAS. On every TOTO draw day it fetches the result, checks your tickets, works out
what the next draws are likely to do (how big the jackpot gets, the chance somebody wins it, when it
cascades) and where the next big prize is, posts two short messages to your Telegram channel, and
keeps everything it reads and writes inside your Obsidian vault.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/architecture-dark.svg">
    <img alt="Huat Bot runs in Docker on the NAS, reads Singapore Pools, writes to the Obsidian vault and posts to Telegram" src="docs/diagrams/architecture-light.svg" width="100%">
  </picture>
</p>

<p align="center"><sub>Made with draw.io. Edit the source:
<a href="https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2Fgcjk768%2FHuat-Bot%2Fmain%2Fdocs%2Fdiagrams%2Farchitecture.drawio">architecture</a> ·
<a href="https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2Fgcjk768%2FHuat-Bot%2Fmain%2Fdocs%2Fdiagrams%2Fdraw-day.drawio">draw day</a> ·
<a href="docs/diagrams">how to regenerate</a></sub></p>

> [!NOTE]
> Every figure comes from Python code. If you switch on the optional commentary, Claude only writes a
> short comment about figures the code already computed. Every draw is independent, so the bot never
> suggests numbers: no pattern in past results makes any set of numbers more likely to win.

## Contents

* [What it does](#what-it-does)
* [The honest odds note](#the-honest-odds-note)
* [Quick start](#quick-start)
* [UGREEN NAS setup, step by step](#ugreen-nas-setup-step-by-step)
* [Telegram bot and channel](#telegram-bot-and-channel)
* [First run, in this order](#first-run-in-this-order)
* [The vault](#the-vault)
* [Schedule and retries](#schedule-and-retries)
* [Commands](#commands)
* [Settings](#settings)
* [How the figures are worked out](#how-the-figures-are-worked-out)
* [Optional Claude commentary](#optional-claude-commentary)
* [Troubleshooting](#troubleshooting)
* [Updating](#updating)
* [Upgrading from the 4D and suggestions version](#upgrading-from-the-4d-and-suggestions-version)
* [Development](#development)

## What it does

| Step | What happens |
| --- | --- |
| Fetch | Downloads only the TOTO draws that are missing, from draw 2995 (9 Oct 2014, the first draw of the current 6 from 49 format). At most 4 requests at a time, with a pause, slowing down if the site complains. Checks that each page is the draw it asked for and that the newest stored draw matches the latest one on the site. |
| Prize rules | Reads the official TOTO prize structure page and says plainly whether the prize percentages were confirmed or the built in values were used. |
| Next draw | Reads the next draw page: date, estimated jackpot and whether it is a normal, cascade, Hongbao or special draw. |
| What TOTO will do | Projects the jackpot draw by draw until it cascades: how big it gets if nobody wins it, how many boards each draw is likely to sell, the chance somebody wins Group 1 at each draw and the chance it is still unwon by then. |
| Next big prize | The biggest jackpot on the way (usually the cascade draw), when it comes, how likely it is to get that far, plus any announced Hongbao or special draw. |
| Buy signal | The average return per $1 for the next draw at its jackpot, labelled HIGH, MEDIUM or LOW. |
| Jackpot history | How often Group 1 is won, how long a jackpot usually lasts, how many cascaded, and the biggest jackpots so far. |
| Tickets | Reads your tickets from `Tickets.md`, checks each one once its result is out, and keeps a ledger with total spent, won and net. |
| Vault | Writes a dashboard, the ledger, a note per draw, a full report per run and an activity log, all in your Obsidian vault. |
| Telegram | Posts two messages: 1) the latest result and your ticket check, 2) the next draw, its jackpot and buy signal, and the next big prize. Each stays under 4,000 characters. A draw is never posted twice. |

What it does not do: it never suggests numbers to pick. Every draw is independent and every set of
six numbers has exactly the same chance, so "hot", "cold" or "overdue" numbers are not a thing.

Message 2 looks like this (from the demo):

```text
Next TOTO draw (draw 4124): Mon 5 Oct 2026, 6.30pm
Estimated jackpot: $2,100,000
Draw type: Normal
Jackpot rollovers so far: 1 of 3, then it cascades
Chance somebody wins Group 1 at this draw: 24%
Buy signal: MEDIUM
Return per $1: $0.45 on average

Next big prize
If nobody wins Group 1 first, the jackpot snowballs to about $4,015,000 at the cascade
draw on Mon 12 Oct 2026 (55% chance it gets that far). The chance somebody wins it before
then is about 62%.
Draw        Jackpot  Unwon  Won
Mon 5 Oct    $2.10m   100%  24%
Thu 8 Oct    $3.00m    76%  27%
Mon 12 Oct   $4.01m    55%  30%
```

## The honest odds note

> Every draw is independent, so past results do not change the odds. TOTO Group 1 is 1 in
> 13,983,816 per board, and any TOTO prize is about 1 in 54. TOTO pays about 54% of sales back as
> prizes, so on average each $1 brings back about 54 cents.

The bot says this in every report. The chances it shows (24%, 62% and so on) are the chances that
**anybody** in Singapore wins Group 1 at a draw, not that you do. A snowballed jackpot can lift one
draw's average return per $1, even above $1, but almost every ticket still wins nothing. Never spend
more than you planned.

## Quick start

Any machine with Docker and Docker Compose:

```bash
git clone https://github.com/gcjk768/Huat-Bot.git
cd Huat-Bot
cp .env.example .env          # then fill in .env (see below)
docker compose build
docker compose run --rm huat-bot python -m huatbot check-site
docker compose run --rm huat-bot python -m huatbot demo
docker compose run --rm huat-bot python -m huatbot run --dry-run
docker compose up -d          # starts the scheduler (serve)
docker compose logs -f
```

Without Docker (Python 3.11 or newer):

```bash
pip install -r requirements.txt
python -m huatbot demo                    # writes ./demo-vault, open it as a vault in Obsidian to look around
VAULT_PATH=/path/to/MyVault python -m huatbot run --dry-run
```

Outside Docker the bot reads `.env` from the folder you run it in (or the file named by `ENV_FILE`).
Values already set in the shell win over the file. Without a Telegram token the run turns into a dry
run and says so.

```bash
VAULT_PATH=/path/to/MyVault python -m huatbot run
```

## UGREEN NAS setup, step by step

These steps use UGOS Pro. Menu names can differ a little between UGOS versions.

### 1. Turn on SSH and find your PUID and PGID

The container runs as your own NAS user, so everything it writes into the vault belongs to you.

1. In UGOS, open **Control Panel**, then **Terminal**, and turn on **SSH**.
2. From a PC, connect: `ssh yourname@your-nas-ip`
3. Run `id`. You will see something like:

   ```text
   uid=1000(yourname) gid=10(admin) groups=10(admin),100(users)
   ```

4. Put the numbers in `.env`: `PUID=1000` (the uid) and `PGID=10` (the gid).

You can turn SSH off again afterwards if you set up the bot through the Docker app.

### 2. Choose the vault folder

Huat Bot works inside an Obsidian vault on the NAS, for example a shared folder `Obsidian` with a vault
called `MyVault`. On UGOS the full path then looks like:

```text
/volume1/Obsidian/MyVault
```

To find the real path, open **File Manager**, right click the vault folder and choose **Properties**
(or run `ls /volume1` over SSH). Make sure the folder already exists: if Docker has to create it, it
belongs to root and the bot cannot write there.

Put the path in `.env` as `VAULT_HOST_PATH`. The bot keeps everything in one folder inside the vault
(`VAULT_FOLDER`, default `Huat Bot`) and never touches `.obsidian` or your other notes.

Your NAS user (the PUID above) needs **read and write** permission on that shared folder. In
**Control Panel**, open **Shared Folder** (or the folder's **Properties** in File Manager) and give
your user read and write access.

### 3. Make sure the vault is synced to your devices

The bot writes plain Markdown and CSV files into the NAS folder. To read them in Obsidian on your phone
and PC, that folder has to sync to your devices. Pick one:

| Option | Works on | Notes |
| --- | --- | --- |
| Syncthing | Windows, Mac, Linux, Android | Run Syncthing on the NAS (as a Docker container) and on each device, and share the vault folder. The most reliable two way sync for a NAS. |
| WebDAV with the Remotely Save plugin | All, including iPhone and iPad | Turn on WebDAV in UGOS file services, then point the Remotely Save community plugin in Obsidian at the vault folder. |
| UGREEN sync on a PC | Windows, Mac | Two way folder sync with the UGREEN NAS desktop app, then open the synced folder as a vault. |
| Open the share directly | Windows, Mac | Map the shared folder over SMB and open it as a vault. Simple, but only while the PC is on the home network. |
| Obsidian Sync | All | Only works if an Obsidian app keeps the NAS folder open and running (for example a PC that opens the SMB share and stays on). The bot itself cannot push to Obsidian Sync. |

Obsidian picks up files changed by the bot on its own. The bot writes every file in one go (a
temporary file, then a rename) and skips files whose content did not change, so sync tools never see
half written notes or needless changes.

### 4. Put the project on the NAS

Pick a folder for Docker projects, for example `/volume1/docker/huat-bot`, and put this repository there:

* over SSH: `git clone https://github.com/gcjk768/Huat-Bot.git /volume1/docker/huat-bot`
* or download the ZIP from GitHub, unzip it on your PC, and upload the folder with File Manager.

### 5. Create the .env file

Copy `.env.example` to `.env` in the same folder and fill it in. At least:

| Variable | Example |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | from BotFather (next section) |
| `TELEGRAM_CHAT_ID` | `@yourchannel` or the numeric id of a private channel |
| `VAULT_HOST_PATH` | `/volume1/Obsidian/MyVault` |
| `PUID`, `PGID` | from the `id` command |

If you want to see the messages in the log before anything is posted, set `DRY_RUN=1` for the first
few days, then set it back to `0` and run `docker compose up -d` (or redeploy the project in the UGOS
Docker app). A plain restart (`docker compose restart`, `docker restart` or the Restart button in the
Docker app) keeps the old values, because `.env` is only read when the container is created. All
variables are listed under [Settings](#settings).

### 6a. Start it with the UGOS Docker app

1. Open the **Docker** app and go to **Project**.
2. Click **Create**. Give it a name (`huat-bot`) and choose the folder from step 4 as the storage path.
3. UGOS finds `docker-compose.yml` in that folder. Use it as it is (if you are asked to paste the
   compose file instead, paste the content of `docker-compose.yml`).
4. Click **Deploy** (or **Build and run**). The first build downloads Python packages and takes a few
   minutes. If your UGOS version cannot build images inside a project, use the SSH route (6b) once to
   build, then manage the running container from the Docker app.
5. The project starts the `huat-bot` container, which runs the scheduler. To run one off commands
   (see [First run](#first-run-in-this-order)), open **Container**, select `huat-bot`, open its
   **Terminal**, and type for example `python -m huatbot check-site`.

### 6b. Or start it over SSH

```bash
cd /volume1/docker/huat-bot
sudo docker compose build
sudo docker compose run --rm huat-bot python -m huatbot check-site
sudo docker compose up -d
sudo docker compose logs -f
```

### 7. Network access

The NAS needs outbound HTTPS (port 443) to:

| Host | Why |
| --- | --- |
| `www.singaporepools.com.sg` | results, draw lists, next draw and jackpot pages |
| `online2.singaporepools.com` | the official TOTO prize structure page |
| `api.telegram.org` | posting the messages |
| `api.anthropic.com` | only if you turn on the Claude commentary |

If a firewall, an ad blocking DNS filter or a VPN on the NAS blocks any of these, `check-site`
shows which one fails.

## Telegram bot and channel

1. In Telegram, open a chat with **@BotFather** and send `/newbot`. Pick a display name and a username
   ending in `bot`. BotFather replies with the token, which looks like `123456789:AAH...`. Put it in
   `.env` as `TELEGRAM_BOT_TOKEN` and keep it secret.
2. Create a channel (New Channel). Public or private both work.
3. Open the channel info, **Administrators**, **Add Admin**, search for your bot's username and add it
   with permission to **post messages**. A bot can only post in a channel where it is an admin.
4. Find the chat id:
   * Public channel: use its username, for example `TELEGRAM_CHAT_ID=@huatbotchannel`.
   * Private channel: post any message in the channel, then open
     `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser (with your token). Look for
     `"channel_post"` and the `"chat"` inside it. The `"id"` there is the chat id. For a channel it
     starts with `-100`, for example `-1001234567890`. Copy all of it, including the minus sign.
     If the list is empty, post another message in the channel and reload.
5. Run `python -m huatbot run --dry-run` first to see the messages, then a real run to check that they
   arrive.

A group chat works the same way (add the bot to the group, the id comes from `"message"` instead of
`"channel_post"`).

## First run, in this order

Run these once before leaving the bot to its schedule. With the Docker app, type the
`python -m huatbot ...` part in the container terminal. Over SSH, put
`docker compose run --rm huat-bot` in front.

| Order | Command | What it does |
| --- | --- | --- |
| 1 | `python -m huatbot check-site` | Reads every Singapore Pools page the bot uses (the draw list, the latest result, the next draw page, the cascade, Hongbao and special draw lists, the prize page) and prints a PASS or FAIL line for each. Nothing is saved. |
| 2 | `python -m huatbot demo` | Runs the whole pipeline on synthetic draws, with no network, and prints the two messages. Nothing is posted. It writes to a separate demo vault, never to your real one. In Docker that demo vault lives inside the container and is thrown away afterwards. |
| 3 | `python -m huatbot run --dry-run` | A real run: fills the vault, downloads the full history, writes the notes and the report, and prints the two messages instead of posting them. The first download is over 1,100 result pages, so give it about 10 to 20 minutes. Later runs only fetch the new draws. |
| 4 | `python -m huatbot serve` | The scheduler. This is what the container runs by default, so `docker compose up -d` (or the Docker app project) starts it. |

After step 3, open the vault in Obsidian and look at `Huat Bot/Home.md`. Edit
`Huat Bot/Settings.md` to set your jackpot alert, and add your tickets to `Huat Bot/Tickets.md`.

## The vault

Everything lives in one folder inside your vault (`VAULT_FOLDER`, default `Huat Bot`):

```text
MyVault/
├── .obsidian/                          never touched
├── (your own notes)                    never touched
└── Huat Bot/
    ├── Settings.md                     you edit: jackpot alert and options as note properties
    ├── Tickets.md                      you edit: one ticket per table row
    ├── Home.md                    next draw, buy signal, the next big prize, latest result, totals
    ├── Ledger.md                       every ticket, what it won, totals, unreadable lines
    ├── Draws/TOTO/2026-10-01 TOTO 4123.md
    ├── Reports/2026-10-02 1930 Report.md
    ├── Logs/2026-10 Activity.md        a row for every fetch, new draw, ticket check, note, post and error
    └── Data/
        ├── toto.csv
        ├── ledger.csv
        ├── state.json                  next draw dates and what was already posted
        └── prize_rules.json            prize rules cache (checked again after 7 days)
```

| You edit | The bot writes |
| --- | --- |
| `Settings.md`: the properties at the top of the note (see [Settings](#settings)). Read at the start of every run. | `Home.md`, `Ledger.md`, draw notes, reports and the activity log. They are rewritten by the bot, so do not edit them (your changes would be replaced). |
| `Tickets.md`: one row per ticket in the table. | `Data/*.csv` and the JSON files. You can open the CSVs in a spreadsheet, but keep the bot stopped while you edit them. |

`Settings.md` and `Tickets.md` are created with defaults and examples on the first run (or with
`init-vault`), and the bot never overwrites them after that.

A ticket row looks like this (copy the examples at the bottom of `Tickets.md`):

| Game | Draw date | Numbers | Bet type | Cost |
| --- | --- | --- | --- | --- |
| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 | Ordinary | $1 |
| TOTO | 5 Oct 2026 | 3 11 19 27 38 45 49 | System 7 | $7 |

Dates can be written `5 Oct 2026`, `Mon 5 Oct 2026` or `5/10/2026` (day first). Bet types are
Ordinary and System 7 to System 12. A row the bot cannot read is listed in `Ledger.md` with the
reason.

| You change a row | What the ledger does |
| --- | --- |
| Fix a typo in a ticket not checked yet | The old row is dropped from the totals and the corrected one is checked. |
| Correct one field (numbers, bet type or cost) of a ticket already checked | The corrected ticket is checked again and replaces the old one, so it is counted once. |
| Delete the row of a ticket not checked yet | It leaves the totals; put the row back and it returns. |
| Delete the row of a ticket already checked | It stays in the ledger, so your spent and won history stays complete. If you add a near identical ticket for the same draw in the same edit, the bot treats it as a correction and says so in `Ledger.md`, with how to undo it. |

## Schedule and retries

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/draw-day-dark.svg">
    <img alt="On a draw day the bot wakes at 7.30pm, waits for the result, then fetches, checks tickets, projects the jackpot, writes the vault and posts" src="docs/diagrams/draw-day-light.svg" width="100%">
  </picture>
</p>

| When | What happens |
| --- | --- |
| Draw days | Monday and Thursday at 6.30pm Singapore time. Hongbao and special draws on other days are picked up from the next draw page and kept in `Data/state.json`. |
| Run time | 7.30pm Singapore time on a draw day (`RUN_AT`). |
| Retries | If the new result is not on the site yet, the bot checks again every 10 minutes (`RETRY_MINUTES`) for up to 2 hours (`RETRY_HOURS`). |
| Gave up | If a result is still missing after that, the bot posts a short notice and logs it. The next run stores the draw and checks your tickets for it, but message 1 only shows the newest draw. |
| No draw today | Nothing is posted; the activity log gets a row. |
| Restart | If the container starts inside the retry window (say the NAS rebooted at 8pm on a draw day), it runs straight away instead of waiting for the next day. |
| No double posts | `Data/state.json` remembers the newest draw posted, so the scheduler never posts the same draw twice, even after a restart. If only the first message went out (Telegram failed half way), the next run sends just the missing one. A manual `run` posts again on purpose. |

The schedule always uses Singapore time, whatever the NAS time zone is.

## Commands

In Telegram (while `serve` runs, only in the bot's own chat and forum topic):

| Command | What it does |
| --- | --- |
| `/huat` | Resends the latest two messages (result and next draw) |
| `/huatnext` | Resends the next draw message |
| `/huathelp` | Lists the commands |

The last message of every post carries 🔄 Latest result and 🔮 Next draw buttons. Commands only
resend what the last run saved: they never contact Singapore Pools or Claude.

On the command line, all commands are `python -m huatbot <command> [options]`. Each one exits with 0 when it worked and 1
when it did not. `python -m huatbot --help` lists them and `python -m huatbot --version` shows the
version.

| Command | What it does |
| --- | --- |
| `run` | One full run now: fetch new draws, check tickets, work out the next draw and the next big prize, write the vault and post the two messages. A manual run posts even if the newest draw was posted before. |
| `serve` | The scheduler and the container's default command: runs at `RUN_AT` on draw days and retries until the results are out. It only posts draws that were not posted yet. |
| `fetch` | Only update `toto.csv` and the next draw information. No analysis, notes or posting. |
| `report` | Print the full report from the stored data. Nothing is fetched, posted or written to the notes. |
| `check-site` | Read every Singapore Pools page the bot uses and print a PASS or FAIL line for each. Saves nothing. |
| `demo` | The whole pipeline on synthetic data in a separate demo vault (`./demo-vault` unless you pass `--vault`). No network, never posts. |
| `init-vault` | Create the bot folder with `Settings.md` and `Tickets.md` in the vault, then stop. |

Every command accepts these options (after the command name):

| Option | Meaning |
| --- | --- |
| `--dry-run` | Print the Telegram messages instead of posting them. `DRY_RUN=1` does the same, for `run` and `serve`. |
| `--no-fetch` | Use only the stored data and do not contact the site (for `run`; `serve` and `fetch` refuse it). |
| `--no-post` | Do not post or print the Telegram messages. |
| `--vault PATH` | Use this vault folder instead of `VAULT_PATH` (for `demo`, instead of `./demo-vault`). |

Examples:

```bash
python -m huatbot run --dry-run                 # print the messages instead of posting
python -m huatbot run --no-fetch --no-post      # rebuild notes from stored data, post nothing
python -m huatbot report > report.md            # the full report as a file
```

## Settings

### In the vault: Settings.md

These are note properties at the top of `Huat Bot/Settings.md`. Change them in Obsidian (the
properties view or source mode). They apply from the next run. Money can be written `10`, `$10` or
`3,000,000`; yes or no settings take `true` or `false`. A value that cannot be used falls back to the
default, and the problem is listed in the report.

| Property | Default | Allowed | What it does |
| --- | --- | --- | --- |
| `jackpot_alert` | 3,000,000 | 0 or more | A jackpot estimate at or above this makes the buy signal HIGH. |
| `alert_on_special_draws` | true | true or false | Any cascade, Hongbao or special draw also makes the buy signal HIGH. |
| `toto_start_draw` | 2995 | 2995 to 99999 | First draw kept in the history (2995 is the first 6 from 49 draw). |
| `draw_notes_backfill` | 50 | 0 to 5,000 | How many recent draws get their own note the first time the vault is filled. |

Settings from the earlier version (budgets, 4D, backtests, System 7) are no longer used. If they are
still in your `Settings.md`, the report lists them once so you can delete them; they do no harm.

### In .env: environment variables

These are read when the container is created. After changing `.env`, run `docker compose up -d`
(it recreates the container) or redeploy the project in the UGOS Docker app; a plain restart keeps
the old values. Outside Docker the bot reads `.env` from the folder you run it in (or `ENV_FILE`), and values already set in the shell win.

| Variable | Default | What it does |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | none | Bot token from BotFather. Without it (or without the chat id) every run is a dry run, with a warning. |
| `TELEGRAM_CHAT_ID` | none | `@channelname` or the numeric chat id. |
| `VAULT_HOST_PATH` | none, required | The vault folder on the NAS. Compose mounts it at `/vault`. |
| `VAULT_PATH` | `/vault` in Docker, `./vault` otherwise | The vault inside the container. Set by compose, leave it. |
| `VAULT_FOLDER` | `Huat Bot` | The bot's folder inside the vault. |
| `DATA_DIR` | `<vault>/<folder>/Data` | Optional: keep the CSV and JSON files somewhere else. |
| `PUID`, `PGID` | 1000, 10 | The NAS user and group the container runs as. |
| `RUN_AT` | `19:30` | Run time on draw days, 24 hour clock, Singapore time. |
| `RETRY_MINUTES` | 10 | Minutes between checks while waiting for a result (1 to 240). |
| `RETRY_HOURS` | 2 | How long to keep checking (0 to 12). |
| `DRY_RUN` | 0 | 1 (or true, yes, on) prints the messages to the log instead of posting. |
| `COMMENTARY` | `off` | `claude` adds a short comment written by `claude -p`. |
| `INSTALL_CLAUDE` | `false` | `true` builds the image with Node.js and Claude Code. Rebuild after changing it. |
| `ANTHROPIC_API_KEY` | none | Only for `COMMENTARY=claude`. |
| `CLAUDE_BIN` | `claude` | Optional: path of the Claude Code command. |
| `LOG_LEVEL` | `INFO` for `serve`, `WARNING` for other commands | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `ENV_FILE` | `.env` | Outside Docker only: the file the bot reads settings like the Telegram token from. |
| `TZ` | `Asia/Singapore` | Time zone for log timestamps. Set by compose. |

## How the figures are worked out

### Boards sold per draw

The result pages do not show sales, but they can be worked out. Group 3 gets 5.5% of the prize pool,
and the result page shows the Group 3 share amount and how many shares won. Share amount times
winning shares, divided by 5.5%, gives the prize pool; the pool divided by 54% gives the number of
boards sold. When Group 3 had no winner, holds money carried over from the draw before, or a cascade
landed there, Group 4 (3%) is used instead. Draws where neither is clean are left out.

Bigger jackpots sell more boards. For a jackpot the bot uses the median sales of past draws with a
jackpot close to it (within 25%), or a fitted line of sales against jackpot when there are too few.

### What the next draws will do

The jackpot snowballs: when nobody wins Group 1, the whole prize rolls into the next draw and 38% of
that draw's prize pool is added. After three draws in a row with no winner, the fourth is the
cascade draw: if nobody wins that one either, the jackpot goes to the Group 2 winners.

For each draw from the next one up to the cascade draw, the bot works out:

| Figure | How |
| --- | --- |
| Jackpot | The next draw page's estimate for the next draw; for later draws, the jackpot before plus 38% of 54% of the boards that jackpot usually sells, rounded to the nearest $1,000. |
| Won | The chance at least one board in the whole draw matches all six numbers: 1 minus e to the power of minus (boards sold divided by 13,983,816). This treats every board as a random pick, so it is a fair estimate rather than an exact figure. |
| Unwon | The chance nobody has won the jackpot before that draw: the Won chances of the draws before it, multiplied out. |
| Return per $1 | As in the buy signal below, at that jackpot. |

The next big prize is the biggest jackpot on that path, usually the cascade draw. A Hongbao or special
draw has its own advertised jackpot, so it is shown as a single draw, and draws announced on the next
draw page are listed too. When the next draw page cannot be read, the next jackpot is worked out from
the stored results the same way, and the report says so.

### Buy signal

1. **Boards sold at this jackpot**, as above.
2. **Return per $1.** The average amount each $1 board brings back: Group 1 allowing for the chance of
   sharing it with other winners at that level of sales, Groups 2 to 4 (shares of the pool, also split
   between winners), the fixed Groups 5 to 7 ($50, $25, $10, worth about $0.24 per $1 together), and on
   cascade and Hongbao draws the chance that the jackpot cascades into Group 2.
3. **Label.** HIGH when the jackpot estimate reaches your `jackpot_alert`, or when the draw is a
   cascade, Hongbao or special draw and `alert_on_special_draws` is true. LOW when the jackpot is the
   $1,000,000 minimum. MEDIUM otherwise.

The return per $1 is usually well below $1. HIGH means a better than usual draw to play if you were
going to play anyway, not a good investment.

### Jackpot history

From every stored draw: the share of draws where Group 1 was won, how many draws a jackpot lasted on
average (until it was won or cascaded), how many cascaded, the typical Group 1 prize when it was won,
the last time it was won and the five biggest jackpots.

## Optional Claude commentary

The bot can add two or three sentences of plain commentary to the second message. Python computes every
figure first; Claude Code (`claude -p`) only receives those figures as JSON and is asked for at most 60
words, with no new numbers and never a number to pick. A reply that contains any number not in the
figures is thrown away, and if anything fails the messages go out without commentary.

1. In `.env`, set `INSTALL_CLAUDE=true`, `COMMENTARY=claude` and `ANTHROPIC_API_KEY=...`.
2. Rebuild: `docker compose up -d --build` (or redeploy the project in the Docker app).
3. Check: `docker compose run --rm huat-bot claude --version`

Each run makes one short `claude -p` call. If `COMMENTARY=claude` but the image was built without
Claude Code, the log says the command was not found and the messages go out without commentary.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `check-site` shows FAIL, or the log says the site is unreachable | The NAS needs outbound HTTPS to `www.singaporepools.com.sg` and `online2.singaporepools.com`. Test from the NAS: `docker compose run --rm huat-bot python -c "import requests; print(requests.get('https://www.singaporepools.com.sg', timeout=30).status_code)"`. Check firewall rules, DNS filters and any VPN on the NAS. When the site cannot be reached the bot carries on with the data it already has and says so. Only a first run with no data at all stops, with a short Telegram notice. A single draw page that keeps failing (gone, or showing something the bot cannot read) is skipped after 3 runs in a row (listed under `skip` in `Data/state.json`; delete it there to try again). Fetch trouble such as timeouts, rate limits or server errors never puts a draw on that list. |
| Only the prize pages fail | They are not critical. The bot uses its built in prize values and the report says the rules were not confirmed. |
| `Permission denied` writing to `/vault` | `PUID` and `PGID` must be a NAS user with read and write access to the vault folder. Over SSH, `ls -ln /volume1/Obsidian/MyVault` shows the owner numbers. Give your user read and write on the shared folder in Control Panel, or if the folder is yours, `sudo chown -R 1000:10 "/volume1/Obsidian/MyVault/Huat Bot"` with your own numbers. |
| Files appear on the NAS but not on the phone | The vault sync is not picking them up. See [step 3](#3-make-sure-the-vault-is-synced-to-your-devices). |
| Nothing was posted | Check the activity log in `Huat Bot/Logs`. Common reasons: the result was already posted (the log says "Nothing new since the last post"; the scheduler never posts a draw twice, see `last_posted` in `Data/state.json`; run `python -m huatbot run` by hand to post again); `DRY_RUN=1` (after changing `.env`, run `docker compose up -d`; a restart keeps the old value); the token or chat id is missing (the run becomes a dry run and says so); it was not a draw day; the result was still not out after the retry window (you get a short notice instead). |
| Telegram says `chat not found` or `Forbidden` | The bot is not an admin of the channel, or the chat id is wrong. For a private channel the id starts with `-100`. |
| Times in the log are 8 hours off | The container clock is in UTC. Compose sets `TZ=Asia/Singapore`; if you run the image some other way, pass `-e TZ=Asia/Singapore`. The schedule itself always uses Singapore time, and `RUN_AT` is Singapore time. Also check the NAS clock (Control Panel, time settings, sync with an NTP server). |
| The run happens at the wrong time | `RUN_AT` is a 24 hour time such as `19:30`. A bad value falls back to 19:30 with a warning in the log. |
| Settings are ignored | The report and the log list every setting that could not be used. Properties must be in the block at the very top of `Settings.md`. |
| Container shows unhealthy | The health check only imports the package. Rebuild the image (`docker compose up -d --build`). |

See the container log with `docker compose logs -f huat-bot` (or the **Log** tab of the container in
the Docker app). Set `LOG_LEVEL=DEBUG` for more detail.

## Updating

Over SSH:

```bash
cd /volume1/docker/huat-bot
git pull
sudo docker compose up -d --build
```

With the Docker app, replace the project files with the new version, then rebuild or redeploy the
project. Your data stays in the vault, so nothing is lost.

## Upgrading from the 4D and suggestions version

Earlier versions also followed 4D and suggested numbers. This version follows TOTO only and never
suggests numbers. After `git pull` and a rebuild:

* Your TOTO history, tickets, ledger and settings carry on as they are.
* Old 4D tickets already checked stay in the ledger and its totals. 4D tickets not checked yet are
  marked as no longer tracked, and new 4D rows in `Tickets.md` are listed in `Ledger.md` as not read.
* The bot no longer writes `Data/fourd.csv`, `Data/backtest_cache.json`, `Draws/4D` or `Suggestions`.
  It leaves them alone, so delete them whenever you like.
* Settings from the old version are listed once in the report so you can delete them.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests run offline: saved HTML pages in `tests/fixtures` and synthetic draws from `huatbot.synth`.
GitHub Actions runs them on Python 3.11 and 3.12 for every push and pull request, and checks that the
Docker image builds. The product requirements are in [docs/PROMPT.md](docs/PROMPT.md) and the diagrams
in [docs/diagrams](docs/diagrams).
