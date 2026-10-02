---
tags: [active]
updated: 2026-10-03
---
# App Overview

Python 3.12, Docker (`docker-compose.yml`, `pull_policy: build`). Runs `python -m huatbot serve` on the NAS.

## Modules (`huatbot/`)
- [[huatbot/scheduler.py]] — draw days (Mon/Thu), waits for the result, retries.
- [[huatbot/runner.py]] — one run: fetch → tickets → outlook → notes → Telegram. Saves the messages for `/huat`.
- [[huatbot/fetch.py]], [[huatbot/parse.py]], [[huatbot/http.py]] — Singapore Pools pages, rate limited.
- [[huatbot/outlook.py]], [[huatbot/buysignal.py]], [[huatbot/sales.py]] — jackpot projection, return per $1.
- [[huatbot/report.py]] — full report + 2 Telegram cards (fleet style: `SECTION_TITLES`, divider, expandable blockquote).
- [[huatbot/telegram.py]] — sendMessage (forum topic via `chat/-topic`), getUpdates, answerCallbackQuery.
- [[huatbot/listener.py]] — `/huat`, `/huatnext`, `/huathelp` + 🔄/🔮 buttons; own topic only; whitelisted callbacks; no fetch or LLM.
- [[huatbot/watch.py]] — hourly next draw page check within `ALERT_HOURS` (9-21), 🆕 card on jackpot estimate change or special draw; snapshot in `Data/watch.json`.
- [[huatbot/commentary.py]] — optional `claude -p --model haiku --no-session-persistence`, numbers validated, vault activity excerpt (≤4k chars) as memory.
- [[huatbot/vault.py]], [[huatbot/notes.py]] — Obsidian output.

## Vault output (NAS `/volume1/James/Obsidian/Huat Bot`, `VAULT_FOLDER=.`)
`Home.md`, `Ledger.md`, `Tickets.md` (you edit), `Settings.md` (you edit), `Activity/YYYY/MM/YYYY-MM-DD.md`, `Reports/YYYY/MM/`, `Draws/TOTO/`, `Data/`.

## Deploy
NAS stack `/volume1/docker/huat-bot` (Dockge). Bot @jameskoh_huat_bot → James Channel topic (see `.env` TELEGRAM_CHAT_ID). Update: `git archive` → tar over SSH (keep `.env`), `docker compose up -d --build`.
