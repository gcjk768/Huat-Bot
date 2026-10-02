---
tags: [active]
updated: 2026-10-03
---
# Changelog

## 2026-10-03
- Telegram cards now match the SG car tracker (@jameskoh_sgcar_bot): no `<pre>` tables; one line per prize group (🏆/🥈/🥉/🎟), one block per ticket (🟢/⚪ + 💰/🎯 line), projection as 📅/🌊 lines, 🌐 Singapore Pools result link; buy-signal reason, return per $1, sales method and legend moved into the closing expandable quote.
- Telegram cards in the fleet style (emoji titles, ━ dividers, 🟢/🟡/🔴 buy signal, 🟢▲/🔴▼ jackpot change, odds + history + commentary in an expandable blockquote).
- Forum topic support (`-100…/topic` or `TELEGRAM_THREAD_ID`); inline buttons on the last message.
- New command listener: `/huat`, `/huatnext`, `/huathelp`.
- Vault: `Activity/YYYY/MM/YYYY-MM-DD.md` bullet lines, `Reports/YYYY/MM/`, `Dashboard.md` → `Home.md`.
- Commentary: haiku by default, no session persistence, activity memory, OTLP usage telemetry (`bot=huat-bot`).
- Finding watcher (`huatbot/watch.py`): hourly light check of the next draw page, 🆕 card when the jackpot estimate moves or a Cascade/Hongbao/special draw is announced; only within `ALERT_HOURS` (default 9-21, clear of the trading desk's US session from 21:30).
- Deployed to the NAS.
