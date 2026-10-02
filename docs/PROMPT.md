# Huat Bot product spec

This is the analyst prompt the bot was first built from, kept here so the code can be checked
against it, followed by the changes asked for since. The latest change (TOTO only, no number
suggestions, at the end) overrides the original prompt where they differ. The bot computes
everything in Python. The prompt is the requirement, not something the bot sends to a model.

## Analyst prompt

```
ROLE
You are my Singapore Pools analyst. Scrape 4D and TOTO results, analyse the history, suggest numbers to buy, and tell me the latest and next prizes so I can decide whether to buy.

MY SETTINGS (edit before running)
Budget per draw: TOTO $10, 4D $5
Jackpot alert: $3,000,000 or more, or any cascade, Hongbao or special draw
History: TOTO from draw 2995 (9 Oct 2014, first draw of the current 6 from 49 format) to the latest. 4D the latest 1,000 draws
My tickets: none yet. Format when I add them: game, draw date, numbers, bet type, cost

STEP 1: GET THE DATA
Write and run a Python script. Save toto.csv and fourd.csv. If the files already exist, fetch only the missing draws.

Draw lists (about 3 years, each <option> has value = draw number):
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_draw_list_en.html
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/fourd_result_draw_list_en.html

One page per draw. The sppl value is base64 of the text "DrawNumber=NNNN". It also works for draws older than the lists, so count draw numbers down to reach older history.
TOTO: https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl=...
  Read th.drawDate, th.drawNumber, td.win1 to td.win6, td.additional, td.jackpotPrize, and every row of table.tableWinningShares (group, share amount, number of winning shares)
4D: https://www.singaporepools.com.sg/en/product/Pages/4d_results.aspx?sppl=...
  Read th.drawDate, th.drawNumber, td.tdFirstPrize, td.tdSecondPrize, td.tdThirdPrize, the 10 numbers in tbody.tbodyStarterPrizes and the 10 in tbody.tbodyConsolationPrizes

Next draw information:
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_next_draw_estimate_en.html
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/fourd_next_draw_info_en.html

TOTO draw types (tag each draw as normal, cascade, Hongbao or special):
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_cascade_draw_list_en.html
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_hongbao_draw_list_en.html
https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_special_draw_list_en.html

Official prize rules (read these, do not rely on memory):
https://online2.singaporepools.com/en/lottery/toto-prize-structure
https://online2.singaporepools.com/en/lottery/4d-prize-structure

Scraping rules: use a browser User Agent, parse with BeautifulSoup, make at most 4 requests at a time with a short pause, and slow down and resume if the site returns errors. Confirm the draw number on each page matches the one requested. Confirm the newest row in each CSV matches the latest draw on the site. If the shell cannot reach the site, say so and work only from what web fetch can read.

STEP 2: ANALYSE TOTO (numbers 1 to 49)
1. Frequency of each number: all history, last 100 draws, last 50 draws
2. Overdue numbers: draws since each number last appeared
3. Most common pairs
4. Usual shape of a winning set: odd/even split, low (1 to 24) vs high (25 to 49) split, and the sum range covering the middle 50% of draws
5. Fairness test: run a chi square test on the frequencies and tell me plainly whether hot and cold numbers are real or just noise
6. Crowd score (which numbers other people buy most). Confirm the percentages on the prize rules page, then for each draw:
   prize pool = Group 3 share amount x Group 3 winning shares / 0.055 (use Group 4 and 0.03 if Group 3 has no winner)
   boards sold = prize pool / 0.54
   expected Group 7 winners = boards sold x 229,600 / 13,983,816
   crowd ratio = actual Group 7 winning shares / expected Group 7 winners
   Run a ridge regression of the crowd ratio on which six numbers were drawn to score each number. A high score means many people buy it, so prizes are split more ways. The average crowd ratio over all draws should be close to 1. If it is not, tell me.

STEP 3: ANALYSE 4D
1. Digit frequency for each position (thousands, hundreds, tens, units) across all 23 prizes per draw
2. Numbers that won more than once, and the most frequent digit sets ignoring order
3. Chi square test on the digits, with the same plain verdict
4. Bet type value: from the official prize table, work out the average return per $1 for Big, Small and iBet, and say which gives the most back

STEP 4: BACKTEST
For each of the last 300 TOTO draws, build each strategy's set using only the draws before it, then score it against the real result using the real prize amounts for that draw. Compare with 1,000 random sets per draw. Show cost, winnings and return per $1 for Hot, Overdue, Balanced, Low Crowd and Random. Do the same for the 4D picks against random numbers. Give a one line verdict per strategy.

STEP 5: SUGGEST NUMBERS (stay within my budget)
TOTO: one set of 6 per strategy, each with a one line reason:
  a) Hot: most frequent in recent draws
  b) Overdue: longest gaps
  c) Balanced: matches the usual odd/even, low/high and sum ranges
  d) Low Crowd: balanced, built from numbers with a low crowd score, with no obvious pattern (no sequences, not all birthday numbers, not a repeat of the last draw)
  Say which sets to buy for my budget. Offer one System 7 set with its cost if the budget allows.
4D: 5 numbers with reasons, the bet type for each (Big, Small or iBet) and the stake, within my budget.

STEP 6: PRIZES AND BUY SIGNAL
1. Latest TOTO draw: date, draw number, winning numbers, additional number, Group 1 prize, full winning shares table
2. Next TOTO draw: date and time, estimated jackpot, draw type, and how many draws in a row have had no Group 1 winner
3. Buy signal: estimate the average return per $1 at this jackpot, including the chance of sharing Group 1 based on typical sales at this jackpot level in the history. Label it HIGH when it meets my jackpot alert, LOW when the jackpot is at the $1,000,000 minimum, otherwise MEDIUM.
4. Latest 4D draw: all 23 winning numbers
5. Next 4D draw: date and time, and the current prize table per $1 for Big and Small

STEP 7: MY TICKETS
Check any tickets in MY SETTINGS against the results and tell me what each one won. Keep ledger.csv with every ticket, its cost and its winnings. Show total spent, total won and net so far.

STEP 8: BE HONEST
Every draw is independent, so past results do not change the odds. Say this once, clearly. State the odds: TOTO Group 1 is 1 in 13,983,816 per board, any TOTO prize is about 1 in 54, and a 4D Big bet wins some prize 23 times in 10,000. If the backtest shows a strategy does no better than random, say so. Never suggest spending above my budget.

OUTPUT
A short report in this order: 1) Next draws, prizes and buy signal 2) Latest results and my ticket check 3) Suggested numbers and total cost 4) Backtest scoreboard 5) Key stats tables 6) Odds note. Use tables. Do not use dashes in the write up.
```

## Build it as a bot

```
BUILD IT AS A BOT
Build this as a Python project that runs in Docker on my UGREEN NAS and posts to my Telegram channel.
1. Schedule: run at 7.30pm Singapore time on draw days (TOTO Mon and Thu, 4D Wed, Sat and Sun, plus special draws). If the new result is not published yet, retry every 10 minutes for up to 2 hours.
2. Storage: keep toto.csv, fourd.csv and ledger.csv in a mounted volume so the history survives restarts.
3. Telegram: read TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from a .env file. Post message 1 with the result and my ticket check, message 2 with the next draw, jackpot and buy signal, message 3 with the suggested numbers and cost. Keep each message under 4,000 characters.
4. All numbers must come from the Python code. If you use claude -p, use it only to write the short commentary from the computed figures.
5. Give me the Dockerfile, the docker compose file, a README with setup steps, and a dry run mode that prints the messages without posting.
```

## Addition: Obsidian vault on the NAS

> add in use obsidian vault inside nas for read and write for all movement

The bot treats an Obsidian vault on the NAS as its single home for everything it reads and writes:

* Reads: `Settings.md` (budgets, jackpot alert, history window as note properties) and `Tickets.md`
  (your tickets, one per table row), both edited in Obsidian.
* Writes: the CSV data (`Data/toto.csv`, `Data/fourd.csv`, `Data/ledger.csv`), one note per draw,
  suggestion notes per upcoming draw, a full report per run, `Dashboard.md`, `Ledger.md`, and an
  activity log (`Logs/YYYY-MM Activity.md`) with a row for every fetch, new draw, ticket check,
  note written, message posted and error.

## Change: TOTO only, no number suggestions

> remove possible winning number. remove 4D too, i want to know about Toto will do and the next big
> prize. improve the code

What this changes in the prompt above:

* **Removed:** everything about 4D (data, analysis, picks, tickets, next draw, schedule days),
  STEP 2 items 1 to 6 (frequency, overdue numbers, pairs, winning set shape, chi square, crowd
  scores), STEP 4 (backtest) and STEP 5 (suggested numbers, System 7 offer), and the budgets. Every
  draw is independent, so the bot never suggests numbers to pick.
* **Kept:** fetching the TOTO history, the official prize rules, the latest result with its winning
  shares table, the next draw with its jackpot and draw type, the buy signal (return per $1 at the
  jackpot, including the chance of sharing Group 1 at typical sales), the ticket check and ledger,
  the honest odds note, the vault, the schedule (Monday and Thursday plus special draws), the
  retries, dry run mode and no dashes in the write up.
* **Added: what TOTO will do.** A draw by draw projection from the next draw to the cascade draw:
  the jackpot if nobody wins it first (38% of 54% of the boards sold added at each rollover), the
  boards each draw is likely to sell (worked out from the Group 3 or Group 4 winning shares of past
  draws with a similar jackpot), the chance somebody wins Group 1 at that draw and the chance it is
  still unwon by then.
* **Added: the next big prize.** The biggest jackpot on that path, when it comes and how likely it is
  to get that far, plus any announced Hongbao or special draw.
* **Added: jackpot history.** How often Group 1 is won, how long a jackpot lasts, how many cascaded,
  the typical prize when won, the last win and the biggest jackpots.
* **Telegram:** two messages. Message 1: the latest result and the ticket check. Message 2: the next
  draw, its jackpot, the buy signal and the next big prize.
* **Report order:** 1) Next draw and the next big prize 2) Latest result and my ticket check
  3) Jackpot history 4) Odds note.
* **Settings.md:** jackpot alert, special draw alert, first draw kept, draw notes backfill. Old
  settings are listed once as no longer used.
