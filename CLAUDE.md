# Project conventions for Claude Code

Standing instructions from the repository owner. Read `docs/vault` before working on a task.

## Web access (Agent Reach)

* When you need the web during development (finding a new source, checking why a page changed,
  researching), use Agent Reach on James's PC (`/agent-reach` skill, `agent-reach doctor` first).
  Backends: `curl https://r.jina.ai/<URL>` for any page, Exa through `mcporter` for search,
  feedparser for RSS, yt-dlp for YouTube, `gh` for GitHub, OpenCLI for Reddit and X.
* New source code in this app starts from those same backends (an official API or feed first,
  then Jina Reader or feedparser) before writing an ad hoc scraper.
* Do not replace a scraper that already works and has tests. Agent Reach is not installed in the
  NAS container, so runtime code must keep working without it.
* Never use Jina or any proxy to get around a 403, a bot challenge or a robots.txt block.
  Credentials stay in `~/.agent-reach/`, never in this repo.
* Setup and fixes: Obsidian Vault note `Tools/Agent Reach Setup`.
