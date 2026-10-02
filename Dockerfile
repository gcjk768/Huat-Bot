# Huat Bot: Singapore Pools 4D and TOTO analyst for a home NAS.
#
#   docker build -t huat-bot .
#   docker build -t huat-bot --build-arg INSTALL_CLAUDE=true .   # adds Node.js and the Claude Code CLI
#
# Build args: INSTALL_CLAUDE (true/false), NODE_MAJOR (24), CLAUDE_CODE_VERSION (latest).
#
# The image runs fine as any user id (docker compose sets `user: PUID:PGID`), so files the
# bot writes into the mounted Obsidian vault belong to you, not to root.

# INSTALL_CLAUDE picks the final stage below, so it must be exactly "true" or "false".
ARG INSTALL_CLAUDE=false
# Node.js major version for the optional Claude Code CLI (an LTS line).
ARG NODE_MAJOR=24

# Official Node.js image, only used as a source of the node binary and npm when
# INSTALL_CLAUDE=true. BuildKit (the default builder) skips it otherwise.
FROM node:${NODE_MAJOR}-trixie-slim AS node


# Base: Python, the bot and its dependencies
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="Huat Bot" \
      org.opencontainers.image.description="Singapore Pools 4D and TOTO analyst that posts to Telegram and keeps everything in an Obsidian vault" \
      org.opencontainers.image.source="https://github.com/gcjk768/Huat-Bot"

ENV TZ=Asia/Singapore \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/huatbot \
    VAULT_PATH=/vault

# Time zone data and CA certificates. The python slim image already ships both, so apt is
# only used if a future base image drops them (saves a slow apt update on a NAS). Not pinned on
# purpose: time zone rules and root certificates must stay current.
# hadolint ignore=DL3008
RUN set -eux; \
    if ! dpkg -s tzdata ca-certificates >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends tzdata ca-certificates; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime; \
    echo "$TZ" > /etc/timezone

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY huatbot/ ./huatbot/
COPY docs/ ./docs/

# Drop any host bytecode that slipped into the build context, then precompile as root so
# a non root user starts fast without needing write access to /app.
# These folders are world writable (sticky, like /tmp) so any PUID:PGID works:
#   HOME              config of the optional Claude CLI
#   /vault            the mount point; without a mount the bot still runs (data is thrown away)
#   /app/demo-vault   where `demo` writes its synthetic vault (thrown away with the container)
RUN set -eux; \
    find /app -name __pycache__ -type d -prune -exec rm -rf {} +; \
    python -m compileall -q /app/huatbot; \
    mkdir -p "$HOME" /vault /app/demo-vault; \
    chmod 1777 "$HOME" /vault /app/demo-vault; \
    python -c "import huatbot"


# Variant without Claude (default)
FROM base AS claude-false


# Variant with Node.js and the Claude Code CLI for the optional `claude -p` commentary
FROM base AS claude-true

# Claude Code version from npm; set an exact version (for example 2.1.287) to pin it.
ARG CLAUDE_CODE_VERSION=latest

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm

ENV DISABLE_AUTOUPDATER=1 \
    DISABLE_TELEMETRY=1

RUN set -eux; \
    ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm; \
    ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx; \
    npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"; \
    npm cache clean --force; \
    rm -rf "$HOME/.npm" /root/.npm /tmp/*; \
    node --version; \
    claude --version


# Final image: one of the two variants above (a stage name, not an image to pull)
# hadolint ignore=DL3006
FROM claude-${INSTALL_CLAUDE}

# Default user when started without `user:` (matches the compose defaults PUID 1000, PGID 10).
USER 1000:10

# Python installs a handler for SIGINT, so `docker stop` ends the bot at once even when it
# runs as PID 1 (compose also sets init: true).
STOPSIGNAL SIGINT

# Cheap liveness check: the package imports (no network, no vault access).
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD ["python", "-c", "import huatbot"]

CMD ["python", "-m", "huatbot", "serve"]
