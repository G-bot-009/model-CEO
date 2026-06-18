#!/usr/bin/env bash
# Auto-deploy G Office: if the tracked branch moved on origin, pull + rebuild.
# Set up once with cron (see deploy/README.md). Safe to run every minute.
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root (deploy/ is one level down)

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git fetch --quiet origin "$BRANCH" || exit 0
LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"

if [ "$LOCAL" != "$REMOTE" ]; then
  echo "$(date '+%F %T') new commit $REMOTE on $BRANCH — deploying…"
  git pull --ff-only origin "$BRANCH"
  docker compose up -d --build
  echo "$(date '+%F %T') ✅ deployed."
fi
