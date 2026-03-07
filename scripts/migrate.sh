#!/usr/bin/env bash
# migrate.sh — Run Alembic database migrations inside the running backend container
# Usage: bash scripts/migrate.sh [revision]
#   revision: optional target revision (default: "head")
set -euo pipefail

REVISION="${1:-head}"

DOCKER_COMPOSE="docker compose"
if ! docker compose version &>/dev/null 2>&1; then
  DOCKER_COMPOSE="docker-compose"
fi

echo "==> Running migrations to: $REVISION"
$DOCKER_COMPOSE exec backend alembic upgrade "$REVISION"
echo "==> Migrations complete."
