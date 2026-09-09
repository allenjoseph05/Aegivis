#!/usr/bin/env bash
# dev-setup.sh — One-command local development setup for Aegivis
# Usage: bash scripts/dev-setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Aegivis — Dev Setup"
echo ""

# --- Check prerequisites ---
for cmd in docker git python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' not found. Please install it and re-run." >&2
    exit 1
  fi
done

DOCKER_COMPOSE="docker compose"
if ! docker compose version &>/dev/null 2>&1; then
  if command -v docker-compose &>/dev/null; then
    DOCKER_COMPOSE="docker-compose"
  else
    echo "ERROR: 'docker compose' (v2) not found." >&2
    exit 1
  fi
fi

# --- Copy .env if not present ---
if [ ! -f "$ROOT_DIR/.env" ]; then
  echo "--> Copying .env.example → .env"
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "    Review .env and fill in any required values (API keys, secrets)."
else
  echo "--> .env already exists, skipping copy."
fi

# --- Build and start services ---
echo ""
echo "--> Building Docker images (this may take a few minutes on first run)..."
cd "$ROOT_DIR"
$DOCKER_COMPOSE build --quiet

echo ""
echo "--> Starting all services..."
$DOCKER_COMPOSE up -d

echo ""
echo "--> Waiting for backend to be healthy..."
for i in $(seq 1 30); do
  if curl -sf -H "X-API-Key: dev-dashboard-key" http://localhost:8000/health &>/dev/null; then
    echo "    Backend is ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Backend did not become healthy within 60s. Check 'docker compose logs backend'." >&2
    exit 1
  fi
  sleep 2
done

# --- Run migrations ---
echo ""
echo "--> Running database migrations..."
$DOCKER_COMPOSE exec -T backend alembic upgrade head

# --- Seed default data ---
echo ""
echo "--> Seeding default data..."
bash "$SCRIPT_DIR/seed-data.sh" || echo "    (seed-data.sh skipped — backend may already be seeded)"

echo ""
echo "==> Setup complete!"
echo ""
echo "  Dashboard:  http://localhost:5173"
echo "  Backend:    http://localhost:8000/docs"
echo "  Proxy:      http://localhost:8080"
echo ""
echo "  API key for backend/dashboard: dev-dashboard-key"
echo "  Header: X-API-Key: dev-dashboard-key"
echo ""
echo "  To stop:   docker compose down"
echo "  To reset:  docker compose down -v   (WARNING: deletes all data)"
