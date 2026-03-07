#!/usr/bin/env bash
# seed-data.sh — Insert default org and API key if they don't already exist
# Usage: bash scripts/seed-data.sh
set -euo pipefail

DOCKER_COMPOSE="docker compose"
if ! docker compose version &>/dev/null 2>&1; then
  DOCKER_COMPOSE="docker-compose"
fi

echo "==> Seeding default org and API key..."

$DOCKER_COMPOSE exec -T postgres psql -U aegivis -d aegivis <<'SQL'
-- Default org / key (idempotent — safe to run multiple times)
INSERT INTO api_keys (org_id, key_hash, name)
VALUES (
  'default-org',
  encode(sha256('dev-dashboard-key'::bytea), 'hex'),
  'Development dashboard key'
)
ON CONFLICT (key_hash) DO NOTHING;

-- Print confirmation
SELECT org_id, name, created_at FROM api_keys WHERE org_id = 'default-org';
SQL

echo "==> Seeding complete."
echo ""
echo "  Use this header to authenticate: X-API-Key: dev-dashboard-key"
