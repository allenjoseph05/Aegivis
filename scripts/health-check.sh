#!/usr/bin/env bash
# health-check.sh — Smoke-test all Aegivis services
# Usage: bash scripts/health-check.sh
set -euo pipefail

API_KEY="${AEGIVIS_API_KEY:-dev-dashboard-key}"
BACKEND_URL="${AEGIVIS_BACKEND_URL:-http://localhost:8000}"
PROXY_URL="${AEGIVIS_PROXY_URL:-http://localhost:8080}"
DASHBOARD_URL="${AEGIVIS_DASHBOARD_URL:-http://localhost:5173}"

PASS=0
FAIL=0

check() {
  local name="$1"
  local url="$2"
  local extra_args="${3:-}"
  local expected="${4:-200}"

  # shellcheck disable=SC2086
  status=$(curl -s -o /dev/null -w "%{http_code}" $extra_args "$url" 2>/dev/null || echo "000")
  if [ "$status" = "$expected" ]; then
    echo "  [PASS] $name ($status)"
    PASS=$((PASS + 1))
  else
    echo "  [FAIL] $name — expected $expected, got $status"
    FAIL=$((FAIL + 1))
  fi
}

echo "==> Aegivis Health Check"
echo ""

echo "-- Backend ($BACKEND_URL) --"
check "Health endpoint"         "$BACKEND_URL/health"        "-H 'X-API-Key: $API_KEY'"
check "Sessions list"           "$BACKEND_URL/v1/sessions"   "-H 'X-API-Key: $API_KEY'"
check "Violations list"         "$BACKEND_URL/v1/violations" "-H 'X-API-Key: $API_KEY'"
check "Agents list"             "$BACKEND_URL/v1/agents"     "-H 'X-API-Key: $API_KEY'"
check "Auth required (no key)"  "$BACKEND_URL/v1/sessions"  ""                           "401"

echo ""
echo "-- Proxy ($PROXY_URL) --"
check "Proxy root (method check)" "$PROXY_URL/" "" "404"

echo ""
echo "-- Dashboard ($DASHBOARD_URL) --"
check "Dashboard index" "$DASHBOARD_URL/" ""

echo ""
echo "==> Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  echo "    Run 'docker compose logs' for details."
  exit 1
fi
