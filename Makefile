# Aegivis — Common developer commands
# Usage: make <target>

.PHONY: up down build restart logs migrate seed health test test-proxy test-backend test-sdk test-dashboard lint format shell-proxy shell-backend clean

# ── Docker Compose ─────────────────────────────────────────────────────────────

up:	## Start all services in detached mode
	docker compose up -d

down:	## Stop all services
	docker compose down

build:	## Rebuild all service images
	docker compose build proxy backend dashboard

restart:	## Rebuild and restart proxy, backend, and dashboard
	docker compose build proxy backend dashboard
	docker compose up -d --no-deps proxy backend dashboard

logs:	## Tail logs for all services (Ctrl-C to stop)
	docker compose logs -f

logs-proxy:	## Tail proxy logs only
	docker compose logs -f proxy

logs-backend:	## Tail backend logs only
	docker compose logs -f backend

# ── Database ───────────────────────────────────────────────────────────────────

migrate:	## Run Alembic migrations to HEAD
	docker compose exec backend alembic upgrade head

migrate-to:	## Run migrations to a specific revision: make migrate-to REV=0005
	docker compose exec backend alembic upgrade $(REV)

seed:	## Insert default org and API key
	bash scripts/seed-data.sh

# ── Health ─────────────────────────────────────────────────────────────────────

health:	## Smoke-test all services
	bash scripts/health-check.sh

# ── Tests (no Docker required) ────────────────────────────────────────────────

test:	## Run all unit test suites (proxy + backend + SDK)
	$(MAKE) test-proxy
	$(MAKE) test-backend
	$(MAKE) test-sdk
	@echo ""
	@echo "==> All unit suites passed."

test-proxy:	## Run proxy unit tests
	cd proxy && python -m pytest tests/ \
		--ignore=tests/test_external_datasets.py \
		--ignore=tests/test_benchmark.py \
		-q

test-backend:	## Run backend unit tests
	cd backend && python -m pytest tests/ -q

test-sdk:	## Run SDK unit tests
	cd sdk && python -m pytest tests/ --ignore=tests/test_integration.py -q

test-integration:	## Run integration tests (requires docker compose up)
	pytest tests/integration/ -m integration -v

test-dashboard:	## Run dashboard unit tests (vitest)
	cd dashboard && npm run test

test-benchmark:	## Run proxy benchmark (requires transformers + torch)
	cd proxy && BENCHMARK_USE_CLASSIFIER=1 python -m pytest tests/test_benchmark.py -v

# ── Linting ────────────────────────────────────────────────────────────────────

lint:	## Run Ruff on proxy, backend, and SDK
	ruff check proxy/app/ backend/app/ sdk/aegivis/

lint-dashboard:	## TypeScript type-check the dashboard
	cd dashboard && npx tsc --noEmit

format:	## Auto-fix Ruff lint issues
	ruff check --fix proxy/app/ backend/app/ sdk/aegivis/

# ── Dev shells ─────────────────────────────────────────────────────────────────

shell-proxy:	## Open a shell in the running proxy container
	docker compose exec proxy bash

shell-backend:	## Open a shell in the running backend container
	docker compose exec backend bash

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:	## Full first-run setup (build, start, migrate, seed)
	bash scripts/dev-setup.sh

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:	## Stop services and remove volumes (WARNING: deletes all data)
	@echo "WARNING: This will delete all data in PostgreSQL and Redis volumes."
	@read -r -p "Continue? [y/N] " confirm && [ "$confirm" = "y" ] || exit 1
	docker compose down -v

# ── Help ───────────────────────────────────────────────────────────────────────

help:	## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
