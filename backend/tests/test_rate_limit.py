"""Tests for the sliding-window rate limiting middleware."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import BackendSettings


def _make_settings(**kwargs) -> BackendSettings:
    """Build a BackendSettings with test defaults, overriding specified fields."""
    return BackendSettings(
        rate_limit_enabled=kwargs.get("rate_limit_enabled", True),
        rate_limit_per_minute=kwargs.get("rate_limit_per_minute", 300),
        rate_limit_ingest_per_minute=kwargs.get("rate_limit_ingest_per_minute", 3000),
        database_url="postgresql+asyncpg://abb:abb@localhost:5432/agentblackbox",
    )


class TestRateLimitDisabled:
    def _make_disabled_app(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from app.middleware.rate_limit import RateLimitMiddleware

        mini = FastAPI()

        @mini.get("/v1/metrics/overview")
        def metrics():
            return {"blocked_count": 0}

        mini.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )
        # Rate limit=1 but disabled — should allow unlimited requests
        s = _make_settings(rate_limit_enabled=False, rate_limit_per_minute=1)
        mini.add_middleware(RateLimitMiddleware, settings=s)
        return mini

    def test_rate_limit_disabled_bypasses_limit(self):
        """When rate_limit_enabled=False, no request should be rate-limited."""
        mini = self._make_disabled_app()
        client = TestClient(mini)
        # Send 5 requests — all should pass even though configured limit=1
        for i in range(5):
            resp = client.get("/v1/metrics/overview")
            assert resp.status_code == 200, f"Request {i} expected 200, got {resp.status_code}"


class TestRateLimitEnforced:
    """Tests using a mini FastAPI app to avoid state leakage between test classes."""

    def _make_app(self, read_limit: int = 300, ingest_limit: int = 3000):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from app.middleware.rate_limit import RateLimitMiddleware

        mini = FastAPI()

        @mini.get("/health")
        def health():
            return {"status": "ok"}

        @mini.get("/v1/metrics/overview")
        def metrics():
            return {"blocked_count": 0}

        @mini.post("/v1/events")
        def ingest_events():
            return {"stored": 0}

        mini.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )
        s = _make_settings(
            rate_limit_enabled=True,
            rate_limit_per_minute=read_limit,
            rate_limit_ingest_per_minute=ingest_limit,
        )
        mini.add_middleware(RateLimitMiddleware, settings=s)
        return mini

    def test_rate_limit_returns_429_after_limit(self):
        """Requests beyond the limit should return 429."""
        limit = 5
        mini = self._make_app(read_limit=limit)
        client = TestClient(mini)

        responses = [client.get("/v1/metrics/overview") for _ in range(limit + 2)]
        statuses = [r.status_code for r in responses]

        # First `limit` should be 200
        assert all(s == 200 for s in statuses[:limit]), f"Expected all 200, got {statuses[:limit]}"
        # Subsequent ones should be 429
        assert statuses[limit] == 429, f"Expected 429 at position {limit}, got {statuses[limit]}"

    def test_rate_limit_exempt_path_not_limited(self):
        """Requests to exempt paths (/health*) are never rate-limited."""
        limit = 3
        mini = self._make_app(read_limit=limit)
        client = TestClient(mini)

        for i in range(limit + 5):
            resp = client.get("/health")
            assert resp.status_code == 200, f"Request {i} to /health got {resp.status_code}"

    def test_ingest_path_higher_limit(self):
        """Ingest paths use the ingest limit, not the read limit."""
        read_limit = 2
        ingest_limit = 5
        mini = self._make_app(read_limit=read_limit, ingest_limit=ingest_limit)
        client = TestClient(mini)

        # POST /v1/events should allow up to ingest_limit requests
        for i in range(ingest_limit):
            resp = client.post("/v1/events")
            assert resp.status_code == 200, f"Ingest request {i} returned {resp.status_code}"

        # The next one should be rate-limited
        resp = client.post("/v1/events")
        assert resp.status_code == 429

        # But /v1/metrics/overview (read tier) was untouched → only 2 allowed
        for i in range(read_limit):
            resp = client.get("/v1/metrics/overview")
            assert resp.status_code == 200, f"Read request {i} returned {resp.status_code}"
        resp = client.get("/v1/metrics/overview")
        assert resp.status_code == 429

    def test_rate_limit_headers_present(self):
        """429 response must include X-RateLimit-Limit and Retry-After headers."""
        limit = 2
        mini = self._make_app(read_limit=limit)
        client = TestClient(mini)

        for _ in range(limit):
            client.get("/v1/metrics/overview")

        resp = client.get("/v1/metrics/overview")
        assert resp.status_code == 429
        assert "X-RateLimit-Limit" in resp.headers, "Missing X-RateLimit-Limit header"
        assert "Retry-After" in resp.headers, "Missing Retry-After header"
        assert resp.headers["X-RateLimit-Limit"] == str(limit)
        assert resp.headers["Retry-After"] == "60"

    def test_rate_limit_remaining_header_on_success(self):
        """Successful responses include X-RateLimit-Remaining."""
        limit = 10
        mini = self._make_app(read_limit=limit)
        client = TestClient(mini)

        resp = client.get("/v1/metrics/overview")
        assert resp.status_code == 200
        assert "X-RateLimit-Remaining" in resp.headers
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert remaining == limit - 1
