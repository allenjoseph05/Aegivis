"""
Sliding-window rate limiting middleware for the Aegivis backend.

Keyed by SHA-256 prefix of X-API-Key header, falling back to client IP.
Two limit tiers:
  - Read  (GET /v1/*):                 rate_limit_per_minute (default 300)
  - Ingest (POST /v1/events, /v1/violations): rate_limit_ingest_per_minute (default 3000)

Exempt paths: /health*, /metrics, /ws/*
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from ..config import BackendSettings

# Paths that are never rate-limited
_EXEMPT_PREFIXES = ("/health", "/metrics", "/ws/", "/docs", "/redoc", "/openapi.json")

# POST paths that use the higher ingest limit
_INGEST_PATHS = ("/v1/events", "/v1/violations")


def _key_for(request: Request) -> str:
    """Return a stable, hashed key for this request's caller."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return "k:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    # Fall back to client IP
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    return "ip:" + ip


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding 1-minute window rate limiter."""

    def __init__(self, app, settings: "BackendSettings") -> None:
        super().__init__(app)
        self._settings = settings
        # key_id → list of timestamps (float seconds)
        self._counts: dict[str, list[float]] = defaultdict(list)

    def _is_exempt(self, path: str) -> bool:
        return any(path.startswith(p) for p in _EXEMPT_PREFIXES)

    def _is_ingest(self, method: str, path: str) -> bool:
        return method == "POST" and any(path.startswith(p) for p in _INGEST_PATHS)

    async def dispatch(self, request: Request, call_next):
        if not self._settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        if self._is_exempt(path):
            return await call_next(request)

        ingest = self._is_ingest(request.method, path)
        limit = (
            self._settings.rate_limit_ingest_per_minute
            if ingest
            else self._settings.rate_limit_per_minute
        )

        # Use separate namespaces so ingest and read quotas don't share a counter
        tier_prefix = "i:" if ingest else "r:"
        key = tier_prefix + _key_for(request)
        now = time.monotonic()
        window_start = now - 60.0

        # Slide: drop timestamps older than 1 minute
        timestamps = self._counts[key]
        # In-place removal of old entries
        while timestamps and timestamps[0] < window_start:
            timestamps.pop(0)

        remaining = limit - len(timestamps)

        if remaining <= 0:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": 60},
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        timestamps.append(now)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        return response
