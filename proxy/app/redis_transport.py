"""
Redis Streams transport.

Publishes audit events and policy violations to Redis Streams instead of
sending them directly to the backend via HTTP.  The backend runs a consumer
group that reads from the streams, persists to PostgreSQL, and acknowledges.

Benefits
--------
* At-least-once delivery: events survive proxy restarts without a local SQLite
  buffer.
* Fan-out: multiple backend consumers can process the same stream in parallel.
* Back-pressure: stream MAXLEN cap prevents unbounded memory growth in Redis.

Fallback
--------
If Redis is unavailable at publish time, the call delegates to the existing
EventTransport (HTTP) so no events are lost.

Stream names
------------
    aegivis:events      — audit events    (MAXLEN ~ 50 000)
    aegivis:violations  — policy violations (MAXLEN ~ 10 000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from .config import settings

logger = logging.getLogger(__name__)

_STREAM_EVENTS     = "aegivis:events"
_STREAM_VIOLATIONS = "aegivis:violations"
_MAXLEN_EVENTS     = 50_000
_MAXLEN_VIOLATIONS = 10_000


class RedisTransport:
    """
    Drop-in replacement for EventTransport that publishes to Redis Streams.

    Falls back to the existing HTTP EventTransport when Redis is unavailable.
    """

    def __init__(self, redis_client, fallback):
        self._redis   = redis_client
        self._fallback = fallback
        self._running  = False

    async def start(self):
        self._running = True
        # Ensure streams exist
        try:
            await self._redis.xadd(
                _STREAM_EVENTS, {"_init": "1"}, maxlen=_MAXLEN_EVENTS, approximate=True
            )
        except Exception:
            pass
        logger.info("RedisTransport started (streams: %s, %s)", _STREAM_EVENTS, _STREAM_VIOLATIONS)

    def enqueue(self, event: dict):
        """Non-blocking: publish event to Redis stream."""
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._xadd_event(event))
            )
        except RuntimeError:
            # No running loop (e.g. test environment) — use fallback
            self._fallback.enqueue(event)

    def enqueue_violation(self, violation: dict):
        """Non-blocking: publish violation to Redis stream."""
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._xadd_violation(violation))
            )
        except RuntimeError:
            self._fallback.enqueue_violation(violation)

    async def _xadd_event(self, event: dict):
        try:
            await self._redis.xadd(
                _STREAM_EVENTS,
                {"event": json.dumps(event, default=str)},
                maxlen=_MAXLEN_EVENTS,
                approximate=True,
            )
        except Exception as e:
            logger.warning("Redis XADD event failed: %s — falling back to HTTP", e)
            self._fallback.enqueue(event)

    async def _xadd_violation(self, violation: dict):
        try:
            await self._redis.xadd(
                _STREAM_VIOLATIONS,
                {"violation": json.dumps(violation, default=str)},
                maxlen=_MAXLEN_VIOLATIONS,
                approximate=True,
            )
        except Exception as e:
            logger.warning("Redis XADD violation failed: %s — falling back to HTTP", e)
            self._fallback.enqueue_violation(violation)

    def buffer_status(self) -> dict:
        return {"events": "redis-stream", "violations": "redis-stream"}

    async def stop(self):
        self._running = False
        if self._fallback:
            await self._fallback.stop()
        logger.info("RedisTransport stopped")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_redis_transport: RedisTransport | None = None


async def get_redis_transport() -> "RedisTransport | None":
    """Create and return a RedisTransport if Redis is available."""
    global _redis_transport
    if _redis_transport is not None:
        return _redis_transport

    try:
        from redis.asyncio import Redis as AsyncRedis
        from .transport import EventTransport

        client = AsyncRedis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_connect_timeout=2,
        )
        await client.ping()

        fallback = EventTransport()
        await fallback.start()

        transport = RedisTransport(client, fallback)
        await transport.start()
        _redis_transport = transport
        return _redis_transport
    except Exception as e:
        logger.warning("RedisTransport init failed: %s — using HTTP transport", e)
        return None
