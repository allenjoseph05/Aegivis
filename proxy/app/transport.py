"""
Async batch transport: buffers events and sends them to the backend in batches.

Design goals:
- Fire-and-forget: never block the agent's LLM response path
- Batch events for efficiency (configurable size + flush interval)
- Disk buffer fallback: if backend unreachable, write to SQLite; retry later
- Background asyncio task handles flushing + retry
- Graceful shutdown flushes remaining events (in-memory + disk)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import httpx

from .buffer import get_buffer
from .config import settings

logger = logging.getLogger(__name__)


class EventTransport:
    """
    Buffers audit events and policy violations, sends to the backend in async batches.
    Falls back to local SQLite disk buffer when backend is unreachable.
    One instance per proxy process. Shared across all requests.
    """

    def __init__(self):
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10_000)
        self._violation_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=5_000)
        self._client: httpx.AsyncClient | None = None
        self._flush_task: asyncio.Task | None = None
        self._retry_task: asyncio.Task | None = None
        self._running = False
        self._buffer = get_buffer(settings.buffer_db_path, settings.buffer_max_events)

    async def start(self):
        """Initialize HTTP client and start background flush + retry loops."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.transport_timeout_s),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": settings.backend_api_key,
            },
        )
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop(), name="aegivis-transport-flush")

        if self._buffer:
            self._retry_task = asyncio.create_task(self._retry_loop(), name="aegivis-buffer-retry")
            sizes = self._buffer.size()
            if sizes["events"] > 0 or sizes["violations"] > 0:
                logger.info(
                    f"LocalBuffer: {sizes['events']} events + {sizes['violations']} violations "
                    f"from previous run will be retried"
                )

        logger.info(f"EventTransport started -> {settings.backend_url}")

    def enqueue(self, event: dict):
        """
        Non-blocking: add event to in-memory queue.
        If queue is full, fall back to disk buffer. If disk disabled, drop.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            if self._buffer:
                self._buffer.write_events([event])
                logger.warning(
                    f"In-memory queue full — event {event.get('event_id', '?')} "
                    f"written to disk buffer"
                )
            else:
                logger.warning(
                    f"EventTransport queue full and disk buffer disabled. "
                    f"Dropping event {event.get('event_id', 'unknown')}."
                )

    def enqueue_violation(self, violation: dict):
        """Non-blocking: add a policy violation to the violation queue."""
        try:
            self._violation_queue.put_nowait(violation)
        except asyncio.QueueFull:
            if self._buffer:
                self._buffer.write_violations([violation])
                logger.warning("Violation queue full — written to disk buffer")
            else:
                logger.warning("Violation queue full and disk buffer disabled — dropping")

    async def _flush_loop(self):
        """Background task: flush batch every N seconds or when batch_size reached."""
        while self._running:
            await asyncio.sleep(settings.batch_flush_interval_s)
            await self._flush_pending()

    async def _retry_loop(self):
        """Background task: retry events/violations that were written to disk buffer."""
        while self._running:
            await asyncio.sleep(settings.buffer_retry_interval_s)
            if self._buffer:
                await self._retry_buffered_events()
                await self._retry_buffered_violations()

    async def _retry_buffered_events(self):
        """Attempt to send disk-buffered events to the backend."""
        if not self._buffer:
            return
        rows = self._buffer.read_events(limit=settings.batch_size * 5)
        if not rows:
            return

        row_ids = [r[0] for r in rows]
        events  = [r[1] for r in rows]
        success = await self._send_batch(events)
        if success:
            self._buffer.delete(row_ids)
            logger.info(f"LocalBuffer: retried and sent {len(events)} buffered events")

    async def _retry_buffered_violations(self):
        """Attempt to send disk-buffered violations to the backend."""
        if not self._buffer:
            return
        rows = self._buffer.read_violations(limit=settings.batch_size * 5)
        if not rows:
            return

        row_ids    = [r[0] for r in rows]
        violations = [r[1] for r in rows]
        success = await self._send_violations(violations)
        if success:
            self._buffer.delete(row_ids)
            logger.info(f"LocalBuffer: retried and sent {len(violations)} buffered violations")

    async def _flush_pending(self):
        """Drain up to batch_size events and violations, send both."""
        batch = []
        while len(batch) < settings.batch_size:
            try:
                event = self._queue.get_nowait()
                batch.append(event)
            except asyncio.QueueEmpty:
                break

        if batch:
            success = await self._send_batch(batch)
            if not success and self._buffer:
                self._buffer.write_events(batch)
                logger.warning(f"Backend unreachable — {len(batch)} events written to disk buffer")

        violations = []
        while len(violations) < settings.batch_size:
            try:
                violations.append(self._violation_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if violations:
            success = await self._send_violations(violations)
            if not success and self._buffer:
                self._buffer.write_violations(violations)

    async def _send_batch(self, events: list[dict]) -> bool:
        """
        POST event batch to backend /v1/events endpoint.
        Returns True on success, False on any failure.
        """
        if not self._client or not events:
            return True

        payload = {
            "events": events,
            "batch_id": str(uuid.uuid4()),
            "sent_at_ns": time.time_ns(),
        }

        try:
            resp = await self._client.post(
                f"{settings.backend_url}/v1/events",
                content=json.dumps(payload, default=str),
            )
            if resp.status_code not in (200, 201, 202):
                logger.warning(
                    f"Backend rejected batch: HTTP {resp.status_code} — {resp.text[:200]}"
                )
                return False
            logger.debug(f"Sent {len(events)} events to backend")
            return True
        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to backend at {settings.backend_url}. "
                f"{'Events will be retried from disk buffer.' if self._buffer else 'Events dropped.'}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to send event batch: {e}")
            return False

    async def _send_violations(self, violations: list[dict]) -> bool:
        """
        POST policy violations to backend /v1/violations endpoint.
        Returns True on success, False on any failure.
        """
        if not self._client or not violations:
            return True
        try:
            resp = await self._client.post(
                f"{settings.backend_url}/v1/violations",
                content=json.dumps(
                    {"violations": violations, "sent_at_ns": time.time_ns()},
                    default=str,
                ),
            )
            if resp.status_code not in (200, 201, 202):
                logger.warning(f"Backend rejected violations: HTTP {resp.status_code}")
                return False
            logger.debug(f"Sent {len(violations)} policy violations to backend")
            return True
        except Exception as e:
            logger.error(f"Failed to send violations: {e}")
            return False

    def buffer_status(self) -> dict:
        """Return current buffer sizes (for /health endpoint)."""
        if self._buffer:
            return self._buffer.size()
        return {"events": 0, "violations": 0, "disk_buffer": "disabled"}

    async def stop(self):
        """Graceful shutdown: flush in-memory queues, persist remainder to disk."""
        self._running = False

        for task in (self._flush_task, self._retry_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Drain remaining in-memory events
        remaining = []
        while not self._queue.empty():
            try:
                remaining.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if remaining:
            success = await self._send_batch(remaining)
            if not success and self._buffer:
                self._buffer.write_events(remaining)
                logger.info(f"Shutdown: {len(remaining)} events written to disk buffer")

        # Drain remaining violations
        remaining_v = []
        while not self._violation_queue.empty():
            try:
                remaining_v.append(self._violation_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if remaining_v:
            success = await self._send_violations(remaining_v)
            if not success and self._buffer:
                self._buffer.write_violations(remaining_v)

        if self._client:
            await self._client.aclose()
        logger.info("EventTransport stopped")


# Singleton instance
_transport: EventTransport | None = None


def get_transport() -> EventTransport:
    """Return the in-memory/HTTP EventTransport (default / fallback)."""
    global _transport
    if _transport is None:
        _transport = EventTransport()
    return _transport


async def get_best_transport() -> "EventTransport":
    """
    Return the best available transport.

    If AEGIVIS_REDIS_URL is set:  RedisTransport (publishes to Redis Streams).
    Otherwise:                 EventTransport (HTTP batches to backend).
    """
    if settings.redis_url:
        try:
            from .redis_transport import get_redis_transport
            rt = await get_redis_transport()
            if rt is not None:
                return rt  # type: ignore[return-value]
        except Exception as e:
            logger.warning("RedisTransport unavailable (%s) — using HTTP transport", e)
    return get_transport()
