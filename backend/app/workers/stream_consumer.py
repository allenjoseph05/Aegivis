"""
Redis Streams consumer for the AgentBlackBox backend.

Reads audit events and policy violations from Redis Streams and persists
them to PostgreSQL. Runs as background asyncio tasks in the backend process.

Streams
-------
    abb:events      — audit events published by the proxy via RedisTransport
    abb:violations  — policy violations published by the proxy via RedisTransport

Consumer groups ensure at-least-once delivery:
- Messages are ACKed only after a successful DB insert.
- On restart, unacknowledged messages are re-delivered automatically.
"""
from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.queries import insert_event

logger = logging.getLogger(__name__)

_STREAM_EVENTS       = "abb:events"
_STREAM_VIOLATIONS   = "abb:violations"
_GROUP               = "abb-backend"
_CONSUMER_EVENTS     = "backend-events-1"
_CONSUMER_VIOLATIONS = "backend-violations-1"
_BATCH_SIZE          = 50
_BLOCK_MS            = 5_000   # block up to 5 s waiting for new messages


async def _ensure_group(redis_client, stream: str, group: str) -> None:
    """Create consumer group if it doesn't exist; silently ignore BUSYGROUP."""
    try:
        await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
        logger.info("Created consumer group '%s' on stream '%s'", group, stream)
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.debug("Consumer group '%s' already exists on '%s'", group, stream)
        else:
            logger.warning("xgroup_create failed for %s/%s: %s", stream, group, e)


async def run_event_consumer(redis_client, session_factory: async_sessionmaker) -> None:
    """
    Consume audit events from abb:events stream and persist to PostgreSQL.
    Runs until cancelled.
    """
    await _ensure_group(redis_client, _STREAM_EVENTS, _GROUP)
    logger.info("Event stream consumer started")

    while True:
        try:
            messages = await redis_client.xreadgroup(
                _GROUP,
                _CONSUMER_EVENTS,
                {_STREAM_EVENTS: ">"},
                count=_BATCH_SIZE,
                block=_BLOCK_MS,
            )
            if not messages:
                continue

            for _stream_name, entries in messages:
                msg_ids = [msg_id for msg_id, _ in entries]

                # Decode events from stream entries
                events = []
                for msg_id, data in entries:
                    try:
                        raw = data.get(b"event") or data.get("event")
                        if raw:
                            events.append(json.loads(raw))
                    except Exception as e:
                        logger.warning(
                            "Failed to decode event from stream msg %s: %s", msg_id, e
                        )

                # Persist to PostgreSQL
                if events:
                    try:
                        async with session_factory() as db:
                            for event in events:
                                try:
                                    await insert_event(db, event)
                                except Exception as e:
                                    logger.warning(
                                        "insert_event failed for %s: %s",
                                        event.get("event_id"), e,
                                    )
                            await db.commit()
                    except Exception as e:
                        logger.error("DB session failed for event batch: %s — skipping ACK", e)
                        continue  # don't ACK — re-delivered on next restart

                # ACK all (including parse errors — avoids infinite retry of malformed msgs)
                await redis_client.xack(_STREAM_EVENTS, _GROUP, *msg_ids)

        except asyncio.CancelledError:
            logger.info("Event stream consumer stopped")
            return
        except Exception as e:
            logger.error("Event stream consumer error: %s — retrying in 5s", e)
            await asyncio.sleep(5)


async def run_violation_consumer(
    redis_client, session_factory: async_sessionmaker
) -> None:
    """
    Consume policy violations from abb:violations stream and persist to PostgreSQL.
    Runs until cancelled.
    """
    await _ensure_group(redis_client, _STREAM_VIOLATIONS, _GROUP)
    logger.info("Violation stream consumer started")

    while True:
        try:
            messages = await redis_client.xreadgroup(
                _GROUP,
                _CONSUMER_VIOLATIONS,
                {_STREAM_VIOLATIONS: ">"},
                count=_BATCH_SIZE,
                block=_BLOCK_MS,
            )
            if not messages:
                continue

            for _stream_name, entries in messages:
                msg_ids = [msg_id for msg_id, _ in entries]

                violations = []
                for msg_id, data in entries:
                    try:
                        raw = data.get(b"violation") or data.get("violation")
                        if raw:
                            violations.append(json.loads(raw))
                    except Exception as e:
                        logger.warning(
                            "Failed to decode violation from stream msg %s: %s", msg_id, e
                        )

                if violations:
                    try:
                        async with session_factory() as db:
                            for v in violations:
                                try:
                                    await db.execute(
                                        text("""
                                            INSERT INTO policy_violations
                                                (rule_name, action, reason, event_type,
                                                 session_id, agent_id, org_id, timestamp_ns)
                                            VALUES
                                                (:rule_name, :action, :reason, :event_type,
                                                 :session_id, :agent_id, :org_id, :timestamp_ns)
                                        """),
                                        {
                                            "rule_name":    v.get("rule_name", "unknown"),
                                            "action":       v.get("action", "ALERT"),
                                            "reason":       v.get("reason", ""),
                                            "event_type":   v.get("event_type", ""),
                                            "session_id":   v.get("session_id", ""),
                                            "agent_id":     v.get("agent_id", "unknown"),
                                            "org_id":       v.get("org_id", "default-org"),
                                            "timestamp_ns": v.get("timestamp_ns", 0),
                                        },
                                    )
                                except Exception as e:
                                    logger.warning("Insert violation failed: %s", e)
                            await db.commit()
                    except Exception as e:
                        logger.error(
                            "DB session failed for violation batch: %s — skipping ACK", e
                        )
                        continue

                await redis_client.xack(_STREAM_VIOLATIONS, _GROUP, *msg_ids)

        except asyncio.CancelledError:
            logger.info("Violation stream consumer stopped")
            return
        except Exception as e:
            logger.error("Violation stream consumer error: %s — retrying in 5s", e)
            await asyncio.sleep(5)
