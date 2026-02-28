"""Named query helpers for common database operations."""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import AuditEventORM

if TYPE_CHECKING:
    from ..services.anomaly import AnomalyFlag


async def insert_event(session: AsyncSession, event: dict) -> None:
    """Insert a single audit event. Raises on duplicate event_id."""
    orm = AuditEventORM(
        event_id=event["event_id"],
        schema_version=event.get("schema_version", "1.0"),
        org_id=event["org_id"],
        session_id=event["session_id"],
        agent_id=event["agent_id"],
        provider=event["provider"],
        model=event["model"],
        interception_layer=event.get("interception_layer", "proxy"),
        run_id=event["run_id"],
        parent_run_id=event.get("parent_run_id"),
        event_type=event["event_type"],
        payload=event["payload"],
        payload_hash=event.get("payload_hash"),
        pii_detected=event.get("pii_detected", []),
        security=event.get("security"),
        timestamp_ns=event["timestamp_ns"],
        sequence_number=event["sequence_number"],
        previous_hash=event["previous_hash"],
        current_hash=event["current_hash"],
    )
    session.add(orm)


async def get_session_events(
    db: AsyncSession,
    session_id: str,
    org_id: str,
    limit: int = 1000,
    offset: int = 0,
) -> list[dict]:
    """Fetch all events for a session, ordered by sequence_number."""
    stmt = (
        select(AuditEventORM)
        .where(AuditEventORM.session_id == session_id)
        .where(AuditEventORM.org_id == org_id)
        .order_by(AuditEventORM.sequence_number)
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return [row.to_dict() for row in result.scalars()]


async def get_session_summary(
    db: AsyncSession,
    session_id: str,
    org_id: str,
) -> dict | None:
    """Compute session summary stats from event table."""
    stmt = text("""
        SELECT
            session_id,
            org_id,
            MIN(agent_id) AS agent_id,
            MIN(provider) AS provider,
            MIN(model) AS model,
            MIN(timestamp_ns) AS started_at_ns,
            MAX(timestamp_ns) AS last_event_ns,
            COUNT(*) AS event_count,
            COUNT(*) FILTER (WHERE event_type = 'LLM_CALL_START') AS llm_call_count,
            COUNT(*) FILTER (WHERE event_type IN ('TOOL_CALL_START')) AS tool_call_count,
            SUM(CAST(payload->>'total_tokens' AS INTEGER)) FILTER (WHERE event_type = 'LLM_CALL_END') AS total_tokens,
            BOOL_AND(current_hash IS NOT NULL) AS has_hashes,
            COUNT(*) FILTER (WHERE event_type = 'SYSTEM_ERROR') > 0 AS has_errors
        FROM audit_events
        WHERE session_id = :session_id AND org_id = :org_id
        GROUP BY session_id, org_id
    """)
    result = await db.execute(stmt, {"session_id": session_id, "org_id": org_id})
    row = result.mappings().first()
    if not row:
        return None
    return dict(row)


async def list_sessions(
    db: AsyncSession,
    org_id: str,
    limit: int = 50,
    offset: int = 0,
    agent_id: str | None = None,
    provider: str | None = None,
) -> list[dict]:
    """List sessions with summary stats, newest first."""
    conditions = "WHERE org_id = :org_id"
    params: dict[str, Any] = {"org_id": org_id, "limit": limit, "offset": offset}

    if agent_id:
        conditions += " AND agent_id = :agent_id"
        params["agent_id"] = agent_id
    if provider:
        conditions += " AND provider = :provider"
        params["provider"] = provider

    stmt = text(f"""
        SELECT
            session_id,
            org_id,
            MIN(agent_id) AS agent_id,
            MIN(provider) AS provider,
            MIN(model) AS model,
            to_timestamp(MIN(timestamp_ns) / 1e9) AS started_at,
            to_timestamp(MAX(timestamp_ns) / 1e9) AS last_event_at,
            COUNT(*) AS event_count,
            COUNT(*) FILTER (WHERE event_type = 'LLM_CALL_START') AS llm_call_count,
            COUNT(*) FILTER (WHERE event_type = 'TOOL_CALL_START') AS tool_call_count,
            COALESCE(SUM(
                CAST(payload->>'total_tokens' AS INTEGER)
            ) FILTER (WHERE event_type = 'LLM_CALL_END' AND payload->>'total_tokens' IS NOT NULL), 0) AS total_tokens,
            COUNT(*) FILTER (WHERE event_type = 'SYSTEM_ERROR') > 0 AS has_errors
        FROM audit_events
        {conditions}
        GROUP BY session_id, org_id
        ORDER BY MAX(timestamp_ns) DESC
        LIMIT :limit OFFSET :offset
    """)
    result = await db.execute(stmt, params)
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def get_last_event_hash(
    db: AsyncSession,
    session_id: str,
) -> str | None:
    """Get the current_hash of the last event in a session."""
    stmt = (
        select(AuditEventORM.current_hash)
        .where(AuditEventORM.session_id == session_id)
        .order_by(AuditEventORM.sequence_number.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    return row


async def event_exists(db: AsyncSession, event_id: str) -> bool:
    """Check if an event already exists (idempotent ingestion)."""
    stmt = select(AuditEventORM.event_id).where(AuditEventORM.event_id == event_id).limit(1)
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None


async def get_session_events_raw(session_id: str, db: AsyncSession) -> list[dict]:
    """
    Fetch all events for a session as plain dicts for the anomaly engine.
    Returns fields needed by detect_anomalies(): event_type, event_id,
    sequence_number, payload, pii_detected, model.
    """
    stmt = text("""
        SELECT event_id, event_type, sequence_number, payload,
               pii_detected, model, current_hash, previous_hash
        FROM audit_events
        WHERE session_id = :session_id
        ORDER BY sequence_number ASC
    """)
    result = await db.execute(stmt, {"session_id": session_id})
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def insert_anomalies(
    session_id: str,
    agent_id: str,
    org_id: str,
    flags: "list[AnomalyFlag]",
    db: AsyncSession,
) -> None:
    """Bulk-insert detected anomaly flags into agent_anomalies."""
    for flag in flags:
        await db.execute(
            text("""
                INSERT INTO agent_anomalies
                    (session_id, agent_id, org_id, rule_id, severity, description,
                     event_id, sequence_number, metadata)
                VALUES
                    (:session_id, :agent_id, :org_id, :rule_id, :severity, :description,
                     :event_id, :sequence_number, :metadata::jsonb)
            """),
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "org_id": org_id,
                "rule_id": flag.rule_id,
                "severity": flag.severity,
                "description": flag.description,
                "event_id": flag.event_id,
                "sequence_number": flag.sequence_number,
                "metadata": json.dumps(flag.metadata),
            },
        )
