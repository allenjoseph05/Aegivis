"""
POST /v1/events - Batch event ingestion endpoint.

Receives event batches from the proxy (and optionally the SDK).
Validates schema, verifies hash chain continuity, and persists to PostgreSQL.
On AGENT_FINISH: runs anomaly detection, persists flags, pushes WebSocket alerts,
and sends email/Slack notifications (all fire-and-forget).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import (
    event_exists,
    get_last_event_hash,
    get_session_events_raw,
    insert_anomalies,
    insert_event,
)
from ...middleware.auth import OrgContext, require_api_key
from ...services.anomaly import detect_anomalies
from ...services.baseline import update_baseline
from ...services.notifications import notify_anomalies
from ..v1.ws import ws_broadcast

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_EVENT_TYPES = {
    "LLM_CALL_START", "LLM_CALL_END", "TOOL_CALL_START", "TOOL_CALL_END",
    "AGENT_FINISH", "SYSTEM_ERROR", "CHECKPOINT",
    # SDK emits AGENT_THOUGHT (custom annotations, tags, errors)
    "AGENT_THOUGHT",
}


class IngestBatchRequest(BaseModel):
    events: list[dict] = Field(min_length=1, max_length=500)
    batch_id: str
    sent_at_ns: int


class IngestBatchResponse(BaseModel):
    accepted: int
    skipped: int
    errors: list[str]


@router.post(
    "/events",
    response_model=IngestBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest event batch from proxy",
)
async def ingest_events(
    batch: IngestBatchRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
) -> IngestBatchResponse:
    """
    Accepts a batch of audit events from the AgentBlackBox proxy.
    Validates each event, checks for duplicates, and persists to PostgreSQL.
    Events whose org_id does not match the authenticated key are rejected.
    """
    accepted = 0
    skipped = 0
    errors: list[str] = []
    # Track agent_finish events to run anomaly detection after commit
    finish_events: list[dict] = []

    for event in batch.events:
        event_id = event.get("event_id")
        if not event_id:
            errors.append("Event missing event_id - skipped")
            skipped += 1
            continue

        # Validate required fields
        missing = _check_required_fields(event)
        if missing:
            errors.append(f"Event {event_id}: missing fields {missing}")
            skipped += 1
            continue

        # Validate event type
        if event.get("event_type") not in VALID_EVENT_TYPES:
            errors.append(f"Event {event_id}: unknown event_type '{event.get('event_type')}'")
            skipped += 1
            continue

        # Org isolation: reject events that don't belong to the authenticated org
        if event.get("org_id") != org_ctx.org_id:
            errors.append(
                f"Event {event_id}: org_id '{event.get('org_id')}' does not match "
                f"authenticated org '{org_ctx.org_id}'"
            )
            skipped += 1
            continue

        # Idempotent: skip duplicates
        try:
            if await event_exists(db, event_id):
                skipped += 1
                continue
        except Exception as e:
            errors.append(f"Event {event_id}: DB check failed - {e}")
            skipped += 1
            continue

        # Persist
        try:
            await insert_event(db, event)
            accepted += 1
        except Exception as e:
            logger.error(f"Failed to insert event {event_id}: {e}")
            errors.append(f"Event {event_id}: insert failed - {str(e)[:100]}")
            skipped += 1
            continue

        # Update agent baseline on session completion
        if event.get("event_type") == "AGENT_FINISH":
            payload = event.get("payload", {})
            try:
                await update_baseline(
                    db,
                    org_id=org_ctx.org_id,
                    agent_id=event.get("agent_id", "unknown"),
                    session_stats={
                        "llm_call_count":      payload.get("total_llm_calls", 0),
                        "tool_call_count":     payload.get("total_tool_calls", 0),
                        "session_duration_ms": payload.get("session_duration_ms", 0),
                    },
                )
            except Exception as e:
                logger.warning(f"Baseline update failed for {event.get('agent_id')}: {e}")
            finish_events.append(event)

    try:
        await db.commit()
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database commit failed: {str(e)[:100]}")

    if errors:
        logger.warning(f"Batch {batch.batch_id}: {accepted} accepted, {skipped} skipped, {len(errors)} errors")

    # Fire-and-forget: anomaly detection + WebSocket push + notifications
    for finish_event in finish_events:
        asyncio.create_task(
            _run_anomaly_pipeline(finish_event, db)
        )

    return IngestBatchResponse(accepted=accepted, skipped=skipped, errors=errors)


async def _run_anomaly_pipeline(finish_event: dict, db: AsyncSession) -> None:
    """
    Run anomaly detection for a completed session.
    Persists flags, pushes to WebSocket, and sends notifications.
    Never raises - all errors are logged as warnings.
    """
    session_id = finish_event.get("session_id", "")
    agent_id = finish_event.get("agent_id", "unknown")
    org_id = finish_event.get("org_id", "default-org")

    try:
        events_raw = await get_session_events_raw(session_id, db)
        flags = detect_anomalies(events_raw, chain_valid=True)
        if not flags:
            return

        await insert_anomalies(session_id, agent_id, org_id, flags, db)
        await db.commit()

        # Push to WebSocket subscribers
        for flag in flags:
            await ws_broadcast({
                "type": "anomaly",
                "session_id": session_id,
                "agent_id": agent_id,
                "rule_id": flag.rule_id,
                "severity": flag.severity,
                "description": flag.description,
            })

        # Slack/email notifications for high/critical only
        await notify_anomalies(flags, session_id, agent_id)

        logger.info(f"Anomaly pipeline: {len(flags)} flag(s) for session {session_id}")
    except Exception as e:
        logger.warning(f"Anomaly pipeline failed for session {session_id}: {e}")


_REQUIRED_FIELDS = {
    "event_id", "schema_version", "org_id", "session_id", "agent_id",
    "provider", "model", "run_id", "event_type", "payload",
    "timestamp_ns", "sequence_number", "previous_hash", "current_hash",
}


def _check_required_fields(event: dict) -> list[str]:
    return [f for f in _REQUIRED_FIELDS if f not in event or event[f] is None]
