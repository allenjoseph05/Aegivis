"""
POST /v1/violations — Ingest policy violations from the proxy.
GET  /v1/violations — Query violations (with filters).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...db.connection import get_session
from ...middleware.auth import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class ViolationRecord(BaseModel):
    rule_name: str
    action: str
    reason: str
    event_type: str
    session_id: str
    agent_id: str
    org_id: str = ""
    timestamp_ns: int


class ViolationBatch(BaseModel):
    violations: list[ViolationRecord]
    sent_at_ns: int


class ViolationResponse(BaseModel):
    accepted: int
    errors: list[str]


@router.post(
    "/violations",
    response_model=ViolationResponse,
    status_code=202,
    summary="Ingest policy violations from proxy",
)
async def ingest_violations(
    batch: ViolationBatch,
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
) -> ViolationResponse:
    accepted = 0
    errors: list[str] = []

    for v in batch.violations:
        try:
            await db.execute(
                text("""
                    INSERT INTO policy_violations
                        (rule_name, action, reason, event_type, session_id, agent_id, org_id, timestamp_ns)
                    VALUES
                        (:rule_name, :action, :reason, :event_type, :session_id, :agent_id, :org_id, :timestamp_ns)
                """),
                {
                    "rule_name": v.rule_name,
                    "action": v.action,
                    "reason": v.reason,
                    "event_type": v.event_type,
                    "session_id": v.session_id,
                    "agent_id": v.agent_id,
                    "org_id": v.org_id,
                    "timestamp_ns": v.timestamp_ns,
                },
            )
            accepted += 1
        except Exception as e:
            errors.append(f"Failed to insert violation: {str(e)[:100]}")

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Commit failed for violations: {e}")

    return ViolationResponse(accepted=accepted, errors=errors)


@router.get(
    "/violations",
    summary="List policy violations",
)
async def list_violations(
    session_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    rule_name: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    filters = []
    params: dict = {"limit": limit, "offset": offset}

    if session_id:
        filters.append("session_id = :session_id")
        params["session_id"] = session_id
    if agent_id:
        filters.append("agent_id = :agent_id")
        params["agent_id"] = agent_id
    if rule_name:
        filters.append("rule_name = :rule_name")
        params["rule_name"] = rule_name
    if action:
        filters.append("action = :action")
        params["action"] = action

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = text(f"""
        SELECT id, rule_name, action, reason, event_type, session_id, agent_id,
               org_id, timestamp_ns, received_at
        FROM policy_violations
        {where}
        ORDER BY timestamp_ns DESC
        LIMIT :limit OFFSET :offset
    """)

    result = await db.execute(sql, params)
    rows = result.fetchall()

    return {
        "violations": [
            {
                "id": r.id,
                "rule_name": r.rule_name,
                "action": r.action,
                "reason": r.reason,
                "event_type": r.event_type,
                "session_id": r.session_id,
                "agent_id": r.agent_id,
                "org_id": r.org_id,
                "timestamp_ns": r.timestamp_ns,
                "received_at": r.received_at.isoformat() if r.received_at else None,
            }
            for r in rows
        ],
        "total": len(rows),
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/violations/summary",
    summary="Policy violation summary by rule",
)
async def violations_summary(
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    result = await db.execute(text("""
        SELECT rule_name, action, COUNT(*) as count,
               MAX(timestamp_ns) as last_fired_ns
        FROM policy_violations
        GROUP BY rule_name, action
        ORDER BY count DESC
    """))
    rows = result.fetchall()
    return {
        "summary": [
            {
                "rule_name": r.rule_name,
                "action": r.action,
                "count": r.count,
                "last_fired_ns": r.last_fired_ns,
            }
            for r in rows
        ]
    }
