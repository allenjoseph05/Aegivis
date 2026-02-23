"""
GET /v1/anomalies - Query persisted anomaly detections.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...db.connection import get_session
from ...middleware.auth import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/anomalies",
    summary="List detected anomalies",
)
async def list_anomalies(
    session_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    rule_id: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    """
    List anomalies detected by the anomaly engine.

    Filter by session_id, agent_id, severity (critical/high/medium/low), or rule_id.
    Results are ordered by detection time descending (newest first).
    """
    filters = []
    params: dict = {"limit": limit, "offset": offset}

    if session_id:
        filters.append("session_id = :session_id")
        params["session_id"] = session_id
    if agent_id:
        filters.append("agent_id = :agent_id")
        params["agent_id"] = agent_id
    if severity:
        filters.append("severity = :severity")
        params["severity"] = severity
    if rule_id:
        filters.append("rule_id = :rule_id")
        params["rule_id"] = rule_id

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = text(f"""
        SELECT id, session_id, agent_id, org_id, rule_id, severity,
               description, event_id, sequence_number, metadata, detected_at
        FROM agent_anomalies
        {where}
        ORDER BY detected_at DESC
        LIMIT :limit OFFSET :offset
    """)

    result = await db.execute(sql, params)
    rows = result.fetchall()

    return {
        "anomalies": [
            {
                "id": r.id,
                "session_id": r.session_id,
                "agent_id": r.agent_id,
                "org_id": r.org_id,
                "rule_id": r.rule_id,
                "severity": r.severity,
                "description": r.description,
                "event_id": r.event_id,
                "sequence_number": r.sequence_number,
                "metadata": r.metadata or {},
                "detected_at": r.detected_at.isoformat() if r.detected_at else None,
            }
            for r in rows
        ],
        "total": len(rows),
        "limit": limit,
        "offset": offset,
    }
