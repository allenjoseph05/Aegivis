"""
POST /v1/violations            — Ingest policy violations from the proxy.
GET  /v1/violations            — Query violations (with filters).
GET  /v1/violations/stats      — Time-bucketed violation counts for dashboard charts.
GET  /v1/violations/summary    — Per-rule/action aggregate counts.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

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
    org_ctx: OrgContext = Depends(require_api_key),
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
                    # Force org_id from authenticated key — never trust client-supplied value
                    "org_id": org_ctx.org_id,
                    "timestamp_ns": v.timestamp_ns,
                },
            )
            accepted += 1
        except Exception as e:
            errors.append(f"Failed to insert violation: {str(e)[:100]}")

    # If every insert failed the DB is likely down — return 503 so the proxy
    # knows to retry instead of silently dropping the violations batch.
    if accepted == 0 and errors:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to store violations: {errors[0]}",
        )

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error("Commit failed for violations batch: %s", e)
        raise HTTPException(status_code=503, detail="Failed to commit violations")

    return ViolationResponse(accepted=accepted, errors=errors)


@router.get(
    "/violations",
    summary="List policy violations",
)
async def list_violations(
    session_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    rule_name: str | None = Query(None),
    action: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    filters = ["org_id = :org_id"]
    params: dict = {"limit": limit, "offset": offset, "org_id": org_ctx.org_id}

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

    where = f"WHERE {' AND '.join(filters)}"
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
    "/violations/count",
    summary="Total violation count (for pagination)",
)
async def count_violations(
    session_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    rule_name: str | None = Query(None),
    action: str | None = Query(None),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    filters = ["org_id = :org_id"]
    params: dict = {"org_id": org_ctx.org_id}
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
    where = f"WHERE {' AND '.join(filters)}"
    result = await db.execute(
        text(f"SELECT COUNT(*) as n FROM policy_violations {where}"), params
    )
    return {"total": result.scalar() or 0}


@router.get(
    "/violations/{violation_id}/context",
    summary="Get violation detail with session context",
)
async def get_violation_context(
    violation_id: int,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    # Fetch the violation itself
    v_row = await db.execute(
        text("""
            SELECT id, rule_name, action, reason, event_type, session_id, agent_id,
                   org_id, timestamp_ns, received_at, is_false_positive, marked_fp_at
            FROM policy_violations
            WHERE id = :id AND org_id = :org_id
        """),
        {"id": violation_id, "org_id": org_ctx.org_id},
    )
    v = v_row.fetchone()
    if not v:
        raise HTTPException(status_code=404, detail="Violation not found")

    # Other violations in the same session (context)
    session_rows = await db.execute(
        text("""
            SELECT id, rule_name, action, reason, event_type, timestamp_ns
            FROM policy_violations
            WHERE session_id = :sid AND org_id = :org_id AND id != :vid
            ORDER BY timestamp_ns
            LIMIT 20
        """),
        {"sid": v.session_id, "org_id": org_ctx.org_id, "vid": violation_id},
    )
    session_violations = [
        {
            "id": r.id,
            "rule_name": r.rule_name,
            "action": r.action,
            "reason": r.reason,
            "event_type": r.event_type,
            "timestamp_ns": r.timestamp_ns,
        }
        for r in session_rows.fetchall()
    ]

    return {
        "violation": {
            "id": v.id,
            "rule_name": v.rule_name,
            "action": v.action,
            "reason": v.reason,
            "event_type": v.event_type,
            "session_id": v.session_id,
            "agent_id": v.agent_id,
            "org_id": v.org_id,
            "timestamp_ns": v.timestamp_ns,
            "received_at": v.received_at.isoformat() if v.received_at else None,
            "is_false_positive": v.is_false_positive,
            "marked_fp_at": v.marked_fp_at.isoformat() if v.marked_fp_at else None,
        },
        "session_violations": session_violations,
        "session_violation_count": len(session_violations),
    }


@router.post(
    "/violations/{violation_id}/mark-fp",
    summary="Mark a violation as a false positive",
)
async def mark_false_positive(
    violation_id: int,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(
        text("""
            UPDATE policy_violations
            SET is_false_positive = TRUE, marked_fp_at = NOW()
            WHERE id = :id AND org_id = :org_id
            RETURNING id
        """),
        {"id": violation_id, "org_id": org_ctx.org_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Violation not found")
    await db.commit()
    return {"status": "marked", "id": violation_id}


@router.post(
    "/violations/{violation_id}/unmark-fp",
    summary="Unmark a false positive",
)
async def unmark_false_positive(
    violation_id: int,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(
        text("""
            UPDATE policy_violations
            SET is_false_positive = FALSE, marked_fp_at = NULL
            WHERE id = :id AND org_id = :org_id
            RETURNING id
        """),
        {"id": violation_id, "org_id": org_ctx.org_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Violation not found")
    await db.commit()
    return {"status": "unmarked", "id": violation_id}


@router.get(
    "/violations/stats",
    summary="Time-bucketed violation counts for trend charts",
)
async def violations_stats(
    interval: str = Query(default="hour", pattern="^(hour|day)$", description="Bucket size: 'hour' or 'day'"),
    days: int = Query(default=7, ge=1, le=90, description="How many days of history to return"),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Returns violation counts bucketed by hour or day for the last N days,
    broken down by action (BLOCK / ALERT). Used by the Analytics dashboard
    for trend charts. All times are UTC.
    """
    bucket_expr = (
        "date_trunc('hour', to_timestamp(timestamp_ns / 1e9))"
        if interval == "hour"
        else "date_trunc('day', to_timestamp(timestamp_ns / 1e9))"
    )
    # Compute threshold as nanoseconds since epoch
    threshold_ns = int((time.time() - days * 86400) * 1e9)

    result = await db.execute(
        text(f"""
            SELECT
                to_char({bucket_expr}, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS bucket,
                action,
                COUNT(*) AS count
            FROM policy_violations
            WHERE org_id = :org_id
              AND timestamp_ns >= :threshold_ns
            GROUP BY bucket, action
            ORDER BY bucket ASC
        """),
        {"org_id": org_ctx.org_id, "threshold_ns": threshold_ns},
    )
    rows = result.fetchall()
    return {
        "interval": interval,
        "days": days,
        "buckets": [
            {"bucket": r.bucket, "action": r.action, "count": r.count}
            for r in rows
        ],
    }


@router.get(
    "/violations/summary",
    summary="Policy violation summary by rule",
)
async def violations_summary(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(text("""
        SELECT rule_name, action, COUNT(*) as count,
               MAX(timestamp_ns) as last_fired_ns
        FROM policy_violations
        WHERE org_id = :org_id
        GROUP BY rule_name, action
        ORDER BY count DESC
    """), {"org_id": org_ctx.org_id})
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
