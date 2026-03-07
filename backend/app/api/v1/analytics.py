"""
GET /v1/analytics/trends  — Daily posture scores and activity for the last N days.
GET /v1/analytics/agents  — Per-agent comparison table.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


def _posture_score(blocked: int, alerted: int, sessions: int) -> int:
    """
    Compute a 0-100 posture score for a single day.
    - Starts at 100
    - Each BLOCK event costs 5 points (max -50)
    - Each ALERT event costs 1 point (max -30)
    - No activity = 70 (not enough data to be confident)
    """
    if sessions == 0:
        return 70
    score = 100
    score -= min(blocked * 5, 50)
    score -= min(alerted * 1, 30)
    return max(0, score)


@router.get("/analytics/trends", summary="Daily posture trends (last N days)")
async def get_trends(
    days: int = Query(default=30, ge=7, le=90),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    since_ns = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1e9)

    # Daily violation counts
    v_result = await db.execute(
        text("""
            SELECT
                DATE_TRUNC('day', TO_TIMESTAMP(timestamp_ns / 1e9)) AS day,
                action,
                COUNT(*) AS cnt
            FROM policy_violations
            WHERE org_id = :org_id AND timestamp_ns >= :since_ns
            GROUP BY day, action
            ORDER BY day
        """),
        {"org_id": org_ctx.org_id, "since_ns": since_ns},
    )
    v_rows = v_result.fetchall()

    # Daily session counts
    s_result = await db.execute(
        text("""
            SELECT
                DATE_TRUNC('day', started_at) AS day,
                COUNT(*) AS cnt
            FROM sessions
            WHERE org_id = :org_id AND started_at >= TO_TIMESTAMP(:since_s)
            GROUP BY day
            ORDER BY day
        """),
        {"org_id": org_ctx.org_id, "since_s": since_ns / 1e9},
    )
    s_rows = s_result.fetchall()

    # Build day-keyed maps
    blocked_by_day: dict[str, int] = {}
    alerted_by_day: dict[str, int] = {}
    for r in v_rows:
        day_str = r.day.strftime("%Y-%m-%d") if r.day else "unknown"
        if r.action == "BLOCK":
            blocked_by_day[day_str] = blocked_by_day.get(day_str, 0) + r.cnt
        elif r.action == "ALERT":
            alerted_by_day[day_str] = alerted_by_day.get(day_str, 0) + r.cnt

    sessions_by_day: dict[str, int] = {}
    for r in s_rows:
        day_str = r.day.strftime("%Y-%m-%d") if r.day else "unknown"
        sessions_by_day[day_str] = sessions_by_day.get(day_str, 0) + r.cnt

    # Fill every day in the range
    points = []
    today = datetime.now(timezone.utc).date()
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        blocked = blocked_by_day.get(d, 0)
        alerted = alerted_by_day.get(d, 0)
        sessions = sessions_by_day.get(d, 0)
        points.append({
            "date": d,
            "posture_score": _posture_score(blocked, alerted, sessions),
            "blocked": blocked,
            "alerted": alerted,
            "sessions": sessions,
        })

    return {"days": days, "points": points}


@router.get("/analytics/agents", summary="Per-agent comparison table")
async def get_agent_comparison(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(
        text("""
            SELECT
                m.agent_id,
                m.session_count,
                m.llm_call_count,
                m.tool_call_count,
                m.error_count,
                m.total_tokens,
                m.anomaly_count,
                m.pii_event_count,
                m.avg_latency_ms,
                m.last_seen,
                COALESCE(v.block_count, 0)  AS block_count,
                COALESCE(v.alert_count, 0)  AS alert_count,
                COALESCE(v.total_violations, 0) AS total_violations
            FROM agent_metrics_view m
            LEFT JOIN (
                SELECT agent_id,
                       COUNT(*) FILTER (WHERE action = 'BLOCK') AS block_count,
                       COUNT(*) FILTER (WHERE action = 'ALERT') AS alert_count,
                       COUNT(*) AS total_violations
                FROM policy_violations
                WHERE org_id = :org_id
                GROUP BY agent_id
            ) v ON v.agent_id = m.agent_id
            WHERE m.org_id = :org_id
            ORDER BY v.total_violations DESC NULLS LAST, m.session_count DESC
        """),
        {"org_id": org_ctx.org_id},
    )
    rows = result.fetchall()

    agents = []
    for r in rows:
        agents.append({
            "agent_id": r.agent_id,
            "session_count": r.session_count,
            "llm_call_count": r.llm_call_count,
            "tool_call_count": r.tool_call_count,
            "error_count": r.error_count,
            "total_tokens": r.total_tokens,
            "anomaly_count": r.anomaly_count,
            "pii_event_count": r.pii_event_count,
            "avg_latency_ms": float(r.avg_latency_ms) if r.avg_latency_ms else None,
            "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            "block_count": r.block_count,
            "alert_count": r.alert_count,
            "total_violations": r.total_violations,
        })

    return {"agents": agents, "total": len(agents)}
