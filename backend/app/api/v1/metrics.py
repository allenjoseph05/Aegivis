"""
GET /v1/metrics/overview -- High-level platform totals.
GET /v1/metrics/agents   -- Per-agent performance + security metrics.
GET /v1/metrics/models   -- Per-model usage statistics.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics/overview", summary="Platform-wide overview metrics")
async def metrics_overview(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    High-level totals across all agents, sessions, and LLM calls.

    Aggregates from audit_events, agent_anomalies, and policy_violations tables,
    scoped to the authenticated organisation.
    """
    # Core event metrics
    events_result = await db.execute(text("""
        SELECT
            COUNT(DISTINCT session_id)                                              AS session_count,
            COUNT(DISTINCT agent_id)                                                AS agent_count,
            COUNT(*) FILTER (WHERE event_type = 'LLM_CALL_START')                  AS llm_call_count,
            COUNT(*) FILTER (WHERE event_type = 'TOOL_CALL_START')                 AS tool_call_count,
            COUNT(*) FILTER (WHERE event_type = 'SYSTEM_ERROR')                    AS error_count,
            COUNT(*) FILTER (WHERE event_type = 'MEMORY_WRITE_BLOCKED')            AS memory_blocked_count,
            COALESCE(SUM(
                CAST(NULLIF(payload->>'total_tokens', '') AS INTEGER)
            ) FILTER (WHERE event_type = 'LLM_CALL_END'
                        AND payload->>'total_tokens' IS NOT NULL), 0)               AS total_tokens,
            COUNT(*) FILTER (WHERE cardinality(pii_detected) > 0)                 AS pii_event_count,
            ROUND(AVG(
                CAST(payload->>'latency_ms' AS FLOAT)
            ) FILTER (WHERE event_type = 'LLM_CALL_END'
                        AND payload->>'latency_ms' IS NOT NULL)::numeric, 2)        AS avg_latency_ms
        FROM audit_events
        WHERE org_id = :org_id
    """), {"org_id": org_ctx.org_id})
    events_row = events_result.fetchone()

    # Anomaly counts
    anom_result = await db.execute(text("""
        SELECT
            COUNT(*)                                                                AS total_anomalies,
            COUNT(*) FILTER (WHERE severity IN ('critical', 'high'))               AS high_severity_anomalies
        FROM agent_anomalies
        WHERE org_id = :org_id
    """), {"org_id": org_ctx.org_id})
    anom_row = anom_result.fetchone()

    # Violation counts
    viol_result = await db.execute(text("""
        SELECT
            COUNT(*)                                                                AS total_violations,
            COUNT(*) FILTER (WHERE action = 'BLOCK')                               AS blocked_count,
            COUNT(*) FILTER (WHERE action = 'ALERT')                               AS alert_count
        FROM policy_violations
        WHERE org_id = :org_id
    """), {"org_id": org_ctx.org_id})
    viol_row = viol_result.fetchone()

    overview: dict = {}
    if events_row:
        overview.update({
            "session_count":    events_row.session_count    or 0,
            "agent_count":      events_row.agent_count      or 0,
            "llm_call_count":   events_row.llm_call_count   or 0,
            "tool_call_count":  events_row.tool_call_count  or 0,
            "error_count":      events_row.error_count      or 0,
            "total_tokens":     events_row.total_tokens     or 0,
            "pii_event_count":        events_row.pii_event_count        or 0,
            "memory_blocked_count":   events_row.memory_blocked_count   or 0,
            "avg_latency_ms":         float(events_row.avg_latency_ms) if events_row.avg_latency_ms is not None else None,
        })
    else:
        overview.update({k: 0 for k in [
            "session_count", "agent_count", "llm_call_count", "tool_call_count",
            "error_count", "total_tokens", "pii_event_count",
        ]})
        overview["memory_blocked_count"] = 0
        overview["avg_latency_ms"] = None

    if anom_row:
        overview["total_anomalies"]         = anom_row.total_anomalies         or 0
        overview["high_severity_anomalies"] = anom_row.high_severity_anomalies or 0
    else:
        overview["total_anomalies"] = 0
        overview["high_severity_anomalies"] = 0

    if viol_row:
        overview["total_violations"] = viol_row.total_violations or 0
        overview["blocked_count"]    = viol_row.blocked_count    or 0
        overview["alert_count"]      = viol_row.alert_count      or 0
    else:
        overview["total_violations"] = 0
        overview["blocked_count"]    = 0
        overview["alert_count"]      = 0

    return overview


@router.get("/metrics/agents", summary="Per-agent performance and security metrics")
async def metrics_agents(
    agent_id: str | None = Query(None, description="Filter to a single agent ID"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Aggregate performance and security metrics grouped by agent_id,
    scoped to the authenticated organisation.
    """
    agent_filter = "AND ae.agent_id = :agent_id" if agent_id else ""
    params: dict = {"limit": limit, "offset": offset, "org_id": org_ctx.org_id}
    if agent_id:
        params["agent_id"] = agent_id

    result = await db.execute(text(f"""
        SELECT
            ae.agent_id,
            COUNT(DISTINCT ae.session_id)                                           AS session_count,
            COUNT(*) FILTER (WHERE ae.event_type = 'LLM_CALL_START')               AS llm_call_count,
            COUNT(*) FILTER (WHERE ae.event_type = 'TOOL_CALL_START')              AS tool_call_count,
            COUNT(*) FILTER (WHERE ae.event_type = 'SYSTEM_ERROR')                 AS error_count,
            COALESCE(SUM(
                CAST(NULLIF(ae.payload->>'total_tokens', '') AS INTEGER)
            ) FILTER (WHERE ae.event_type = 'LLM_CALL_END'
                        AND ae.payload->>'total_tokens' IS NOT NULL), 0)            AS total_tokens,
            ROUND(AVG(
                CAST(ae.payload->>'latency_ms' AS FLOAT)
            ) FILTER (WHERE ae.event_type = 'LLM_CALL_END'
                        AND ae.payload->>'latency_ms' IS NOT NULL)::numeric, 2)     AS avg_latency_ms,
            COUNT(*) FILTER (WHERE cardinality(ae.pii_detected) > 0)              AS pii_event_count,
            COALESCE(MAX(
                CAST(ae.payload->'security'->>'injection_score' AS FLOAT)
            ) FILTER (WHERE ae.payload->'security' IS NOT NULL), 0.0)              AS injection_score_max,
            COALESCE(an.anomaly_count, 0)                                           AS anomaly_count,
            to_timestamp(MIN(ae.timestamp_ns) / 1e9)                               AS first_seen,
            to_timestamp(MAX(ae.timestamp_ns) / 1e9)                               AS last_seen
        FROM audit_events ae
        LEFT JOIN (
            SELECT agent_id, COUNT(*) AS anomaly_count
            FROM agent_anomalies
            WHERE org_id = :org_id
            GROUP BY agent_id
        ) an ON ae.agent_id = an.agent_id
        WHERE ae.org_id = :org_id {agent_filter}
        GROUP BY ae.agent_id, an.anomaly_count
        ORDER BY MAX(ae.timestamp_ns) DESC
        LIMIT :limit OFFSET :offset
    """), params)
    rows = result.fetchall()

    return {
        "agents": [
            {
                "agent_id":           r.agent_id,
                "session_count":      r.session_count,
                "llm_call_count":     r.llm_call_count,
                "tool_call_count":    r.tool_call_count,
                "error_count":        r.error_count,
                "total_tokens":       r.total_tokens,
                "avg_latency_ms":     float(r.avg_latency_ms) if r.avg_latency_ms is not None else None,
                "pii_event_count":    r.pii_event_count,
                "injection_score_max": float(r.injection_score_max) if r.injection_score_max is not None else 0.0,
                "anomaly_count":      r.anomaly_count,
                "first_seen":         r.first_seen.isoformat() if r.first_seen else None,
                "last_seen":          r.last_seen.isoformat() if r.last_seen else None,
            }
            for r in rows
        ],
        "total": len(rows),
        "limit": limit,
        "offset": offset,
    }


@router.get("/metrics/models", summary="Per-model usage statistics")
async def metrics_models(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Usage statistics grouped by LLM model, scoped to the authenticated organisation.
    """
    result = await db.execute(text("""
        SELECT
            model,
            MIN(provider)                                                           AS provider,
            COUNT(DISTINCT session_id)                                              AS session_count,
            COUNT(*) FILTER (WHERE event_type = 'LLM_CALL_START')                  AS call_count,
            COUNT(*) FILTER (WHERE event_type = 'SYSTEM_ERROR')                    AS error_count,
            COALESCE(SUM(
                CAST(NULLIF(payload->>'total_tokens', '') AS INTEGER)
            ) FILTER (WHERE event_type = 'LLM_CALL_END'
                        AND payload->>'total_tokens' IS NOT NULL), 0)               AS total_tokens,
            ROUND(AVG(
                CAST(payload->>'latency_ms' AS FLOAT)
            ) FILTER (WHERE event_type = 'LLM_CALL_END'
                        AND payload->>'latency_ms' IS NOT NULL)::numeric, 2)        AS avg_latency_ms,
            to_timestamp(MIN(timestamp_ns) / 1e9)                                   AS first_used,
            to_timestamp(MAX(timestamp_ns) / 1e9)                                   AS last_used
        FROM audit_events
        WHERE org_id = :org_id
          AND model IS NOT NULL AND model != '' AND model != 'unknown'
        GROUP BY model
        ORDER BY call_count DESC
    """), {"org_id": org_ctx.org_id})
    rows = result.fetchall()

    return {
        "models": [
            {
                "model":          r.model,
                "provider":       r.provider,
                "session_count":  r.session_count,
                "call_count":     r.call_count,
                "error_count":    r.error_count,
                "total_tokens":   r.total_tokens,
                "avg_latency_ms": float(r.avg_latency_ms) if r.avg_latency_ms is not None else None,
                "first_used":     r.first_used.isoformat() if r.first_used else None,
                "last_used":      r.last_used.isoformat() if r.last_used else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }
