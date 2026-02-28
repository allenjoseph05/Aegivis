"""
Policy Builder endpoints:

GET /v1/policy/observed-tools  — per-agent, per-tool usage (last N days)
GET /v1/policy/suggestions     — auto-generated policy rules from observed traffic
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/policy/observed-tools", summary="Observed tool usage per agent (last 30 days)")
async def get_observed_tools(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Returns per-agent, per-tool activity stats derived from stored events.
    Used by the Policy Builder to show 'what your agents actually do' so
    the operator can create BLOCK/ALERT rules from observed behavior.
    """
    stmt = text("""
        SELECT
            ae.agent_id,
            ae.payload->>'tool_name'                  AS tool_name,
            COUNT(*)                                   AS call_count,
            COUNT(pv.id) FILTER (WHERE pv.id IS NOT NULL) AS violation_count,
            to_timestamp(MAX(ae.timestamp_ns) / 1e9)  AS last_seen
        FROM audit_events ae
        LEFT JOIN policy_violations pv
            ON pv.agent_id = ae.agent_id
            AND pv.session_id = ae.session_id
            AND pv.org_id = ae.org_id
        WHERE ae.org_id      = :org_id
          AND ae.event_type  = 'TOOL_CALL_START'
          AND ae.timestamp_ns > (EXTRACT(EPOCH FROM NOW()) - :window_s) * 1e9
          AND ae.payload->>'tool_name' IS NOT NULL
        GROUP BY ae.agent_id, ae.payload->>'tool_name'
        ORDER BY call_count DESC
        LIMIT 500
    """)

    try:
        result = await db.execute(stmt, {
            "org_id": org_ctx.org_id,
            "window_s": days * 86400,
        })
        rows = result.mappings().all()
    except Exception as exc:
        logger.warning("observed-tools query failed: %s", exc)
        rows = []

    return {
        "tools": [
            {
                "agent_id":        r["agent_id"],
                "tool_name":       r["tool_name"],
                "call_count":      r["call_count"],
                "violation_count": r["violation_count"] or 0,
                "last_seen":       r["last_seen"].isoformat() if r["last_seen"] else None,
            }
            for r in rows
        ],
        "window_days": days,
        "org_id": org_ctx.org_id,
    }


@router.get("/policy/suggestions", summary="Auto-generated policy rules from observed traffic")
async def get_policy_suggestions(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Analyses stored events and returns 0-N ready-to-apply policy rule suggestions.
    """
    org_id = org_ctx.org_id
    window_s = days * 86400
    suggestions: list[dict] = []

    # ── 1. Model allowlist ───────────────────────────────────────────────────
    try:
        models_rows = (
            await db.execute(
                text("""
                    SELECT model, COUNT(*) AS calls
                    FROM audit_events
                    WHERE org_id      = :org_id
                      AND event_type  = 'LLM_CALL_START'
                      AND timestamp_ns > (EXTRACT(EPOCH FROM NOW()) - :window_s) * 1e9
                      AND model IS NOT NULL AND model != '' AND model != 'unknown'
                    GROUP BY model
                    ORDER BY calls DESC
                    LIMIT 20
                """),
                {"org_id": org_id, "window_s": window_s},
            )
        ).mappings().all()
    except Exception as exc:
        logger.warning("suggestions: model query failed: %s", exc)
        await db.rollback()
        models_rows = []

    if models_rows:
        model_names = [r["model"] for r in models_rows]
        top = ", ".join(
            f"{r['model']} ({r['calls']} calls)" for r in models_rows[:3]
        )
        suffix = f" +{len(model_names) - 3} more" if len(model_names) > 3 else ""
        suggestions.append(
            {
                "id": "model-allowlist",
                "title": "Model Allowlist",
                "description": "Block any model not observed in production traffic.",
                "evidence": f"Detected: {top}{suffix}",
                "rule": {
                    "name": "unapproved-model-block",
                    "event_types": ["LLM_CALL_START"],
                    "conditions": [
                        {"field": "model", "op": "not_in", "value": model_names}
                    ],
                    "action": "BLOCK",
                    "reason": (
                        f"Agent used an unapproved model. "
                        f"Approved: {', '.join(model_names)}"
                    ),
                    "enabled": False,
                },
            }
        )

    # ── 2. Tool call rate cap ────────────────────────────────────────────────
    try:
        rate_row = (
            await db.execute(
                text("""
                    SELECT
                        percentile_cont(0.95)
                            WITHIN GROUP (ORDER BY tool_count) AS p95,
                        MAX(tool_count)  AS max_count,
                        COUNT(*)         AS session_count
                    FROM (
                        SELECT session_id, COUNT(*) AS tool_count
                        FROM audit_events
                        WHERE org_id     = :org_id
                          AND event_type = 'TOOL_CALL_START'
                          AND timestamp_ns > (EXTRACT(EPOCH FROM NOW()) - :window_s) * 1e9
                        GROUP BY session_id
                    ) sub
                """),
                {"org_id": org_id, "window_s": window_s},
            )
        ).mappings().first()
    except Exception as exc:
        logger.warning("suggestions: rate query failed: %s", exc)
        await db.rollback()
        rate_row = None

    if rate_row and rate_row["p95"] and int(rate_row["session_count"]) >= 5:
        p95 = int(rate_row["p95"])
        cap = max(p95 * 2, p95 + 10)
        suggestions.append(
            {
                "id": "tool-rate-cap",
                "title": "Tool Call Rate Cap",
                "description": "Tighten the default cap (50) based on your actual traffic pattern.",
                "evidence": (
                    f"p95 = {p95} tool calls/session across "
                    f"{rate_row['session_count']} sessions. "
                    f"Suggested cap: {cap}."
                ),
                "rule": {
                    "name": "tool-call-loop-protection",
                    "event_types": ["TOOL_CALL_START"],
                    "conditions": [
                        {"field": "tool_call_count", "op": "gt", "value": cap}
                    ],
                    "action": "BLOCK",
                    "reason": (
                        f"Tool call cap based on observed p95 ({p95} calls/session). "
                        f"Cap: {cap}."
                    ),
                    "enabled": True,
                },
            }
        )

    # ── Minimum data check ───────────────────────────────────────────────────
    try:
        total = (
            await db.execute(
                text("""
                    SELECT COUNT(*) FROM audit_events
                    WHERE org_id = :org_id
                      AND timestamp_ns > (EXTRACT(EPOCH FROM NOW()) - :window_s) * 1e9
                """),
                {"org_id": org_id, "window_s": window_s},
            )
        ).scalar() or 0
    except Exception:
        total = 0

    return {
        "suggestions": suggestions,
        "window_days": days,
        "total_events_analyzed": int(total),
        "insufficient_data": int(total) < 10,
    }
