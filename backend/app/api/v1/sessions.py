"""
GET /v1/sessions                    — List sessions with summary stats
GET /v1/sessions/{id}               — Session detail + summary
GET /v1/sessions/{id}/events        — Paginated audit events for a session
GET /v1/sessions/{id}/pdg           — Data flow graph (PDG) reconstruction
GET /v1/sessions/{id}/reasoning     — Extended thinking traces
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import get_session_summary, list_sessions
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions", summary="List sessions")
async def list_sessions_endpoint(
    agent_id: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    List agent sessions with summary statistics.
    Returns sessions ordered by most recent activity.
    """
    sessions = await list_sessions(
        db,
        org_id=org_ctx.org_id,
        limit=limit,
        offset=offset,
        agent_id=agent_id,
        provider=provider,
    )

    # Serialize datetime objects
    result = []
    for s in sessions:
        row = dict(s)
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
        result.append(row)

    return {
        "sessions": result,
        "count": len(result),
        "offset": offset,
        "limit": limit,
    }


@router.get("/sessions/{session_id}", summary="Get session detail")
async def get_session_detail(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Get summary statistics for a specific session."""
    summary = await get_session_summary(db, session_id=session_id, org_id=org_ctx.org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    result = dict(summary)
    for k, v in result.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()

    return result


@router.get("/sessions/{session_id}/events", summary="List audit events for a session")
async def list_session_events(
    session_id: str,
    event_type: str | None = Query(default=None, description="Filter by event type (e.g. TOOL_CALL_START)"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    List audit events for a session with pagination. Events are returned
    oldest-first (ascending sequence_number). Use limit/offset to page through
    large sessions. Payload and security JSONB fields are excluded by default
    for performance — fetch individual events via /v1/events/{id} if needed.
    """
    filters = ["session_id = :session_id", "org_id = :org_id"]
    params: dict = {
        "session_id": session_id,
        "org_id": org_ctx.org_id,
        "limit": limit,
        "offset": offset,
    }

    if event_type:
        filters.append("event_type = :event_type")
        params["event_type"] = event_type

    where = " AND ".join(filters)
    rows = await db.execute(
        text(f"""
            SELECT event_id, session_id, org_id, run_id, event_type, agent_id,
                   model, provider, sequence_number, timestamp_ns,
                   current_hash, previous_hash
            FROM audit_events
            WHERE {where}
            ORDER BY sequence_number ASC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    events = [dict(r) for r in rows.mappings().all()]

    count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
    count_row = await db.execute(
        text(f"SELECT COUNT(*) FROM audit_events WHERE {where}"),
        count_params,
    )
    total = count_row.scalar() or 0

    return {
        "session_id": session_id,
        "events": events,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/sessions/{session_id}/pdg", summary="Get session data flow graph")
async def get_session_pdg(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Reconstruct the Session PDG from stored TOOL_CALL_START security annotations."""
    rows = await db.execute(
        text("""
            SELECT event_id, timestamp_ns,
                   payload->>'tool_name' AS tool_name,
                   security->'pdg_edges'   AS pdg_edges,
                   security->'pdg_summary' AS pdg_summary
            FROM audit_events
            WHERE session_id = :session_id
              AND org_id     = :org_id
              AND event_type = 'TOOL_CALL_START'
              AND security IS NOT NULL
              AND (security->'pdg_edges') IS NOT NULL
              AND jsonb_array_length(security->'pdg_edges') > 0
            ORDER BY timestamp_ns ASC
        """),
        {"session_id": session_id, "org_id": org_ctx.org_id},
    )

    all_edges: list[dict] = []
    untrusted_sources: set[str] = set()
    tools_with_violations: set[str] = set()

    for row in rows.mappings().all():
        tool_name = row["tool_name"] or "unknown"
        edges = row["pdg_edges"] or []
        if isinstance(edges, str):
            import json as _json
            try:
                edges = _json.loads(edges)
            except Exception:
                edges = []
        for edge in (edges if isinstance(edges, list) else []):
            frag = edge.get("source_fragment") or edge.get("fragment") or ""
            all_edges.append({
                "tool_name": tool_name,
                "source_tool": edge.get("source_tool"),
                "source_fragment": frag,
                "sink_tool": edge.get("sink_tool") or tool_name,
                "sink_arg": edge.get("sink_arg") or edge.get("arg"),
                "hop_count": edge.get("hop_count", 1),
            })
            if frag:
                untrusted_sources.add(frag)
        if edges:
            tools_with_violations.add(tool_name)

    return {
        "session_id": session_id,
        "total_edges": len(all_edges),
        "total_untrusted_sources": len(untrusted_sources),
        "untrusted_sources": sorted(untrusted_sources),
        "tools_with_violations": sorted(tools_with_violations),
        "edges": all_edges,
    }


@router.get("/sessions/{session_id}/reasoning", summary="Get reasoning traces")
async def get_session_reasoning(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return all LLM_CALL_END events that contain extended thinking blocks."""
    rows = await db.execute(
        text("""
            SELECT event_id, run_id, model, timestamp_ns,
                   payload->'thinking_blocks' AS thinking_blocks,
                   jsonb_array_length(payload->'thinking_blocks') AS block_count
            FROM audit_events
            WHERE session_id = :session_id
              AND org_id = :org_id
              AND event_type = 'LLM_CALL_END'
              AND (payload->>'has_reasoning_trace')::boolean = true
            ORDER BY timestamp_ns ASC
        """),
        {"session_id": session_id, "org_id": org_ctx.org_id},
    )
    reasoning = [dict(r) for r in rows.mappings().all()]
    return {
        "session_id": session_id,
        "reasoning": reasoning,
        "total_calls_with_reasoning": len(reasoning),
    }

