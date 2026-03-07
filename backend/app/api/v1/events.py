"""GET /v1/sessions/{id}/events — Retrieve events for a session."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import get_session_events, get_session_summary
from ...middleware.auth import OrgContext, require_api_key

router = APIRouter()


@router.get("/sessions/{session_id}/events", summary="Get session events")
async def get_events(
    session_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Retrieve all events for a session, ordered by sequence_number.
    Used by dashboard event timeline and forensic replay.
    """
    # Check session exists (scoped to org)
    summary = await get_session_summary(db, session_id=session_id, org_id=org_ctx.org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    events = await get_session_events(
        db,
        session_id=session_id,
        org_id=org_ctx.org_id,
        limit=limit,
        offset=offset,
        event_type=event_type,
    )

    return {
        "session_id": session_id,
        "events": events,
        "count": len(events),
        "offset": offset,
        "limit": limit,
    }
