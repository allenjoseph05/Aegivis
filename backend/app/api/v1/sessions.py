"""
GET /v1/sessions         — List sessions with summary stats
GET /v1/sessions/{id}    — Session detail + summary
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...db.connection import get_session
from ...db.queries import get_session_summary, list_sessions
from ...middleware.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_ORG = "default-org"


@router.get("/sessions", summary="List sessions")
async def list_sessions_endpoint(
    org_id: str = Query(default=DEFAULT_ORG),
    agent_id: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    """
    List agent sessions with summary statistics.
    Returns sessions ordered by most recent activity.
    """
    sessions = await list_sessions(
        db,
        org_id=org_id,
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
    org_id: str = Query(default=DEFAULT_ORG),
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    """Get summary statistics for a specific session."""
    summary = await get_session_summary(db, session_id=session_id, org_id=org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    result = dict(summary)
    for k, v in result.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()

    return result
