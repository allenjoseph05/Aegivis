"""GET /v1/sessions/{id}/verify — On-demand hash chain verification."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import get_session_events, get_session_summary
from ...middleware.auth import OrgContext, require_api_key
from ...services.hash_verifier import verify_session_chain

router = APIRouter()


@router.get("/sessions/{session_id}/verify", summary="Verify session hash chain")
async def verify_chain(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Recompute and verify the SHA-256 hash chain for a session.

    Returns:
    - valid: true if chain is intact
    - total_events: number of events checked
    - first_failed_sequence: sequence number of first tampered event (if any)
    - error_message: description of failure (if any)
    """
    summary = await get_session_summary(db, session_id=session_id, org_id=org_ctx.org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    events = await get_session_events(db, session_id=session_id, org_id=org_ctx.org_id, limit=50000)
    result = verify_session_chain(session_id, events)

    return result.to_dict()
