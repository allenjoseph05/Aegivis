"""GET /v1/sessions/{id}/verify — On-demand hash chain verification."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import get_session_events, get_session_summary
from ...middleware.auth import require_api_key
from ...services.hash_verifier import verify_session_chain

router = APIRouter()


@router.get("/sessions/{session_id}/verify", summary="Verify session hash chain")
async def verify_chain(
    session_id: str,
    org_id: str = Query(default="default-org"),
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    """
    Recompute and verify the SHA-256 hash chain for a session.

    Returns:
    - valid: true if chain is intact
    - total_events: number of events checked
    - first_failed_sequence: sequence number of first tampered event (if any)
    - error_message: description of failure (if any)

    This is mathematically guaranteed: any modification to any historical event
    causes all subsequent hashes to diverge, making tampering detectable.
    """
    summary = await get_session_summary(db, session_id=session_id, org_id=org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    events = await get_session_events(db, session_id=session_id, org_id=org_id, limit=50000)
    result = verify_session_chain(session_id, events)

    return result.to_dict()
