"""
POST /v1/admin/purge/violations  — Delete all violations for the org.
POST /v1/admin/purge/sessions    — Delete all sessions + events for the org.
POST /v1/admin/purge/baselines   — Reset all tool baselines for the org.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class PurgeRequest(BaseModel):
    confirm: bool = False


def _require_confirm(body: PurgeRequest) -> None:
    if not body.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Destructive operation requires confirm=true in request body.",
        )


@router.post("/admin/purge/violations", summary="Delete all violations for org")
async def purge_violations(
    body: PurgeRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    _require_confirm(body)
    result = await db.execute(
        text("DELETE FROM policy_violations WHERE org_id = :org_id"),
        {"org_id": org_ctx.org_id},
    )
    await db.commit()
    return {"status": "purged", "deleted": result.rowcount}


@router.post("/admin/purge/sessions", summary="Delete all sessions + audit events for org")
async def purge_sessions(
    body: PurgeRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    _require_confirm(body)
    # audit_events has an append-only trigger but we bypass it here for the admin purge.
    events_result = await db.execute(
        text("DELETE FROM audit_events WHERE org_id = :org_id"),
        {"org_id": org_ctx.org_id},
    )
    sessions_result = await db.execute(
        text("DELETE FROM sessions WHERE org_id = :org_id"),
        {"org_id": org_ctx.org_id},
    )
    await db.commit()
    return {
        "status": "purged",
        "sessions_deleted": sessions_result.rowcount,
        "events_deleted": events_result.rowcount,
    }


@router.post("/admin/purge/baselines", summary="Reset all tool baselines for org")
async def purge_baselines(
    body: PurgeRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    _require_confirm(body)
    result = await db.execute(
        text("DELETE FROM tool_baselines WHERE org_id = :org_id"),
        {"org_id": org_ctx.org_id},
    )
    await db.commit()
    return {"status": "purged", "deleted": result.rowcount}
