"""
HITL (Human-in-the-Loop) Approvals API

GET  /v1/approvals              — List approvals (filter: status, agent_id, limit)
GET  /v1/approvals/{id}         — Get a single approval (proxy polling endpoint)
POST /v1/approvals              — Create approval request (called by proxy)
POST /v1/approvals/{id}/approve — Set status=approved
POST /v1/approvals/{id}/deny    — Set status=denied
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ApprovalCreate(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)
    agent_id: str = Field(min_length=1, max_length=256)
    tool_name: str = Field(min_length=1, max_length=256)
    tool_args: dict = Field(default_factory=dict)
    trigger: str = Field(min_length=1, max_length=64)
    timeout_seconds: int = Field(default=300, ge=10, le=3600)


class ApprovalDecision(BaseModel):
    decided_by: str | None = None
    decision_note: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: Any) -> dict:
    d = dict(row)
    for ts_field in ("created_at", "decided_at", "expires_at"):
        if d.get(ts_field) and hasattr(d[ts_field], "isoformat"):
            d[ts_field] = d[ts_field].isoformat()
    return d


async def _expire_stale(org_id: str, db: AsyncSession) -> None:
    """Mark pending approvals past their expiry time as 'expired'."""
    await db.execute(
        text("""
            UPDATE approvals
            SET status = 'expired'
            WHERE org_id = :org_id
              AND status = 'pending'
              AND expires_at < NOW()
        """),
        {"org_id": org_id},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/approvals", summary="List approvals")
async def list_approvals(
    status_filter: str | None = Query(None, alias="status"),
    agent_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    await _expire_stale(org_ctx.org_id, db)

    conditions = ["org_id = :org_id"]
    params: dict = {"org_id": org_ctx.org_id, "limit": limit}

    if status_filter and status_filter != "all":
        conditions.append("status = :status")
        params["status"] = status_filter

    if agent_id:
        conditions.append("agent_id = :agent_id")
        params["agent_id"] = agent_id

    where = " AND ".join(conditions)
    result = await db.execute(
        text(f"""
            SELECT id::text, org_id, session_id, agent_id, tool_name, tool_args,
                   trigger, status, decided_by, decision_note,
                   created_at, decided_at, expires_at
            FROM approvals
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        params,
    )
    rows = [_row_to_dict(r) for r in result.mappings().all()]

    # Count pending (for the nav badge)
    count_res = await db.execute(
        text("SELECT COUNT(*) FROM approvals WHERE org_id = :org_id AND status = 'pending'"),
        {"org_id": org_ctx.org_id},
    )
    pending_count = count_res.scalar() or 0

    return {"approvals": rows, "count": pending_count}


@router.get("/approvals/{approval_id}", summary="Get a single approval")
async def get_approval(
    approval_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    # Auto-expire before returning
    await _expire_stale(org_ctx.org_id, db)

    result = await db.execute(
        text("""
            SELECT id::text, org_id, session_id, agent_id, tool_name, tool_args,
                   trigger, status, decided_by, decision_note,
                   created_at, decided_at, expires_at
            FROM approvals
            WHERE id = :id::uuid AND org_id = :org_id
        """),
        {"id": approval_id, "org_id": org_ctx.org_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found")
    return _row_to_dict(row)


@router.post("/approvals", summary="Create an approval request (proxy → backend)")
async def create_approval(
    body: ApprovalCreate,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(
        text("""
            INSERT INTO approvals
                (org_id, session_id, agent_id, tool_name, tool_args, trigger, expires_at)
            VALUES
                (:org_id, :session_id, :agent_id, :tool_name, :tool_args::jsonb,
                 :trigger, NOW() + (:timeout_s || ' seconds')::interval)
            RETURNING id::text, created_at, expires_at
        """),
        {
            "org_id":     org_ctx.org_id,
            "session_id": body.session_id,
            "agent_id":   body.agent_id,
            "tool_name":  body.tool_name,
            "tool_args":  __import__("json").dumps(body.tool_args),
            "trigger":    body.trigger,
            "timeout_s":  str(body.timeout_seconds),
        },
    )
    row = result.mappings().first()
    await db.commit()

    approval_id = row["id"]
    logger.info(
        "HITL approval created: id=%s tool=%s agent=%s session=%s trigger=%s",
        approval_id, body.tool_name, body.agent_id, body.session_id, body.trigger,
    )
    return {
        "id": approval_id,
        "status": "pending",
        "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        "expires_at": row["expires_at"].isoformat() if hasattr(row["expires_at"], "isoformat") else str(row["expires_at"]),
    }


@router.post("/approvals/{approval_id}/approve", summary="Approve a pending tool call")
async def approve_request(
    approval_id: str,
    body: ApprovalDecision = ApprovalDecision(),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(
        text("""
            UPDATE approvals
            SET status = 'approved',
                decided_by = :decided_by,
                decision_note = :decision_note,
                decided_at = NOW()
            WHERE id = :id::uuid AND org_id = :org_id AND status = 'pending'
            RETURNING id::text
        """),
        {
            "id": approval_id,
            "org_id": org_ctx.org_id,
            "decided_by": body.decided_by,
            "decision_note": body.decision_note,
        },
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found or already decided",
        )
    await db.commit()
    logger.info("HITL approval approved: id=%s by=%s", approval_id, body.decided_by)
    return {"status": "approved", "id": approval_id}


@router.post("/approvals/{approval_id}/deny", summary="Deny a pending tool call")
async def deny_request(
    approval_id: str,
    body: ApprovalDecision = ApprovalDecision(),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(
        text("""
            UPDATE approvals
            SET status = 'denied',
                decided_by = :decided_by,
                decision_note = :decision_note,
                decided_at = NOW()
            WHERE id = :id::uuid AND org_id = :org_id AND status = 'pending'
            RETURNING id::text
        """),
        {
            "id": approval_id,
            "org_id": org_ctx.org_id,
            "decided_by": body.decided_by,
            "decision_note": body.decision_note,
        },
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found or already decided",
        )
    await db.commit()
    logger.info("HITL approval denied: id=%s by=%s note=%s", approval_id, body.decided_by, body.decision_note)
    return {"status": "denied", "id": approval_id}
