"""
GET  /v1/agents           — List all registered agents
POST /v1/agents           — Register a new agent
GET  /v1/agents/{id}      — Get a specific agent
PATCH /v1/agents/{id}     — Update agent (purpose, allowed_tools, owner)
DELETE /v1/agents/{id}    — Remove agent from registry
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Pydantic models ──────────────────────────────────────────────────────────

class AgentCreate(BaseModel):
    agent_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    declared_purpose: str = Field(default="", max_length=500)
    allowed_tools: list[str] = Field(default_factory=list)
    owner: str | None = None


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=128)
    declared_purpose: str | None = Field(default=None, max_length=500)
    allowed_tools: list[str] | None = None
    owner: str | None = None


class AgentResponse(BaseModel):
    agent_id: str
    org_id: str
    name: str
    declared_purpose: str
    allowed_tools: list[str]
    owner: str | None
    registered_at: str
    updated_at: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _row_to_dict(row: Any) -> dict:
    d = dict(row)
    for k in ("registered_at", "updated_at"):
        if d.get(k) and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    if d.get("allowed_tools") is None:
        d["allowed_tools"] = []
    return d


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/agents", summary="List registered agents")
async def list_agents(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """List all agents in the registry for the authenticated org."""
    result = await db.execute(text("""
        SELECT agent_id, org_id, name, declared_purpose, allowed_tools, owner,
               registered_at, updated_at
        FROM agents
        WHERE org_id = :org_id
        ORDER BY registered_at DESC
    """), {"org_id": org_ctx.org_id})
    rows = result.mappings().all()
    return {"agents": [_row_to_dict(r) for r in rows], "count": len(rows)}


@router.post("/agents", status_code=status.HTTP_201_CREATED, summary="Register an agent")
async def register_agent(
    body: AgentCreate,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Register a new agent with its declared purpose and allowed tools."""
    import json

    try:
        await db.execute(text("""
            INSERT INTO agents (agent_id, org_id, name, declared_purpose, allowed_tools, owner)
            VALUES (:agent_id, :org_id, :name, :declared_purpose, CAST(:allowed_tools AS jsonb), :owner)
        """), {
            "agent_id":        body.agent_id,
            "org_id":          org_ctx.org_id,
            "name":            body.name,
            "declared_purpose": body.declared_purpose,
            "allowed_tools":   json.dumps(body.allowed_tools),
            "owner":           body.owner,
        })
        await db.commit()
    except Exception as exc:
        await db.rollback()
        if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{body.agent_id}' is already registered in org '{org_ctx.org_id}'",
            )
        raise HTTPException(status_code=500, detail=str(exc)[:200])

    return {"status": "registered", "agent_id": body.agent_id}


@router.get("/agents/{agent_id}", summary="Get agent details")
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(text("""
        SELECT agent_id, org_id, name, declared_purpose, allowed_tools, owner,
               registered_at, updated_at
        FROM agents WHERE agent_id = :agent_id AND org_id = :org_id
    """), {"agent_id": agent_id, "org_id": org_ctx.org_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _row_to_dict(row)


@router.patch("/agents/{agent_id}", summary="Update agent registration")
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Update agent name, declared purpose, allowed tools, or owner."""
    import json

    updates: list[str] = ["updated_at = NOW()"]
    params: dict = {"agent_id": agent_id, "org_id": org_ctx.org_id}

    if body.name is not None:
        updates.append("name = :name")
        params["name"] = body.name
    if body.declared_purpose is not None:
        updates.append("declared_purpose = :declared_purpose")
        params["declared_purpose"] = body.declared_purpose
    if body.allowed_tools is not None:
        updates.append("allowed_tools = CAST(:allowed_tools AS jsonb)")
        params["allowed_tools"] = json.dumps(body.allowed_tools)
    if body.owner is not None:
        updates.append("owner = :owner")
        params["owner"] = body.owner

    result = await db.execute(text(f"""
        UPDATE agents SET {', '.join(updates)}
        WHERE agent_id = :agent_id AND org_id = :org_id
        RETURNING agent_id
    """), params)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    await db.commit()
    return {"status": "updated", "agent_id": agent_id}


@router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Remove agent")
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(text(
        "DELETE FROM agents WHERE agent_id = :agent_id AND org_id = :org_id RETURNING agent_id"
    ), {"agent_id": agent_id, "org_id": org_ctx.org_id})
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    await db.commit()
