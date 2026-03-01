"""
Tool Baseline API  —  /v1/baselines

Manages the per-agent tool approval workflow.

Discovery → Review → Approve/Deny → Enforce
--------------------------------------------
1. The proxy reports every tool it observes via POST /observe.
2. Operators see pending tools in the dashboard.
3. They approve or deny each tool (individually or all at once).
4. The proxy fetches the approved set via GET /approved and enforces it.

Endpoints
---------
GET  /v1/baselines
    Summary of all agents: how many tools are pending / approved / denied.

GET  /v1/baselines/{agent_id}/tools
    Full tool list for one agent with their current status and schema.

GET  /v1/baselines/{agent_id}/tools/approved
    Approved tool names only — consumed by the proxy cache.
    Returns 404 when the agent has no approved tools yet (audit mode).

POST /v1/baselines/{agent_id}/tools/observe
    Called by the proxy to upsert newly discovered tools.
    Idempotent: existing tools get their call_count and last_seen_at updated.

POST /v1/baselines/{agent_id}/tools/{tool_name}/approve
POST /v1/baselines/{agent_id}/tools/{tool_name}/deny
    Set the status of a single tool.

POST /v1/baselines/{agent_id}/tools/approve-all
    Approve every tool currently in 'pending' status for this agent.

Legacy endpoints (kept for backwards compatibility)
---------------------------------------------------
GET  /v1/baselines/{agent_id}          — behavioural baseline (drift check)
POST /v1/baselines/{agent_id}/drift    — drift check
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key
from ...services.baseline import check_drift, get_baseline

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class ObservedTool(BaseModel):
    name: str
    description: str = ""
    schema_: dict = {}


class ObserveRequest(BaseModel):
    tools: List[dict]   # list of {name, description, schema}
    session_id: str = ""
    org_id: str = ""    # ignored — taken from auth context


class DriftCheckRequest(BaseModel):
    session_stats: dict


# ─────────────────────────────────────────────────────────────────────────────
# Overview: all agents with baseline counts
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/baselines", summary="All agents with tool baseline summary")
async def list_baselines(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Returns one row per agent that has at least one observed tool.
    Each row includes counts of pending / approved / denied tools so the
    operator can quickly see which agents need review.
    """
    rows = await db.execute(text("""
        SELECT
            agent_id,
            COUNT(*) FILTER (WHERE status = 'pending')  AS pending,
            COUNT(*) FILTER (WHERE status = 'approved') AS approved,
            COUNT(*) FILTER (WHERE status = 'denied')   AS denied,
            COUNT(*)                                    AS total,
            MAX(last_seen_at)                           AS last_seen_at
        FROM agent_tool_baselines
        WHERE org_id = :org_id
        GROUP BY agent_id
        ORDER BY MAX(last_seen_at) DESC
    """), {"org_id": org_ctx.org_id})

    agents = []
    for r in rows.mappings():
        agents.append({
            "agent_id":    r["agent_id"],
            "pending":     r["pending"],
            "approved":    r["approved"],
            "denied":      r["denied"],
            "total":       r["total"],
            "last_seen_at": r["last_seen_at"].isoformat() if r["last_seen_at"] else None,
            # Enforcement is active once at least one tool is approved
            "enforcement_active": r["approved"] > 0,
        })
    return {"baselines": agents, "count": len(agents)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool list for one agent
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/baselines/{agent_id}/tools", summary="All tools for an agent")
async def list_agent_tools(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Returns all tools observed for *agent_id* with their current status,
    full schema, and usage statistics.
    """
    rows = await db.execute(text("""
        SELECT
            tool_name,
            tool_schema,
            status,
            call_count,
            first_seen_at,
            last_seen_at,
            approved_at
        FROM agent_tool_baselines
        WHERE org_id = :org_id AND agent_id = :agent_id
        ORDER BY
            CASE status WHEN 'pending' THEN 0 WHEN 'approved' THEN 1 ELSE 2 END,
            call_count DESC
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id})

    tools = []
    for r in rows.mappings():
        tools.append({
            "tool_name":     r["tool_name"],
            "tool_schema":   r["tool_schema"] or {},
            "status":        r["status"],
            "call_count":    r["call_count"],
            "first_seen_at": r["first_seen_at"].isoformat() if r["first_seen_at"] else None,
            "last_seen_at":  r["last_seen_at"].isoformat()  if r["last_seen_at"]  else None,
            "approved_at":   r["approved_at"].isoformat()   if r["approved_at"]   else None,
        })

    return {
        "agent_id": agent_id,
        "org_id":   org_ctx.org_id,
        "tools":    tools,
        "counts": {
            "pending":  sum(1 for t in tools if t["status"] == "pending"),
            "approved": sum(1 for t in tools if t["status"] == "approved"),
            "denied":   sum(1 for t in tools if t["status"] == "denied"),
            "total":    len(tools),
        },
        # Once any tool is approved, the proxy enforces the approved set.
        "enforcement_active": any(t["status"] == "approved" for t in tools),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Approved tool names — consumed by the proxy cache
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/baselines/{agent_id}/tools/approved",
    summary="Approved tool names (for proxy enforcement cache)",
)
async def get_approved_tools(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Returns the list of approved tool names for *agent_id*.

    The proxy calls this endpoint to populate its local cache.  If no
    tools are approved yet (agent is in audit mode) a 404 is returned so
    the proxy knows enforcement is not yet active.
    """
    rows = await db.execute(text("""
        SELECT tool_name
        FROM agent_tool_baselines
        WHERE org_id = :org_id AND agent_id = :agent_id AND status = 'approved'
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id})

    approved = [r[0] for r in rows]

    if not approved:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No approved tools for agent '{agent_id}' — agent is in audit mode. "
                "Approve at least one tool in the Baselines dashboard to activate enforcement."
            ),
        )

    return {"agent_id": agent_id, "approved_tools": approved, "count": len(approved)}


# ─────────────────────────────────────────────────────────────────────────────
# Observe (called by proxy — upsert discovered tools)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/baselines/{agent_id}/tools/observe",
    summary="Record newly observed tools (called by proxy)",
    status_code=200,
)
async def observe_tools(
    agent_id: str,
    body: ObserveRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Upsert tool definitions discovered by the proxy.

    - New tools are inserted with ``status = 'pending'``.
    - Existing tools get their ``call_count`` incremented and
      ``last_seen_at`` updated; their status is NOT changed.

    This endpoint is idempotent and designed for fire-and-forget calls
    from the proxy.  Schema changes (same name, different parameters)
    update the stored schema.
    """
    upserted = 0
    for tool in body.tools:
        name = tool.get("name", "").strip()
        if not name:
            continue
        schema_json = {
            "description": tool.get("description", ""),
            "schema": tool.get("schema", {}),
        }
        await db.execute(text("""
            INSERT INTO agent_tool_baselines
                (org_id, agent_id, tool_name, tool_schema, first_seen_at, last_seen_at, call_count, status)
            VALUES
                (:org_id, :agent_id, :tool_name, :schema::jsonb, NOW(), NOW(), 1, 'pending')
            ON CONFLICT (org_id, agent_id, tool_name) DO UPDATE SET
                tool_schema  = EXCLUDED.tool_schema,
                last_seen_at = NOW(),
                call_count   = agent_tool_baselines.call_count + 1
        """), {
            "org_id":    org_ctx.org_id,
            "agent_id":  agent_id,
            "tool_name": name,
            "schema":    __import__("json").dumps(schema_json),
        })
        upserted += 1

    await db.commit()
    logger.info(
        "Tool observe: agent=%s session=%s upserted=%d",
        agent_id, body.session_id, upserted,
    )
    return {"upserted": upserted}


# ─────────────────────────────────────────────────────────────────────────────
# Approve / deny individual tool
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/baselines/{agent_id}/tools/{tool_name}/approve",
    summary="Approve a tool for an agent",
)
async def approve_tool(
    agent_id: str,
    tool_name: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Mark *tool_name* as approved for *agent_id*.
    Once at least one tool is approved, the proxy switches from audit mode
    to enforcement mode for this agent.
    """
    result = await db.execute(text("""
        UPDATE agent_tool_baselines
        SET status = 'approved', approved_at = NOW()
        WHERE org_id = :org_id AND agent_id = :agent_id AND tool_name = :tool_name
        RETURNING tool_name
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id, "tool_name": tool_name})
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(404, f"Tool '{tool_name}' not found for agent '{agent_id}'")

    logger.info("Tool approved: agent=%s tool=%s org=%s", agent_id, tool_name, org_ctx.org_id)
    return {"agent_id": agent_id, "tool_name": tool_name, "status": "approved"}


@router.post(
    "/baselines/{agent_id}/tools/{tool_name}/deny",
    summary="Deny a tool for an agent",
)
async def deny_tool(
    agent_id: str,
    tool_name: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Mark *tool_name* as denied for *agent_id*.
    The proxy will block any tool call using this name.
    """
    result = await db.execute(text("""
        UPDATE agent_tool_baselines
        SET status = 'denied', approved_at = NOW()
        WHERE org_id = :org_id AND agent_id = :agent_id AND tool_name = :tool_name
        RETURNING tool_name
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id, "tool_name": tool_name})
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(404, f"Tool '{tool_name}' not found for agent '{agent_id}'")

    logger.info("Tool denied: agent=%s tool=%s org=%s", agent_id, tool_name, org_ctx.org_id)
    return {"agent_id": agent_id, "tool_name": tool_name, "status": "denied"}


# ─────────────────────────────────────────────────────────────────────────────
# Approve all pending tools at once
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/baselines/{agent_id}/tools/approve-all",
    summary="Approve all pending tools for an agent",
)
async def approve_all_tools(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Approve every tool currently in 'pending' status for *agent_id*.
    Useful when first baselining a new agent — review the list and approve
    everything the agent currently uses in one click.
    """
    result = await db.execute(text("""
        UPDATE agent_tool_baselines
        SET status = 'approved', approved_at = NOW()
        WHERE org_id = :org_id AND agent_id = :agent_id AND status = 'pending'
        RETURNING tool_name
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id})
    await db.commit()

    approved_names = [r[0] for r in result]
    logger.info(
        "Approve-all: agent=%s approved=%d tools=%s",
        agent_id, len(approved_names), approved_names,
    )
    return {
        "agent_id": agent_id,
        "approved_count": len(approved_names),
        "approved_tools": approved_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: behavioural baseline (drift check) — kept for backwards compat
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/baselines/{agent_id}", summary="Get agent behavioral baseline (legacy)")
async def get_agent_baseline(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    baseline = await get_baseline(db, org_id=org_ctx.org_id, agent_id=agent_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail=f"No baseline found for agent '{agent_id}'")
    return {
        "agent_id": agent_id,
        "org_id": org_ctx.org_id,
        **baseline,
        "last_updated": (
            baseline["last_updated"].isoformat()
            if baseline.get("last_updated") else None
        ),
    }


@router.post("/baselines/{agent_id}/drift", summary="Check session drift from baseline (legacy)")
async def check_agent_drift(
    agent_id: str,
    body: DriftCheckRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await check_drift(
        db,
        org_id=org_ctx.org_id,
        agent_id=agent_id,
        session_stats=body.session_stats,
    )
    return {
        "agent_id": result.agent_id,
        "org_id": result.org_id,
        "drifted": result.drifted,
        "drift_fields": result.drift_fields,
        "details": result.details,
    }
