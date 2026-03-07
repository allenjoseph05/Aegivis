"""
Capability Manifest API — Phase 12.

GET  /v1/manifests/{agent_id}/active      — get active manifest for agent
POST /v1/manifests/{agent_id}/generate    — generate + sign manifest from observations
GET  /v1/manifests/{agent_id}/compliance  — live compliance stats (blocked vs allowed)
DELETE /v1/manifests/{agent_id}           — revoke current manifest
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateManifestRequest(BaseModel):
    validity_days: int = 30
    enforcement_mode: str = "block"   # "block" | "alert"
    notes: str = ""
    # Override auto-generated fields (optional — for manual manifest creation)
    permitted_tools: list[dict] | None = None
    permitted_network_destinations: list[str] | None = None


# ---------------------------------------------------------------------------
# Signing helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sign_manifest(body: dict) -> str:
    """Sign manifest body with HMAC-SHA256 using the configured signing key."""
    key = settings.manifest_signing_key or "dev-insecure-key"
    canonical = _canonical_json(body)
    digest = hmac.new(key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# GET /v1/manifests/{agent_id}/active
# ---------------------------------------------------------------------------

@router.get("/manifests/{agent_id}/active", summary="Get active capability manifest")
async def get_active_manifest(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return the currently active, non-expired, non-revoked manifest for this agent."""
    row = await db.execute(text("""
        SELECT
            manifest_id, agent_id, org_id,
            issued_at, expires_at, version,
            permitted_tools, permitted_network_destinations, ifc_policy,
            manifest_json, signature, notes
        FROM capability_manifests
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND revoked_at IS NULL
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY issued_at DESC
        LIMIT 1
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id})
    manifest = row.mappings().fetchone()
    if not manifest:
        raise HTTPException(status_code=404, detail=f"No active manifest for agent '{agent_id}'")

    result = dict(manifest)
    for k, v in result.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        elif hasattr(v, "__json__"):
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# POST /v1/manifests/{agent_id}/generate
# ---------------------------------------------------------------------------

@router.post("/manifests/{agent_id}/generate", summary="Generate and sign capability manifest")
async def generate_manifest(
    agent_id: str,
    req: GenerateManifestRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Generate a capability manifest from observed tool behavior, sign it,
    store it, and revoke any previous manifests for this agent.
    """
    org_id = org_ctx.org_id

    # Step 1: Gather observed tool behavior from audit_events (last 30 days)
    if req.permitted_tools is None:
        obs_tools = await _observe_tools(db, org_id, agent_id)
    else:
        obs_tools = req.permitted_tools

    if req.permitted_network_destinations is None:
        obs_dests = await _observe_destinations(db, org_id, agent_id, obs_tools)
    else:
        obs_dests = req.permitted_network_destinations

    # Step 2: Build manifest body
    manifest_id = f"mfst_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=req.validity_days)

    body = {
        "manifest_id": manifest_id,
        "agent_id": agent_id,
        "org_id": org_id,
        "issued_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "version": 1,
        "permitted_tools": obs_tools,
        "permitted_network_destinations": obs_dests,
        "ifc_policy": {
            "external_sources": [
                t["name"] for t in obs_tools
                if t.get("trust_classification") == "external_source"
            ],
            "privileged_sinks": [
                t["name"] for t in obs_tools
                if t.get("trust_classification") == "internal_sink"
            ],
            "block_external_to_sink": True,
        },
    }

    # Step 3: Sign
    signature = _sign_manifest(body)
    body["signature"] = signature
    manifest_json = _canonical_json(body)

    # Step 4: Revoke any existing active manifests for this agent
    await db.execute(text("""
        UPDATE capability_manifests
        SET revoked_at = NOW()
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND revoked_at IS NULL
    """), {"org_id": org_id, "agent_id": agent_id})

    # Step 5: Insert new manifest
    await db.execute(text("""
        INSERT INTO capability_manifests (
            manifest_id, org_id, agent_id, issued_at, expires_at, version,
            permitted_tools, permitted_network_destinations, ifc_policy,
            manifest_json, signature, notes
        ) VALUES (
            :manifest_id, :org_id, :agent_id, :issued_at, :expires_at, :version,
            :permitted_tools::jsonb, :permitted_network_destinations::jsonb, :ifc_policy::jsonb,
            :manifest_json, :signature, :notes
        )
    """), {
        "manifest_id": manifest_id,
        "org_id": org_id,
        "agent_id": agent_id,
        "issued_at": now,
        "expires_at": expires,
        "version": 1,
        "permitted_tools": json.dumps(obs_tools),
        "permitted_network_destinations": json.dumps(obs_dests),
        "ifc_policy": json.dumps(body["ifc_policy"]),
        "manifest_json": manifest_json,
        "signature": signature,
        "notes": req.notes or f"Auto-generated from {len(obs_tools)} observed tools",
    })
    await db.commit()

    logger.info("Generated manifest %s for agent %s (org %s)", manifest_id, agent_id, org_id)
    return {
        **body,
        "manifest_json": manifest_json,
        "tools_count": len(obs_tools),
        "destinations_count": len(obs_dests),
    }


# ---------------------------------------------------------------------------
# DELETE /v1/manifests/{agent_id}
# ---------------------------------------------------------------------------

@router.delete("/manifests/{agent_id}", summary="Revoke active manifest")
async def revoke_manifest(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(text("""
        UPDATE capability_manifests
        SET revoked_at = NOW()
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND revoked_at IS NULL
        RETURNING manifest_id
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id})
    await db.commit()
    revoked = [row[0] for row in result.fetchall()]
    return {"revoked": revoked, "count": len(revoked)}


# ---------------------------------------------------------------------------
# GET /v1/manifests/{agent_id}/compliance
# ---------------------------------------------------------------------------

@router.get("/manifests/{agent_id}/compliance", summary="Live manifest compliance stats")
async def get_compliance(
    agent_id: str,
    hours: int = Query(default=24, ge=1, le=720),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return live compliance stats: allowed vs blocked tool calls against the manifest."""
    result = await db.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE event_type = 'TOOL_CALL_START'
                AND (security->>'manifest_violation') IS NULL) AS allowed,
            COUNT(*) FILTER (WHERE event_type = 'TOOL_CALL_START'
                AND (security->>'manifest_violation') IS NOT NULL) AS blocked
        FROM audit_events
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND timestamp > NOW() - INTERVAL '1 hour' * :hours
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id, "hours": hours})
    row = result.fetchone()
    return {
        "agent_id": agent_id,
        "window_hours": hours,
        "allowed_tool_calls": row[0] if row else 0,
        "blocked_tool_calls": row[1] if row else 0,
    }


# ---------------------------------------------------------------------------
# GET /v1/manifests/{agent_id}/list
# ---------------------------------------------------------------------------

@router.get("/manifests/{agent_id}/list", summary="List all manifests for agent")
async def list_manifests(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(text("""
        SELECT manifest_id, issued_at, expires_at, revoked_at, version, notes
        FROM capability_manifests
        WHERE org_id = :org_id AND agent_id = :agent_id
        ORDER BY issued_at DESC
        LIMIT 20
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id})
    rows = result.mappings().fetchall()
    manifests = []
    for r in rows:
        row = dict(r)
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
        manifests.append(row)
    return {"manifests": manifests, "count": len(manifests)}


# ---------------------------------------------------------------------------
# Observation helpers — mine tool behavior from audit trail
# ---------------------------------------------------------------------------

async def _observe_tools(db: AsyncSession, org_id: str, agent_id: str) -> list[dict]:
    """
    Mine the last 30 days of tool call events to build a permitted_tools list.
    Returns a list of tool dicts suitable for the manifest.
    """
    result = await db.execute(text("""
        SELECT
            ae.event_data->>'tool_name'                                       AS tool_name,
            COUNT(*)                                                           AS call_count,
            jsonb_agg(DISTINCT ae.event_data->'tool_arguments') FILTER (WHERE ae.event_data->'tool_arguments' IS NOT NULL) AS sample_args
        FROM audit_events ae
        WHERE ae.org_id = :org_id
          AND ae.agent_id = :agent_id
          AND ae.event_type = 'TOOL_CALL_START'
          AND ae.timestamp > NOW() - INTERVAL '30 days'
          AND ae.event_data->>'tool_name' IS NOT NULL
        GROUP BY ae.event_data->>'tool_name'
        ORDER BY call_count DESC
    """), {"org_id": org_id, "agent_id": agent_id})
    rows = result.mappings().fetchall()

    tools = []
    for r in rows:
        tool_name = r["tool_name"]
        call_count = r["call_count"]
        # Derive permitted_arguments from observed samples (simplified: collect unique arg keys)
        sample_args_list = r.get("sample_args") or []
        arg_keys: set[str] = set()
        for sample in sample_args_list:
            if isinstance(sample, dict):
                arg_keys.update(sample.keys())

        permitted_arguments: dict[str, dict] = {}
        for k in sorted(arg_keys):
            permitted_arguments[k] = {"type": "string"}  # basic type; operator can refine

        # Classify trust based on tool name heuristics
        trust = _classify_tool_trust(tool_name)

        tools.append({
            "name": tool_name,
            "trust_classification": trust,
            "permitted_arguments": permitted_arguments,
            "requires_hitl": trust == "internal_sink",
            "hitl_reason": "Writes data — review required" if trust == "internal_sink" else "",
            "notes": f"Observed {call_count} times in last 30 days",
        })

    return tools


async def _observe_destinations(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
    tools: list[dict],
) -> list[str]:
    """
    Mine observed network destinations from the security JSONB column.
    Returns a deduplicated list of hostnames.
    """
    result = await db.execute(text("""
        SELECT DISTINCT ae.security->>'ssrf_destination' AS dest
        FROM audit_events ae
        WHERE ae.org_id = :org_id
          AND ae.agent_id = :agent_id
          AND ae.timestamp > NOW() - INTERVAL '30 days'
          AND ae.security->>'ssrf_destination' IS NOT NULL
    """), {"org_id": org_id, "agent_id": agent_id})
    rows = result.fetchall()
    dests = [row[0] for row in rows if row[0]]

    # Always include the LLM provider endpoints
    dests += ["api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com"]
    return sorted(set(dests))


# Tool name → trust classification heuristic (same logic as IFC source labeling)
_EXTERNAL_SOURCE_RE = re.compile(
    r"search|fetch|browse|retrieve|scrape|crawl|download|read|get|query|"
    r"lookup|rss|news|email_read|inbox|web|http_get|request|external",
    re.IGNORECASE,
)
_SINK_RE = re.compile(
    r"send|email|mail|webhook|upload|push|publish|notify|post|export|"
    r"relay|forward|submit|dispatch|write|save|store|insert|create",
    re.IGNORECASE,
)


def _classify_tool_trust(tool_name: str) -> str:
    if _EXTERNAL_SOURCE_RE.search(tool_name):
        return "external_source"
    if _SINK_RE.search(tool_name):
        return "internal_sink"
    return "internal"
