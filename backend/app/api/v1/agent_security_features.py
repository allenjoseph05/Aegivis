"""
GET  /v1/agents/{agent_id}/security-features  — full 26-feature catalog with three-level resolution
PATCH /v1/agents/{agent_id}/security-features — upsert agent-level overrides (empty = reset to org/default)

Resolution order (highest → lowest priority):
  1. Agent-level override  (agent_security_profiles table)
  2. Org-level override    (org_settings table)
  3. Catalog default       (FEATURE_CATALOG default field)

Each feature in the response carries:
  is_agent_override  — True when set at agent level
  is_org_override    — True when set at org level (but not agent level)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key
from .security_features import (
    FEATURE_CATALOG,
    SECURITY_FEATURE_KEYS,
    _CATALOG_BY_KEY,
    _load_overrides,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_agent_overrides(
    org_id: str, agent_id: str, db: AsyncSession
) -> dict[str, str]:
    """Load all agent-level key/value overrides from agent_security_profiles."""
    rows = await db.execute(
        text(
            "SELECT key, value FROM agent_security_profiles "
            "WHERE org_id = :org_id AND agent_id = :agent_id AND key = ANY(:keys)"
        ),
        {"org_id": org_id, "agent_id": agent_id, "keys": list(SECURITY_FEATURE_KEYS)},
    )
    return {r.key: r.value for r in rows.mappings().all() if r.value is not None}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AgentFeatureUpdateRequest(BaseModel):
    updates: dict[str, str]  # key → new value; empty string = reset to org/default


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/agents/{agent_id}/security-features")
async def get_agent_security_features(
    agent_id: str,
    org_ctx: OrgContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_session),
):
    """
    Return the full feature catalog with three-level resolution for one agent.
    """
    org_overrides = await _load_overrides(org_ctx.org_id, db)
    agent_overrides = await _load_agent_overrides(org_ctx.org_id, agent_id, db)

    features = []
    for feat in FEATURE_CATALOG:
        key = feat["key"]
        is_agent = key in agent_overrides
        is_org = (not is_agent) and (key in org_overrides)

        if is_agent:
            value = agent_overrides[key]
        elif is_org:
            value = org_overrides[key]
        else:
            value = _CATALOG_BY_KEY[key]["default"]

        out = dict(feat)
        out["value"] = value
        out["is_agent_override"] = is_agent
        out["is_org_override"] = is_org
        features.append(out)

    return {"features": features, "agent_id": agent_id, "org_id": org_ctx.org_id}


@router.patch("/agents/{agent_id}/security-features")
async def update_agent_security_features(
    agent_id: str,
    body: AgentFeatureUpdateRequest,
    org_ctx: OrgContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_session),
):
    """
    Upsert agent-level security feature overrides.
    Pass empty string to remove the agent override (falls back to org/default).
    Unknown keys are silently ignored.
    """
    saved: list[str] = []

    for key, value in body.updates.items():
        if key not in SECURITY_FEATURE_KEYS:
            logger.warning("Ignoring unknown security feature key: %s", key)
            continue

        if value is None or value == "":
            # Delete agent-level override → falls back to org/default
            await db.execute(
                text(
                    "DELETE FROM agent_security_profiles "
                    "WHERE org_id = :org_id AND agent_id = :agent_id AND key = :key"
                ),
                {"org_id": org_ctx.org_id, "agent_id": agent_id, "key": key},
            )
        else:
            await db.execute(
                text(
                    "INSERT INTO agent_security_profiles "
                    "  (org_id, agent_id, key, value, updated_at) "
                    "VALUES (:org_id, :agent_id, :key, :value, NOW()) "
                    "ON CONFLICT (org_id, agent_id, key) DO UPDATE "
                    "SET value = EXCLUDED.value, updated_at = NOW()"
                ),
                {
                    "org_id": org_ctx.org_id,
                    "agent_id": agent_id,
                    "key": key,
                    "value": str(value),
                },
            )
        saved.append(key)

    await db.commit()
    logger.info(
        "Agent security features updated for org=%s agent=%s keys=%s",
        org_ctx.org_id, agent_id, saved,
    )
    return {"updated": saved, "count": len(saved)}
