"""
GET /v1/baselines/{agent_id} — Get behavioral baseline for an agent.
GET /v1/baselines/{agent_id}/drift  — Check if a session drifted from baseline.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import require_api_key
from ...services.baseline import check_drift, get_baseline

router = APIRouter()


@router.get("/baselines/{agent_id}", summary="Get agent behavioral baseline")
async def get_agent_baseline(
    agent_id: str,
    org_id: str = "default-org",
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    baseline = await get_baseline(db, org_id=org_id, agent_id=agent_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail=f"No baseline found for agent '{agent_id}'")
    return {
        "agent_id": agent_id,
        "org_id": org_id,
        **baseline,
        "last_updated": (
            baseline["last_updated"].isoformat()
            if baseline.get("last_updated") else None
        ),
    }


class DriftCheckRequest(BaseModel):
    session_stats: dict
    org_id: str = "default-org"


@router.post("/baselines/{agent_id}/drift", summary="Check session drift from baseline")
async def check_agent_drift(
    agent_id: str,
    body: DriftCheckRequest,
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    result = await check_drift(
        db,
        org_id=body.org_id,
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
