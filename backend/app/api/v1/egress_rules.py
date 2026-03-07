"""
Egress Rules API — dashboard-configurable network destination allowlist.

GET    /v1/egress-rules              — list rules for org
POST   /v1/egress-rules              — add a destination
DELETE /v1/egress-rules/{destination} — remove a destination

The proxy fetches this list (with 60s cache) instead of reading a static
env var, so rules take effect within one cache TTL of saving.

Destination formats accepted:
  api.openai.com          — exact hostname
  *.anthropic.com         — wildcard subdomain
  10.0.0.0/8              — CIDR range (IPv4)
  db.internal:5432        — hostname with port
"""
from __future__ import annotations

import logging
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class AddEgressRuleRequest(BaseModel):
    destination: str
    notes: str = ""


# ---------------------------------------------------------------------------
# GET /v1/egress-rules
# ---------------------------------------------------------------------------

@router.get("/egress-rules", summary="List egress allowlist rules")
async def list_egress_rules(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    result = await db.execute(text("""
        SELECT destination, notes, created_at, created_by
        FROM egress_rules
        WHERE org_id = :org_id
        ORDER BY destination
    """), {"org_id": org_ctx.org_id})
    rows = result.mappings().fetchall()
    rules = [
        {
            "destination": r["destination"],
            "notes": r["notes"] or "",
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "created_by": r["created_by"],
        }
        for r in rows
    ]
    return {"rules": rules, "count": len(rules)}


# ---------------------------------------------------------------------------
# POST /v1/egress-rules
# ---------------------------------------------------------------------------

@router.post("/egress-rules", summary="Add egress allowlist rule", status_code=201)
async def add_egress_rule(
    req: AddEgressRuleRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    destination = req.destination.strip().lower()
    if not destination:
        raise HTTPException(status_code=422, detail="destination must not be empty")

    try:
        await db.execute(text("""
            INSERT INTO egress_rules (org_id, destination, notes)
            VALUES (:org_id, :destination, :notes)
            ON CONFLICT (org_id, destination) DO UPDATE SET notes = EXCLUDED.notes
        """), {"org_id": org_ctx.org_id, "destination": destination, "notes": req.notes})
        await db.commit()
    except Exception as exc:
        logger.warning("Failed to add egress rule %s: %s", destination, exc)
        raise HTTPException(status_code=500, detail="Failed to save rule")

    logger.info("Egress rule added: org=%s dest=%s", org_ctx.org_id, destination)
    return {"destination": destination, "added": True}


# ---------------------------------------------------------------------------
# DELETE /v1/egress-rules/{destination}
# ---------------------------------------------------------------------------

@router.delete("/egress-rules/{destination:path}", summary="Remove egress allowlist rule")
async def delete_egress_rule(
    destination: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    dest = unquote(destination).strip().lower()
    result = await db.execute(text("""
        DELETE FROM egress_rules
        WHERE org_id = :org_id AND destination = :destination
        RETURNING destination
    """), {"org_id": org_ctx.org_id, "destination": dest})
    await db.commit()
    deleted = [row[0] for row in result.fetchall()]
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Rule '{dest}' not found")
    return {"destination": dest, "deleted": True}
