"""POST /v1/reports/generate — Generate compliance reports."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import get_session_events, get_session_summary
from ...middleware.auth import require_api_key
from ...services.compliance import REPORT_GENERATORS, generate_report
from ...services.hash_verifier import verify_session_chain

router = APIRouter()

VALID_REGULATIONS = list(REPORT_GENERATORS.keys())


class GenerateReportRequest(BaseModel):
    session_id: str
    org_id: str = "default-org"
    regulation: str  # "eu_ai_act" | "gdpr" | "hipaa" | "soc2"


@router.post("/reports/generate", summary="Generate compliance report")
async def generate_compliance_report(
    req: GenerateReportRequest,
    db: AsyncSession = Depends(get_session),
    _api_key: str = Depends(require_api_key),
):
    """
    Generate a structured compliance report for a session.

    Supported regulations:
    - eu_ai_act: EU AI Act Article 12
    - gdpr: GDPR Article 22
    - hipaa: HIPAA §164.312
    - soc2: SOC 2 Type II
    """
    if req.regulation not in VALID_REGULATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown regulation '{req.regulation}'. Valid: {VALID_REGULATIONS}",
        )

    summary = await get_session_summary(db, session_id=req.session_id, org_id=req.org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")

    events = await get_session_events(db, session_id=req.session_id, org_id=req.org_id, limit=50000)
    chain_result = verify_session_chain(req.session_id, events)

    report = generate_report(
        regulation=req.regulation,
        session_id=req.session_id,
        org_id=req.org_id,
        events=events,
        chain_valid=chain_result.valid,
    )

    return report.to_dict()
