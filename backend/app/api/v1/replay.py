"""GET /v1/sessions/{id}/replay — Forensic replay report."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...db.queries import get_session_events, get_session_summary
from ...middleware.auth import OrgContext, require_api_key
from ...services.anomaly import detect_anomalies
from ...services.forensic_replay import build_forensic_report
from ...services.hash_verifier import verify_session_chain

router = APIRouter()


@router.get("/sessions/{session_id}/replay", summary="Get forensic replay report")
async def get_forensic_replay(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Generate a forensic replay report for a session, scoped to the authenticated org.

    Fetches up to 50 000 events per session in one query. Full chain verification
    and anomaly detection require all events to be in memory simultaneously, so
    streaming / chunked pagination is not supported. Sessions larger than 50 000
    events are uncommon in practice (~4 h+ of continuous agent operation) but if
    encountered the endpoint will return a partial report for the first 50 000 events.
    """
    summary = await get_session_summary(db, session_id=session_id, org_id=org_ctx.org_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    events = await get_session_events(db, session_id=session_id, org_id=org_ctx.org_id, limit=50000)

    # Verify chain
    chain_result = verify_session_chain(session_id, events)

    # Build forensic report
    report = build_forensic_report(session_id, events, chain_valid=chain_result.valid)

    # Run anomaly detection
    anomalies = detect_anomalies(events, chain_valid=chain_result.valid)

    return {
        "session_id": session_id,
        "chain_valid": chain_result.valid,
        "chain_check": chain_result.to_dict(),
        "report": {
            "org_id": report.org_id,
            "agent_id": report.agent_id,
            "provider": report.provider,
            "model": report.model,
            "started_at_ns": report.started_at_ns,
            "ended_at_ns": report.ended_at_ns,
            "duration_ms": report.duration_ms,
            "total_events": report.total_events,
            "llm_call_count": report.llm_call_count,
            "tool_call_count": report.tool_call_count,
            "error_count": report.error_count,
            "pii_exposure_count": report.pii_exposure_count,
            "total_tokens": report.total_tokens,
            "timeline": [
                {
                    "sequence_number": e.sequence_number,
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "label": e.label,
                    "severity": e.severity,
                    "timestamp_ns": e.timestamp_ns,
                    "summary": e.summary,
                    "details": e.details,
                    "pii_detected": e.pii_detected,
                    "run_id": e.run_id,
                    "parent_run_id": e.parent_run_id,
                }
                for e in report.timeline
            ],
            "data_access_map": [
                {
                    "tool_name": d.tool_name,
                    "call_count": d.call_count,
                    "first_seen_ns": d.first_seen_ns,
                    "last_seen_ns": d.last_seen_ns,
                }
                for d in report.data_access_map
            ],
            "risk_flags": [
                {
                    "severity": f.severity,
                    "flag_type": f.flag_type,
                    "description": f.description,
                    "sequence_number": f.sequence_number,
                    "event_id": f.event_id,
                }
                for f in report.risk_flags
            ],
        },
        "anomalies": [
            {
                "rule_id": a.rule_id,
                "severity": a.severity,
                "description": a.description,
                "event_id": a.event_id,
                "sequence_number": a.sequence_number,
                "metadata": a.metadata,
            }
            for a in anomalies
        ],
    }
