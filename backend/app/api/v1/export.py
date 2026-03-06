"""
SIEM export endpoints.

Transforms audit events into formats consumed by common SIEM / data-lake
platforms and either streams them back to the caller or pushes them directly
to a remote endpoint.

Endpoints
---------
GET  /v1/export/jsonlines      — Stream events as newline-delimited JSON.
POST /v1/export/splunk         — Push to a Splunk HTTP Event Collector.
POST /v1/export/elasticsearch  — Push to an Elasticsearch Bulk API endpoint.
GET  /v1/export/audit-report   — Org-wide compliance audit report (OWASP/EU AI Act/HIPAA/SOC 2).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key
from ...services.export import (
    push_to_elasticsearch,
    push_to_splunk,
    stream_jsonlines,
)
from ...services.audit_report import build_audit_report

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class SplunkPushRequest(BaseModel):
    """Configuration for a Splunk HEC push."""

    hec_url: str = Field(
        ...,
        description=(
            "Full Splunk HEC endpoint, e.g. "
            "https://splunk.example.com:8088/services/collector/event"
        ),
    )
    hec_token: str = Field(..., description="Splunk HEC token (without 'Splunk ' prefix).")
    index: str = Field("aegivis", description="Splunk index name.")
    source: str = Field("aegivis-proxy", description="Splunk source field.")
    # Optional event filters
    session_id: str | None = Field(None, description="Filter to a single session.")
    agent_id:   str | None = Field(None, description="Filter to a single agent.")
    from_ts_ns: int | None = Field(None, description="Start timestamp (nanoseconds).")
    to_ts_ns:   int | None = Field(None, description="End timestamp (nanoseconds).")
    limit: int = Field(5_000, ge=1, le=50_000, description="Maximum events to export.")


class ElasticPushRequest(BaseModel):
    """Configuration for an Elasticsearch Bulk API push."""

    es_url: str = Field(
        ...,
        description="Elasticsearch base URL, e.g. https://es.example.com:9200",
    )
    api_key: str | None = Field(
        None,
        description="Elasticsearch API key (without 'ApiKey ' prefix). Omit for open clusters.",
    )
    index: str = Field("aegivis", description="Elasticsearch index name.")
    # Optional event filters
    session_id: str | None = Field(None, description="Filter to a single session.")
    agent_id:   str | None = Field(None, description="Filter to a single agent.")
    from_ts_ns: int | None = Field(None, description="Start timestamp (nanoseconds).")
    to_ts_ns:   int | None = Field(None, description="End timestamp (nanoseconds).")
    limit: int = Field(5_000, ge=1, le=50_000, description="Maximum events to export.")


class PushResult(BaseModel):
    """Summary returned after a Splunk or Elasticsearch push."""

    sent: int = Field(..., description="Number of events successfully delivered.")
    batches: int = Field(..., description="Number of HTTP batch requests made.")
    errors: list[str] = Field(default_factory=list, description="Per-batch error messages, if any.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/export/jsonlines",
    summary="Stream events as JSON Lines (NDJSON)",
    response_class=StreamingResponse,
)
async def export_jsonlines(
    session_id:   str | None       = Query(None, description="Filter to a single session."),
    agent_id:     str | None       = Query(None, description="Filter to a single agent."),
    from_ts_ns:   int | None       = Query(None, description="Start timestamp (nanoseconds)."),
    to_ts_ns:     int | None       = Query(None, description="End timestamp (nanoseconds)."),
    event_types:  list[str] | None = Query(None, description="Restrict to these event types."),
    limit:        int                 = Query(10_000, ge=1, le=100_000,
                                             description="Maximum events to stream."),
    db:           AsyncSession        = Depends(get_session),
    org_ctx:      OrgContext          = Depends(require_api_key),
):
    """
    Stream audit events in JSON Lines format (one JSON object per line),
    scoped to the authenticated organisation.
    """
    generator = stream_jsonlines(
        db,
        org_id      = org_ctx.org_id,
        session_id  = session_id,
        agent_id    = agent_id,
        from_ts_ns  = from_ts_ns,
        to_ts_ns    = to_ts_ns,
        event_types = event_types,
        limit       = limit,
    )
    return StreamingResponse(
        generator,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": 'attachment; filename="aegivis-events.ndjson"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/export/splunk",
    response_model=PushResult,
    summary="Push events to Splunk HEC",
)
async def export_to_splunk(
    req: SplunkPushRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
) -> PushResult:
    """
    Fetch audit events and push them to a Splunk HTTP Event Collector,
    scoped to the authenticated organisation.
    """
    try:
        result = await push_to_splunk(
            db,
            org_id     = org_ctx.org_id,
            hec_url    = req.hec_url,
            hec_token  = req.hec_token,
            index      = req.index,
            source     = req.source,
            session_id = req.session_id,
            agent_id   = req.agent_id,
            from_ts_ns = req.from_ts_ns,
            to_ts_ns   = req.to_ts_ns,
            limit      = req.limit,
        )
    except Exception as exc:
        logger.error("Splunk push failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Splunk push error: {exc}")

    return PushResult(**result)


@router.post(
    "/export/elasticsearch",
    response_model=PushResult,
    summary="Push events to Elasticsearch",
)
async def export_to_elasticsearch(
    req: ElasticPushRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
) -> PushResult:
    """
    Fetch audit events and push them to Elasticsearch via the Bulk API,
    scoped to the authenticated organisation.
    """
    try:
        result = await push_to_elasticsearch(
            db,
            org_id     = org_ctx.org_id,
            es_url     = req.es_url,
            api_key    = req.api_key,
            index      = req.index,
            session_id = req.session_id,
            agent_id   = req.agent_id,
            from_ts_ns = req.from_ts_ns,
            to_ts_ns   = req.to_ts_ns,
            limit      = req.limit,
        )
    except Exception as exc:
        logger.error("Elasticsearch push failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Elasticsearch push error: {exc}")

    return PushResult(**result)


@router.get(
    "/export/audit-report",
    summary="Generate org-wide compliance audit report",
    response_description="Compliance report with control pass/fail mappings.",
)
async def export_audit_report(
    from_date: str | None = Query(
        None,
        description="Period start date (ISO 8601: YYYY-MM-DD). Defaults to 30 days ago.",
    ),
    to_date: str | None = Query(
        None,
        description="Period end date (ISO 8601: YYYY-MM-DD). Defaults to today.",
    ),
    framework: str = Query(
        "soc2",
        description="Compliance framework: owasp_asi_2026 | eu_ai_act | hipaa | soc2 | gdpr",
    ),
    agent_id: str | None = Query(None, description="Filter to a single agent."),
    session_id: str | None = Query(None, description="Filter to a single session."),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Generate an org-wide compliance audit report for a date range,
    scoped to the authenticated organisation.
    """
    now = datetime.now(timezone.utc)
    try:
        if from_date:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        else:
            # When scoping to a single session use a wide default range so the
            # caller does not need to know the session's exact date.
            from_dt = datetime(2020, 1, 1, tzinfo=timezone.utc) if session_id else now - timedelta(days=30)

        if to_date:
            to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, microsecond=999999,
                tzinfo=timezone.utc,
            )
        else:
            to_dt = now
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format — use YYYY-MM-DD: {exc}",
        )

    if from_dt >= to_dt:
        raise HTTPException(
            status_code=422,
            detail="from_date must be before to_date",
        )

    from_ts_ns = int(from_dt.timestamp() * 1_000_000_000)
    to_ts_ns = int(to_dt.timestamp() * 1_000_000_000)

    try:
        report = await build_audit_report(
            db,
            org_id=org_ctx.org_id,
            from_ts_ns=from_ts_ns,
            to_ts_ns=to_ts_ns,
            framework=framework,
            agent_id=agent_id,
            session_id=session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Audit report generation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Report generation error: {exc}"
        )

    return report
