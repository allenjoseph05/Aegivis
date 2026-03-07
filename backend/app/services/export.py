"""
SIEM Export Service.

Transforms audit events into SIEM-compatible payloads and delivers them to:

  * JSON Lines  — newline-delimited JSON for any SIEM or data lake.
  * Splunk HEC  — Splunk HTTP Event Collector (https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector).
  * Elasticsearch — Bulk API (https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html).

All exporters follow the same data shape:

  normalise_event(row) → dict
      A flat dict with every audit-event field plus derived helpers
      (timestamp_iso, risk_flags, security fields expanded to top-level).

The network calls use a shared ``httpx.AsyncClient`` passed by callers so
the event loop is not blocked and connections are reused across batch calls.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Maximum events per Splunk/Elasticsearch batch request.
_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Event normalisation
# ---------------------------------------------------------------------------

def normalise_event(row: Any) -> dict[str, Any]:
    """
    Convert a raw DB row / dict into a flat SIEM-friendly event.

    The ``security`` sub-object (stored as JSONB) is expanded to top-level
    ``security_*`` keys so SIEM platforms can index them without JSON parsing.
    """
    # Accept both SQLAlchemy Row objects and plain dicts
    if hasattr(row, "_mapping"):
        r: dict = dict(row._mapping)
    elif hasattr(row, "keys"):
        r = dict(row)
    else:
        r = row

    # Shallow-copy payload so we don't mutate the caller's dict when popping 'security'
    payload: dict = dict(r.get("payload") or {})
    security: dict = payload.pop("security", {}) if isinstance(payload, dict) else {}

    evt: dict[str, Any] = {
        # Core identifiers
        "event_id":           r.get("event_id"),
        "schema_version":     r.get("schema_version", "1.0"),
        "org_id":             r.get("org_id"),
        "session_id":         r.get("session_id"),
        "agent_id":           r.get("agent_id"),
        "run_id":             r.get("run_id"),
        "parent_run_id":      r.get("parent_run_id"),
        "event_type":         r.get("event_type"),
        "provider":           r.get("provider"),
        "model":              r.get("model"),
        "interception_layer": r.get("interception_layer"),
        # Timing
        "timestamp_ns":  r.get("timestamp_ns"),
        "timestamp_iso": _ns_to_iso(r.get("timestamp_ns")),
        # Hash chain
        "sequence_number": r.get("sequence_number"),
        "previous_hash":   r.get("previous_hash"),
        "current_hash":    r.get("current_hash"),
        # PII
        "pii_detected": r.get("pii_detected") or [],
        # Payload fields (top-level for easy indexing)
        **_flatten_payload(payload),
        # Security fields (top-level)
        "security_injection_score": security.get("injection_score"),
        "security_injection_label": security.get("injection_label"),
        "security_credential_detected": security.get("credential_detected", False),
        "security_rce_detected":  security.get("rce_detected", False),
        "security_ssrf_detected": security.get("ssrf_detected", False),
        "security_crescendo_detected": (security.get("crescendo") or {}).get("detected", False),
        "security_crescendo_drift": (security.get("crescendo") or {}).get("drift_score"),
        "security_output_detected": (security.get("output") or {}).get("detected", False),
    }
    return evt


def _ns_to_iso(ns: int | None) -> str | None:
    if not ns:
        return None
    import datetime
    try:
        ts = ns / 1_000_000_000
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
    except Exception:
        return None


def _flatten_payload(payload: dict) -> dict[str, Any]:
    """Extract the most useful payload fields as top-level keys."""
    out: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out
    # LLM_CALL_END latency + token usage
    out["latency_ms"]         = payload.get("latency_ms")
    out["total_tokens"]       = payload.get("total_tokens")
    out["prompt_tokens"]      = payload.get("prompt_tokens")
    out["completion_tokens"]  = payload.get("completion_tokens")
    out["finish_reason"]      = payload.get("finish_reason")
    out["http_status"]        = payload.get("http_status")
    # Tool calls
    out["tool_name"]          = payload.get("tool_name")
    out["tool_call_id"]       = payload.get("tool_call_id")
    # Errors
    out["error_message"]      = payload.get("error_message")
    out["error_code"]         = payload.get("error_code")
    # Agent finish
    out["total_llm_calls"]    = payload.get("total_llm_calls")
    out["total_tool_calls"]   = payload.get("total_tool_calls")
    out["session_duration_ms"]= payload.get("session_duration_ms")
    return out


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

async def _fetch_events(
    db: AsyncSession,
    *,
    org_id: str,
    session_id: str | None,
    agent_id: str | None,
    from_ts_ns: int | None,
    to_ts_ns: int | None,
    event_types: list[str] | None,
    limit: int,
    offset: int,
) -> list[Any]:
    """Fetch raw event rows matching the given filters, scoped to an org."""
    filters = ["org_id = :org_id"]
    params: dict[str, Any] = {"limit": limit, "offset": offset, "org_id": org_id}

    if session_id:
        filters.append("session_id = :session_id")
        params["session_id"] = session_id
    if agent_id:
        filters.append("agent_id = :agent_id")
        params["agent_id"] = agent_id
    if from_ts_ns:
        filters.append("timestamp_ns >= :from_ts_ns")
        params["from_ts_ns"] = from_ts_ns
    if to_ts_ns:
        filters.append("timestamp_ns <= :to_ts_ns")
        params["to_ts_ns"] = to_ts_ns
    if event_types:
        placeholders = ", ".join(f":et{i}" for i in range(len(event_types)))
        filters.append(f"event_type IN ({placeholders})")
        for i, et in enumerate(event_types):
            params[f"et{i}"] = et

    where = f"WHERE {' AND '.join(filters)}"
    sql = text(f"""
        SELECT event_id, schema_version, org_id, session_id, agent_id,
               provider, model, interception_layer, run_id, parent_run_id,
               event_type, payload, pii_detected,
               timestamp_ns, sequence_number, previous_hash, current_hash
        FROM audit_events
        {where}
        ORDER BY timestamp_ns ASC
        LIMIT :limit OFFSET :offset
    """)
    result = await db.execute(sql, params)
    return result.fetchall()


# ---------------------------------------------------------------------------
# JSON Lines exporter
# ---------------------------------------------------------------------------

async def stream_jsonlines(
    db: AsyncSession,
    *,
    org_id: str,
    session_id: str | None = None,
    agent_id:   str | None = None,
    from_ts_ns: int | None = None,
    to_ts_ns:   int | None = None,
    event_types: list[str] | None = None,
    limit: int = 10_000,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator that yields newline-delimited JSON bytes.

    Each line is one audit event serialised with ``normalise_event()``.
    The caller is responsible for streaming these bytes to the HTTP client.

    Usage in a FastAPI route:
        return StreamingResponse(
            stream_jsonlines(db, session_id=sid),
            media_type="application/x-ndjson",
        )
    """
    offset = 0
    batch_size = min(limit, 1000)
    remaining = limit

    while remaining > 0:
        fetch_size = min(batch_size, remaining)
        rows = await _fetch_events(
            db,
            org_id       = org_id,
            session_id   = session_id,
            agent_id     = agent_id,
            from_ts_ns   = from_ts_ns,
            to_ts_ns     = to_ts_ns,
            event_types  = event_types,
            limit        = fetch_size,
            offset       = offset,
        )
        if not rows:
            break

        for row in rows:
            line = json.dumps(normalise_event(row), default=str) + "\n"
            yield line.encode("utf-8")

        offset    += len(rows)
        remaining -= len(rows)

        if len(rows) < fetch_size:
            break  # no more rows


# ---------------------------------------------------------------------------
# Splunk HEC exporter
# ---------------------------------------------------------------------------

def _to_splunk_event(evt: dict[str, Any], index: str, source: str) -> dict[str, Any]:
    """Wrap a normalised event in the Splunk HEC envelope."""
    ts_ns = evt.get("timestamp_ns")
    return {
        "time":       ts_ns / 1_000_000_000 if ts_ns else None,
        "index":      index,
        "source":     source,
        "sourcetype": "aegivis",
        "host":       evt.get("agent_id", "unknown"),
        "event":      evt,
    }


async def push_to_splunk(
    db: AsyncSession,
    *,
    org_id: str,
    hec_url: str,
    hec_token: str,
    index: str = "aegivis",
    session_id: str | None = None,
    agent_id:   str | None = None,
    from_ts_ns: int | None = None,
    to_ts_ns:   int | None = None,
    limit: int = 5_000,
    source: str = "aegivis-proxy",
) -> dict[str, Any]:
    """
    Fetch events and push them to a Splunk HEC endpoint in batches.

    Returns a summary dict: {"sent": N, "batches": N, "errors": [...]}.
    """
    rows = await _fetch_events(
        db,
        org_id      = org_id,
        session_id  = session_id,
        agent_id    = agent_id,
        from_ts_ns  = from_ts_ns,
        to_ts_ns    = to_ts_ns,
        event_types = None,
        limit       = limit,
        offset      = 0,
    )

    total_sent = 0
    batches    = 0
    errors: list[str] = []

    async with httpx.AsyncClient(
        timeout = httpx.Timeout(30.0),
        headers = {
            "Authorization": f"Splunk {hec_token}",
            "Content-Type":  "application/json",
        },
    ) as client:
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            # Splunk HEC accepts a newline-separated stream of JSON objects
            body = "\n".join(
                json.dumps(_to_splunk_event(normalise_event(r), index, source), default=str)
                for r in batch
            )
            try:
                resp = await client.post(hec_url, content=body.encode())
                if resp.status_code not in (200, 201):
                    errors.append(
                        f"Batch {batches + 1}: HTTP {resp.status_code} — {resp.text[:200]}"
                    )
                else:
                    total_sent += len(batch)
                batches += 1
            except Exception as exc:
                errors.append(f"Batch {batches + 1}: {exc}")
                logger.warning("Splunk push error (batch %d): %s", batches, exc)

    return {"sent": total_sent, "batches": batches, "errors": errors}


# ---------------------------------------------------------------------------
# Elasticsearch exporter
# ---------------------------------------------------------------------------

def _elastic_bulk_body(events: list[dict[str, Any]], index: str) -> bytes:
    """
    Build an Elasticsearch bulk request body.

    Format (two lines per document):
        {"index": {"_index": "<index>", "_id": "<event_id>"}}
        {<event document>}
    """
    lines: list[str] = []
    for evt in events:
        doc_id = evt.get("event_id", "")
        meta   = json.dumps({"index": {"_index": index, "_id": doc_id}})
        doc    = json.dumps(evt, default=str)
        lines.append(meta)
        lines.append(doc)
    # Bulk API requires a trailing newline
    return ("\n".join(lines) + "\n").encode("utf-8")


async def push_to_elasticsearch(
    db: AsyncSession,
    *,
    org_id: str,
    es_url: str,
    api_key: str | None = None,
    index: str = "aegivis",
    session_id: str | None = None,
    agent_id:   str | None = None,
    from_ts_ns: int | None = None,
    to_ts_ns:   int | None = None,
    limit: int = 5_000,
) -> dict[str, Any]:
    """
    Fetch events and push them to Elasticsearch via the Bulk API.

    Returns a summary dict: {"sent": N, "batches": N, "errors": [...]}.
    """
    rows = await _fetch_events(
        db,
        org_id      = org_id,
        session_id  = session_id,
        agent_id    = agent_id,
        from_ts_ns  = from_ts_ns,
        to_ts_ns    = to_ts_ns,
        event_types = None,
        limit       = limit,
        offset      = 0,
    )

    total_sent = 0
    batches    = 0
    errors: list[str] = []

    bulk_url = f"{es_url.rstrip('/')}/_bulk"
    headers: dict[str, str] = {"Content-Type": "application/x-ndjson"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), headers=headers) as client:
        for i in range(0, len(rows), _BATCH_SIZE):
            batch       = rows[i : i + _BATCH_SIZE]
            norm_events = [normalise_event(r) for r in batch]
            body        = _elastic_bulk_body(norm_events, index)
            try:
                resp = await client.post(bulk_url, content=body)
                if resp.status_code not in (200, 201):
                    errors.append(
                        f"Batch {batches + 1}: HTTP {resp.status_code} — {resp.text[:200]}"
                    )
                else:
                    resp_data = resp.json()
                    if resp_data.get("errors"):
                        failed = [
                            item for item in resp_data.get("items", [])
                            if "error" in (item.get("index") or {})
                        ]
                        if failed:
                            errors.append(
                                f"Batch {batches + 1}: {len(failed)} index errors"
                            )
                    total_sent += len(batch)
                batches += 1
            except Exception as exc:
                errors.append(f"Batch {batches + 1}: {exc}")
                logger.warning("Elasticsearch push error (batch %d): %s", batches, exc)

    return {"sent": total_sent, "batches": batches, "errors": errors}
