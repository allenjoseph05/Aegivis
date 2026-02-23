"""
Forensic replay service: reconstructs a human-readable incident report
from a session's raw event sequence.

Output format designed for compliance investigators:
- Chronological timeline of all agent decisions
- Data access map (what external systems were called)
- Risk flags (anomalies, errors, PII exposure)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EVENT_TYPE_LABELS = {
    "LLM_CALL_START": "LLM Request",
    "LLM_CALL_END": "LLM Response",
    "TOOL_CALL_START": "Tool Invoked",
    "TOOL_CALL_END": "Tool Result",
    "AGENT_FINISH": "Agent Finished",
    "SYSTEM_ERROR": "Error",
    "CHECKPOINT": "Integrity Checkpoint",
}

EVENT_TYPE_SEVERITY = {
    "SYSTEM_ERROR": "error",
    "TOOL_CALL_START": "info",
    "TOOL_CALL_END": "info",
    "LLM_CALL_START": "debug",
    "LLM_CALL_END": "debug",
    "AGENT_FINISH": "success",
    "CHECKPOINT": "debug",
}


@dataclass
class TimelineEntry:
    sequence_number: int
    event_id: str
    event_type: str
    label: str
    severity: str
    timestamp_ns: int
    summary: str
    details: dict
    pii_detected: list[str]
    run_id: str
    parent_run_id: str | None


@dataclass
class DataAccessEntry:
    tool_name: str
    call_count: int
    first_seen_ns: int
    last_seen_ns: int
    sample_input: Any


@dataclass
class RiskFlag:
    severity: str  # "critical" | "high" | "medium" | "low"
    flag_type: str
    description: str
    sequence_number: int | None
    event_id: str | None


@dataclass
class ForensicReport:
    session_id: str
    org_id: str
    agent_id: str
    provider: str
    model: str
    started_at_ns: int
    ended_at_ns: int
    duration_ms: float
    total_events: int
    llm_call_count: int
    tool_call_count: int
    error_count: int
    pii_exposure_count: int
    total_tokens: int
    timeline: list[TimelineEntry]
    data_access_map: list[DataAccessEntry]
    risk_flags: list[RiskFlag]
    chain_valid: bool


def build_forensic_report(
    session_id: str,
    events: list[dict],
    chain_valid: bool = True,
) -> ForensicReport:
    """
    Build a ForensicReport from an ordered list of event dicts.
    """
    if not events:
        return ForensicReport(
            session_id=session_id,
            org_id="unknown",
            agent_id="unknown",
            provider="unknown",
            model="unknown",
            started_at_ns=0,
            ended_at_ns=0,
            duration_ms=0,
            total_events=0,
            llm_call_count=0,
            tool_call_count=0,
            error_count=0,
            pii_exposure_count=0,
            total_tokens=0,
            timeline=[],
            data_access_map=[],
            risk_flags=[],
            chain_valid=chain_valid,
        )

    org_id = events[0].get("org_id", "unknown")
    agent_id = events[0].get("agent_id", "unknown")
    provider = events[0].get("provider", "unknown")
    model = events[0].get("model", "unknown")
    started_at_ns = events[0].get("timestamp_ns", 0)
    ended_at_ns = events[-1].get("timestamp_ns", 0)
    duration_ms = (ended_at_ns - started_at_ns) / 1_000_000

    timeline: list[TimelineEntry] = []
    data_access: dict[str, DataAccessEntry] = {}
    risk_flags: list[RiskFlag] = []
    llm_calls = 0
    tool_calls = 0
    errors = 0
    pii_exposures = 0
    total_tokens = 0

    for event in events:
        et = event.get("event_type", "")
        payload = event.get("payload", {})
        pii = event.get("pii_detected", [])
        seq = event.get("sequence_number", 0)

        if pii:
            pii_exposures += 1

        # Build summary
        summary = _summarize_event(et, payload, event.get("provider", provider))

        entry = TimelineEntry(
            sequence_number=seq,
            event_id=event.get("event_id", ""),
            event_type=et,
            label=EVENT_TYPE_LABELS.get(et, et),
            severity=EVENT_TYPE_SEVERITY.get(et, "info"),
            timestamp_ns=event.get("timestamp_ns", 0),
            summary=summary,
            details=_extract_details(et, payload),
            pii_detected=pii,
            run_id=event.get("run_id", ""),
            parent_run_id=event.get("parent_run_id"),
        )
        timeline.append(entry)

        # Count stats
        if et == "LLM_CALL_START":
            llm_calls += 1
        elif et == "TOOL_CALL_START":
            tool_calls += 1
            tool_name = payload.get("tool_name", "unknown")
            if tool_name not in data_access:
                data_access[tool_name] = DataAccessEntry(
                    tool_name=tool_name,
                    call_count=0,
                    first_seen_ns=event.get("timestamp_ns", 0),
                    last_seen_ns=0,
                    sample_input=payload.get("tool_input"),
                )
            data_access[tool_name].call_count += 1
            data_access[tool_name].last_seen_ns = event.get("timestamp_ns", 0)
        elif et == "SYSTEM_ERROR":
            errors += 1
            risk_flags.append(RiskFlag(
                severity="high",
                flag_type="PROVIDER_ERROR",
                description=f"LLM provider error: {payload.get('error_message', 'unknown')}",
                sequence_number=seq,
                event_id=event.get("event_id"),
            ))
        elif et == "LLM_CALL_END":
            usage = payload.get("token_usage") or {}
            total_tokens += usage.get("total_tokens", 0) or 0

        # PII risk flag
        if pii and et in ("LLM_CALL_START", "TOOL_CALL_END"):
            risk_flags.append(RiskFlag(
                severity="medium",
                flag_type="PII_DETECTED",
                description=f"PII detected in {et}: {', '.join(pii)}",
                sequence_number=seq,
                event_id=event.get("event_id"),
            ))

    if not chain_valid:
        risk_flags.append(RiskFlag(
            severity="critical",
            flag_type="CHAIN_INTEGRITY_VIOLATION",
            description="Hash chain integrity check failed - audit trail may have been tampered",
            sequence_number=None,
            event_id=None,
        ))

    return ForensicReport(
        session_id=session_id,
        org_id=org_id,
        agent_id=agent_id,
        provider=provider,
        model=model,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        duration_ms=duration_ms,
        total_events=len(events),
        llm_call_count=llm_calls,
        tool_call_count=tool_calls,
        error_count=errors,
        pii_exposure_count=pii_exposures,
        total_tokens=total_tokens,
        timeline=timeline,
        data_access_map=list(data_access.values()),
        risk_flags=sorted(risk_flags, key=lambda f: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(f.severity, 4)),
        chain_valid=chain_valid,
    )


def _summarize_event(event_type: str, payload: dict, provider: str) -> str:
    if event_type == "LLM_CALL_START":
        msgs = payload.get("messages", [])
        n_tools = len(payload.get("tools_available", []))
        tool_str = f", {n_tools} tools available" if n_tools else ""
        last_user = next((m.get("content", "")[:80] for m in reversed(msgs) if m.get("role") == "user"), "")
        return f"Request to {provider}: {len(msgs)} messages{tool_str}. Last user: '{last_user}...'"
    elif event_type == "LLM_CALL_END":
        text = payload.get("response_text") or ""
        tcs = payload.get("tool_calls", [])
        if tcs:
            names = ", ".join(tc.get("name", "?") for tc in tcs)
            return f"Decided to call: {names}"
        return f"Response: '{text[:100]}...' (finish: {payload.get('finish_reason')})"
    elif event_type == "TOOL_CALL_START":
        return f"Calling tool: {payload.get('tool_name', '?')}"
    elif event_type == "TOOL_CALL_END":
        out = payload.get("tool_output_masked") or ""
        return f"Tool {payload.get('tool_name', '?')} returned: '{out[:80]}...'"
    elif event_type == "AGENT_FINISH":
        out = payload.get("final_output") or ""
        return f"Agent finished: '{out[:120]}...'"
    elif event_type == "SYSTEM_ERROR":
        return f"Error {payload.get('http_status', '')}: {payload.get('error_message', '')[:100]}"
    elif event_type == "CHECKPOINT":
        return f"Merkle checkpoint: {payload.get('events_covered', 0)} events verified"
    return event_type


def _extract_details(event_type: str, payload: dict) -> dict:
    if event_type == "LLM_CALL_START":
        return {
            "message_count": len(payload.get("messages", [])),
            "tool_count": len(payload.get("tools_available", [])),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
        }
    elif event_type == "LLM_CALL_END":
        usage = payload.get("token_usage") or {}
        return {
            "finish_reason": payload.get("finish_reason"),
            "latency_ms": payload.get("latency_ms"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "tool_calls": [tc.get("name") for tc in (payload.get("tool_calls") or [])],
        }
    elif event_type in ("TOOL_CALL_START", "TOOL_CALL_END"):
        return {
            "tool_name": payload.get("tool_name"),
            "tool_call_id": payload.get("tool_call_id"),
        }
    return payload
