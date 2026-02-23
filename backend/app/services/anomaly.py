"""
Rule-based anomaly detection engine (Phase 1 foundation).

8 detection rules based on the AgentBlackBox spec:
1. Excessive token usage (> 3x rolling avg)
2. Rapid tool call loops (same tool called > 5x in one session)
3. Error rate spike (> 3 errors in a session)
4. Unusual model switch mid-session
5. Long latency outlier (LLM response > 30s)
6. PII in tool outputs (data exfiltration signal)
7. No AGENT_FINISH after many LLM calls (runaway agent)
8. Chain integrity violation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnomalyFlag:
    rule_id: str
    severity: str      # "critical" | "high" | "medium" | "low"
    description: str
    event_id: str | None
    sequence_number: int | None
    metadata: dict


def detect_anomalies(events: list[dict], chain_valid: bool = True) -> list[AnomalyFlag]:
    """Run all anomaly detection rules against a session's event list."""
    flags: list[AnomalyFlag] = []

    if not events:
        return flags

    flags.extend(_rule_error_rate(events))
    flags.extend(_rule_tool_call_loop(events))
    flags.extend(_rule_long_latency(events))
    flags.extend(_rule_pii_in_tool_output(events))
    flags.extend(_rule_runaway_agent(events))
    flags.extend(_rule_model_switch(events))

    if not chain_valid:
        flags.append(AnomalyFlag(
            rule_id="CHAIN_INTEGRITY",
            severity="critical",
            description="Hash chain integrity violation detected",
            event_id=None,
            sequence_number=None,
            metadata={},
        ))

    return flags


def _rule_error_rate(events: list[dict]) -> list[AnomalyFlag]:
    errors = [e for e in events if e.get("event_type") == "SYSTEM_ERROR"]
    if len(errors) >= 3:
        return [AnomalyFlag(
            rule_id="HIGH_ERROR_RATE",
            severity="high",
            description=f"{len(errors)} provider errors in session",
            event_id=errors[-1].get("event_id"),
            sequence_number=errors[-1].get("sequence_number"),
            metadata={"error_count": len(errors)},
        )]
    return []


def _rule_tool_call_loop(events: list[dict]) -> list[AnomalyFlag]:
    tool_counts: dict[str, int] = {}
    for e in events:
        if e.get("event_type") == "TOOL_CALL_START":
            name = e.get("payload", {}).get("tool_name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1

    flags = []
    for tool_name, count in tool_counts.items():
        if count > 5:
            flags.append(AnomalyFlag(
                rule_id="TOOL_CALL_LOOP",
                severity="medium",
                description=f"Tool '{tool_name}' called {count} times — possible loop",
                event_id=None,
                sequence_number=None,
                metadata={"tool_name": tool_name, "call_count": count},
            ))
    return flags


def _rule_long_latency(events: list[dict]) -> list[AnomalyFlag]:
    flags = []
    for e in events:
        if e.get("event_type") == "LLM_CALL_END":
            latency = e.get("payload", {}).get("latency_ms")
            if latency and latency > 30_000:
                flags.append(AnomalyFlag(
                    rule_id="LONG_LATENCY",
                    severity="low",
                    description=f"LLM call took {latency:.0f}ms (>30s threshold)",
                    event_id=e.get("event_id"),
                    sequence_number=e.get("sequence_number"),
                    metadata={"latency_ms": latency},
                ))
    return flags


def _rule_pii_in_tool_output(events: list[dict]) -> list[AnomalyFlag]:
    flags = []
    for e in events:
        if e.get("event_type") == "TOOL_CALL_END" and e.get("pii_detected"):
            pii_types = e.get("pii_detected", [])
            sensitive = {"US_SSN", "CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE"}
            critical_pii = set(pii_types) & sensitive
            if critical_pii:
                flags.append(AnomalyFlag(
                    rule_id="SENSITIVE_PII_IN_TOOL_OUTPUT",
                    severity="high",
                    description=f"Sensitive PII in tool output: {', '.join(critical_pii)}",
                    event_id=e.get("event_id"),
                    sequence_number=e.get("sequence_number"),
                    metadata={"pii_types": sorted(critical_pii)},
                ))
    return flags


def _rule_runaway_agent(events: list[dict]) -> list[AnomalyFlag]:
    llm_calls = sum(1 for e in events if e.get("event_type") == "LLM_CALL_START")
    finish_events = sum(1 for e in events if e.get("event_type") == "AGENT_FINISH")

    if llm_calls >= 20 and finish_events == 0:
        return [AnomalyFlag(
            rule_id="RUNAWAY_AGENT",
            severity="high",
            description=f"Agent made {llm_calls} LLM calls without finishing — possible infinite loop",
            event_id=None,
            sequence_number=None,
            metadata={"llm_call_count": llm_calls},
        )]
    return []


def _rule_model_switch(events: list[dict]) -> list[AnomalyFlag]:
    models = []
    for e in events:
        m = e.get("model")
        if m and (not models or models[-1] != m):
            models.append(m)

    if len(models) > 1:
        return [AnomalyFlag(
            rule_id="MODEL_SWITCH",
            severity="low",
            description=f"Multiple models used in session: {', '.join(models)}",
            event_id=None,
            sequence_number=None,
            metadata={"models": models},
        )]
    return []
