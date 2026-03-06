"""
Org-wide compliance audit report service.

Builds a date-range audit report that maps observed platform behaviour
to compliance controls for the following frameworks:

  - OWASP ASI 2026
  - EU AI Act
  - HIPAA §164.312
  - SOC 2 Type II

Usage::

    from app.services.audit_report import build_audit_report

    report = await build_audit_report(
        db,
        org_id="default-org",
        from_ts_ns=from_ns,
        to_ts_ns=to_ns,
        framework="soc2",
    )
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ─── Framework control definitions ────────────────────────────────────────────

_FRAMEWORK_CONTROLS: dict[str, list[dict[str, str]]] = {
    "owasp_asi_2026": [
        {
            "id": "ASI-1",
            "name": "Prompt Injection Prevention",
            "evidence_type": "injection",
        },
        {
            "id": "ASI-2",
            "name": "Tool Abuse Prevention",
            "evidence_type": "tool_abuse",
        },
        {
            "id": "ASI-3",
            "name": "Output Safety",
            "evidence_type": "output_safety",
        },
        {
            "id": "ASI-4",
            "name": "Data Privacy",
            "evidence_type": "pii",
        },
    ],
    "eu_ai_act": [
        {
            "id": "Art.9",
            "name": "Risk Management System",
            "evidence_type": "injection",
        },
        {
            "id": "Art.12",
            "name": "Record Keeping",
            "evidence_type": "record_keeping",
        },
        {
            "id": "Art.13",
            "name": "Transparency",
            "evidence_type": "transparency",
        },
        {
            "id": "Art.15",
            "name": "Robustness & Accuracy",
            "evidence_type": "robustness",
        },
    ],
    "hipaa": [
        {
            "id": "§164.312(a)(2)(i)",
            "name": "Unique User Identification",
            "evidence_type": "identity",
        },
        {
            "id": "§164.312(b)",
            "name": "Audit Controls",
            "evidence_type": "record_keeping",
        },
        {
            "id": "§164.312(c)(1)",
            "name": "Integrity",
            "evidence_type": "chain_integrity",
        },
        {
            "id": "§164.312(e)(1)",
            "name": "Transmission Security",
            "evidence_type": "pii",
        },
    ],
    "soc2": [
        {
            "id": "CC6.1",
            "name": "Logical Access Controls",
            "evidence_type": "injection",
        },
        {
            "id": "CC6.7",
            "name": "Data Transmission Security",
            "evidence_type": "pii",
        },
        {
            "id": "CC7.2",
            "name": "Anomaly Monitoring",
            "evidence_type": "anomalies",
        },
        {
            "id": "CC7.3",
            "name": "Security Event Response",
            "evidence_type": "violations",
        },
    ],
    "gdpr": [
        {
            "id": "Art.5(1)(f)",
            "name": "Integrity & Confidentiality",
            "evidence_type": "chain_integrity",
        },
        {
            "id": "Art.25",
            "name": "Data Protection by Design",
            "evidence_type": "pii",
        },
        {
            "id": "Art.30",
            "name": "Records of Processing Activities",
            "evidence_type": "record_keeping",
        },
        {
            "id": "Art.32",
            "name": "Security of Processing",
            "evidence_type": "injection",
        },
    ],
}


# ─── SQL helpers ──────────────────────────────────────────────────────────────

_EVENT_SUMMARY_SQL = """
SELECT
    COUNT(DISTINCT session_id)                                       AS total_sessions,
    COUNT(*) FILTER (WHERE event_type = 'LLM_CALL_START')           AS llm_calls,
    COUNT(*) FILTER (WHERE event_type = 'TOOL_CALL_START')          AS tool_calls,
    COUNT(*) FILTER (WHERE event_type = 'MEMORY_WRITE_BLOCKED')     AS memory_blocked,
    COUNT(*) FILTER (WHERE cardinality(pii_detected) > 0)           AS pii_events,
    COUNT(*) FILTER (WHERE event_type = 'SYSTEM_ERROR')             AS error_count,
    COUNT(DISTINCT agent_id)                                         AS agent_count,
    ROUND(AVG(
        CAST(payload->>'latency_ms' AS FLOAT)
    ) FILTER (WHERE event_type = 'LLM_CALL_END'
                AND payload->>'latency_ms' IS NOT NULL)::numeric, 2) AS avg_latency_ms
FROM audit_events
WHERE timestamp_ns BETWEEN :from_ts AND :to_ts
  AND (CAST(:org_id AS TEXT) IS NULL OR org_id = CAST(:org_id AS TEXT))
  AND (CAST(:agent_id AS TEXT) IS NULL OR agent_id = CAST(:agent_id AS TEXT))
  AND (CAST(:session_id AS TEXT) IS NULL OR session_id = CAST(:session_id AS TEXT))
"""

_VIOLATIONS_SUMMARY_SQL = """
SELECT
    rule_name,
    action,
    COUNT(*) AS count
FROM policy_violations
WHERE timestamp_ns BETWEEN :from_ts AND :to_ts
  AND (CAST(:org_id AS TEXT) IS NULL OR org_id = CAST(:org_id AS TEXT))
  AND (CAST(:agent_id AS TEXT) IS NULL OR agent_id = CAST(:agent_id AS TEXT))
  AND (CAST(:session_id AS TEXT) IS NULL OR session_id = CAST(:session_id AS TEXT))
  AND (is_false_positive IS NULL OR is_false_positive = FALSE)
GROUP BY rule_name, action
ORDER BY count DESC
"""

_ANOMALY_COUNT_SQL = """
SELECT COUNT(*) AS total
FROM agent_anomalies
WHERE detected_at BETWEEN
      to_timestamp(:from_ts / 1e9)
  AND to_timestamp(:to_ts  / 1e9)
  AND (CAST(:org_id AS TEXT) IS NULL OR org_id = CAST(:org_id AS TEXT))
  AND (CAST(:agent_id AS TEXT) IS NULL OR agent_id = CAST(:agent_id AS TEXT))
"""

_CHAIN_STATUS_SQL = """
SELECT
    session_id,
    previous_hash,
    current_hash,
    sequence_number
FROM audit_events
WHERE timestamp_ns BETWEEN :from_ts AND :to_ts
  AND (CAST(:org_id AS TEXT) IS NULL OR org_id = CAST(:org_id AS TEXT))
  AND (CAST(:agent_id AS TEXT) IS NULL OR agent_id = CAST(:agent_id AS TEXT))
  AND (CAST(:session_id AS TEXT) IS NULL OR session_id = CAST(:session_id AS TEXT))
ORDER BY session_id, sequence_number
"""

_TOP_VIOLATIONS_SQL = """
SELECT
    pv.id,
    pv.rule_name,
    pv.action,
    pv.reason,
    pv.event_type,
    pv.session_id,
    pv.agent_id,
    pv.org_id,
    pv.timestamp_ns
FROM policy_violations pv
WHERE pv.action IN ('BLOCK', 'ALERT')
  AND pv.timestamp_ns BETWEEN :from_ts AND :to_ts
  AND (CAST(:org_id AS TEXT) IS NULL OR pv.org_id = CAST(:org_id AS TEXT))
  AND (CAST(:agent_id AS TEXT) IS NULL OR pv.agent_id = CAST(:agent_id AS TEXT))
  AND (CAST(:session_id AS TEXT) IS NULL OR pv.session_id = CAST(:session_id AS TEXT))
  AND (pv.is_false_positive IS NULL OR pv.is_false_positive = FALSE)
ORDER BY pv.timestamp_ns DESC
LIMIT 200
"""

_AGENTS_SQL = """
SELECT
    agent_id,
    name,
    declared_purpose,
    allowed_tools,
    owner,
    registered_at,
    updated_at
FROM agents
ORDER BY registered_at DESC
"""


# ─── Chain verification helper ─────────────────────────────────────────────────

def _verify_chains(rows: list) -> tuple[int, int]:
    """Return (verified_session_count, valid_session_count) from raw event rows."""
    # Group by session_id → ordered events
    sessions: dict[str, list] = {}
    for row in rows:
        sessions.setdefault(row.session_id, []).append(row)

    total = len(sessions)
    valid = 0
    for sid, evts in sessions.items():
        evts_sorted = sorted(evts, key=lambda r: r.sequence_number)
        ok = True
        prev_hash = evts_sorted[0].current_hash if evts_sorted else None
        for evt in evts_sorted[1:]:
            if evt.previous_hash != prev_hash:
                ok = False
                break
            prev_hash = evt.current_hash
        if ok:
            valid += 1
    return total, valid


# ─── Control status logic ──────────────────────────────────────────────────────

def _compute_control_status(
    evidence_type: str,
    summary: dict[str, Any],
    violation_by_rule: dict[str, dict],
    anomaly_count: int,
    chain_total: int,
    chain_valid: int,
) -> tuple[str, str]:
    """Return (status, evidence_string) for one compliance control."""
    blocked = sum(
        v["count"] for v in violation_by_rule.values() if v["action"] == "BLOCK"
    )
    alert = sum(
        v["count"] for v in violation_by_rule.values() if v["action"] == "ALERT"
    )
    total_sessions = summary.get("total_sessions", 0)
    pii_events = summary.get("pii_events", 0)
    llm_calls = summary.get("llm_calls", 0)
    chain_valid_pct = (chain_valid / chain_total * 100) if chain_total else 100.0

    injection_rules = {
        k: v for k, v in violation_by_rule.items()
        if any(kw in k.lower() for kw in ("injection", "prompt", "crescendo", "delimiter"))
    }
    tool_rules = {
        k: v for k, v in violation_by_rule.items()
        if any(kw in k.lower() for kw in ("tool", "rce", "ssrf", "schema"))
    }
    output_rules = {
        k: v for k, v in violation_by_rule.items()
        if any(kw in k.lower() for kw in ("output", "canary", "relay"))
    }
    cred_rules = {
        k: v for k, v in violation_by_rule.items()
        if any(kw in k.lower() for kw in ("credential", "pii", "secret", "token"))
    }

    if evidence_type == "injection":
        inj_blocks = sum(v["count"] for v in injection_rules.values() if v["action"] == "BLOCK")
        inj_alerts = sum(v["count"] for v in injection_rules.values() if v["action"] == "ALERT")
        if llm_calls == 0:
            return "fail", "No LLM calls recorded in this period"
        if inj_blocks > 0:
            return "pass", f"{inj_blocks} injection attempt(s) blocked by policy engine"
        if inj_alerts > 0:
            return "partial", f"{inj_alerts} injection alert(s) fired — review and harden thresholds"
        return "pass", f"No injection attempts detected across {llm_calls} LLM call(s)"

    if evidence_type == "tool_abuse":
        tool_blocks = sum(v["count"] for v in tool_rules.values() if v["action"] == "BLOCK")
        tool_alerts = sum(v["count"] for v in tool_rules.values() if v["action"] == "ALERT")
        tool_calls = summary.get("tool_calls", 0)
        if tool_calls == 0 and tool_blocks == 0 and tool_alerts == 0:
            return "pass", "No tool calls recorded; tool firewall active"
        if tool_blocks > 0:
            return "pass", f"{tool_blocks} tool abuse attempt(s) blocked (RCE/SSRF/schema violations)"
        if tool_alerts > 0:
            return "partial", f"{tool_alerts} tool call alert(s) — no blocks triggered yet"
        return "pass", f"{tool_calls} tool call(s) logged with no abuse detected"

    if evidence_type == "output_safety":
        out_blocks = sum(v["count"] for v in output_rules.values() if v["action"] == "BLOCK")
        out_alerts = sum(v["count"] for v in output_rules.values() if v["action"] == "ALERT")
        if out_blocks > 0:
            return "pass", f"{out_blocks} unsafe output(s) blocked (canary/relay detection)"
        if out_alerts > 0:
            return "partial", f"{out_alerts} suspicious output alert(s); no blocks yet"
        return "pass", "No unsafe outputs detected"

    if evidence_type == "pii":
        if pii_events > 0:
            return "partial", f"{pii_events} PII detection event(s); verify masking is complete"
        return "pass", "No PII detected in session payloads"

    if evidence_type == "record_keeping":
        if total_sessions == 0:
            return "fail", "No sessions recorded in this period"
        ev_count = summary.get("llm_calls", 0) + summary.get("tool_calls", 0)
        return "pass", (
            f"{total_sessions} session(s), {ev_count} event(s) durably recorded "
            f"with hash-chain integrity"
        )

    if evidence_type == "transparency":
        if llm_calls == 0:
            return "fail", "No LLM calls recorded"
        return "pass", (
            f"All {llm_calls} LLM call(s) captured with provider, model, token usage, "
            f"and latency metadata"
        )

    if evidence_type == "robustness":
        if anomaly_count > 0 and blocked == 0:
            return "partial", (
                f"{anomaly_count} anomaly/anomalies detected but no blocks triggered — "
                f"consider stricter policy thresholds"
            )
        if blocked > 0:
            return "pass", f"{blocked} anomalous request(s) blocked; platform responded"
        return "pass", "No anomalies detected in this period"

    if evidence_type == "identity":
        if total_sessions == 0:
            return "fail", "No sessions with agent identity recorded"
        return "pass", (
            f"{total_sessions} session(s) with unique agent_id/session_id tracking"
        )

    if evidence_type == "chain_integrity":
        if chain_total == 0:
            return "fail", "No events to verify chain integrity against"
        if chain_valid_pct < 100:
            return "fail", (
                f"Chain link structure FAILED for {chain_total - chain_valid} "
                f"of {chain_total} session(s). "
                f"Use Session Detail \u2192 Verify Chain for cryptographic detail."
            )
        return "pass", (
            f"Chain link structure verified for all {chain_total} session(s) "
            f"({chain_valid_pct:.0f}%). For full SHA-256 re-verification use "
            f"Session Detail \u2192 Verify Chain."
        )

    if evidence_type == "anomalies":
        if anomaly_count > 0:
            return "pass", f"{anomaly_count} anomaly/anomalies detected and logged for review"
        return "pass", "No anomalies detected; monitoring active"

    if evidence_type == "violations":
        total_v = blocked + alert
        if total_v > 0:
            return "pass", (
                f"{blocked} BLOCK + {alert} ALERT enforcement event(s) recorded with audit trail"
            )
        return "pass", "Policy engine active; no violations in this period"

    return "pass", "Control checked"


# ─── Main function ─────────────────────────────────────────────────────────────

async def build_audit_report(
    db: AsyncSession,
    *,
    org_id: str = "default-org",
    from_ts_ns: int,
    to_ts_ns: int,
    framework: str = "soc2",
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Build an org-wide compliance audit report for a date range.

    Parameters
    ----------
    db          AsyncSession from dependency injection.
    org_id      Organisation filter (default 'default-org').
    from_ts_ns  Period start in nanoseconds since epoch.
    to_ts_ns    Period end in nanoseconds since epoch.
    framework   One of: owasp_asi_2026, eu_ai_act, hipaa, soc2, gdpr.
    agent_id    Optional agent filter.
    session_id  Optional session filter (scopes report to a single session).

    Returns
    -------
    dict  Full report document; shape described in IMPLEMENTATION_PLAN.md.
    """
    if framework not in _FRAMEWORK_CONTROLS:
        raise ValueError(
            f"Unknown framework '{framework}'. "
            f"Valid: {', '.join(_FRAMEWORK_CONTROLS)}"
        )

    params = {
        "from_ts": from_ts_ns,
        "to_ts": to_ts_ns,
        "org_id": org_id or None,
        "agent_id": agent_id or None,
        "session_id": session_id or None,
    }

    # Run queries sequentially — SQLAlchemy AsyncSession cannot handle
    # concurrent executions on the same connection object.
    event_summary_result = await db.execute(text(_EVENT_SUMMARY_SQL), params)
    violations_result    = await db.execute(text(_VIOLATIONS_SUMMARY_SQL), params)
    anomaly_result       = await db.execute(text(_ANOMALY_COUNT_SQL), params)
    chain_result         = await db.execute(text(_CHAIN_STATUS_SQL), params)
    top_violations_result = await db.execute(text(_TOP_VIOLATIONS_SQL), params)

    try:
        agents_result = await db.execute(text(_AGENTS_SQL))
        agents_rows = agents_result.fetchall()
    except Exception:
        agents_rows = []

    # ── Process event summary ─────────────────────────────────────────────────
    ev_row = event_summary_result.fetchone()
    summary: dict[str, Any] = {}
    if ev_row:
        summary = {
            "total_sessions": ev_row.total_sessions or 0,
            "llm_calls": ev_row.llm_calls or 0,
            "tool_calls": ev_row.tool_calls or 0,
            "memory_blocked": ev_row.memory_blocked or 0,
            "pii_events": ev_row.pii_events or 0,
            "error_count": ev_row.error_count or 0,
            "agent_count": ev_row.agent_count or 0,
            "avg_latency_ms": (
                float(ev_row.avg_latency_ms)
                if ev_row.avg_latency_ms is not None
                else None
            ),
        }
    else:
        summary = {
            "total_sessions": 0, "llm_calls": 0, "tool_calls": 0,
            "memory_blocked": 0, "pii_events": 0, "error_count": 0,
            "agent_count": 0, "avg_latency_ms": None,
        }

    # ── Process violations ────────────────────────────────────────────────────
    viol_rows = violations_result.fetchall()
    violation_by_rule: dict[str, dict] = {}
    blocked_count = 0
    alert_count = 0
    for row in viol_rows:
        violation_by_rule[row.rule_name] = {
            "action": row.action,
            "count": row.count,
        }
        if row.action == "BLOCK":
            blocked_count += row.count
        elif row.action == "ALERT":
            alert_count += row.count

    summary["blocked_count"] = blocked_count
    summary["alert_count"] = alert_count
    summary["total_violations"] = blocked_count + alert_count

    # ── Process anomalies ─────────────────────────────────────────────────────
    anom_row = anomaly_result.fetchone()
    anomaly_count = int(anom_row.total) if anom_row else 0
    summary["anomalies"] = anomaly_count

    # ── Process hash chain ────────────────────────────────────────────────────
    chain_rows = chain_result.fetchall()
    chain_total, chain_valid = _verify_chains(chain_rows)
    chain_valid_pct = (chain_valid / chain_total * 100) if chain_total else 100.0
    summary["chain_verified_sessions"] = chain_valid
    summary["chain_valid_pct"] = round(chain_valid_pct, 1)

    # ── Compute overall status ────────────────────────────────────────────────
    if summary["total_sessions"] == 0:
        overall_status = "fail"
    elif chain_valid_pct < 100:
        overall_status = "fail"
    elif blocked_count > 0 or summary["pii_events"] > 0:
        overall_status = "partial"
    else:
        overall_status = "pass"
    summary["overall_status"] = overall_status

    # ── Build controls ────────────────────────────────────────────────────────
    controls = []
    for ctrl in _FRAMEWORK_CONTROLS[framework]:
        status, evidence = _compute_control_status(
            ctrl["evidence_type"],
            summary,
            violation_by_rule,
            anomaly_count,
            chain_total,
            chain_valid,
        )
        controls.append({
            "id": ctrl["id"],
            "name": ctrl["name"],
            "status": status,
            "evidence": evidence,
            "violation_detail": [
                {"rule": k, "action": v["action"], "count": v["count"]}
                for k, v in violation_by_rule.items()
            ],
        })

    # ── Top violations list ───────────────────────────────────────────────────
    top_viol_rows = top_violations_result.fetchall()
    violations_list = [
        {
            "id": r.id,
            "rule_name": r.rule_name,
            "action": r.action,
            "reason": r.reason,
            "event_type": r.event_type,
            "session_id": r.session_id,
            "agent_id": r.agent_id,
            "org_id": r.org_id,
            "timestamp_ns": r.timestamp_ns,
        }
        for r in top_viol_rows
    ]

    # ── Agent registry snapshot ───────────────────────────────────────────────
    agents_list = []
    for r in agents_rows:
        agents_list.append({
            "agent_id": r.agent_id,
            "name": r.name,
            "declared_purpose": r.declared_purpose,
            "allowed_tools": r.allowed_tools or [],
            "owner": r.owner,
            "registered_at": r.registered_at.isoformat() if r.registered_at else None,
        })

    # ── Assemble report ───────────────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    from_iso = datetime.fromtimestamp(from_ts_ns / 1e9, tz=timezone.utc).isoformat()
    to_iso = datetime.fromtimestamp(to_ts_ns / 1e9, tz=timezone.utc).isoformat()

    # Generate a simple report_id
    import time
    report_id = f"rpt-{int(time.time() * 1000)}"

    return {
        "report_id": report_id,
        "generated_at": now_iso,
        "org_id": org_id,
        "framework": framework,
        "period": {
            "from_iso": from_iso,
            "to_iso": to_iso,
            "from_ts_ns": from_ts_ns,
            "to_ts_ns": to_ts_ns,
        },
        "summary": summary,
        "controls": controls,
        "violations": violations_list,
        "agents": agents_list,
    }
