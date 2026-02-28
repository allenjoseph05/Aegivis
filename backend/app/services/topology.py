"""
Agent Topology Service.

Computes a directed graph of agent-to-agent relationships by analysing
the parent_run_id cross-references in the audit_events table.

Graph model
-----------
* Node  = a distinct agent_id with aggregated performance + security stats.
* Edge  = a directed call from one agent to another, inferred when an LLM
          call's parent_run_id belongs to a run owned by a *different* agent.

Risk scoring
------------
risk_score ∈ [0, 1] is computed from three signals:

    violation component = min(0.4, violation_count × 0.05)
    anomaly   component = min(0.3, anomaly_count  × 0.10)
    injection component = injection_score_max > 0.4
                          → (injection_score_max − 0.4) / 0.6 × 0.3

    risk_score = clamp(sum of components, 0, 1)

Levels
    [0.00 – 0.20)  →  "low"
    [0.20 – 0.40)  →  "medium"
    [0.40 – 0.70)  →  "high"
    [0.70 – 1.00]  →  "critical"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class TopologyNode:
    """Represents a single agent in the topology graph."""
    agent_id:          str
    session_count:     int
    llm_call_count:    int
    tool_call_count:   int
    error_count:       int
    violation_count:   int
    anomaly_count:     int
    pii_event_count:   int
    injection_score_max: float
    risk_score:        float
    risk_level:        str          # "low" | "medium" | "high" | "critical"
    first_seen:        str | None
    last_seen:         str | None
    providers:         list[str] = field(default_factory=list)
    models:            list[str]  = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id":            self.agent_id,
            "session_count":       self.session_count,
            "llm_call_count":      self.llm_call_count,
            "tool_call_count":     self.tool_call_count,
            "error_count":         self.error_count,
            "violation_count":     self.violation_count,
            "anomaly_count":       self.anomaly_count,
            "pii_event_count":     self.pii_event_count,
            "injection_score_max": self.injection_score_max,
            "risk_score":          self.risk_score,
            "risk_level":          self.risk_level,
            "first_seen":          self.first_seen,
            "last_seen":           self.last_seen,
            "providers":           self.providers,
            "models":              self.models,
        }


@dataclass
class TopologyEdge:
    """Represents a directed call from one agent to another."""
    source:         str   # calling agent
    target:         str   # called agent
    call_count:     int
    avg_latency_ms: float | None
    first_seen:     str | None
    last_seen:      str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source":         self.source,
            "target":         self.target,
            "call_count":     self.call_count,
            "avg_latency_ms": self.avg_latency_ms,
            "first_seen":     self.first_seen,
            "last_seen":      self.last_seen,
        }


@dataclass
class TopologyGraph:
    """Complete graph: nodes + directed edges."""
    nodes:       list[TopologyNode]
    edges:       list[TopologyEdge]
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes":       [n.to_dict() for n in self.nodes],
            "edges":       [e.to_dict() for e in self.edges],
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "computed_at": self.computed_at,
        }


# ---------------------------------------------------------------------------
# Risk scoring helpers
# ---------------------------------------------------------------------------

def _compute_risk(
    violation_count: int,
    anomaly_count: int,
    injection_score_max: float,
) -> tuple[float, str]:
    """
    Return (risk_score, risk_level).

    The formula is designed so that any *single* signal stays below 0.40,
    requiring multiple signals to reach HIGH or CRITICAL.
    """
    score = 0.0
    score += min(0.4, violation_count * 0.05)
    score += min(0.3, anomaly_count  * 0.10)
    if injection_score_max > 0.4:
        score += (injection_score_max - 0.4) / 0.6 * 0.3
    score = round(min(1.0, score), 3)

    if score >= 0.70:
        level = "critical"
    elif score >= 0.40:
        level = "high"
    elif score >= 0.20:
        level = "medium"
    else:
        level = "low"

    return score, level


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

async def _fetch_agent_stats(db: AsyncSession, org_id: str) -> list[dict]:
    """
    Return per-agent aggregate stats from audit_events, agent_anomalies,
    and policy_violations, filtered to a single org.
    """
    result = await db.execute(text("""
        SELECT
            ae.agent_id,
            COUNT(DISTINCT ae.session_id)                                           AS session_count,
            COUNT(*) FILTER (WHERE ae.event_type = 'LLM_CALL_START')               AS llm_call_count,
            COUNT(*) FILTER (WHERE ae.event_type = 'TOOL_CALL_START')              AS tool_call_count,
            COUNT(*) FILTER (WHERE ae.event_type = 'SYSTEM_ERROR')                 AS error_count,
            COUNT(*) FILTER (WHERE cardinality(ae.pii_detected) > 0)              AS pii_event_count,
            COALESCE(MAX(
                CAST(ae.payload->'security'->>'injection_score' AS FLOAT)
            ) FILTER (WHERE ae.payload->'security' IS NOT NULL), 0.0)              AS injection_score_max,
            COALESCE(an.anomaly_count, 0)                                           AS anomaly_count,
            COALESCE(pv.violation_count, 0)                                         AS violation_count,
            to_timestamp(MIN(ae.timestamp_ns) / 1e9)                               AS first_seen,
            to_timestamp(MAX(ae.timestamp_ns) / 1e9)                               AS last_seen
        FROM audit_events ae
        LEFT JOIN (
            SELECT agent_id, COUNT(*) AS anomaly_count
            FROM agent_anomalies
            WHERE org_id = :org_id
            GROUP BY agent_id
        ) an ON ae.agent_id = an.agent_id
        LEFT JOIN (
            SELECT agent_id, COUNT(*) AS violation_count
            FROM policy_violations
            WHERE org_id = :org_id
            GROUP BY agent_id
        ) pv ON ae.agent_id = pv.agent_id
        WHERE ae.org_id = :org_id
        GROUP BY ae.agent_id, an.anomaly_count, pv.violation_count
        ORDER BY MAX(ae.timestamp_ns) DESC
    """), {"org_id": org_id})
    return [dict(r) for r in result.mappings().all()]


async def _fetch_agent_providers_models(db: AsyncSession, org_id: str) -> dict[str, dict[str, list[str]]]:
    """Return {agent_id: {"providers": [...], "models": [...]}} from audit_events."""
    result = await db.execute(text("""
        SELECT agent_id,
               ARRAY_AGG(DISTINCT provider) AS providers,
               ARRAY_AGG(DISTINCT model)    AS models
        FROM audit_events
        WHERE org_id = :org_id
          AND provider IS NOT NULL
          AND model    IS NOT NULL
          AND model != '' AND model != 'unknown'
        GROUP BY agent_id
    """), {"org_id": org_id})
    out: dict[str, dict[str, list[str]]] = {}
    for r in result.mappings().all():
        out[r["agent_id"]] = {
            "providers": list(r["providers"] or []),
            "models":    list(r["models"]    or []),
        }
    return out


async def _fetch_edges(db: AsyncSession, org_id: str, min_calls: int = 1) -> list[dict]:
    """
    Return directed edges between agents within a single org.

    An edge source→target exists when an LLM_CALL_START event for *target*
    has a parent_run_id that belongs to a run owned by *source*.
    """
    result = await db.execute(text("""
        SELECT
            parent.agent_id                                                         AS source,
            child.agent_id                                                          AS target,
            COUNT(*)                                                                AS call_count,
            ROUND(AVG(
                CAST(latency.payload->>'latency_ms' AS FLOAT)
            )::numeric, 2)                                                          AS avg_latency_ms,
            to_timestamp(MIN(child.timestamp_ns) / 1e9)                            AS first_seen,
            to_timestamp(MAX(child.timestamp_ns) / 1e9)                            AS last_seen
        FROM audit_events child
        JOIN audit_events parent
          ON child.parent_run_id = parent.run_id
         AND child.agent_id     != parent.agent_id
         AND child.org_id        = parent.org_id
        LEFT JOIN audit_events latency
          ON latency.run_id    = child.run_id
         AND latency.event_type = 'LLM_CALL_END'
        WHERE child.event_type = 'LLM_CALL_START'
          AND child.org_id     = :org_id
        GROUP BY parent.agent_id, child.agent_id
        HAVING COUNT(*) >= :min_calls
        ORDER BY COUNT(*) DESC
    """), {"org_id": org_id, "min_calls": min_calls})
    return [dict(r) for r in result.mappings().all()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def compute_topology(
    db: AsyncSession,
    *,
    org_id: str,
    include_isolated: bool = True,
    min_edge_calls: int = 1,
) -> TopologyGraph:
    """
    Build the full agent topology graph for a single org.

    Parameters
    ----------
    db:
        Active async SQLAlchemy session.
    org_id:
        Organisation to scope all queries to.
    include_isolated:
        When True (default), include agents with no edges (stand-alone agents).
        Set False to show only agents that call or are called by other agents.
    min_edge_calls:
        Minimum cross-agent calls needed to include an edge.
    """
    import datetime

    agent_rows        = await _fetch_agent_stats(db, org_id)
    providers_models  = await _fetch_agent_providers_models(db, org_id)
    edge_rows         = await _fetch_edges(db, org_id, min_calls=min_edge_calls)

    # Build set of connected agents for filtering
    connected_agents: set[str] = set()
    for e in edge_rows:
        connected_agents.add(e["source"])
        connected_agents.add(e["target"])

    nodes: list[TopologyNode] = []
    for row in agent_rows:
        aid = row["agent_id"]
        if not include_isolated and aid not in connected_agents:
            continue

        injection_max = float(row["injection_score_max"] or 0.0)
        risk_score, risk_level = _compute_risk(
            violation_count     = int(row["violation_count"] or 0),
            anomaly_count       = int(row["anomaly_count"]   or 0),
            injection_score_max = injection_max,
        )
        pm = providers_models.get(aid, {})
        nodes.append(TopologyNode(
            agent_id           = aid,
            session_count      = int(row["session_count"]   or 0),
            llm_call_count     = int(row["llm_call_count"]  or 0),
            tool_call_count    = int(row["tool_call_count"] or 0),
            error_count        = int(row["error_count"]     or 0),
            violation_count    = int(row["violation_count"] or 0),
            anomaly_count      = int(row["anomaly_count"]   or 0),
            pii_event_count    = int(row["pii_event_count"] or 0),
            injection_score_max = injection_max,
            risk_score         = risk_score,
            risk_level         = risk_level,
            first_seen         = row["first_seen"].isoformat() if row["first_seen"] else None,
            last_seen          = row["last_seen"].isoformat()  if row["last_seen"]  else None,
            providers          = pm.get("providers", []),
            models             = pm.get("models",    []),
        ))

    edges: list[TopologyEdge] = []
    for row in edge_rows:
        edges.append(TopologyEdge(
            source         = row["source"],
            target         = row["target"],
            call_count     = int(row["call_count"] or 0),
            avg_latency_ms = float(row["avg_latency_ms"]) if row["avg_latency_ms"] is not None else None,
            first_seen     = row["first_seen"].isoformat() if row["first_seen"] else None,
            last_seen      = row["last_seen"].isoformat()  if row["last_seen"]  else None,
        ))

    return TopologyGraph(
        nodes       = nodes,
        edges       = edges,
        computed_at = datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
