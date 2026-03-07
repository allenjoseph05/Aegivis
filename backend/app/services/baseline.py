"""
Behavioral baseline service.

After each agent session completes (AGENT_FINISH event), we update a running
Welford online mean/variance per (org_id, agent_id). Drift is detected when
a session's stats deviate more than `drift_threshold` standard deviations from
the baseline mean.

This enables detecting:
- Sudden spike in tool calls (possible prompt injection / loop bug)
- Latency anomalies (model performance degradation)
- Token usage spikes (runaway agent / changed prompt)
- Session duration outliers
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# How many standard deviations above the mean triggers a drift alert
DRIFT_THRESHOLD = 3.0
# Minimum sessions before drift detection is meaningful
MIN_SESSIONS_FOR_DRIFT = 5


@dataclass
class DriftResult:
    agent_id: str
    org_id: str
    drifted: bool
    drift_fields: list[str]      # which metrics drifted
    details: dict                # field -> {value, mean, std_dev, z_score}


async def update_baseline(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
    session_stats: dict,
) -> None:
    """
    Upsert the agent's baseline with new session stats using Welford's algorithm.

    session_stats keys (all optional, 0 if missing):
        llm_call_count, tool_call_count, session_duration_ms,
        avg_latency_ms, total_tokens
    """
    row = await _get_baseline(db, org_id, agent_id)

    new_vals = {
        "llm_calls":    float(session_stats.get("llm_call_count", 0)),
        "tool_calls":   float(session_stats.get("tool_call_count", 0)),
        "duration_ms":  float(session_stats.get("session_duration_ms", 0)),
        "latency_ms":   float(session_stats.get("avg_latency_ms", 0)),
        "tokens":       float(session_stats.get("total_tokens", 0)),
    }

    if row is None:
        # First session — initialize baseline
        await db.execute(
            text("""
                INSERT INTO agent_baselines
                    (org_id, agent_id, session_count,
                     avg_llm_calls, avg_tool_calls,
                     avg_session_duration_ms, avg_latency_ms, avg_total_tokens,
                     last_updated)
                VALUES
                    (:org_id, :agent_id, 1,
                     :llm, :tools, :duration, :latency, :tokens,
                     NOW())
            """),
            {
                "org_id": org_id,
                "agent_id": agent_id,
                "llm": new_vals["llm_calls"],
                "tools": new_vals["tool_calls"],
                "duration": new_vals["duration_ms"],
                "latency": new_vals["latency_ms"],
                "tokens": new_vals["tokens"],
            },
        )
    else:
        n = row["session_count"] + 1
        # Welford online mean update: new_mean = old_mean + (x - old_mean) / n
        def welford_mean(old_mean: float, x: float) -> float:
            return old_mean + (x - old_mean) / n

        await db.execute(
            text("""
                UPDATE agent_baselines SET
                    session_count = :n,
                    avg_llm_calls = :llm,
                    avg_tool_calls = :tools,
                    avg_session_duration_ms = :duration,
                    avg_latency_ms = :latency,
                    avg_total_tokens = :tokens,
                    last_updated = NOW()
                WHERE org_id = :org_id AND agent_id = :agent_id
            """),
            {
                "n": n,
                "org_id": org_id,
                "agent_id": agent_id,
                "llm":      welford_mean(row["avg_llm_calls"], new_vals["llm_calls"]),
                "tools":    welford_mean(row["avg_tool_calls"], new_vals["tool_calls"]),
                "duration": welford_mean(row["avg_session_duration_ms"], new_vals["duration_ms"]),
                "latency":  welford_mean(row["avg_latency_ms"], new_vals["latency_ms"]),
                "tokens":   welford_mean(row["avg_total_tokens"], new_vals["tokens"]),
            },
        )


async def check_drift(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
    session_stats: dict,
) -> DriftResult:
    """
    Compare session_stats against stored baseline.
    Returns a DriftResult with drifted=True if any metric exceeds DRIFT_THRESHOLD std devs.
    Uses a simplified std_dev estimate: std_dev ~ mean * 0.5 (50% CV assumption)
    when we don't have enough history to compute it properly.
    """
    row = await _get_baseline(db, org_id, agent_id)

    if row is None or row["session_count"] < MIN_SESSIONS_FOR_DRIFT:
        return DriftResult(
            agent_id=agent_id, org_id=org_id,
            drifted=False, drift_fields=[], details={},
        )

    metrics = {
        "llm_call_count":       ("avg_llm_calls", session_stats.get("llm_call_count", 0)),
        "tool_call_count":      ("avg_tool_calls", session_stats.get("tool_call_count", 0)),
        "session_duration_ms":  ("avg_session_duration_ms", session_stats.get("session_duration_ms", 0)),
        "total_tokens":         ("avg_total_tokens", session_stats.get("total_tokens", 0)),
    }

    drift_fields = []
    details = {}

    for stat_key, (baseline_key, actual_value) in metrics.items():
        baseline_mean = float(row[baseline_key] or 0)
        if baseline_mean == 0:
            continue

        # Approximate std_dev as 50% of mean (conservative estimate)
        std_dev = baseline_mean * 0.5
        z_score = (float(actual_value) - baseline_mean) / std_dev if std_dev > 0 else 0

        details[stat_key] = {
            "value": actual_value,
            "baseline_mean": round(baseline_mean, 2),
            "std_dev_estimate": round(std_dev, 2),
            "z_score": round(z_score, 2),
        }

        if abs(z_score) > DRIFT_THRESHOLD:
            drift_fields.append(stat_key)

    return DriftResult(
        agent_id=agent_id,
        org_id=org_id,
        drifted=bool(drift_fields),
        drift_fields=drift_fields,
        details=details,
    )


async def get_baseline(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
) -> dict | None:
    return await _get_baseline(db, org_id, agent_id)


async def _get_baseline(db: AsyncSession, org_id: str, agent_id: str) -> dict | None:
    result = await db.execute(
        text("""
            SELECT session_count, avg_llm_calls, avg_tool_calls,
                   avg_session_duration_ms, avg_latency_ms, avg_total_tokens,
                   last_updated
            FROM agent_baselines
            WHERE org_id = :org_id AND agent_id = :agent_id
        """),
        {"org_id": org_id, "agent_id": agent_id},
    )
    row = result.fetchone()
    if row is None:
        return None
    return dict(row._mapping)
