"""
Behavioral Scope Baseline — Phase 20.

Tracks how often each agent calls each tool across sessions and flags
statistically anomalous deviations using z-score analysis.

The design goal is to catch:
  - Sudden bursts: an agent that normally calls ``delete_file`` zero times per
    session calling it 50 times in a single session.
  - Novel tools: an agent suddenly using a tool it has never used before.
  - Frequency spikes: an agent sending 10× its normal number of emails.

No ML, no regex, no external dependencies required.

Architecture:
  BaselineRecord    — aggregated per-tool stats (mean, variance, call count).
  BaselineStore     — in-process dict of records with configurable window size.
  score_anomaly()   — z-score analysis; returns AnomalyResult.
  observe()         — update stats after a session ends (or mid-session).
  BaselineBackend   — ABC for pluggable persistence (in-process or Redis).
  InMemoryBackend   — default: in-process per-deployment stats; survives restarts
                      only as long as the proxy process lives.

Graceful degradation:
  If no stats are available for a (org, agent, tool) triple (new agent, new tool,
  or first N sessions), ``observe()`` updates the record and ``score_anomaly()``
  returns ``AnomalyResult(has_baseline=False)`` — no false positives.

Integration in intercept.py (TOOL_CALL_START, after compound sequence hook):
    anomaly = await baseline_store.score_anomaly(org_id, agent_id, tc_name, call_count)
    if anomaly.has_baseline and anomaly.is_anomalous:
        # fire "behavioral-baseline-violation" ALERT violation

Call-count metric:
    The metric tracked is the number of times the tool has been called in the
    current session up to this point.  On TOOL_CALL_START, session.tool_call_count
    counts ALL tools; instead, intercept.py should pass the per-tool call count
    from session.tool_call_counts[tc_name] (see note below).

    For simplicity in this initial implementation, the metric is the per-tool
    call count within the session: ``session.tool_call_counts.get(tc_name, 0)``.
    intercept.py increments this before calling score_anomaly.

Z-score computation:
    Welford's online algorithm for numerically stable variance tracking.
    Mean and variance are updated incrementally with each new observation.
    After ``min_observations`` data points the baseline is considered reliable.

Thresholds (all configurable):
    min_observations:  Minimum session observations before baseline is active.
                       Default 5 — avoids early false positives.
    alert_z_score:     Z-score above which an ALERT is fired.  Default 3.0.
    block_z_score:     Z-score above which a BLOCK is fired.  Default 5.0.
    max_tool_count:    Maximum call count stored per observation (caps outliers).
                       Default 1_000.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tool baseline record (Welford online algorithm)
# ---------------------------------------------------------------------------

@dataclass
class BaselineRecord:
    """
    Running mean and variance for one (org, agent, tool) triple.

    Uses Welford's numerically stable online algorithm so we never store all
    past observations in memory — only the three running values below.
    """
    count: int = 0          # number of sessions observed
    mean: float = 0.0       # running mean of per-session call counts
    _M2: float = 0.0        # sum of squared deviations (Welford)

    @property
    def variance(self) -> float:
        """Sample variance (Bessel-corrected).  0.0 if count < 2."""
        if self.count < 2:
            return 0.0
        return self._M2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Standard deviation.  0.0 if count < 2."""
        return math.sqrt(self.variance)

    def update(self, value: float) -> None:
        """
        Update running stats with a new observation.

        Welford's algorithm — O(1), numerically stable, single pass.
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._M2 += delta * delta2

    def z_score(self, value: float) -> float:
        """
        Compute z-score for ``value`` given current baseline.

        Returns 0.0 if std_dev is effectively zero (no variance in history).
        """
        sd = self.std_dev
        if sd < 1e-9:
            # Baseline has no variance: any non-zero value is anomalous
            # Return a large z-score if the observed value differs from mean
            if abs(value - self.mean) > 0.5:
                return float("inf")
            return 0.0
        return (value - self.mean) / sd

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "variance": round(self.variance, 4),
        }


# ---------------------------------------------------------------------------
# AnomalyResult
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """
    Result of checking a single tool call against the behavioral baseline.

    Attributes:
        tool_name:      Tool that was checked.
        has_baseline:   False if insufficient history — no alert fired.
        is_anomalous:   True if z-score exceeds alert threshold.
        should_block:   True if z-score exceeds block threshold.
        z_score:        Computed z-score (float).
        observed_count: The per-session call count observed for this tool.
        baseline:       Dict snapshot of the baseline record.
        reason:         Human-readable reason string.
    """
    tool_name: str
    has_baseline: bool = False
    is_anomalous: bool = False
    should_block: bool = False
    z_score: float = 0.0
    observed_count: int = 0
    baseline: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "has_baseline": self.has_baseline,
            "is_anomalous": self.is_anomalous,
            "should_block": self.should_block,
            "z_score": self.z_score if math.isfinite(self.z_score) else 9999.0,
            "observed_count": self.observed_count,
            "baseline": self.baseline,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# BaselineBackend ABC + InMemoryBackend
# ---------------------------------------------------------------------------

class BaselineBackend(ABC):
    """
    Pluggable persistence backend for baseline records.

    The default InMemoryBackend stores records in-process.  A Redis backend
    can be implemented by subclassing and providing serialisation to Redis
    hashes, with TTL-based expiry for rolling windows.
    """

    @abstractmethod
    def get_record(self, key: str) -> BaselineRecord:
        """Return the baseline record for ``key``, creating one if absent."""

    @abstractmethod
    def save_record(self, key: str, record: BaselineRecord) -> None:
        """Persist ``record`` under ``key``."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all records.  Used for test isolation."""


class InMemoryBackend(BaselineBackend):
    """
    In-process baseline backend.

    Records survive for the lifetime of the proxy process.  On restart the
    baseline is rebuilt from new observations (first ``min_observations``
    sessions produce no alerts).
    """

    def __init__(self) -> None:
        self._store: dict[str, BaselineRecord] = {}

    def get_record(self, key: str) -> BaselineRecord:
        if key not in self._store:
            self._store[key] = BaselineRecord()
        return self._store[key]

    def save_record(self, key: str, record: BaselineRecord) -> None:
        self._store[key] = record

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# BehavioralBaselineStore — main interface
# ---------------------------------------------------------------------------

class BehavioralBaselineStore:
    """
    Main interface for the behavioral baseline.

    Usage in intercept.py::

        # On TOOL_CALL_START:
        per_tool_count = state.tool_call_counts.get(tc_name, 0) + 1
        anomaly = baseline_store.score_anomaly(org_id, agent_id, tc_name,
                                               per_tool_count)
        if anomaly.has_baseline and anomaly.is_anomalous:
            # fire ALERT or BLOCK violation

        # On SESSION_END (or periodically):
        for tool_name, call_count in session.tool_call_counts.items():
            baseline_store.observe(org_id, agent_id, tool_name, call_count)
    """

    def __init__(
        self,
        backend: BaselineBackend | None = None,
        min_observations: int = 5,
        alert_z_score: float = 3.0,
        block_z_score: float = 5.0,
        max_tool_count: int = 1_000,
    ) -> None:
        """
        Args:
            backend:           Persistence backend.  Defaults to InMemoryBackend.
            min_observations:  Sessions to observe before baseline activates.
                               Default 5.
            alert_z_score:     Z-score threshold for ALERT.  Default 3.0.
            block_z_score:     Z-score threshold for BLOCK.  Default 5.0.
            max_tool_count:    Cap on observed count before storing (prevents
                               runaway outliers from warping the mean).  1_000.
        """
        self._backend = backend if backend is not None else InMemoryBackend()
        self._min_obs = min_observations
        self._alert_z = alert_z_score
        self._block_z = block_z_score
        self._max_count = max_tool_count

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(
        self,
        org_id: str,
        agent_id: str,
        tool_name: str,
        session_call_count: int,
    ) -> None:
        """
        Record how many times ``tool_name`` was called in a completed session.

        Should be called once per tool per session (e.g. on SESSION_END or when
        the session is evicted from the session tracker).

        Args:
            org_id:             Organisation identifier.
            agent_id:           Agent identifier.
            tool_name:          Tool name.
            session_call_count: How many times the tool was called this session.
        """
        key = _key(org_id, agent_id, tool_name)
        record = self._backend.get_record(key)
        capped = min(float(max(0, session_call_count)), float(self._max_count))
        record.update(capped)
        self._backend.save_record(key, record)
        logger.debug(
            "[BASELINE:OBSERVE] key=%s count=%d → mean=%.2f std=%.2f n=%d",
            key, session_call_count, record.mean, record.std_dev, record.count,
        )

    def score_anomaly(
        self,
        org_id: str,
        agent_id: str,
        tool_name: str,
        current_count: int,
    ) -> AnomalyResult:
        """
        Check whether ``current_count`` calls to ``tool_name`` in this session
        is anomalous compared to the baseline.

        Args:
            org_id:        Organisation identifier.
            agent_id:      Agent identifier.
            tool_name:     Tool name.
            current_count: Number of times the tool has been called so far in
                           the current session (including the call being checked).

        Returns:
            AnomalyResult.  If ``has_baseline=False``, no alert should be fired.
        """
        key = _key(org_id, agent_id, tool_name)
        record = self._backend.get_record(key)

        result = AnomalyResult(
            tool_name=tool_name,
            observed_count=current_count,
            baseline=record.to_dict(),
        )

        if record.count < self._min_obs:
            result.reason = (
                f"Insufficient baseline data ({record.count}/{self._min_obs} sessions observed)"
            )
            return result

        result.has_baseline = True
        z = record.z_score(float(current_count))
        # Cap z for display — math.inf is not JSON-serialisable
        result.z_score = z if math.isfinite(z) else 9999.0

        if abs(z) >= self._block_z or not math.isfinite(z):
            result.is_anomalous = True
            result.should_block = True
            result.reason = (
                f"Tool '{tool_name}' called {current_count}× this session — "
                f"z-score={result.z_score:.1f} EXCEEDS block threshold "
                f"({self._block_z:.1f}) — baseline: mean={record.mean:.1f} "
                f"std={record.std_dev:.1f} over {record.count} sessions"
            )
            logger.warning(
                "[BASELINE:BLOCK] key=%s count=%d z=%.1f mean=%.1f std=%.1f",
                key, current_count, result.z_score, record.mean, record.std_dev,
            )

        elif abs(z) >= self._alert_z:
            result.is_anomalous = True
            result.should_block = False
            result.reason = (
                f"Tool '{tool_name}' called {current_count}× this session — "
                f"z-score={result.z_score:.1f} exceeds alert threshold "
                f"({self._alert_z:.1f}) — baseline: mean={record.mean:.1f} "
                f"std={record.std_dev:.1f} over {record.count} sessions"
            )
            logger.info(
                "[BASELINE:ALERT] key=%s count=%d z=%.1f mean=%.1f std=%.1f",
                key, current_count, result.z_score, record.mean, record.std_dev,
            )

        else:
            result.reason = (
                f"Tool '{tool_name}' count {current_count} is within baseline "
                f"(z={result.z_score:.2f}, threshold={self._alert_z:.1f})"
            )

        return result

    def record_count(self, org_id: str, agent_id: str, tool_name: str) -> int:
        """Return the number of sessions observed for the given triple."""
        key = _key(org_id, agent_id, tool_name)
        return self._backend.get_record(key).count

    def clear(self) -> None:
        """Clear all baseline data.  Used for test isolation."""
        self._backend.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(org_id: str, agent_id: str, tool_name: str) -> str:
    """Compose a deterministic backend key from the three identifiers."""
    # Use | as separator — none of the identifiers should contain | in practice
    return f"{org_id}|{agent_id}|{tool_name}"


#: Module-level singleton for use by intercept.py.
baseline_store = BehavioralBaselineStore()
