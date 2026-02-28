"""
Markov Sequence Model — Phase 3.3

Per-agent event-type transition probability model.
Flags anomalous agent behaviour sequences that deviate from learned norms.

Design
------
- Seeded with domain priors so it works from session 1 (no cold-start gap)
- Per-agent counts with fallback to global when agent has < MIN_AGENT_SAMPLES
- Laplace smoothing (alpha=0.01) prevents zero-probability transitions
- Thread-safe via module-level lock (transitions called from thread pool)
- Never raises
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prior probabilities seeded from domain knowledge
# ---------------------------------------------------------------------------

# Maps (from_event, to_event) -> prior count weight
# These are scaled counts (not raw probabilities) that seed the model.
# Laplace alpha means even unseeded transitions have alpha probability.
_PRIOR_WEIGHTS: dict[tuple[str, str], float] = {
    ("LLM_CALL_START",  "LLM_CALL_END"):    70.0,   # normal: call starts -> ends
    ("LLM_CALL_END",    "TOOL_CALL_START"):  40.0,   # LLM requests a tool
    ("LLM_CALL_END",    "AGENT_FINISH"):     55.0,   # LLM finishes with no tool
    ("LLM_CALL_END",    "LLM_CALL_START"):   15.0,   # multi-turn without tool
    ("TOOL_CALL_END",   "LLM_CALL_START"):   90.0,   # after tool, LLM resumes
    ("TOOL_CALL_START", "TOOL_CALL_END"):    85.0,   # tool completes
    ("LLM_CALL_START",  "SYSTEM_ERROR"):      5.0,   # rare but valid
    ("TOOL_CALL_START",  "SYSTEM_ERROR"):     3.0,   # tool errors
    ("AGENT_FINISH",    "LLM_CALL_START"):    5.0,   # rare multi-episode
    ("CHECKPOINT",      "LLM_CALL_START"):    8.0,   # after checkpoint
}

_MIN_AGENT_SAMPLES = 5        # threshold before using per-agent matrix
_LAPLACE_ALPHA = 0.01         # Laplace smoothing constant
_KNOWN_EVENTS = frozenset({
    "LLM_CALL_START", "LLM_CALL_END",
    "TOOL_CALL_START", "TOOL_CALL_END",
    "AGENT_FINISH", "SYSTEM_ERROR", "CHECKPOINT", "AGENT_THOUGHT",
})


# ---------------------------------------------------------------------------
# Model internals
# ---------------------------------------------------------------------------

class _MarkovMatrix:
    """Transition count matrix with Laplace smoothing."""

    def __init__(self) -> None:
        # counts[from_event][to_event] = raw count
        self._counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._total_transitions = 0
        self._lock = threading.Lock()

        # Seed with priors
        for (from_e, to_e), weight in _PRIOR_WEIGHTS.items():
            self._counts[from_e][to_e] += weight
            self._total_transitions += 1

    def observe(self, from_event: str, to_event: str) -> None:
        with self._lock:
            self._counts[from_event][to_event] += 1.0
            self._total_transitions += 1

    def probability(self, from_event: str, to_event: str) -> float:
        """P(to_event | from_event) with Laplace smoothing."""
        with self._lock:
            row = self._counts.get(from_event)
            if row is None:
                # Unknown from_event: uniform prior across all known events
                return _LAPLACE_ALPHA / (_LAPLACE_ALPHA * len(_KNOWN_EVENTS))

            total_from = sum(row.values()) + _LAPLACE_ALPHA * len(_KNOWN_EVENTS)
            count = row.get(to_event, 0.0) + _LAPLACE_ALPHA
            return count / total_from

    @property
    def total_transitions(self) -> int:
        return self._total_transitions


# Global matrix (shared across all agents; used as fallback)
_global_matrix = _MarkovMatrix()

# Per-agent matrices
_agent_matrices: dict[str, _MarkovMatrix] = {}
_agent_lock = threading.Lock()


def _get_or_create_agent_matrix(agent_id: str) -> _MarkovMatrix:
    with _agent_lock:
        if agent_id not in _agent_matrices:
            _agent_matrices[agent_id] = _MarkovMatrix()
        return _agent_matrices[agent_id]


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class MarkovScanResult:
    from_event: str
    to_event: str
    probability: float
    is_anomaly: bool

    def to_dict(self) -> dict:
        return {
            "from_event":  self.from_event,
            "to_event":    self.to_event,
            "probability": round(self.probability, 4),
            "is_anomaly":  self.is_anomaly,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def observe_transition(from_evt: str, to_evt: str, agent_id: str) -> None:
    """
    Record an observed event-type transition.
    Updates both the per-agent matrix and the global matrix.
    Never raises.
    """
    try:
        _global_matrix.observe(from_evt, to_evt)
        agent_mat = _get_or_create_agent_matrix(agent_id)
        agent_mat.observe(from_evt, to_evt)
    except Exception as exc:
        logger.debug("observe_transition error (skipped): %s", exc)


def score_transition(
    from_evt: str,
    to_evt: str,
    agent_id: str,
    threshold: float = 0.05,
) -> MarkovScanResult:
    """
    Score an observed transition.

    Returns probability of (to_evt | from_evt) under the learned model.
    Low probability = anomalous.

    Args:
        from_evt:  Previous event type.
        to_evt:    Current event type.
        agent_id:  Agent identifier (used to select per-agent matrix).
        threshold: P(to | from) below this value is flagged as anomaly.

    Returns:
        MarkovScanResult.  Never raises.
    """
    try:
        agent_mat = _get_or_create_agent_matrix(agent_id)

        # Use per-agent matrix only if it has enough observations
        use_agent = agent_mat.total_transitions >= _MIN_AGENT_SAMPLES
        matrix = agent_mat if use_agent else _global_matrix

        prob = matrix.probability(from_evt, to_evt)
        is_anomaly = prob < threshold

        return MarkovScanResult(
            from_event=from_evt,
            to_event=to_evt,
            probability=prob,
            is_anomaly=is_anomaly,
        )
    except Exception as exc:
        logger.debug("score_transition error (skipped): %s", exc)
        return MarkovScanResult(
            from_event=from_evt,
            to_event=to_evt,
            probability=1.0,
            is_anomaly=False,
        )


def get_agent_transition_count(agent_id: str) -> int:
    """Return total transitions observed for a given agent (0 if unknown)."""
    with _agent_lock:
        mat = _agent_matrices.get(agent_id)
        return mat.total_transitions if mat else 0
