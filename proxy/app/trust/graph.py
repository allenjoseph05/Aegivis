"""
Trust Graph — Phase 9C

Tracks a per-session and per-agent trust score [0.0, 1.0].

Trust starts at 1.0 (fully trusted) and degrades based on:
  - Each BLOCK violation: −0.05 per incident (max −0.40)
  - Injection score contribution: −score × 0.30
  - Propagated from parent: inherited trust degradation

Trust recovery:
  - Each clean LLM call (no violations): +0.005 per call

Attack vector countered:
  Compromised parent agents (e.g. a data ingestion agent that was injected)
  that spawn child agents inherit the compromised trust score. A canary leakage
  in any ancestor automatically degrades all descendants to near-zero trust,
  which triggers BLOCK on their subsequent high-risk tool calls.

Design notes:
  - TrustGraph is an in-memory dict keyed by session_id. Ephemeral.
  - Trust is NOT persisted to Redis (trust resets on proxy restart, which is
    acceptable — an attacker cannot "carry over" a good trust score).
  - Trust propagation is one-directional: parent → children (never reverse).
  - The graph stores the spawn tree lazily (session registers its parent on
    first call, same as spawn chain tracking in session.py).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Trust score constants
_TRUST_INIT = 1.0
_TRUST_FLOOR = 0.0

# Degradation deltas
_DELTA_PER_BLOCK_VIOLATION = 0.05   # subtracted per BLOCK event
_DELTA_MAX_FROM_VIOLATIONS = 0.40   # cap on cumulative violation penalty
_DELTA_INJECTION_FACTOR   = 0.30   # multiplied by injection_score
_DELTA_CANARY_LEAK        = 0.60   # canary leakage: heavy hit to all descendants

# Recovery
_RECOVERY_PER_CLEAN_CALL  = 0.005

# Canary-leak rules (any of these → treat as canary exfiltration)
_CANARY_RULES: frozenset[str] = frozenset({
    "canary-leak",
    "canary-token-leak",
    "data-exfiltration-attempt",
})


@dataclass
class TrustEntry:
    """Per-session trust record."""
    session_id: str
    agent_id: str
    parent_session_id: str | None
    parent_agent_id: str | None
    trust_score: float = _TRUST_INIT
    violation_count: int = 0
    clean_call_count: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_updated: float = field(default_factory=time.monotonic)

    def apply_degradation(self, delta: float) -> None:
        self.trust_score = max(_TRUST_FLOOR, self.trust_score - delta)
        self.last_updated = time.monotonic()

    def apply_recovery(self) -> None:
        self.clean_call_count += 1
        # Slow recovery, capped at original score before violations
        self.trust_score = min(_TRUST_INIT, self.trust_score + _RECOVERY_PER_CLEAN_CALL)
        self.last_updated = time.monotonic()


class TrustGraph:
    """
    In-memory trust graph.

    One instance per proxy process. Sessions register when first seen;
    trust degrades on violations and propagates to all descendant sessions.

    Thread safety: asyncio single-threaded — no locking needed.
    """

    def __init__(self) -> None:
        # session_id → TrustEntry
        self._nodes: dict[str, TrustEntry] = {}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(
        self,
        session_id: str,
        agent_id: str,
        parent_session_id: str | None = None,
        parent_agent_id: str | None = None,
        initial_trust: float | None = None,
    ) -> TrustEntry:
        """
        Register a session with the graph.

        If the session is already registered, returns the existing entry.
        If a parent is known, the child inherits the parent's current trust score.
        """
        if session_id in self._nodes:
            return self._nodes[session_id]

        # Inherit parent trust score as starting point
        start_trust = _TRUST_INIT
        if initial_trust is not None:
            start_trust = initial_trust
        elif parent_session_id and parent_session_id in self._nodes:
            parent_trust = self._nodes[parent_session_id].trust_score
            # Child starts at parent's current score (trust is inherited)
            start_trust = parent_trust

        entry = TrustEntry(
            session_id=session_id,
            agent_id=agent_id,
            parent_session_id=parent_session_id,
            parent_agent_id=parent_agent_id,
            trust_score=start_trust,
        )
        self._nodes[session_id] = entry
        logger.debug(
            "Trust: registered session=%s agent=%s parent=%s trust=%.3f",
            session_id, agent_id, parent_session_id, start_trust,
        )
        return entry

    # -------------------------------------------------------------------------
    # Score accessors
    # -------------------------------------------------------------------------

    def get_trust(self, session_id: str) -> float:
        """Return trust score for a session (1.0 if not registered)."""
        entry = self._nodes.get(session_id)
        return entry.trust_score if entry else _TRUST_INIT

    def get_entry(self, session_id: str) -> TrustEntry | None:
        return self._nodes.get(session_id)

    # -------------------------------------------------------------------------
    # Degradation
    # -------------------------------------------------------------------------

    def on_violation(
        self,
        session_id: str,
        rule_name: str,
        action: str,
        injection_score: float = 0.0,
    ) -> float:
        """
        Apply trust degradation for a violation and propagate to descendants.

        Returns the new trust score for the affected session.
        """
        if session_id not in self._nodes:
            return _TRUST_INIT

        entry = self._nodes[session_id]
        entry.violation_count += 1

        is_canary = rule_name in _CANARY_RULES
        is_block  = action.upper() == "BLOCK"

        if is_canary:
            # Canary leakage: heavy hit to this session
            entry.apply_degradation(_DELTA_CANARY_LEAK)
            logger.warning(
                "Trust: CANARY LEAK session=%s — trust %.3f → %.3f",
                session_id, entry.trust_score + _DELTA_CANARY_LEAK, entry.trust_score,
            )
        elif is_block:
            block_delta = min(
                _DELTA_PER_BLOCK_VIOLATION,
                _DELTA_MAX_FROM_VIOLATIONS - (entry.violation_count - 1) * _DELTA_PER_BLOCK_VIOLATION,
            )
            block_delta = max(0.0, block_delta)
            inj_delta   = injection_score * _DELTA_INJECTION_FACTOR
            total_delta = block_delta + inj_delta
            entry.apply_degradation(total_delta)
            logger.info(
                "Trust: BLOCK session=%s rule=%s — trust %.3f (delta=−%.3f)",
                session_id, rule_name, entry.trust_score, total_delta,
            )
        else:
            # ALERT — lighter touch
            inj_delta = injection_score * _DELTA_INJECTION_FACTOR * 0.3
            entry.apply_degradation(inj_delta)

        # Propagate to all descendant sessions
        self._propagate_downward(session_id, is_canary=is_canary)

        return entry.trust_score

    def _propagate_downward(self, source_session_id: str, *, is_canary: bool) -> None:
        """Apply reduced trust degradation to all sessions that descend from source."""
        source_entry = self._nodes.get(source_session_id)
        if source_entry is None:
            return

        for sid, entry in self._nodes.items():
            if sid == source_session_id:
                continue
            if not self._is_descendant(sid, source_session_id):
                continue

            if is_canary:
                # Canary propagates fully to all descendants
                entry.apply_degradation(_DELTA_CANARY_LEAK)
                logger.warning(
                    "Trust: CANARY propagated to child session=%s — trust %.3f",
                    sid, entry.trust_score,
                )
            else:
                # Non-canary: propagate 50% of source's trust loss
                source_loss = _TRUST_INIT - source_entry.trust_score
                child_delta = source_loss * 0.50
                if child_delta > 0.01:
                    entry.apply_degradation(child_delta)
                    logger.debug(
                        "Trust: propagated %.3f to child session=%s — trust %.3f",
                        child_delta, sid, entry.trust_score,
                    )

    def _is_descendant(self, session_id: str, ancestor_id: str) -> bool:
        """Walk parent pointers to determine if session_id descends from ancestor_id."""
        visited: set[str] = set()
        current = session_id
        while current is not None:
            if current in visited:
                break
            visited.add(current)
            entry = self._nodes.get(current)
            if entry is None:
                break
            if entry.parent_session_id == ancestor_id:
                return True
            current = entry.parent_session_id
        return False

    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    def on_clean_call(self, session_id: str) -> None:
        """Record a clean LLM call (no violations) and apply slow trust recovery."""
        entry = self._nodes.get(session_id)
        if entry:
            entry.apply_recovery()

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the full graph for the REST API."""
        return {
            "node_count": len(self._nodes),
            "nodes": [
                {
                    "session_id":        e.session_id,
                    "agent_id":          e.agent_id,
                    "parent_session_id": e.parent_session_id,
                    "parent_agent_id":   e.parent_agent_id,
                    "trust_score":       round(e.trust_score, 4),
                    "violation_count":   e.violation_count,
                    "clean_call_count":  e.clean_call_count,
                    "last_updated":      e.last_updated,
                }
                for e in self._nodes.values()
            ],
        }

    def reset_session(self, session_id: str) -> bool:
        """Reset a specific session's trust score to 1.0. Returns True if found."""
        entry = self._nodes.get(session_id)
        if entry is None:
            return False
        entry.trust_score = _TRUST_INIT
        entry.violation_count = 0
        entry.clean_call_count = 0
        entry.last_updated = time.monotonic()
        logger.info("Trust: reset session=%s to 1.0", session_id)
        return True

    def reset_all(self) -> int:
        """Reset all trust scores. Returns number of nodes reset."""
        count = len(self._nodes)
        self._nodes.clear()
        logger.info("Trust: full graph reset (%d nodes cleared)", count)
        return count

    def active_count(self) -> int:
        return len(self._nodes)


# ---------------------------------------------------------------------------
# Module-level singleton (one graph per proxy process)
# ---------------------------------------------------------------------------

_graph: TrustGraph | None = None


def get_trust_graph() -> TrustGraph:
    global _graph
    if _graph is None:
        _graph = TrustGraph()
    return _graph
