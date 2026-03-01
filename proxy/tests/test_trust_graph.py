"""
Unit tests for proxy/app/trust/graph.py — Phase 9C

12 tests. No network, no ML, no Docker required.
Run: cd proxy && python -m pytest tests/test_trust_graph.py -v
"""
import pytest

from app.trust.graph import (
    TrustGraph,
    _TRUST_INIT,
    _DELTA_CANARY_LEAK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_with_parent_child() -> tuple[TrustGraph, str, str]:
    """Return (graph, parent_sid, child_sid) with parent already registered."""
    g = TrustGraph()
    p = "sess-parent"
    c = "sess-child"
    g.register(p, "agent-parent")
    g.register(c, "agent-child", parent_session_id=p)
    return g, p, c


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_new_session_starts_at_full_trust(self):
        g = TrustGraph()
        entry = g.register("sess-1", "agent-1")
        assert entry.trust_score == pytest.approx(1.0)

    def test_double_registration_returns_same_entry(self):
        g = TrustGraph()
        e1 = g.register("sess-1", "agent-1")
        e2 = g.register("sess-1", "agent-1")
        assert e1 is e2

    def test_child_inherits_parent_trust(self):
        g = TrustGraph()
        # Parent degrades
        g.register("sess-parent", "agent-parent")
        g.on_violation("sess-parent", "high-injection-score", "BLOCK")
        parent_trust = g.get_trust("sess-parent")
        # Child inherits parent's degraded trust
        e = g.register("sess-child", "agent-child", parent_session_id="sess-parent")
        assert e.trust_score == pytest.approx(parent_trust)

    def test_unregistered_session_returns_full_trust(self):
        g = TrustGraph()
        assert g.get_trust("unknown-session") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Violation degradation
# ---------------------------------------------------------------------------

class TestViolationDegradation:
    def test_block_violation_decreases_trust(self):
        g = TrustGraph()
        g.register("sess-1", "agent-1")
        new_trust = g.on_violation("sess-1", "high-injection-score", "BLOCK")
        assert new_trust < 1.0

    def test_canary_leak_applies_heavy_penalty(self):
        g = TrustGraph()
        g.register("sess-1", "agent-1")
        new_trust = g.on_violation("sess-1", "canary-token-leak", "BLOCK")
        # Canary penalty is _DELTA_CANARY_LEAK = 0.60
        assert new_trust == pytest.approx(1.0 - _DELTA_CANARY_LEAK, abs=0.001)

    def test_trust_never_goes_below_zero(self):
        g = TrustGraph()
        g.register("sess-1", "agent-1")
        for _ in range(30):
            g.on_violation("sess-1", "high-injection-score", "BLOCK", injection_score=0.99)
        assert g.get_trust("sess-1") >= 0.0

    def test_alert_violation_smaller_penalty_than_block(self):
        g1 = TrustGraph()
        g1.register("sess-1", "agent-1")
        g1.on_violation("sess-1", "injection-score-alert", "ALERT", injection_score=0.5)
        alert_trust = g1.get_trust("sess-1")

        g2 = TrustGraph()
        g2.register("sess-1", "agent-1")
        g2.on_violation("sess-1", "high-injection-score", "BLOCK", injection_score=0.5)
        block_trust = g2.get_trust("sess-1")

        assert alert_trust > block_trust


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

class TestTrustPropagation:
    def test_canary_leak_propagates_to_child(self):
        g, parent, child = _make_graph_with_parent_child()
        g.on_violation(parent, "canary-token-leak", "BLOCK")
        # Child should also have degraded trust (canary propagation)
        child_trust = g.get_trust(child)
        assert child_trust < 1.0

    def test_block_propagates_partially_to_child(self):
        g, parent, child = _make_graph_with_parent_child()
        g.on_violation(parent, "high-injection-score", "BLOCK")
        parent_trust = g.get_trust(parent)
        child_trust  = g.get_trust(child)
        # Child degrades but less than parent
        assert child_trust < 1.0
        assert child_trust > parent_trust

    def test_sibling_not_affected_by_other_sibling_violation(self):
        g = TrustGraph()
        g.register("sess-parent", "agent-parent")
        g.register("sess-child-a", "agent-a", parent_session_id="sess-parent")
        g.register("sess-child-b", "agent-b", parent_session_id="sess-parent")
        # Violation on child-a should NOT propagate to child-b (sibling, not descendant)
        g.on_violation("sess-child-a", "high-injection-score", "BLOCK")
        assert g.get_trust("sess-child-b") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recovery + reset
# ---------------------------------------------------------------------------

class TestRecoveryAndReset:
    def test_clean_calls_recover_trust(self):
        g = TrustGraph()
        g.register("sess-1", "agent-1")
        g.on_violation("sess-1", "high-injection-score", "BLOCK")
        degraded = g.get_trust("sess-1")
        # Multiple clean calls
        for _ in range(50):
            g.on_clean_call("sess-1")
        recovered = g.get_trust("sess-1")
        assert recovered > degraded

    def test_reset_session_restores_full_trust(self):
        g = TrustGraph()
        g.register("sess-1", "agent-1")
        g.on_violation("sess-1", "canary-token-leak", "BLOCK")
        assert g.get_trust("sess-1") < 1.0
        g.reset_session("sess-1")
        assert g.get_trust("sess-1") == pytest.approx(1.0)

    def test_to_dict_includes_all_nodes(self):
        g = TrustGraph()
        g.register("sess-1", "agent-1")
        g.register("sess-2", "agent-2")
        d = g.to_dict()
        assert d["node_count"] == 2
        session_ids = {n["session_id"] for n in d["nodes"]}
        assert {"sess-1", "sess-2"} == session_ids
