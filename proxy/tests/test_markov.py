"""Tests for proxy.app.security.markov (Phase 3.3)."""
import pytest
from app.security.markov import (
    score_transition,
    observe_transition,
    get_agent_transition_count,
    MarkovScanResult,
    _global_matrix,
    _PRIOR_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Prior probabilities
# ---------------------------------------------------------------------------

def test_common_transition_high_probability():
    """LLM_CALL_START -> LLM_CALL_END should be very likely (seeded prior)."""
    r = score_transition("LLM_CALL_START", "LLM_CALL_END", "test-agent-1")
    assert r.probability > 0.30
    assert r.is_anomaly is False


def test_tool_end_to_llm_start_high_probability():
    """TOOL_CALL_END -> LLM_CALL_START is the most common tool flow transition."""
    r = score_transition("TOOL_CALL_END", "LLM_CALL_START", "test-agent-2")
    assert r.probability > 0.50
    assert r.is_anomaly is False


def test_agent_finish_after_llm_end():
    """LLM_CALL_END -> AGENT_FINISH should be common."""
    r = score_transition("LLM_CALL_END", "AGENT_FINISH", "test-agent-3")
    assert r.probability > 0.20
    assert r.is_anomaly is False


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_unknown_transition_is_anomaly():
    """AGENT_FINISH -> TOOL_CALL_START is highly unusual."""
    r = score_transition("AGENT_FINISH", "TOOL_CALL_START", "test-agent-4")
    # Should be low probability -- not in priors and unusual
    assert r.is_anomaly is True or r.probability < 0.15


def test_system_error_after_start_is_low():
    """LLM_CALL_START -> SYSTEM_ERROR has low prior (5%)."""
    r = score_transition("LLM_CALL_START", "SYSTEM_ERROR", "test-agent-5")
    assert r.probability < 0.20


# ---------------------------------------------------------------------------
# observe_transition
# ---------------------------------------------------------------------------

def test_observe_increases_count():
    initial = get_agent_transition_count("observe-test-agent")
    observe_transition("LLM_CALL_START", "LLM_CALL_END", "observe-test-agent")
    assert get_agent_transition_count("observe-test-agent") > initial


def test_observe_then_score_learned():
    """After many observations, the learned probability should update."""
    agent = "learning-agent-unique"
    # Observe AGENT_FINISH -> LLM_CALL_START 20 times (unusual transition)
    for _ in range(20):
        observe_transition("AGENT_FINISH", "LLM_CALL_START", agent)

    r = score_transition("AGENT_FINISH", "LLM_CALL_START", agent)
    # After learning, probability should be higher than before
    assert r.probability > 0.0


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_fields():
    r = score_transition("LLM_CALL_START", "LLM_CALL_END", "struct-test")
    assert hasattr(r, "from_event")
    assert hasattr(r, "to_event")
    assert hasattr(r, "probability")
    assert hasattr(r, "is_anomaly")
    assert 0.0 <= r.probability <= 1.0


def test_to_dict():
    r = score_transition("LLM_CALL_END", "TOOL_CALL_START", "dict-test")
    d = r.to_dict()
    assert "from_event" in d
    assert "to_event" in d
    assert "probability" in d
    assert "is_anomaly" in d


def test_never_raises_bad_event():
    """Unknown event types should not raise."""
    r = score_transition("UNKNOWN_EVENT", "ALSO_UNKNOWN", "bad-agent")
    assert isinstance(r, MarkovScanResult)
    assert 0.0 <= r.probability <= 1.0


def test_threshold_parameter():
    """Custom threshold changes is_anomaly classification."""
    r_strict = score_transition("LLM_CALL_END", "AGENT_FINISH", "threshold-test", threshold=0.99)
    r_loose = score_transition("LLM_CALL_END", "AGENT_FINISH", "threshold-test", threshold=0.0)
    # strict threshold: almost everything is anomaly
    # loose threshold: nothing is anomaly
    assert r_strict.is_anomaly is True
    assert r_loose.is_anomaly is False
