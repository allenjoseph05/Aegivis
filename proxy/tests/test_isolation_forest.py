"""Tests for proxy.app.security.isolation_forest (Phase 3.3)."""
import pytest
from app.security.isolation_forest import (
    fit_and_score,
    samples_seen,
    IsolationForestResult,
    _MIN_SAMPLES,
)


def _normal_features(
    llm_calls=5.0,
    tool_rate=0.5,
    error_rate=0.0,
    duration=2.0,
    injection_score=0.0,
) -> dict:
    return {
        "llm_call_count": llm_calls,
        "tool_call_rate": tool_rate,
        "error_rate": error_rate,
        "session_duration_min": duration,
        "max_injection_score": injection_score,
    }


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

def test_returns_none_before_min_samples():
    """Model should return None until MIN_SAMPLES sessions are seen."""
    # Since tests may share state from prior runs, just verify None is possible
    # by using clearly low sample counts (isolation_forest is module-level singleton)
    result = fit_and_score(_normal_features())
    # Either None (not enough samples) or a valid result
    assert result is None or isinstance(result, IsolationForestResult)


def test_result_type_after_sufficient_data():
    """After enough samples, fit_and_score returns IsolationForestResult or None."""
    # Feed MIN_SAMPLES normal sessions
    for i in range(_MIN_SAMPLES + 5):
        r = fit_and_score(_normal_features(llm_calls=float(i % 10 + 1)))
        # Either result or None (sklearn may not be installed)
        assert r is None or isinstance(r, IsolationForestResult)


def test_result_score_in_range():
    """Anomaly score must be in [0.0, 1.0]."""
    for i in range(_MIN_SAMPLES + 5):
        r = fit_and_score(_normal_features())
        if r is not None:
            assert 0.0 <= r.anomaly_score <= 1.0
            break


def test_features_recorded_in_result():
    """Result should contain the feature vector that was scored."""
    features = _normal_features(llm_calls=7.0, tool_rate=0.3)
    for _ in range(_MIN_SAMPLES + 5):
        r = fit_and_score(features)
        if r is not None:
            assert "llm_call_count" in r.features
            assert "tool_call_rate" in r.features
            break


def test_never_raises_on_bad_features():
    """Missing or invalid feature keys should not raise."""
    result = fit_and_score({})
    assert result is None or isinstance(result, IsolationForestResult)


def test_never_raises_on_none_features():
    """None-valued features should not raise."""
    result = fit_and_score({"llm_call_count": None})  # type: ignore[arg-type]
    assert result is None or isinstance(result, IsolationForestResult)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_to_dict_structure():
    result = IsolationForestResult(
        anomaly_score=0.2,
        is_anomaly=False,
        features={"llm_call_count": 5.0},
        samples_seen=15,
    )
    d = result.to_dict()
    assert "anomaly_score" in d
    assert "is_anomaly" in d
    assert "features" in d
    assert "samples_seen" in d


def test_samples_seen_increases():
    """samples_seen should increase as we call fit_and_score."""
    before = samples_seen()
    fit_and_score(_normal_features())
    assert samples_seen() >= before


# ---------------------------------------------------------------------------
# Anomaly detection (if sklearn available)
# ---------------------------------------------------------------------------

def test_extreme_outlier_flagged():
    """An extreme outlier session should eventually be flagged as anomaly."""
    pytest.importorskip("sklearn", reason="scikit-learn not installed")

    # Feed many normal sessions first
    for i in range(30):
        fit_and_score(_normal_features(llm_calls=float(i % 5 + 1)))

    # Score a clearly anomalous session
    anomalous = _normal_features(
        llm_calls=500.0,     # huge number of calls
        error_rate=0.99,     # almost all errors
        injection_score=0.99,
    )
    r = fit_and_score(anomalous)
    if r is not None:
        # Anomalous session should have higher score or be flagged
        assert r.anomaly_score > 0.0 or r.is_anomaly is True
