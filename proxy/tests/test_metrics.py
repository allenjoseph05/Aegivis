"""Tests for proxy.app.metrics (Phase 3.4)."""
import pytest


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def test_metrics_module_imports():
    """Module must import without error regardless of prometheus-client."""
    import app.metrics as m
    assert hasattr(m, "requests_total")
    assert hasattr(m, "violations_total")
    assert hasattr(m, "security_scans_total")
    assert hasattr(m, "events_enqueued_total")
    assert hasattr(m, "llm_latency_ms")
    assert hasattr(m, "security_scan_ms")
    assert hasattr(m, "active_sessions")


def test_get_metrics_app_returns_something_or_none():
    """get_metrics_app() must return ASGI app or None, never raise."""
    from app.metrics import get_metrics_app
    result = get_metrics_app()
    assert result is None or callable(result) or hasattr(result, "__call__")


# ---------------------------------------------------------------------------
# No-op behavior when prometheus-client unavailable
# ---------------------------------------------------------------------------

def test_requests_total_noop_does_not_raise():
    from app.metrics import requests_total
    # Whether real or noop, calling labels().inc() must not raise
    try:
        requests_total.labels(provider="openai", status="200").inc()
    except Exception as e:
        pytest.fail(f"requests_total.labels().inc() raised: {e}")


def test_violations_total_noop_does_not_raise():
    from app.metrics import violations_total
    try:
        violations_total.labels(rule_name="test-rule", action="BLOCK").inc()
    except Exception as e:
        pytest.fail(f"violations_total raised: {e}")


def test_llm_latency_noop_does_not_raise():
    from app.metrics import llm_latency_ms
    try:
        llm_latency_ms.labels(provider="anthropic", model="claude-3").observe(250.0)
    except Exception as e:
        pytest.fail(f"llm_latency_ms raised: {e}")


def test_active_sessions_noop_does_not_raise():
    from app.metrics import active_sessions
    try:
        active_sessions.set(5)
    except Exception as e:
        pytest.fail(f"active_sessions raised: {e}")


def test_security_scans_total_noop():
    from app.metrics import security_scans_total
    try:
        security_scans_total.labels(scanner="injection").inc()
    except Exception as e:
        pytest.fail(f"security_scans_total raised: {e}")


# ---------------------------------------------------------------------------
# Backend metrics module
# ---------------------------------------------------------------------------

def test_backend_metrics_import():
    """Verify the backend metrics module is importable from the proxy tests."""
    # This import would fail only if there's a syntax error in the file;
    # we do a relative path import here since tests run from proxy/
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "backend_metrics",
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "backend", "app", "metrics.py"
        ),
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "events_ingested_total")
        assert hasattr(mod, "get_metrics_app")
