"""Tests for proxy.app.tracing (Phase 3.4)."""
import pytest


def test_tracing_module_imports():
    """Module must import without error."""
    from app.tracing import setup_tracing, get_tracer
    assert callable(setup_tracing)
    assert callable(get_tracer)


def test_get_tracer_returns_tracer():
    """get_tracer() returns a tracer-like object (real or no-op)."""
    from app.tracing import get_tracer
    tracer = get_tracer("test")
    assert hasattr(tracer, "start_as_current_span")


def test_noop_tracer_context_manager():
    """No-op tracer must work as context manager without raising."""
    from app.tracing import _NoOpTracer
    tracer = _NoOpTracer()
    with tracer.start_as_current_span("test-span", attributes={"key": "value"}):
        pass  # must not raise


def test_setup_tracing_returns_bool():
    """setup_tracing() must return True or False, never raise."""
    from app.tracing import setup_tracing
    result = setup_tracing("test-service", "http://localhost:4317")
    assert isinstance(result, bool)


def test_noop_span_attributes():
    """No-op span methods must not raise."""
    from app.tracing import _NoOpSpan
    span = _NoOpSpan()
    span.set_attribute("key", "value")
    span.set_status("ok")
    span.record_exception(ValueError("test"))
