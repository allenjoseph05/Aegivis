"""
Optional SDK integration tests — require a running Aegivis stack.

Run:
    docker compose up -d
    cd sdk && python -m pytest tests/test_integration.py -m integration -v

Or with custom endpoints:
    AEGIVIS_BACKEND_URL=http://localhost:8000 \
    AEGIVIS_API_KEY=dev-dashboard-key \
    python -m pytest tests/test_integration.py -m integration -v

These tests are excluded from the default unit-test run.  They verify that
the SDK's fire-and-forget event emission actually reaches the backend.
"""
from __future__ import annotations

import os
import time

import pytest

BACKEND_URL = os.getenv("AEGIVIS_BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("AEGIVIS_API_KEY", "dev-dashboard-key")
AGENT_ID = "sdk-integration-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_events(
    agent_id: str,
    event_type: str,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.25,
) -> list[dict]:
    """Poll the backend until matching events appear or timeout expires."""
    import httpx

    deadline = time.monotonic() + timeout
    client = httpx.Client(
        base_url=BACKEND_URL,
        headers={"X-API-Key": API_KEY},
        timeout=10,
    )
    with client:
        while time.monotonic() < deadline:
            resp = client.get(
                "/v1/events",
                params={"agent_id": agent_id, "event_type": event_type, "limit": 10},
            )
            if resp.status_code == 200:
                events = resp.json().get("events", [])
                if events:
                    return events
            time.sleep(poll_interval)
    return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend_available() -> bool:
    """Skip all tests in this module if the backend is not reachable."""
    import httpx

    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test 1 — instrument() emits TOOL_EXEC_START + TOOL_EXEC_END events
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_instrument_emits_tool_events(backend_available: bool):
    """
    Wrapping a plain function with instrument() and calling it should produce
    TOOL_EXEC_START and TOOL_EXEC_END events in the backend within 5 seconds.
    """
    if not backend_available:
        pytest.skip("Aegivis backend not reachable — run docker compose up -d first")

    from aegivis import instrument

    # Wrap a simple function.
    @instrument.tool(agent_id=AGENT_ID)
    def fetch_weather(city: str) -> str:
        return f"Sunny in {city}"

    # Call the tool — events are fired in a background daemon thread.
    result = fetch_weather("London")
    assert result == "Sunny in London"

    # Allow the fire-and-forget thread to complete.
    time.sleep(0.5)

    # Verify TOOL_EXEC_START arrived.
    start_events = _wait_for_events(AGENT_ID, "TOOL_EXEC_START")
    assert start_events, (
        "No TOOL_EXEC_START events found for agent_id=sdk-integration-test. "
        "Check that the backend is running and /v1/ingest is reachable."
    )
    assert any(
        e.get("payload", {}).get("tool_name") == "fetch_weather"
        for e in start_events
    ), f"Expected tool_name='fetch_weather' in {start_events}"

    # Verify TOOL_EXEC_END arrived.
    end_events = _wait_for_events(AGENT_ID, "TOOL_EXEC_END")
    assert end_events, "No TOOL_EXEC_END events found"
    assert any(
        e.get("payload", {}).get("tool_name") == "fetch_weather"
        for e in end_events
    ), f"Expected tool_name='fetch_weather' in {end_events}"


# ---------------------------------------------------------------------------
# Test 2 — instrument() emits TOOL_EXEC_ERROR on exception
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_instrument_emits_error_event(backend_available: bool):
    """
    When a wrapped tool raises an exception, TOOL_EXEC_ERROR must be emitted.
    """
    if not backend_available:
        pytest.skip("Aegivis backend not reachable — run docker compose up -d first")

    from aegivis import instrument

    ERROR_AGENT = f"{AGENT_ID}-error"

    @instrument.tool(agent_id=ERROR_AGENT)
    def failing_tool() -> str:
        raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        failing_tool()

    time.sleep(0.5)

    error_events = _wait_for_events(ERROR_AGENT, "TOOL_EXEC_ERROR")
    assert error_events, "No TOOL_EXEC_ERROR events found"
    assert any(
        e.get("payload", {}).get("error_type") == "RuntimeError"
        for e in error_events
    ), f"Expected error_type='RuntimeError' in {error_events}"


# ---------------------------------------------------------------------------
# Test 3 — MemoryEventReporter delivers MEMORY_WRITE_SCANNED event
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_memory_reporter_delivers_event(backend_available: bool):
    """
    MemoryEventReporter.report() should deliver a MEMORY_WRITE_SCANNED event
    to the backend.
    """
    if not backend_available:
        pytest.skip("Aegivis backend not reachable — run docker compose up -d first")

    from aegivis.client import MemoryEventReporter

    REPORTER_AGENT = f"{AGENT_ID}-reporter"

    reporter = MemoryEventReporter(backend_url=BACKEND_URL, api_key=API_KEY)
    reporter.report(
        event_type="MEMORY_WRITE_SCANNED",
        text_preview="The capital of France is Paris.",
        score=0.05,
        matched_phrases=[],
        agent_id=REPORTER_AGENT,
        session_id="sdk-integ-sess-001",
    )

    time.sleep(0.5)

    events = _wait_for_events(REPORTER_AGENT, "MEMORY_WRITE_SCANNED")
    assert events, "No MEMORY_WRITE_SCANNED events found"
    assert events[0].get("payload", {}).get("score") == pytest.approx(0.05, abs=0.01)
