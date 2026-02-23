"""
Tests for the anomaly detection API endpoint and detection engine.

Run from backend/ directory:
    python -m pytest tests/test_anomalies.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.anomaly import (
    AnomalyFlag,
    detect_anomalies,
    _rule_error_rate,
    _rule_tool_call_loop,
    _rule_runaway_agent,
    _rule_long_latency,
    _rule_pii_in_tool_output,
    _rule_model_switch,
)

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-proxy-key"}


# ---------------------------------------------------------------------------
# REST API tests
# ---------------------------------------------------------------------------

class TestAnomaliesEndpoint:
    def test_requires_auth(self):
        resp = client.get("/v1/anomalies")
        assert resp.status_code == 401

    def test_invalid_api_key(self):
        resp = client.get("/v1/anomalies", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_list_anomalies_empty(self):
        """GET /v1/anomalies returns 200 with empty list when no anomalies."""
        resp = client.get("/v1/anomalies", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "anomalies" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body
        assert body["anomalies"] == []
        assert body["total"] == 0

    def test_list_anomalies_with_data(self, mock_db_session):
        """GET /v1/anomalies returns rows when DB has data."""
        from datetime import datetime, timezone

        row = MagicMock()
        row.id = 1
        row.session_id = "sess_abc"
        row.agent_id = "agent-x"
        row.org_id = "org-1"
        row.rule_id = "HIGH_ERROR_RATE"
        row.severity = "high"
        row.description = "3 errors in session"
        row.event_id = None
        row.sequence_number = None
        row.metadata = {"error_count": 3}
        row.detected_at = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)

        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row]
        mock_db_session.execute = AsyncMock(return_value=result_mock)

        resp = client.get("/v1/anomalies", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        anomaly = body["anomalies"][0]
        assert anomaly["rule_id"] == "HIGH_ERROR_RATE"
        assert anomaly["severity"] == "high"
        assert anomaly["session_id"] == "sess_abc"

    def test_filter_by_session_id(self, mock_db_session):
        """session_id query param is forwarded to the SQL filter."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_db_session.execute = AsyncMock(return_value=result_mock)

        resp = client.get("/v1/anomalies?session_id=sess_xyz", headers=HEADERS)
        assert resp.status_code == 200
        # Check that execute was called (SQL was built with filter)
        mock_db_session.execute.assert_called_once()

    def test_filter_by_severity(self, mock_db_session):
        """severity=critical query param is accepted."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_db_session.execute = AsyncMock(return_value=result_mock)

        resp = client.get("/v1/anomalies?severity=critical", headers=HEADERS)
        assert resp.status_code == 200

    def test_pagination_params(self):
        """limit and offset query params are accepted."""
        resp = client.get("/v1/anomalies?limit=10&offset=5", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 10
        assert body["offset"] == 5

    def test_limit_max_500(self):
        """limit over 500 is rejected by validation."""
        resp = client.get("/v1/anomalies?limit=999", headers=HEADERS)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Anomaly detection engine unit tests
# ---------------------------------------------------------------------------

def _make_event(event_type: str, **kwargs) -> dict:
    base = {
        "event_id": f"evt_{event_type.lower()}",
        "event_type": event_type,
        "sequence_number": 1,
        "payload": {},
        "pii_detected": [],
        "model": "gpt-4",
    }
    base.update(kwargs)
    return base


class TestAnomalyDetectionEngine:
    def test_empty_events_returns_no_flags(self):
        flags = detect_anomalies([])
        assert flags == []

    def test_high_error_rate_fires_at_three_errors(self):
        """Rule: HIGH_ERROR_RATE triggers when >= 3 SYSTEM_ERROR events."""
        events = [_make_event("SYSTEM_ERROR") for _ in range(3)]
        flags = detect_anomalies(events)
        rule_ids = [f.rule_id for f in flags]
        assert "HIGH_ERROR_RATE" in rule_ids

    def test_high_error_rate_no_fire_below_threshold(self):
        """HIGH_ERROR_RATE does NOT fire for 2 errors."""
        events = [_make_event("SYSTEM_ERROR") for _ in range(2)]
        flags = _rule_error_rate(events)
        assert flags == []

    def test_high_error_rate_severity_is_high(self):
        events = [_make_event("SYSTEM_ERROR") for _ in range(3)]
        flags = _rule_error_rate(events)
        assert flags[0].severity == "high"
        assert flags[0].rule_id == "HIGH_ERROR_RATE"

    def test_tool_call_loop_fires_above_five(self):
        """TOOL_CALL_LOOP triggers when same tool called > 5 times."""
        events = [
            _make_event("TOOL_CALL_START", payload={"tool_name": "web_search"})
            for _ in range(6)
        ]
        flags = _rule_tool_call_loop(events)
        assert len(flags) == 1
        assert flags[0].rule_id == "TOOL_CALL_LOOP"
        assert flags[0].metadata["tool_name"] == "web_search"

    def test_tool_call_loop_no_fire_at_five(self):
        """TOOL_CALL_LOOP does NOT fire at exactly 5 calls."""
        events = [
            _make_event("TOOL_CALL_START", payload={"tool_name": "search"})
            for _ in range(5)
        ]
        flags = _rule_tool_call_loop(events)
        assert flags == []

    def test_runaway_agent_fires_at_twenty_llm_calls(self):
        """RUNAWAY_AGENT triggers at 20+ LLM calls with no AGENT_FINISH."""
        events = [_make_event("LLM_CALL_START") for _ in range(20)]
        flags = _rule_runaway_agent(events)
        assert len(flags) == 1
        assert flags[0].rule_id == "RUNAWAY_AGENT"
        assert flags[0].severity == "high"

    def test_runaway_agent_no_fire_with_finish(self):
        """RUNAWAY_AGENT does NOT fire if AGENT_FINISH is present."""
        events = [_make_event("LLM_CALL_START") for _ in range(25)]
        events.append(_make_event("AGENT_FINISH"))
        flags = _rule_runaway_agent(events)
        assert flags == []

    def test_long_latency_fires_above_30s(self):
        """LONG_LATENCY triggers when LLM call took > 30,000ms."""
        events = [_make_event("LLM_CALL_END", payload={"latency_ms": 35000})]
        flags = _rule_long_latency(events)
        assert len(flags) == 1
        assert flags[0].rule_id == "LONG_LATENCY"

    def test_long_latency_no_fire_under_30s(self):
        events = [_make_event("LLM_CALL_END", payload={"latency_ms": 29000})]
        flags = _rule_long_latency(events)
        assert flags == []

    def test_pii_in_tool_output_fires_for_ssn(self):
        """SENSITIVE_PII_IN_TOOL_OUTPUT fires when SSN found in tool output."""
        events = [_make_event("TOOL_CALL_END", pii_detected=["US_SSN"])]
        flags = _rule_pii_in_tool_output(events)
        assert len(flags) == 1
        assert flags[0].rule_id == "SENSITIVE_PII_IN_TOOL_OUTPUT"
        assert flags[0].severity == "high"

    def test_pii_non_sensitive_no_fire(self):
        """Non-sensitive PII (EMAIL_ADDRESS) does not trigger the rule."""
        events = [_make_event("TOOL_CALL_END", pii_detected=["EMAIL_ADDRESS"])]
        flags = _rule_pii_in_tool_output(events)
        assert flags == []

    def test_model_switch_fires_for_multiple_models(self):
        """MODEL_SWITCH fires when events use different models."""
        events = [
            _make_event("LLM_CALL_START", model="gpt-4"),
            _make_event("LLM_CALL_START", model="gpt-3.5-turbo"),
        ]
        flags = _rule_model_switch(events)
        assert len(flags) == 1
        assert flags[0].rule_id == "MODEL_SWITCH"

    def test_chain_integrity_violation_flag(self):
        """chain_valid=False adds CHAIN_INTEGRITY critical flag."""
        events = [_make_event("LLM_CALL_START")]
        flags = detect_anomalies(events, chain_valid=False)
        rule_ids = [f.rule_id for f in flags]
        assert "CHAIN_INTEGRITY" in rule_ids
        chain_flag = next(f for f in flags if f.rule_id == "CHAIN_INTEGRITY")
        assert chain_flag.severity == "critical"

    def test_multiple_rules_fire_simultaneously(self):
        """Multiple rules can fire for the same session."""
        events = (
            [_make_event("SYSTEM_ERROR") for _ in range(3)] +
            [_make_event("LLM_CALL_START") for _ in range(20)]
        )
        flags = detect_anomalies(events)
        rule_ids = {f.rule_id for f in flags}
        assert "HIGH_ERROR_RATE" in rule_ids
        assert "RUNAWAY_AGENT" in rule_ids
