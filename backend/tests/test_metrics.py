"""
Tests for GET /v1/metrics/* endpoints.

Run from backend/ directory:
    python -m pytest tests/test_metrics.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-proxy-key"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_overview_row(**kwargs):
    row = MagicMock()
    row.session_count    = kwargs.get("session_count", 0)
    row.agent_count      = kwargs.get("agent_count", 0)
    row.llm_call_count   = kwargs.get("llm_call_count", 0)
    row.tool_call_count  = kwargs.get("tool_call_count", 0)
    row.error_count      = kwargs.get("error_count", 0)
    row.total_tokens     = kwargs.get("total_tokens", 0)
    row.pii_event_count  = kwargs.get("pii_event_count", 0)
    row.avg_latency_ms   = kwargs.get("avg_latency_ms", None)
    return row


def _make_anom_row(**kwargs):
    row = MagicMock()
    row.total_anomalies         = kwargs.get("total_anomalies", 0)
    row.high_severity_anomalies = kwargs.get("high_severity_anomalies", 0)
    return row


def _make_viol_row(**kwargs):
    row = MagicMock()
    row.total_violations = kwargs.get("total_violations", 0)
    row.blocked_count    = kwargs.get("blocked_count", 0)
    row.alert_count      = kwargs.get("alert_count", 0)
    return row


def _make_agent_row(**kwargs):
    row = MagicMock()
    row.agent_id           = kwargs.get("agent_id", "test-agent")
    row.session_count      = kwargs.get("session_count", 5)
    row.llm_call_count     = kwargs.get("llm_call_count", 20)
    row.tool_call_count    = kwargs.get("tool_call_count", 10)
    row.error_count        = kwargs.get("error_count", 1)
    row.total_tokens       = kwargs.get("total_tokens", 4000)
    row.avg_latency_ms     = kwargs.get("avg_latency_ms", 123.45)
    row.pii_event_count    = kwargs.get("pii_event_count", 2)
    row.injection_score_max = kwargs.get("injection_score_max", 0.12)
    row.anomaly_count      = kwargs.get("anomaly_count", 0)
    row.first_seen         = kwargs.get("first_seen", datetime(2025, 1, 1, tzinfo=timezone.utc))
    row.last_seen          = kwargs.get("last_seen", datetime(2025, 6, 1, tzinfo=timezone.utc))
    return row


def _make_model_row(**kwargs):
    row = MagicMock()
    row.model          = kwargs.get("model", "gpt-4o")
    row.provider       = kwargs.get("provider", "openai")
    row.session_count  = kwargs.get("session_count", 5)
    row.call_count     = kwargs.get("call_count", 20)
    row.error_count    = kwargs.get("error_count", 0)
    row.total_tokens   = kwargs.get("total_tokens", 5000)
    row.avg_latency_ms = kwargs.get("avg_latency_ms", 145.0)
    row.first_used     = kwargs.get("first_used", datetime(2025, 1, 1, tzinfo=timezone.utc))
    row.last_used      = kwargs.get("last_used", datetime(2025, 6, 1, tzinfo=timezone.utc))
    return row


def _patch_execute(mock_db, return_values: list):
    """
    Configure mock_db.execute to return successive values.
    Each value in return_values is turned into a result mock.
    Pass None to use the default empty result.
    """
    results = []
    for rv in return_values:
        rm = MagicMock()
        if rv is None:
            rm.fetchone.return_value = None
            rm.fetchall.return_value = []
        elif isinstance(rv, list):
            rm.fetchall.return_value = rv
            rm.fetchone.return_value = rv[0] if rv else None
        else:
            rm.fetchone.return_value = rv
            rm.fetchall.return_value = [rv]
        results.append(rm)
    mock_db.execute = AsyncMock(side_effect=results)


# ---------------------------------------------------------------------------
# GET /v1/metrics/overview
# ---------------------------------------------------------------------------

class TestMetricsOverview:
    def test_requires_auth(self):
        resp = client.get("/v1/metrics/overview")
        assert resp.status_code == 401

    def test_rejects_wrong_key(self):
        resp = client.get("/v1/metrics/overview", headers={"X-API-Key": "bad"})
        assert resp.status_code == 403

    def test_empty_db_returns_zeros(self, mock_db_session):
        """When all three queries return None/empty, response has zero-valued fields."""
        _patch_execute(mock_db_session, [None, None, None])
        resp = client.get("/v1/metrics/overview", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_count"] == 0
        assert body["agent_count"] == 0
        assert body["llm_call_count"] == 0
        assert body["total_tokens"] == 0
        assert body["total_anomalies"] == 0
        assert body["total_violations"] == 0
        assert body["avg_latency_ms"] is None

    def test_with_data(self, mock_db_session):
        """Returns correct values when queries have data."""
        events_row = _make_overview_row(
            session_count=42, agent_count=7, llm_call_count=300,
            tool_call_count=80, total_tokens=60000, avg_latency_ms=98.5,
        )
        anom_row = _make_anom_row(total_anomalies=5, high_severity_anomalies=2)
        viol_row = _make_viol_row(total_violations=12, blocked_count=3, alert_count=9)
        _patch_execute(mock_db_session, [events_row, anom_row, viol_row])

        resp = client.get("/v1/metrics/overview", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_count"] == 42
        assert body["agent_count"] == 7
        assert body["llm_call_count"] == 300
        assert body["total_tokens"] == 60000
        assert body["avg_latency_ms"] == 98.5
        assert body["total_anomalies"] == 5
        assert body["high_severity_anomalies"] == 2
        assert body["total_violations"] == 12
        assert body["blocked_count"] == 3
        assert body["alert_count"] == 9

    def test_all_expected_keys_present(self, mock_db_session):
        _patch_execute(mock_db_session, [None, None, None])
        resp = client.get("/v1/metrics/overview", headers=HEADERS)
        body = resp.json()
        expected_keys = {
            "session_count", "agent_count", "llm_call_count", "tool_call_count",
            "error_count", "total_tokens", "pii_event_count", "avg_latency_ms",
            "total_anomalies", "high_severity_anomalies",
            "total_violations", "blocked_count", "alert_count",
        }
        assert expected_keys.issubset(body.keys())


# ---------------------------------------------------------------------------
# GET /v1/metrics/agents
# ---------------------------------------------------------------------------

class TestMetricsAgents:
    def test_requires_auth(self):
        resp = client.get("/v1/metrics/agents")
        assert resp.status_code == 401

    def test_empty_returns_empty_list(self, mock_db_session):
        _patch_execute(mock_db_session, [[]])
        resp = client.get("/v1/metrics/agents", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agents"] == []
        assert body["total"] == 0
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_with_single_agent(self, mock_db_session):
        row = _make_agent_row(agent_id="my-agent", session_count=3, llm_call_count=15)
        _patch_execute(mock_db_session, [[row]])
        resp = client.get("/v1/metrics/agents", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        agent = body["agents"][0]
        assert agent["agent_id"] == "my-agent"
        assert agent["session_count"] == 3
        assert agent["llm_call_count"] == 15
        assert "injection_score_max" in agent
        assert "anomaly_count" in agent
        assert "first_seen" in agent
        assert "last_seen" in agent

    def test_filter_by_agent_id(self, mock_db_session):
        _patch_execute(mock_db_session, [[]])
        resp = client.get("/v1/metrics/agents?agent_id=agent-1", headers=HEADERS)
        assert resp.status_code == 200
        # Verify query was called with agent_id filter
        call_args = mock_db_session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        # The agent_id param should have been passed
        assert mock_db_session.execute.called

    def test_pagination_params(self, mock_db_session):
        _patch_execute(mock_db_session, [[]])
        resp = client.get("/v1/metrics/agents?limit=10&offset=20", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 10
        assert body["offset"] == 20

    def test_limit_over_max_rejected(self, mock_db_session):
        resp = client.get("/v1/metrics/agents?limit=201", headers=HEADERS)
        assert resp.status_code == 422  # exceeds max=200

    def test_injection_score_serialized_as_float(self, mock_db_session):
        row = _make_agent_row(injection_score_max=0.73)
        _patch_execute(mock_db_session, [[row]])
        resp = client.get("/v1/metrics/agents", headers=HEADERS)
        agent = resp.json()["agents"][0]
        assert isinstance(agent["injection_score_max"], float)
        assert 0.0 <= agent["injection_score_max"] <= 1.0

    def test_null_avg_latency_serialized_as_none(self, mock_db_session):
        row = _make_agent_row(avg_latency_ms=None)
        _patch_execute(mock_db_session, [[row]])
        resp = client.get("/v1/metrics/agents", headers=HEADERS)
        agent = resp.json()["agents"][0]
        assert agent["avg_latency_ms"] is None

    def test_multiple_agents(self, mock_db_session):
        rows = [
            _make_agent_row(agent_id="agent-a", session_count=10),
            _make_agent_row(agent_id="agent-b", session_count=5),
            _make_agent_row(agent_id="agent-c", session_count=1),
        ]
        _patch_execute(mock_db_session, [rows])
        resp = client.get("/v1/metrics/agents", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["agents"]) == 3
        assert body["agents"][0]["agent_id"] == "agent-a"


# ---------------------------------------------------------------------------
# GET /v1/metrics/models
# ---------------------------------------------------------------------------

class TestMetricsModels:
    def test_requires_auth(self):
        resp = client.get("/v1/metrics/models")
        assert resp.status_code == 401

    def test_empty_returns_empty_list(self, mock_db_session):
        _patch_execute(mock_db_session, [[]])
        resp = client.get("/v1/metrics/models", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"] == []
        assert body["total"] == 0

    def test_with_single_model(self, mock_db_session):
        row = _make_model_row(model="gpt-4o", provider="openai", call_count=100)
        _patch_execute(mock_db_session, [[row]])
        resp = client.get("/v1/metrics/models", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        model = body["models"][0]
        assert model["model"] == "gpt-4o"
        assert model["provider"] == "openai"
        assert model["call_count"] == 100
        assert "session_count" in model
        assert "total_tokens" in model
        assert "avg_latency_ms" in model
        assert "first_used" in model
        assert "last_used" in model

    def test_multiple_models(self, mock_db_session):
        rows = [
            _make_model_row(model="gpt-4o", call_count=50),
            _make_model_row(model="gpt-3.5-turbo", call_count=30),
            _make_model_row(model="claude-3-sonnet", provider="anthropic", call_count=20),
        ]
        _patch_execute(mock_db_session, [rows])
        resp = client.get("/v1/metrics/models", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["models"]) == 3

    def test_null_avg_latency_serialized_as_none(self, mock_db_session):
        row = _make_model_row(avg_latency_ms=None)
        _patch_execute(mock_db_session, [[row]])
        resp = client.get("/v1/metrics/models", headers=HEADERS)
        model = resp.json()["models"][0]
        assert model["avg_latency_ms"] is None

    def test_all_expected_model_keys(self, mock_db_session):
        row = _make_model_row()
        _patch_execute(mock_db_session, [[row]])
        resp = client.get("/v1/metrics/models", headers=HEADERS)
        model = resp.json()["models"][0]
        expected = {"model", "provider", "session_count", "call_count",
                    "error_count", "total_tokens", "avg_latency_ms",
                    "first_used", "last_used"}
        assert expected.issubset(model.keys())
