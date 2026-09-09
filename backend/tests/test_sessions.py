"""Tests for GET /v1/sessions endpoint — filter params and basic behavior."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-dashboard-key"}


class TestListSessions:
    def test_requires_auth(self):
        resp = client.get("/v1/sessions")
        assert resp.status_code == 401

    def test_invalid_api_key(self):
        resp = client.get("/v1/sessions", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code in (401, 403)

    def test_returns_empty_list_when_no_sessions(self, mock_db_session):
        """With mocked DB returning empty rows, response has sessions: []."""
        resp = client.get("/v1/sessions", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert data["sessions"] == []
        assert "count" in data
        assert "limit" in data
        assert "offset" in data

    def test_agent_id_filter_passed_to_query(self, mock_db_session):
        """
        When agent_id query param is provided, db.execute is called with
        params dict containing 'agent_id'.
        """
        resp = client.get("/v1/sessions?agent_id=finance-agent", headers=HEADERS)
        assert resp.status_code == 200

        assert mock_db_session.execute.called
        _stmt, params = mock_db_session.execute.call_args.args
        assert params.get("agent_id") == "finance-agent"

    def test_provider_filter_passed_to_query(self, mock_db_session):
        """When provider query param is provided, params dict contains 'provider'."""
        resp = client.get("/v1/sessions?provider=anthropic", headers=HEADERS)
        assert resp.status_code == 200

        _stmt, params = mock_db_session.execute.call_args.args
        assert params.get("provider") == "anthropic"

    def test_both_filters_passed_to_query(self, mock_db_session):
        """Both agent_id and provider are forwarded to the query."""
        resp = client.get(
            "/v1/sessions?agent_id=my-agent&provider=openai",
            headers=HEADERS,
        )
        assert resp.status_code == 200

        _stmt, params = mock_db_session.execute.call_args.args
        assert params.get("agent_id") == "my-agent"
        assert params.get("provider") == "openai"

    def test_no_filters_omits_agent_id_from_params(self, mock_db_session):
        """Without filters, params should not have agent_id or provider keys."""
        resp = client.get("/v1/sessions", headers=HEADERS)
        assert resp.status_code == 200

        _stmt, params = mock_db_session.execute.call_args.args
        assert "agent_id" not in params
        assert "provider" not in params

    def test_limit_and_offset_passed_to_query(self, mock_db_session):
        """limit and offset query params are forwarded."""
        resp = client.get("/v1/sessions?limit=10&offset=20", headers=HEADERS)
        assert resp.status_code == 200

        data = resp.json()
        assert data["limit"] == 10
        assert data["offset"] == 20

        _stmt, params = mock_db_session.execute.call_args.args
        assert params["limit"] == 10
        assert params["offset"] == 20

    def test_limit_max_capped_at_200(self):
        """limit > 200 returns a validation error."""
        resp = client.get("/v1/sessions?limit=500", headers=HEADERS)
        assert resp.status_code == 422

    def test_offset_cannot_be_negative(self):
        """offset < 0 returns a validation error."""
        resp = client.get("/v1/sessions?offset=-1", headers=HEADERS)
        assert resp.status_code == 422


class TestGetSessionReasoning:
    """Tests for GET /v1/sessions/{session_id}/reasoning."""

    def test_requires_auth(self):
        resp = client.get("/v1/sessions/sess_abc/reasoning")
        assert resp.status_code == 401

    def test_returns_empty_when_no_reasoning_traces(self, mock_db_session):
        """With mocked DB returning empty rows, returns empty reasoning list."""
        resp = client.get("/v1/sessions/sess_abc/reasoning", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_abc"
        assert data["reasoning"] == []
        assert data["total_calls_with_reasoning"] == 0

    def test_session_id_and_org_id_passed_to_query(self, mock_db_session):
        """session_id and org_id are forwarded to the SQL query."""
        resp = client.get("/v1/sessions/sess_test_01/reasoning", headers=HEADERS)
        assert resp.status_code == 200

        _stmt, params = mock_db_session.execute.call_args.args
        assert params["session_id"] == "sess_test_01"
        assert "org_id" in params

    def test_reasoning_list_reflects_db_rows(self, mock_db_session):
        """When DB returns rows, reasoning list contains them."""
        from unittest.mock import MagicMock

        # Build a mock row that behaves like a RowMapping
        mock_row = MagicMock()
        mock_row.keys.return_value = ["event_id", "run_id", "model", "timestamp_ns",
                                       "thinking_blocks", "block_count"]
        mock_row.__iter__ = MagicMock(return_value=iter([
            ("event_id", "ev_001"), ("run_id", "run_001"), ("model", "claude-3-5-sonnet-20241022"),
            ("timestamp_ns", 1_700_000_000_000_000_000), ("thinking_blocks", [{"thinking": "Step 1"}]),
            ("block_count", 1),
        ]))
        mock_row._mapping = {
            "event_id": "ev_001", "run_id": "run_001", "model": "claude-3-5-sonnet-20241022",
            "timestamp_ns": 1_700_000_000_000_000_000, "thinking_blocks": [{"thinking": "Step 1"}],
            "block_count": 1,
        }

        # Configure mock so mappings().all() returns the row
        result_mock = mock_db_session.execute.return_value
        mappings_mock = MagicMock()
        mappings_mock.all.return_value = [mock_row._mapping]
        result_mock.mappings.return_value = mappings_mock

        resp = client.get("/v1/sessions/sess_with_traces/reasoning", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_calls_with_reasoning"] == 1
        assert len(data["reasoning"]) == 1
        assert data["reasoning"][0]["event_id"] == "ev_001"
