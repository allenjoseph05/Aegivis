"""Tests for policy violations ingestion endpoint."""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.connection import get_session

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-proxy-key"}


def _make_violation(rule: str = "test-rule") -> dict:
    return {
        "rule_name": rule,
        "action": "ALERT",
        "reason": "Test policy violation",
        "event_type": "TOOL_CALL_START",
        "session_id": "sess_test",
        "agent_id": "test-agent",
        "org_id": "test-org",
        "timestamp_ns": time.time_ns(),
    }


class TestViolationsEndpoint:
    def test_requires_auth(self):
        resp = client.post(
            "/v1/violations",
            json={"violations": [], "sent_at_ns": time.time_ns()},
        )
        assert resp.status_code == 401

    def test_invalid_api_key(self):
        resp = client.post(
            "/v1/violations",
            json={"violations": [], "sent_at_ns": time.time_ns()},
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 403

    def test_empty_batch_accepted(self):
        resp = client.post(
            "/v1/violations",
            json={"violations": [], "sent_at_ns": time.time_ns()},
            headers=HEADERS,
        )
        assert resp.status_code == 202
        assert resp.json()["accepted"] == 0

    def test_valid_violation_accepted(self):
        resp = client.post(
            "/v1/violations",
            json={
                "violations": [_make_violation()],
                "sent_at_ns": time.time_ns(),
            },
            headers=HEADERS,
        )
        assert resp.status_code == 202
        assert resp.json()["accepted"] == 1

    def test_multiple_violations(self):
        violations = [_make_violation(f"rule-{i}") for i in range(5)]
        resp = client.post(
            "/v1/violations",
            json={"violations": violations, "sent_at_ns": time.time_ns()},
            headers=HEADERS,
        )
        assert resp.status_code == 202
        assert resp.json()["accepted"] == 5

    def test_list_violations_requires_auth(self):
        resp = client.get("/v1/violations")
        assert resp.status_code == 401

    def test_list_violations_returns_structure(self):
        resp = client.get("/v1/violations", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "violations" in body
        assert "total" in body
        assert "limit" in body

    def test_violations_summary_requires_auth(self):
        resp = client.get("/v1/violations/summary")
        assert resp.status_code == 401

    def test_violations_summary_returns_structure(self):
        resp = client.get("/v1/violations/summary", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "summary" in body

    def test_all_inserts_fail_returns_503(self):
        """When every DB insert fails, endpoint must return 503 (not 202 with accepted=0)."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB connection lost"))
        mock_db.rollback = AsyncMock()

        async def _failing_session():
            yield mock_db

        app.dependency_overrides[get_session] = _failing_session
        try:
            resp = client.post(
                "/v1/violations",
                json={"violations": [_make_violation()], "sent_at_ns": time.time_ns()},
                headers=HEADERS,
            )
        finally:
            app.dependency_overrides.pop(get_session, None)

        assert resp.status_code == 503
