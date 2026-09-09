"""
Tests for the notification / integration settings API.

Covers:
  GET  /v1/settings         — read current effective settings
  PATCH /v1/settings        — update keys with validation
  POST  /v1/settings/test   — fire a test notification (unconfigured → ok: false)

No network or SMTP connections are made; external HTTP calls are mocked.
Run: cd backend && python -m pytest tests/test_org_settings.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-dashboard-key"}


def _make_settings_db_mock(rows: list[tuple[str, str]] | None = None) -> AsyncMock:
    """Return an AsyncMock for db.execute whose result.mappings().all() yields rows.

    ``rows`` is a list of (key, value) tuples.  Defaults to [] (no DB overrides).
    """
    row_objects = [SimpleNamespace(key=k, value=v) for k, v in (rows or [])]
    result_mock = MagicMock()
    result_mock.mappings.return_value.all.return_value = row_objects
    # Also satisfy the fetchall / scalars patterns used by other endpoints
    result_mock.fetchall.return_value = []
    result_mock.fetchone.return_value = None
    result_mock.scalars.return_value.all.return_value = []
    return AsyncMock(return_value=result_mock)


# ---------------------------------------------------------------------------
# GET /v1/settings
# ---------------------------------------------------------------------------

class TestGetSettings:
    def test_requires_auth(self):
        resp = client.get("/v1/settings")
        assert resp.status_code == 401

    def test_returns_settings_dict(self, mock_db_session):
        resp = client.get("/v1/settings", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "settings" in data
        s = data["settings"]
        # All expected keys are present
        for key in (
            "slack_webhook_url", "smtp_host", "smtp_port", "smtp_user",
            "smtp_password", "smtp_from", "smtp_to", "pagerduty_routing_key",
            "alert_webhook_url", "alert_webhook_secret",
            "alert_min_severity", "alert_cooldown_s",
            "splunk_hec_url", "splunk_hec_token",
            "elastic_url", "elastic_api_key",
        ):
            assert key in s, f"Expected key '{key}' in settings response"

    def test_defaults_applied_when_no_db_overrides(self, mock_db_session):
        """With empty DB result, env defaults should be returned."""
        mock_db_session.execute = _make_settings_db_mock()  # no DB overrides
        resp = client.get("/v1/settings", headers=HEADERS)
        assert resp.status_code == 200
        s = resp.json()["settings"]
        # alert_min_severity defaults to MEDIUM
        assert s["alert_min_severity"] == "MEDIUM"
        # alert_cooldown_s defaults to 600 (may be int or str depending on config)
        assert str(s["alert_cooldown_s"]) == "600"


# ---------------------------------------------------------------------------
# PATCH /v1/settings
# ---------------------------------------------------------------------------

class TestUpdateSettings:
    def test_requires_auth(self):
        resp = client.patch("/v1/settings", json={"updates": {"alert_min_severity": "HIGH"}})
        assert resp.status_code == 401

    def test_valid_update_returns_ok(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {"alert_min_severity": "HIGH"}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["updated_count"] == 1

    def test_multiple_updates_counted(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {
                "alert_min_severity": "CRITICAL",
                "alert_cooldown_s": "300",
            }},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["updated_count"] == 2

    def test_rejects_unknown_key(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {"malicious_key": "value"}},
            headers=HEADERS,
        )
        assert resp.status_code == 400
        assert "malicious_key" in resp.json()["detail"]

    def test_rejects_non_integer_for_smtp_port(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {"smtp_port": "not-a-number"}},
            headers=HEADERS,
        )
        assert resp.status_code == 400
        assert "smtp_port" in resp.json()["detail"]

    def test_accepts_valid_smtp_port(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {"smtp_port": "587"}},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_rejects_invalid_alert_min_severity(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {"alert_min_severity": "EXTREME"}},
            headers=HEADERS,
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "alert_min_severity" in detail

    def test_valid_severity_values_accepted(self, mock_db_session):
        for sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            resp = client.patch(
                "/v1/settings",
                json={"updates": {"alert_min_severity": sev}},
                headers=HEADERS,
            )
            assert resp.status_code == 200, f"Severity '{sev}' should be accepted"

    def test_null_value_deletes_key(self, mock_db_session):
        """Setting a key to null reverts it to the env default (DELETE from DB)."""
        resp = client.patch(
            "/v1/settings",
            json={"updates": {"slack_webhook_url": None}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        # DB DELETE should have been called (commit still called once)
        assert mock_db_session.commit.called

    def test_empty_updates_returns_ok(self, mock_db_session):
        resp = client.patch(
            "/v1/settings",
            json={"updates": {}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["updated_count"] == 0


# ---------------------------------------------------------------------------
# POST /v1/settings/test — test notification channel
# ---------------------------------------------------------------------------

class TestTestChannel:
    def test_requires_auth(self):
        resp = client.post("/v1/settings/test", json={"channel": "slack"})
        assert resp.status_code == 401

    def test_unknown_channel_returns_400(self, mock_db_session):
        resp = client.post(
            "/v1/settings/test",
            json={"channel": "telegram"},
            headers=HEADERS,
        )
        assert resp.status_code == 400
        assert "telegram" in resp.json()["detail"]

    def test_slack_unconfigured_returns_not_ok(self, mock_db_session):
        """When slack_webhook_url is not set, test returns ok=False with a message."""
        resp = client.post(
            "/v1/settings/test",
            json={"channel": "slack"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "Slack" in data["message"] or "not configured" in data["message"]

    def test_email_unconfigured_returns_not_ok(self, mock_db_session):
        resp = client.post(
            "/v1/settings/test",
            json={"channel": "email"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "SMTP" in data["message"] or "not configured" in data["message"]

    def test_pagerduty_unconfigured_returns_not_ok(self, mock_db_session):
        resp = client.post(
            "/v1/settings/test",
            json={"channel": "pagerduty"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "PagerDuty" in data["message"] or "not configured" in data["message"]

    def test_webhook_unconfigured_returns_not_ok(self, mock_db_session):
        resp = client.post(
            "/v1/settings/test",
            json={"channel": "webhook"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "Webhook" in data["message"] or "not configured" in data["message"]

    def test_slack_configured_sends_and_returns_ok(self, mock_db_session):
        """
        When slack_webhook_url is set, test POSTs to Slack and returns ok=True.
        Mocks the HTTP call so no real network access occurs.
        """
        # Make the DB return a row with the Slack URL set
        mock_db_session.execute = _make_settings_db_mock([
            ("slack_webhook_url", "https://hooks.slack.com/fake"),
        ])

        # Mock the httpx.AsyncClient POST inside org_settings.py
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_resp)

        # httpx is imported inline inside the endpoint, so patch at the httpx module level
        with patch("httpx.AsyncClient", return_value=mock_async_client):
            resp = client.post(
                "/v1/settings/test",
                json={"channel": "slack"},
                headers=HEADERS,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "sent" in data["message"].lower() or "success" in data["message"].lower()
