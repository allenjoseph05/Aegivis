"""
Tests for GET/PATCH /v1/agents/{agent_id}/security-features — Part C per-agent override resolution.

Three-level resolution: agent-level → org-level → catalog default.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-dashboard-key"}
AGENT_ID = "test-agent-1"
BASE_URL = f"/v1/agents/{AGENT_ID}/security-features"

# A known valid feature key from the catalog (first entry)
KEY_STRUCTURAL = "injection_structural_enabled"
KEY_ML = "injection_ml_async_enabled"
KEY_HITL = "hitl_enabled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(key: str, value: str) -> MagicMock:
    """Create a mock DB row with .key and .value attributes."""
    row = MagicMock()
    row.key = key
    row.value = value
    return row


def _make_execute_result(rows: list[MagicMock]) -> MagicMock:
    """Return a CursorResult mock that yields the given rows from mappings().all()."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    return result


# ---------------------------------------------------------------------------
# GET /v1/agents/{agent_id}/security-features
# ---------------------------------------------------------------------------

class TestGetAgentSecurityFeatures:
    def test_requires_auth(self):
        resp = client.get(BASE_URL)
        assert resp.status_code == 401

    def test_returns_full_catalog_with_defaults(self, mock_db_session):
        """When no overrides exist, all 26 features use catalog defaults."""
        empty = _make_execute_result([])
        mock_db_session.execute = AsyncMock(return_value=empty)

        resp = client.get(BASE_URL, headers=HEADERS)
        assert resp.status_code == 200

        data = resp.json()
        assert data["agent_id"] == AGENT_ID
        assert "features" in data
        assert len(data["features"]) == 26

    def test_all_flags_false_when_no_overrides(self, mock_db_session):
        """is_agent_override and is_org_override are both False when no overrides exist."""
        empty = _make_execute_result([])
        mock_db_session.execute = AsyncMock(return_value=empty)

        resp = client.get(BASE_URL, headers=HEADERS)
        data = resp.json()

        for feat in data["features"]:
            assert feat["is_agent_override"] is False
            assert feat["is_org_override"] is False

    def test_default_value_matches_catalog(self, mock_db_session):
        """When no overrides exist, value equals the catalog default."""
        empty = _make_execute_result([])
        mock_db_session.execute = AsyncMock(return_value=empty)

        resp = client.get(BASE_URL, headers=HEADERS)
        data = resp.json()

        for feat in data["features"]:
            assert feat["value"] == feat["default"]

    def test_agent_override_takes_priority_over_default(self, mock_db_session):
        """Agent-level override wins over catalog default."""
        org_result = _make_execute_result([])                          # no org override
        agent_result = _make_execute_result([
            _make_row(KEY_STRUCTURAL, "false"),
        ])
        mock_db_session.execute = AsyncMock(side_effect=[org_result, agent_result])

        resp = client.get(BASE_URL, headers=HEADERS)
        assert resp.status_code == 200

        data = resp.json()
        feat = next(f for f in data["features"] if f["key"] == KEY_STRUCTURAL)
        assert feat["value"] == "false"
        assert feat["is_agent_override"] is True
        assert feat["is_org_override"] is False

    def test_org_override_used_when_no_agent_override(self, mock_db_session):
        """Org override is surfaced with is_org_override=True when no agent override exists."""
        org_result = _make_execute_result([
            _make_row(KEY_ML, "false"),
        ])
        agent_result = _make_execute_result([])                        # no agent override
        mock_db_session.execute = AsyncMock(side_effect=[org_result, agent_result])

        resp = client.get(BASE_URL, headers=HEADERS)
        assert resp.status_code == 200

        data = resp.json()
        feat = next(f for f in data["features"] if f["key"] == KEY_ML)
        assert feat["value"] == "false"
        assert feat["is_agent_override"] is False
        assert feat["is_org_override"] is True

    def test_agent_override_beats_org_override(self, mock_db_session):
        """When both org and agent overrides exist for the same key, agent wins."""
        org_result = _make_execute_result([
            _make_row(KEY_HITL, "false"),
        ])
        agent_result = _make_execute_result([
            _make_row(KEY_HITL, "true"),
        ])
        mock_db_session.execute = AsyncMock(side_effect=[org_result, agent_result])

        resp = client.get(BASE_URL, headers=HEADERS)
        assert resp.status_code == 200

        data = resp.json()
        feat = next(f for f in data["features"] if f["key"] == KEY_HITL)
        assert feat["value"] == "true"
        assert feat["is_agent_override"] is True
        assert feat["is_org_override"] is False   # shadowed by agent override

    def test_org_override_does_not_affect_is_agent_override(self, mock_db_session):
        """A key with org override but no agent override has is_org_override=True, not is_agent_override."""
        org_result = _make_execute_result([
            _make_row(KEY_STRUCTURAL, "false"),
        ])
        agent_result = _make_execute_result([])
        mock_db_session.execute = AsyncMock(side_effect=[org_result, agent_result])

        resp = client.get(BASE_URL, headers=HEADERS)
        data = resp.json()

        feat = next(f for f in data["features"] if f["key"] == KEY_STRUCTURAL)
        assert feat["is_agent_override"] is False
        assert feat["is_org_override"] is True

    def test_response_envelope_contains_agent_and_org_id(self, mock_db_session):
        """Response envelope carries agent_id and org_id."""
        empty = _make_execute_result([])
        mock_db_session.execute = AsyncMock(return_value=empty)

        resp = client.get(BASE_URL, headers=HEADERS)
        data = resp.json()

        assert data["agent_id"] == AGENT_ID
        assert "org_id" in data


# ---------------------------------------------------------------------------
# PATCH /v1/agents/{agent_id}/security-features
# ---------------------------------------------------------------------------

class TestPatchAgentSecurityFeatures:
    def test_requires_auth(self):
        resp = client.patch(BASE_URL, json={"updates": {}})
        assert resp.status_code == 401

    def test_patch_valid_key_returns_updated(self, mock_db_session):
        """PATCH with a valid key returns updated: [key] and count: 1."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {KEY_STRUCTURAL: "false"}},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert KEY_STRUCTURAL in data["updated"]
        assert data["count"] == 1

    def test_patch_commits_to_db(self, mock_db_session):
        """PATCH always calls db.commit() exactly once."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {KEY_STRUCTURAL: "true"}},
        )
        assert resp.status_code == 200
        mock_db_session.commit.assert_called_once()

    def test_patch_non_empty_uses_upsert_sql(self, mock_db_session):
        """Saving a non-empty value executes INSERT ... ON CONFLICT ... DO UPDATE."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {KEY_STRUCTURAL: "true"}},
        )
        assert resp.status_code == 200

        sqls = [str(c.args[0]) for c in mock_db_session.execute.call_args_list if c.args]
        assert any("ON CONFLICT" in sql for sql in sqls)

    def test_patch_empty_string_triggers_delete(self, mock_db_session):
        """Empty string value triggers DELETE (reset to org/default) and is still in 'updated'."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {KEY_STRUCTURAL: ""}},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert KEY_STRUCTURAL in data["updated"]
        assert data["count"] == 1

        # Confirm DELETE SQL was issued (not INSERT)
        sqls = [str(c.args[0]) for c in mock_db_session.execute.call_args_list if c.args]
        assert any("DELETE" in sql for sql in sqls)
        assert not any("INSERT" in sql for sql in sqls)

    def test_patch_unknown_key_is_ignored(self, mock_db_session):
        """Unknown feature keys are silently ignored; 'updated' is empty."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {"totally_fake_key": "true"}},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["updated"] == []
        assert data["count"] == 0
        # No DB writes should have happened
        mock_db_session.execute.assert_not_called()

    def test_patch_multiple_valid_keys(self, mock_db_session):
        """Multiple valid keys are all recorded in 'updated'."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {
                KEY_STRUCTURAL: "false",
                KEY_ML: "false",
                KEY_HITL: "true",
            }},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["count"] == 3
        assert set(data["updated"]) == {KEY_STRUCTURAL, KEY_ML, KEY_HITL}

    def test_patch_mixed_valid_and_unknown_keys(self, mock_db_session):
        """Mix of valid and unknown keys — only valid ones appear in 'updated'."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {
                KEY_STRUCTURAL: "true",
                "not_a_real_feature": "yes",
            }},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert KEY_STRUCTURAL in data["updated"]
        assert "not_a_real_feature" not in data["updated"]
        assert data["count"] == 1

    def test_patch_params_forwarded_correctly(self, mock_db_session):
        """The SQL params include org_id, agent_id, key, and value."""
        resp = client.patch(
            BASE_URL,
            headers=HEADERS,
            json={"updates": {KEY_STRUCTURAL: "false"}},
        )
        assert resp.status_code == 200

        # Find the INSERT call params
        params_list = [c.args[1] for c in mock_db_session.execute.call_args_list if len(c.args) > 1]
        upsert_params = next(
            (p for p in params_list if "agent_id" in p and "INSERT" not in str(p)),
            None,
        )
        # At minimum, one call must include agent_id == AGENT_ID
        all_params = [c.args[1] for c in mock_db_session.execute.call_args_list if len(c.args) > 1]
        assert any(p.get("agent_id") == AGENT_ID for p in all_params)
