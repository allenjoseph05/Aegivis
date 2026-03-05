"""
Unit tests for backend/app/services/notifications.py — Phase 9F

14 tests. No network, no SMTP, no Docker required.
Run: cd backend && python -m pytest tests/test_alerting.py -v
"""
from __future__ import annotations

import time
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_violation(
    rule_name: str = "injection-score-alert",
    action: str = "ALERT",
    reason: str = "Test reason",
    agent_id: str = "test-agent",
    session_id: str = "sess-abc123",
) -> dict:
    return {
        "rule_name":    rule_name,
        "action":       action,
        "reason":       reason,
        "agent_id":     agent_id,
        "session_id":   session_id,
        "event_type":   "LLM_CALL_START",
        "timestamp_ns": 1_700_000_000_000_000_000,
        "org_id":       "default-org",
    }


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

from app.services.notifications import _classify_severity


class TestSeverityClassification:
    def test_critical_rules_map_to_critical(self):
        v = _make_violation(rule_name="canary-token-leak", action="BLOCK")
        assert _classify_severity(v) == "CRITICAL"

    def test_data_exfiltration_is_critical(self):
        v = _make_violation(rule_name="data-exfiltration-attempt", action="BLOCK")
        assert _classify_severity(v) == "CRITICAL"

    def test_system_prompt_mutation_is_critical(self):
        v = _make_violation(rule_name="system-prompt-mutation", action="BLOCK")
        assert _classify_severity(v) == "CRITICAL"

    def test_block_action_is_high(self):
        v = _make_violation(rule_name="high-injection-score", action="BLOCK")
        assert _classify_severity(v) == "HIGH"

    def test_named_high_rule_is_high(self):
        v = _make_violation(rule_name="rce-attempt", action="ALERT")
        assert _classify_severity(v) == "HIGH"

    def test_alert_action_default_medium(self):
        v = _make_violation(rule_name="unknown-custom-rule", action="ALERT")
        assert _classify_severity(v) == "MEDIUM"

    def test_unknown_action_is_low(self):
        v = _make_violation(rule_name="some-rule", action="LOG")
        assert _classify_severity(v) == "LOW"

    def test_medium_rule_is_medium(self):
        v = _make_violation(rule_name="goal-drift-alert", action="ALERT")
        assert _classify_severity(v) == "MEDIUM"


# ---------------------------------------------------------------------------
# AlertThrottle
# ---------------------------------------------------------------------------

from app.services.notifications import AlertThrottle


class TestAlertThrottle:
    def test_first_call_allowed(self):
        throttle = AlertThrottle(cooldown_s=60.0)
        assert throttle.should_send("rule-a", "sess-1") is True

    def test_second_call_within_cooldown_suppressed(self):
        throttle = AlertThrottle(cooldown_s=60.0)
        throttle.should_send("rule-a", "sess-1")   # first — allowed
        assert throttle.should_send("rule-a", "sess-1") is False

    def test_different_sessions_not_throttled(self):
        throttle = AlertThrottle(cooldown_s=60.0)
        throttle.should_send("rule-a", "sess-1")
        # Different session → not throttled
        assert throttle.should_send("rule-a", "sess-2") is True

    def test_different_rules_not_throttled(self):
        throttle = AlertThrottle(cooldown_s=60.0)
        throttle.should_send("rule-a", "sess-1")
        assert throttle.should_send("rule-b", "sess-1") is True

    def test_after_cooldown_allowed_again(self):
        throttle = AlertThrottle(cooldown_s=0.01)   # 10ms cooldown
        throttle.should_send("rule-a", "sess-1")
        time.sleep(0.02)
        assert throttle.should_send("rule-a", "sess-1") is True

    def test_evict_old_removes_stale_entries(self):
        throttle = AlertThrottle(cooldown_s=1.0)
        # Force-add a stale entry by manipulating internal cache
        import time as _t
        throttle._cache["stale:key"] = type("E", (), {
            "last_sent_ts": _t.monotonic() - 9999,
            "send_count": 1,
        })()
        evicted = throttle.evict_old(max_age_s=60.0)
        assert evicted >= 1
        assert "stale:key" not in throttle._cache


# ---------------------------------------------------------------------------
# Slack Block Kit formatting
# ---------------------------------------------------------------------------

from app.services.notifications import _slack_blocks


class TestSlackBlockKit:
    def test_block_kit_has_attachments(self):
        v = _make_violation(rule_name="canary-token-leak", action="BLOCK")
        payload = _slack_blocks(v, "CRITICAL")
        assert "attachments" in payload
        assert len(payload["attachments"]) == 1

    def test_block_kit_color_critical_is_red(self):
        v = _make_violation()
        payload = _slack_blocks(v, "CRITICAL")
        assert payload["attachments"][0]["color"] == "#FF0000"

    def test_block_kit_contains_rule_name(self):
        v = _make_violation(rule_name="data-exfiltration-attempt")
        payload = _slack_blocks(v, "CRITICAL")
        attachment_text = str(payload)
        assert "data-exfiltration-attempt" in attachment_text


# ---------------------------------------------------------------------------
# PagerDuty payload
# ---------------------------------------------------------------------------

from app.services.notifications import _pagerduty_payload


class TestPagerDutyPayload:
    def test_payload_has_required_keys(self):
        v = _make_violation(rule_name="canary-token-leak", action="BLOCK")
        # Temporarily set routing key on settings mock
        import unittest.mock as mock
        with mock.patch("app.services.notifications.settings") as s:
            s.pagerduty_routing_key = "test-key-abc"
            payload = _pagerduty_payload(v, "CRITICAL")

        assert payload["event_action"] == "trigger"
        assert "dedup_key" in payload
        assert "payload" in payload
        assert payload["payload"]["severity"] == "critical"

    def test_dedup_key_is_stable_within_hour(self):
        v = _make_violation(rule_name="rce-attempt", session_id="sess-xyz")
        import unittest.mock as mock
        with mock.patch("app.services.notifications.settings") as s:
            s.pagerduty_routing_key = "k"
            p1 = _pagerduty_payload(v, "HIGH")
            p2 = _pagerduty_payload(v, "HIGH")
        assert p1["dedup_key"] == p2["dedup_key"]
