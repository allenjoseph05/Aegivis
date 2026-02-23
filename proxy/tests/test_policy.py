"""Tests for the policy enforcement engine."""
from __future__ import annotations

import pytest
from proxy.app.policy import PolicyAction, PolicyEngine, PolicyRule, PolicyViolation


def _make_engine(rules_data: list[dict]) -> PolicyEngine:
    return PolicyEngine.from_rules_list(rules_data)


def _make_event(
    event_type: str = "LLM_CALL_START",
    pii_detected: list[str] | None = None,
    payload: dict | None = None,
    **kwargs,
) -> dict:
    return {
        "event_type": event_type,
        "session_id": "sess_test",
        "agent_id": "test-agent",
        "org_id": "test-org",
        "model": "gpt-4o",
        "provider": "openai",
        "pii_detected": pii_detected or [],
        "payload": payload or {},
        "sequence_number": kwargs.get("sequence_number", 0),
    }


def _make_state(tool_call_count: int = 0, llm_call_count: int = 0, **kwargs) -> dict:
    import time
    return {
        "tool_call_count": tool_call_count,
        "llm_call_count": llm_call_count,
        "started_at_ns": kwargs.get("started_at_ns", time.time_ns()),
    }


class TestPolicyEngineBasics:
    def test_empty_rules_no_violations(self):
        engine = _make_engine([])
        result = engine.evaluate(_make_event(), _make_state())
        assert result == []

    def test_disabled_rule_not_fired(self):
        engine = _make_engine([{
            "name": "disabled-rule",
            "event_types": ["LLM_CALL_START"],
            "conditions": [],
            "action": "BLOCK",
            "reason": "should not fire",
            "enabled": False,
        }])
        result = engine.evaluate(_make_event("LLM_CALL_START"), _make_state())
        assert result == []

    def test_wrong_event_type_not_fired(self):
        engine = _make_engine([{
            "name": "tool-only-rule",
            "event_types": ["TOOL_CALL_START"],
            "conditions": [],
            "action": "ALERT",
            "reason": "tool alert",
            "enabled": True,
        }])
        result = engine.evaluate(_make_event("LLM_CALL_START"), _make_state())
        assert result == []

    def test_wildcard_event_type_fires_on_all(self):
        engine = _make_engine([{
            "name": "catch-all",
            "event_types": ["*"],
            "conditions": [],
            "action": "LOG",
            "reason": "log everything",
            "enabled": True,
        }])
        for et in ["LLM_CALL_START", "TOOL_CALL_START", "AGENT_FINISH"]:
            result = engine.evaluate(_make_event(et), _make_state())
            assert len(result) == 1


class TestConditionOperators:
    def test_gt_condition(self):
        engine = _make_engine([{
            "name": "tool-loop",
            "event_types": ["TOOL_CALL_START"],
            "conditions": [{"field": "tool_call_count", "op": "gt", "value": 50}],
            "action": "BLOCK",
            "reason": "too many tool calls",
            "enabled": True,
        }])
        # Under threshold
        result = engine.evaluate(_make_event("TOOL_CALL_START"), _make_state(tool_call_count=50))
        assert result == []
        # Over threshold
        result = engine.evaluate(_make_event("TOOL_CALL_START"), _make_state(tool_call_count=51))
        assert len(result) == 1
        assert result[0].action == PolicyAction.BLOCK

    def test_not_empty_condition_pii(self):
        engine = _make_engine([{
            "name": "pii-alert",
            "event_types": ["LLM_CALL_START"],
            "conditions": [{"field": "pii_detected", "op": "not_empty"}],
            "action": "ALERT",
            "reason": "PII found",
            "enabled": True,
        }])
        # No PII
        result = engine.evaluate(_make_event("LLM_CALL_START", pii_detected=[]), _make_state())
        assert result == []
        # With PII
        result = engine.evaluate(_make_event("LLM_CALL_START", pii_detected=["EMAIL_ADDRESS"]), _make_state())
        assert len(result) == 1
        assert result[0].action == PolicyAction.ALERT

    def test_in_condition(self):
        engine = _make_engine([{
            "name": "auth-error",
            "event_types": ["SYSTEM_ERROR"],
            "conditions": [{"field": "http_status", "op": "in", "value": [401, 403]}],
            "action": "ALERT",
            "reason": "auth error",
            "enabled": True,
        }])
        result = engine.evaluate(
            _make_event("SYSTEM_ERROR", payload={"http_status": 401}),
            _make_state(),
        )
        assert len(result) == 1

        result = engine.evaluate(
            _make_event("SYSTEM_ERROR", payload={"http_status": 500}),
            _make_state(),
        )
        assert result == []

    def test_any_in_condition_pii_types(self):
        engine = _make_engine([{
            "name": "critical-pii",
            "event_types": ["LLM_CALL_START"],
            "conditions": [{
                "field": "pii_detected",
                "op": "any_in",
                "value": ["US_SSN", "CREDIT_CARD"],
            }],
            "action": "ALERT",
            "reason": "critical PII",
            "enabled": True,
        }])
        result = engine.evaluate(
            _make_event("LLM_CALL_START", pii_detected=["EMAIL_ADDRESS", "US_SSN"]),
            _make_state(),
        )
        assert len(result) == 1

        result = engine.evaluate(
            _make_event("LLM_CALL_START", pii_detected=["EMAIL_ADDRESS"]),
            _make_state(),
        )
        assert result == []

    def test_eq_condition(self):
        engine = _make_engine([{
            "name": "model-check",
            "event_types": ["LLM_CALL_START"],
            "conditions": [{"field": "model", "op": "eq", "value": "gpt-4"}],
            "action": "ALERT",
            "reason": "legacy model",
            "enabled": True,
        }])
        event = _make_event("LLM_CALL_START")
        event["model"] = "gpt-4"
        result = engine.evaluate(event, _make_state())
        assert len(result) == 1

        event["model"] = "gpt-4o"
        result = engine.evaluate(event, _make_state())
        assert result == []

    def test_payload_field_accessor(self):
        engine = _make_engine([{
            "name": "latency-alert",
            "event_types": ["LLM_CALL_END"],
            "conditions": [{"field": "payload.latency_ms", "op": "gt", "value": 5000}],
            "action": "ALERT",
            "reason": "slow call",
            "enabled": True,
        }])
        result = engine.evaluate(
            _make_event("LLM_CALL_END", payload={"latency_ms": 6000}),
            _make_state(),
        )
        assert len(result) == 1

        result = engine.evaluate(
            _make_event("LLM_CALL_END", payload={"latency_ms": 1000}),
            _make_state(),
        )
        assert result == []


class TestPolicyActions:
    def test_block_returns_immediately(self):
        """BLOCK rule short-circuits — later ALERT rules not returned."""
        engine = _make_engine([
            {
                "name": "block-rule",
                "event_types": ["*"],
                "conditions": [],
                "action": "BLOCK",
                "reason": "blocked",
                "enabled": True,
            },
            {
                "name": "alert-rule",
                "event_types": ["*"],
                "conditions": [],
                "action": "ALERT",
                "reason": "alerted",
                "enabled": True,
            },
        ])
        result = engine.evaluate(_make_event(), _make_state())
        assert len(result) == 1
        assert result[0].action == PolicyAction.BLOCK
        assert result[0].rule_name == "block-rule"

    def test_allow_short_circuits(self):
        """ALLOW rule vetoes all subsequent violations for that event."""
        engine = _make_engine([
            {
                "name": "allow-rule",
                "event_types": ["*"],
                "conditions": [],
                "action": "ALLOW",
                "reason": "explicitly allowed",
                "enabled": True,
            },
            {
                "name": "block-rule",
                "event_types": ["*"],
                "conditions": [],
                "action": "BLOCK",
                "reason": "would be blocked",
                "enabled": True,
            },
        ])
        result = engine.evaluate(_make_event(), _make_state())
        assert result == []

    def test_multiple_alert_rules_all_returned(self):
        engine = _make_engine([
            {
                "name": "alert-1",
                "event_types": ["*"],
                "conditions": [],
                "action": "ALERT",
                "reason": "alert 1",
                "enabled": True,
            },
            {
                "name": "alert-2",
                "event_types": ["*"],
                "conditions": [],
                "action": "ALERT",
                "reason": "alert 2",
                "enabled": True,
            },
        ])
        result = engine.evaluate(_make_event(), _make_state())
        assert len(result) == 2
        rule_names = {v.rule_name for v in result}
        assert rule_names == {"alert-1", "alert-2"}

    def test_violation_fields_populated(self):
        engine = _make_engine([{
            "name": "test-rule",
            "event_types": ["LLM_CALL_START"],
            "conditions": [],
            "action": "ALERT",
            "reason": "test reason",
            "enabled": True,
        }])
        result = engine.evaluate(_make_event("LLM_CALL_START"), _make_state())
        assert len(result) == 1
        v = result[0]
        assert v.rule_name == "test-rule"
        assert v.action == PolicyAction.ALERT
        assert v.reason == "test reason"
        assert v.session_id == "sess_test"
        assert v.agent_id == "test-agent"
        assert v.timestamp_ns > 0


class TestMultipleConditions:
    def test_all_conditions_must_match(self):
        """AND logic: both conditions must be true."""
        engine = _make_engine([{
            "name": "compound-rule",
            "event_types": ["TOOL_CALL_START"],
            "conditions": [
                {"field": "tool_call_count", "op": "gt", "value": 10},
                {"field": "llm_call_count", "op": "gt", "value": 5},
            ],
            "action": "ALERT",
            "reason": "both exceeded",
            "enabled": True,
        }])

        # Both met
        result = engine.evaluate(
            _make_event("TOOL_CALL_START"),
            _make_state(tool_call_count=11, llm_call_count=6),
        )
        assert len(result) == 1

        # Only tool count met
        result = engine.evaluate(
            _make_event("TOOL_CALL_START"),
            _make_state(tool_call_count=11, llm_call_count=3),
        )
        assert result == []

        # Only llm count met
        result = engine.evaluate(
            _make_event("TOOL_CALL_START"),
            _make_state(tool_call_count=5, llm_call_count=6),
        )
        assert result == []


class TestDefaultPoliciesLoad:
    def test_default_yaml_loads_without_error(self):
        from pathlib import Path
        yaml_path = Path(__file__).parent.parent / "app" / "policies" / "default.yaml"
        assert yaml_path.exists(), f"Default policy file not found at {yaml_path}"

        try:
            engine = PolicyEngine.from_yaml(yaml_path)
            assert engine.rule_count > 0
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_rules_summary_returns_list(self):
        engine = _make_engine([{
            "name": "test",
            "event_types": ["*"],
            "conditions": [],
            "action": "LOG",
            "reason": "test",
            "enabled": True,
        }])
        summary = engine.rules_summary()
        assert isinstance(summary, list)
        assert len(summary) == 1
        assert summary[0]["name"] == "test"
