"""
Tests for Phase 3.1 Iteration 3 — Tool Permissions Engine.

Test structure
--------------
TestToolMatching        — tools list, wildcard, except_tools
TestAgentMatching       — agents list, wildcard, all-agents (empty)
TestArgConditions       — all operators, nested args, no conditions
TestEngineCheck         — end-to-end rule evaluation + short-circuit
TestDisabledRules       — disabled rules are skipped
TestMultipleRules       — ALERT accumulation, BLOCK short-circuit
TestArgConditionOps     — full operator coverage (mirrors policy.py)
TestNormaliseArgs       — JSON string → dict normalisation
TestViolationFields     — violation record has correct fields
TestEngineFactory       — from_rules_list parsing, from_yaml path
TestRulesSummary        — summary output shape
TestIntegrationWithPolicy — engine returns PolicyViolation objects
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from proxy.app.tool_permissions import (
    ToolPermissionsEngine,
    ToolPermissionRule,
    _agent_matches,
    _args_match,
    _eval_cond,
    _normalise_args,
    _tool_matches,
)
from proxy.app.policy import PolicyAction, PolicyViolation


# ===========================================================================
# Helpers
# ===========================================================================

def _make_engine(*rules: dict) -> ToolPermissionsEngine:
    """Create an engine from a list of raw rule dicts."""
    return ToolPermissionsEngine.from_rules_list(list(rules))


def _make_rule(
    *,
    name: str = "test-rule",
    tools: list[str] | None = None,
    agents: list[str] | None = None,
    except_tools: list[str] | None = None,
    arg_conditions: list[dict] | None = None,
    action: str = "BLOCK",
    reason: str = "test",
    enabled: bool = True,
) -> ToolPermissionRule:
    return ToolPermissionRule(
        name=name,
        tools=tools or ["*"],
        agents=agents or [],
        action=PolicyAction(action),
        reason=reason,
        except_tools=except_tools or [],
        arg_conditions=arg_conditions or [],
        enabled=enabled,
    )


# ===========================================================================
# Tool matching
# ===========================================================================

class TestToolMatching:
    def test_exact_match_in_list(self):
        rule = _make_rule(tools=["bash", "sh"])
        assert _tool_matches(rule, "bash")
        assert _tool_matches(rule, "sh")

    def test_exact_match_not_in_list(self):
        rule = _make_rule(tools=["bash", "sh"])
        assert not _tool_matches(rule, "python3")

    def test_wildcard_matches_everything(self):
        rule = _make_rule(tools=["*"])
        assert _tool_matches(rule, "any_tool_name")
        assert _tool_matches(rule, "read_file")
        assert _tool_matches(rule, "delete_file")

    def test_wildcard_with_except_tools_allows_exceptions(self):
        rule = _make_rule(tools=["*"], except_tools=["read_file", "search_web"])
        assert not _tool_matches(rule, "read_file")   # allowed through
        assert not _tool_matches(rule, "search_web")  # allowed through
        assert _tool_matches(rule, "delete_file")     # blocked

    def test_wildcard_except_tools_non_matching(self):
        rule = _make_rule(tools=["*"], except_tools=["safe_tool"])
        assert _tool_matches(rule, "dangerous_tool")

    def test_empty_tools_list_matches_nothing(self):
        # An explicitly empty tools list is a no-op rule (matches nothing).
        # Omitting tools in YAML defaults to ["*"] via _parse_rules.
        rule = ToolPermissionRule(
            name="no-tools", tools=[], agents=[],
            action=PolicyAction.BLOCK, reason="test",
        )
        assert not _tool_matches(rule, "any_tool")
        assert not _tool_matches(rule, "bash")

    def test_case_sensitive_tool_name(self):
        rule = _make_rule(tools=["Bash"])
        assert _tool_matches(rule, "Bash")
        assert not _tool_matches(rule, "bash")  # case-sensitive


# ===========================================================================
# Agent matching
# ===========================================================================

class TestAgentMatching:
    def test_empty_agents_matches_all(self):
        rule = _make_rule(agents=[])
        assert _agent_matches(rule, "any-agent")
        assert _agent_matches(rule, "other-agent")

    def test_wildcard_agents_matches_all(self):
        rule = _make_rule(agents=["*"])
        assert _agent_matches(rule, "any-agent")

    def test_exact_agent_match(self):
        rule = _make_rule(agents=["trusted-agent"])
        assert _agent_matches(rule, "trusted-agent")
        assert not _agent_matches(rule, "other-agent")

    def test_multiple_agents_in_list(self):
        rule = _make_rule(agents=["agent-a", "agent-b"])
        assert _agent_matches(rule, "agent-a")
        assert _agent_matches(rule, "agent-b")
        assert not _agent_matches(rule, "agent-c")

    def test_unknown_agent_with_specific_list(self):
        rule = _make_rule(agents=["known-agent"])
        assert not _agent_matches(rule, "unknown-agent")


# ===========================================================================
# Argument conditions
# ===========================================================================

class TestArgConditions:
    def test_no_conditions_always_fires(self):
        rule = _make_rule(arg_conditions=[])
        assert _args_match(rule, {})
        assert _args_match(rule, {"path": "/tmp/file"})

    def test_eq_condition_matches(self):
        rule = _make_rule(arg_conditions=[
            {"field": "mode", "op": "eq", "value": "write"}
        ])
        assert _args_match(rule, {"mode": "write"})
        assert not _args_match(rule, {"mode": "read"})

    def test_contains_condition(self):
        rule = _make_rule(arg_conditions=[
            {"field": "path", "op": "contains", "value": ".."}
        ])
        assert _args_match(rule, {"path": "/home/user/../etc/passwd"})
        assert not _args_match(rule, {"path": "/home/user/file.txt"})

    def test_matches_regex_condition(self):
        rule = _make_rule(arg_conditions=[
            {"field": "path", "op": "matches_regex", "value": "^/(etc|bin|sys)"}
        ])
        assert _args_match(rule, {"path": "/etc/passwd"})
        assert _args_match(rule, {"path": "/bin/sh"})
        assert not _args_match(rule, {"path": "/home/user/file"})

    def test_not_empty_condition(self):
        rule = _make_rule(arg_conditions=[
            {"field": "url", "op": "not_empty"}
        ])
        assert _args_match(rule, {"url": "http://example.com"})
        assert not _args_match(rule, {"url": ""})
        assert not _args_match(rule, {})

    def test_empty_condition(self):
        rule = _make_rule(arg_conditions=[
            {"field": "payload", "op": "empty"}
        ])
        assert _args_match(rule, {"payload": ""})
        assert _args_match(rule, {})  # missing key = None = falsy
        assert not _args_match(rule, {"payload": "something"})

    def test_and_logic_all_must_match(self):
        rule = _make_rule(arg_conditions=[
            {"field": "mode", "op": "eq", "value": "write"},
            {"field": "path", "op": "contains", "value": ".."},
        ])
        # Both conditions must match
        assert _args_match(rule, {"mode": "write", "path": "../etc/passwd"})
        # Only one matches — rule should NOT fire
        assert not _args_match(rule, {"mode": "read", "path": "../etc/passwd"})
        assert not _args_match(rule, {"mode": "write", "path": "/safe/path"})

    def test_missing_field_treated_as_none(self):
        rule = _make_rule(arg_conditions=[
            {"field": "missing_key", "op": "eq", "value": "expected"}
        ])
        # Actual = None, expected = "expected" → no match
        assert not _args_match(rule, {})


# ===========================================================================
# Operator coverage
# ===========================================================================

class TestArgConditionOps:
    def _check(self, op: str, field_val, expected):
        return _eval_cond(
            {"field": "f", "op": op, "value": expected},
            {"f": field_val},
        )

    def test_eq(self):
        assert self._check("eq", "hello", "hello")
        assert not self._check("eq", "hello", "world")

    def test_neq(self):
        assert self._check("neq", "hello", "world")
        assert not self._check("neq", "hello", "hello")

    def test_gt(self):
        assert self._check("gt", 10, 5)
        assert not self._check("gt", 5, 10)

    def test_gte(self):
        assert self._check("gte", 10, 10)
        assert not self._check("gte", 9, 10)

    def test_lt(self):
        assert self._check("lt", 3, 10)
        assert not self._check("lt", 10, 3)

    def test_lte(self):
        assert self._check("lte", 10, 10)
        assert not self._check("lte", 11, 10)

    def test_contains(self):
        assert self._check("contains", "hello world", "world")
        assert not self._check("contains", "hello world", "xyz")

    def test_not_contains(self):
        assert self._check("not_contains", "hello world", "xyz")
        assert not self._check("not_contains", "hello world", "world")

    def test_not_empty(self):
        assert _eval_cond({"field": "f", "op": "not_empty"}, {"f": "value"})
        assert not _eval_cond({"field": "f", "op": "not_empty"}, {"f": ""})

    def test_empty(self):
        assert _eval_cond({"field": "f", "op": "empty"}, {"f": ""})
        assert not _eval_cond({"field": "f", "op": "empty"}, {"f": "something"})

    def test_matches_regex(self):
        assert self._check("matches_regex", "/etc/passwd", "^/etc")
        assert not self._check("matches_regex", "/home/user", "^/etc")

    def test_in_list(self):
        assert self._check("in", "banana", ["apple", "banana", "cherry"])
        assert not self._check("in", "durian", ["apple", "banana", "cherry"])

    def test_not_in_list(self):
        assert self._check("not_in", "durian", ["apple", "banana"])
        assert not self._check("not_in", "apple", ["apple", "banana"])

    def test_invalid_op_returns_false(self):
        result = _eval_cond({"field": "f", "op": "unknown_op", "value": "x"}, {"f": "x"})
        assert result is False


# ===========================================================================
# Args normalisation
# ===========================================================================

class TestNormaliseArgs:
    def test_dict_returned_unchanged(self):
        d = {"key": "value"}
        assert _normalise_args(d) is d

    def test_json_string_parsed_to_dict(self):
        s = '{"cmd": "ls", "path": "/tmp"}'
        result = _normalise_args(s)
        assert result == {"cmd": "ls", "path": "/tmp"}

    def test_invalid_json_returns_empty(self):
        assert _normalise_args("not json at all") == {}

    def test_json_array_returns_empty(self):
        # JSON arrays are not dicts — return empty
        assert _normalise_args("[1, 2, 3]") == {}

    def test_empty_string_returns_empty(self):
        assert _normalise_args("") == {}

    def test_none_like_returns_empty(self):
        assert _normalise_args({}) == {}


# ===========================================================================
# Engine.check — end-to-end
# ===========================================================================

class TestEngineCheck:
    def test_empty_engine_no_violations(self):
        engine = ToolPermissionsEngine([])
        violations = engine.check("any_tool", "any_agent", {})
        assert violations == []

    def test_block_exact_tool_name(self):
        engine = _make_engine({
            "name": "block-bash",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "No shell",
        })
        violations = engine.check("bash", "agent1", {})
        assert len(violations) == 1
        assert violations[0].action == PolicyAction.BLOCK
        assert violations[0].rule_name == "block-bash"

    def test_tool_not_in_list_no_violation(self):
        engine = _make_engine({
            "name": "block-bash",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "No shell",
        })
        assert engine.check("python3", "agent1", {}) == []

    def test_wildcard_blocks_all(self):
        engine = _make_engine({
            "name": "block-all",
            "tools": ["*"],
            "action": "BLOCK",
            "reason": "No tools",
        })
        assert len(engine.check("any_tool", "agent1", {})) == 1
        assert len(engine.check("another_tool", "agent1", {})) == 1

    def test_wildcard_with_except_tools_allows(self):
        engine = _make_engine({
            "name": "restrict-agent",
            "tools": ["*"],
            "except_tools": ["read_file", "search_web"],
            "action": "BLOCK",
            "reason": "Restricted",
        })
        assert engine.check("read_file", "agent1", {}) == []
        assert engine.check("search_web", "agent1", {}) == []
        assert len(engine.check("delete_file", "agent1", {})) == 1

    def test_agent_specific_rule_fires_only_for_that_agent(self):
        engine = _make_engine({
            "name": "restrict-agent-x",
            "tools": ["*"],
            "agents": ["agent-x"],
            "action": "BLOCK",
            "reason": "Agent X restricted",
        })
        assert len(engine.check("any_tool", "agent-x", {})) == 1
        assert engine.check("any_tool", "agent-y", {}) == []

    def test_multiple_agents_in_rule(self):
        engine = _make_engine({
            "name": "restrict-agents",
            "tools": ["dangerous_tool"],
            "agents": ["agent-a", "agent-b"],
            "action": "BLOCK",
            "reason": "Restricted",
        })
        assert len(engine.check("dangerous_tool", "agent-a", {})) == 1
        assert len(engine.check("dangerous_tool", "agent-b", {})) == 1
        assert engine.check("dangerous_tool", "agent-c", {}) == []

    def test_arg_condition_triggers_rule(self):
        engine = _make_engine({
            "name": "path-traversal",
            "tools": ["read_file"],
            "arg_conditions": [
                {"field": "path", "op": "contains", "value": ".."}
            ],
            "action": "BLOCK",
            "reason": "Path traversal",
        })
        assert len(engine.check("read_file", "agent1", {"path": "../etc/passwd"})) == 1
        assert engine.check("read_file", "agent1", {"path": "/home/user/file"}) == []

    def test_arg_condition_not_met_no_violation(self):
        engine = _make_engine({
            "name": "dangerous-mode",
            "tools": ["file_op"],
            "arg_conditions": [
                {"field": "mode", "op": "eq", "value": "delete"}
            ],
            "action": "BLOCK",
            "reason": "Delete forbidden",
        })
        assert engine.check("file_op", "agent1", {"mode": "read"}) == []

    def test_string_args_normalised_before_condition_check(self):
        engine = _make_engine({
            "name": "mode-check",
            "tools": ["write_file"],
            "arg_conditions": [
                {"field": "path", "op": "contains", "value": ".."}
            ],
            "action": "BLOCK",
            "reason": "Traversal in JSON string args",
        })
        # Raw JSON string
        args_str = json.dumps({"path": "../secret/file"})
        assert len(engine.check("write_file", "agent1", args_str)) == 1

    def test_block_violation_has_correct_session_info(self):
        engine = _make_engine({
            "name": "block-bash",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "No shell",
        })
        violations = engine.check(
            "bash", "my-agent", {},
            session_id="sess-abc", org_id="org-1",
        )
        assert violations[0].session_id == "sess-abc"
        assert violations[0].agent_id == "my-agent"
        assert violations[0].org_id == "org-1"
        assert violations[0].event_type == "TOOL_CALL_START"


# ===========================================================================
# Disabled rules
# ===========================================================================

class TestDisabledRules:
    def test_disabled_rule_never_fires(self):
        engine = _make_engine({
            "name": "disabled-block",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "Disabled",
            "enabled": False,
        })
        assert engine.check("bash", "agent1", {}) == []

    def test_disabled_rule_in_mixed_set(self):
        engine = _make_engine(
            {
                "name": "disabled",
                "tools": ["bash"],
                "action": "BLOCK",
                "reason": "Disabled",
                "enabled": False,
            },
            {
                "name": "active",
                "tools": ["sh"],
                "action": "BLOCK",
                "reason": "Active",
                "enabled": True,
            },
        )
        assert engine.check("bash", "agent1", {}) == []   # disabled
        assert len(engine.check("sh", "agent1", {})) == 1  # active


# ===========================================================================
# Multiple rules — accumulation and short-circuit
# ===========================================================================

class TestMultipleRules:
    def test_alert_rules_accumulate(self):
        """Multiple ALERT rules can fire for the same tool call."""
        engine = _make_engine(
            {"name": "alert-1", "tools": ["tool"], "action": "ALERT", "reason": "a1"},
            {"name": "alert-2", "tools": ["tool"], "action": "ALERT", "reason": "a2"},
        )
        violations = engine.check("tool", "agent1", {})
        assert len(violations) == 2
        rule_names = {v.rule_name for v in violations}
        assert rule_names == {"alert-1", "alert-2"}

    def test_first_block_short_circuits(self):
        """After the first BLOCK, no further rules should be evaluated."""
        engine = _make_engine(
            {"name": "block-first", "tools": ["tool"], "action": "BLOCK", "reason": "b"},
            {"name": "alert-after", "tools": ["tool"], "action": "ALERT", "reason": "a"},
        )
        violations = engine.check("tool", "agent1", {})
        # Only the BLOCK is returned — the ALERT rule never fires
        assert len(violations) == 1
        assert violations[0].rule_name == "block-first"
        assert violations[0].action == PolicyAction.BLOCK

    def test_alert_then_block_both_recorded(self):
        """An ALERT before a BLOCK accumulates before the BLOCK short-circuits."""
        engine = _make_engine(
            {"name": "alert-first", "tools": ["tool"], "action": "ALERT", "reason": "a"},
            {"name": "block-second", "tools": ["tool"], "action": "BLOCK", "reason": "b"},
        )
        violations = engine.check("tool", "agent1", {})
        assert len(violations) == 2
        assert violations[0].rule_name == "alert-first"
        assert violations[0].action == PolicyAction.ALERT
        assert violations[1].rule_name == "block-second"
        assert violations[1].action == PolicyAction.BLOCK

    def test_log_action_does_not_short_circuit(self):
        engine = _make_engine(
            {"name": "log-rule", "tools": ["tool"], "action": "LOG", "reason": "log"},
            {"name": "alert-rule", "tools": ["tool"], "action": "ALERT", "reason": "a"},
        )
        violations = engine.check("tool", "agent1", {})
        assert len(violations) == 2


# ===========================================================================
# Engine factory methods
# ===========================================================================

class TestEngineFactory:
    def test_from_rules_list_parses_correctly(self):
        engine = ToolPermissionsEngine.from_rules_list([
            {
                "name": "test-rule",
                "tools": ["bash", "sh"],
                "agents": ["agent-x"],
                "except_tools": [],
                "action": "BLOCK",
                "reason": "No shell",
                "enabled": True,
            }
        ])
        assert engine.rule_count == 1
        assert engine.enabled_rule_count == 1

    def test_from_rules_list_scalar_tools_normalised(self):
        """tools as a string should be converted to a list."""
        engine = ToolPermissionsEngine.from_rules_list([
            {"name": "r", "tools": "bash", "action": "BLOCK", "reason": "r"}
        ])
        assert engine.check("bash", "a", {})  # should fire

    def test_from_rules_list_malformed_rule_skipped(self):
        """Rules without a 'name' key should be skipped gracefully."""
        engine = ToolPermissionsEngine.from_rules_list([
            {"tools": ["bash"], "action": "BLOCK", "reason": "no name"},
            {"name": "valid", "tools": ["sh"], "action": "BLOCK", "reason": "ok"},
        ])
        # Malformed rule skipped, valid rule loaded
        assert engine.rule_count == 1

    def test_empty_rules_list(self):
        engine = ToolPermissionsEngine.from_rules_list([])
        assert engine.rule_count == 0
        assert engine.check("any_tool", "agent", {}) == []

    def test_from_yaml_loads_bundled_file(self):
        """The bundled tool_permissions.yaml should load without errors."""
        bundled = Path(__file__).parent.parent / "app" / "policies" / "tool_permissions.yaml"
        if not bundled.exists():
            pytest.skip("Bundled tool_permissions.yaml not found")
        engine = ToolPermissionsEngine.from_yaml(bundled)
        # All rules are disabled by default — no violations
        assert engine.check("any_tool", "any_agent", {}) == []

    def test_from_yaml_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ToolPermissionsEngine.from_yaml(Path("/nonexistent/path.yaml"))


# ===========================================================================
# Rule count properties
# ===========================================================================

class TestRuleCountProperties:
    def test_rule_count_includes_disabled(self):
        engine = _make_engine(
            {"name": "enabled", "tools": ["t"], "action": "BLOCK", "reason": "r", "enabled": True},
            {"name": "disabled", "tools": ["t"], "action": "BLOCK", "reason": "r", "enabled": False},
        )
        assert engine.rule_count == 2

    def test_enabled_rule_count_excludes_disabled(self):
        engine = _make_engine(
            {"name": "enabled", "tools": ["t"], "action": "BLOCK", "reason": "r", "enabled": True},
            {"name": "disabled", "tools": ["t"], "action": "BLOCK", "reason": "r", "enabled": False},
        )
        assert engine.enabled_rule_count == 1

    def test_all_disabled_enabled_count_is_zero(self):
        engine = _make_engine(
            {"name": "r1", "tools": ["t"], "action": "BLOCK", "reason": "r", "enabled": False},
            {"name": "r2", "tools": ["t"], "action": "BLOCK", "reason": "r", "enabled": False},
        )
        assert engine.enabled_rule_count == 0
        assert engine.rule_count == 2


# ===========================================================================
# Rules summary
# ===========================================================================

class TestRulesSummary:
    def test_summary_includes_all_rules(self):
        engine = _make_engine(
            {"name": "rule-a", "tools": ["bash"], "action": "BLOCK", "reason": "r"},
            {"name": "rule-b", "tools": ["sh"], "action": "ALERT", "reason": "r"},
        )
        summary = engine.rules_summary()
        assert len(summary) == 2
        names = {r["name"] for r in summary}
        assert names == {"rule-a", "rule-b"}

    def test_summary_fields_present(self):
        engine = _make_engine({
            "name": "test",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "test reason",
        })
        s = engine.rules_summary()[0]
        for key in ("name", "tools", "agents", "except_tools",
                    "action", "reason", "enabled", "arg_conditions"):
            assert key in s, f"Missing key: {key}"

    def test_summary_action_is_string(self):
        engine = _make_engine({
            "name": "test",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "r",
        })
        s = engine.rules_summary()[0]
        assert s["action"] == "BLOCK"  # string, not enum


# ===========================================================================
# PolicyViolation fields
# ===========================================================================

class TestViolationFields:
    def test_violation_is_policy_violation_instance(self):
        engine = _make_engine({
            "name": "block-bash",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "test",
        })
        violations = engine.check("bash", "agent1", {},
                                  session_id="s1", org_id="o1")
        assert isinstance(violations[0], PolicyViolation)

    def test_violation_to_dict_serialisable(self):
        engine = _make_engine({
            "name": "block-bash",
            "tools": ["bash"],
            "action": "BLOCK",
            "reason": "no shell",
        })
        v = engine.check("bash", "a", {}, session_id="s", org_id="o")[0]
        d = v.to_dict()
        assert d["rule_name"] == "block-bash"
        assert d["action"] == "BLOCK"
        assert d["event_type"] == "TOOL_CALL_START"
        assert d["session_id"] == "s"
        assert d["agent_id"] == "a"
        assert d["org_id"] == "o"
        assert isinstance(d["timestamp_ns"], int)

    def test_alert_violation_action_is_alert(self):
        engine = _make_engine({
            "name": "alert-on-send",
            "tools": ["send_email"],
            "action": "ALERT",
            "reason": "log outbound",
        })
        violations = engine.check("send_email", "agent1", {})
        assert violations[0].action == PolicyAction.ALERT
