"""
Policy enforcement engine for AgentBlackBox proxy.

Rules are loaded from YAML files and evaluated against each event before/after
forwarding to the LLM provider. When a BLOCK action fires, the proxy returns
403 to the agent without forwarding. ALERT actions log and annotate events.

Rule schema (YAML):
  rules:
    - name: string
      event_types: ["TOOL_CALL_START"] or ["*"] for all types
      conditions:                # all conditions must match (AND logic)
        - field: tool_call_count
          op: gt
          value: 50
      action: BLOCK | ALERT | RATE_LIMIT | LOG | ALLOW
      reason: "Human-readable explanation"
      enabled: true

Available fields:
  event-level:  event_type, pii_detected, provider, model, tool_name,
                latency_ms, finish_reason, http_status, error_message,
                sequence_number, payload.<key>
  session-level: tool_call_count, llm_call_count, session_duration_ms

Available operators: eq, neq, gt, gte, lt, lte, in, not_in,
                     contains, not_contains, not_empty, empty,
                     matches_regex, any_in
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml
    _yaml_available = True
except ImportError:
    _yaml_available = False
    logger.warning("PyYAML not installed. Policy engine will use no rules. pip install pyyaml")


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ALERT = "ALERT"
    RATE_LIMIT = "RATE_LIMIT"
    LOG = "LOG"


@dataclass
class PolicyViolation:
    rule_name: str
    action: PolicyAction
    reason: str
    event_type: str
    session_id: str
    agent_id: str
    org_id: str = ""
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "action": self.action.value,
            "reason": self.reason,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "org_id": self.org_id,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class PolicyRule:
    name: str
    event_types: list[str]
    conditions: list[dict]
    action: PolicyAction
    reason: str
    enabled: bool = True


class PolicyEngine:
    """Evaluates policy rules against events and session state."""

    def __init__(self, rules: list[PolicyRule]):
        self._rules = rules
        # rate_limit_counters: (rule_name, session_id) -> list of timestamp_ns
        self._rate_limit_counters: dict[tuple[str, str], list[int]] = {}
        self._rate_limit_window_s: int = 60

    @classmethod
    def from_yaml(cls, path: Path) -> "PolicyEngine":
        if not _yaml_available:
            return cls([])
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._parse_rules(data.get("rules", []), source=str(path))

    @classmethod
    def from_rules_list(cls, rules_data: list[dict]) -> "PolicyEngine":
        return cls._parse_rules(rules_data, source="api")

    @classmethod
    def _parse_rules(cls, rules_data: list[dict], source: str) -> "PolicyEngine":
        rules = []
        for r in rules_data:
            try:
                rules.append(PolicyRule(
                    name=r["name"],
                    event_types=r.get("event_types", ["*"]),
                    conditions=r.get("conditions", []),
                    action=PolicyAction(r.get("action", "LOG").upper()),
                    reason=r.get("reason", "Policy violation"),
                    enabled=r.get("enabled", True),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed policy rule '{r.get('name')}' from {source}: {e}")
        logger.info(f"Loaded {len(rules)} policy rules from {source}")
        return cls(rules)

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def evaluate(
        self,
        event: dict,
        session_state: dict,
    ) -> list[PolicyViolation]:
        """
        Evaluate all applicable rules against the event + session state.
        Returns list of violations (may be empty).
        ALLOW rules short-circuit further evaluation for that event.
        """
        violations: list[PolicyViolation] = []
        event_type = event.get("event_type", "")
        session_id = event.get("session_id", "")
        agent_id = event.get("agent_id", "")
        org_id = event.get("org_id", "")

        for rule in self._rules:
            if not rule.enabled:
                continue
            if "*" not in rule.event_types and event_type not in rule.event_types:
                continue
            if not self._matches_all(rule.conditions, event, session_state):
                continue

            # ALLOW short-circuits — no further violations for this event
            if rule.action == PolicyAction.ALLOW:
                return []

            violation = PolicyViolation(
                rule_name=rule.name,
                action=rule.action,
                reason=rule.reason,
                event_type=event_type,
                session_id=session_id,
                agent_id=agent_id,
                org_id=org_id,
            )
            violations.append(violation)

            if rule.action == PolicyAction.BLOCK:
                # Return immediately on first BLOCK — don't evaluate remaining rules
                return violations

        return violations

    def _matches_all(
        self, conditions: list[dict], event: dict, session_state: dict
    ) -> bool:
        return all(self._matches_one(c, event, session_state) for c in conditions)

    def _get_field(self, field_name: str, event: dict, session_state: dict) -> Any:
        payload = event.get("payload", {}) or {}

        mapping: dict[str, Any] = {
            "event_type":          event.get("event_type"),
            "pii_detected":        event.get("pii_detected", []),
            "provider":            event.get("provider"),
            "model":               event.get("model"),
            "sequence_number":     event.get("sequence_number", 0),
            "agent_id":            event.get("agent_id"),
            "tool_name":           payload.get("tool_name"),
            "latency_ms":          payload.get("latency_ms"),
            "finish_reason":       payload.get("finish_reason"),
            "http_status":         payload.get("http_status"),
            "error_message":       payload.get("error_message", ""),
            "tool_call_count":     session_state.get("tool_call_count", 0),
            "llm_call_count":      session_state.get("llm_call_count", 0),
            "session_duration_ms": (
                (time.time_ns() - session_state.get("started_at_ns", time.time_ns())) / 1_000_000
                if "started_at_ns" in session_state else 0
            ),
        }

        if field_name in mapping:
            return mapping[field_name]

        # payload.<key> accessor
        if field_name.startswith("payload."):
            return payload.get(field_name[8:])

        return None

    def _matches_one(
        self, condition: dict, event: dict, session_state: dict
    ) -> bool:
        field_name = condition.get("field", "")
        op = condition.get("op", "eq")
        expected = condition.get("value")
        actual = self._get_field(field_name, event, session_state)

        try:
            if op == "eq":
                return actual == expected
            elif op == "neq":
                return actual != expected
            elif op == "gt":
                return actual is not None and float(actual) > float(expected)
            elif op == "gte":
                return actual is not None and float(actual) >= float(expected)
            elif op == "lt":
                return actual is not None and float(actual) < float(expected)
            elif op == "lte":
                return actual is not None and float(actual) <= float(expected)
            elif op == "in":
                return actual in (expected or [])
            elif op == "not_in":
                return actual not in (expected or [])
            elif op == "contains":
                return expected in str(actual or "")
            elif op == "not_contains":
                return expected not in str(actual or "")
            elif op == "not_empty":
                return bool(actual)
            elif op == "empty":
                return not bool(actual)
            elif op == "matches_regex":
                return bool(re.search(str(expected), str(actual or "")))
            elif op == "any_in":
                # actual is a list; true if any element appears in expected list
                return bool(set(actual or []) & set(expected or []))
        except Exception as e:
            logger.debug(f"Condition eval error (field={field_name} op={op}): {e}")
        return False

    def rules_summary(self) -> list[dict]:
        return [
            {
                "name": r.name,
                "event_types": r.event_types,
                "action": r.action.value,
                "reason": r.reason,
                "enabled": r.enabled,
                "conditions": r.conditions,
            }
            for r in self._rules
        ]


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    global _engine
    if _engine is None:
        _engine = _load_default()
    return _engine


def reload_policy_engine(
    rules_data: list[dict] | None = None,
    yaml_path: Path | None = None,
) -> int:
    """Reload policies from YAML or from a list of rule dicts. Returns rule count."""
    global _engine
    if yaml_path:
        _engine = PolicyEngine.from_yaml(yaml_path)
    elif rules_data is not None:
        _engine = PolicyEngine.from_rules_list(rules_data)
    else:
        _engine = _load_default()
    return _engine.rule_count


def _load_default() -> PolicyEngine:
    default = Path(__file__).parent / "policies" / "default.yaml"
    if default.exists():
        return PolicyEngine.from_yaml(default)
    logger.warning("No default policy file found at proxy/app/policies/default.yaml")
    return PolicyEngine([])
