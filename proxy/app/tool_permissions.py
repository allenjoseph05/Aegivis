"""
Tool Permissions Engine for Aegivis proxy.

Evaluates per-tool, per-agent access control rules defined in a YAML file.
These rules are checked at every TOOL_CALL_START event, BEFORE the main
policy engine, giving them first-priority veto power over tool execution.

Rule schema (YAML):
  permissions:
    - name: string
      tools: ["tool_name", ...] or ["*"] for all tools
      agents: ["agent_id", ...]  # optional; omit or ["*"] for all agents
      except_tools: ["tool_name", ...]  # optional allowlist when tools=["*"]
      arg_conditions:           # optional: ALL conditions must match (AND logic)
        - field: <key>          # key inside the tool arguments dict
          op: eq|neq|gt|gte|lt|lte|contains|not_contains|
              matches_regex|empty|not_empty|in|not_in
          value: <expected>     # omit for empty/not_empty operators
      action: BLOCK | ALERT | LOG
      reason: "Human-readable explanation"
      enabled: true

Rule evaluation:
  1. Rule must be enabled.
  2. Tool name must match the `tools` list (respecting `except_tools` allowlist).
  3. Agent ID must match the `agents` list (empty or ["*"] = all agents).
  4. All `arg_conditions` must match (AND logic).
  5. If all of the above pass: rule fires.
     - BLOCK short-circuits evaluation and returns immediately.
     - ALERT and LOG accumulate into the violations list.

Condition operators mirror those in policy.py for a consistent experience.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import PolicyAction, PolicyViolation

logger = logging.getLogger(__name__)

try:
    import yaml
    _yaml_available = True
except ImportError:
    _yaml_available = False
    logger.warning(
        "PyYAML not installed. Tool permissions engine will use no rules. "
        "pip install pyyaml"
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolPermissionRule:
    """A single tool permission / access control rule."""
    name:           str
    tools:          list[str]    # ["*"] = all tools; else exact names
    agents:         list[str]    # [] or ["*"] = all agents; else exact IDs
    action:         PolicyAction
    reason:         str
    except_tools:   list[str]   = field(default_factory=list)  # allowlist when tools=["*"]
    arg_conditions: list[dict]  = field(default_factory=list)  # AND conditions on tc_args
    enabled:        bool        = True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ToolPermissionsEngine:
    """
    Evaluates tool permission rules against a single tool call.

    Instantiate via ``from_yaml()`` or ``from_rules_list()``.
    The global singleton is accessed via ``get_tool_permissions_engine()``.
    """

    def __init__(self, rules: list[ToolPermissionRule]) -> None:
        self._rules = rules

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "ToolPermissionsEngine":
        if not _yaml_available:
            return cls([])
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._parse_rules(data.get("permissions", []), source=str(path))

    @classmethod
    def from_rules_list(cls, rules_data: list[dict]) -> "ToolPermissionsEngine":
        return cls._parse_rules(rules_data, source="api")

    @classmethod
    def _parse_rules(
        cls, rules_data: list[dict], source: str
    ) -> "ToolPermissionsEngine":
        rules: list[ToolPermissionRule] = []
        for r in rules_data:
            try:
                # Normalise scalar → list for tools / agents / except_tools
                tools = r.get("tools", ["*"])
                if isinstance(tools, str):
                    tools = [tools]
                agents = r.get("agents", [])
                if isinstance(agents, str):
                    agents = [agents]
                except_tools = r.get("except_tools", [])
                if isinstance(except_tools, str):
                    except_tools = [except_tools]

                rules.append(ToolPermissionRule(
                    name=r["name"],
                    tools=tools,
                    agents=agents,
                    action=PolicyAction(r.get("action", "BLOCK").upper()),
                    reason=r.get("reason", "Tool permission violation"),
                    except_tools=except_tools,
                    arg_conditions=r.get("arg_conditions", []),
                    enabled=r.get("enabled", True),
                ))
            except Exception as e:
                logger.warning(
                    "Skipping malformed tool permission rule '%s' from %s: %s",
                    r.get("name"), source, e,
                )
        logger.info(
            "Loaded %d tool permission rules from %s (%d enabled)",
            len(rules),
            source,
            sum(1 for r in rules if r.enabled),
        )
        return cls(rules)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def enabled_rule_count(self) -> int:
        return sum(1 for r in self._rules if r.enabled)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def check(
        self,
        tool_name: str,
        agent_id: str,
        tool_args: dict | str,
        *,
        session_id: str = "",
        org_id: str = "",
    ) -> list[PolicyViolation]:
        """
        Evaluate all enabled rules against a single tool call.

        Returns:
            List of PolicyViolation objects.  Empty list = permitted.
            If a BLOCK rule fires, only that violation is returned
            (evaluation stops immediately to avoid spurious extra records).

        Args:
            tool_name:  Name of the tool the LLM is invoking.
            agent_id:   Agent making the call (from session / header).
            tool_args:  Parsed tool arguments dict (or raw JSON string).
            session_id: Session ID (embedded in the violation record).
            org_id:     Organisation ID (embedded in the violation record).
        """
        parsed_args = _normalise_args(tool_args)
        violations: list[PolicyViolation] = []

        for rule in self._rules:
            if not rule.enabled:
                continue
            if not _tool_matches(rule, tool_name):
                continue
            if not _agent_matches(rule, agent_id):
                continue
            if not _args_match(rule, parsed_args):
                continue

            violation = PolicyViolation(
                rule_name=rule.name,
                action=rule.action,
                reason=rule.reason,
                event_type="TOOL_CALL_START",
                session_id=session_id,
                agent_id=agent_id,
                org_id=org_id,
                timestamp_ns=time.time_ns(),
            )
            violations.append(violation)

            if rule.action == PolicyAction.BLOCK:
                # Short-circuit — first BLOCK ends evaluation
                return violations

        return violations

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def rules_summary(self) -> list[dict]:
        """JSON-serialisable summary of all rules (enabled and disabled)."""
        return [
            {
                "name":           r.name,
                "tools":          r.tools,
                "agents":         r.agents,
                "except_tools":   r.except_tools,
                "action":         r.action.value,
                "reason":         r.reason,
                "enabled":        r.enabled,
                "arg_conditions": r.arg_conditions,
            }
            for r in self._rules
        ]


# ---------------------------------------------------------------------------
# Matching helpers (module-level for testability)
# ---------------------------------------------------------------------------

def _normalise_args(tool_args: dict | str) -> dict:
    """Convert a raw JSON string to a dict; return as-is if already a dict."""
    if isinstance(tool_args, dict):
        return tool_args
    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _tool_matches(rule: ToolPermissionRule, tool_name: str) -> bool:
    """Return True when this rule applies to *tool_name*."""
    if "*" in rule.tools:
        # Wildcard — blocks all tools except those in the allowlist
        return tool_name not in rule.except_tools
    return tool_name in rule.tools


def _agent_matches(rule: ToolPermissionRule, agent_id: str) -> bool:
    """Return True when this rule applies to *agent_id*."""
    if not rule.agents or "*" in rule.agents:
        return True
    return agent_id in rule.agents


def _args_match(rule: ToolPermissionRule, parsed_args: dict) -> bool:
    """Return True when all arg_conditions match (AND logic)."""
    if not rule.arg_conditions:
        return True  # No conditions = always fires when tool/agent match
    return all(_eval_cond(cond, parsed_args) for cond in rule.arg_conditions)


def _eval_cond(cond: dict, args: dict) -> bool:
    """
    Evaluate a single condition dict against the tool args.

    Operators mirror policy.py ``_matches_one`` for consistency.
    """
    field_name = cond.get("field", "")
    op         = cond.get("op", "eq")
    expected   = cond.get("value")
    actual: Any = args.get(field_name) if isinstance(args, dict) else None

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
        elif op == "in":
            return actual in (expected or [])
        elif op == "not_in":
            return actual not in (expected or [])
    except Exception as e:
        logger.debug(
            "ToolPermissions condition eval error (field=%s op=%s): %s",
            field_name, op, e,
        )
    return False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_engine: ToolPermissionsEngine | None = None


def get_tool_permissions_engine() -> ToolPermissionsEngine:
    global _engine
    if _engine is None:
        _engine = _load_default()
    return _engine


def reload_tool_permissions_engine(
    rules_data: list[dict] | None = None,
    yaml_path: Path | None = None,
) -> int:
    """
    Reload the tool permissions engine.  Returns the total rule count.

    Priority:
      1. ``yaml_path`` — explicit path to a YAML file.
      2. ``rules_data`` — list of rule dicts (from API body).
      3. Neither — reload from default locations.
    """
    global _engine
    if yaml_path:
        _engine = ToolPermissionsEngine.from_yaml(yaml_path)
    elif rules_data is not None:
        _engine = ToolPermissionsEngine.from_rules_list(rules_data)
    else:
        _engine = _load_default()
    return _engine.rule_count


def _load_default() -> ToolPermissionsEngine:
    """
    Load engine from AEGIVIS_TOOL_PERMISSIONS_YAML or the bundled example file.
    All rules in the bundled example are disabled, so this is a no-op
    unless the user explicitly sets enabled: true on a rule.
    """
    from .config import settings

    if settings.tool_permissions_yaml:
        p = Path(settings.tool_permissions_yaml)
        if p.exists():
            return ToolPermissionsEngine.from_yaml(p)
        logger.warning(
            "AEGIVIS_TOOL_PERMISSIONS_YAML=%s not found; "
            "starting with no tool permission rules.",
            settings.tool_permissions_yaml,
        )
        return ToolPermissionsEngine([])

    # Auto-load the bundled example file (all rules are disabled by default)
    default = Path(__file__).parent / "policies" / "tool_permissions.yaml"
    if default.exists():
        return ToolPermissionsEngine.from_yaml(default)

    return ToolPermissionsEngine([])
