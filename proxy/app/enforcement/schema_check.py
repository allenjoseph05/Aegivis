"""
enforcement/schema_check.py — Tool argument JSON Schema validation.

Validates tool call arguments against the JSON Schema declared in the
tool definition. This is Phase 2 enforcement: the LLM is supposed to
follow the schema it was given; deviations are a signal of:

  - Prompt injection telling the LLM to use a different argument shape
  - A tool being called with argument types that shouldn't be accepted
  - additionalProperties being exploited to smuggle extra data

Design
------
This module uses Python's standard library only (no jsonschema package).
It implements a subset of JSON Schema (Draft 7) that covers the common
tool definition patterns:
  - type (string, number, integer, boolean, array, object, null)
  - required (list of required property names)
  - properties (per-property type constraints)
  - additionalProperties (false = no extra keys allowed)
  - enum (allowed value set)
  - minLength / maxLength (string constraints)
  - minimum / maximum (number constraints)
  - items (array element type)

For full JSON Schema validation, install jsonschema:
    pip install jsonschema
and it will be used automatically (more complete, ~2ms overhead).

No external dependencies required. Falls back to stdlib-only subset if
jsonschema is not installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SchemaViolation:
    """A single schema violation found in tool arguments."""
    field:       str    # dotted path (e.g. "args.command", "args.urls[0]")
    expected:    str    # what the schema requires
    actual:      str    # what was provided
    severity:    str    # "error" | "warning"


@dataclass
class SchemaCheckResult:
    """Result of validating tool arguments against the tool's declared schema."""
    valid:       bool
    violations:  list[SchemaViolation] = field(default_factory=list)
    schema_found: bool = False          # False if no schema was found for this tool


# ---------------------------------------------------------------------------
# jsonschema-based validation (preferred when available)
# ---------------------------------------------------------------------------

def _validate_jsonschema(args: dict, schema: dict) -> list[SchemaViolation]:
    """Validate using the jsonschema package (preferred path)."""
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        violations: list[SchemaViolation] = []
        for error in validator.iter_errors(args):
            path = ".".join(str(p) for p in error.absolute_path) or "(root)"
            violations.append(SchemaViolation(
                field=f"args.{path}",
                expected=error.schema.get("type", str(error.schema))[:80],
                actual=str(error.instance)[:80],
                severity="error",
            ))
        return violations
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Stdlib-only subset validator
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP = {
    "string":  str,
    "number":  (int, float),
    "integer": int,
    "boolean": bool,
    "array":   list,
    "object":  dict,
    "null":    type(None),
}


def _check_type(value: object, expected_type: str, path: str) -> list[SchemaViolation]:
    """Check that *value* matches the JSON Schema *expected_type*."""
    violations: list[SchemaViolation] = []
    py_type = _JSON_TYPE_MAP.get(expected_type)
    if py_type is None:
        return violations   # unknown type, skip
    # JSON Schema: "integer" also accepts float with no fractional part
    if expected_type == "integer" and isinstance(value, float) and value == int(value):
        return violations
    if not isinstance(value, py_type):
        violations.append(SchemaViolation(
            field=path,
            expected=f"type={expected_type}",
            actual=f"type={type(value).__name__}",
            severity="error",
        ))
    return violations


def _validate_stdlib(args: dict, schema: dict, path: str = "args") -> list[SchemaViolation]:
    """
    Stdlib-only JSON Schema validation.

    Supports: type, required, properties, additionalProperties, enum,
              minLength, maxLength, minimum, maximum, items.
    """
    violations: list[SchemaViolation] = []

    if not isinstance(args, dict):
        violations.append(SchemaViolation(
            field=path, expected="type=object",
            actual=f"type={type(args).__name__}", severity="error",
        ))
        return violations

    props_schema: dict = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    # Required properties
    for req in required:
        if req not in args:
            violations.append(SchemaViolation(
                field=f"{path}.{req}", expected="present (required)",
                actual="missing", severity="error",
            ))

    # Per-property checks
    for key, value in args.items():
        field_path = f"{path}.{key}"
        prop_schema = props_schema.get(key)

        # additionalProperties: false
        if not prop_schema and schema.get("additionalProperties") is False:
            violations.append(SchemaViolation(
                field=field_path, expected="not present (additionalProperties=false)",
                actual=f"present (value={str(value)[:40]})", severity="warning",
            ))
            continue

        if not prop_schema:
            continue

        # Type check
        expected_type = prop_schema.get("type")
        if expected_type:
            violations.extend(_check_type(value, expected_type, field_path))

        # Enum check
        enum_values = prop_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            violations.append(SchemaViolation(
                field=field_path,
                expected=f"one of {enum_values[:5]}",
                actual=str(value)[:40],
                severity="error",
            ))

        # String constraints
        if isinstance(value, str):
            min_len = prop_schema.get("minLength")
            max_len = prop_schema.get("maxLength")
            if min_len is not None and len(value) < min_len:
                violations.append(SchemaViolation(
                    field=field_path,
                    expected=f"minLength={min_len}",
                    actual=f"length={len(value)}",
                    severity="error",
                ))
            if max_len is not None and len(value) > max_len:
                violations.append(SchemaViolation(
                    field=field_path,
                    expected=f"maxLength={max_len}",
                    actual=f"length={len(value)}",
                    severity="error",
                ))

        # Number constraints
        if isinstance(value, (int, float)):
            minimum = prop_schema.get("minimum")
            maximum = prop_schema.get("maximum")
            if minimum is not None and value < minimum:
                violations.append(SchemaViolation(
                    field=field_path,
                    expected=f"minimum={minimum}",
                    actual=str(value),
                    severity="error",
                ))
            if maximum is not None and value > maximum:
                violations.append(SchemaViolation(
                    field=field_path,
                    expected=f"maximum={maximum}",
                    actual=str(value),
                    severity="error",
                ))

        # Array item type
        if isinstance(value, list) and "items" in prop_schema:
            item_schema = prop_schema["items"]
            item_type = item_schema.get("type")
            if item_type:
                for i, item in enumerate(value[:20]):   # cap at 20 items
                    violations.extend(
                        _check_type(item, item_type, f"{field_path}[{i}]")
                    )

        # Nested object
        if isinstance(value, dict) and expected_type == "object":
            violations.extend(_validate_stdlib(value, prop_schema, field_path))

    return violations


# ---------------------------------------------------------------------------
# Tool schema lookup
# ---------------------------------------------------------------------------

def _find_schema_for_tool(tool_name: str, tools: list[dict]) -> dict | None:
    """
    Find the JSON Schema for *tool_name* from the tools[] array sent to the LLM.

    Handles both OpenAI function-calling format:
        {"type": "function", "function": {"name": ..., "parameters": {...}}}
    and raw function format:
        {"name": ..., "parameters": {...}}
    """
    for tool in tools:
        # OpenAI format
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        if name == tool_name:
            schema = fn.get("parameters") or fn.get("input_schema")
            if isinstance(schema, dict):
                return schema
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_tool_args(
    tool_name: str,
    tool_args: dict | str,
    tools: list[dict],
) -> SchemaCheckResult:
    """
    Validate tool arguments against the tool's declared JSON Schema.

    Args:
        tool_name: The name of the tool being called.
        tool_args: The argument dict the LLM produced.
        tools:     The tools[] array from the LLM request (contains schemas).

    Returns:
        SchemaCheckResult.
        - schema_found=False: no schema available, result is valid by default.
        - valid=True: arguments conform to the schema.
        - valid=False: schema violations found (see .violations for details).
    """
    if not tools:
        return SchemaCheckResult(valid=True, schema_found=False)

    schema = _find_schema_for_tool(tool_name, tools)
    if schema is None:
        return SchemaCheckResult(valid=True, schema_found=False)

    if not isinstance(tool_args, dict):
        return SchemaCheckResult(
            valid=False,
            schema_found=True,
            violations=[SchemaViolation(
                field="args", expected="type=object",
                actual=f"type={type(tool_args).__name__}", severity="error",
            )],
        )

    # Prefer jsonschema if available
    try:
        import jsonschema   # noqa: F401
        violations = _validate_jsonschema(tool_args, schema)
    except ImportError:
        violations = _validate_stdlib(tool_args, schema)

    if violations:
        logger.warning(
            "[SCHEMA-CHECK] tool=%r violations=%d first=%r",
            tool_name, len(violations),
            f"{violations[0].field}: {violations[0].expected}" if violations else "",
        )

    return SchemaCheckResult(
        valid=len(violations) == 0,
        violations=violations,
        schema_found=True,
    )
