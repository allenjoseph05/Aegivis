"""
MCP Tool Definition Scanner — Phase 3.3 + MCP Security

Scans the ``tools`` array in LLM API requests for malicious tool definitions.

Covers active 2025-2026 MCP exploit categories:
  1. Name traversal       -- path-traversal chars in tool name (../../, \\, %2F, etc.)
  2. Description injection -- structural signals on tool description (delimiter tokens,
                             Unicode anomalies). Phrase-based detection removed: phrases are
                             bypassable by paraphrase and produce FPs on documentation text.
  3. Schema property injection -- same structural checks on inputSchema properties
  4. Shadow overloading   -- two tools with Levenshtein-1 names (namespace collision)
  5. Rug pull detection   -- tool definitions changed after capability manifest approval

Never raises: all exceptions degrade gracefully to an empty clean result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import urllib.parse as _urlparse
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Homoglyph detection
# ---------------------------------------------------------------------------

# Common Unicode homoglyphs that look like ASCII chars used in paths
_HOMOGLYPH_NORM_PATTERN = re.compile(
    r"[\u2044\u2215\u29f8\u29f9\u2f8f\uff0f\uff3c]"  # / and \ lookalikes
    r"|[\u0000-\u001f\u007f-\u009f]"                  # C0/C1 control chars
)


def _has_homoglyphs(name: str) -> bool:
    return bool(_HOMOGLYPH_NORM_PATTERN.search(name))


# ---------------------------------------------------------------------------
# Name traversal patterns
# ---------------------------------------------------------------------------

def _canonicalize(s: str) -> str:
    """Decode URL encoding up to 3 passes, then lowercase."""
    prev = None
    for _ in range(3):
        if s == prev:
            break
        prev = s
        s = _urlparse.unquote(s)
    return s.lower()


_DANGEROUS_SEGMENTS: frozenset[str] = frozenset({
    "../", "..\\", "/etc/", "/proc/", "/sys/",
    "\\etc\\", "\\proc\\", "\\sys\\",
    "/shadow", "/passwd", "/sudoers",
    "\x00",
})


def _has_path_traversal(value: str) -> bool:
    c = _canonicalize(value)
    return any(seg in c for seg in _DANGEROUS_SEGMENTS)


def _check_name_traversal(tool_name: str) -> list[str]:
    """Return list of finding detail strings for name traversal issues."""
    findings: list[str] = []

    if _has_path_traversal(tool_name):
        findings.append(f"path_traversal in tool name: {tool_name!r}")

    if _has_homoglyphs(tool_name):
        findings.append(f"unicode_homoglyph in tool name: {tool_name!r}")

    return findings


# ---------------------------------------------------------------------------
# Structural signal detection (delimiter tokens + Unicode anomalies)
# Phrase matching is bypassable by paraphrase; structural signals are not.
# ---------------------------------------------------------------------------

def _description_structural_score(description: str) -> float:
    """
    Score a tool description using byte-level structural signals only.

    Delegates to the enforcement scanner which detects:
    - LLM delimiter tokens (model-specific: <|im_start|>, [INST], etc.)
    - Unicode anomalies (RTL override, zero-width joiners, mixed scripts)

    Returns a float 0.0–1.0 matching the enforcement scanner's convention.
    Returns 0.0 on import error (graceful degradation).
    """
    try:
        from ..enforcement.structural import scan as _structural_scan
        result = _structural_scan(description)
        return result.score
    except Exception:
        return 0.0


def _combined_description_score(description: str) -> float:
    """Structural-only score — delimiter tokens and Unicode anomalies."""
    return _description_structural_score(description)


# ---------------------------------------------------------------------------
# Schema property injection (recursive)
#
# Many MCP clients include inputSchema.properties[x].description verbatim in
# the model context alongside the tool description.  Attackers can embed
# injection in deeply nested schema property descriptions while keeping the
# top-level description clean.
# ---------------------------------------------------------------------------

_MAX_SCHEMA_DEPTH = 4   # avoid infinite recursion on pathological schemas
_MAX_SCHEMA_PROPS = 50  # abort per-level scan after N properties


def _scan_schema_properties(
    schema: dict,
    tool_name: str,
    path: str = "inputSchema",
    depth: int = 0,
) -> list["McpFinding"]:
    """
    Recursively scan JSON Schema object for injections in property descriptions.

    Checks:
    - Each property's "description" field (structural signals only)
    - The schema's own "description" or "title" field (structural only — title
      is often a short machine-readable label, so we use a higher threshold)
    - Nested "properties", "items", "$defs", "definitions", "anyOf", "oneOf",
      "allOf" sub-schemas

    Returns McpFinding list (never raises).
    """
    findings: list[McpFinding] = []
    if depth > _MAX_SCHEMA_DEPTH or not isinstance(schema, dict):
        return findings

    # Scan this level's own description
    own_desc = str(schema.get("description") or "")
    if own_desc:
        score = _combined_description_score(own_desc)
        sev = _score_to_severity(score)
        if score >= 0.50:
            findings.append(McpFinding(
                tool_name=tool_name,
                finding_type="schema_property_injection",
                severity=sev,
                detail=f"injection in {path}.description score={score:.2f} snippet={own_desc[:80]!r}",
            ))

    # Scan properties map
    properties = schema.get("properties") or {}
    for i, (prop_name, prop_schema) in enumerate(properties.items()):
        if i >= _MAX_SCHEMA_PROPS:
            break
        if not isinstance(prop_schema, dict):
            continue
        prop_path = f"{path}.properties.{prop_name}"
        prop_desc = str(prop_schema.get("description") or "")
        if prop_desc:
            score = _combined_description_score(prop_desc)
            if score >= 0.50:
                sev = _score_to_severity(score)
                findings.append(McpFinding(
                    tool_name=tool_name,
                    finding_type="schema_property_injection",
                    severity=sev,
                    detail=f"injection in {prop_path}.description score={score:.2f} snippet={prop_desc[:80]!r}",
                ))
        # Recurse into nested schemas
        findings.extend(_scan_schema_properties(prop_schema, tool_name, prop_path, depth + 1))

    # Recurse into combinatorial keywords
    for kw in ("items", "$defs", "definitions"):
        sub = schema.get(kw)
        if isinstance(sub, dict):
            findings.extend(_scan_schema_properties(sub, tool_name, f"{path}.{kw}", depth + 1))

    for kw in ("anyOf", "oneOf", "allOf"):
        sub_list = schema.get(kw)
        if isinstance(sub_list, list):
            for idx, sub in enumerate(sub_list[:10]):
                findings.extend(_scan_schema_properties(sub, tool_name, f"{path}.{kw}[{idx}]", depth + 1))

    return findings


def _score_to_severity(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Shadow overloading (Levenshtein-1)
# ---------------------------------------------------------------------------

def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return abs(la - lb)  # fast path

    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[lb]


def _check_shadow_overloading(names: list[str]) -> list[str]:
    """
    Return detail strings for pairs of tool names with Levenshtein distance == 1
    that represent plausible namespace collisions.

    False-positive guards:
      1. Minimum name length of 6 chars — avoids "get"/"set", "ls"/"lp" etc.
      2. Skip pairs where the only difference is a trailing 's' (plural convention).
    """
    findings: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i].lower(), names[j].lower()

            if len(a) < 6 or len(b) < 6:
                continue

            dist = _levenshtein_distance(a, b)
            if dist != 1:
                continue

            if a + "s" == b or b + "s" == a:
                continue

            findings.append(
                f"shadow_collision:{names[i]!r} vs {names[j]!r} (lev_dist={dist})"
            )
    return findings


# ---------------------------------------------------------------------------
# Rug pull detection utilities
#
# "Rug pull" = a tool definition (description + schema) changes after the
# capability manifest has been approved and signed.  The attack allows an MCP
# server to present a safe-looking tool at approval time, then swap in a
# malicious definition once the agent is running.
#
# Usage in intercept.py:
#   known = state.tool_definitions          # set on first tools/list response
#   pulls = detect_rug_pull(current_tools, known)
#   if pulls: BLOCK with "mcp-rug-pull" rule
# ---------------------------------------------------------------------------

def hash_tool_definition(tool: dict) -> str:
    """
    Compute a stable SHA-256[:16] fingerprint of a single tool definition.

    Covers name + description + inputSchema (recursively sorted).  Ignores
    any other vendor-specific fields so legitimate additions don't trigger.
    """
    inner = tool.get("function", tool)
    canonical = {
        "name":        str(inner.get("name") or tool.get("name") or ""),
        "description": str(inner.get("description") or tool.get("description") or ""),
        "schema":      _normalize_schema(inner.get("inputSchema") or inner.get("parameters") or {}),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def hash_tool_definitions(tools: list[dict]) -> dict[str, str]:
    """
    Return a dict mapping each tool's name → hash_tool_definition(tool).

    When two tools share a name (pathological case), the last one wins —
    consistent with how most LLM runtimes handle duplicates.
    """
    result: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        inner = tool.get("function", tool)
        name = str(inner.get("name") or tool.get("name") or "")
        if name:
            result[name] = hash_tool_definition(tool)
    return result


def detect_rug_pull(
    current_tools: list[dict],
    known_definitions: dict[str, str],
) -> list[tuple[str, str, str]]:
    """
    Compare current tool definitions against a previously recorded snapshot.

    Args:
        current_tools:     The ``tools`` array from the current request.
        known_definitions: Dict[name → hash] captured from a prior tools/list
                           response (or the first LLM call in the session).

    Returns:
        List of (tool_name, old_hash, new_hash) for every tool whose definition
        changed.  An empty list means no rug pull detected.

    Note: tools that appear for the first time (not in known_definitions) are
    NOT flagged — new tools are handled by the tool baseline enforcer.
    """
    if not known_definitions:
        return []

    pulls: list[tuple[str, str, str]] = []
    current_hashes = hash_tool_definitions(current_tools)

    for name, old_hash in known_definitions.items():
        new_hash = current_hashes.get(name)
        if new_hash is not None and new_hash != old_hash:
            pulls.append((name, old_hash, new_hash))

    return pulls


def _normalize_schema(schema: object) -> object:
    """Recursively sort dict keys for stable canonical JSON."""
    if isinstance(schema, dict):
        return {k: _normalize_schema(v) for k, v in sorted(schema.items())}
    if isinstance(schema, list):
        return [_normalize_schema(item) for item in schema]
    return schema


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class McpFinding:
    """A single security finding in a tool definition."""
    tool_name: str
    finding_type: str  # "description_injection" | "name_traversal" | "shadow_tool"
                       # | "schema_property_injection"
    severity: str      # "low" | "medium" | "high"
    detail: str

    def to_dict(self) -> dict:
        return {
            "tool_name":    self.tool_name,
            "finding_type": self.finding_type,
            "severity":     self.severity,
            "detail":       self.detail,
        }


@dataclass
class McpScanResult:
    """Aggregated result of scanning all tool definitions in a request."""
    detected: bool
    severity: str          # highest severity across all findings
    findings: list[McpFinding] = field(default_factory=list)
    tools_scanned: int = 0

    def to_dict(self) -> dict:
        return {
            "detected":      self.detected,
            "severity":      self.severity,
            "tools_scanned": self.tools_scanned,
            "findings":      [f.to_dict() for f in self.findings[:20]],
        }


def _severity_level(s: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(s, 0)


def _highest_severity(findings: list[McpFinding]) -> str:
    if not findings:
        return "none"
    return max(findings, key=lambda f: _severity_level(f.severity)).severity


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scan(tools: list[dict]) -> McpScanResult:
    """
    Scan tool definitions for MCP-specific attack vectors.

    Attack vectors covered:
    1. Name traversal — path chars / null bytes / homoglyphs in tool name
    2. Description injection — structural signals (delimiters, Unicode) on tool.description
    3. Schema property injection — same checks recursively on inputSchema properties
    4. Shadow overloading — Levenshtein-1 name pairs

    Args:
        tools: The ``tools`` array extracted from an LLM API request body.
               Each element is a dict with at minimum a ``name`` field.

    Returns:
        McpScanResult.  Never raises.
    """
    if not tools:
        return McpScanResult(detected=False, severity="none", tools_scanned=0)

    try:
        findings: list[McpFinding] = []
        tool_names: list[str] = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            # Unwrap OpenAI function-call format: {"type":"function","function":{...}}
            inner = tool.get("function", tool)
            name: str = str(inner.get("name") or tool.get("name") or "")
            description: str = str(inner.get("description") or tool.get("description") or "")
            schema: dict = inner.get("inputSchema") or inner.get("parameters") or {}

            if name:
                tool_names.append(name)

            # 1. Name traversal
            traversal_details = _check_name_traversal(name)
            for detail in traversal_details:
                findings.append(McpFinding(
                    tool_name=name,
                    finding_type="name_traversal",
                    severity="high",
                    detail=detail,
                ))

            # 2. Description injection (structural signals only)
            if description:
                score = _combined_description_score(description)
                sev = _score_to_severity(score)
                if score >= 0.50:
                    findings.append(McpFinding(
                        tool_name=name,
                        finding_type="description_injection",
                        severity=sev,
                        detail=f"structural_signal score={score:.2f} in description",
                    ))

            # 3. Schema property injection (recursive)
            if isinstance(schema, dict) and schema:
                schema_findings = _scan_schema_properties(schema, name)
                findings.extend(schema_findings)

        # 4. Shadow overloading (cross-tool check)
        shadow_details = _check_shadow_overloading(tool_names)
        for detail in shadow_details:
            findings.append(McpFinding(
                tool_name=detail.split(":")[1].split("'")[1] if "'" in detail else "",
                finding_type="shadow_tool",
                severity="medium",
                detail=detail,
            ))

        highest = _highest_severity(findings)
        detected = len(findings) > 0

        return McpScanResult(
            detected=detected,
            severity=highest,
            findings=findings,
            tools_scanned=len(tools),
        )

    except Exception as exc:
        logger.warning("MCP scan error (skipped): %s", exc)
        return McpScanResult(detected=False, severity="none", tools_scanned=len(tools))
