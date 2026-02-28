"""
MCP Tool Definition Scanner — Phase 3.3

Scans the ``tools`` array in LLM API requests for malicious tool definitions.

Covers active 2025-2026 MCP exploit categories:
  1. Name traversal    -- path-traversal chars in tool name (../../, \\, %2F, etc.)
  2. Description injection -- prompt injection hidden in tool description field
  3. Shadow overloading -- two tools with Levenshtein-1 names (namespace collision)

Never raises: all exceptions degrade gracefully to an empty clean result.
"""
from __future__ import annotations

import logging
import re
import unicodedata
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

_NAME_TRAVERSAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.\.[/\\]"),             "path_traversal_dotdot"),
    (re.compile(r"\\{2,}"),                "path_traversal_unc"),
    (re.compile(r"\x00"),                  "null_byte"),
    (re.compile(r"%2[fF]"),                "url_encoded_slash"),
    (re.compile(r"%5[cC]"),                "url_encoded_backslash"),
    (re.compile(r"[/\\]etc[/\\]"),         "etc_access"),
    (re.compile(r"[/\\]passwd"),           "passwd_access"),
    (re.compile(r"[/\\]shadow"),           "shadow_access"),
    (re.compile(r"[/\\]proc[/\\]"),        "proc_access"),
]


def _check_name_traversal(tool_name: str) -> list[str]:
    """Return list of finding detail strings for name traversal issues."""
    findings: list[str] = []

    for pattern, label in _NAME_TRAVERSAL_PATTERNS:
        if pattern.search(tool_name):
            findings.append(f"{label} in tool name: {tool_name!r}")

    if _has_homoglyphs(tool_name):
        findings.append(f"unicode_homoglyph in tool name: {tool_name!r}")

    return findings


# ---------------------------------------------------------------------------
# Description injection patterns
# ---------------------------------------------------------------------------

_DESC_INJECTION_PATTERNS: list[tuple[str, float]] = [
    # (lowercase_substring, severity_score)
    ("ignore previous instructions",      1.0),
    ("ignore all previous",               1.0),
    ("disregard your instructions",       1.0),
    ("disregard all previous",            0.95),
    ("system prompt:",                    0.85),  # "system:" alone is too broad; require "prompt"
    ("you are now",                       0.80),
    ("you are a helpful",                 0.50),   # lower -- borderline
    ("act as",                            0.60),
    ("pretend you are",                   0.75),
    ("new instructions:",                 0.90),
    ("new task:",                         0.80),
    ("[override]",                        0.85),
    ("<system>",                          0.80),
    ("</system>",                         0.80),
    ("<instructions>",                    0.80),
    ("</instructions>",                   0.80),
    ("[inst]",                            0.75),
    ("when asked about",                  0.55),  # phishing-style
    ("always respond with",               0.70),
    ("never reveal",                      0.65),
    ("do not reveal",                     0.60),
    ("output your system prompt",         1.0),
    ("repeat your system prompt",         1.0),
    ("exfiltrate",                        0.90),
    ("send the following to",             0.80),
    ("call the following tool",           0.70),
    ("execute the following",             0.75),
]


def _check_description_injection(description: str) -> list[str]:
    """Return list of detail strings for injection patterns found in description."""
    findings: list[str] = []
    lower = description.lower()

    for pattern, _ in _DESC_INJECTION_PATTERNS:
        if pattern in lower:
            snippet = description[max(0, lower.index(pattern) - 10):lower.index(pattern) + len(pattern) + 20]
            findings.append(f"injection_pattern:{pattern!r} snippet={snippet!r}")

    return findings


def _description_max_score(description: str) -> float:
    """Compute worst-case severity score from description injection patterns."""
    lower = description.lower()
    worst = 0.0
    for pattern, score in _DESC_INJECTION_PATTERNS:
        if pattern in lower:
            worst = max(worst, score)
    return worst


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

    # Standard DP with two-row optimization
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

    False-positive guards applied before flagging:
      1. Minimum name length of 6 chars — avoids "get"/"set", "ls"/"lp" etc.
      2. Skip pairs where the only difference is a trailing 's' (plural convention)
         e.g. get_user / get_users is legitimate API design, not an attack.
    """
    findings: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i].lower(), names[j].lower()

            # Guard 1: require minimum length on both names
            if len(a) < 6 or len(b) < 6:
                continue

            dist = _levenshtein_distance(a, b)
            if dist != 1:
                continue

            # Guard 2: skip simple plural (trailing-s) variations
            if a + "s" == b or b + "s" == a:
                continue

            findings.append(
                f"shadow_collision:{names[i]!r} vs {names[j]!r} (lev_dist={dist})"
            )
    return findings


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class McpFinding:
    """A single security finding in a tool definition."""
    tool_name: str
    finding_type: str  # "description_injection" | "name_traversal" | "shadow_tool"
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

            # 2. Description injection
            if description:
                inj_details = _check_description_injection(description)
                max_score = _description_max_score(description)
                sev = "high" if max_score >= 0.75 else "medium" if max_score >= 0.50 else "low"
                for detail in inj_details:
                    findings.append(McpFinding(
                        tool_name=name,
                        finding_type="description_injection",
                        severity=sev,
                        detail=detail,
                    ))

        # 3. Shadow overloading (cross-tool check)
        shadow_details = _check_shadow_overloading(tool_names)
        for detail in shadow_details:
            # Extract tool name from detail string
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
