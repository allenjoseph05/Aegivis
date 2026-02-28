"""
enforcement/content_policy.py — Tool argument content policy enforcement.

This module is the SYNC PATH scanner for TOOL_CALL_START events.
It checks tool arguments for:

  1. Remote Code Execution (RCE) patterns
     Python AST analysis (stdlib), shell structural analysis,
     SQL injection patterns. Optional: bashlex + sqlglot for deeper parsing.

  2. Server-Side Request Forgery (SSRF) patterns
     RFC 1918 private IP ranges, cloud metadata endpoints (169.254.169.254),
     non-standard IP formats (hex/octal/decimal), optional DNS resolution.

No ML dependencies. All detection is deterministic. Target latency: <5ms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Re-export from existing security/ implementations.
# These modules are already pure-stdlib + optional bashlex/sqlglot.
# They will be migrated into enforcement/ in a later cleanup phase.
from ..security.rce_scanner import (   # noqa: F401
    RceScanResult,
    scan as scan_rce,
)
from ..security.ssrf_scanner import (  # noqa: F401
    SsrfScanResult,
    scan as scan_ssrf,
)


# ---------------------------------------------------------------------------
# Combined result type
# ---------------------------------------------------------------------------

@dataclass
class ToolScanResult:
    """
    Combined RCE + SSRF scan result for a single tool call.

    Matches the shape that intercept.py already reads from
    security/__init__.py's ToolSecurityScanResult, making it a
    drop-in replacement when intercept.py is migrated.
    """
    rce:          RceScanResult
    ssrf:         SsrfScanResult
    rce_detected:  bool = False
    ssrf_detected: bool = False

    def to_event_dict(self) -> dict:
        return {
            "rce_detected":  self.rce_detected,
            "rce_confidence": self.rce.confidence,
            "rce_language":  self.rce.language,
            "rce_patterns":  self.rce.dangerous_patterns,
            "ssrf_detected": self.ssrf_detected,
            "ssrf_category": self.ssrf.ssrf_category,
            "ssrf_urls_scanned": self.ssrf.urls_scanned,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_tool_call(tool_name: str, tool_args: dict | str) -> ToolScanResult:
    """
    Scan tool call arguments for RCE and SSRF threats.

    Args:
        tool_name: The LLM-chosen tool name (used for safe-context dampening).
        tool_args: The argument dict (or raw string if JSON parsing failed).

    Returns:
        ToolScanResult with both RCE and SSRF findings.
        rce_detected / ssrf_detected are True when confidence exceeds thresholds.
    """
    rce_result  = scan_rce(tool_name, tool_args)
    ssrf_result = scan_ssrf(tool_args)

    return ToolScanResult(
        rce=rce_result,
        ssrf=ssrf_result,
        rce_detected=rce_result.detected,
        ssrf_detected=ssrf_result.detected,
    )
