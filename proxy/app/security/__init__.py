"""
Aegivis security scanning package.

Injection detection is handled exclusively by the enforcement/ sync-path
module (enforcement.scan_messages). This package covers the remaining
security concerns that are orthogonal to injection detection:

  Tool call scanning    — RCE + SSRF on tool arguments (TOOL_CALL_START)
  Tool output scanning  — injection in tool return values (TOOL_CALL_END)
  MCP scanning          — malicious tool definitions
  Canary / spotlighting — exfiltration detection + IPI defense (forward path)
  Output scanning       — post-response canary leak / credential exposure
  Behavioral analysis   — Markov sequence + Isolation Forest (AGENT_FINISH)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .rce_scanner import RceScanResult, scan as _scan_rce
from .ssrf_scanner import SsrfScanResult, scan as _scan_ssrf

if TYPE_CHECKING:
    from .tool_output_scanner import ToolOutputScanResult
    from .mcp_scanner import McpScanResult
    from .isolation_forest import IsolationForestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool call scanning  (RCE + SSRF)
# ---------------------------------------------------------------------------

@dataclass
class ToolSecurityScanResult:
    """Combined RCE + SSRF scan result for a single tool call."""
    rce:  RceScanResult
    ssrf: SsrfScanResult

    @property
    def rce_detected(self) -> bool:
        return self.rce.detected

    @property
    def ssrf_detected(self) -> bool:
        return self.ssrf.detected

    def to_event_dict(self) -> dict:
        """Compact dict embedded in TOOL_CALL_START event ``security`` key."""
        return {
            "rce_detected":      self.rce.detected,
            "rce_confidence":    round(self.rce.confidence, 4),
            "rce_language":      self.rce.language,
            "rce_patterns":      self.rce.dangerous_patterns[:5],
            "ssrf_detected":     self.ssrf.detected,
            "ssrf_category":     self.ssrf.ssrf_category,
            "ssrf_urls_scanned": self.ssrf.urls_scanned,
            "ssrf_blocked_count": len(self.ssrf.matches),
            "ssrf_reasons": [m.reason for m in self.ssrf.matches[:3]],
        }


def scan_tool_call(
    tool_name: str,
    tool_args: dict | str,
) -> ToolSecurityScanResult:
    """Run RCE and SSRF scanners on the arguments of a single tool call."""
    rce_result  = _scan_rce(tool_name, tool_args)
    ssrf_result = _scan_ssrf(tool_args)
    return ToolSecurityScanResult(rce=rce_result, ssrf=ssrf_result)


# ---------------------------------------------------------------------------
# Tool output scanning
# ---------------------------------------------------------------------------

def scan_tool_output(
    tool_name: str,
    output: str | None,
) -> "ToolOutputScanResult":
    """Scan a tool return value for injection attempts."""
    from .tool_output_scanner import scan as _scan_to
    return _scan_to(tool_name, output)


# ---------------------------------------------------------------------------
# MCP tool definition scanning
# ---------------------------------------------------------------------------

def scan_tool_definitions(tools: list[dict]) -> "McpScanResult":
    """Scan tool definitions for MCP-specific attack vectors."""
    from .mcp_scanner import scan as _scan_mcp
    return _scan_mcp(tools)


# ---------------------------------------------------------------------------
# Behavioral analysis (Markov + Isolation Forest)
# ---------------------------------------------------------------------------

def observe_event(from_evt: str, to_evt: str, agent_id: str) -> None:
    """Record an event-type transition in the Markov model."""
    from .markov import observe_transition
    observe_transition(from_evt, to_evt, agent_id)


def score_session_end(features: dict) -> "IsolationForestResult | None":
    """Score session features via Isolation Forest at AGENT_FINISH."""
    from .isolation_forest import fit_and_score
    return fit_and_score(features)
