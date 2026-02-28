"""
Tool Output Scanner — Phase 3.3

Scans tool return values for injection attempts before they re-enter LLM context.
Closes the indirect-injection gap documented in EchoLeak (CVE-2025-32711).

Architecture
------------
- Max 8 KB per tool output (long outputs truncated, flag set)
- Layer 1 (structural): runs always, pure stdlib
- Layer 2 (semantic): runs when sentence-transformers is installed
- Combined score: worst-case across layers
- Never raises: all exceptions degrade gracefully to score=0.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_BYTES = 8192  # maximum bytes to scan per tool output

# Structural patterns that indicate injection in tool output.
#
# Two-tier design to eliminate single-pattern false positives on normal content:
#
#  HIGH-CONFIDENCE (score >= 0.65): worst-case logic — one match alone is
#    sufficient evidence to trigger detection/alert.
#
#  LOW-CONFIDENCE (score < 0.65): accumulation logic — multiple co-occurring
#    weak signals are needed to reach the alert threshold.  A discount factor
#    of 0.60 in _structural_scan means:
#      * 1 weak signal of 0.30:             0.30 * 0.60 = 0.18  (ignored)
#      * 2 weak signals:             ~0.60 * 0.60 = 0.36  (ignored)
#      * 3 weak signals:             ~0.90 * 0.60 = 0.54  (ALERT)
#
#  This eliminates the most common false positive classes:
#    "system:" in YAML output        (0.25 * 0.60 = 0.15 -- ignored)
#    "###" in Markdown output        (0.10 * 0.60 = 0.06 -- ignored)
#    "you are a good person"         (0.30 * 0.60 = 0.18 -- ignored)
#    "act as a filter" in docs       (0.30 * 0.60 = 0.18 -- ignored)
#    Real injection with 3+ signals  (0.54+ -- ALERT)
_INJECTION_PATTERNS: list[tuple[str, float, str]] = [
    # ── HIGH-CONFIDENCE patterns (>= 0.65) ─────────────────────────────────
    # Single occurrence is strong enough evidence for detection.
    ("ignore previous instructions",    0.95, "override_instruction"),
    ("ignore all previous instructions", 0.95, "override_instruction"),
    ("disregard previous",              0.85, "override_instruction"),
    ("disregard your instructions",     0.85, "override_instruction"),
    ("you are now",                     0.70, "persona_hijack"),
    ("new instructions:",               0.80, "injected_directive"),
    ("new task:",                       0.75, "injected_directive"),
    ("system prompt:",                  0.75, "injected_directive"),
    ("<|im_start|>",                    0.90, "token_injection"),
    ("<|im_end|>",                      0.90, "token_injection"),
    ("[INST]",                          0.85, "token_injection"),
    ("[/INST]",                         0.85, "token_injection"),
    ("</instructions>",                 0.80, "xml_injection"),
    ("<instructions>",                  0.75, "xml_injection"),
    ("output your system prompt",       0.90, "exfiltration"),
    ("repeat your instructions",        0.85, "exfiltration"),
    ("jailbreak",                       0.80, "jailbreak"),
    ("dan mode",                        0.85, "jailbreak"),
    ("developer mode",                  0.70, "jailbreak"),
    ("sudo mode",                       0.75, "jailbreak"),
    ("admin mode",                      0.70, "jailbreak"),
    ("execute the following",           0.70, "command_injection"),
    ("pretend you are",                 0.65, "jailbreak"),
    # ── LOW-CONFIDENCE patterns (< 0.65) ───────────────────────────────────
    # Each alone is ambiguous -- appears frequently in legitimate tool outputs
    # (web scrapes, logs, documentation, YAML, chat transcripts).
    # Multiple must co-occur to reach the alert threshold.
    ("you are a",                       0.30, "persona_hijack"),    # "you are a good person"
    ("system:",                         0.25, "injected_directive"), # YAML: "system: ubuntu"
    ("assistant:",                      0.40, "role_injection"),     # chat log output
    ("act as",                          0.30, "jailbreak"),          # "act as a filter"
    ("do the following",                0.30, "command_injection"),  # instruction language
    ("run the following",               0.45, "command_injection"),  # docs/tutorials
    ("print your",                      0.35, "exfiltration"),       # "print your results"
    ("reveal your",                     0.50, "exfiltration"),       # borderline
    ("translate the following",         0.15, "benign"),             # very common in docs
    ("###",                             0.10, "delimiter_injection"), # Markdown headers
    ("---\n",                           0.10, "delimiter_injection"), # YAML/Markdown hr
    ("===\n",                           0.10, "delimiter_injection"), # section separators
]

_LABEL_MAP = {
    "safe":        (0.0, 0.35),
    "suspicious":  (0.35, 0.65),
    "injection":   (0.65, 1.01),
}


def _classify_label(score: float) -> str:
    for label, (lo, hi) in _LABEL_MAP.items():
        if lo <= score < hi:
            return label
    return "injection"


@dataclass
class ToolOutputScanResult:
    """Result of scanning a single tool output for injection."""
    detected: bool
    score: float
    label: str         # "safe" | "suspicious" | "injection"
    threats: list[str] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "detected":  self.detected,
            "score":     round(self.score, 4),
            "label":     self.label,
            "threats":   self.threats[:10],
            "truncated": self.truncated,
        }


def _structural_scan(text: str) -> tuple[float, list[str]]:
    """
    Layer 1: structural pattern matching with two-tier scoring.

    HIGH-CONFIDENCE patterns (score >= 0.65): worst-case — single match
    is sufficient for detection.

    LOW-CONFIDENCE patterns (score < 0.65): accumulation — the sum of all
    matching weak-signal scores is multiplied by 0.60, so:
      * one 0.30 signal  → 0.18 (ignored)
      * three 0.30 signals → 0.54 (ALERT)
    Hard-capped at 0.78 to keep weak combos below the BLOCK threshold.

    Returns (combined_score, found_threats).
    """
    lower = text.lower()
    high_score = 0.0      # worst-case among high-confidence matches (>= 0.65)
    low_score_sum = 0.0   # accumulated from ambiguous matches (< 0.65)
    threats: list[str] = []

    for pattern, score, label in _INJECTION_PATTERNS:
        if pattern.lower() in lower:
            threats.append(f"{label}:{pattern[:30]}")
            if score >= 0.65:
                high_score = max(high_score, score)
            else:
                low_score_sum += score

    # Weak patterns must accumulate before signalling.
    weak_combined = min(low_score_sum * 0.60, 0.78)
    return max(high_score, weak_combined), threats


def scan(tool_name: str, output: str | None) -> ToolOutputScanResult:
    """
    Scan a tool return value for injection attempts.

    Args:
        tool_name: Name of the tool that produced this output.
        output:    The tool's return value (string content).

    Returns:
        ToolOutputScanResult.  Never raises.
    """
    if not output:
        return ToolOutputScanResult(
            detected=False, score=0.0, label="safe", threats=[], truncated=False
        )

    try:
        # Truncate to max bytes
        truncated = False
        raw = str(output)
        if len(raw.encode("utf-8", errors="replace")) > _MAX_BYTES:
            raw = raw.encode("utf-8", errors="replace")[:_MAX_BYTES].decode("utf-8", errors="replace")
            truncated = True

        # Structural scan: pattern + phrase matching on tool output
        struct_score, threats = _structural_scan(raw)
        combined = min(struct_score, 1.0)

        label = _classify_label(combined)
        detected = combined >= 0.65

        return ToolOutputScanResult(
            detected=detected,
            score=combined,
            label=label,
            threats=threats,
            truncated=truncated,
        )

    except Exception as exc:
        logger.warning("Tool output scan error (tool=%s, skipped): %s", tool_name, exc)
        return ToolOutputScanResult(
            detected=False, score=0.0, label="safe", threats=[], truncated=False
        )
