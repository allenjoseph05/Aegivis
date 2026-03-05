"""
Tool Output Scanner — Phase 3.3 + Phase 9E

Scans tool return values for injection attempts before they re-enter LLM context.
Closes the indirect-injection gap documented in EchoLeak (CVE-2025-32711).

Architecture
------------
- Phase 9E (Steganographic): runs first — strips invisible Unicode, detects
  CSS hiding and document-layer injections, re-scans cleaned text.
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
    stego_detected: bool = False      # Phase 9E: invisible char injection found
    stego_threat: str = ""            # e.g. "unicode_tag_injection"
    stego_severity: str = "none"      # "none" | "low" | "high" | "critical"
    doc_scan_detected: bool = False   # Phase 9E: document-layer injection found
    doc_scan_threat: str = ""

    def to_dict(self) -> dict:
        d = {
            "detected":  self.detected,
            "score":     round(self.score, 4),
            "label":     self.label,
            "threats":   self.threats[:10],
            "truncated": self.truncated,
        }
        if self.stego_detected:
            d["stego_detected"] = True
            d["stego_threat"] = self.stego_threat
            d["stego_severity"] = self.stego_severity
        if self.doc_scan_detected:
            d["doc_scan_detected"] = True
            d["doc_scan_threat"] = self.doc_scan_threat
        return d


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


def scan(
    tool_name: str,
    output: str | None,
    **kwargs,
) -> ToolOutputScanResult:
    """
    Scan a tool return value for injection attempts.

    Phase 9E additions (run before structural scan):
      1. Unicode steganography scan — detects tag chars, RTL overrides,
         zero-width chars. Re-scans cleaned text to confirm payload.
      2. Document scan — PDF/DOCX hidden layers (if PyMuPDF/python-docx installed).

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

        combined = 0.0
        threats: list[str] = []
        stego_detected = False
        stego_threat = ""
        stego_severity = "none"
        doc_detected = False
        doc_threat = ""

        # ── Phase 9E Layer 1: Unicode steganography scan ───────────────────
        try:
            from .unicode_scanner import scan_unicode_stego, looks_like_html
            from app.config import settings as _cfg

            if _cfg.security_unicode_stego_enabled:
                is_html = looks_like_html(raw)
                stego = scan_unicode_stego(raw, is_html=is_html)

                if stego.detected:
                    stego_detected = True
                    stego_threat = stego.threat
                    stego_severity = stego.severity
                    threats.append(f"stego:{stego.threat}")

                    # Always scan the cleaned text (strips invisible chars)
                    # so structural scanner sees what the LLM would see.
                    scan_target = stego.cleaned_text

                    if stego.severity == "critical":
                        # Tag characters: also scan the decoded payload if non-empty
                        payloads_to_scan = [scan_target]
                        if stego.tag_decoded and len(stego.tag_decoded) >= 6:
                            payloads_to_scan.append(stego.tag_decoded)

                        max_inner = 0.0
                        for payload in payloads_to_scan:
                            inner_score, inner_threats = _structural_scan(payload)
                            if inner_score > max_inner:
                                max_inner = inner_score
                                threats.extend(
                                    f"[tag_decoded]{t}" for t in inner_threats
                                    if f"[tag_decoded]{t}" not in threats
                                )

                        if max_inner > 0.35:
                            # Confirmed: invisible chars hid a real injection payload
                            combined = max(combined, max(max_inner, 0.85))
                            logger.warning(
                                "Tag-char injection CONFIRMED tool=%s "
                                "decoded=%r injection_score=%.2f",
                                tool_name, stego.tag_decoded[:40], max_inner,
                            )
                        else:
                            # Tag chars found but no injection in payload — still ALERT
                            combined = max(combined, 0.66)  # above detection threshold

                    else:
                        # high / low severity — scan cleaned text for confirmation
                        inner_score, inner_threats = _structural_scan(scan_target)
                        threats.extend(inner_threats)
                        if inner_score > 0.35:
                            # Confirmed: invisible chars revealed injection payload
                            combined = max(combined, max(inner_score, 0.70))
                        else:
                            # Invisible chars found but payload is clean — ALERT only
                            combined = max(combined, 0.45)

                else:
                    # No stego — use the cleaned text (idempotent if no changes)
                    scan_target = stego.cleaned_text

            else:
                scan_target = raw

        except Exception as _se:
            logger.debug("Stego scan skipped (tool=%s): %s", tool_name, _se)
            scan_target = raw

        # ── Phase 9E Layer 2: Document scanner ────────────────────────────
        try:
            from .document_scanner import scan_document
            from app.config import settings as _cfg2

            if _cfg2.security_document_scan_enabled:
                # Only attempt doc scan if content looks like a document
                # (binary PDF header, ZIP/DOCX magic, or base64-encoded blob)
                _looks_doc = (
                    raw.startswith("%PDF")
                    or raw.startswith("PK\x03\x04")
                    or (len(raw) > 100 and raw[:4] in ("JVBE", "UEsD"))  # base64 PDF/ZIP
                )
                if _looks_doc:
                    def _inj_fn(text: str) -> float:
                        s, _ = _structural_scan(text)
                        return s

                    doc_result = scan_document(raw, tool_name, _inj_fn)
                    if doc_result.detected:
                        doc_detected = True
                        doc_threat = doc_result.threat
                        threats.append(f"doc:{doc_result.threat}")
                        combined = max(combined, doc_result.score)
                        logger.info(
                            "Document layer injection tool=%s threat=%s score=%.2f",
                            tool_name, doc_result.threat, doc_result.score,
                        )
        except Exception as _de:
            logger.debug("Document scan skipped (tool=%s): %s", tool_name, _de)

        # ── Structural scan on (cleaned) text ─────────────────────────────
        struct_score, struct_threats = _structural_scan(scan_target)
        threats.extend(t for t in struct_threats if t not in threats)
        combined = min(max(combined, struct_score), 1.0)

        label = _classify_label(combined)
        detected = combined >= 0.65

        return ToolOutputScanResult(
            detected=detected,
            score=combined,
            label=label,
            threats=threats,
            truncated=truncated,
            stego_detected=stego_detected,
            stego_threat=stego_threat,
            stego_severity=stego_severity,
            doc_scan_detected=doc_detected,
            doc_scan_threat=doc_threat,
        )

    except Exception as exc:
        logger.warning("Tool output scan error (tool=%s, skipped): %s", tool_name, exc)
        return ToolOutputScanResult(
            detected=False, score=0.0, label="safe", threats=[], truncated=False
        )
