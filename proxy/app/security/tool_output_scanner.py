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

# Injection phrase patterns — REMOVED.
# Phrase/string matching is bypassable by any paraphrase and creates false
# positives on normal tool output (YAML with "system:", Markdown with "###",
# chat transcripts with "assistant:"). Detection is now split between:
#
#   Structural (sync, fast):   enforcement/structural.py token_score
#     → LLM delimiter tokens (<|im_start|>, [INST], etc.) and Unicode anomalies
#   Semantic (async, ML):      AEGIVIS_SECURITY_TOOL_OUTPUT_ML_SCAN_ENABLED=true
#     → DeBERTa classifier detects paraphrased injection
#
# Phase 9E stego detection (unicode_scanner + document_scanner) is unchanged.

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
    Structural scan using enforcement/structural.py token_score.

    Detects: LLM delimiter tokens (<|im_start|>, [INST], etc.) and Unicode
    anomalies (invisible chars, directional overrides). Does NOT use phrase
    lists — those are bypassable by paraphrase and have been removed.

    Returns (score, threats) matching the original interface so stego +
    document scanner callers are unchanged.
    """
    try:
        from ..enforcement.structural import scan as _eng_scan
        result = _eng_scan(text)
        threats: list[str] = []
        if result.delimiter_hits > 0:
            threats.append(f"delimiter_injection:{result.delimiter_hits}_hits")
        if result.override_score > 0:
            threats.append("direction_override")
        if result.invisible_score > 0:
            threats.append("invisible_unicode")
        return result.token_score, threats
    except Exception:
        return 0.0, []


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
