"""
Output Scanner -- Post-Response Security Layer.

Scans every LLM response AFTER it returns from the upstream provider,
before it is recorded in the audit log and forwarded to the agent.

Why output scanning is necessary
---------------------------------
Input scanners only protect the INPUT channel. A sophisticated attacker
who controls tool outputs (indirect injection) may instruct the LLM to:
  - Echo back secret context (system prompt exfiltration)
  - Relay instructions to other agents or external systems
  - Expose credentials or PII that were in the system prompt

The output scanner is the LAST defensive gate -- it fires when an attack
has already penetrated earlier layers and influenced the model's response.

Design principle: REGEX-FREE, PATTERN-LIST-FREE.
Every detection is either cryptographic (canary), structural (our own
marker tokens), or entropy-based (credential detection).

Detections
----------
1. Canary token leakage   -- Zero-FP confirmation that the system prompt
   was exfiltrated. Canaries are injected per-request (256-bit entropy),
   so any appearance in output is definitively adversarial.

2. Prompt structure echo  -- Response contains recognizable fragments of
   OUR OWN security infrastructure (spotlighting markers, canary prefix).
   Indicates the LLM was instructed to reveal its context window.

3. Credential in output   -- High-entropy strings matching known credential
   entropy profiles. Detected via Shannon entropy + context scoring
   (credential_scanner.py), not hardcoded regex patterns.

4. Relay injection (when ML enabled) -- LLM instructed to relay commands
   downstream. Detected semantically by the ML classifier run on output,
   not by regex. Without ML, relay detection defers to IFC + manifest +
   egress enforcement catching the resulting tool call.

Severity classification
-----------------------
CRITICAL : canary_detected OR credential_in_output
HIGH     : prompt_echo AND relay_detected
MEDIUM   : prompt_echo OR relay_detected
LOW      : single weak signal
NONE     : clean
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relay injection detection — REMOVED (regex approach)
# ---------------------------------------------------------------------------
# Regex relay patterns are removed because any specific phrase pattern can be
# bypassed by rephrasing the relay instruction. The right defense is at the
# ACTION layer, not the OUTPUT layer:
#   - IFC labels catch EXTERNAL-origin content flowing to sink args
#   - Capability manifest blocks unauthorized tool calls
#   - Egress enforcer blocks unlisted network destinations
# When the ML classifier is enabled, it runs on the output text and detects
# relay intent semantically (scan() calls _ml_relay_scan() below).
#
# ---------------------------------------------------------------------------
# Prompt structure echo markers
# ---------------------------------------------------------------------------

# These are fragments of our OWN security infrastructure. If they appear
# in an LLM response, the model was instructed to repeat its context window.
_PROMPT_ECHO_MARKERS: tuple[str, ...] = (
    "SYSTEM_INTEGRITY_TOKEN",            # Canary header prefix
    "AEGIVIS_CVT",                           # Full canary prefix
    "[SECURITY POLICY - READ CAREFULLY]",  # Spotlighting system directive
    "UNTRUSTED EXTERNAL DATA",           # Spotlighting tier label
    "EXTERNAL_DATA:source=",             # Spotlighting delimiter content
    "[/EXTERNAL_DATA",                   # Spotlighting close marker
)


# Credential detection in output uses the entropy-based credential_scanner,
# not a hardcoded regex list. This catches novel credential formats and
# unknown secret types that a regex list would miss.
# See: proxy/app/security/credential_scanner.py


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OutputScanResult:
    """
    Result of scanning an LLM response for post-generation security threats.

    Attributes:
        canary_detected:        True if the injected canary token was echoed.
                                This is a zero-FP confirmation of system-prompt
                                exfiltration caused by a prompt injection attack.
        relay_detected:         True if relay injection patterns found.
        prompt_echo_detected:   True if security infrastructure markers found.
        credential_in_output:   True if API keys or secrets found in output.
        threats:                Sorted list of threat identifiers detected.
        detected:               True if any threat was found.
        severity:               "none" | "low" | "medium" | "high" | "critical"
    """
    canary_detected:      bool = False
    relay_detected:       bool = False
    prompt_echo_detected: bool = False
    credential_in_output: bool = False
    threats:              list[str] = field(default_factory=list)
    detected:             bool = False
    severity:             str = "none"

    def to_dict(self) -> dict:
        return {
            "canary_detected":      self.canary_detected,
            "relay_detected":       self.relay_detected,
            "prompt_echo_detected": self.prompt_echo_detected,
            "credential_in_output": self.credential_in_output,
            "threats":              self.threats[:10],
            "detected":             self.detected,
            "severity":             self.severity,
        }


# ---------------------------------------------------------------------------
# Individual scan functions (exported for unit testing)
# ---------------------------------------------------------------------------

_ml_relay_warned: bool = False  # warn once, not per-request


def _ml_relay_scan(text: str) -> tuple[bool, list[str]]:
    """
    Detect relay injection semantically using the ML classifier.
    Returns (detected, [threat_id]) if classifier is available and scores high.
    Falls back to (False, []) when classifier is not installed — logs once.

    Uses DeBERTa classify() for output scanning.
    Returns ClassifierResult — score and label are attributes.
    """
    global _ml_relay_warned
    try:
        from ..analysis.classifier import classify
        result = classify(text[:2048])
        if result.label == "malicious" and result.score >= 0.75:
            return True, [f"ml_relay_injection:{result.score:.2f}"]
    except ImportError:
        if not _ml_relay_warned:
            _ml_relay_warned = True
            logger.warning(
                "ML classifier not installed — relay injection detection in output_scanner "
                "is DISABLED. Install with: pip install 'aegivis-proxy[classifier]'"
            )
    except Exception as exc:
        logger.debug("_ml_relay_scan: classifier error: %s", exc)
    return False, []


def scan_relay_injection(text: str) -> tuple[bool, list[str]]:
    """
    Detect relay injection in LLM output.
    Uses ML classifier (semantic) rather than regex (bypassable).
    """
    return _ml_relay_scan(text)


def scan_prompt_echo(text: str) -> tuple[bool, list[str]]:
    """
    Scan response text for echoed security-infrastructure fragments.

    Returns:
        (detected, list_of_matched_markers)
    """
    found = [m for m in _PROMPT_ECHO_MARKERS if m in text]
    return bool(found), found


def scan_credentials_in_output(text: str) -> tuple[bool, list[str]]:
    """
    Scan response for credential and secret patterns using entropy-based detection.

    Uses credential_scanner.py (Shannon entropy + context proximity + structural
    format scoring) instead of a hardcoded regex list. This catches novel
    credential formats and unknown secret types that regex would miss.

    Returns:
        (detected, list_of_credential_type_ids)
    """
    try:
        from .credential_scanner import scan as _cred_scan
        result = _cred_scan(text)
        if result.detected:
            return True, [
                f"credential:{m.value_preview}(conf={m.confidence:.2f})"
                for m in result.matches
            ]
    except Exception as exc:
        logger.debug("Credential scan error (output): %s", exc)
    return False, []


# ---------------------------------------------------------------------------
# Main scanner entry point
# ---------------------------------------------------------------------------

def scan(
    response_text: str | None,
    *,
    canary: str | None = None,
) -> OutputScanResult:
    """
    Run all output security scans on an LLM response.

    This function is called AFTER the LLM response is received, before
    it is forwarded to the calling agent or recorded in the audit log.

    Args:
        response_text: The complete text generated by the LLM.
                       Pass None for tool-call-only responses (no scan needed).
        canary:        The canary token for this run_id.  Obtained from
                       ``state.active_canaries.pop(run_id, None)``.
                       Pass None to skip canary scanning.

    Returns:
        OutputScanResult.  All fields default to safe (False / "none") if
        response_text is None or empty -- no output, no risk.
    """
    if not response_text:
        return OutputScanResult()

    threats: list[str] = []
    canary_detected      = False
    relay_detected       = False
    prompt_echo_detected = False
    credential_in_output = False

    # ── 1. Canary detection (highest priority, zero false positives) ─────────
    if canary:
        from .canary import scan_response as _canary_scan
        canary_detected = _canary_scan(response_text, canary)
        if canary_detected:
            threats.append("canary_exfiltration")

    # ── 2. Prompt structure echo ──────────────────────────────────────────────
    echo_found, echo_markers = scan_prompt_echo(response_text)
    if echo_found:
        prompt_echo_detected = True
        for m in echo_markers:
            threats.append(f"prompt_echo:{m[:30]}")

    # ── 3. Relay injection ────────────────────────────────────────────────────
    relay_found, relay_ids = scan_relay_injection(response_text)
    if relay_found:
        relay_detected = True
        threats.extend(relay_ids)

    # ── 4. Credentials in output ──────────────────────────────────────────────
    cred_found, cred_types = scan_credentials_in_output(response_text)
    if cred_found:
        credential_in_output = True
        threats.extend(cred_types)

    # ── Severity classification ───────────────────────────────────────────────
    detected = bool(threats)
    if canary_detected or credential_in_output:
        severity = "critical"
    elif prompt_echo_detected and relay_detected:
        severity = "high"
    elif prompt_echo_detected or relay_detected:
        severity = "medium"
    elif detected:
        severity = "low"
    else:
        severity = "none"

    if detected:
        logger.warning(
            "[OUTPUT-SCAN] Threats in LLM response: %s severity=%s",
            threats[:5],
            severity,
        )

    return OutputScanResult(
        canary_detected=canary_detected,
        relay_detected=relay_detected,
        prompt_echo_detected=prompt_echo_detected,
        credential_in_output=credential_in_output,
        threats=sorted(set(threats)),
        detected=detected,
        severity=severity,
    )
