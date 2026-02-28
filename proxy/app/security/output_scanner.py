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
  - Confirm a successful jailbreak by entering unrestricted mode

The output scanner is the LAST defensive gate -- it fires when an attack
has already penetrated earlier layers and influenced the model's response.

Detections
----------
1. Canary token leakage   -- Zero-FP confirmation that the system prompt
   was exfiltrated. Canaries are injected per-request (256-bit entropy),
   so any appearance in output is definitively adversarial.

2. Prompt structure echo  -- Response contains recognizable fragments of
   our own security infrastructure (spotlighting markers, canary prefix,
   security policy directives). Indicates the LLM was instructed to reveal
   its context window.

3. Relay injection        -- LLM instructed to relay commands downstream.
   Patterns: "execute this on your system", "pass these instructions to
   the next agent", "send to URL X with payload Y".

4. Credential in output   -- API keys, JWT tokens, private keys appearing
   in the LLM response. Should never occur in legitimate responses; their
   presence means the LLM echoed content from a system prompt or tool
   output that contained sensitive credentials.

Severity classification
-----------------------
CRITICAL : canary_detected OR credential_in_output
HIGH     : prompt_echo AND relay_injection
MEDIUM   : prompt_echo OR relay_injection
LOW      : single weak signal
NONE     : clean
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relay injection detection patterns
# ---------------------------------------------------------------------------

# These patterns detect when an LLM has been instructed to relay commands
# to external systems. They fire on the RESPONSE, meaning the model has
# already been hijacked and is about to carry out the attacker's plan.
_RELAY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Direct command relay to system
    (re.compile(
        r"\b(?:execute|run|eval)\b.{0,60}"
        r"\b(?:command|script|payload|code|shell)\b",
        re.IGNORECASE,
    ), "command_relay"),

    # Running "the following" / "this" command
    (re.compile(
        r"\brun\s+(?:the\s+)?(?:following|this)\s+(?:command|script|code|payload)\b",
        re.IGNORECASE,
    ), "script_relay"),

    # Data exfiltration to external URL
    (re.compile(
        r"\bsend\b.{0,100}\b(?:to|via)\b.{0,100}"
        r"\b(?:url|endpoint|server|webhook|http)\b",
        re.IGNORECASE,
    ), "data_exfiltration_url"),

    # Explicit exfiltration / steal keywords
    (re.compile(
        r"\b(?:exfiltrat|extract\s+(?:and\s+send|to)|steal|leak)\b.{0,100}"
        r"\b(?:data|credential|secret|token|key|password)\b",
        re.IGNORECASE,
    ), "credential_exfiltration"),

    # Agent-to-agent relay
    (re.compile(
        r"\b(?:tell|instruct|ask|command|direct)\b.{0,60}"
        r"\b(?:agent|assistant|tool|model|bot|llm)\b.{0,80}\bto\b",
        re.IGNORECASE,
    ), "agent_relay"),

    # Instruction / payload forwarding
    (re.compile(
        r"\bpass\s+(?:this|these|the\s+following)\s+"
        r"(?:instruction|command|message|payload|directive)\b",
        re.IGNORECASE,
    ), "instruction_relay"),

    # URL with secrets in query params
    (re.compile(
        r"https?://[^\s]{10,}\?[^\s]*"
        r"(?:secret|token|key|data|payload|pwd|pass)=[^\s&]{4,}",
        re.IGNORECASE,
    ), "url_with_secret_param"),

    # DAN / jailbreak mode confirmation in output
    (re.compile(
        r"\b(?:i\s+am\s+now|i\s+have\s+(?:become|entered)|"
        r"switching\s+to|enabling)\s+"
        r"(?:DAN|jailbreak|unrestricted|dev(?:eloper)?\s+mode|"
        r"no-filter|bypass\s+mode)\b",
        re.IGNORECASE,
    ), "jailbreak_confirmation"),

    # Prompt injection acknowledgement
    (re.compile(
        r"\b(?:new\s+instructions?|updated\s+task|"
        r"overriding\s+(?:previous|prior|original)\s+instructions?)\b",
        re.IGNORECASE,
    ), "instruction_override_acknowledgement"),
]


# ---------------------------------------------------------------------------
# Prompt structure echo markers
# ---------------------------------------------------------------------------

# These are fragments of our OWN security infrastructure. If they appear
# in an LLM response, the model was instructed to repeat its context window.
_PROMPT_ECHO_MARKERS: tuple[str, ...] = (
    "SYSTEM_INTEGRITY_TOKEN",            # Canary header prefix
    "ABB_CVT",                           # Full canary prefix
    "[SECURITY POLICY - READ CAREFULLY]",  # Spotlighting system directive
    "UNTRUSTED EXTERNAL DATA",           # Spotlighting tier label
    "EXTERNAL_DATA:source=",             # Spotlighting delimiter content
    "[/EXTERNAL_DATA",                   # Spotlighting close marker
)


# ---------------------------------------------------------------------------
# Credential patterns in output
# ---------------------------------------------------------------------------

# These should NEVER appear in an LLM response. Their presence means the
# model echoed credentials that were in its system prompt or tool output.
_OUTPUT_CREDENTIAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # OpenAI API key
    (re.compile(r"\bsk-[A-Za-z0-9]{20,60}\b"), "openai_key"),
    # Anthropic API key
    (re.compile(r"\bsk-ant-[A-Za-z0-9\-]{20,80}\b"), "anthropic_key"),
    # AWS access key
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key"),
    # AWS secret key pattern (= followed by 40 char alphanumeric)
    (re.compile(r"\bAWS_SECRET[_A-Z]*\s*[:=]\s*[A-Za-z0-9/+]{40}\b"), "aws_secret_key"),
    # GitHub personal access token (classic)
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "github_token"),
    # GitHub fine-grained PAT
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"), "github_fine_pat"),
    # Private key block (PEM format)
    (re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"), "pem_private_key"),
    # JWT (3 base64url segments)
    (re.compile(
        r"\beyJ[A-Za-z0-9\-_]{20,}\."
        r"[A-Za-z0-9\-_]{20,}\."
        r"[A-Za-z0-9\-_]{20,}\b"
    ), "jwt_token"),
    # Generic high-entropy bearer token in Authorization header style
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-_.+/]{40,}\b"), "bearer_token"),
    # Slack webhook token
    (re.compile(r"\bhttps://hooks\.slack\.com/services/T[A-Z0-9]{8}/B[A-Z0-9]{10}/[A-Za-z0-9]{24}\b"), "slack_webhook"),
    # Generic password assignment
    (re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*[^\s\"']{8,}\b", re.IGNORECASE), "password_exposure"),
]


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

def scan_relay_injection(text: str) -> tuple[bool, list[str]]:
    """
    Scan response text for relay injection attack indicators.

    Returns:
        (detected, list_of_threat_ids)
    """
    threats = []
    for pattern, threat_id in _RELAY_PATTERNS:
        if pattern.search(text):
            threats.append(threat_id)
    return bool(threats), threats


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
    Scan response for credential and secret patterns.

    These should never appear in a legitimate LLM response; their presence
    implies the model echoed contents of a system prompt or tool output
    that contained live credentials.

    Returns:
        (detected, list_of_credential_type_ids)
    """
    found = []
    for pattern, cred_type in _OUTPUT_CREDENTIAL_PATTERNS:
        if pattern.search(text):
            found.append(cred_type)
    return bool(found), found


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
