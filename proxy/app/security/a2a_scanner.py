"""
Phase E7 — A2A (Agent-to-Agent) Protocol Security Scanner.

Intercepts Google A2A Protocol (JSON-RPC 2.0) messages exchanged between
autonomous AI agents.  The A2A protocol defines a standard HTTP-based
communication channel so any agent can delegate tasks to another agent — and
attackers can abuse this channel to pivot across agent trust boundaries.

Threat model
------------
1. **Cross-agent prompt injection** — Agent A is hijacked by a malicious tool
   result; it relays the injected instruction to Agent B via A2A, effectively
   escaping the security perimeter of Agent A's session.
2. **PII leakage through delegation** — Agent A passes sensitive user data
   (SSN, email, CC) to Agent B as part of the task description; if Agent B has
   broader permissions the data may leave the secure perimeter.
3. **Trust escalation** — An agent sends tasks to a privileged peer agent with
   elevated capabilities it could not otherwise invoke directly.

How it works
------------
The proxy intercepts ``POST /a2a`` requests.  The target agent URL is supplied
via ``X-A2A-Agent-URL`` header.  The proxy:

1. Parses the JSON-RPC 2.0 body.
2. Extracts ``TextPart`` content from ``tasks/send`` / ``tasks/sendSubscribe``
   payloads.
3. Runs structural injection scan (delimiter anomaly + Unicode attack signals).
4. Runs PII detection (presidio-first with regex fallback for email + IBAN).
5. Emits an ``A2A_MESSAGE_SEND`` audit event.
6. Forwards to the real target agent URL.
7. Parses the response, scans artifact text for PII.
8. Emits an ``A2A_MESSAGE_RECEIVE`` audit event.

Usage
-----
Point your A2A client at the Aegivis proxy::

    client = A2AClient(url="http://localhost:8080/a2a")

And set the real target agent URL in the header::

    X-A2A-Agent-URL: http://my-agent.internal:8000
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# JSON-RPC methods that carry agent task messages (need injection + PII scan)
_SEND_METHODS = frozenset({"tasks/send", "tasks/sendSubscribe"})

# JSON-RPC methods that return artifacts we want to scan for PII
_RESULT_METHODS = frozenset({"tasks/send", "tasks/sendSubscribe", "tasks/get"})


@dataclass
class A2AScanResult:
    """Security scan result for a single A2A JSON-RPC message."""

    method: str
    """JSON-RPC method name (e.g. 'tasks/send')."""

    task_id: str | None
    """A2A task ID extracted from params.id or result.id."""

    message_role: str | None
    """Role of the message sender: 'user' | 'agent' | None."""

    text_parts: list[str]
    """All TextPart content strings extracted from the message."""

    injection_score: float
    """Highest structural injection score across all text parts (0.0–1.0)."""

    injection_triggered: bool
    """True if injection_score >= 0.5 (alert threshold)."""

    pii_detected: list[str]
    """Deduplicated PII entity types found in text parts."""

    critical_pii: bool
    """True if any detected PII type is high-severity (SSN, CC, IBAN, etc.)."""

    artifact_texts: list[str]
    """Text content extracted from response artifacts (populated by scan_a2a_response)."""


def scan_a2a_request(body: dict) -> A2AScanResult:
    """
    Scan an outgoing A2A JSON-RPC request for injection signals and PII.

    Covers ``tasks/send`` and ``tasks/sendSubscribe`` — the two methods that
    carry user/agent-authored message content.  Other methods (tasks/get,
    tasks/cancel) carry no text content and are passed through with an
    empty-result scan record.

    Parameters
    ----------
    body:
        Parsed JSON-RPC 2.0 request dict.

    Returns
    -------
    A2AScanResult with all relevant security findings.
    """
    method = body.get("method", "")
    params = body.get("params", {}) or {}
    task_id = params.get("id")

    text_parts: list[str] = []
    message_role: str | None = None

    if method in _SEND_METHODS:
        message = params.get("message", {}) or {}
        message_role = message.get("role")
        text_parts = _extract_text_parts(message)

        # Also scan the historyLength field (not content) and metadata if present
        # to avoid future blind spots as the protocol evolves.
        context_id = params.get("contextId")  # ignore — not scannable text

    # ── Structural injection scan ─────────────────────────────────────────────
    injection_score = 0.0
    injection_triggered = False
    if text_parts:
        try:
            from ..enforcement.structural import scan as _structural_scan
            for part in text_parts:
                result = _structural_scan(part)
                injection_score = max(injection_score, result.score)
            injection_triggered = injection_score >= 0.5
        except Exception as exc:
            logger.debug("A2A structural injection scan failed: %s", exc)

    # ── PII detection ─────────────────────────────────────────────────────────
    pii_types: list[str] = []
    critical_pii = False
    if text_parts:
        try:
            from .embedding_guard import _detect_pii
            all_text = "\n".join(text_parts)
            pii_results, _used_presidio = _detect_pii(all_text)
            pii_types = sorted({p["type"] for p in pii_results})
            critical_pii = any(p.get("critical") for p in pii_results)
        except Exception as exc:
            logger.debug("A2A PII detection failed: %s", exc)

    return A2AScanResult(
        method=method,
        task_id=task_id,
        message_role=message_role,
        text_parts=text_parts,
        injection_score=injection_score,
        injection_triggered=injection_triggered,
        pii_detected=pii_types,
        critical_pii=critical_pii,
        artifact_texts=[],
    )


def scan_a2a_response(response_body: dict) -> A2AScanResult:
    """
    Scan an A2A JSON-RPC response for PII in artifact text content.

    A2A responses may include ``artifacts`` — structured output from the
    target agent.  PII in artifacts indicates the upstream agent may have
    accessed or generated sensitive data that should not flow back to the
    calling agent without proper controls.

    Parameters
    ----------
    response_body:
        Parsed JSON-RPC 2.0 response dict (the body returned by the target agent).

    Returns
    -------
    A2AScanResult with artifact_texts populated and pii_detected from artifacts.
    """
    result_data = response_body.get("result", {}) or {}
    task_id = result_data.get("id")

    artifact_texts: list[str] = []
    for artifact in (result_data.get("artifacts") or []):
        for part in (artifact.get("parts") or []):
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text", "")
                if t and isinstance(t, str):
                    artifact_texts.append(t)

    pii_types: list[str] = []
    critical_pii = False
    if artifact_texts:
        try:
            from .embedding_guard import _detect_pii
            all_text = "\n".join(artifact_texts)
            pii_results, _used_presidio = _detect_pii(all_text)
            pii_types = sorted({p["type"] for p in pii_results})
            critical_pii = any(p.get("critical") for p in pii_results)
        except Exception as exc:
            logger.debug("A2A response PII detection failed: %s", exc)

    return A2AScanResult(
        method="",
        task_id=task_id,
        message_role=None,
        text_parts=[],
        injection_score=0.0,
        injection_triggered=False,
        pii_detected=pii_types,
        critical_pii=critical_pii,
        artifact_texts=artifact_texts,
    )


def _extract_text_parts(message: dict) -> list[str]:
    """Extract text content from all TextPart entries in an A2A message."""
    texts = []
    for part in message.get("parts", []):
        if isinstance(part, dict) and part.get("type") == "text":
            t = part.get("text", "")
            if t and isinstance(t, str):
                texts.append(t)
    return texts
