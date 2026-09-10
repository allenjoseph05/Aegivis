"""
Email Preview Sandbox Adapter — Phase 16.6.

Performs structural analysis of email tool calls WITHOUT sending anything.
Checks recipient count, body length, PII signals in the body, and attachment
sizes.  Returns safe=False when the scope exceeds configurable thresholds.

Detects tool calls by:
  1. Tool name tokenises to contain "email", "mail", "send", "notify", etc.
  2. Args contain a recipient-shaped key ("to", "recipient", "cc", …).

No regex.  All detection is frozenset intersection on lowercased tokens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..sandbox import SandboxAdapter, SandboxContext, SandboxResult
from ._common import tokenize as _tokenize


# ── Thresholds ────────────────────────────────────────────────────────────────

_BULK_RECIPIENT_THRESHOLD = 10          # >= this → unsafe (bulk send)
_MASS_RECIPIENT_THRESHOLD = 50          # >= this → mass-send signal
_LARGE_ATTACHMENT_BYTES   = 10 * 1024 * 1024   # 10 MB
_MANY_ATTACHMENTS         = 5


# ── Token sets ────────────────────────────────────────────────────────────────

_EMAIL_TOOL_TOKENS: frozenset[str] = frozenset({
    "email", "mail", "send", "notify", "message", "smtp",
    "gmail", "outlook", "sendgrid", "mailgun", "postmark",
})

_RECIPIENT_ARG_KEYS: frozenset[str] = frozenset({
    "to", "recipient", "recipients", "cc", "bcc",
    "address", "addresses", "email", "emails",
    "to_email", "to_address", "destination",
})

_BODY_ARG_KEYS: frozenset[str] = frozenset({
    "body", "content", "text", "message", "html",
    "body_html", "body_text", "template", "payload",
})

_PII_SIGNAL_TOKENS: frozenset[str] = frozenset({
    "ssn", "social", "security", "password", "passwd",
    "secret", "token", "credit", "card", "cvv", "pin",
    "dob", "birthdate", "national", "license", "passport",
    "iban", "routing", "account", "private", "key",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lower_arg_keys(args: dict) -> frozenset[str]:
    return frozenset(str(k).lower() for k in args)


# ── EmailPreviewAdapter ───────────────────────────────────────────────────────

class EmailPreviewAdapter(SandboxAdapter):
    """
    Structural preview of email send operations.

    Returns safe=False when:
      - recipient_count >= BULK_RECIPIENT_THRESHOLD (10)
      - PII-signal tokens appear in the message body

    Returns a warning signal (safe stays True) when:
      - total attachment bytes > 10 MB
      - attachment count > 5
    """

    name = "email"

    async def can_handle(self, tool_name: str, args: dict) -> bool:
        if _tokenize(tool_name) & _EMAIL_TOOL_TOKENS:
            return True
        # Also match when args have a recipient-shaped key, even if tool name differs
        return bool(_lower_arg_keys(args) & _RECIPIENT_ARG_KEYS)

    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        t0 = time.monotonic()

        recipients    = _extract_recipients(args)
        body          = _extract_body(args)
        attachments   = _extract_attachments(args)

        recipient_count       = len(recipients)
        body_len              = len(body)
        attachment_count      = len(attachments)
        total_attachment_bytes = sum(_estimate_size(a) for a in attachments)

        # PII check — tokenize body (strip punctuation from each word so
        # "ssn:", "password." etc. still match the PII signal frozenset).
        body_tokens = frozenset(_tokenize(body))
        pii_found   = sorted(body_tokens & _PII_SIGNAL_TOKENS)

        signals: list[str] = []
        safe = True

        if recipient_count >= _MASS_RECIPIENT_THRESHOLD:
            signals.append(f"mass-send:{recipient_count}-recipients")
            safe = False
        elif recipient_count >= _BULK_RECIPIENT_THRESHOLD:
            signals.append(f"bulk-send:{recipient_count}-recipients")
            safe = False

        if pii_found:
            signals.append(f"pii-in-body:{','.join(pii_found[:3])}")
            safe = False

        if total_attachment_bytes > _LARGE_ATTACHMENT_BYTES:
            signals.append(f"large-attachment:{total_attachment_bytes // 1024}KB")

        if attachment_count > _MANY_ATTACHMENTS:
            signals.append(f"many-attachments:{attachment_count}")

        if not signals:
            signals.append("email-preview-ok")

        # Build human-readable preview
        preview_parts: list[str] = []
        if recipients:
            summary = ", ".join(recipients[:3])
            if recipient_count > 3:
                summary += f" (+{recipient_count - 3} more)"
            preview_parts.append(f"To: {summary}")
        if body:
            snippet = body[:200]
            if len(body) > 200:
                snippet += "…"
            preview_parts.append(f"Body: {snippet}")

        return SandboxResult(
            adapter="email",
            executed=False,   # never actually sends
            safe=safe,
            scope_estimate={
                "recipient_count":       recipient_count,
                "body_length":           body_len,
                "attachment_count":      attachment_count,
                "total_attachment_bytes": total_attachment_bytes,
                "pii_signals":           pii_found,
            },
            preview=" | ".join(preview_parts)[:500],
            signals=signals,
            latency_ms=(time.monotonic() - t0) * 1000,
            error=None,
        )


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_recipients(args: dict) -> list[str]:
    for key in _RECIPIENT_ARG_KEYS:
        val = args.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip():
            # Split on comma / semicolon
            return [r.strip() for r in val.replace(";", ",").split(",") if r.strip()]
        if isinstance(val, list):
            return [str(r).strip() for r in val if r]
    return []


def _extract_body(args: dict) -> str:
    for key in _BODY_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_attachments(args: dict) -> list:
    for key in ("attachments", "files", "attach", "file"):
        val = args.get(key)
        if isinstance(val, list):
            return val
        if val is not None:
            return [val]
    return []


def _estimate_size(attachment) -> int:
    """Best-effort byte estimate of a single attachment."""
    if isinstance(attachment, dict):
        explicit = attachment.get("size") or attachment.get("file_size")
        if isinstance(explicit, int):
            return explicit
        content = attachment.get("content") or attachment.get("data") or ""
        return len(str(content))
    return len(str(attachment))
