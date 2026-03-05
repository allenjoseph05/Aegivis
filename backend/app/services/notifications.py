"""
Alerting + Notification service — Phase 9F

Routes policy violations and anomaly flags to configured channels based on
severity, with anti-fatigue deduplication and per-rule cooldowns.

Channels (all optional — disabled when env var not set):
  1. Slack Block Kit       — ABB_SLACK_WEBHOOK_URL
  2. SMTP email            — ABB_SMTP_HOST + ABB_SMTP_TO
  3. PagerDuty Events v2   — ABB_PAGERDUTY_ROUTING_KEY
  4. Generic webhook       — ABB_ALERT_WEBHOOK_URL

Severity tiers:
  CRITICAL — canary leakage, data-exfiltration, system-prompt-mutation
             → all channels
  HIGH     — any BLOCK action, spawn-depth-exceeded, ml-injection-block
             → Slack + email + PagerDuty
  MEDIUM   — ALERT actions with moderate score (rag, stego, goal-drift)
             → Slack only
  LOW      — informational alerts
             → dashboard only (no external channel)

Anti-fatigue:
  AlertThrottle deduplicates by (rule_name, session_id) with a configurable
  cooldown window (default 600 s). Identical violations within the window are
  suppressed to prevent alert storms.
"""
from __future__ import annotations

import asyncio
import email.mime.text
import hashlib
import json
import logging
import smtplib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from ..config import settings

if TYPE_CHECKING:
    from .anomaly import AnomalyFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

# Rules that always fire at CRITICAL tier
_CRITICAL_RULES: frozenset[str] = frozenset({
    "canary-leak",
    "canary-token-leak",
    "data-exfiltration-attempt",
    "system-prompt-mutation",
    "credential-in-llm-response",
})

# Rules that fire at HIGH tier
_HIGH_RULES: frozenset[str] = frozenset({
    "spawn-depth-exceeded",
    "ml-injection-block",
    "tool-call-loop-protection",
    "pii-in-llm-request",
    "rag-poisoning-block",
    "unicode-stego-block",
    "rce-attempt",
    "ssrf-attempt",
    "data-flow-suspicious",
})

# Rules that fire at MEDIUM tier (ALERT actions)
_MEDIUM_RULES: frozenset[str] = frozenset({
    "high-injection-score",
    "injection-score-alert",
    "rag-poisoning-alert",
    "goal-drift-alert",
    "intent-scope-violation",
    "consequence-high-risk",
    "context-injection-alert",
})


def _classify_severity(violation: dict) -> str:
    """
    Map a violation to a severity tier string.

    Priority order:
      1. CRITICAL rules — always CRITICAL
      2. BLOCK action   — HIGH (or CRITICAL if rule is in _CRITICAL_RULES)
      3. Named MEDIUM / HIGH rules
      4. Default — MEDIUM for ALERT action, LOW otherwise
    """
    rule = violation.get("rule_name", "")
    action = violation.get("action", "ALERT").upper()

    if rule in _CRITICAL_RULES:
        return "CRITICAL"
    if action == "BLOCK":
        return "HIGH" if rule not in _HIGH_RULES else "HIGH"
    if rule in _HIGH_RULES:
        return "HIGH"
    if rule in _MEDIUM_RULES:
        return "MEDIUM"
    if action == "ALERT":
        return "MEDIUM"
    return "LOW"


_SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


def _severity_meets_minimum(severity: str) -> bool:
    """Return True if severity >= configured minimum (ABB_ALERT_MIN_SEVERITY)."""
    min_sev = getattr(settings, "alert_min_severity", "MEDIUM").upper()
    return _SEVERITY_ORDER.get(severity, 0) >= _SEVERITY_ORDER.get(min_sev, 2)


# ---------------------------------------------------------------------------
# Anti-fatigue throttle
# ---------------------------------------------------------------------------

@dataclass
class _ThrottleEntry:
    last_sent_ts: float
    send_count: int = 0


class AlertThrottle:
    """
    Per-(rule_name, session_id) cooldown deduplicator.

    Thread-safe for asyncio (no threads involved), not safe for multi-process
    (in-process cache only; good enough for single backend pod).
    """

    def __init__(self, cooldown_s: float = 600.0) -> None:
        self._cooldown_s = cooldown_s
        self._cache: dict[str, _ThrottleEntry] = {}

    def should_send(self, rule: str, session_id: str) -> bool:
        """Return True if the alert should be sent (not throttled)."""
        key = f"{rule}:{session_id}"
        now = time.monotonic()
        entry = self._cache.get(key)
        if entry is None or (now - entry.last_sent_ts) >= self._cooldown_s:
            self._cache[key] = _ThrottleEntry(last_sent_ts=now, send_count=1)
            return True
        entry.send_count += 1
        return False

    def evict_old(self, max_age_s: float = 3600.0) -> int:
        """Remove stale entries. Returns number evicted."""
        cutoff = time.monotonic() - max_age_s
        stale = [k for k, v in self._cache.items() if v.last_sent_ts < cutoff]
        for k in stale:
            del self._cache[k]
        return len(stale)


# Module-level throttle instance (shared across all calls in one process)
_throttle = AlertThrottle(cooldown_s=600.0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI = {
    "CRITICAL": ":rotating_light:",
    "HIGH":     ":red_circle:",
    "MEDIUM":   ":large_orange_circle:",
    "LOW":      ":large_yellow_circle:",
}

_SEVERITY_COLOR = {
    "CRITICAL": "#FF0000",
    "HIGH":     "#FF6600",
    "MEDIUM":   "#FFA500",
    "LOW":      "#FFDD00",
}


def _slack_blocks(violation: dict, severity: str) -> dict:
    """Build a Slack Block Kit message payload for a violation."""
    rule    = violation.get("rule_name", "unknown")
    action  = violation.get("action", "ALERT")
    reason  = violation.get("reason", "")
    agent   = violation.get("agent_id", "unknown")
    session = violation.get("session_id", "")
    ts_ns   = violation.get("timestamp_ns", 0)

    emoji = _SEVERITY_EMOJI.get(severity, ":warning:")
    color = _SEVERITY_COLOR.get(severity, "#FFA500")

    header_text = f"{emoji} *{severity}* | `{rule}` | Action: `{action}`"

    fields = [
        {"type": "mrkdwn", "text": f"*Agent*\n`{agent}`"},
        {"type": "mrkdwn", "text": f"*Session*\n`{session[:16]}…`" if len(session) > 16 else f"*Session*\n`{session}`"},
        {"type": "mrkdwn", "text": f"*Reason*\n{reason[:200]}"},
    ]

    return {
        "attachments": [{
            "color": color,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": header_text}},
                {"type": "section", "fields": fields},
                {"type": "context", "elements": [
                    {"type": "mrkdwn", "text": f"AgentBlackBox | ts={ts_ns}"}
                ]},
            ],
        }]
    }


def _email_body(violation: dict, severity: str) -> tuple[str, str]:
    """Return (subject, body) for SMTP."""
    rule    = violation.get("rule_name", "unknown")
    action  = violation.get("action", "ALERT")
    reason  = violation.get("reason", "")
    agent   = violation.get("agent_id", "unknown")
    session = violation.get("session_id", "")

    subject = f"[AgentBlackBox] {severity} — {action}: {rule}"
    body = (
        f"AgentBlackBox security event\n"
        f"{'=' * 50}\n\n"
        f"Severity:   {severity}\n"
        f"Rule:       {rule}\n"
        f"Action:     {action}\n"
        f"Agent:      {agent}\n"
        f"Session:    {session}\n"
        f"Reason:     {reason}\n\n"
        f"{'=' * 50}\n"
        f"Review violations at your AgentBlackBox dashboard.\n"
    )
    return subject, body


def _pagerduty_payload(violation: dict, severity: str) -> dict:
    """Build PagerDuty Events API v2 trigger payload."""
    rule     = violation.get("rule_name", "unknown")
    action   = violation.get("action", "ALERT")
    reason   = violation.get("reason", "")
    agent    = violation.get("agent_id", "unknown")
    session  = violation.get("session_id", "")
    ts_ns    = violation.get("timestamp_ns", 0)

    # PagerDuty severity mapping
    pd_severity = {
        "CRITICAL": "critical",
        "HIGH":     "error",
        "MEDIUM":   "warning",
        "LOW":      "info",
    }.get(severity, "warning")

    # Stable dedup key — same rule + session within a window → same incident
    dedup_key = hashlib.sha256(
        f"{rule}:{session}:{ts_ns // 3_600_000_000_000}".encode()
    ).hexdigest()[:32]

    return {
        "routing_key": settings.pagerduty_routing_key,
        "event_action": "trigger",
        "dedup_key": dedup_key,
        "payload": {
            "summary": f"[{action}] {rule} — agent {agent}",
            "severity": pd_severity,
            "source": "agentblackbox-proxy",
            "component": rule,
            "group": agent,
            "class": action,
            "custom_details": {
                "reason":     reason,
                "session_id": session,
                "agent_id":   agent,
            },
        },
    }


# ---------------------------------------------------------------------------
# Channel senders (all fire-and-forget, never raise)
# ---------------------------------------------------------------------------

async def _send_slack(violation: dict, severity: str) -> None:
    url = settings.slack_webhook_url
    if not url:
        return
    try:
        payload = _slack_blocks(violation, severity)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code not in (200, 204):
                logger.warning("Slack alert returned %s: %s", resp.status_code, resp.text[:200])
            else:
                logger.info("Slack alert sent: rule=%s sev=%s", violation.get("rule_name"), severity)
    except Exception as e:
        logger.warning("Slack send failed (non-critical): %s", e)


def _smtp_send_sync(subject: str, body: str) -> None:
    """Synchronous SMTP — run in thread executor."""
    msg = email.mime.text.MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from
    msg["To"] = settings.smtp_to
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        smtp.ehlo()
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.sendmail(settings.smtp_from, settings.smtp_to.split(","), msg.as_string())


async def _send_email(violation: dict, severity: str) -> None:
    if not settings.smtp_host or not settings.smtp_to:
        return
    try:
        subject, body = _email_body(violation, severity)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _smtp_send_sync, subject, body)
        logger.info("SMTP alert sent: rule=%s sev=%s", violation.get("rule_name"), severity)
    except Exception as e:
        logger.warning("SMTP send failed (non-critical): %s", e)


async def _send_pagerduty(violation: dict, severity: str) -> None:
    key = getattr(settings, "pagerduty_routing_key", "")
    if not key:
        return
    try:
        payload = _pagerduty_payload(violation, severity)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
            )
            if resp.status_code not in (200, 201, 202):
                logger.warning(
                    "PagerDuty returned %s: %s", resp.status_code, resp.text[:200]
                )
            else:
                logger.info(
                    "PagerDuty alert sent: rule=%s sev=%s dedup=%s",
                    violation.get("rule_name"), severity,
                    payload["dedup_key"],
                )
    except Exception as e:
        logger.warning("PagerDuty send failed (non-critical): %s", e)


async def _send_webhook(violation: dict, severity: str) -> None:
    url = getattr(settings, "alert_webhook_url", "")
    if not url:
        return
    try:
        payload = {
            "severity": severity,
            "violation": violation,
            "source": "agentblackbox",
        }
        headers = {"Content-Type": "application/json"}
        secret = getattr(settings, "alert_webhook_secret", "")
        if secret:
            import hmac
            body_bytes = json.dumps(payload).encode()
            sig = hmac.new(secret.encode(), body_bytes, "sha256").hexdigest()
            headers["X-ABB-Signature"] = f"sha256={sig}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code not in (200, 201, 202, 204):
                logger.warning(
                    "Webhook returned %s: %s", resp.status_code, resp.text[:200]
                )
            else:
                logger.info(
                    "Webhook alert sent: rule=%s sev=%s",
                    violation.get("rule_name"), severity,
                )
    except Exception as e:
        logger.warning("Webhook send failed (non-critical): %s", e)


# ---------------------------------------------------------------------------
# Main dispatch entry point
# ---------------------------------------------------------------------------

async def dispatch_violation_alert(violation: dict) -> None:
    """
    Route a policy violation to all appropriate alert channels.

    Applies anti-fatigue throttling, classifies severity, then fires
    channels concurrently according to the severity routing table:

      CRITICAL → Slack + email + PagerDuty + webhook
      HIGH     → Slack + email + PagerDuty
      MEDIUM   → Slack only
      LOW      → (no external channel — dashboard only)
    """
    severity = _classify_severity(violation)

    # Filter LOW violations and anything below the configured minimum
    if severity == "LOW" or not _severity_meets_minimum(severity):
        return

    # Anti-fatigue throttle
    rule    = violation.get("rule_name", "")
    session = violation.get("session_id", "")
    if not _throttle.should_send(rule, session):
        logger.debug("Alert throttled: rule=%s session=%s", rule, session)
        return

    logger.info(
        "Dispatching %s alert: rule=%s action=%s",
        severity, rule, violation.get("action"),
    )

    tasks: list = []

    if severity in ("CRITICAL", "HIGH", "MEDIUM"):
        tasks.append(_send_slack(violation, severity))

    if severity in ("CRITICAL", "HIGH"):
        tasks.append(_send_email(violation, severity))
        tasks.append(_send_pagerduty(violation, severity))

    if severity == "CRITICAL":
        tasks.append(_send_webhook(violation, severity))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Backwards-compatible simple helpers (used by ingest.py + external code)
# ---------------------------------------------------------------------------

async def notify_block(violation: dict) -> None:
    """Legacy: send SMTP email for a BLOCK violation. Prefer dispatch_violation_alert."""
    await _send_email(violation, "HIGH")


async def notify_alert(violation: dict) -> None:
    """Legacy: send Slack message for an ALERT violation. Prefer dispatch_violation_alert."""
    await _send_slack(violation, "MEDIUM")


async def notify_anomalies(
    flags: "list[AnomalyFlag]",
    session_id: str,
    agent_id: str,
) -> None:
    """
    Send Slack Block Kit message for HIGH or CRITICAL anomaly flags.
    Skips silently if ABB_SLACK_WEBHOOK_URL is not configured.
    """
    if not settings.slack_webhook_url:
        return
    notable = [f for f in flags if f.severity in ("high", "critical")]
    if not notable:
        return
    try:
        lines = [
            f":rotating_light: *Anomalies detected* — agent `{agent_id}`, session `{session_id}`"
        ]
        for flag in notable:
            emoji = ":red_circle:" if flag.severity == "critical" else ":large_orange_circle:"
            lines.append(
                f"{emoji} [{flag.severity.upper()}] `{flag.rule_id}`: {flag.description}"
            )
        payload = {"text": "\n".join(lines)}
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(settings.slack_webhook_url, json=payload)
        logger.info("Slack anomaly alert sent: %s", [f.rule_id for f in notable])
    except Exception as e:
        logger.warning("notify_anomalies failed (non-critical): %s", e)
