"""
Notification service for policy violations and anomalies.

Sends email via SMTP for BLOCK events and Slack messages for ALERT/anomaly events.
All functions are fire-and-forget: they log warnings on failure but never
propagate exceptions to the caller.

Configure via environment variables:
    ABB_SMTP_HOST, ABB_SMTP_PORT, ABB_SMTP_USER, ABB_SMTP_PASSWORD,
    ABB_SMTP_FROM, ABB_SMTP_TO, ABB_SLACK_WEBHOOK_URL
"""
from __future__ import annotations

import asyncio
import email.mime.text
import logging
import smtplib
from typing import TYPE_CHECKING

import httpx

from ..config import settings

if TYPE_CHECKING:
    from .anomaly import AnomalyFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _send_smtp(subject: str, body: str) -> None:
    """Synchronous SMTP send — run in a thread pool executor."""
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


async def _post_slack(text: str) -> None:
    """POST a plain-text message to the configured Slack webhook."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            settings.slack_webhook_url,
            json={"text": text},
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def notify_block(violation: dict) -> None:
    """
    Send SMTP email for a BLOCK policy action.
    Skips silently if ABB_SMTP_HOST is not configured.
    """
    if not settings.smtp_host or not settings.smtp_to:
        return
    try:
        rule = violation.get("rule_name", "unknown")
        reason = violation.get("reason", "")
        session_id = violation.get("session_id", "")
        agent_id = violation.get("agent_id", "")
        subject = f"[AgentBlackBox] BLOCKED: {rule}"
        body = (
            f"AgentBlackBox policy engine blocked an LLM call.\n\n"
            f"Rule:       {rule}\n"
            f"Reason:     {reason}\n"
            f"Agent:      {agent_id}\n"
            f"Session:    {session_id}\n"
            f"Event type: {violation.get('event_type', '')}\n"
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send_smtp, subject, body)
        logger.info(f"SMTP alert sent for BLOCK rule={rule}")
    except Exception as e:
        logger.warning(f"SMTP notify_block failed (non-critical): {e}")


async def notify_alert(violation: dict) -> None:
    """
    Send Slack message for an ALERT policy action.
    Skips silently if ABB_SLACK_WEBHOOK_URL is not configured.
    """
    if not settings.slack_webhook_url:
        return
    try:
        rule = violation.get("rule_name", "unknown")
        reason = violation.get("reason", "")
        agent_id = violation.get("agent_id", "")
        session_id = violation.get("session_id", "")
        text = (
            f":warning: *ALERT*: `{rule}` fired for agent `{agent_id}`\n"
            f"Reason: {reason}\n"
            f"Session: `{session_id}`"
        )
        await _post_slack(text)
        logger.info(f"Slack alert sent for ALERT rule={rule}")
    except Exception as e:
        logger.warning(f"Slack notify_alert failed (non-critical): {e}")


async def notify_anomalies(
    flags: "list[AnomalyFlag]",
    session_id: str,
    agent_id: str,
) -> None:
    """
    Send Slack message for HIGH or CRITICAL anomaly flags.
    Skips silently if ABB_SLACK_WEBHOOK_URL is not configured.
    """
    if not settings.slack_webhook_url:
        return
    notable = [f for f in flags if f.severity in ("high", "critical")]
    if not notable:
        return
    try:
        lines = [f":rotating_light: *Anomalies detected* — agent `{agent_id}`, session `{session_id}`"]
        for flag in notable:
            emoji = ":red_circle:" if flag.severity == "critical" else ":large_orange_circle:"
            lines.append(f"{emoji} [{flag.severity.upper()}] `{flag.rule_id}`: {flag.description}")
        await _post_slack("\n".join(lines))
        logger.info(f"Slack anomaly alert sent: {[f.rule_id for f in notable]}")
    except Exception as e:
        logger.warning(f"Slack notify_anomalies failed (non-critical): {e}")
