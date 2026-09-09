"""
GET  /v1/settings       — Get org notification/integration settings (DB overlaid on env defaults)
PATCH /v1/settings      — Update one or more settings keys
POST /v1/settings/test  — Fire a test alert on a specific channel
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Allowlisted keys — only these may be persisted (prevents arbitrary injection)
# ---------------------------------------------------------------------------

# Keys whose value must be a valid integer
_INTEGER_KEYS: frozenset[str] = frozenset({"smtp_port", "alert_cooldown_s"})
# Keys whose value must be one of the listed options
_ENUM_KEYS: dict[str, frozenset[str]] = {
    "alert_min_severity": frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"}),
}

ALLOWED_KEYS: frozenset[str] = frozenset({
    "slack_webhook_url",
    "smtp_host", "smtp_port", "smtp_user", "smtp_password", "smtp_from", "smtp_to",
    "pagerduty_routing_key",
    "alert_webhook_url", "alert_webhook_secret",
    "alert_min_severity", "alert_cooldown_s",
    "splunk_hec_url", "splunk_hec_token",
    "elastic_url", "elastic_api_key",
})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SettingsUpdateRequest(BaseModel):
    updates: dict[str, str | None]


class TestChannelRequest(BaseModel):
    channel: str  # "slack" | "email" | "pagerduty" | "webhook"


# ---------------------------------------------------------------------------
# Helper: load org settings from DB, merge with env defaults
# ---------------------------------------------------------------------------

async def _load_effective(org_id: str, db: AsyncSession) -> dict:
    from ...config import settings as _s

    result = await db.execute(
        text("SELECT key, value FROM org_settings WHERE org_id = :org_id"),
        {"org_id": org_id},
    )
    overrides: dict[str, str | None] = {r.key: r.value for r in result.mappings().all()}

    def _ov(k: str, default: str = "") -> str:
        v = overrides.get(k)
        return v if v is not None else (getattr(_s, k, None) or default)

    return {
        "slack_webhook_url":      _ov("slack_webhook_url"),
        "smtp_host":              _ov("smtp_host"),
        "smtp_port":              _ov("smtp_port", str(getattr(_s, "smtp_port", 587))),
        "smtp_user":              _ov("smtp_user"),
        "smtp_password":          _ov("smtp_password"),
        "smtp_from":              _ov("smtp_from"),
        "smtp_to":                _ov("smtp_to"),
        "pagerduty_routing_key":  _ov("pagerduty_routing_key"),
        "alert_webhook_url":      _ov("alert_webhook_url"),
        "alert_webhook_secret":   _ov("alert_webhook_secret"),
        "alert_min_severity":     _ov("alert_min_severity", "MEDIUM"),
        "alert_cooldown_s":       _ov("alert_cooldown_s", "600"),
        "splunk_hec_url":         _ov("splunk_hec_url"),
        "splunk_hec_token":       _ov("splunk_hec_token"),
        "elastic_url":            _ov("elastic_url"),
        "elastic_api_key":        _ov("elastic_api_key"),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/settings", summary="Get org notification/integration settings")
async def get_settings(
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    effective = await _load_effective(org_ctx.org_id, db)
    return {"settings": effective}


@router.patch("/settings", summary="Update org settings keys")
async def update_settings(
    body: SettingsUpdateRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    bad_keys = set(body.updates.keys()) - ALLOWED_KEYS
    if bad_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown settings key(s): {sorted(bad_keys)}",
        )

    # Validate value types for keys with known constraints
    for key, value in body.updates.items():
        if value is None:
            continue
        if key in _INTEGER_KEYS:
            try:
                int(value)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Key '{key}' requires an integer value, got: {value!r}",
                )
        if key in _ENUM_KEYS:
            allowed = _ENUM_KEYS[key]
            if value not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Key '{key}' must be one of {sorted(allowed)}, got: {value!r}",
                )

    updated = 0
    for key, value in body.updates.items():
        if value is None:
            # Delete key — revert to env default
            await db.execute(
                text("DELETE FROM org_settings WHERE org_id = :org_id AND key = :key"),
                {"org_id": org_ctx.org_id, "key": key},
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO org_settings (org_id, key, value, updated_at)
                    VALUES (:org_id, :key, :value, NOW())
                    ON CONFLICT (org_id, key)
                    DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """),
                {"org_id": org_ctx.org_id, "key": key, "value": value},
            )
        updated += 1

    await db.commit()
    logger.info("Org settings updated: org=%s keys=%s", org_ctx.org_id, list(body.updates.keys()))
    return {"status": "ok", "updated_count": updated}


@router.post("/settings/test", summary="Fire a test alert on a notification channel")
async def test_channel(
    body: TestChannelRequest,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    valid_channels = {"slack", "email", "pagerduty", "webhook"}
    if body.channel not in valid_channels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown channel '{body.channel}'. Must be one of: {sorted(valid_channels)}",
        )

    cfg = await _load_effective(org_ctx.org_id, db)

    test_violation = {
        "rule_name": "test-notification",
        "action": "ALERT",
        "reason": "This is a test notification from Aegivis",
        "agent_id": "test-agent",
        "session_id": "test-session",
        "timestamp_ns": 0,
    }

    try:
        from ...services.notifications import (
            _send_slack, _send_email, _send_pagerduty, _send_webhook,
        )

        if body.channel == "slack":
            url = cfg.get("slack_webhook_url", "")
            if not url:
                return {"ok": False, "message": "Slack webhook URL is not configured"}
            import httpx
            payload = {
                "text": ":white_check_mark: *Aegivis test notification* — Slack is configured correctly."
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code in (200, 204):
                return {"ok": True, "message": "Slack test message sent successfully"}
            return {"ok": False, "message": f"Slack returned HTTP {resp.status_code}: {resp.text[:100]}"}

        elif body.channel == "email":
            if not cfg.get("smtp_host") or not cfg.get("smtp_to"):
                return {"ok": False, "message": "SMTP host and recipient address are not configured"}
            await _send_email(test_violation, "MEDIUM")
            return {"ok": True, "message": "Test email dispatched (check inbox)"}

        elif body.channel == "pagerduty":
            if not cfg.get("pagerduty_routing_key"):
                return {"ok": False, "message": "PagerDuty routing key is not configured"}
            await _send_pagerduty(test_violation, "MEDIUM")
            return {"ok": True, "message": "Test PagerDuty event sent"}

        elif body.channel == "webhook":
            url = cfg.get("alert_webhook_url", "")
            if not url:
                return {"ok": False, "message": "Webhook URL is not configured"}
            import httpx, json, hmac
            payload = {"severity": "MEDIUM", "violation": test_violation, "source": "aegivis-test"}
            headers: dict[str, str] = {"Content-Type": "application/json"}
            secret = cfg.get("alert_webhook_secret", "")
            if secret:
                body_bytes = json.dumps(payload).encode()
                sig = hmac.new(secret.encode(), body_bytes, "sha256").hexdigest()
                headers["X-Aegivis-Signature"] = f"sha256={sig}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (200, 201, 202, 204):
                return {"ok": True, "message": "Webhook test payload delivered successfully"}
            return {"ok": False, "message": f"Webhook returned HTTP {resp.status_code}: {resp.text[:100]}"}

    except Exception as exc:
        logger.warning("Test notification failed: channel=%s error=%s", body.channel, exc)
        return {"ok": False, "message": str(exc)}
