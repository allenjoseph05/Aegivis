"""Backend configuration via environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIVIS_", case_sensitive=False)

    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://aegivis:abb@localhost:5432/aegivis"

    # Auth
    api_keys: list[str] = ["dev-proxy-key", "dev-dashboard-key"]

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Pagination
    default_page_size: int = 50
    max_page_size: int = 200

    # SMTP email alerts (all optional — leave empty to disable)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "alerts@aegivis.local"
    smtp_to: str = ""

    # Slack webhook alerts (optional — leave empty to disable)
    slack_webhook_url: str = ""

    # PagerDuty Events API v2 (optional — leave empty to disable)
    pagerduty_routing_key: str = ""

    # Generic webhook alert channel (optional — leave empty to disable)
    alert_webhook_url: str = ""
    alert_webhook_secret: str = ""  # HMAC-SHA256 signing secret

    # Anti-fatigue: suppress duplicate alerts within this window (seconds)
    alert_cooldown_s: int = 600
    # Minimum severity to send external alerts ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    alert_min_severity: str = "MEDIUM"

    # SIEM export — Splunk HEC (optional defaults; overridden per-request)
    splunk_hec_url:   str = ""
    splunk_hec_token: str = ""
    splunk_index:     str = "aegivis"

    # SIEM export — Elasticsearch (optional defaults; overridden per-request)
    elastic_url:      str = ""
    elastic_api_key:  str = ""
    elastic_index:    str = "aegivis"

    # -------------------------------------------------------------------------
    # Prometheus metrics (Phase 3.4)
    # -------------------------------------------------------------------------
    metrics_enabled: bool = True

    # -------------------------------------------------------------------------
    # OpenTelemetry distributed tracing (Phase 3.4)
    # -------------------------------------------------------------------------
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "aegivis-backend"

    # -------------------------------------------------------------------------
    # Rate limiting — sliding-window counter keyed by API key or client IP
    # -------------------------------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 300          # read/general endpoints (GET /v1/*)
    rate_limit_ingest_per_minute: int = 3000  # ingest endpoints (POST /v1/events, violations)

    # -------------------------------------------------------------------------
    # Redis (optional) — Redis Streams event pipeline + distributed state
    # Set AEGIVIS_REDIS_URL to enable (e.g. "redis://redis:6379/0")
    # -------------------------------------------------------------------------
    redis_url: str = ""

    # -------------------------------------------------------------------------
    # Capability Manifests (Phase 12)
    # -------------------------------------------------------------------------
    # HMAC-SHA256 signing key for capability manifests.
    # Must match AEGIVIS_MANIFEST_SIGNING_KEY on the proxy.
    # If empty, signing/verification is skipped (development only).
    # Production: set to a 32+ char random string.
    # Example: AEGIVIS_MANIFEST_SIGNING_KEY=$(openssl rand -hex 32)
    manifest_signing_key: str = ""


settings = BackendSettings()
