"""Backend configuration via environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ABB_", case_sensitive=False)

    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://abb:abb@localhost:5432/agentblackbox"

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
    smtp_from: str = "alerts@agentblackbox.local"
    smtp_to: str = ""

    # Slack webhook alerts (optional — leave empty to disable)
    slack_webhook_url: str = ""

    # SIEM export — Splunk HEC (optional defaults; overridden per-request)
    splunk_hec_url:   str = ""
    splunk_hec_token: str = ""
    splunk_index:     str = "agentblackbox"

    # SIEM export — Elasticsearch (optional defaults; overridden per-request)
    elastic_url:      str = ""
    elastic_api_key:  str = ""
    elastic_index:    str = "agentblackbox"

    # -------------------------------------------------------------------------
    # Prometheus metrics (Phase 3.4)
    # -------------------------------------------------------------------------
    metrics_enabled: bool = True

    # -------------------------------------------------------------------------
    # OpenTelemetry distributed tracing (Phase 3.4)
    # -------------------------------------------------------------------------
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "agentblackbox-backend"

    # -------------------------------------------------------------------------
    # Redis (optional) — Redis Streams event pipeline + distributed state
    # Set ABB_REDIS_URL to enable (e.g. "redis://redis:6379/0")
    # -------------------------------------------------------------------------
    redis_url: str = ""


settings = BackendSettings()
