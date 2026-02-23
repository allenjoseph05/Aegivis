"""Proxy configuration via environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProxySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ABB_", case_sensitive=False)

    # Backend ingestion endpoint
    backend_url: str = "http://localhost:8000"
    backend_api_key: str = "dev-proxy-key"

    # Organization (used in event envelope)
    org_id: str = "default-org"

    # Proxy server
    host: str = "0.0.0.0"
    port: int = 8080

    # Real LLM provider URLs (defaults are the public APIs)
    openai_upstream: str = "https://api.openai.com"
    anthropic_upstream: str = "https://api.anthropic.com"
    google_upstream: str = "https://generativelanguage.googleapis.com"
    azure_upstream: str = ""         # set to https://{resource}.openai.azure.com
    ollama_upstream: str = "http://localhost:11434"

    # Batch transport
    batch_size: int = 10
    batch_flush_interval_s: float = 2.0
    transport_timeout_s: float = 10.0

    # PII settings
    pii_enabled: bool = True
    pii_language: str = "en"

    # Hash chain
    checkpoint_interval: int = 1000   # CHECKPOINT every N events per session

    # Policy engine
    # Path to a custom YAML policy file. If empty, uses proxy/app/policies/default.yaml.
    policy_yaml: str = ""
    # Comma-separated list of rules to disable by name (e.g. "high-latency-alert,pii-in-llm-request")
    policy_disabled_rules: str = ""

    # Agent identity
    # Comma-separated agent_id:key pairs. e.g. "my-agent:abb_abc123,other-agent:abb_def456"
    agent_keys: str = ""
    # If true, requests without a valid X-ABB-Agent-Key are rejected with 401
    require_agent_key: bool = False

    # Violations reporting — send policy violations to backend
    violations_enabled: bool = True

    # Reliability: local disk buffer when backend is unreachable
    # Path to a SQLite file used to persist unsent events. Empty = disabled.
    buffer_db_path: str = "abb_buffer.db"   # relative to CWD; set "" to disable
    buffer_retry_interval_s: float = 30.0   # how often to retry flushing buffered events
    buffer_max_events: int = 50_000         # max events to buffer before dropping

    # Reliability: session state persistence across proxy restarts
    # Path to a JSON file for session state. Empty = disabled (in-memory only).
    session_state_path: str = "abb_sessions.json"

    # Logging
    log_level: str = "INFO"
    log_request_bodies: bool = False  # set True only in dev, never in prod


settings = ProxySettings()
