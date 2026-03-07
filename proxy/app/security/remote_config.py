"""
Community Remote Security Configuration — environment variables only.

Reads security feature flags from local ProxySettings (env vars).
No database or network calls. All security features are fully functional;
configuration is done via AEGIVIS_* environment variables.

The enterprise edition adds dashboard-managed per-org and per-agent overrides
via the ``org_settings`` and ``agent_security_profiles`` database tables. The
full DB-backed implementation lives in
``aegivis_enterprise.proxy_ext.remote_config``.

Usage (identical API to the enterprise edition):
    cfg = await get_security_config(org_id)
    if cfg.ifc_enabled:
        ...
    if cfg.egress_mode == "allowlist":
        ...

Priority in community:
  1. Local env var (ProxySettings / config.py) — only source available
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

# Simple in-process cache — env vars are read once and cached for the
# lifetime of the process (no TTL needed; env vars don't change at runtime).
_config_cache: dict[str, "SecurityConfig"] = {}


@dataclass
class SecurityConfig:
    """Resolved security feature flags for one request."""

    # Injection Detection
    injection_structural_enabled: bool = True
    injection_ml_async_enabled: bool = False
    injection_ml_sync_enabled: bool = False
    injection_ml_threshold: float = 0.85
    injection_rolling_window_enabled: bool = True
    tool_output_ml_scan_enabled: bool = False

    # Data Flow Control
    ifc_enabled: bool = True
    ifc_mode: str = "block"            # "block" | "alert"
    taint_tracking_enabled: bool = True
    taint_mode: str = "block"
    pdg_enabled: bool = True
    pdg_mode: str = "alert"

    # Access Control
    manifest_enforce_enabled: bool = True
    manifest_mode: str = "block"
    hitl_enabled: bool = False
    spawn_depth_max: int = 10
    rate_limit_tokens: int = 0
    rate_limit_llm_calls: int = 0

    # Content Protection
    pii_detection_enabled: bool = True
    unicode_stego_enabled: bool = True
    document_scan_enabled: bool = True
    canary_enabled: bool = True
    mcp_scanning_enabled: bool = True

    # Behavioral
    behavioral_enabled: bool = True
    system_prompt_mutation_enabled: bool = True
    system_prompt_mutation_mode: str = "block"


def _from_local_settings(settings) -> SecurityConfig:
    """Build a SecurityConfig from local ProxySettings (env vars)."""
    return SecurityConfig(
        injection_ml_async_enabled=settings.analysis_classifier_enabled,
        injection_ml_sync_enabled=settings.analysis_classifier_sync_enabled,
        injection_ml_threshold=settings.analysis_classifier_threshold,
        injection_rolling_window_enabled=True,
        tool_output_ml_scan_enabled=settings.security_tool_output_ml_scan_enabled,

        ifc_enabled=settings.security_ifc_enabled,
        ifc_mode="block" if settings.security_ifc_block else "alert",
        taint_tracking_enabled=settings.security_taint_tracking_enabled,

        manifest_enforce_enabled=settings.manifest_enforce,
        spawn_depth_max=settings.max_spawn_depth,
        rate_limit_tokens=settings.rate_limit_max_tokens_per_session,
        rate_limit_llm_calls=settings.rate_limit_max_llm_calls_per_session,

        pii_detection_enabled=settings.pii_enabled,
        unicode_stego_enabled=settings.security_unicode_stego_enabled,
        document_scan_enabled=settings.security_document_scan_enabled,
        canary_enabled=settings.security_canary_enabled,
        mcp_scanning_enabled=settings.security_mcp_scanning_enabled,
        behavioral_enabled=settings.security_behavioral_enabled,
    )


async def get_security_config(org_id: str, agent_id: str = "") -> SecurityConfig:
    """
    Return the effective SecurityConfig for *org_id* / *agent_id*.

    Community edition: reads from env vars only.
    org_id and agent_id are accepted for API compatibility with the enterprise
    edition but are not used — all orgs and agents share the same
    env-var-driven config in community mode.

    Enterprise edition adds per-org and per-agent DB overrides on top.
    """
    from ..config import settings as _settings

    # Cache key mirrors enterprise signature for drop-in compatibility
    cache_key = f"{org_id}:{agent_id}"
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    cfg = _from_local_settings(_settings)
    _config_cache[cache_key] = cfg
    return cfg


def invalidate_config_cache(org_id: str, agent_id: str = "") -> None:
    """Clear cached config entry.

    In community mode this is effectively a no-op since env vars are fixed,
    but the function is provided for API compatibility with the enterprise
    edition (which invalidates after dashboard changes).
    """
    cache_key = f"{org_id}:{agent_id}"
    _config_cache.pop(cache_key, None)
