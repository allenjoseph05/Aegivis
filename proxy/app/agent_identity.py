"""
Agent identity validation for AgentBlackBox proxy.

Agents can optionally present a pre-shared key via the X-ABB-Agent-Key header.
The proxy validates it against configured keys and resolves the canonical agent_id.

Configuration (env vars):
    ABB_AGENT_KEYS=agent-alpha:key-abc123,agent-beta:key-def456

If ABB_AGENT_KEYS is empty or ABB_REQUIRE_AGENT_KEY=false (default), the proxy
accepts anonymous agents and uses X-ABB-Agent-ID or "unknown-agent" as the ID.

When ABB_REQUIRE_AGENT_KEY=true, requests without a valid key are rejected with 401.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import time

from .config import settings

logger = logging.getLogger(__name__)

# Parsed from ABB_AGENT_KEYS on first access
_key_map: dict[str, str] | None = None   # key -> agent_id


def _parse_key_map() -> dict[str, str]:
    """Parse ABB_AGENT_KEYS=agent1:key1,agent2:key2 into {key: agent_id}."""
    result: dict[str, str] = {}
    raw = settings.agent_keys.strip()
    if not raw:
        return result
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        agent_id, key = pair.split(":", 1)
        result[key.strip()] = agent_id.strip()
    return result


def _get_key_map() -> dict[str, str]:
    global _key_map
    if _key_map is None:
        _key_map = _parse_key_map()
    return _key_map


def validate_agent_key(
    agent_key: str | None,
    agent_id_header: str | None,
) -> tuple[bool, str, str | None]:
    """
    Validate X-ABB-Agent-Key against configured keys.

    Returns:
        (is_valid, resolved_agent_id, error_message)
        - is_valid: True if key is valid OR if key auth is not required
        - resolved_agent_id: canonical agent_id from key map, or from header, or "unknown-agent"
        - error_message: set only if invalid
    """
    key_map = _get_key_map()

    if agent_key:
        if agent_key in key_map:
            return True, key_map[agent_key], None
        else:
            if settings.require_agent_key:
                return False, "unknown", "Invalid X-ABB-Agent-Key"
            else:
                # Key provided but unknown — warn and continue as unidentified
                logger.warning(f"Unknown agent key presented, treating as anonymous")
                resolved = agent_id_header or "unknown-agent"
                return True, resolved, None
    else:
        if settings.require_agent_key and key_map:
            return False, "unknown", "X-ABB-Agent-Key header required"
        resolved = agent_id_header or "unknown-agent"
        return True, resolved, None


def generate_agent_key(agent_id: str) -> str:
    """Generate a new secure key for an agent (for local dev / testing)."""
    raw = f"{agent_id}:{time.time_ns()}:{secrets.token_hex(16)}"
    return "abb_" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def invalidate_key_cache() -> None:
    """Force re-parse of ABB_AGENT_KEYS on next call (after config reload)."""
    global _key_map
    _key_map = None
