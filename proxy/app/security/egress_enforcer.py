"""
Hard Egress Enforcement — Phase 13.

Replaces the probabilistic SSRF scanner with a deterministic allowlist-based
network destination checker. If a tool call's network destination is not in
the operator's approved list, the call is BLOCKED — not alerted, BLOCKED.

This is the last proxy-level backstop before data reaches the network:
  - Even if every other layer fails (classifier missed it, IFC label was wrong)
  - Even if injection fully succeeded and the agent is 100% compromised
  - If the destination hostname is not in the allowlist → BLOCK

The check is a simple set membership test. It does not read content.
It cannot be prompt-injected because it does not process text instructions.

Allowlist sources (merged, highest-priority first):
  1. Dashboard rules (GET /v1/egress-rules, 60s cache per org)
  2. AEGIVIS_SECURITY_EGRESS_ALLOWLIST env var (static fallback)

Dashboard rules take effect within one cache TTL of being saved.
If the backend is unreachable, the cached list (or env var) is used.

Destination formats:
    api.openai.com           — exact hostname
    *.anthropic.com          — wildcard subdomain (all subdomains match)
    db.internal:5432         — internal hostname with port
    192.168.1.100            — internal IP (exact match)
    10.0.0.0/8               — CIDR (IPv4 ranges)

Network destination args extracted from:
  url, endpoint, webhook_url, webhook, api_url, base_url,
  callback_url, callback, uri, host, server, destination, target

When AEGIVIS_SECURITY_EGRESS_MODE=audit: logs violations but does not block.
When AEGIVIS_SECURITY_EGRESS_MODE=allowlist (default): blocks unlisted destinations.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument keys that represent network destinations
# ---------------------------------------------------------------------------

_NETWORK_ARG_KEYS: frozenset[str] = frozenset({
    "url", "endpoint", "webhook_url", "webhook", "api_url", "base_url",
    "callback_url", "callback", "uri", "host", "server", "destination",
    "target", "addr", "address",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EgressViolation:
    tool_name: str
    arg_key: str
    destination: str        # the hostname/IP found in the arg
    raw_value: str          # full arg value (truncated for logging)


# ---------------------------------------------------------------------------
# EgressEnforcer
# ---------------------------------------------------------------------------

class EgressEnforcer:
    """
    Deterministic allowlist-based network destination checker.

    Loaded from the operator's configured allowlist. Checks every tool call
    argument that looks like a network destination against the allowlist.
    If a destination is not listed → violation (block or alert per mode).
    """

    def __init__(self, allowed: list[str]) -> None:
        """
        Args:
            allowed: list of allowlist entries (hostname, wildcard, CIDR, etc.)
        """
        self._exact: set[str] = set()           # exact hostname[:port] matches (lowercased)
        self._host_only: set[str] = set()       # hostname without port (lowercased), from exact entries
        self._wildcard_domains: list[str] = []   # "*.example.com" → ".example.com" (lowercased)
        self._cidr_ranges: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._compile(allowed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True when no allowlist entries are configured (audit mode only)."""
        return (
            not self._exact
            and not self._wildcard_domains
            and not self._cidr_ranges
        )

    def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> list[EgressViolation]:
        """
        Scan tool arguments for network destinations and check each against
        the allowlist.

        Returns a list of EgressViolation objects (empty = all allowed).
        """
        violations: list[EgressViolation] = []
        for arg_key, arg_val in _flatten_args(args):
            leaf = _leaf_key(arg_key)
            if leaf not in _NETWORK_ARG_KEYS:
                continue
            if not isinstance(arg_val, str) or len(arg_val) < 4:
                continue
            dest = _extract_hostname(arg_val)
            if dest is None:
                continue
            if not self._is_allowed(dest):
                violations.append(EgressViolation(
                    tool_name=tool_name,
                    arg_key=arg_key,
                    destination=dest,
                    raw_value=arg_val[:200],
                ))
        return violations

    def allowed_list(self) -> list[str]:
        """Return all configured allowlist entries (for diagnostics)."""
        entries = list(self._exact)
        entries += [f"*.{d.lstrip('.')}" for d in self._wildcard_domains]
        return sorted(entries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compile(self, entries: list[str]) -> None:
        for raw in entries:
            entry = raw.strip().lower()   # normalize to lowercase at compile time
            if not entry:
                continue
            # CIDR range (e.g. 10.0.0.0/8)
            if "/" in entry:
                try:
                    self._cidr_ranges.append(ipaddress.ip_network(entry, strict=False))
                    continue
                except ValueError:
                    pass
            # Wildcard subdomain (e.g. *.example.com)
            if entry.startswith("*."):
                self._wildcard_domains.append(entry[1:])  # store as ".example.com" (already lower)
                continue
            # Exact match (hostname or hostname:port) — store both full and host-only forms
            self._exact.add(entry)
            self._host_only.add(entry.split(":")[0])   # allow bare hostname when port entry exists

    def _is_allowed(self, destination: str) -> bool:
        """Return True if the destination is in the allowlist."""
        dest_lower = destination.lower()
        # Compute host_only (strip port). For IPv6 like "2001:db8::1", splitting
        # on ":" gives the wrong result — detect this case with ip_address().
        try:
            ipaddress.ip_address(dest_lower)
            host_only = dest_lower  # bare IP address — no port to strip
        except ValueError:
            host_only = dest_lower.split(":")[0]  # hostname or hostname:port

        # 1. Exact match (full "host:port" or bare "host")
        if dest_lower in self._exact:
            return True
        # 2. Host-only match (allowlist has "host:port" but destination is bare "host")
        if host_only in self._host_only:
            return True

        # 2. Wildcard subdomain match
        for suffix in self._wildcard_domains:
            # ".example.com" matches "api.example.com" and "example.com"
            if dest_lower.endswith(suffix) or dest_lower == suffix.lstrip("."):
                return True
            if host_only.endswith(suffix) or host_only == suffix.lstrip("."):
                return True

        # 3. CIDR range match (IP destinations only)
        if self._cidr_ranges:
            try:
                ip = ipaddress.ip_address(host_only)
                for net in self._cidr_ranges:
                    if ip in net:
                        return True
            except ValueError:
                pass  # not an IP address

        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_hostname(value: str) -> str | None:
    """
    Extract the hostname (and optional port) from a URL or bare hostname string.

    Returns None if the value does not appear to be a network destination.
    """
    value = value.strip()
    if not value:
        return None

    # Try parsing as a URL first
    if "://" in value:
        try:
            parsed = urlparse(value)
            if parsed.hostname:
                host = parsed.hostname
                if parsed.port:
                    return f"{host}:{parsed.port}"
                return host
        except Exception:
            pass

    # Bare hostname or hostname:port (no scheme)
    if re.match(r'^[a-zA-Z0-9.\-]+(:\d+)?$', value):
        return value.lower()

    # IP address possibly with port
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}(:\d+)?$', value):
        return value

    return None


def _leaf_key(key_path: str) -> str:
    """Extract the leaf key name from a dotted/indexed key path."""
    key_path = re.sub(r'\[\d+\]$', '', key_path)
    return key_path.rsplit(".", 1)[-1]


def _flatten_args(args: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Recursively flatten args dict to (key_path, value) pairs."""
    result: list[tuple[str, str]] = []
    for k, v in args.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            result.append((key, v))
        elif isinstance(v, dict):
            result.extend(_flatten_args(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    result.append((f"{key}[{i}]", item))
                elif isinstance(item, dict):
                    result.extend(_flatten_args(item, f"{key}[{i}]"))
    return result


# ---------------------------------------------------------------------------
# Dashboard-backed allowlist cache
# ---------------------------------------------------------------------------

_CACHE_TTL_S: float = 60.0

# (org_id) → (destinations: list[str], loaded_at: float)
_allowlist_cache: dict[str, tuple[list[str], float]] = {}


async def get_egress_allowlist(org_id: str) -> list[str]:
    """
    Return the merged egress allowlist for *org_id*.

    Priority:
      1. Dashboard rules (GET /v1/egress-rules, 60s cache)
      2. AEGIVIS_SECURITY_EGRESS_ALLOWLIST env var (static, always included)

    Falls back to env-var-only if the backend is unreachable.
    """
    from ..config import settings

    env_entries = [
        e.strip() for e in settings.security_egress_allowlist.split(",") if e.strip()
    ]

    now = time.monotonic()
    cached = _allowlist_cache.get(org_id)
    if cached and (now - cached[1]) < _CACHE_TTL_S:
        db_entries = cached[0]
    else:
        db_entries = await _fetch_from_backend(org_id, settings)
        _allowlist_cache[org_id] = (db_entries, now)

    # Merge: dashboard rules + env var (deduplicated, order preserved)
    seen: set[str] = set()
    merged: list[str] = []
    for e in db_entries + env_entries:
        if e not in seen:
            seen.add(e)
            merged.append(e)
    return merged


async def _fetch_from_backend(org_id: str, settings) -> list[str]:
    """Fetch egress rules from backend API. Returns [] on any error."""
    try:
        import httpx
        url = f"{settings.backend_url}/v1/egress-rules"
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url, headers={
                "X-API-Key": settings.api_keys[0] if settings.api_keys else "dev-dashboard-key",
                "X-Org-Id": org_id,
            })
        if resp.status_code == 200:
            data = resp.json()
            return [r["destination"] for r in data.get("rules", [])]
        _logger.debug("Egress rules fetch returned %d for org %s", resp.status_code, org_id)
    except Exception as exc:
        _logger.debug("Egress rules fetch failed for org %s: %s", org_id, exc)
    return []


def invalidate_egress_cache(org_id: str) -> None:
    """Invalidate cached egress rules for an org (call after rule change)."""
    _allowlist_cache.pop(org_id, None)
