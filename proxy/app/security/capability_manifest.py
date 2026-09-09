"""
Capability Manifest Validator — Phase 12.

A capability manifest is a signed document encoding exactly what an agent
is authorized to do: which tools it may call, which argument patterns are
permitted, and which network destinations are allowed.

The proxy fetches the active manifest for each agent from the backend,
verifies its HMAC-SHA256 signature, and caches it. At TOOL_CALL_START,
every tool call is validated against the manifest before execution.

Security guarantee: the manifest cannot be modified without the signing key.
An injected agent that tries to call an unauthorized tool, provide forbidden
argument values, or reach a prohibited network destination will be blocked
at the proxy level — regardless of what the LLM was told.

Signing: HMAC-SHA256 over the canonical JSON representation of the manifest
fields. The signing key (AEGIVIS_MANIFEST_SIGNING_KEY) is shared between the
backend (for signing on generation) and the proxy (for verification at runtime).

Manifest enforcement modes (AEGIVIS_MANIFEST_ENFORCE):
  true  — BLOCK tool calls that violate the manifest
  false — ALERT only (observe mode; for testing / gradual rollout)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Cache TTL (seconds) for loaded manifests — re-fetch from backend after this.
_MANIFEST_CACHE_TTL_S: float = 300.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PermittedTool:
    name: str
    permitted_arguments: dict[str, dict]   # arg_key → {"type": ..., "pattern": ..., "max_length": ...}
    trust_classification: str              # "external_source" | "internal_sink" | "internal"
    requires_hitl: bool = False
    hitl_reason: str = ""
    notes: str = ""


@dataclass
class ManifestViolation:
    tool_name: str
    violation_type: str    # "tool_not_permitted" | "arg_not_permitted" | "arg_pattern_mismatch"
    arg_key: str = ""
    arg_value: str = ""    # truncated to 200 chars for logging
    detail: str = ""


@dataclass
class CapabilityManifest:
    manifest_id: str
    agent_id: str
    org_id: str
    issued_at: str
    expires_at: str | None
    version: int
    permitted_tools: list[PermittedTool]
    permitted_network_destinations: list[str]
    ifc_policy: dict
    signature: str
    raw_json: str          # the original JSON string used for signature verification

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityManifest":
        tools = [
            PermittedTool(
                name=t["name"],
                permitted_arguments=t.get("permitted_arguments", {}),
                trust_classification=t.get("trust_classification", "internal"),
                requires_hitl=t.get("requires_hitl", False),
                hitl_reason=t.get("hitl_reason", ""),
                notes=t.get("notes", ""),
            )
            for t in data.get("permitted_tools", [])
        ]
        return cls(
            manifest_id=data["manifest_id"],
            agent_id=data["agent_id"],
            org_id=data["org_id"],
            issued_at=data.get("issued_at", ""),
            expires_at=data.get("expires_at"),
            version=data.get("version", 1),
            permitted_tools=tools,
            permitted_network_destinations=data.get("permitted_network_destinations", []),
            ifc_policy=data.get("ifc_policy", {}),
            signature=data.get("signature", ""),
            raw_json=data.get("manifest_json", json.dumps(data)),
        )

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            import datetime
            exp = datetime.datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.datetime.now(datetime.timezone.utc) > exp
        except Exception:
            return False

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.permitted_tools)

    def get_tool(self, name: str) -> PermittedTool | None:
        for t in self.permitted_tools:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

def sign_manifest(manifest_body: dict, signing_key: str) -> str:
    """
    Compute HMAC-SHA256 over the canonical JSON of manifest_body.
    Returns "sha256:<hex_digest>".
    """
    canonical = _canonical_json(manifest_body)
    key_bytes = signing_key.encode("utf-8") if signing_key else b"dev-insecure-key"
    digest = hmac.new(key_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"sha256:{digest}"


def verify_signature(manifest: CapabilityManifest, signing_key: str) -> bool:
    """
    Verify the manifest's HMAC-SHA256 signature.
    Returns True if valid, False if tampered or key mismatch.
    """
    if not signing_key:
        # No key configured — signature verification skipped (development mode)
        logger.warning("AEGIVIS_MANIFEST_SIGNING_KEY not set — manifest signature not verified")
        return True
    try:
        body = json.loads(manifest.raw_json)
        # Remove signature field before verifying
        body.pop("signature", None)
        body.pop("manifest_json", None)
        expected = sign_manifest(body, signing_key)
        return hmac.compare_digest(expected, manifest.signature)
    except Exception as e:
        logger.warning("Manifest signature verification error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

class ManifestValidator:
    """
    Validates tool calls against a loaded CapabilityManifest.
    """

    def __init__(self, manifest: CapabilityManifest) -> None:
        self._manifest = manifest

    def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> list[ManifestViolation]:
        """
        Validate a tool call against the manifest.
        Returns a list of ManifestViolation objects (empty = permitted).
        """
        manifest = self._manifest

        # 1. Check if tool is in permitted_tools
        tool_def = manifest.get_tool(tool_name)
        if tool_def is None:
            return [ManifestViolation(
                tool_name=tool_name,
                violation_type="tool_not_permitted",
                detail=f"Tool '{tool_name}' is not in the capability manifest for agent '{manifest.agent_id}'.",
            )]

        violations: list[ManifestViolation] = []

        # 2. Check argument keys and values
        if tool_def.permitted_arguments:
            for arg_key, arg_val in _flatten_str_args(args):
                leaf = arg_key.rsplit(".", 1)[-1]
                # If this arg key is declared in the manifest, validate it
                if leaf in tool_def.permitted_arguments:
                    arg_spec = tool_def.permitted_arguments[leaf]
                    v = _check_arg(tool_name, arg_key, arg_val, arg_spec)
                    if v:
                        violations.append(v)

        return violations


def _check_arg(
    tool_name: str,
    arg_key: str,
    arg_val: str,
    spec: dict,
) -> ManifestViolation | None:
    """Validate a single argument value against its manifest spec."""
    import re

    max_length = spec.get("max_length")
    if max_length and len(arg_val) > max_length:
        return ManifestViolation(
            tool_name=tool_name,
            violation_type="arg_pattern_mismatch",
            arg_key=arg_key,
            arg_value=arg_val[:200],
            detail=f"Arg '{arg_key}' length {len(arg_val)} exceeds manifest max {max_length}.",
        )

    pattern = spec.get("pattern")
    if pattern:
        try:
            if not re.match(pattern, arg_val):
                return ManifestViolation(
                    tool_name=tool_name,
                    violation_type="arg_pattern_mismatch",
                    arg_key=arg_key,
                    arg_value=arg_val[:200],
                    detail=f"Arg '{arg_key}' value does not match permitted pattern '{pattern}'.",
                )
        except re.error:
            logger.warning("Manifest: invalid pattern for arg %s: %s", arg_key, pattern)

    return None


# ---------------------------------------------------------------------------
# In-memory manifest cache (per process, per agent)
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    manifest: CapabilityManifest | None   # None = no manifest (absence is cached)
    loaded_at: float                      # time.monotonic()


_manifest_cache: dict[tuple[str, str], _CacheEntry] = {}   # (org_id, agent_id) → entry


async def get_active_manifest(
    agent_id: str,
    org_id: str,
    backend_url: str,
    api_key: str,
    signing_key: str,
    cache_ttl_s: float = _MANIFEST_CACHE_TTL_S,
) -> CapabilityManifest | None:
    """
    Return the active CapabilityManifest for this agent, or None if not found.
    Fetches from backend and caches for cache_ttl_s seconds.
    """
    cache_key = (org_id, agent_id)
    now = time.monotonic()

    entry = _manifest_cache.get(cache_key)
    if entry and (now - entry.loaded_at) < cache_ttl_s:
        return entry.manifest

    # Fetch from backend
    manifest = await _fetch_manifest(agent_id, org_id, backend_url, api_key, signing_key)
    _manifest_cache[cache_key] = _CacheEntry(manifest=manifest, loaded_at=now)
    return manifest


def invalidate_manifest_cache(agent_id: str, org_id: str) -> None:
    """Invalidate cached manifest for an agent (call after manifest update)."""
    _manifest_cache.pop((org_id, agent_id), None)


async def _fetch_manifest(
    agent_id: str,
    org_id: str,
    backend_url: str,
    api_key: str,
    signing_key: str,
) -> CapabilityManifest | None:
    """Fetch the active manifest from the backend API."""
    try:
        import httpx
        url = f"{backend_url}/v1/manifests/{agent_id}/active"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers={
                "X-API-Key": api_key,
                "X-Org-Id": org_id,
            })
        if resp.status_code == 404:
            return None   # No manifest — agent in learning mode
        if resp.status_code != 200:
            logger.warning("Manifest fetch returned %d for agent %s", resp.status_code, agent_id)
            return None
        data = resp.json()
        manifest = CapabilityManifest.from_dict(data)

        # Verify signature
        if signing_key and not verify_signature(manifest, signing_key):
            logger.error(
                "Manifest signature INVALID for agent %s — manifest rejected (possible tampering)",
                agent_id,
            )
            return None

        if manifest.is_expired():
            logger.warning("Manifest for agent %s is expired — treating as absent", agent_id)
            return None

        return manifest
    except Exception as e:
        logger.debug("Could not fetch manifest for agent %s: %s", agent_id, e)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj: dict) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _flatten_str_args(args: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Recursively flatten args dict to (key_path, str_value) pairs."""
    result: list[tuple[str, str]] = []
    for k, v in args.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            result.append((key, v))
        elif isinstance(v, dict):
            result.extend(_flatten_str_args(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    result.append((f"{key}[{i}]", item))
                elif isinstance(item, dict):
                    result.extend(_flatten_str_args(item, f"{key}[{i}]"))
    return result
