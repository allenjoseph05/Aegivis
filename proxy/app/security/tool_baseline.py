"""
Tool Baseline Enforcement — detects new / unexpected tools mid-session
and enforces operator-approved tool sets at TOOL_CALL_START.

How it works
------------
Every LLM API call (OpenAI, Anthropic, Google, etc.) carries a ``tools[]``
array that declares exactly which tools the model is allowed to invoke.
This is an API-level guarantee: the model physically cannot call a tool
that is not listed in that array.

We exploit this invariant in two ways:

1. **Mid-session tool-set mutation detection** (zero first-call risk)
   On the first API call in a session we hash the tools[] array and store
   it on the session.  Every subsequent call recomputes the hash and
   compares.  If the set changed (tool added, removed, or schema changed)
   we emit a BLOCK violation immediately — before any tool call is made.

   Attack scenarios caught:
   - Prompt injection convincing the orchestrator to add a new tool
   - MCP server poisoning that injects extra tool definitions
   - Compromised orchestrator swapping its own tool registry mid-flight

2. **Approved-baseline enforcement** (at TOOL_CALL_START)
   Operators review the discovered tools per agent in the dashboard and
   mark each one as *approved* or *denied*.  Once a baseline exists, any
   tool call that uses a tool name not in the approved set is blocked.

   Tools are reported to the backend asynchronously (fire-and-forget) so
   the operator can review them.  The approved list is cached locally in
   the proxy process for ``tool_baseline_cache_ttl_s`` seconds to avoid a
   round-trip on every request.

Audit mode
----------
Agents with no approved baseline in the database are in **audit mode**:
tool calls are observed and reported but not blocked.  Operators switch
an agent from audit → enforce by approving its tool baseline in the
dashboard.

Provider compatibility
----------------------
Both OpenAI-style (``tool.function.name``) and Anthropic/MCP-style
(``tool.name``) tool definitions are handled transparently.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool normalisation — handles OpenAI and Anthropic/MCP formats
# ---------------------------------------------------------------------------

def _normalize_tool(tool: dict) -> dict | None:
    """
    Return a canonical ``{name, description, schema}`` dict for a single
    tool definition regardless of provider format.  Returns None if the
    tool has no name.
    """
    if not isinstance(tool, dict):
        return None

    if "function" in tool:
        # OpenAI / OpenAI-compatible format:
        # {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        fn = tool["function"]
        name = fn.get("name", "")
        desc = fn.get("description", "")
        schema = fn.get("parameters", {})
    else:
        # Anthropic / MCP format:
        # {"name": ..., "description": ..., "input_schema": ...}
        name = tool.get("name", "")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", tool.get("parameters", {}))

    if not name:
        return None
    return {"name": name, "description": desc, "schema": schema}


def extract_tools(body: dict) -> list[dict]:
    """
    Return the raw tools[] list from a request body, or [] if absent.
    Works for all providers — they all use the key ``tools``.
    """
    tools = body.get("tools")
    return tools if isinstance(tools, list) else []


def extract_tool_names(tools: list[dict]) -> set[str]:
    """Return the set of tool names from a tools[] array."""
    names: set[str] = set()
    for t in tools:
        n = _normalize_tool(t)
        if n:
            names.add(n["name"])
    return names


def hash_tool_set(tools: list[dict]) -> str:
    """
    Compute a stable, order-independent SHA-256[:16] fingerprint of a
    tools[] array.

    The hash covers both the tool names AND their parameter schemas so
    that a schema change (same name, different parameters) is also
    detected as a mutation.
    """
    normalized: dict[str, Any] = {}
    for t in tools:
        n = _normalize_tool(t)
        if n:
            normalized[n["name"]] = n["schema"]
    canonical = json.dumps(
        {k: normalized[k] for k in sorted(normalized)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def normalize_tools_for_storage(tools: list[dict]) -> list[dict]:
    """
    Return a list of normalized tool dicts suitable for database storage
    (name + description + schema).  Used when reporting observed tools to
    the backend.
    """
    result = []
    for t in tools:
        n = _normalize_tool(t)
        if n:
            result.append(n)
    return result


# ---------------------------------------------------------------------------
# Approved-baseline cache
# ---------------------------------------------------------------------------

# Cache entry: (approved_tool_names | None, cached_at_monotonic)
# None means "no baseline exists" (audit mode).
_baseline_cache: dict[str, tuple[frozenset[str] | None, float]] = {}
_CACHE_SENTINEL = object()   # distinguishes "not in cache" from cached-None


def invalidate_cache(agent_id: str, org_id: str) -> None:
    """Remove a cached baseline entry (e.g. after an approval decision)."""
    _baseline_cache.pop(f"{org_id}:{agent_id}", None)


async def get_approved_baseline(
    agent_id: str,
    org_id: str,
    backend_url: str,
    api_key: str,
    cache_ttl_s: float = 300.0,
) -> frozenset[str] | None:
    """
    Return the set of approved tool names for *agent_id*, or None if no
    baseline has been approved yet (audit mode).

    Results are cached for *cache_ttl_s* seconds.  On any backend error
    we fail **open** (return None) so that a backend outage does not block
    legitimate agent traffic.

    Args:
        agent_id:    The agent whose baseline to look up.
        org_id:      Organisation identifier (for multi-tenancy).
        backend_url: Base URL of the Aegivis backend.
        api_key:     API key for backend authentication.
        cache_ttl_s: Seconds before the cached entry expires.

    Returns:
        frozenset of approved tool names, or None (audit mode / error).
    """
    import httpx

    cache_key = f"{org_id}:{agent_id}"
    now = time.monotonic()

    cached = _baseline_cache.get(cache_key)
    if cached is not None and (now - cached[1]) < cache_ttl_s:
        return cached[0]

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                f"{backend_url}/v1/baselines/{agent_id}/tools/approved",
                headers={"X-API-Key": api_key, "X-Org-Id": org_id},
            )
            if resp.status_code == 404:
                # No baseline exists → audit mode
                _baseline_cache[cache_key] = (None, now)
                return None
            resp.raise_for_status()
            data = resp.json()
            approved = frozenset(data.get("approved_tools", []))
            _baseline_cache[cache_key] = (approved, now)
            return approved
    except Exception as exc:
        logger.debug(
            "Baseline cache fetch failed for agent=%s org=%s (fail-open): %s",
            agent_id, org_id, exc,
        )
        # Fail open: do not block traffic on backend error
        return None


# ---------------------------------------------------------------------------
# Backend reporting (fire-and-forget)
# ---------------------------------------------------------------------------

async def report_tools_observed(
    agent_id: str,
    org_id: str,
    tools: list[dict],
    session_id: str,
    backend_url: str,
    api_key: str,
) -> None:
    """
    Notify the backend about the tools seen in a session so the operator
    can review and approve them.

    Called asynchronously via ``asyncio.create_task()`` — failures are
    logged at DEBUG level and never propagate.

    Args:
        agent_id:    The agent that owns this session.
        org_id:      Organisation identifier.
        tools:       Raw tools[] from the request body.
        session_id:  Session that first surfaced these tools.
        backend_url: Base URL of the Aegivis backend.
        api_key:     API key for backend authentication.
    """
    import httpx

    normalized = normalize_tools_for_storage(tools)
    if not normalized:
        return

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{backend_url}/v1/baselines/{agent_id}/tools/observe",
                headers={"X-API-Key": api_key, "X-Org-Id": org_id},
                json={"tools": normalized, "session_id": session_id, "org_id": org_id},
            )
        logger.debug(
            "Reported %d tools for agent=%s session=%s",
            len(normalized), agent_id, session_id,
        )
    except Exception as exc:
        logger.debug(
            "Tool observation report failed for agent=%s (non-critical): %s",
            agent_id, exc,
        )
