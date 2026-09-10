"""Live Model Shadowing — fire-and-forget shadow call after primary response.

Runs completely out of the hot path: the primary response is already sent to
the agent before this module is called. Uses asyncio.create_task() so there
is zero latency impact on the agent.

What it does:
  1. Swaps `model` in the original request body to the configured shadow model
  2. Makes the same HTTP call to the same provider upstream
  3. Extracts tokens, tool calls, and response length
  4. Computes: cost delta, latency delta, response-length ratio, tool-call agreement
  5. POSTs the comparison to the backend for storage and dashboard display

Only non-streaming requests are shadowed (streaming assembled calls are TODO).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# ── Inline rate table (USD per 1M tokens, input/output) ──────────────────────
# Duplicated from backend cost_analyzer intentionally — proxy has no backend deps.

_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":       (0.15,  0.60),
    "gpt-4o":            (2.50,  10.00),
    "gpt-4-turbo":       (10.00, 30.00),
    "gpt-4":             (30.00, 60.00),
    "gpt-3.5-turbo":     (0.50,  1.50),
    "claude-opus-4":     (15.00, 75.00),
    "claude-sonnet-4":   (3.00,  15.00),
    "claude-haiku-4":    (0.25,  1.25),
    "claude-3-5-sonnet": (3.00,  15.00),
    "claude-3-5-haiku":  (0.80,  4.00),
    "claude-3-haiku":    (0.25,  1.25),
    "claude-3-sonnet":   (3.00,  15.00),
    "claude-3-opus":     (15.00, 75.00),
    "llama":             (0.0,   0.0),
    "mistral":           (0.0,   0.0),
    "mixtral":           (0.0,   0.0),
    "gemma":             (0.0,   0.0),
}
_DEFAULT_RATE = (1.00, 3.00)
_INPUT_SPLIT  = 0.70
_OUTPUT_SPLIT = 0.30


def _get_rate(model: str) -> tuple[float, float]:
    m = model.lower()
    for key, rate in _RATES.items():
        if key in m:
            return rate
    return _DEFAULT_RATE


def _estimate_cost(tokens: int, model: str) -> float:
    in_r, out_r = _get_rate(model)
    return (tokens * _INPUT_SPLIT * in_r + tokens * _OUTPUT_SPLIT * out_r) / 1_000_000


# ── Response parsing (provider-agnostic) ──────────────────────────────────────

def _extract_tokens(body: dict) -> int:
    usage = body.get("usage") or {}
    if "total_tokens" in usage:
        return int(usage["total_tokens"] or 0)
    # Anthropic format
    return int((usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0))


def _extract_tool_calls(body: dict) -> list[str]:
    # OpenAI format
    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        tcs = msg.get("tool_calls") or []
        if tcs:
            return [
                (tc.get("function") or {}).get("name") or tc.get("name") or ""
                for tc in tcs
            ]
    # Anthropic format
    content = body.get("content") or []
    return [
        b.get("name", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def _extract_response_len(body: dict) -> int:
    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        return len(msg.get("content") or "")
    content = body.get("content") or []
    return sum(len(b.get("text") or "") for b in content if isinstance(b, dict))


def _primary_tool_names(tool_calls: list[dict]) -> list[str]:
    """Extract names from the primary parsed_resp tool_calls list."""
    names = []
    for tc in (tool_calls or []):
        name = (
            tc.get("name")
            or (tc.get("function") or {}).get("name")
            or ""
        )
        if name:
            names.append(name)
    return names


# ── Main shadow task ──────────────────────────────────────────────────────────

async def fire_shadow_comparison(
    *,
    org_id: str,
    session_id: str,
    agent_id: str,
    provider: str,
    upstream_url: str,
    forward_headers: dict[str, str],
    request_body_bytes: bytes,
    primary_model: str,
    primary_latency_ms: float,
    primary_tokens: int,
    primary_tool_calls: list[dict],
    primary_response_len: int,
) -> None:
    """
    Fire-and-forget shadow comparison task.
    Must be called via asyncio.create_task() — never awaited directly on the hot path.
    """
    shadow_model = settings.shadow_model
    if not shadow_model or not upstream_url:
        return

    # ── Swap model in request body ────────────────────────────────────────────
    try:
        body = json.loads(request_body_bytes)
    except Exception:
        return

    body["model"] = shadow_model
    # Remove stream flag — always do non-streaming shadow call
    body.pop("stream", None)
    shadow_bytes = json.dumps(body).encode("utf-8")

    # Build headers — strip content-length (httpx recomputes)
    shadow_headers = {
        k: v for k, v in forward_headers.items()
        if k.lower() not in ("content-length",)
    }

    # ── Make shadow call ──────────────────────────────────────────────────────
    t_start = time.time()
    shadow_error: str | None = None
    shadow_tokens = 0
    shadow_tool_names: list[str] = []
    shadow_response_len = 0
    shadow_latency_ms = 0.0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            resp = await client.request(
                method="POST",
                url=upstream_url,
                content=shadow_bytes,
                headers=shadow_headers,
            )
        shadow_latency_ms = (time.time() - t_start) * 1000

        if resp.status_code == 200:
            try:
                rb = resp.json()
                shadow_tokens       = _extract_tokens(rb)
                shadow_tool_names   = _extract_tool_calls(rb)
                shadow_response_len = _extract_response_len(rb)
            except Exception as pe:
                shadow_error = f"parse_error: {pe}"
        else:
            shadow_error = f"http_{resp.status_code}"

    except Exception as ce:
        shadow_latency_ms = (time.time() - t_start) * 1000
        shadow_error = str(ce)[:300]
        logger.debug("Shadow call failed (ignored): %s", shadow_error)

    # ── Compute comparison metrics ────────────────────────────────────────────
    primary_cost  = _estimate_cost(primary_tokens, primary_model)
    shadow_cost   = _estimate_cost(shadow_tokens, shadow_model)
    prim_tools    = sorted(_primary_tool_names(primary_tool_calls))
    tool_agreement = prim_tools == sorted(shadow_tool_names)

    latency_delta_pct = (
        (shadow_latency_ms - primary_latency_ms) / primary_latency_ms * 100
        if primary_latency_ms > 0 else 0.0
    )
    cost_delta_pct = (
        (shadow_cost - primary_cost) / primary_cost * 100
        if primary_cost > 0 else 0.0
    )
    response_len_ratio = (
        shadow_response_len / primary_response_len
        if primary_response_len > 0 else 0.0
    )

    comparison: dict[str, Any] = {
        "id":                   str(uuid.uuid4()),
        "org_id":               org_id,
        "session_id":           session_id,
        "agent_id":             agent_id,
        "primary_model":        primary_model,
        "shadow_model":         shadow_model,
        "provider":             provider,
        "primary_latency_ms":   round(primary_latency_ms, 1),
        "primary_tokens":       primary_tokens,
        "primary_cost_usd":     round(primary_cost, 6),
        "primary_tool_calls":   prim_tools,
        "primary_response_len": primary_response_len,
        "shadow_latency_ms":    round(shadow_latency_ms, 1),
        "shadow_tokens":        shadow_tokens,
        "shadow_cost_usd":      round(shadow_cost, 6),
        "shadow_tool_calls":    shadow_tool_names,
        "shadow_response_len":  shadow_response_len,
        "shadow_error":         shadow_error,
        "tool_call_agreement":  tool_agreement,
        "latency_delta_pct":    round(latency_delta_pct, 1),
        "cost_delta_pct":       round(cost_delta_pct, 1),
        "response_len_ratio":   round(response_len_ratio, 3),
    }

    # ── POST to backend ───────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            await client.post(
                f"{settings.backend_url}/v1/shadow/comparisons",
                json=comparison,
                headers={
                    "X-API-Key":    settings.backend_api_key,
                    "Content-Type": "application/json",
                },
            )
        logger.debug(
            "Shadow comparison stored: %s vs %s latency_delta=%.1f%% cost_delta=%.1f%%",
            primary_model, shadow_model, latency_delta_pct, cost_delta_pct,
        )
    except Exception as be:
        logger.debug("Shadow backend POST failed (ignored): %s", be)
