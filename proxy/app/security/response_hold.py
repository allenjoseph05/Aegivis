"""
Response-Hold HITL Gate — Phase 19.

Transforms HITL from "block the NEXT LLM call" to "block THIS response before
the agent ever sees the tool_calls."  The agent never executes a tool that has
not been approved by a human reviewer.

Current HITL flow (Phases 9 / 15):
    LLM responds with tool_call → agent receives it → tool executes
    → agent sends tool_result → NEXT LLM call BLOCKED until approved
    Problem: tool already executed; we can only retroactively block.

Phase 19 flow:
    LLM responds with tool_call → Proxy HOLDS response (not returned to agent)
    → Proxy creates approval record → Polls until human decides
    → Approved: release original response to agent (tool executes normally)
    → Denied: return synthetic denial response (no tool_calls; agent stops)

Architecture:
    poll_hold_gate()     — async polling loop; respects timeout; survives
                           transient network errors.
    synthesize_denial()  — generates a structurally valid LLM response with
                           no tool_calls so the agent SDK parses it cleanly
                           and does not attempt to execute the blocked action.

Provider support:
    Non-streaming only.  Streaming responses are already in-flight by the time
    tool_calls are parsed; holding them is a future enhancement (Phase 19.1).

    Formats supported:
        "anthropic"                → Anthropic Messages API format
        all other providers        → OpenAI Chat Completion format
        (OpenAI-compat providers like Groq, Together, Mistral, etc. all use
         the OpenAI schema — handled by the else branch.)

Integration (main.py non-streaming path, after process_response()):
    should_deny, denial_body = await context.maybe_hold_response(
        session_id, agent_id, provider, model)
    if should_deny:
        return JSONResponse(status_code=200, content=denial_body, headers={...})
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

#: Denial message returned to the agent when a hold is not approved.
#: Written as a fact, not a threat — the agent should propagate it to the user.
_DENIAL_TEXT = (
    "This action requires human approval that was not granted. "
    "The security system has prevented this tool call from executing. "
    "Please contact your supervisor or security team if you believe this decision is incorrect."
)


# ---------------------------------------------------------------------------
# HoldDecision
# ---------------------------------------------------------------------------

@dataclass
class HoldDecision:
    """
    Result of polling the approvals endpoint for a response-hold.

    Attributes:
        approved:     True if the reviewer approved the tool call.
        decision:     Raw decision string: "approved", "denied", "expired",
                      "timeout" (proxy deadline), "no_backend" (backend unreachable).
        approval_id:  The approval record ID that was polled.
        latency_ms:   Wall-clock time from first poll to decision.
    """
    approved: bool
    decision: str
    approval_id: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

async def poll_hold_gate(
    approval_id: str,
    timeout_s: int,
    backend_url: str,
    api_key: str,
    poll_interval_s: float = 2.0,
) -> HoldDecision:
    """
    Poll the approvals endpoint until a human decision is made.

    Args:
        approval_id:     The approval record to poll.
        timeout_s:       Maximum wall-clock seconds to wait before giving up.
                         On timeout, ``decision="timeout"`` and ``approved=False``.
        backend_url:     Base URL of the Aegivis backend (e.g. "http://localhost:8000").
        api_key:         Backend API key (X-API-Key header).
        poll_interval_s: Seconds between polls.  Default 2 s.

    Returns:
        HoldDecision with approved=True (continue) or approved=False (deny).

    Error handling:
        Transient network errors are caught and retried until the deadline.
        If the backend is completely unreachable for the full timeout, the
        decision is "timeout" and approved=False (conservative — deny).
    """
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    decision = "pending"

    logger.info(
        "[RESPONSE-HOLD] Polling approval id=%s timeout=%ds", approval_id, timeout_s
    )

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(
                    f"{backend_url}/v1/approvals/{approval_id}",
                    headers={"X-API-Key": api_key},
                )
                if resp.status_code == 200:
                    decision = resp.json().get("status", "pending")
                    if decision in ("approved", "denied", "expired"):
                        break
                else:
                    logger.debug(
                        "[RESPONSE-HOLD] Unexpected status %d from approvals endpoint",
                        resp.status_code,
                    )
            except Exception as exc:
                logger.debug("[RESPONSE-HOLD] Poll error (retrying): %s", exc)

            # Sleep, but wake early if deadline passes mid-sleep
            remaining = deadline - time.monotonic()
            await asyncio.sleep(min(poll_interval_s, max(0.0, remaining)))
        else:
            decision = "timeout"

    latency_ms = (time.monotonic() - t0) * 1000.0
    approved = decision == "approved"

    logger.info(
        "[RESPONSE-HOLD] Decision: id=%s decision=%s approved=%s latency=%.0fms",
        approval_id, decision, approved, latency_ms,
    )

    return HoldDecision(
        approved=approved,
        decision=decision,
        approval_id=approval_id,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Synthetic denial response
# ---------------------------------------------------------------------------

def synthesize_denial(provider: str, model: str, message: str = _DENIAL_TEXT) -> dict:
    """
    Generate a structurally valid LLM response containing no tool_calls.

    The agent SDK receives a normal-looking response that it can parse without
    errors.  The content explains that the action was not approved.  The agent
    is expected to propagate this message to the end user and stop.

    Args:
        provider: Provider string from the request context ("anthropic", "openai",
                  "groq", "together", etc.).  Only "anthropic" produces Anthropic
                  format; all others produce OpenAI Chat Completion format.
        model:    The model string from the original request (passed through so
                  the agent SDK validation passes).
        message:  The denial text.  Defaults to _DENIAL_TEXT.

    Returns:
        A dict that is JSON-serialisable and parseable by the provider's SDK.
    """
    if provider == "anthropic":
        return _anthropic_denial(model, message)
    return _openai_denial(model, message)


def _anthropic_denial(model: str, message: str) -> dict:
    """
    Anthropic Messages API format denial response.

    Shape mirrors: https://docs.anthropic.com/en/api/messages
    ``stop_reason`` is ``"end_turn"`` — the agent sees a normal conversation
    ending, not an error, so it does not retry.
    """
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": len(message.split()),
        },
    }


def _openai_denial(model: str, message: str) -> dict:
    """
    OpenAI Chat Completion format denial response.

    Shape mirrors: https://platform.openai.com/docs/api-reference/chat
    Used for OpenAI and all OpenAI-compatible providers (Groq, Together,
    Mistral, Fireworks, DeepSeek, Cerebras, etc.).
    ``finish_reason`` is ``"stop"`` — clean conversation end, no tool execution.
    """
    import time as _time
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(message.split()),
            "total_tokens": len(message.split()),
        },
    }
