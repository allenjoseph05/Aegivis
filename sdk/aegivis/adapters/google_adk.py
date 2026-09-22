"""
Google Agent Development Kit (ADK) adapter for Aegivis.

Instruments Google ADK ``Runner`` instances to emit structured audit events
for every tool call, LLM generation, and agent step.

How it works
------------
Google ADK's ``Runner.run_async()`` is an async generator that yields ``Event``
objects.  Each event carries a ``content`` field with ``Part`` objects — either
``FunctionCall`` (the agent requests a tool) or ``FunctionResponse`` (the tool
result fed back).  The adapter wraps ``run_async()`` to intercept these events
as they stream, forwarding them to Aegivis in real time.

Install::

    pip install 'aegivis[google-adk]'

Usage::

    from google.adk.runners import Runner
    from google.adk.agents import LlmAgent
    from aegivis.adapters.google_adk import instrument_runner

    agent  = LlmAgent(name="my-agent", model="gemini-2.0-flash", tools=[...])
    runner = Runner(agent=agent, app_name="myapp", session_service=session_svc)
    instrument_runner(runner, agent_id="my-adk-agent")

    async for event in runner.run_async(user_id="u1", session_id="s1",
                                        new_message=...):
        ...   # Aegivis events fire automatically as tool calls happen

Events emitted
--------------
- ``TOOL_EXEC_END``   — every ``FunctionResponse`` event (tool completed)
- ``AGENT_THOUGHT``   — agent text responses and model turns
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


def instrument_runner(
    runner: Any,
    *,
    agent_id: str = "",
    backend_url: str = "",
    api_key: str = "",
) -> Any:
    """
    Instrument a Google ADK ``Runner`` to emit Aegivis audit events.

    Wraps ``runner.run_async()`` in-place to intercept ADK events as they
    stream.  ``FunctionResponse`` events → ``TOOL_EXEC_END``.
    Text model responses → ``AGENT_THOUGHT``.

    Parameters
    ----------
    runner      A ``google.adk.runners.Runner`` instance.
    agent_id    Label used in audit events. Defaults to runner's app_name.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.

    Returns
    -------
    The same runner (instrumented in-place).
    """
    _url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
    _key = api_key or os.environ.get("AEGIVIS_API_KEY", "")
    _id  = agent_id or getattr(runner, "app_name", "google-adk-agent")

    _orig_run_async = runner.run_async

    async def _hooked_run_async(*args, **kwargs) -> AsyncIterator:
        # Track pending function calls to pair with responses
        pending_calls: dict[str, str] = {}  # function_name → call_id

        async for event in _orig_run_async(*args, **kwargs):
            content = getattr(event, "content", None)
            if content is not None:
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    # FunctionCall — record intent
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        fn_name = getattr(fc, "name", "unknown")
                        pending_calls[fn_name] = json.dumps(
                            dict(getattr(fc, "args", {}) or {}), default=str
                        )[:500]

                    # FunctionResponse — tool completed
                    fr = getattr(part, "function_response", None)
                    if fr is not None:
                        fn_name = getattr(fr, "name", "unknown")
                        response = getattr(fr, "response", None)
                        _fire(_url, _key, "TOOL_EXEC_END", _id, {
                            "tool_name":   fn_name,
                            "tool_args":   pending_calls.pop(fn_name, None),
                            "tool_output": _safe_json(response),
                            "framework":   "google-adk",
                        })

                    # Text part — agent thought / response
                    text = getattr(part, "text", None)
                    if text:
                        _fire(_url, _key, "AGENT_THOUGHT", _id, {
                            "event":    "model_response",
                            "preview":  text[:500],
                            "framework": "google-adk",
                        })

            yield event

    runner.run_async = _hooked_run_async
    return runner


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        if isinstance(obj, str):
            return obj[:1000]
        return json.dumps(obj, default=str)[:1000]
    except Exception:
        return str(obj)[:1000]


def _fire(url: str, key: str, event_type: str, agent_id: str, payload: dict) -> None:
    if not url:
        return
    event = {
        "event_type":   event_type,
        "agent_id":     agent_id,
        "timestamp_ns": time.time_ns(),
        "payload":      payload,
    }
    headers = {"Content-Type": "application/json", "X-API-Key": key}

    def _post() -> None:
        try:
            body = json.dumps(event).encode()
            req = urllib.request.Request(
                url.rstrip("/") + "/v1/ingest",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception as exc:
            logger.debug("Aegivis fire-and-forget failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()
