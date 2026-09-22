"""
PydanticAI adapter for Aegivis.

Instruments PydanticAI ``Agent`` instances to emit structured audit events
for every tool call and agent run.

How it works
------------
PydanticAI's ``agent.run()`` / ``agent.run_sync()`` return a ``RunResult``
whose ``all_messages()`` list contains the full conversation, including
``ToolCallPart`` (the LLM requesting a tool) and ``ToolReturnPart`` (the
result fed back).  The adapter wraps those methods, inspects the result
messages post-run, and fires ``TOOL_EXEC_END`` events for each tool call.

For real-time tool visibility (not post-hoc), also pass tools through
``aegivis.instrument(tools)`` before registering them with the agent.

Install::

    pip install 'aegivis[pydantic-ai]'

Usage::

    from pydantic_ai import Agent
    from aegivis.adapters.pydantic_ai import instrument_agent

    agent = Agent("openai:gpt-4o", tools=[...])
    instrument_agent(agent, agent_id="my-pydantic-agent")

    result = await agent.run("What is the weather?")
    # TOOL_EXEC_END + AGENT_THOUGHT events emitted automatically

Sync usage::

    result = agent.run_sync("What is the weather?")
    # Same events emitted via daemon thread
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def instrument_agent(
    agent: Any,
    *,
    agent_id: str = "",
    backend_url: str = "",
    api_key: str = "",
) -> Any:
    """
    Instrument a PydanticAI ``Agent`` to emit Aegivis audit events.

    Wraps ``agent.run()`` (async) and ``agent.run_sync()`` (sync) in-place.
    After each run, inspects ``result.all_messages()`` and fires:

    - ``TOOL_EXEC_END`` for every ``ToolCallPart`` / ``ToolReturnPart`` pair
    - ``AGENT_THOUGHT`` for the final response

    Parameters
    ----------
    agent       A ``pydantic_ai.Agent`` instance.
    agent_id    Label used in audit events. Defaults to agent's model name.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.

    Returns
    -------
    The same agent (instrumented in-place).
    """
    _url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
    _key = api_key or os.environ.get("AEGIVIS_API_KEY", "")

    # Resolve agent_id from model name if not provided
    _id = agent_id
    if not _id:
        try:
            _id = str(agent.model)
        except Exception:
            _id = "pydantic-ai-agent"

    def _emit_run_events(result: Any) -> None:
        """Parse RunResult messages and fire audit events."""
        try:
            messages = result.all_messages()
        except Exception:
            return

        # Map ToolCallPart → ToolReturnPart by tool_call_id / index
        tool_calls: dict[str, dict] = {}

        for msg in messages:
            parts = getattr(msg, "parts", []) or []
            for part in parts:
                part_type = type(part).__name__
                if part_type == "ToolCallPart":
                    call_id = getattr(part, "tool_call_id", None) or id(part)
                    tool_calls[str(call_id)] = {
                        "name": getattr(part, "tool_name", "unknown"),
                        "args": _safe_json(getattr(part, "args", None)),
                        "started_at": time.time_ns(),
                    }
                elif part_type == "ToolReturnPart":
                    call_id = getattr(part, "tool_call_id", None)
                    tc = tool_calls.get(str(call_id), {})
                    _fire(_url, _key, "TOOL_EXEC_END", _id, {
                        "tool_name":  tc.get("name", "unknown"),
                        "tool_args":  tc.get("args"),
                        "tool_output": _safe_json(getattr(part, "content", None)),
                        "framework":  "pydantic-ai",
                    })
                elif part_type == "TextPart":
                    text = getattr(part, "content", "")
                    if text:
                        _fire(_url, _key, "AGENT_THOUGHT", _id, {
                            "event":    "response",
                            "preview":  text[:500],
                            "framework": "pydantic-ai",
                        })

    # ── Wrap async agent.run ────────────────────────────────────────────────
    if hasattr(agent, "run"):
        _orig_run = agent.run.__func__ if hasattr(agent.run, "__func__") else None

        import asyncio
        import inspect

        if inspect.iscoroutinefunction(getattr(type(agent), "run", None)):
            _orig_run_method = type(agent).run

            async def _hooked_run(self_agent, user_prompt, *args, **kwargs):
                result = await _orig_run_method(self_agent, user_prompt, *args, **kwargs)
                _emit_run_events(result)
                return result

            try:
                type(agent).run = _hooked_run
            except (TypeError, AttributeError):
                pass  # frozen or C-extension class — skip patching

    # ── Wrap sync agent.run_sync ────────────────────────────────────────────
    if hasattr(agent, "run_sync"):
        _orig_run_sync = type(agent).run_sync

        def _hooked_run_sync(self_agent, user_prompt, *args, **kwargs):
            result = _orig_run_sync(self_agent, user_prompt, *args, **kwargs)
            _emit_run_events(result)
            return result

        try:
            type(agent).run_sync = _hooked_run_sync
        except (TypeError, AttributeError):
            pass

    return agent


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
