"""
Smolagents adapter for Aegivis.

Instruments HuggingFace ``smolagents`` agents to emit structured audit events
for every tool call and agent step.

How it works
------------
Smolagents agents (``CodeAgent``, ``ToolCallingAgent``) expose their tools via
``agent.tools`` (a dict of name → Tool).  Each ``Tool`` is callable — we wrap
``tool.__call__`` on every tool in the toolbox.  We also wrap ``agent.run()``
to capture the final response as an ``AGENT_THOUGHT`` event.

Install::

    pip install 'aegivis[smolagents]'

Usage::

    from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
    from aegivis.adapters.smolagents import instrument_agent

    model = HfApiModel()
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
    instrument_agent(agent, agent_id="my-smolagent")

    result = agent.run("What is the capital of France?")
    # TOOL_EXEC_END + AGENT_THOUGHT events emitted automatically

Events emitted
--------------
- ``TOOL_EXEC_END``   — every successful tool call (name, args, output)
- ``TOOL_EXEC_ERROR`` — tool calls that raised an exception
- ``AGENT_THOUGHT``   — final agent response text
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
    Instrument a smolagents agent to emit Aegivis audit events.

    Wraps every tool in ``agent.tools`` in-place to capture tool calls.
    Also wraps ``agent.run()`` to emit a final ``AGENT_THOUGHT`` event.

    Parameters
    ----------
    agent       A smolagents ``CodeAgent`` or ``ToolCallingAgent`` instance.
    agent_id    Label used in audit events. Defaults to type name.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.

    Returns
    -------
    The same agent (instrumented in-place).
    """
    _url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
    _key = api_key or os.environ.get("AEGIVIS_API_KEY", "")
    _id  = agent_id or type(agent).__name__

    # ── Instrument all tools in the toolbox ───────────────────────────────
    # We replace each tool in the dict with a wrapper callable rather than
    # patching tool.__call__ — Python's special method lookup bypasses
    # instance-level __call__ patches for dunder methods.
    tools: dict[str, Any] = getattr(agent, "tools", {}) or {}
    for tool_name, tool in list(tools.items()):
        tools[tool_name] = _wrap_tool(tool, tool_name, _id, _url, _key)

    # ── Wrap agent.run() for final response capture ───────────────────────
    if hasattr(agent, "run") and callable(agent.run):
        _orig_run = agent.run

        def _hooked_run(task, *args, **kwargs):
            result = _orig_run(task, *args, **kwargs)
            _fire(_url, _key, "AGENT_THOUGHT", _id, {
                "event":    "agent_run_complete",
                "task":     str(task)[:300],
                "preview":  str(result)[:500] if result is not None else None,
                "framework": "smolagents",
            })
            return result

        agent.run = _hooked_run

    return agent


def _wrap_tool(tool: Any, tool_name: str, agent_id: str, url: str, key: str) -> Any:
    """
    Return a callable wrapper around a smolagents Tool.

    We return a new wrapper object rather than patching ``tool.__call__``
    in-place because Python's special-method lookup (``obj(...)`` syntax)
    resolves ``__call__`` on the *type*, not the instance — so an
    instance-level patch is silently bypassed.
    """
    if not callable(tool):
        return tool

    class _ToolWrapper:
        """Thin callable wrapper that forwards to the original tool."""

        def __call__(self, *args, **kwargs):
            t_start = time.time_ns()
            try:
                result = tool(*args, **kwargs)
                _fire(url, key, "TOOL_EXEC_END", agent_id, {
                    "tool_name":   tool_name,
                    "tool_args":   _safe_json({"args": args, "kwargs": kwargs}),
                    "tool_output": _safe_json(result),
                    "latency_ms":  (time.time_ns() - t_start) / 1e6,
                    "framework":   "smolagents",
                })
                return result
            except Exception as exc:
                _fire(url, key, "TOOL_EXEC_ERROR", agent_id, {
                    "tool_name": tool_name,
                    "error":     str(exc)[:500],
                    "framework": "smolagents",
                })
                raise

        # Forward attribute access so the agent can still read .name, .description, etc.
        def __getattr__(self, item):
            return getattr(tool, item)

    return _ToolWrapper()


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
