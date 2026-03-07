"""
OpenAI Agents SDK tracing adapter for Aegivis.

Implements the ``TracingProcessor`` interface to emit structured audit events
for every tool call, LLM generation, agent step, and handoff processed by the
OpenAI Agents SDK (``openai-agents`` package, 2025).

Install::

    pip install 'aegivis[openai-agents]'

Usage::

    from agents import set_trace_processors
    from aegivis.adapters.openai_agents import AegivisTracingProcessor

    set_trace_processors([AegivisTracingProcessor(agent_id="my-agent")])

    # Then run your agent normally — events are emitted automatically.
    result = await Runner.run(agent, "Hello")

Events emitted
--------------
- ``TOOL_EXEC_END``   — every function/tool span (name, input, output)
- ``TOOL_EXEC_ERROR`` — function spans that ended with an error
- ``AGENT_THOUGHT``   — LLM generation spans (model, usage), agent steps,
                        and handoffs (from_agent → to_agent)
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

# ---------------------------------------------------------------------------
# Optional base class — graceful degradation when openai-agents is absent
# ---------------------------------------------------------------------------

try:
    from agents.tracing import TracingProcessor as _TracingProcessor  # type: ignore
    _BASE: type = _TracingProcessor
except ImportError:
    _BASE = object


class AegivisTracingProcessor(_BASE):
    """
    OpenAI Agents SDK ``TracingProcessor`` that emits Aegivis audit events.

    Captures:

    - **Tool/function calls** → ``TOOL_EXEC_END`` (or ``TOOL_EXEC_ERROR``)
    - **LLM generation calls** → ``AGENT_THOUGHT`` with model and usage info
    - **Agent handoffs** → ``AGENT_THOUGHT`` noting from/to agent names
    - **Agent steps** → ``AGENT_THOUGHT`` with agent name

    Parameters
    ----------
    agent_id    Identifier attached to every emitted event.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL`` env var.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY`` env var.
    """

    def __init__(
        self,
        *,
        agent_id: str = "",
        backend_url: str = "",
        api_key: str = "",
    ) -> None:
        if _BASE is not object:
            super().__init__()
        self._agent_id = agent_id
        self._backend_url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
        self._api_key = api_key or os.environ.get("AEGIVIS_API_KEY", "")

    # ── TracingProcessor protocol ──────────────────────────────────────────

    def on_trace_start(self, trace: Any) -> None:  # noqa: ARG002
        pass

    def on_trace_end(self, trace: Any) -> None:  # noqa: ARG002
        pass

    def on_span_start(self, span: Any) -> None:  # noqa: ARG002
        pass

    def on_span_end(self, span: Any) -> None:
        span_type = getattr(span, "type", None) or ""
        data = getattr(span, "span_data", None)
        error = getattr(span, "error", None)

        if span_type == "function":
            name = getattr(data, "name", "unknown")
            inp = getattr(data, "input", None)
            out = getattr(data, "output", None)
            if error:
                self._fire("TOOL_EXEC_ERROR", {
                    "tool_name": name,
                    "input": inp,
                    "error": str(error),
                })
            else:
                self._fire("TOOL_EXEC_END", {
                    "tool_name": name,
                    "input": inp,
                    "output": str(out)[:500] if out is not None else None,
                })

        elif span_type == "generation":
            model = getattr(data, "model", None)
            usage = getattr(data, "usage", None)
            self._fire("AGENT_THOUGHT", {
                "event": "llm_generation",
                "model": model,
                "usage": usage,
                "error": str(error) if error else None,
            })

        elif span_type == "handoff":
            self._fire("AGENT_THOUGHT", {
                "event": "handoff",
                "from_agent": getattr(data, "from_agent", None),
                "to_agent": getattr(data, "to_agent", None),
            })

        elif span_type == "agent":
            self._fire("AGENT_THOUGHT", {
                "event": "agent_step",
                "agent_name": getattr(data, "name", None),
                "error": str(error) if error else None,
            })

        elif span_type == "guardrail":
            triggered = getattr(data, "triggered", None)
            if triggered:
                self._fire("AGENT_THOUGHT", {
                    "event": "guardrail_triggered",
                    "name": getattr(data, "name", None),
                })

    # ── Internal fire-and-forget ───────────────────────────────────────────

    def _fire(self, event_type: str, payload: dict) -> None:
        if not self._backend_url:
            return
        event = {
            "event_type": event_type,
            "agent_id": self._agent_id,
            "timestamp_ns": time.time_ns(),
            "payload": payload,
        }
        url = self._backend_url.rstrip("/") + "/v1/ingest"
        headers = {"Content-Type": "application/json", "X-API-Key": self._api_key}

        def _post() -> None:
            try:
                body = json.dumps(event).encode()
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=3):
                    pass
            except Exception as exc:  # noqa: BLE001
                logger.debug("Aegivis fire-and-forget failed: %s", exc)

        threading.Thread(target=_post, daemon=True).start()
