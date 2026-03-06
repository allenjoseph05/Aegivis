"""
CrewAI adapter for Aegivis.

Provides a ``step_callback`` compatible with ``Crew(step_callback=...)`` that
records every agent step as an ``AGENT_THOUGHT`` event in the Aegivis audit
trail.  Also exposes ``make_task_callback()`` for task-level recording.

Install::

    pip install 'aegivis[crewai]'

Usage::

    from aegivis.adapters.crewai import AegivisCrewAICallback

    cb = AegivisCrewAICallback(agent_id="my-crew")
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        step_callback=cb,
    )

    # Optionally attach per-task callbacks too:
    research_task = Task(
        description="...",
        callback=cb.make_task_callback(),
    )

Events emitted
--------------
- ``AGENT_THOUGHT`` with ``event="agent_action"`` — for tool-calling steps
  (includes ``tool`` name and ``tool_input``)
- ``AGENT_THOUGHT`` with ``event="agent_finish"`` — for steps that produce a
  final answer (includes ``output``)
- ``AGENT_THOUGHT`` with ``event="task_complete"`` — when a task finishes
  (via ``make_task_callback()``)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class AegivisCrewAICallback:
    """
    CrewAI ``step_callback`` that emits ``AGENT_THOUGHT`` events to Aegivis.

    Parameters
    ----------
    agent_id    Identifier attached to every emitted event.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.
    """

    def __init__(
        self,
        *,
        agent_id: str = "crewai-agent",
        backend_url: str = "",
        api_key: str = "",
    ) -> None:
        self._agent_id = agent_id
        self._backend_url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
        self._api_key = api_key or os.environ.get("AEGIVIS_API_KEY", "")

    def __call__(self, step_output: Any) -> None:
        """Called by CrewAI after every agent step."""
        payload = _extract_step_payload(step_output)
        _fire(self._backend_url, self._api_key, "AGENT_THOUGHT", self._agent_id, payload)

    def make_task_callback(self) -> Any:
        """
        Return a callable suitable for ``Task(callback=...)``.

        Records task completion as an ``AGENT_THOUGHT`` event with
        ``event="task_complete"``.
        """
        url, key, agent_id = self._backend_url, self._api_key, self._agent_id

        def _task_callback(task_output: Any) -> None:
            _fire(url, key, "AGENT_THOUGHT", agent_id, {
                "event": "task_complete",
                "output": str(getattr(task_output, "raw", task_output))[:500],
                "description": str(getattr(task_output, "description", ""))[:200],
            })

        return _task_callback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_step_payload(step_output: Any) -> dict:
    """Convert a CrewAI step output (AgentAction or AgentFinish) to a payload dict."""
    # AgentFinish: has .return_values (dict) and optionally .log
    if hasattr(step_output, "return_values"):
        return {
            "event": "agent_finish",
            "output": str(step_output.return_values)[:500],
            "log": str(getattr(step_output, "log", ""))[:200],
        }
    # AgentAction: has .tool, .tool_input, and optionally .log
    if hasattr(step_output, "tool"):
        return {
            "event": "agent_action",
            "tool": str(step_output.tool),
            "tool_input": str(getattr(step_output, "tool_input", ""))[:300],
            "log": str(getattr(step_output, "log", ""))[:200],
        }
    # Fallback for unrecognised step types
    return {"event": "step", "raw": str(step_output)[:500]}


def _fire(url: str, key: str, event_type: str, agent_id: str, payload: dict) -> None:
    if not url:
        return
    event = {
        "event_type": event_type,
        "agent_id": agent_id,
        "timestamp_ns": time.time_ns(),
        "payload": payload,
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
        except Exception as exc:  # noqa: BLE001
            logger.debug("Aegivis fire-and-forget failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()
