"""
Haystack native pipeline adapter for Aegivis.

Instruments Haystack v2 ``Pipeline`` instances to emit structured audit
events for every component execution — including LLM generators, retrievers,
web search components, and custom tools.

Why a native adapter alongside LiteLLM
---------------------------------------
The ``litellm`` adapter captures LLM API calls when Haystack uses LiteLLM as
its LLM backend.  This adapter operates at the **pipeline component level**,
capturing the full component execution (input → output) for *every* component
type: retrievers, rankers, generators, routers, and custom components.  This
gives full pipeline observability, not just LLM calls.

How it works
------------
Haystack v2 ``Pipeline.run(inputs)`` internally calls each component's
``.run(**kwargs)`` method in topological order.  This adapter wraps each
registered component's ``.run()`` method at instrument-time, emitting a
``TOOL_EXEC_END`` event when the component completes.  The pipeline's own
``run()`` is also wrapped to emit a summary ``AGENT_THOUGHT`` event.

Install::

    pip install 'aegivis[haystack]'

Usage::

    from haystack import Pipeline
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from aegivis.adapters.haystack import instrument_pipeline

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=ds))
    pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o"))
    instrument_pipeline(pipeline, agent_id="rag-pipeline")

    result = pipeline.run({"retriever": {"query": "What is RAG?"}})
    # TOOL_EXEC_END events fire for each component automatically

Events emitted
--------------
- ``TOOL_EXEC_END``   — every component.run() completion (name, inputs, outputs)
- ``TOOL_EXEC_ERROR`` — component.run() that raised an exception
- ``AGENT_THOUGHT``   — pipeline.run() summary (components run, total latency)
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


def instrument_pipeline(
    pipeline: Any,
    *,
    agent_id: str = "",
    backend_url: str = "",
    api_key: str = "",
) -> Any:
    """
    Instrument a Haystack v2 ``Pipeline`` to emit Aegivis audit events.

    Wraps each component's ``.run()`` method in-place (at instrument-time).
    Also wraps ``pipeline.run()`` for a summary event.

    Parameters
    ----------
    pipeline    A Haystack ``Pipeline`` instance (v2+).
    agent_id    Label used in audit events. Defaults to "haystack-pipeline".
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.

    Returns
    -------
    The same pipeline (instrumented in-place).
    """
    _url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
    _key = api_key or os.environ.get("AEGIVIS_API_KEY", "")
    _id  = agent_id or "haystack-pipeline"

    # ── Instrument each component ────────────────────────────────────────
    # Haystack v2: pipeline.graph is a networkx DiGraph where nodes have
    # a "component" attribute holding the component instance.
    try:
        graph = getattr(pipeline, "graph", None)
        if graph is not None:
            for node_name, node_data in graph.nodes(data=True):
                component = node_data.get("instance") or node_data.get("component")
                if component is not None and hasattr(component, "run"):
                    _wrap_component(component, str(node_name), _id, _url, _key)
    except Exception as exc:
        logger.debug("Could not iterate pipeline graph: %s", exc)

    # Fallback: Haystack may expose components via pipeline.components dict
    components = getattr(pipeline, "components", None)
    if isinstance(components, dict):
        for comp_name, component in components.items():
            if hasattr(component, "run") and not getattr(component, "_aegivis_instrumented", False):
                _wrap_component(component, comp_name, _id, _url, _key)

    # ── Wrap pipeline.run() for summary event ────────────────────────────
    if hasattr(pipeline, "run"):
        _orig_pipeline_run = pipeline.run

        def _hooked_pipeline_run(inputs, *args, **kwargs):
            t0 = time.time()
            result = _orig_pipeline_run(inputs, *args, **kwargs)
            elapsed_ms = (time.time() - t0) * 1000
            output_keys = list(result.keys()) if isinstance(result, dict) else []
            _fire(_url, _key, "AGENT_THOUGHT", _id, {
                "event":       "pipeline_run_complete",
                "output_keys": output_keys,
                "latency_ms":  round(elapsed_ms, 2),
                "framework":   "haystack",
            })
            return result

        pipeline.run = _hooked_pipeline_run

    return pipeline


def _wrap_component(
    component: Any,
    comp_name: str,
    agent_id: str,
    url: str,
    key: str,
) -> None:
    """Wrap a Haystack component's .run() method to emit audit events."""
    _orig_run = component.run

    def _hooked_run(**kwargs):
        t_start = time.time_ns()
        try:
            result = _orig_run(**kwargs)
            latency_ms = (time.time_ns() - t_start) / 1e6
            _fire(url, key, "TOOL_EXEC_END", agent_id, {
                "tool_name":   comp_name,
                "component":   type(component).__name__,
                "tool_args":   _safe_json(kwargs),
                "tool_output": _safe_json(result),
                "latency_ms":  round(latency_ms, 2),
                "framework":   "haystack",
            })
            return result
        except Exception as exc:
            _fire(url, key, "TOOL_EXEC_ERROR", agent_id, {
                "tool_name": comp_name,
                "component": type(component).__name__,
                "error":     str(exc)[:500],
                "framework": "haystack",
            })
            raise

    component.run = _hooked_run
    component._aegivis_instrumented = True  # idempotency guard


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
