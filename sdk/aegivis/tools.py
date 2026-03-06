"""
General Tool Instrumentation SDK for Aegivis.

Wraps tool functions to emit TOOL_EXEC_START / TOOL_EXEC_END / TOOL_EXEC_ERROR
events to the Aegivis backend. Gives you visibility into actual tool
execution (duration, errors, return values) — the proxy sees what the LLM
*requested*; this SDK sees what the tool actually *did*.

Quick start:

    from aegivis import instrument

    # Wrap a list of tools (LangChain, AutoGen, raw callables):
    tools = instrument(tools, agent_id="my-agent")

    # Or wrap a single function with a decorator:
    @instrument.tool
    def send_email(to: str, body: str) -> str:
        ...

    # Or with explicit config:
    @instrument.tool(agent_id="finance-agent")
    def process_payment(amount: float) -> str:
        ...

Events emitted
--------------
TOOL_EXEC_START  — fires before the tool runs (payload: tool_name, args_preview)
TOOL_EXEC_END    — fires after successful return (payload: duration_ms, return_preview)
TOOL_EXEC_ERROR  — fires on exception (payload: duration_ms, error_type, error_message)

All events are POSTed to AEGIVIS_BACKEND_URL in a background daemon thread —
never blocks the caller. Silent on network failure (unless AEGIVIS_DEBUG=1).

Tool type detection
-------------------
- LangChain BaseTool: has .name (str) + .run() method → wraps .run() and .arun()
- AutoGen FunctionTool: has ._func attribute → wraps ._func
- Async callable: detected via asyncio.iscoroutinefunction → async wrapper
- Any other callable: sync wrapper

Environment variable overrides
-------------------------------
AEGIVIS_BACKEND_URL        — backend URL (default: http://localhost:8000)
AEGIVIS_BACKEND_API_KEY    — API key (default: dev-proxy-key)
AEGIVIS_ORG_ID             — org ID (default: default-org)
AEGIVIS_AGENT_ID           — agent ID (set automatically by abb.session())
AEGIVIS_SESSION_ID         — session ID (set automatically by abb.session())
AEGIVIS_DEBUG              — set to "1" to log network errors to stderr
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

_BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000")
_API_KEY = os.environ.get("AEGIVIS_BACKEND_API_KEY", "dev-proxy-key")
_ORG_ID = os.environ.get("AEGIVIS_ORG_ID", "default-org")
_DEBUG = os.environ.get("AEGIVIS_DEBUG", "") == "1"

# Max concurrent background event threads. Prevents unbounded thread growth
# when the backend is slow or unreachable.
_MAX_EVENT_THREADS = int(os.environ.get("AEGIVIS_MAX_EVENT_THREADS", "20"))
_thread_sem = threading.Semaphore(_MAX_EVENT_THREADS)

# Mock type names that should never be instrumented. MagicMock has auto-created
# .run (callable) and .name (str) which triggers the LangChain heuristic.
_MOCK_TYPE_NAMES: frozenset[str] = frozenset({
    "MagicMock", "NonCallableMagicMock", "AsyncMock", "Mock",
    "NonCallableMock", "MagicProxy", "patch",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_preview(value: Any, max_len: int = 300) -> str:
    """Convert a value to a truncated, JSON-safe string preview."""
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s[:max_len] + ("…" if len(s) > max_len else "")


def _fire_event(event: dict, backend_url: str, api_key: str) -> None:
    """POST one event to the backend in a daemon thread (fire-and-forget).

    Uses a bounded semaphore (_MAX_EVENT_THREADS) so the thread count
    stays finite even when the backend is slow or unreachable.
    """
    if not backend_url:
        return

    if not _thread_sem.acquire(blocking=False):
        if _DEBUG:
            logger.warning("ABB instrument: event thread pool saturated — dropping event")
        return

    def _post() -> None:
        try:
            import httpx  # noqa: PLC0415
            payload = {
                "events": [event],
                "batch_id": str(uuid.uuid4()),
                "sent_at_ns": time.time_ns(),
            }
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    f"{backend_url.rstrip('/')}/v1/ingest",
                    content=json.dumps(payload, default=str),
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": api_key,
                    },
                )
        except Exception as exc:
            if _DEBUG:
                logger.warning("ABB instrument: failed to post event: %s", exc)
        finally:
            _thread_sem.release()

    t = threading.Thread(target=_post, daemon=True)
    t.start()


def _make_event(
    event_type: str,
    tool_name: str,
    payload_extra: dict,
    agent_id: str,
    session_id: str,
    org_id: str,
) -> dict:
    """Build a backend-compatible event dict for a tool exec event."""
    return {
        "event_id": f"sdk_{uuid.uuid4().hex[:20]}",
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": "sdk",
        "model": "sdk",
        "interception_layer": "sdk_instrument",
        "run_id": str(uuid.uuid4()),
        "parent_run_id": None,
        "event_type": event_type,
        "payload": {
            "tool_name": tool_name,
            "source": "sdk_instrument",
            **payload_extra,
        },
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": time.time_ns(),
        "sequence_number": 0,
        "previous_hash": "SDK_INSTRUMENT",
        "current_hash": "SDK_INSTRUMENT",
    }


class _InstrumentConfig:
    """Resolved runtime configuration for one instrumented tool set."""

    def __init__(
        self,
        agent_id: str = "",
        session_id: str = "",
        backend_url: str = "",
        api_key: str = "",
        org_id: str = "",
    ):
        self.agent_id = agent_id or os.environ.get("AEGIVIS_AGENT_ID", "sdk-agent")
        self.session_id = (
            session_id
            or os.environ.get("AEGIVIS_SESSION_ID", f"sdk_{uuid.uuid4().hex[:12]}")
        )
        self.backend_url = backend_url or _BACKEND_URL
        self.api_key = api_key or _API_KEY
        self.org_id = org_id or _ORG_ID


# ---------------------------------------------------------------------------
# Sync / async wrappers
# ---------------------------------------------------------------------------

def _wrap_sync(fn: Callable, tool_name: str, cfg: _InstrumentConfig) -> Callable:
    """Wrap a sync callable to emit TOOL_EXEC_* events."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        _fire_event(
            _make_event(
                "TOOL_EXEC_START",
                tool_name,
                {"args_preview": _safe_preview({"args": list(args), "kwargs": kwargs})},
                cfg.agent_id,
                cfg.session_id,
                cfg.org_id,
            ),
            cfg.backend_url,
            cfg.api_key,
        )

        try:
            result = fn(*args, **kwargs)
            duration_ms = (time.time_ns() - start_ns) / 1_000_000
            _fire_event(
                _make_event(
                    "TOOL_EXEC_END",
                    tool_name,
                    {
                        "duration_ms": round(duration_ms, 2),
                        "return_preview": _safe_preview(result),
                    },
                    cfg.agent_id,
                    cfg.session_id,
                    cfg.org_id,
                ),
                cfg.backend_url,
                cfg.api_key,
            )
            return result

        except Exception as exc:
            duration_ms = (time.time_ns() - start_ns) / 1_000_000
            _fire_event(
                _make_event(
                    "TOOL_EXEC_ERROR",
                    tool_name,
                    {
                        "duration_ms": round(duration_ms, 2),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300],
                    },
                    cfg.agent_id,
                    cfg.session_id,
                    cfg.org_id,
                ),
                cfg.backend_url,
                cfg.api_key,
            )
            raise

    return wrapper


def _wrap_async(fn: Callable, tool_name: str, cfg: _InstrumentConfig) -> Callable:
    """Wrap an async callable to emit TOOL_EXEC_* events."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        _fire_event(
            _make_event(
                "TOOL_EXEC_START",
                tool_name,
                {"args_preview": _safe_preview({"args": list(args), "kwargs": kwargs})},
                cfg.agent_id,
                cfg.session_id,
                cfg.org_id,
            ),
            cfg.backend_url,
            cfg.api_key,
        )

        try:
            result = await fn(*args, **kwargs)
            duration_ms = (time.time_ns() - start_ns) / 1_000_000
            _fire_event(
                _make_event(
                    "TOOL_EXEC_END",
                    tool_name,
                    {
                        "duration_ms": round(duration_ms, 2),
                        "return_preview": _safe_preview(result),
                    },
                    cfg.agent_id,
                    cfg.session_id,
                    cfg.org_id,
                ),
                cfg.backend_url,
                cfg.api_key,
            )
            return result

        except Exception as exc:
            duration_ms = (time.time_ns() - start_ns) / 1_000_000
            _fire_event(
                _make_event(
                    "TOOL_EXEC_ERROR",
                    tool_name,
                    {
                        "duration_ms": round(duration_ms, 2),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300],
                    },
                    cfg.agent_id,
                    cfg.session_id,
                    cfg.org_id,
                ),
                cfg.backend_url,
                cfg.api_key,
            )
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Tool type detection + wrapping
# ---------------------------------------------------------------------------

def _tool_name(obj: Any) -> str:
    """Extract a display name from a tool object or callable."""
    if hasattr(obj, "name") and isinstance(getattr(obj, "name"), str):
        return obj.name
    return getattr(obj, "__name__", type(obj).__name__)


def _wrap_one(obj: Any, cfg: _InstrumentConfig) -> Any:
    """
    Instrument a single tool object.

    Detection priority:
      0. Mock objects         — skip (MagicMock matches LangChain heuristic)
      1. LangChain BaseTool  — has .run() + .name str → wrap .run() and .arun()
      2. AutoGen FunctionTool — has ._func callable → wrap ._func
      3. Async callable       — asyncio.iscoroutinefunction → async wrapper
      4. Sync callable        — any callable → sync wrapper
      5. Unknown              — returned unchanged with a debug log
    """
    # 0. Guard: unittest.mock objects auto-generate .run and .name attributes
    #    that would falsely match the LangChain heuristic. Skip them.
    if type(obj).__name__ in _MOCK_TYPE_NAMES:
        logger.debug("ABB instrument: skipping mock object %s", type(obj).__name__)
        return obj

    # 1. LangChain BaseTool
    if hasattr(obj, "run") and callable(obj.run) and isinstance(getattr(obj, "name", None), str):
        name = _tool_name(obj)
        obj.run = _wrap_sync(obj.run, name, cfg)
        if hasattr(obj, "arun") and callable(obj.arun):
            obj.arun = _wrap_async(obj.arun, name, cfg)
        return obj

    # 2. AutoGen FunctionTool (has ._func)
    if hasattr(obj, "_func") and callable(obj._func):
        name = getattr(obj, "name", None) or getattr(obj._func, "__name__", "tool")
        obj._func = _wrap_sync(obj._func, name, cfg)
        return obj

    # 3. Async callable
    if inspect.iscoroutinefunction(obj):
        return _wrap_async(obj, _tool_name(obj), cfg)

    # 4. Sync callable
    if callable(obj):
        return _wrap_sync(obj, _tool_name(obj), cfg)

    # 5. Unknown — pass through
    logger.debug(
        "ABB instrument: unrecognised tool type %s, skipping", type(obj).__name__
    )
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def instrument(
    tools: Any,
    *,
    agent_id: str = "",
    session_id: str = "",
    backend_url: str = "",
    api_key: str = "",
    org_id: str = "",
) -> Any:
    """
    Instrument one or more tools to emit execution events to Aegivis.

    Args:
        tools:        A single tool or a list of tools (LangChain BaseTool,
                      AutoGen FunctionTool, or any plain callable).
        agent_id:     Override the agent ID (defaults to AEGIVIS_AGENT_ID env var).
        session_id:   Override the session ID (defaults to AEGIVIS_SESSION_ID env var).
        backend_url:  Backend URL (defaults to AEGIVIS_BACKEND_URL env var).
        api_key:      API key (defaults to AEGIVIS_BACKEND_API_KEY env var).
        org_id:       Org ID (defaults to AEGIVIS_ORG_ID env var).

    Returns:
        The instrumented tool or list of tools (same structural type as input).

    Examples::

        # Wrap a LangChain tool list
        tools = instrument(tools, agent_id="research-agent")

        # Wrap a single function
        send_email = instrument(send_email, agent_id="finance-agent")
    """
    cfg = _InstrumentConfig(
        agent_id=agent_id,
        session_id=session_id,
        backend_url=backend_url,
        api_key=api_key,
        org_id=org_id,
    )
    if isinstance(tools, list):
        return [_wrap_one(t, cfg) for t in tools]
    return _wrap_one(tools, cfg)


# ---------------------------------------------------------------------------
# @instrument.tool decorator
# ---------------------------------------------------------------------------

class _ToolDecorator:
    """
    Provides the ``@instrument.tool`` decorator syntax.

    Usage::

        @instrument.tool
        def my_tool(x: str) -> str: ...

        @instrument.tool(agent_id="my-agent")
        def my_tool(x: str) -> str: ...
    """

    def __call__(
        self,
        fn: Callable | None = None,
        *,
        agent_id: str = "",
        session_id: str = "",
        backend_url: str = "",
        api_key: str = "",
        org_id: str = "",
    ) -> Any:
        # Used as @instrument.tool (no parentheses) — fn is the decorated function
        if fn is not None and callable(fn):
            return instrument(
                fn,
                agent_id=agent_id,
                session_id=session_id,
                backend_url=backend_url,
                api_key=api_key,
                org_id=org_id,
            )
        # Used as @instrument.tool(...) — return a parameterised decorator
        def decorator(func: Callable) -> Callable:
            return instrument(
                func,
                agent_id=agent_id,
                session_id=session_id,
                backend_url=backend_url,
                api_key=api_key,
                org_id=org_id,
            )
        return decorator


# Attach .tool as an attribute so `instrument.tool` works
instrument.tool = _ToolDecorator()  # type: ignore[attr-defined]
