"""
General Tool Instrumentation SDK for Aegivis.

Wraps tool functions to emit TOOL_EXEC_START / TOOL_EXEC_END / TOOL_EXEC_ERROR
events to the Aegivis backend.  Optionally enforces an execution gate that
classifies actual argument values and blocks / alerts before the tool runs.

Quick start — observability only (unchanged from prior versions)::

    from aegivis import instrument

    tools = instrument(tools, agent_id="my-agent")

    @instrument.tool
    def send_email(to: str, body: str) -> str: ...

    @instrument.tool(agent_id="finance-agent")
    def process_payment(amount: float) -> str: ...

Execution gate — opt-in per tool::

    from aegivis import instrument
    from aegivis.security.arg_classifier import Signal

    @instrument.tool(
        timeout_s=10,
        block_on=frozenset({Signal.CREDENTIAL}),
        alert_on=frozenset({Signal.SHELL_METACHAR, Signal.NETWORK_DESTINATION}),
        scan_return=True,
    )
    def call_api(endpoint: str, token: str) -> dict: ...

Events emitted
--------------
TOOL_EXEC_START         — fires before execution (includes gate decision)
TOOL_EXEC_END           — fires after successful return
TOOL_EXEC_ERROR         — fires on exception (includes ToolExecutionBlocked)
TOOL_EXEC_ALERT         — fires when alert_on signal detected (before exec)
TOOL_EXEC_RETURN_ALERT  — fires when return value contains credential

All events are POSTed to AEGIVIS_BACKEND_URL in a background daemon thread.
Never blocks the caller.  Silent on network failure unless AEGIVIS_DEBUG=1.

Environment variables
---------------------
AEGIVIS_BACKEND_URL        backend URL (default: http://localhost:8000)
AEGIVIS_BACKEND_API_KEY    API key (default: dev-proxy-key)
AEGIVIS_ORG_ID             org ID (default: default-org)
AEGIVIS_AGENT_ID           agent ID (set automatically by abb.session())
AEGIVIS_SESSION_ID         session ID (set automatically by abb.session())
AEGIVIS_DEBUG              set to "1" to log network errors to stderr
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable

from .security.arg_classifier import ArgClassifier, ClassifierConfig
from .security.exec_policy import (
    GateConfig,
    PolicyAction,
    ToolArgumentViolation,
    ToolBudgetExceeded,
    ToolConcurrencyLimitExceeded,
    ToolExecutionBlocked,
    ToolExecutionTimeout,
    check_hitl_async,
    check_hitl_sync,
    evaluate_policy,
    scan_return_value,
)
from .security.taint_tracker import DEFAULT_UNTRUSTED_TOOLS, SDKTaintTracker

logger = logging.getLogger(__name__)

_BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000")
_API_KEY     = os.environ.get("AEGIVIS_BACKEND_API_KEY", "dev-proxy-key")
_ORG_ID      = os.environ.get("AEGIVIS_ORG_ID", "default-org")
_DEBUG       = os.environ.get("AEGIVIS_DEBUG", "") == "1"

# Bounded pool for background event posting — prevents unbounded thread growth
_MAX_EVENT_THREADS = int(os.environ.get("AEGIVIS_MAX_EVENT_THREADS", "20"))
_thread_sem = threading.Semaphore(_MAX_EVENT_THREADS)

# Shared executor for sync timeout enforcement — created lazily, never shut down
_timeout_executor: concurrent.futures.ThreadPoolExecutor | None = None
_timeout_executor_lock = threading.Lock()

# Mock type names: MagicMock has auto-created .run and .name that falsely
# trigger the LangChain heuristic.
_MOCK_TYPE_NAMES: frozenset[str] = frozenset({
    "MagicMock", "NonCallableMagicMock", "AsyncMock",
    "Mock", "NonCallableMock", "MagicProxy", "patch",
})

# Per-session concurrency semaphores.
# Key: "{session_id}:{max_concurrent}"
# Sync tools use threading semaphores; async tools use asyncio semaphores.
# Both dicts grow only as sessions are created — no eviction needed for
# typical session lifetimes (process-scoped, bounded by agent deployments).
_sync_semaphores: dict[str, threading.BoundedSemaphore] = {}
_sync_semaphores_lock = threading.Lock()
_async_semaphores: dict[str, asyncio.Semaphore] = {}

# Per-session taint trackers for cross-tool taint flow detection.
# Keyed by session_id. Created lazily on first taint-enabled tool call.
_session_taint_trackers: dict[str, SDKTaintTracker] = {}
_taint_trackers_lock = threading.Lock()

# Per-session call budget counters.
# _session_call_counts: session_id → total calls across all tools
# _tool_call_counts:    "session_id:tool_name" → calls for that specific tool
# Both are checked and incremented atomically under a single lock.
_session_call_counts: dict[str, int] = {}
_tool_call_counts: dict[str, int] = {}
_call_counts_lock = threading.Lock()


def _get_sync_semaphore(session_id: str, max_concurrent: int) -> threading.BoundedSemaphore:
    """Return (or create) the threading semaphore for this session+limit."""
    key = f"{session_id}:{max_concurrent}"
    with _sync_semaphores_lock:
        if key not in _sync_semaphores:
            _sync_semaphores[key] = threading.BoundedSemaphore(max_concurrent)
        return _sync_semaphores[key]


def _get_async_semaphore(session_id: str, max_concurrent: int) -> asyncio.Semaphore:
    """Return (or create) the asyncio semaphore for this session+limit.

    Called only from async context so no separate async lock is needed —
    coroutines in a single event loop don't interleave between the dict
    lookup and the assignment.
    """
    key = f"{session_id}:{max_concurrent}"
    if key not in _async_semaphores:
        _async_semaphores[key] = asyncio.Semaphore(max_concurrent)
    return _async_semaphores[key]


def _get_taint_tracker(session_id: str, gate: GateConfig) -> SDKTaintTracker:
    """Return (or create) the taint tracker for this session.

    Uses gate.taint_untrusted_tools if non-empty, otherwise falls back to
    DEFAULT_UNTRUSTED_TOOLS.  Once created, the tracker is shared by all
    instrumented tools in the same session regardless of per-tool config.
    """
    with _taint_trackers_lock:
        if session_id not in _session_taint_trackers:
            untrusted = gate.taint_untrusted_tools or DEFAULT_UNTRUSTED_TOOLS
            _session_taint_trackers[session_id] = SDKTaintTracker(
                untrusted_tools=untrusted
            )
        return _session_taint_trackers[session_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_timeout_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level timeout executor, creating it on first call."""
    global _timeout_executor
    if _timeout_executor is None:
        with _timeout_executor_lock:
            if _timeout_executor is None:
                _timeout_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=20,
                    thread_name_prefix="aegivis-timeout",
                )
    return _timeout_executor


def _safe_preview(value: Any, max_len: int = 300) -> str:
    """Convert a value to a truncated, JSON-safe string preview."""
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s[:max_len] + ("…" if len(s) > max_len else "")


def _fire_event(event: dict, backend_url: str, api_key: str) -> None:
    """POST one event to the backend in a daemon thread (fire-and-forget)."""
    if not backend_url:
        return

    if not _thread_sem.acquire(blocking=False):
        if _DEBUG:
            logger.warning("aegivis instrument: event thread pool saturated — dropping event")
        return

    def _post() -> None:
        try:
            import httpx  # noqa: PLC0415

            payload = {
                "events":     [event],
                "batch_id":   str(uuid.uuid4()),
                "sent_at_ns": time.time_ns(),
            }
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    f"{backend_url.rstrip('/')}/v1/ingest",
                    content=json.dumps(payload, default=str),
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key":    api_key,
                    },
                )
        except Exception as exc:
            if _DEBUG:
                logger.warning("aegivis instrument: failed to post event: %s", exc)
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
    """Build a backend-compatible event dict."""
    return {
        "event_id":           f"sdk_{uuid.uuid4().hex[:20]}",
        "schema_version":     "1.0",
        "org_id":             org_id,
        "session_id":         session_id,
        "agent_id":           agent_id,
        "provider":           "sdk",
        "model":              "sdk",
        "interception_layer": "sdk_instrument",
        "run_id":             str(uuid.uuid4()),
        "parent_run_id":      None,
        "event_type":         event_type,
        "payload": {
            "tool_name": tool_name,
            "source":    "sdk_instrument",
            **payload_extra,
        },
        "payload_hash":    None,
        "pii_detected":    [],
        "timestamp_ns":    time.time_ns(),
        "sequence_number": 0,
        "previous_hash":   "SDK_INSTRUMENT",
        "current_hash":    "SDK_INSTRUMENT",
    }


# ---------------------------------------------------------------------------
# Resolved runtime configuration per tool
# ---------------------------------------------------------------------------

class _InstrumentConfig:
    """All resolved runtime settings for one instrumented tool."""

    def __init__(
        self,
        agent_id:          str                  = "",
        session_id:        str                  = "",
        backend_url:       str                  = "",
        api_key:           str                  = "",
        org_id:            str                  = "",
        gate:              GateConfig | None     = None,
        classifier_config: ClassifierConfig | None = None,
    ):
        self.agent_id    = agent_id   or os.environ.get("AEGIVIS_AGENT_ID",   "sdk-agent")
        self.session_id  = (
            session_id or os.environ.get("AEGIVIS_SESSION_ID", f"sdk_{uuid.uuid4().hex[:12]}")
        )
        self.backend_url = backend_url or _BACKEND_URL
        self.api_key     = api_key     or _API_KEY
        self.org_id      = org_id      or _ORG_ID
        self.gate        = gate        or GateConfig()
        self.classifier  = ArgClassifier(classifier_config)


# ---------------------------------------------------------------------------
# Allowlist enforcement — runs first, before any other gate logic
# ---------------------------------------------------------------------------

def _check_allowlist(tool_name: str, cfg: _InstrumentConfig) -> None:
    """
    Raise ``ToolExecutionBlocked`` if ``allowed_tools`` is set and ``tool_name``
    is not in it.

    ``allowed_tools=None`` (default) is pass-through — no restriction.
    ``allowed_tools=frozenset()`` blocks every tool.
    """
    allowed = cfg.gate.allowed_tools
    if allowed is not None and tool_name not in allowed:
        _fire_event(
            _make_event(
                "TOOL_EXEC_BLOCKED",
                tool_name,
                {"reason": f"'{tool_name}' is not in the declared tool allowlist"},
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )
        raise ToolExecutionBlocked(
            tool_name,
            f"'{tool_name}' is not in the declared tool allowlist",
            [],
        )


# ---------------------------------------------------------------------------
# Call budget enforcement
# ---------------------------------------------------------------------------

def _check_and_increment_budget(tool_name: str, cfg: _InstrumentConfig) -> None:
    """
    Enforce per-session and per-tool call budgets.

    Checks both limits atomically under a single lock before incrementing.
    If either limit is exhausted, fires ``TOOL_EXEC_BLOCKED`` and raises
    ``ToolBudgetExceeded``.  The counter is NOT incremented on a blocked call.

    ``max_calls_per_session=None`` and ``max_calls_per_tool=None`` (defaults)
    are pass-through — no budget is enforced.
    """
    max_session = cfg.gate.max_calls_per_session
    max_tool    = cfg.gate.max_calls_per_tool

    if max_session is None and max_tool is None:
        return

    session_key = cfg.session_id
    tool_key    = f"{cfg.session_id}:{tool_name}"

    with _call_counts_lock:
        session_count = _session_call_counts.get(session_key, 0)
        tool_count    = _tool_call_counts.get(tool_key, 0)

        if max_session is not None and session_count >= max_session:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_BLOCKED",
                    tool_name,
                    {
                        "reason": (
                            f"session call budget of {max_session} exhausted "
                            f"({session_count} calls already made)"
                        ),
                        "budget_type": "session",
                        "limit": max_session,
                        "actual": session_count,
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            raise ToolBudgetExceeded(tool_name, "session", max_session, session_count)

        if max_tool is not None and tool_count >= max_tool:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_BLOCKED",
                    tool_name,
                    {
                        "reason": (
                            f"per-tool call budget of {max_tool} exhausted "
                            f"({tool_count} calls to '{tool_name}' already made)"
                        ),
                        "budget_type": "tool",
                        "limit": max_tool,
                        "actual": tool_count,
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            raise ToolBudgetExceeded(tool_name, "tool", max_tool, tool_count)

        # Both checks passed — increment atomically
        _session_call_counts[session_key] = session_count + 1
        _tool_call_counts[tool_key]       = tool_count + 1


# ---------------------------------------------------------------------------
# Argument constraint enforcement (path prefixes + URL domains)
# ---------------------------------------------------------------------------

import urllib.parse as _urllib_parse  # noqa: E402 — placed here for clarity

_URL_SCHEMES: frozenset[str] = frozenset({"http", "https", "ftp", "ws", "wss"})


def _looks_like_path(value: str) -> bool:
    """True if value is structurally path-shaped (Unix, home-relative, or Windows)."""
    return bool(
        value.startswith("/")
        or value.startswith("~/")
        or value.startswith("./")
        or value.startswith("../")
        or (len(value) > 2 and value[1] == ":" and value[2] in "/\\")
    )


def _looks_like_url(value: str) -> bool:
    """True if value has a network URL scheme and a netloc."""
    try:
        p = _urllib_parse.urlparse(value)
        return p.scheme.lower() in _URL_SCHEMES and bool(p.netloc)
    except Exception:
        return False


def _url_hostname(value: str) -> str:
    """Extract the hostname from a URL string, lower-cased."""
    try:
        return (_urllib_parse.urlparse(value).hostname or "").lower()
    except Exception:
        return ""


def _domain_allowed(host: str, allowed_domains: frozenset[str]) -> bool:
    """
    True if host is in allowed_domains or is a subdomain of one.

    ``{"example.com"}`` allows ``example.com`` and ``api.example.com``
    but not ``evil-example.com``.
    """
    host = host.lower()
    for domain in allowed_domains:
        domain = domain.lower()
        if host == domain or host.endswith("." + domain):
            return True
    return False


def _scan_constraints(
    value: Any,
    path_prefixes: frozenset[str] | None,
    url_domains: frozenset[str] | None,
    violations: list[tuple[str, str, str]],
    depth: int = 0,
) -> None:
    """
    Recursively scan a value for path/URL strings that violate constraints.

    Appends ``(constraint_type, value, allowed_repr)`` tuples to ``violations``.
    Stops after depth 4 and samples up to 20 list items.
    """
    if depth > 4:
        return

    if isinstance(value, str):
        if len(value) < 2:
            return
        if path_prefixes is not None and _looks_like_path(value):
            if not any(value.startswith(p) for p in path_prefixes):
                violations.append(("path_prefix", value, str(sorted(path_prefixes))))
        if url_domains is not None and _looks_like_url(value):
            host = _url_hostname(value)
            if not _domain_allowed(host, url_domains):
                violations.append(("url_domain", value, str(sorted(url_domains))))

    elif isinstance(value, dict):
        for v in value.values():
            _scan_constraints(v, path_prefixes, url_domains, violations, depth + 1)

    elif isinstance(value, (list, tuple)):
        for item in value[:20]:
            _scan_constraints(item, path_prefixes, url_domains, violations, depth + 1)


def _check_arg_constraints(
    tool_name: str,
    args: tuple,
    kwargs: dict,
    cfg: _InstrumentConfig,
) -> None:
    """
    Enforce path_prefixes and url_domains constraints on tool arguments.

    Scans all positional and keyword arguments recursively.
    Raises ``ToolArgumentViolation`` on the first violation found.
    Fires ``TOOL_EXEC_BLOCKED`` event before raising.
    No-ops if both constraints are ``None``.
    """
    path_prefixes = cfg.gate.path_prefixes
    url_domains   = cfg.gate.url_domains

    if path_prefixes is None and url_domains is None:
        return

    violations: list[tuple[str, str, str]] = []
    for arg in args:
        _scan_constraints(arg, path_prefixes, url_domains, violations)
        if violations:
            break
    if not violations:
        for v in kwargs.values():
            _scan_constraints(v, path_prefixes, url_domains, violations)
            if violations:
                break

    if violations:
        constraint_type, bad_value, allowed_repr = violations[0]
        _fire_event(
            _make_event(
                "TOOL_EXEC_BLOCKED",
                tool_name,
                {
                    "reason": (
                        f"{constraint_type} constraint violated: "
                        f"'{bad_value[:80]}' not in allowed set"
                    ),
                    "constraint_type": constraint_type,
                    "value_preview":   bad_value[:80],
                    "allowed":         allowed_repr,
                },
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )
        raise ToolArgumentViolation(tool_name, constraint_type, bad_value, allowed_repr)


# ---------------------------------------------------------------------------
# Taint tracking helpers
# ---------------------------------------------------------------------------

def _check_taint(
    tool_name: str,
    args: tuple,
    kwargs: dict,
    cfg: _InstrumentConfig,
) -> None:
    """
    Check whether any stored untrusted fragment appears in the arguments.

    Fires ``TOOL_EXEC_TAINT_FLOW`` event for every detected flow.
    If ``gate.taint_block`` is True, raises ``ToolExecutionBlocked``.
    Silently no-ops on any error (fail-open).
    """
    if not cfg.gate.taint_tracking:
        return
    try:
        tracker = _get_taint_tracker(cfg.session_id, cfg.gate)
        flows = tracker.check_args(tool_name, args, kwargs)
        if flows:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_TAINT_FLOW",
                    tool_name,
                    {"taint_flows": [f.to_dict() for f in flows]},
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            if cfg.gate.taint_block:
                raise ToolExecutionBlocked(
                    tool_name,
                    f"Taint flow detected: untrusted data from '{flows[0].source_tool}'",
                    [],
                )
    except ToolExecutionBlocked:
        raise
    except Exception as exc:
        logger.warning("[gate] taint check error (failing open): %s", exc)


def _record_taint(tool_name: str, result: Any, cfg: _InstrumentConfig) -> None:
    """
    Record fragments from a tool's return value for future taint checking.

    Silently no-ops if taint_tracking is disabled or on any error.
    """
    if not cfg.gate.taint_tracking:
        return
    try:
        tracker = _get_taint_tracker(cfg.session_id, cfg.gate)
        tracker.record_return(tool_name, result)
    except Exception as exc:
        logger.warning("[gate] taint record error (ignoring): %s", exc)


# ---------------------------------------------------------------------------
# Execution gate — runs before fn
# ---------------------------------------------------------------------------

def _run_gate_sync(
    tool_name: str,
    args: tuple,
    kwargs: dict,
    cfg: _InstrumentConfig,
) -> list:
    """
    Run the pre-execution gate for a sync tool call.

    Returns the list of detected ArgSignals (may be empty).
    Raises ToolExecutionBlocked if the policy decision is BLOCK.
    Fires TOOL_EXEC_ALERT event (fire-and-forget) for ALERT decision.
    """
    gate = cfg.gate
    signals = cfg.classifier.classify_call(args, kwargs)

    if signals:
        decision = evaluate_policy(signals, gate)

        if decision.action is PolicyAction.BLOCK:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_BLOCKED",
                    tool_name,
                    {
                        "reason":  decision.reason,
                        "signals": [s.to_dict() for s in decision.triggered_by],
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            raise ToolExecutionBlocked(tool_name, decision.reason, decision.triggered_by)

        if decision.action is PolicyAction.ALERT:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_ALERT",
                    tool_name,
                    {
                        "reason":  decision.reason,
                        "signals": [s.to_dict() for s in decision.triggered_by],
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )

    if gate.hitl:
        check_hitl_sync(
            tool_name=tool_name,
            signals=signals,
            backend_url=cfg.backend_url,
            api_key=cfg.api_key,
            session_id=cfg.session_id,
            agent_id=cfg.agent_id,
            org_id=cfg.org_id,
            timeout_s=gate.hitl_timeout_s,
        )

    return signals


async def _run_gate_async(
    tool_name: str,
    args: tuple,
    kwargs: dict,
    cfg: _InstrumentConfig,
) -> list:
    """Async version of _run_gate_sync."""
    gate = cfg.gate
    signals = cfg.classifier.classify_call(args, kwargs)

    if signals:
        decision = evaluate_policy(signals, gate)

        if decision.action is PolicyAction.BLOCK:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_BLOCKED",
                    tool_name,
                    {
                        "reason":  decision.reason,
                        "signals": [s.to_dict() for s in decision.triggered_by],
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            raise ToolExecutionBlocked(tool_name, decision.reason, decision.triggered_by)

        if decision.action is PolicyAction.ALERT:
            _fire_event(
                _make_event(
                    "TOOL_EXEC_ALERT",
                    tool_name,
                    {
                        "reason":  decision.reason,
                        "signals": [s.to_dict() for s in decision.triggered_by],
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )

    if gate.hitl:
        await check_hitl_async(
            tool_name=tool_name,
            signals=signals,
            backend_url=cfg.backend_url,
            api_key=cfg.api_key,
            session_id=cfg.session_id,
            agent_id=cfg.agent_id,
            org_id=cfg.org_id,
            timeout_s=gate.hitl_timeout_s,
        )

    return signals


# ---------------------------------------------------------------------------
# Return value scan — runs after fn
# ---------------------------------------------------------------------------

def _maybe_scan_return(
    result: Any,
    tool_name: str,
    cfg: _InstrumentConfig,
) -> None:
    """Fire TOOL_EXEC_RETURN_ALERT if the return value contains credentials."""
    if not cfg.gate.scan_return:
        return
    findings = scan_return_value(result, cfg.classifier._cfg)
    if findings:
        _fire_event(
            _make_event(
                "TOOL_EXEC_RETURN_ALERT",
                tool_name,
                {"credential_findings": findings},
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )


# ---------------------------------------------------------------------------
# Sync / async wrappers
# ---------------------------------------------------------------------------

def _wrap_sync(fn: Callable, tool_name: str, cfg: _InstrumentConfig) -> Callable:
    """Wrap a sync callable with gate, timeout, and event emission."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # ── Allowlist check ──────────────────────────────────────────────
        _check_allowlist(tool_name, cfg)

        # ── Budget check ─────────────────────────────────────────────────
        _check_and_increment_budget(tool_name, cfg)

        # ── Argument constraints ─────────────────────────────────────────
        _check_arg_constraints(tool_name, args, kwargs, cfg)

        # ── Taint check ──────────────────────────────────────────────────
        try:
            _check_taint(tool_name, args, kwargs, cfg)
        except ToolExecutionBlocked:
            raise
        except Exception as taint_exc:
            logger.warning("[gate] taint check error (failing open): %s", taint_exc)

        # ── Pre-execution gate ───────────────────────────────────────────
        try:
            signals = _run_gate_sync(tool_name, args, kwargs, cfg)
        except ToolExecutionBlocked:
            raise
        except Exception as gate_exc:
            logger.warning("[gate] pre-execution check error (failing open): %s", gate_exc)
            signals = []

        # ── Fire TOOL_EXEC_START ─────────────────────────────────────────
        start_ns = time.time_ns()
        _fire_event(
            _make_event(
                "TOOL_EXEC_START",
                tool_name,
                {
                    "args_preview": _safe_preview({"args": list(args), "kwargs": kwargs}),
                    "gate_signals": [s.to_dict() for s in signals],
                },
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )

        # ── Concurrency limit ────────────────────────────────────────────
        _sem: threading.BoundedSemaphore | None = None
        if cfg.gate.max_concurrent is not None:
            _sem = _get_sync_semaphore(cfg.session_id, cfg.gate.max_concurrent)
            if not _sem.acquire(blocking=False):
                raise ToolConcurrencyLimitExceeded(tool_name, cfg.gate.max_concurrent)

        # ── Execute with optional timeout ────────────────────────────────
        try:
            timeout_s = cfg.gate.timeout_s
            if timeout_s is not None:
                executor = _get_timeout_executor()
                future = executor.submit(fn, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    # Thread keeps running (Python limitation) but caller
                    # receives the timeout exception immediately.
                    raise ToolExecutionTimeout(tool_name, timeout_s)
            else:
                result = fn(*args, **kwargs)

        except (ToolExecutionTimeout, ToolExecutionBlocked, ToolConcurrencyLimitExceeded, ToolBudgetExceeded, ToolArgumentViolation):
            raise

        except Exception as exc:
            duration_ms = (time.time_ns() - start_ns) / 1_000_000
            _fire_event(
                _make_event(
                    "TOOL_EXEC_ERROR",
                    tool_name,
                    {
                        "duration_ms":   round(duration_ms, 2),
                        "error_type":    type(exc).__name__,
                        "error_message": str(exc)[:300],
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            raise

        finally:
            if _sem is not None:
                _sem.release()

        # ── Post-execution return scan ───────────────────────────────────
        try:
            _maybe_scan_return(result, tool_name, cfg)
        except Exception as scan_exc:
            logger.warning("[gate] return scan error (ignoring): %s", scan_exc)

        # ── Taint record ─────────────────────────────────────────────────
        _record_taint(tool_name, result, cfg)

        # ── Fire TOOL_EXEC_END ───────────────────────────────────────────
        duration_ms = (time.time_ns() - start_ns) / 1_000_000
        _fire_event(
            _make_event(
                "TOOL_EXEC_END",
                tool_name,
                {
                    "duration_ms":   round(duration_ms, 2),
                    "return_preview": _safe_preview(result),
                },
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )

        return result

    return wrapper


def _wrap_async(fn: Callable, tool_name: str, cfg: _InstrumentConfig) -> Callable:
    """Wrap an async callable with gate, timeout, and event emission."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # ── Allowlist check ──────────────────────────────────────────────
        _check_allowlist(tool_name, cfg)

        # ── Budget check ─────────────────────────────────────────────────
        _check_and_increment_budget(tool_name, cfg)

        # ── Argument constraints ─────────────────────────────────────────
        _check_arg_constraints(tool_name, args, kwargs, cfg)

        # ── Taint check ──────────────────────────────────────────────────
        try:
            _check_taint(tool_name, args, kwargs, cfg)
        except ToolExecutionBlocked:
            raise
        except Exception as taint_exc:
            logger.warning("[gate] async taint check error (failing open): %s", taint_exc)

        # ── Pre-execution gate ───────────────────────────────────────────
        try:
            signals = await _run_gate_async(tool_name, args, kwargs, cfg)
        except ToolExecutionBlocked:
            raise
        except Exception as gate_exc:
            logger.warning("[gate] async pre-execution check error (failing open): %s", gate_exc)
            signals = []

        # ── Fire TOOL_EXEC_START ─────────────────────────────────────────
        start_ns = time.time_ns()
        _fire_event(
            _make_event(
                "TOOL_EXEC_START",
                tool_name,
                {
                    "args_preview": _safe_preview({"args": list(args), "kwargs": kwargs}),
                    "gate_signals": [s.to_dict() for s in signals],
                },
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )

        # ── Concurrency limit ────────────────────────────────────────────
        _asem: asyncio.Semaphore | None = None
        if cfg.gate.max_concurrent is not None:
            _asem = _get_async_semaphore(cfg.session_id, cfg.gate.max_concurrent)
            if _asem.locked():
                raise ToolConcurrencyLimitExceeded(tool_name, cfg.gate.max_concurrent)
            await _asem.acquire()

        # ── Execute with optional timeout ────────────────────────────────
        try:
            timeout_s = cfg.gate.timeout_s
            if timeout_s is not None:
                try:
                    result = await asyncio.wait_for(
                        fn(*args, **kwargs),
                        timeout=timeout_s,
                    )
                except asyncio.TimeoutError:
                    raise ToolExecutionTimeout(tool_name, timeout_s)
            else:
                result = await fn(*args, **kwargs)

        except (ToolExecutionTimeout, ToolExecutionBlocked, ToolConcurrencyLimitExceeded, ToolBudgetExceeded, ToolArgumentViolation):
            raise

        except Exception as exc:
            duration_ms = (time.time_ns() - start_ns) / 1_000_000
            _fire_event(
                _make_event(
                    "TOOL_EXEC_ERROR",
                    tool_name,
                    {
                        "duration_ms":   round(duration_ms, 2),
                        "error_type":    type(exc).__name__,
                        "error_message": str(exc)[:300],
                    },
                    cfg.agent_id, cfg.session_id, cfg.org_id,
                ),
                cfg.backend_url, cfg.api_key,
            )
            raise

        finally:
            if _asem is not None:
                _asem.release()

        # ── Post-execution return scan ───────────────────────────────────
        try:
            _maybe_scan_return(result, tool_name, cfg)
        except Exception as scan_exc:
            logger.warning("[gate] async return scan error (ignoring): %s", scan_exc)

        # ── Taint record ─────────────────────────────────────────────────
        _record_taint(tool_name, result, cfg)

        # ── Fire TOOL_EXEC_END ───────────────────────────────────────────
        duration_ms = (time.time_ns() - start_ns) / 1_000_000
        _fire_event(
            _make_event(
                "TOOL_EXEC_END",
                tool_name,
                {
                    "duration_ms":    round(duration_ms, 2),
                    "return_preview": _safe_preview(result),
                },
                cfg.agent_id, cfg.session_id, cfg.org_id,
            ),
            cfg.backend_url, cfg.api_key,
        )

        return result

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
      1. LangChain BaseTool   — has .run() + .name str → wrap .run() and .arun()
      2. AutoGen FunctionTool — has ._func callable → wrap ._func
      3. Async callable       — asyncio.iscoroutinefunction → async wrapper
      4. Sync callable        — any callable → sync wrapper
      5. Unknown              — returned unchanged with a debug log
    """
    if type(obj).__name__ in _MOCK_TYPE_NAMES:
        logger.debug("aegivis instrument: skipping mock object %s", type(obj).__name__)
        return obj

    # 1. LangChain BaseTool
    if (
        hasattr(obj, "run")
        and callable(obj.run)
        and isinstance(getattr(obj, "name", None), str)
    ):
        name = _tool_name(obj)
        obj.run = _wrap_sync(obj.run, name, cfg)
        if hasattr(obj, "arun") and callable(obj.arun):
            obj.arun = _wrap_async(obj.arun, name, cfg)
        return obj

    # 2. AutoGen FunctionTool
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

    # 5. Unknown
    logger.debug(
        "aegivis instrument: unrecognised tool type %s — skipping",
        type(obj).__name__,
    )
    return obj


# ---------------------------------------------------------------------------
# Public API — instrument()
# ---------------------------------------------------------------------------

def instrument(
    tools: Any,
    *,
    agent_id:              str                           = "",
    session_id:            str                           = "",
    backend_url:           str                           = "",
    api_key:               str                           = "",
    org_id:                str                           = "",
    # Gate options
    block_on:              frozenset[str] | set[str] | None = None,
    alert_on:              frozenset[str] | set[str] | None = None,
    timeout_s:             float | None                  = None,
    hitl:                  bool                          = False,
    hitl_timeout_s:        float                         = 30.0,
    scan_return:           bool                          = True,
    max_concurrent:        int | None                    = None,
    taint_tracking:        bool                          = False,
    taint_block:           bool                          = False,
    taint_untrusted_tools: frozenset[str] | set[str] | None = None,
    allowed_tools:         frozenset[str] | set[str] | None = None,
    max_calls_per_session: int | None                    = None,
    max_calls_per_tool:    int | None                    = None,
    path_prefixes:         frozenset[str] | set[str] | None = None,
    url_domains:           frozenset[str] | set[str] | None = None,
    classifier_config:     ClassifierConfig | None       = None,
) -> Any:
    """
    Instrument one or more tools to emit execution events to Aegivis.

    All gate parameters are optional and off by default.  Passing none of
    them gives identical behaviour to the previous version of this function.

    Args:
        tools:             A single tool or a list of tools.
        agent_id:          Override the agent ID.
        session_id:        Override the session ID.
        backend_url:       Backend URL.
        api_key:           API key.
        org_id:            Org ID.
        block_on:          Signal types that block execution before the tool
                           runs.  Example: ``frozenset({"credential"})``.
        alert_on:          Signal types that fire an alert event but allow
                           execution to continue.
        timeout_s:         Hard execution deadline in seconds.
        hitl:              Require synchronous human approval before each
                           tool execution.
        hitl_timeout_s:    Max seconds to wait for a HITL decision.
        scan_return:       Scan the return value for credentials.
        max_concurrent:        Maximum number of concurrent executions of any
                               instrumented tool within the same session.
                               Raises ``ToolConcurrencyLimitExceeded`` immediately
                               if the limit is already reached (no queuing).
        taint_tracking:        Enable cross-tool taint flow tracking.  When True,
                               return values from untrusted tools are stored as
                               fragment sets; subsequent tool arguments are checked
                               against those fragments before execution.
        taint_block:           If True, raise ``ToolExecutionBlocked`` when a taint
                               flow is detected.  If False (default), only fire a
                               ``TOOL_EXEC_TAINT_FLOW`` event and allow execution.
        taint_untrusted_tools: Tool names whose return values are tracked as
                               untrusted.  Defaults to ``DEFAULT_UNTRUSTED_TOOLS``
                               (web search, file read, email read, etc.).
        allowed_tools:         Explicit set of tool names permitted to execute.
                               ``None`` (default) allows all tools.  Any tool not
                               in this set raises ``ToolExecutionBlocked`` before
                               any other gate logic.
        max_calls_per_session: Maximum total tool executions across all tools
                               in the session.  Raises ``ToolBudgetExceeded``
                               when exhausted.
        max_calls_per_tool:    Maximum executions of each individual tool within
                               the session.  Raises ``ToolBudgetExceeded`` when
                               exhausted.
        path_prefixes:         Allowed path prefixes for any path-shaped argument.
                               Example: ``{"/data/", "/tmp/"}``.  Any path arg
                               not under these prefixes raises
                               ``ToolArgumentViolation``.
        url_domains:           Allowed hostnames for any URL-shaped argument.
                               Supports subdomain matching: ``{"example.com"}``
                               also allows ``api.example.com``.
        classifier_config:     Override classifier thresholds.

    Returns:
        The instrumented tool or list of tools.
    """
    gate = GateConfig(
        block_on=frozenset(block_on) if block_on is not None else frozenset(),
        alert_on=frozenset(alert_on) if alert_on is not None else frozenset(),
        timeout_s=timeout_s,
        hitl=hitl,
        hitl_timeout_s=hitl_timeout_s,
        scan_return=scan_return,
        max_concurrent=max_concurrent,
        taint_tracking=taint_tracking,
        taint_block=taint_block,
        taint_untrusted_tools=(
            frozenset(taint_untrusted_tools)
            if taint_untrusted_tools is not None
            else frozenset()
        ),
        allowed_tools=frozenset(allowed_tools) if allowed_tools is not None else None,
        max_calls_per_session=max_calls_per_session,
        max_calls_per_tool=max_calls_per_tool,
        path_prefixes=frozenset(path_prefixes) if path_prefixes is not None else None,
        url_domains=frozenset(url_domains) if url_domains is not None else None,
    )
    cfg = _InstrumentConfig(
        agent_id=agent_id,
        session_id=session_id,
        backend_url=backend_url,
        api_key=api_key,
        org_id=org_id,
        gate=gate,
        classifier_config=classifier_config,
    )
    if isinstance(tools, list):
        return [_wrap_one(t, cfg) for t in tools]
    return _wrap_one(tools, cfg)


# ---------------------------------------------------------------------------
# @instrument.tool decorator
# ---------------------------------------------------------------------------

class _ToolDecorator:
    """Provides the ``@instrument.tool`` decorator syntax."""

    def __call__(
        self,
        fn:                    Callable | None              = None,
        *,
        agent_id:              str                          = "",
        session_id:            str                          = "",
        backend_url:           str                          = "",
        api_key:               str                          = "",
        org_id:                str                          = "",
        block_on:              frozenset[str] | set[str] | None = None,
        alert_on:              frozenset[str] | set[str] | None = None,
        timeout_s:             float | None                 = None,
        hitl:                  bool                         = False,
        hitl_timeout_s:        float                        = 30.0,
        scan_return:           bool                         = True,
        max_concurrent:        int | None                   = None,
        taint_tracking:        bool                         = False,
        taint_block:           bool                         = False,
        taint_untrusted_tools: frozenset[str] | set[str] | None = None,
        allowed_tools:         frozenset[str] | set[str] | None = None,
        max_calls_per_session: int | None                    = None,
        max_calls_per_tool:    int | None                    = None,
        path_prefixes:         frozenset[str] | set[str] | None = None,
        url_domains:           frozenset[str] | set[str] | None = None,
        classifier_config:     ClassifierConfig | None      = None,
    ) -> Any:
        _kwargs = dict(
            agent_id=agent_id, session_id=session_id, backend_url=backend_url,
            api_key=api_key, org_id=org_id, block_on=block_on, alert_on=alert_on,
            timeout_s=timeout_s, hitl=hitl, hitl_timeout_s=hitl_timeout_s,
            scan_return=scan_return, max_concurrent=max_concurrent,
            taint_tracking=taint_tracking, taint_block=taint_block,
            taint_untrusted_tools=taint_untrusted_tools,
            allowed_tools=allowed_tools,
            max_calls_per_session=max_calls_per_session,
            max_calls_per_tool=max_calls_per_tool,
            path_prefixes=path_prefixes,
            url_domains=url_domains,
            classifier_config=classifier_config,
        )
        # @instrument.tool (no parentheses)
        if fn is not None and callable(fn):
            return instrument(fn, **_kwargs)
        # @instrument.tool(...) — return parameterised decorator
        def decorator(func: Callable) -> Callable:
            return instrument(func, **_kwargs)
        return decorator


instrument.tool = _ToolDecorator()  # type: ignore[attr-defined]
