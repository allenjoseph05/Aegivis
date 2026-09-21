"""
Execution Gate Policy Engine — SDK Execution Gate.

Evaluates ArgSignals from the classifier against developer-declared rules
and decides whether a tool call should:

  - ``allow``  — proceed with execution (no signals matched policy)
  - ``alert``  — proceed but fire an asynchronous warning event
  - ``block``  — raise ToolExecutionBlocked before fn is called

Two-tier evaluation:

  Tier 1 — Local (< 1 ms, no network):
    Compares detected signals against ``block_on`` and ``alert_on`` sets.
    Raises ``ToolExecutionBlocked`` if any signal is in ``block_on``.

  Tier 2 — Backend HITL (optional, ~100–500 ms, requires backend):
    Only when ``hitl=True`` on the decorated tool.
    POSTs arg signals to ``/v1/approvals``, polls for a human decision.
    Raises ``ToolExecutionBlocked`` if denied or if the deadline elapses.
    Uses synchronous httpx for sync wrappers, async httpx for async wrappers.

Design principles:
  - Fail-open on classifier / policy errors: if our code throws, the tool
    runs normally.  The policy must not break functionality.
  - Explicit over inferred: ``block_on`` and ``alert_on`` are declared by
    the developer; defaults are empty (observe-only out of the box).
  - No hardcoded signal names in logic: all sets are parameters.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .arg_classifier import ArgSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolExecutionBlocked(RuntimeError):
    """
    Raised by the execution gate when a tool call is denied.

    Attributes:
        tool_name:  The name of the tool that was blocked.
        reason:     Human-readable explanation (signal type or HITL decision).
        signals:    The ArgSignal list that triggered the block.
    """

    def __init__(
        self,
        tool_name: str,
        reason: str,
        signals: list[ArgSignal],
    ) -> None:
        self.tool_name = tool_name
        self.reason = reason
        self.signals = signals
        super().__init__(
            f"Tool '{tool_name}' blocked by Aegivis execution gate: {reason}"
        )


class ToolExecutionTimeout(RuntimeError):
    """
    Raised when a tool call exceeds its configured time limit.

    Attributes:
        tool_name:  The name of the timed-out tool.
        timeout_s:  The limit that was exceeded, in seconds.
    """

    def __init__(self, tool_name: str, timeout_s: float) -> None:
        self.tool_name = tool_name
        self.timeout_s = timeout_s
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_s}s"
        )


class ToolConcurrencyLimitExceeded(RuntimeError):
    """
    Raised when a tool call is rejected because the session has reached
    its maximum concurrent tool execution limit.

    Fails fast (does not queue) — the caller should retry or report the
    congestion rather than block the event loop or thread pool.

    Attributes:
        tool_name:      The tool that was rejected.
        max_concurrent: The limit that was exceeded.
    """

    def __init__(self, tool_name: str, max_concurrent: int) -> None:
        self.tool_name = tool_name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Tool '{tool_name}' rejected: session concurrency limit "
            f"{max_concurrent} already reached"
        )


class ToolArgumentViolation(RuntimeError):
    """
    Raised when a tool argument violates a declared constraint.

    Checked before execution — the tool body is never called.

    Attributes:
        tool_name:       The tool whose argument violated a constraint.
        constraint_type: ``"path_prefix"`` or ``"url_domain"``.
        value:           The argument value that was rejected.
        allowed:         Human-readable representation of the allowed set.
    """

    def __init__(
        self,
        tool_name: str,
        constraint_type: str,
        value: str,
        allowed: str,
    ) -> None:
        self.tool_name = tool_name
        self.constraint_type = constraint_type
        self.value = value
        self.allowed = allowed
        super().__init__(
            f"Tool '{tool_name}' argument blocked by {constraint_type} constraint: "
            f"'{value[:80]}' not in allowed set {allowed}"
        )


class ToolBudgetExceeded(RuntimeError):
    """
    Raised when a tool call is rejected because the session has exhausted
    its configured call budget.

    Attributes:
        tool_name:   The tool that was blocked.
        budget_type: ``"session"`` (total calls across all tools) or
                     ``"tool"`` (per-tool call limit).
        limit:       The configured maximum.
        actual:      Current call count that triggered the block.
    """

    def __init__(
        self,
        tool_name: str,
        budget_type: str,
        limit: int,
        actual: int,
    ) -> None:
        self.tool_name = tool_name
        self.budget_type = budget_type
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"Tool '{tool_name}' blocked: {budget_type} call budget of "
            f"{limit} exhausted ({actual} calls already made)"
        )


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

class PolicyAction(str, Enum):
    ALLOW = "allow"
    ALERT = "alert"
    BLOCK = "block"


@dataclass
class GateConfig:
    """
    Per-tool gate configuration.

    All fields are optional.  Zero configuration → gate is pass-through
    (events still fire; no blocking or alerting).

    Attributes:
        block_on:            Signal types that cause an immediate block.
                             Example: ``frozenset({"credential"})``
        alert_on:            Signal types that fire an alert event but allow
                             execution to continue.
        timeout_s:           Hard execution deadline in seconds.
                             ``None`` disables the timeout.
        hitl:                If True, make a synchronous HITL approval request
                             to the backend before execution.
        hitl_timeout_s:      Maximum seconds to wait for a HITL decision.
                             If the deadline elapses, the call is blocked.
        scan_return:         If True, scan the return value for credential-shaped
                             content and fire a return-alert event if found.
        max_concurrent:      Maximum concurrent tool executions per session.
                             ``None`` disables the limit.
        taint_tracking:      If True, enable cross-tool taint flow tracking.
                             Fragments from untrusted tools are stored and
                             checked against subsequent tool arguments.
        taint_block:         If True, raise ``ToolExecutionBlocked`` when a
                             taint flow is detected.  If False (default), only
                             fire a ``TOOL_EXEC_TAINT_FLOW`` event.
        taint_untrusted_tools: Tool names whose return values are tracked as
                             untrusted.  Defaults to ``DEFAULT_UNTRUSTED_TOOLS``
                             from the taint tracker module.
        allowed_tools:       Explicit set of tool names permitted to execute.
                             ``None`` (default) disables the allowlist — all
                             tools are allowed.  An empty frozenset blocks every
                             tool.  Any tool whose name is not in the set raises
                             ``ToolExecutionBlocked`` before any other gate logic
                             runs.
        max_calls_per_session: Maximum total tool executions across all tools
                             in the session.  ``None`` disables the limit.
                             Raises ``ToolBudgetExceeded`` when exhausted.
        max_calls_per_tool:  Maximum executions of this specific tool within
                             the session.  ``None`` disables the limit.
                             Raises ``ToolBudgetExceeded`` when exhausted.
        path_prefixes:       Allowed path prefixes for any path-shaped argument.
                             ``None`` disables the constraint.  Example:
                             ``frozenset({"/data/", "/tmp/"})`` — the tool may
                             only access files under those directories.
                             Raises ``ToolArgumentViolation`` on violation.
        url_domains:         Allowed hostnames for any URL-shaped argument.
                             ``None`` disables the constraint.  Supports
                             subdomain matching: ``{"example.com"}`` also allows
                             ``api.example.com``.
                             Raises ``ToolArgumentViolation`` on violation.
    """
    block_on:               frozenset[str]       = field(default_factory=frozenset)
    alert_on:               frozenset[str]       = field(default_factory=frozenset)
    timeout_s:              float | None         = None
    hitl:                   bool                 = False
    hitl_timeout_s:         float                = 30.0
    scan_return:            bool                 = True
    max_concurrent:         int | None           = None
    taint_tracking:         bool                 = False
    taint_block:            bool                 = False
    taint_untrusted_tools:  frozenset[str]       = field(default_factory=frozenset)
    allowed_tools:          frozenset[str] | None = None
    max_calls_per_session:  int | None           = None
    max_calls_per_tool:     int | None           = None
    path_prefixes:          frozenset[str] | None = None
    url_domains:            frozenset[str] | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """Result of evaluating signals against a GateConfig."""
    action:          PolicyAction
    triggered_by:    list[ArgSignal]   # signals that caused the decision
    reason:          str


# ---------------------------------------------------------------------------
# Local policy evaluation (Tier 1)
# ---------------------------------------------------------------------------

def evaluate_policy(
    signals: list[ArgSignal],
    config: GateConfig,
) -> PolicyDecision:
    """
    Evaluate detected signals against the gate config.

    Blocking takes priority over alerting: if a signal appears in both
    ``block_on`` and ``alert_on``, it blocks.

    Returns a ``PolicyDecision`` — never raises.
    """
    block_triggers = [s for s in signals if s.signal in config.block_on]
    if block_triggers:
        return PolicyDecision(
            action=PolicyAction.BLOCK,
            triggered_by=block_triggers,
            reason=(
                f"Signal(s) {[s.signal for s in block_triggers]} matched block_on policy"
            ),
        )

    alert_triggers = [s for s in signals if s.signal in config.alert_on]
    if alert_triggers:
        return PolicyDecision(
            action=PolicyAction.ALERT,
            triggered_by=alert_triggers,
            reason=(
                f"Signal(s) {[s.signal for s in alert_triggers]} matched alert_on policy"
            ),
        )

    return PolicyDecision(
        action=PolicyAction.ALLOW,
        triggered_by=[],
        reason="No signals matched block_on or alert_on",
    )


# ---------------------------------------------------------------------------
# Backend HITL — synchronous (Tier 2, sync tools)
# ---------------------------------------------------------------------------

def check_hitl_sync(
    tool_name: str,
    signals: list[ArgSignal],
    backend_url: str,
    api_key: str,
    session_id: str,
    agent_id: str,
    org_id: str,
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> None:
    """
    Create a HITL approval request and poll for a human decision.

    Raises ``ToolExecutionBlocked`` if:
      - The decision is ``denied`` or ``expired``.
      - The polling deadline elapses without a decision.

    Fails open (returns without raising) if:
      - The backend is unreachable.
      - The approval ID cannot be obtained.

    This keeps the gate non-fatal when the backend is unavailable.
    """
    if not backend_url:
        logger.warning("[gate/hitl] backend_url not configured — skipping HITL check")
        return

    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        logger.warning("[gate/hitl] httpx not available — skipping HITL check")
        return

    approval_id = _create_approval(
        tool_name=tool_name,
        signals=signals,
        backend_url=backend_url,
        api_key=api_key,
        session_id=session_id,
        agent_id=agent_id,
        org_id=org_id,
    )
    if not approval_id:
        return  # fail open

    _poll_approval_sync(
        approval_id=approval_id,
        tool_name=tool_name,
        signals=signals,
        backend_url=backend_url,
        api_key=api_key,
        deadline=time.monotonic() + timeout_s,
        poll_interval_s=poll_interval_s,
    )


async def check_hitl_async(
    tool_name: str,
    signals: list[ArgSignal],
    backend_url: str,
    api_key: str,
    session_id: str,
    agent_id: str,
    org_id: str,
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> None:
    """Async version of ``check_hitl_sync``."""
    if not backend_url:
        logger.warning("[gate/hitl] backend_url not configured — skipping HITL check")
        return

    try:
        import httpx   # noqa: PLC0415
        import asyncio  # noqa: PLC0415
    except ImportError:
        logger.warning("[gate/hitl] httpx not available — skipping HITL check")
        return

    approval_id = _create_approval(
        tool_name=tool_name,
        signals=signals,
        backend_url=backend_url,
        api_key=api_key,
        session_id=session_id,
        agent_id=agent_id,
        org_id=org_id,
    )
    if not approval_id:
        return

    await _poll_approval_async(
        approval_id=approval_id,
        tool_name=tool_name,
        signals=signals,
        backend_url=backend_url,
        api_key=api_key,
        deadline=time.monotonic() + timeout_s,
        poll_interval_s=poll_interval_s,
    )


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

def _create_approval(
    tool_name: str,
    signals: list[ArgSignal],
    backend_url: str,
    api_key: str,
    session_id: str,
    agent_id: str,
    org_id: str,
) -> str | None:
    """POST a new approval request. Returns the approval_id or None on error."""
    try:
        import httpx  # noqa: PLC0415

        payload = {
            "tool_name":  tool_name,
            "session_id": session_id,
            "agent_id":   agent_id,
            "org_id":     org_id,
            "signals":    [s.to_dict() for s in signals],
            "source":     "sdk_gate",
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{backend_url.rstrip('/')}/v1/approvals",
                json=payload,
                headers={"X-API-Key": api_key},
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "[gate/hitl] approval creation returned %s — skipping HITL",
                    resp.status_code,
                )
                return None
            return resp.json().get("id")
    except Exception as exc:
        logger.warning("[gate/hitl] approval creation failed: %s — skipping HITL", exc)
        return None


def _poll_approval_sync(
    approval_id: str,
    tool_name: str,
    signals: list[ArgSignal],
    backend_url: str,
    api_key: str,
    deadline: float,
    poll_interval_s: float,
) -> None:
    """Poll approval endpoint until decided or deadline exceeded."""
    import httpx  # noqa: PLC0415

    url = f"{backend_url.rstrip('/')}/v1/approvals/{approval_id}"
    headers = {"X-API-Key": api_key}

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                status = resp.json().get("status", "pending")
                if status == "approved":
                    return
                if status in ("denied", "expired"):
                    raise ToolExecutionBlocked(
                        tool_name,
                        f"HITL decision: {status}",
                        signals,
                    )
        except ToolExecutionBlocked:
            raise
        except Exception as exc:
            logger.debug("[gate/hitl] poll error (retrying): %s", exc)

        time.sleep(poll_interval_s)

    raise ToolExecutionBlocked(tool_name, "HITL timeout — no decision received", signals)


async def _poll_approval_async(
    approval_id: str,
    tool_name: str,
    signals: list[ArgSignal],
    backend_url: str,
    api_key: str,
    deadline: float,
    poll_interval_s: float,
) -> None:
    """Async poll approval endpoint until decided or deadline exceeded."""
    import httpx   # noqa: PLC0415
    import asyncio  # noqa: PLC0415

    url = f"{backend_url.rstrip('/')}/v1/approvals/{approval_id}"
    headers = {"X-API-Key": api_key}

    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                status = resp.json().get("status", "pending")
                if status == "approved":
                    return
                if status in ("denied", "expired"):
                    raise ToolExecutionBlocked(
                        tool_name,
                        f"HITL decision: {status}",
                        signals,
                    )
        except ToolExecutionBlocked:
            raise
        except Exception as exc:
            logger.debug("[gate/hitl] async poll error (retrying): %s", exc)

        await asyncio.sleep(poll_interval_s)

    raise ToolExecutionBlocked(tool_name, "HITL timeout — no decision received", signals)


# ---------------------------------------------------------------------------
# Return value scanner
# ---------------------------------------------------------------------------

def scan_return_value(
    result: Any,
    classifier_config: Any,  # ClassifierConfig — avoid circular import
) -> list[dict[str, str]]:
    """
    Scan a tool's return value for credential-shaped content.

    Traverses strings, dicts, and lists (limited depth and breadth).
    Returns a list of finding dicts with keys ``path``, ``preview``.
    """
    from .arg_classifier import (  # noqa: PLC0415
        _is_credential,
        _credential_preview,
    )

    findings: list[dict[str, str]] = []
    _scan_node(
        result,
        "$",
        classifier_config,
        findings,
        depth=0,
        _is_credential=_is_credential,
        _credential_preview=_credential_preview,
    )
    return findings


def _scan_node(
    value: Any,
    path: str,
    cfg: Any,
    findings: list[dict[str, str]],
    depth: int,
    _is_credential: Any,
    _credential_preview: Any,
) -> None:
    if depth > 5:
        return

    if isinstance(value, str):
        if _is_credential(
            value,
            cfg.credential_min_length,
            cfg.credential_entropy_threshold,
            cfg.credential_prefixes,
        ):
            findings.append({"path": path, "preview": _credential_preview(value)})

    elif isinstance(value, dict):
        for k, v in value.items():
            _scan_node(
                v, f"{path}[{k!r}]", cfg, findings, depth + 1,
                _is_credential, _credential_preview,
            )

    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value[:20]):   # sample up to 20 items
            _scan_node(
                v, f"{path}[{i}]", cfg, findings, depth + 1,
                _is_credential, _credential_preview,
            )
