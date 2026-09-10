"""
Core Sandbox Adapter Framework — Phase 16.0.

Provides a first-match-wins dispatch registry so any number of adapter
plug-ins can be registered.  At TOOL_CALL_START the proxy calls
``SandboxRegistry.run(tool_name, args, ctx)``; the first adapter whose
``can_handle()`` returns True runs a dry-run and returns a ``SandboxResult``.

Design principles:
  - Fail-open: adapter errors → safe=True with error field populated.
  - Timeout: asyncio.wait_for(timeout=ctx.timeout_ms / 1000) → safe result.
  - can_handle() errors → skip that adapter, try the next.
  - No adapter matches → return None (proxy continues normally).

Module-level singleton:
    from .sandbox import SandboxRegistry, SandboxContext
    result = await SandboxRegistry.run(tc_name, args, ctx)
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── SandboxResult ─────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    """
    Standardised result returned by any sandbox adapter.

    Attributes:
        adapter:        Short name of the adapter that ran ("email", "git", …).
        executed:       True if the adapter actually ran a dry-run operation
                        (e.g. ``git push --dry-run``); False if purely structural.
        safe:           True = proxy should proceed; False = escalate/block.
        scope_estimate: Adapter-specific scope dict (serialisable to JSONB).
        preview:        Human-readable summary (≤ 500 chars).
        signals:        Flags raised by the adapter (≤ 10 items).
        latency_ms:     Wall-clock time the adapter took, in milliseconds.
        error:          Error message when the adapter itself failed; safe=True.
    """

    adapter:        str
    executed:       bool
    safe:           bool
    scope_estimate: dict                = field(default_factory=dict)
    preview:        str                 = ""
    signals:        list[str]           = field(default_factory=list)
    latency_ms:     float               = 0.0
    error:          str | None          = None

    def to_dict(self) -> dict:
        """JSON-serialisable dict suitable for storage in the ``security`` JSONB column."""
        return {
            "adapter":        self.adapter,
            "executed":       self.executed,
            "safe":           self.safe,
            "scope_estimate": self.scope_estimate,
            "preview":        self.preview[:500],
            "signals":        self.signals[:10],
            "latency_ms":     round(self.latency_ms, 2),
            "error":          self.error,
        }


# ── SandboxContext ────────────────────────────────────────────────────────────

@dataclass
class SandboxContext:
    """
    Caller-supplied context forwarded to every adapter's ``dry_run()``.

    Attributes:
        session_id:  Active session identifier.
        agent_id:    Agent identifier (for logging / policy lookup).
        org_id:      Organisation identifier.
        spawn_depth: Agent spawn depth (0 = root).
        timeout_ms:  Hard timeout for the entire dry-run, in milliseconds.
        blast_score: Blast radius score for the triggering tool call.
        verb_class:  Blast radius verb class ("safe", "medium", "critical").
    """

    session_id:  str
    agent_id:    str
    org_id:      str
    spawn_depth: int   = 0
    timeout_ms:  int   = 5_000
    blast_score: float = 0.0
    verb_class:  str   = "safe"


# ── SandboxAdapter ABC ────────────────────────────────────────────────────────

class SandboxAdapter(ABC):
    """
    Abstract base class that every sandbox adapter must subclass.

    Subclasses must define:
        name (class attribute): short snake_case identifier.
        can_handle():           True if this adapter applies to the tool call.
        dry_run():              Perform the dry-run and return a SandboxResult.
    """

    name: str = ""

    @abstractmethod
    async def can_handle(self, tool_name: str, args: dict) -> bool:
        """Return True if this adapter can dry-run the given tool call."""

    @abstractmethod
    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        """Run the dry-run and return a SandboxResult."""


# ── SandboxAdapterRegistry ────────────────────────────────────────────────────

class SandboxAdapterRegistry:
    """
    Thread-safe (GIL is sufficient) ordered registry of SandboxAdapters.

    Dispatch is first-match-wins: the first adapter whose ``can_handle()``
    returns True handles the call.  Adapters are checked in registration order.
    """

    def __init__(self) -> None:
        self._adapters: list[SandboxAdapter] = []

    def register(self, adapter: SandboxAdapter) -> None:
        """Append an adapter to the registry."""
        self._adapters.append(adapter)
        logger.debug("[SANDBOX] Registered adapter: %s", adapter.name)

    def registered_names(self) -> list[str]:
        """Return the names of all registered adapters in order."""
        return [a.name for a in self._adapters]

    def _clear(self) -> None:
        """Reset the registry. For unit tests only."""
        self._adapters.clear()

    async def run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult | None:
        """
        Dispatch to the first matching adapter and return its SandboxResult.

        Returns None if no adapter matched.

        Error handling:
          - can_handle() raises  → log warning, skip adapter (fail-open).
          - dry_run() raises     → return SandboxResult(safe=True, error=…).
          - asyncio.TimeoutError → return SandboxResult(safe=True, error=timeout).
        """
        for adapter in self._adapters:
            # ── can_handle check ──────────────────────────────────────────────
            try:
                if not await adapter.can_handle(tool_name, args):
                    continue
            except Exception as exc:
                logger.warning(
                    "[SANDBOX] can_handle() raised for adapter=%s tool=%s: %s",
                    adapter.name, tool_name, exc,
                )
                continue  # skip — fail-open

            # ── dry_run with timeout ──────────────────────────────────────────
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    adapter.dry_run(tool_name, args, ctx),
                    timeout=ctx.timeout_ms / 1000.0,
                )
                logger.debug(
                    "[SANDBOX] adapter=%s tool=%s safe=%s latency=%.1fms",
                    adapter.name, tool_name, result.safe,
                    (time.monotonic() - t0) * 1000,
                )
                return result

            except asyncio.TimeoutError:
                latency = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[SANDBOX] dry_run() timed out: adapter=%s tool=%s timeout_ms=%d",
                    adapter.name, tool_name, ctx.timeout_ms,
                )
                return SandboxResult(
                    adapter=adapter.name,
                    executed=False,
                    safe=True,   # fail-open on timeout
                    scope_estimate={},
                    preview="Sandbox dry-run timed out.",
                    signals=["timeout"],
                    latency_ms=latency,
                    error=f"Dry-run timed out after {ctx.timeout_ms} ms",
                )

            except Exception as exc:
                latency = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[SANDBOX] dry_run() raised: adapter=%s tool=%s error=%r",
                    adapter.name, tool_name, exc,
                )
                return SandboxResult(
                    adapter=adapter.name,
                    executed=False,
                    safe=True,   # fail-open on adapter errors
                    scope_estimate={},
                    preview=f"Sandbox error: {exc!r}",
                    signals=["error"],
                    latency_ms=latency,
                    error=str(exc),
                )

        return None  # no adapter matched


# ── Module-level singleton ────────────────────────────────────────────────────

#: Global registry — import and call ``SandboxRegistry.run(...)`` from intercept.py.
SandboxRegistry = SandboxAdapterRegistry()
