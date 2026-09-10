"""
Unit tests for the Sandbox Adapter Framework — Phase 16.0.

Tests: SandboxResult, SandboxContext, SandboxAdapter ABC,
       SandboxAdapterRegistry dispatch and error handling.
"""
from __future__ import annotations

import asyncio
import pytest

from app.security.sandbox import (
    SandboxAdapter,
    SandboxAdapterRegistry,
    SandboxContext,
    SandboxResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ctx(**kw) -> SandboxContext:
    defaults = dict(
        session_id="sess-1", agent_id="agent-1", org_id="org-1",
        timeout_ms=1000,
    )
    defaults.update(kw)
    return SandboxContext(**defaults)


def make_result(**kw) -> SandboxResult:
    defaults = dict(
        adapter="test", executed=True, safe=True,
        scope_estimate={"rows": 5},
        preview="preview text",
        signals=["ok"],
        latency_ms=12.3,
        error=None,
    )
    defaults.update(kw)
    return SandboxResult(**defaults)


# ── SandboxResult ─────────────────────────────────────────────────────────────

class TestSandboxResult:
    def test_to_dict_contains_all_keys(self):
        r = make_result()
        d = r.to_dict()
        assert set(d) == {
            "adapter", "executed", "safe", "scope_estimate",
            "preview", "signals", "latency_ms", "error",
        }

    def test_preview_capped_at_500(self):
        r = make_result(preview="x" * 600)
        assert len(r.to_dict()["preview"]) == 500

    def test_signals_capped_at_10(self):
        r = make_result(signals=[str(i) for i in range(20)])
        assert len(r.to_dict()["signals"]) == 10

    def test_latency_rounded(self):
        r = make_result(latency_ms=12.3456789)
        assert r.to_dict()["latency_ms"] == 12.35

    def test_safe_false_preserved(self):
        r = make_result(safe=False)
        assert r.to_dict()["safe"] is False

    def test_error_none_preserved(self):
        r = make_result(error=None)
        assert r.to_dict()["error"] is None

    def test_error_string_preserved(self):
        r = make_result(error="something went wrong")
        assert r.to_dict()["error"] == "something went wrong"

    def test_scope_estimate_preserved(self):
        r = make_result(scope_estimate={"a": 1, "b": [2, 3]})
        assert r.to_dict()["scope_estimate"] == {"a": 1, "b": [2, 3]}


# ── SandboxContext ────────────────────────────────────────────────────────────

class TestSandboxContext:
    def test_defaults(self):
        ctx = SandboxContext(session_id="s", agent_id="a", org_id="o")
        assert ctx.spawn_depth == 0
        assert ctx.timeout_ms  == 5_000
        assert ctx.blast_score == 0.0
        assert ctx.verb_class  == "safe"

    def test_custom_values(self):
        ctx = make_ctx(blast_score=0.9, verb_class="critical", timeout_ms=2000)
        assert ctx.blast_score == 0.9
        assert ctx.verb_class  == "critical"
        assert ctx.timeout_ms  == 2000


# ── SandboxAdapter ABC ────────────────────────────────────────────────────────

class TestSandboxAdapterABC:
    def test_abstract_methods_enforced(self):
        """Cannot instantiate SandboxAdapter directly."""
        with pytest.raises(TypeError):
            SandboxAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class Dummy(SandboxAdapter):
            name = "dummy"
            async def can_handle(self, tool_name, args): return True
            async def dry_run(self, tool_name, args, ctx): return make_result()

        d = Dummy()
        assert d.name == "dummy"


# ── SandboxAdapterRegistry ────────────────────────────────────────────────────

class _AlwaysMatch(SandboxAdapter):
    name = "always"
    def __init__(self, safe=True, signal="ok"):
        self._safe = safe
        self._signal = signal
    async def can_handle(self, tool_name, args):
        return True
    async def dry_run(self, tool_name, args, ctx):
        return make_result(adapter=self.name, safe=self._safe, signals=[self._signal])


class _NeverMatch(SandboxAdapter):
    name = "never"
    async def can_handle(self, tool_name, args):
        return False
    async def dry_run(self, tool_name, args, ctx):
        return make_result(adapter=self.name)


class _MatchByName(SandboxAdapter):
    def __init__(self, match_name: str):
        self.name = f"match_{match_name}"
        self._match = match_name
    async def can_handle(self, tool_name, args):
        return tool_name == self._match
    async def dry_run(self, tool_name, args, ctx):
        return make_result(adapter=self.name)


class _CanHandleRaises(SandboxAdapter):
    name = "can_handle_raises"
    async def can_handle(self, tool_name, args):
        raise RuntimeError("whoops")
    async def dry_run(self, tool_name, args, ctx):
        return make_result(adapter=self.name)


class _DryRunRaises(SandboxAdapter):
    name = "dry_run_raises"
    async def can_handle(self, tool_name, args):
        return True
    async def dry_run(self, tool_name, args, ctx):
        raise ValueError("dry run error")


class _Slow(SandboxAdapter):
    name = "slow"
    async def can_handle(self, tool_name, args):
        return True
    async def dry_run(self, tool_name, args, ctx):
        await asyncio.sleep(10)
        return make_result(adapter=self.name)


class TestSandboxAdapterRegistry:
    def setup_method(self):
        self.reg = SandboxAdapterRegistry()

    def run(self, coro):
        return asyncio.run(coro)

    def test_register_and_names(self):
        self.reg.register(_AlwaysMatch())
        self.reg.register(_NeverMatch())
        assert self.reg.registered_names() == ["always", "never"]

    def test_no_adapter_returns_none(self):
        result = self.run(self.reg.run("my_tool", {}, make_ctx()))
        assert result is None

    def test_never_match_returns_none(self):
        self.reg.register(_NeverMatch())
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result is None

    def test_always_match_returns_result(self):
        self.reg.register(_AlwaysMatch(safe=True))
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result is not None
        assert result.safe is True
        assert result.adapter == "always"

    def test_unsafe_adapter_propagates_safe_false(self):
        self.reg.register(_AlwaysMatch(safe=False, signal="danger"))
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result.safe is False
        assert "danger" in result.signals

    def test_first_match_wins(self):
        """With two matching adapters, only the first runs."""
        a = _MatchByName("my_tool")
        a.name = "first"
        b = _MatchByName("my_tool")
        b.name = "second"
        self.reg.register(a)
        self.reg.register(b)
        result = self.run(self.reg.run("my_tool", {}, make_ctx()))
        assert result.adapter == "first"

    def test_skip_when_can_handle_raises(self):
        """can_handle() raising skips the adapter; subsequent adapters tried."""
        self.reg.register(_CanHandleRaises())
        self.reg.register(_AlwaysMatch(safe=True))
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result is not None
        assert result.adapter == "always"

    def test_dry_run_raises_returns_safe_error(self):
        """dry_run() raising returns safe=True error result (fail-open)."""
        self.reg.register(_DryRunRaises())
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result is not None
        assert result.safe is True
        assert result.error is not None
        assert "error" in result.signals

    def test_timeout_returns_safe_result(self):
        """A slow adapter is killed after timeout_ms; result is safe=True."""
        self.reg.register(_Slow())
        ctx = make_ctx(timeout_ms=50)   # 50 ms timeout
        result = self.run(self.reg.run("tool", {}, ctx))
        assert result is not None
        assert result.safe is True
        assert "timeout" in result.signals
        assert result.error is not None

    def test_clear_empties_registry(self):
        self.reg.register(_AlwaysMatch())
        self.reg._clear()
        assert self.reg.registered_names() == []
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result is None

    def test_registry_order_preserved(self):
        """Adapters checked in registration order."""
        for i in range(5):
            a = _NeverMatch()
            a.name = f"never_{i}"
            self.reg.register(a)
        winner = _AlwaysMatch()
        winner.name = "winner"
        self.reg.register(winner)
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        assert result.adapter == "winner"

    def test_to_dict_after_error(self):
        self.reg.register(_DryRunRaises())
        result = self.run(self.reg.run("tool", {}, make_ctx()))
        d = result.to_dict()
        assert "adapter" in d
        assert "error" in d
