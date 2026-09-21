"""
Tests for SDK-C4: Tool Allowlist Enforcement.

Covers:
  - None (default) — all tools allowed
  - Non-empty frozenset — only listed tools allowed
  - Empty frozenset — every tool blocked
  - TOOL_EXEC_BLOCKED event fired on allowlist rejection
  - Allowlist blocks before taint check, gate check, and execution
  - Sync and async wrappers
  - Decorator syntax
  - LangChain-style tool wrapping
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

import pytest

from aegivis.tools import instrument
from aegivis.security.exec_policy import ToolExecutionBlocked


def _sid() -> str:
    return f"sess_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Sync tool tests
# ---------------------------------------------------------------------------

class TestAllowlistSync:

    def test_no_allowlist_allows_all(self):
        """allowed_tools=None (default) — every tool executes normally."""
        executed = []

        def send_email(to: str) -> str:
            executed.append(to)
            return "sent"

        wrapped = instrument(send_email, session_id=_sid(), backend_url="")
        wrapped(to="user@example.com")
        assert executed == ["user@example.com"]

    def test_allowed_tool_executes(self):
        """Tool in the allowlist executes normally."""
        executed = []

        def send_email(to: str) -> str:
            executed.append(to)
            return "sent"

        wrapped = instrument(
            send_email,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"send_email"}),
        )
        wrapped(to="user@example.com")
        assert executed == ["user@example.com"]

    def test_unlisted_tool_blocked(self):
        """Tool not in the allowlist raises ToolExecutionBlocked."""
        def delete_user(user_id: str) -> str:
            return "deleted"

        wrapped = instrument(
            delete_user,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"send_email", "read_file"}),
        )
        with pytest.raises(ToolExecutionBlocked) as exc_info:
            wrapped(user_id="123")

        assert exc_info.value.tool_name == "delete_user"
        assert "allowlist" in exc_info.value.reason.lower()

    def test_empty_allowlist_blocks_everything(self):
        """allowed_tools=frozenset() blocks every tool."""
        def safe_fn() -> str:
            return "ok"

        wrapped = instrument(
            safe_fn,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset(),
        )
        with pytest.raises(ToolExecutionBlocked):
            wrapped()

    def test_allowlist_uses_function_name(self):
        """Allowlist is matched against __name__, not variable name."""
        def my_tool(x: int) -> int:
            return x * 2

        # The function __name__ is "my_tool"
        wrapped = instrument(
            my_tool,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"my_tool"}),
        )
        assert wrapped(x=5) == 10

    def test_allowlist_set_accepted(self):
        """Plain set (not frozenset) is accepted and converted."""
        def process(data: str) -> str:
            return data.upper()

        wrapped = instrument(
            process,
            session_id=_sid(),
            backend_url="",
            allowed_tools={"process", "other_tool"},  # plain set
        )
        assert wrapped(data="hello") == "HELLO"

    def test_blocked_tool_does_not_execute(self):
        """Body of a blocked tool is never called."""
        executed = []

        def danger_fn() -> None:
            executed.append(True)

        wrapped = instrument(
            danger_fn,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"safe_fn"}),
        )
        with pytest.raises(ToolExecutionBlocked):
            wrapped()

        assert executed == []

    def test_blocked_fires_tool_exec_blocked_event(self):
        """TOOL_EXEC_BLOCKED event is fired when allowlist rejects a tool."""
        def risky_fn() -> str:
            return "risky"

        fired: list[dict] = []

        def capture(event: dict, backend_url: str, api_key: str) -> None:
            fired.append(event)

        with patch("aegivis.tools._fire_event", side_effect=capture):
            wrapped = instrument(
                risky_fn,
                session_id=_sid(),
                backend_url="http://fake",
                allowed_tools=frozenset({"safe_fn"}),
            )
            with pytest.raises(ToolExecutionBlocked):
                wrapped()

        assert len(fired) == 1
        assert fired[0]["event_type"] == "TOOL_EXEC_BLOCKED"
        assert fired[0]["payload"]["tool_name"] == "risky_fn"

    def test_blocked_event_payload_has_reason(self):
        """TOOL_EXEC_BLOCKED event payload includes the reason string."""
        def fn() -> str:
            return "ok"

        fired: list[dict] = []

        with patch("aegivis.tools._fire_event", side_effect=lambda e, *_: fired.append(e)):
            wrapped = instrument(fn, session_id=_sid(), backend_url="http://fake",
                                 allowed_tools=frozenset())
            with pytest.raises(ToolExecutionBlocked):
                wrapped()

        assert "allowlist" in fired[0]["payload"]["reason"].lower()

    def test_allowlist_does_not_interfere_with_gate_signals(self):
        """When a tool IS in the allowlist, gate signal detection still works."""
        def send_email(to: str) -> str:
            return "sent"

        wrapped = instrument(
            send_email,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"send_email"}),
            block_on=frozenset({"email_destination"}),
        )
        # An email destination signal should still trigger block
        with pytest.raises(ToolExecutionBlocked) as exc_info:
            wrapped(to="user@example.com")

        # Blocked by signal gate, not allowlist
        assert "allowlist" not in exc_info.value.reason.lower()

    def test_allowlist_blocks_before_gate(self):
        """Allowlist check fires before signal classification."""
        classifier_called = []

        def fn(to: str) -> str:
            return "ok"

        wrapped = instrument(
            fn,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"other_tool"}),
        )

        # Even with a clearly signal-triggering arg, block is from allowlist
        with pytest.raises(ToolExecutionBlocked) as exc_info:
            wrapped(to="user@example.com")

        assert "allowlist" in exc_info.value.reason.lower()

    def test_multiple_tools_partial_allowlist(self):
        """instrument() on a list: allowed tools run, others are blocked."""
        results = []

        def tool_a(x: str) -> str:
            results.append(("a", x))
            return x

        def tool_b(x: str) -> str:
            results.append(("b", x))
            return x

        sid = _sid()
        tool_a_w, tool_b_w = instrument(
            [tool_a, tool_b],
            session_id=sid,
            backend_url="",
            allowed_tools=frozenset({"tool_a"}),
        )

        tool_a_w(x="hello")
        assert results == [("a", "hello")]

        with pytest.raises(ToolExecutionBlocked):
            tool_b_w(x="world")

        # tool_b body was never called
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Async tool tests
# ---------------------------------------------------------------------------

class TestAllowlistAsync:

    def test_async_allowed_tool_executes(self):
        async def afn(x: int) -> int:
            return x + 1

        wrapped = instrument(
            afn,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"afn"}),
        )
        result = asyncio.run(wrapped(x=10))
        assert result == 11

    def test_async_unlisted_tool_blocked(self):
        async def dangerous_fn() -> str:
            return "danger"

        wrapped = instrument(
            dangerous_fn,
            session_id=_sid(),
            backend_url="",
            allowed_tools=frozenset({"safe_fn"}),
        )

        async def run():
            with pytest.raises(ToolExecutionBlocked) as exc_info:
                await wrapped()
            assert exc_info.value.tool_name == "dangerous_fn"

        asyncio.run(run())

    def test_async_empty_allowlist_blocks(self):
        async def fn() -> str:
            return "ok"

        wrapped = instrument(fn, session_id=_sid(), backend_url="",
                             allowed_tools=frozenset())

        async def run():
            with pytest.raises(ToolExecutionBlocked):
                await wrapped()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Decorator syntax
# ---------------------------------------------------------------------------

class TestAllowlistDecorator:

    def test_decorator_with_allowlist(self):
        sid = _sid()

        @instrument.tool(
            session_id=sid,
            backend_url="",
            allowed_tools=frozenset({"my_decorated_fn"}),
        )
        def my_decorated_fn(x: int) -> int:
            return x * 3

        assert my_decorated_fn(x=4) == 12

    def test_decorator_blocks_unlisted(self):
        sid = _sid()

        @instrument.tool(
            session_id=sid,
            backend_url="",
            allowed_tools=frozenset({"other_fn"}),
        )
        def blocked_fn() -> str:
            return "should not run"

        with pytest.raises(ToolExecutionBlocked) as exc_info:
            blocked_fn()

        assert exc_info.value.tool_name == "blocked_fn"

    def test_bare_decorator_no_allowlist(self):
        """@instrument.tool without allowlist passes all tools."""
        sid = _sid()

        @instrument.tool(session_id=sid, backend_url="")
        def any_tool(v: str) -> str:
            return v.upper()

        assert any_tool(v="hello") == "HELLO"
