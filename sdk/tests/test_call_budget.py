"""
Tests for SDK-C5: Per-session call budgets.

Covers:
  - max_calls_per_session: total calls across all tools
  - max_calls_per_tool: per-tool call limit
  - Both limits simultaneously
  - ToolBudgetExceeded exception attributes
  - TOOL_EXEC_BLOCKED event fired on budget exhaustion
  - Blocked calls do NOT increment the counter
  - Different sessions are independent
  - Async wrappers
  - Decorator syntax
  - Interaction with allowlist and gate (budget checked second)
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from unittest.mock import patch

import pytest

from aegivis.tools import instrument, _session_call_counts, _tool_call_counts, _call_counts_lock
from aegivis.security.exec_policy import ToolBudgetExceeded, ToolExecutionBlocked


def _sid() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def _make_tool(name_suffix: str = ""):
    """Return a simple sync tool function."""
    executed = []

    def tool_fn(x: str = "ok") -> str:
        executed.append(x)
        return x

    tool_fn.__name__ = f"tool_{name_suffix}" if name_suffix else "tool_fn"
    return tool_fn, executed


# ---------------------------------------------------------------------------
# ToolBudgetExceeded exception
# ---------------------------------------------------------------------------

class TestToolBudgetExceededExc:

    def test_session_budget_attributes(self):
        exc = ToolBudgetExceeded("send_email", "session", 5, 5)
        assert exc.tool_name == "send_email"
        assert exc.budget_type == "session"
        assert exc.limit == 5
        assert exc.actual == 5

    def test_tool_budget_attributes(self):
        exc = ToolBudgetExceeded("delete_user", "tool", 3, 3)
        assert exc.tool_name == "delete_user"
        assert exc.budget_type == "tool"
        assert exc.limit == 3
        assert exc.actual == 3

    def test_message_contains_key_info(self):
        exc = ToolBudgetExceeded("fn", "session", 10, 10)
        msg = str(exc)
        assert "fn" in msg
        assert "session" in msg
        assert "10" in msg

    def test_is_runtime_error(self):
        assert isinstance(ToolBudgetExceeded("fn", "tool", 1, 1), RuntimeError)


# ---------------------------------------------------------------------------
# Session-level budget (max_calls_per_session)
# ---------------------------------------------------------------------------

class TestSessionBudget:

    def test_calls_within_budget_succeed(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_session=3)

        for _ in range(3):
            wrapped(x="ok")

        assert len(executed) == 3

    def test_call_exceeding_budget_blocked(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_session=2)

        wrapped(x="1")
        wrapped(x="2")

        with pytest.raises(ToolBudgetExceeded) as exc_info:
            wrapped(x="3")

        assert exc_info.value.budget_type == "session"
        assert exc_info.value.limit == 2
        assert exc_info.value.actual == 2

    def test_blocked_call_does_not_execute(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_session=1)

        wrapped(x="first")
        with pytest.raises(ToolBudgetExceeded):
            wrapped(x="second")

        assert executed == ["first"]

    def test_blocked_call_does_not_increment_counter(self):
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_session=1)

        wrapped()
        with pytest.raises(ToolBudgetExceeded):
            wrapped()
        # Second block attempt: counter is still 1, not 2
        with pytest.raises(ToolBudgetExceeded) as exc_info:
            wrapped()
        assert exc_info.value.actual == 1

    def test_session_budget_shared_across_different_tools(self):
        """Session budget counts calls from ALL tools, not per-tool."""
        sid = _sid()

        def tool_a() -> str:
            return "a"

        def tool_b() -> str:
            return "b"

        a = instrument(tool_a, session_id=sid, backend_url="", max_calls_per_session=2)
        b = instrument(tool_b, session_id=sid, backend_url="", max_calls_per_session=2)

        a()  # session count → 1
        b()  # session count → 2

        with pytest.raises(ToolBudgetExceeded) as exc_info:
            a()  # session count would be 3 > 2

        assert exc_info.value.budget_type == "session"

    def test_different_sessions_independent(self):
        sid_a = _sid()
        sid_b = _sid()
        fn, _ = _make_tool()

        a = instrument(fn, session_id=sid_a, backend_url="", max_calls_per_session=1)
        b = instrument(fn, session_id=sid_b, backend_url="", max_calls_per_session=1)

        a()   # session A: count → 1
        b()   # session B: count → 1 (independent)

        with pytest.raises(ToolBudgetExceeded):
            a()  # session A exhausted

        # session B still has its own budget
        with pytest.raises(ToolBudgetExceeded):
            b()  # session B exhausted too, but independently

    def test_no_session_budget_allows_unlimited(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="")

        for _ in range(50):
            wrapped()

        assert len(executed) == 50

    def test_session_budget_one_allows_exactly_one(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_session=1)

        wrapped()
        with pytest.raises(ToolBudgetExceeded):
            wrapped()
        assert len(executed) == 1

    def test_budget_blocked_fires_tool_exec_blocked_event(self):
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="http://fake",
                             max_calls_per_session=1)

        fired: list[dict] = []
        with patch("aegivis.tools._fire_event", side_effect=lambda e, *_: fired.append(e)):
            wrapped()  # allowed
            with pytest.raises(ToolBudgetExceeded):
                wrapped()  # blocked

        blocked_events = [e for e in fired if e["event_type"] == "TOOL_EXEC_BLOCKED"]
        assert len(blocked_events) == 1
        payload = blocked_events[0]["payload"]
        assert payload["budget_type"] == "session"
        assert payload["limit"] == 1

    def test_budget_event_payload_has_required_fields(self):
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="http://fake",
                             max_calls_per_session=1)

        fired: list[dict] = []
        with patch("aegivis.tools._fire_event", side_effect=lambda e, *_: fired.append(e)):
            wrapped()
            with pytest.raises(ToolBudgetExceeded):
                wrapped()

        blocked = [e for e in fired if e["event_type"] == "TOOL_EXEC_BLOCKED"][0]
        assert "budget_type" in blocked["payload"]
        assert "limit" in blocked["payload"]
        assert "actual" in blocked["payload"]
        assert "reason" in blocked["payload"]


# ---------------------------------------------------------------------------
# Per-tool budget (max_calls_per_tool)
# ---------------------------------------------------------------------------

class TestToolBudget:

    def test_tool_within_budget_succeeds(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_tool=3)

        for _ in range(3):
            wrapped()

        assert len(executed) == 3

    def test_tool_exceeding_budget_blocked(self):
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_tool=2)

        wrapped()
        wrapped()

        with pytest.raises(ToolBudgetExceeded) as exc_info:
            wrapped()

        assert exc_info.value.budget_type == "tool"
        assert exc_info.value.limit == 2

    def test_tool_budget_per_tool_name_independent(self):
        """Different tool names have independent per-tool counters."""
        sid = _sid()

        def sender() -> str:
            return "sent"

        def reader() -> str:
            return "read"

        s = instrument(sender, session_id=sid, backend_url="", max_calls_per_tool=1)
        r = instrument(reader, session_id=sid, backend_url="", max_calls_per_tool=1)

        s()  # sender count → 1
        r()  # reader count → 1

        with pytest.raises(ToolBudgetExceeded) as exc_info:
            s()  # sender exhausted
        assert exc_info.value.budget_type == "tool"

        with pytest.raises(ToolBudgetExceeded):
            r()  # reader also exhausted (independently)

    def test_tool_budget_blocked_does_not_execute(self):
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_tool=1)

        wrapped()
        with pytest.raises(ToolBudgetExceeded):
            wrapped()

        assert len(executed) == 1

    def test_tool_budget_blocked_fires_event(self):
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="http://fake",
                             max_calls_per_tool=1)

        fired: list[dict] = []
        with patch("aegivis.tools._fire_event", side_effect=lambda e, *_: fired.append(e)):
            wrapped()
            with pytest.raises(ToolBudgetExceeded):
                wrapped()

        blocked = [e for e in fired if e["event_type"] == "TOOL_EXEC_BLOCKED"]
        assert len(blocked) == 1
        assert blocked[0]["payload"]["budget_type"] == "tool"


# ---------------------------------------------------------------------------
# Both limits simultaneously
# ---------------------------------------------------------------------------

class TestBothBudgets:

    def test_session_limit_hit_first(self):
        """When session limit < tool limit, session triggers first."""
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="",
                             max_calls_per_session=1, max_calls_per_tool=5)

        wrapped()
        with pytest.raises(ToolBudgetExceeded) as exc_info:
            wrapped()
        assert exc_info.value.budget_type == "session"

    def test_tool_limit_hit_first(self):
        """When tool limit < session limit, tool budget triggers first."""
        sid = _sid()
        fn, _ = _make_tool("x")
        wrapped = instrument(fn, session_id=sid, backend_url="",
                             max_calls_per_session=10, max_calls_per_tool=1)
        wrapped()
        with pytest.raises(ToolBudgetExceeded) as exc_info:
            wrapped()
        assert exc_info.value.budget_type == "tool"


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

class TestBudgetAsync:

    def test_async_within_budget(self):
        sid = _sid()

        async def afn() -> str:
            return "ok"

        wrapped = instrument(afn, session_id=sid, backend_url="", max_calls_per_session=2)

        async def run():
            await wrapped()
            await wrapped()

        asyncio.run(run())

    def test_async_exceeds_budget(self):
        sid = _sid()

        async def afn() -> str:
            return "ok"

        wrapped = instrument(afn, session_id=sid, backend_url="", max_calls_per_session=1)

        async def run():
            await wrapped()
            with pytest.raises(ToolBudgetExceeded):
                await wrapped()

        asyncio.run(run())

    def test_async_per_tool_budget(self):
        sid = _sid()

        async def afn() -> str:
            return "ok"

        wrapped = instrument(afn, session_id=sid, backend_url="", max_calls_per_tool=2)

        async def run():
            await wrapped()
            await wrapped()
            with pytest.raises(ToolBudgetExceeded) as exc_info:
                await wrapped()
            assert exc_info.value.budget_type == "tool"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Decorator syntax
# ---------------------------------------------------------------------------

class TestBudgetDecorator:

    def test_decorator_session_budget(self):
        sid = _sid()

        @instrument.tool(session_id=sid, backend_url="", max_calls_per_session=2)
        def my_fn() -> str:
            return "ok"

        my_fn()
        my_fn()
        with pytest.raises(ToolBudgetExceeded):
            my_fn()

    def test_decorator_tool_budget(self):
        sid = _sid()

        @instrument.tool(session_id=sid, backend_url="", max_calls_per_tool=1)
        def my_fn() -> str:
            return "ok"

        my_fn()
        with pytest.raises(ToolBudgetExceeded) as exc_info:
            my_fn()
        assert exc_info.value.budget_type == "tool"


# ---------------------------------------------------------------------------
# Interaction with other gate features
# ---------------------------------------------------------------------------

class TestBudgetInteraction:

    def test_budget_checked_after_allowlist(self):
        """Allowlist fires before budget — blocked by allowlist, counter not incremented."""
        sid = _sid()
        fn, _ = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="",
                             allowed_tools=frozenset({"other_tool"}),
                             max_calls_per_session=5)

        # Blocked by allowlist — budget counter should NOT increment
        for _ in range(3):
            with pytest.raises(ToolExecutionBlocked):
                wrapped()

        # Remove allowlist restriction — budget should still have full capacity
        fn2, executed = _make_tool()
        fn2.__name__ = fn.__name__
        wrapped2 = instrument(fn2, session_id=sid, backend_url="",
                              max_calls_per_session=5)
        # All 5 calls should succeed (allowlist blocks didn't eat the budget)
        for _ in range(5):
            wrapped2()
        assert len(executed) == 5

    def test_no_budget_no_limit(self):
        """Without budget params, no limits apply."""
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="")

        for _ in range(100):
            wrapped()

        assert len(executed) == 100

    def test_thread_safe_concurrent_budget(self):
        """Concurrent calls under the budget limit all succeed; excess are blocked."""
        sid = _sid()
        fn, executed = _make_tool()
        wrapped = instrument(fn, session_id=sid, backend_url="", max_calls_per_session=10)

        errors: list[Exception] = []
        budget_errors: list[ToolBudgetExceeded] = []

        def call_many():
            for _ in range(5):
                try:
                    wrapped()
                except ToolBudgetExceeded as e:
                    budget_errors.append(e)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=call_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Exactly 10 calls succeeded, 10 were blocked
        assert len(executed) == 10
        assert len(budget_errors) == 10
