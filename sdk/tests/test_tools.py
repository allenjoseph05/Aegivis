"""
Tests for sdk/aegivis/tools.py — General Tool Instrumentation SDK.

All tests mock _fire_event so no network calls are made.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, call, patch

import pytest

from aegivis.tools import (
    _InstrumentConfig,
    _fire_event,
    _safe_preview,
    _tool_name,
    _wrap_one,
    _wrap_sync,
    _wrap_async,
    instrument,
)
from aegivis import instrument as instrument_top_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs) -> _InstrumentConfig:
    return _InstrumentConfig(
        agent_id="test-agent",
        session_id="test-session",
        backend_url="http://fake-backend",
        api_key="test-key",
        org_id="test-org",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _safe_preview
# ---------------------------------------------------------------------------

def test_safe_preview_short_string():
    assert _safe_preview("hello") == '"hello"'


def test_safe_preview_truncates():
    long = "x" * 400
    preview = _safe_preview(long)
    assert len(preview) <= 305  # 300 chars + ellipsis


def test_safe_preview_handles_non_serialisable():
    class Unser:
        def __repr__(self): return "Unser()"
    result = _safe_preview(Unser())
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _tool_name
# ---------------------------------------------------------------------------

def test_tool_name_from_name_attr():
    obj = MagicMock()
    obj.name = "my_tool"
    assert _tool_name(obj) == "my_tool"


def test_tool_name_from_dunder_name():
    def my_function(): pass
    assert _tool_name(my_function) == "my_function"


def test_tool_name_fallback():
    assert _tool_name(42) == "int"


# ---------------------------------------------------------------------------
# Sync callable wrapping
# ---------------------------------------------------------------------------

def test_sync_callable_returns_result():
    fn = MagicMock(return_value="done")
    with patch("aegivis.tools._fire_event"):
        wrapped = _wrap_sync(fn, "test_tool", _cfg())
        result = wrapped("arg1", key="val")
    assert result == "done"
    fn.assert_called_once_with("arg1", key="val")


def test_sync_callable_fires_start_and_end():
    fn = MagicMock(return_value="ok")
    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_sync(fn, "my_tool", _cfg())
        wrapped("x")
    assert mock_fire.call_count == 2
    start_event = mock_fire.call_args_list[0][0][0]
    end_event = mock_fire.call_args_list[1][0][0]
    assert start_event["event_type"] == "TOOL_EXEC_START"
    assert end_event["event_type"] == "TOOL_EXEC_END"
    assert start_event["payload"]["tool_name"] == "my_tool"
    assert "duration_ms" in end_event["payload"]


def test_sync_callable_fires_error_on_exception():
    def failing(*a, **kw):
        raise ValueError("boom")

    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_sync(failing, "bad_tool", _cfg())
        with pytest.raises(ValueError, match="boom"):
            wrapped()

    assert mock_fire.call_count == 2
    start_event = mock_fire.call_args_list[0][0][0]
    err_event = mock_fire.call_args_list[1][0][0]
    assert start_event["event_type"] == "TOOL_EXEC_START"
    assert err_event["event_type"] == "TOOL_EXEC_ERROR"
    assert err_event["payload"]["error_type"] == "ValueError"
    assert err_event["payload"]["error_message"] == "boom"


def test_sync_callable_preserves_name():
    def my_named_fn(): pass
    with patch("aegivis.tools._fire_event"):
        wrapped = _wrap_sync(my_named_fn, "my_named_fn", _cfg())
    assert wrapped.__name__ == "my_named_fn"


# ---------------------------------------------------------------------------
# Async callable wrapping
# ---------------------------------------------------------------------------

def test_async_callable_returns_result():
    async def async_fn(x):
        return x * 2

    with patch("aegivis.tools._fire_event"):
        wrapped = _wrap_async(async_fn, "async_tool", _cfg())
        result = asyncio.run(wrapped(5))
    assert result == 10


def test_async_callable_fires_start_and_end():
    async def async_fn():
        return "async_result"

    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_async(async_fn, "async_tool", _cfg())
        asyncio.run(wrapped())

    assert mock_fire.call_count == 2
    assert mock_fire.call_args_list[0][0][0]["event_type"] == "TOOL_EXEC_START"
    assert mock_fire.call_args_list[1][0][0]["event_type"] == "TOOL_EXEC_END"


def test_async_callable_fires_error():
    async def async_failing():
        raise RuntimeError("async boom")

    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_async(async_failing, "fail_tool", _cfg())
        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(wrapped())

    err_event = mock_fire.call_args_list[1][0][0]
    assert err_event["event_type"] == "TOOL_EXEC_ERROR"
    assert err_event["payload"]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# LangChain BaseTool detection
# ---------------------------------------------------------------------------

class FakeLangChainTool:
    """Minimal LangChain BaseTool-like object."""

    name: str = "fake_lc_tool"

    def run(self, *args, **kwargs):
        return "lc_result"

    async def arun(self, *args, **kwargs):
        return "lc_async_result"


def test_langchain_tool_run_wrapped():
    tool = FakeLangChainTool()
    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_one(tool, _cfg())
        result = wrapped.run("input")
    assert result == "lc_result"
    assert mock_fire.call_count == 2
    assert mock_fire.call_args_list[0][0][0]["payload"]["tool_name"] == "fake_lc_tool"


def test_langchain_tool_arun_wrapped():
    tool = FakeLangChainTool()
    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_one(tool, _cfg())
        result = asyncio.run(wrapped.arun("input"))
    assert result == "lc_async_result"
    assert mock_fire.call_count == 2


# ---------------------------------------------------------------------------
# AutoGen FunctionTool detection
# ---------------------------------------------------------------------------

class FakeAutoGenTool:
    """Minimal AutoGen FunctionTool-like object."""

    name = "autogen_search"

    def __init__(self):
        self._func = self._real_func

    def _real_func(self, query: str) -> str:
        return f"result for {query}"


def test_autogen_tool_func_wrapped():
    tool = FakeAutoGenTool()
    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_one(tool, _cfg())
        result = wrapped._func("hello")
    assert result == "result for hello"
    assert mock_fire.call_count == 2


# ---------------------------------------------------------------------------
# Plain async callable
# ---------------------------------------------------------------------------

def test_plain_async_callable_wrapped():
    async def fetch(url: str) -> str:
        return f"content of {url}"

    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = _wrap_one(fetch, _cfg())
        result = asyncio.run(wrapped("http://example.com"))
    assert result == "content of http://example.com"
    assert mock_fire.call_args_list[0][0][0]["event_type"] == "TOOL_EXEC_START"


# ---------------------------------------------------------------------------
# instrument() API
# ---------------------------------------------------------------------------

def test_instrument_single_callable():
    def fn(): return "single"
    with patch("aegivis.tools._fire_event"):
        wrapped = instrument(fn, agent_id="a", backend_url="http://b")
        result = wrapped()
    assert result == "single"


def test_instrument_list_of_callables():
    def fn1(): return "one"
    def fn2(): return "two"
    with patch("aegivis.tools._fire_event"):
        wrapped = instrument([fn1, fn2], agent_id="a", backend_url="http://b")
    assert len(wrapped) == 2
    assert wrapped[0]() == "one"
    assert wrapped[1]() == "two"


def test_instrument_list_of_langchain_tools():
    t1 = FakeLangChainTool()
    t1.name = "tool_1"
    t2 = FakeLangChainTool()
    t2.name = "tool_2"
    with patch("aegivis.tools._fire_event"):
        wrapped = instrument([t1, t2], backend_url="http://b")
    assert len(wrapped) == 2


def test_instrument_event_payload_shape():
    # Use a plain function — MagicMock has .name + .run so it looks like a LangChain tool
    def plain_fn(*args, **kwargs):
        return 42

    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = instrument(plain_fn, agent_id="agt", session_id="sess", org_id="org", backend_url="http://b")
        wrapped("hello", key="world")
        # Each call: _fire_event(event_dict, backend_url, api_key)
        assert mock_fire.call_count == 2
        start = mock_fire.call_args_list[0][0][0]
        assert start["agent_id"] == "agt"
        assert start["session_id"] == "sess"
        assert start["org_id"] == "org"
        assert start["payload"]["source"] == "sdk_instrument"
        assert "args_preview" in start["payload"]
        end = mock_fire.call_args_list[1][0][0]
        assert end["payload"]["return_preview"] == "42"
        assert end["payload"]["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# @instrument.tool decorator
# ---------------------------------------------------------------------------

def test_decorator_no_parens():
    @instrument.tool
    def my_func(x):
        return x + 1

    with patch("aegivis.tools._fire_event") as mock_fire:
        result = my_func(10)
    assert result == 11
    assert mock_fire.call_count == 2


def test_decorator_with_parens():
    @instrument.tool(agent_id="deco-agent", backend_url="http://fake")
    def another_func(x):
        return x * 3

    with patch("aegivis.tools._fire_event") as mock_fire:
        result = another_func(5)
    assert result == 15
    assert mock_fire.call_count == 2
    start = mock_fire.call_args_list[0][0][0]
    assert start["agent_id"] == "deco-agent"


def test_decorator_async_no_parens():
    @instrument.tool
    async def async_fn(x):
        return x ** 2

    with patch("aegivis.tools._fire_event"):
        result = asyncio.run(async_fn(4))
    assert result == 16


# ---------------------------------------------------------------------------
# Top-level import from aegivis
# ---------------------------------------------------------------------------

def test_top_level_instrument_import():
    def fn(): return "imported"
    with patch("aegivis.tools._fire_event"):
        wrapped = instrument_top_level(fn, backend_url="http://b")
        result = wrapped()
    assert result == "imported"


# ---------------------------------------------------------------------------
# _fire_event — no network call when backend_url is empty
# ---------------------------------------------------------------------------

def test_fire_event_noop_when_no_url():
    # Should not start any thread and should not raise
    import threading
    before = threading.active_count()
    _fire_event({"event_type": "TOOL_EXEC_START"}, "", "key")
    # No new threads should have been spawned
    assert threading.active_count() <= before + 1


# ---------------------------------------------------------------------------
# Error event contains error_message truncation
# ---------------------------------------------------------------------------

def test_error_message_truncated():
    long_error = "x" * 500

    def failing():
        raise RuntimeError(long_error)

    with patch("aegivis.tools._fire_event") as mock_fire:
        wrapped = instrument(failing, backend_url="http://b")
        with pytest.raises(RuntimeError):
            wrapped()
        err_event = mock_fire.call_args_list[1][0][0]
        assert len(err_event["payload"]["error_message"]) <= 300
