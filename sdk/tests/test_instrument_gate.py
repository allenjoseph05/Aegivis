"""
End-to-end tests for the SDK Execution Gate integrated into instrument().

Covers: blocking, alerting, timeout (sync + async), return scan,
backward compatibility, decorator syntax, and fail-open behaviour.

All tests set backend_url="" so no real network calls are made.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from aegivis.security.arg_classifier import Signal
from aegivis.security.exec_policy import (
    ToolConcurrencyLimitExceeded,
    ToolExecutionBlocked,
    ToolExecutionTimeout,
)
from aegivis.tools import instrument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_BACKEND = ""  # disables all HTTP traffic for every test


def make_gate_tool(fn, *, block_on=None, alert_on=None, timeout_s=None, scan_return=True):
    """Convenience wrapper: instrument fn with gate enabled but no backend."""
    return instrument(
        fn,
        backend_url=_NO_BACKEND,
        block_on=block_on,
        alert_on=alert_on,
        timeout_s=timeout_s,
        scan_return=scan_return,
    )


# ---------------------------------------------------------------------------
# Backward compatibility — zero config behaves identically to before
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_sync_tool_runs_unchanged(self):
        def add(a, b):
            return a + b

        wrapped = instrument(add, backend_url=_NO_BACKEND)
        assert wrapped(2, 3) == 5

    def test_return_value_preserved(self):
        def greet(name):
            return f"hello, {name}"

        wrapped = instrument(greet, backend_url=_NO_BACKEND)
        assert wrapped("world") == "hello, world"

    @pytest.mark.asyncio
    async def test_async_tool_runs_unchanged(self):
        async def fetch(url):
            return {"status": 200, "url": url}

        wrapped = instrument(fetch, backend_url=_NO_BACKEND)
        result = await wrapped("https://example.com")
        assert result["status"] == 200

    def test_list_of_tools_instrumented(self):
        def tool_a():
            return "a"
        def tool_b():
            return "b"

        wrapped = instrument([tool_a, tool_b], backend_url=_NO_BACKEND)
        assert len(wrapped) == 2
        assert wrapped[0]() == "a"
        assert wrapped[1]() == "b"

    def test_exceptions_propagate(self):
        def boom():
            raise ValueError("intentional error")

        wrapped = instrument(boom, backend_url=_NO_BACKEND)
        with pytest.raises(ValueError, match="intentional error"):
            wrapped()

    def test_functools_wraps_preserves_name(self):
        def my_special_tool():
            pass

        wrapped = instrument(my_special_tool, backend_url=_NO_BACKEND)
        assert wrapped.__name__ == "my_special_tool"


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------


class TestBlocking:
    def test_credential_blocks_sync_tool(self):
        def send(token):
            return "sent"

        wrapped = make_gate_tool(send, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked) as exc_info:
            wrapped("sk-supersecretkey1234567890")

        assert exc_info.value.tool_name == "send"

    def test_email_blocks_sync_tool(self):
        def notify(to):
            return "ok"

        wrapped = make_gate_tool(notify, block_on=frozenset({Signal.EMAIL_DESTINATION}))
        with pytest.raises(ToolExecutionBlocked):
            notify_w = wrapped
            notify_w("attacker@evil.com")

    def test_blocked_tool_does_not_execute(self):
        executed = []

        def sensitive(token):
            executed.append(True)
            return "done"

        wrapped = make_gate_tool(sensitive, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped("sk-supersecretkey1234567890")

        assert executed == [], "Tool body must not run when blocked"

    def test_blocked_tool_reason_mentions_signal(self):
        def tool(token):
            return "x"

        wrapped = make_gate_tool(tool, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked) as exc_info:
            wrapped("sk-supersecretkey1234567890")

        assert "credential" in exc_info.value.reason.lower()

    def test_safe_args_not_blocked(self):
        def tool(name):
            return f"hello {name}"

        wrapped = make_gate_tool(tool, block_on=frozenset({Signal.CREDENTIAL}))
        result = wrapped("alice")
        assert result == "hello alice"

    @pytest.mark.asyncio
    async def test_credential_blocks_async_tool(self):
        async def async_send(token):
            return "sent"

        wrapped = make_gate_tool(async_send, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            await wrapped("sk-supersecretkey1234567890")

    @pytest.mark.asyncio
    async def test_async_blocked_tool_does_not_execute(self):
        executed = []

        async def async_tool(token):
            executed.append(True)
            return "done"

        wrapped = make_gate_tool(async_tool, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            await wrapped("sk-supersecretkey1234567890")

        assert executed == []

    def test_block_via_kwargs(self):
        def tool(destination):
            return "sent"

        wrapped = make_gate_tool(tool, block_on=frozenset({Signal.EMAIL_DESTINATION}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped(destination="evil@attacker.com")

    def test_nested_credential_in_dict_blocks(self):
        def tool(config):
            return "ok"

        wrapped = make_gate_tool(tool, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped({"auth": {"token": "sk-supersecretkey12345678901234"}})

    def test_block_priority_over_alert(self):
        """When a signal is in both block_on and alert_on, block must win."""
        def tool(token):
            return "x"

        wrapped = instrument(
            tool,
            backend_url=_NO_BACKEND,
            block_on=frozenset({Signal.CREDENTIAL}),
            alert_on=frozenset({Signal.CREDENTIAL}),
        )
        with pytest.raises(ToolExecutionBlocked):
            wrapped("sk-supersecretkey1234567890")


# ---------------------------------------------------------------------------
# Alert — execution continues
# ---------------------------------------------------------------------------


class TestAlert:
    def test_alert_allows_execution(self):
        results = []

        def send(to):
            results.append(to)
            return "ok"

        wrapped = make_gate_tool(send, alert_on=frozenset({Signal.EMAIL_DESTINATION}))
        result = wrapped("user@example.com")
        assert result == "ok"
        assert "user@example.com" in results

    @pytest.mark.asyncio
    async def test_async_alert_allows_execution(self):
        async def send(to):
            return f"sent to {to}"

        wrapped = make_gate_tool(send, alert_on=frozenset({Signal.EMAIL_DESTINATION}))
        result = await wrapped("user@example.com")
        assert result == "sent to user@example.com"

    def test_irrelevant_signal_does_not_alert_block(self):
        """alert_on=email should not affect a tool called with a file path."""
        def tool(path):
            return f"read {path}"

        wrapped = make_gate_tool(tool, alert_on=frozenset({Signal.EMAIL_DESTINATION}))
        result = wrapped("/etc/hosts")
        assert result == "read /etc/hosts"


# ---------------------------------------------------------------------------
# Timeout — sync
# ---------------------------------------------------------------------------


class TestSyncTimeout:
    def test_fast_tool_completes(self):
        def fast():
            return 42

        wrapped = make_gate_tool(fast, timeout_s=5.0)
        assert wrapped() == 42

    def test_slow_tool_raises_timeout(self):
        def slow():
            time.sleep(10)
            return "done"

        wrapped = make_gate_tool(slow, timeout_s=0.1)
        with pytest.raises(ToolExecutionTimeout) as exc_info:
            wrapped()

        assert exc_info.value.timeout_s == 0.1

    def test_timeout_raises_tool_execution_timeout(self):
        def slow():
            time.sleep(10)

        wrapped = make_gate_tool(slow, timeout_s=0.1)
        with pytest.raises(ToolExecutionTimeout):
            wrapped()

    def test_timeout_exception_has_tool_name(self):
        def very_slow_operation():
            time.sleep(10)

        wrapped = make_gate_tool(very_slow_operation, timeout_s=0.1)
        with pytest.raises(ToolExecutionTimeout) as exc_info:
            wrapped()

        assert exc_info.value.tool_name == "very_slow_operation"

    def test_no_timeout_by_default(self):
        def slow_but_fine():
            time.sleep(0.05)
            return "done"

        # No timeout_s set → should complete normally
        wrapped = instrument(slow_but_fine, backend_url=_NO_BACKEND)
        assert wrapped() == "done"


# ---------------------------------------------------------------------------
# Timeout — async
# ---------------------------------------------------------------------------


class TestAsyncTimeout:
    @pytest.mark.asyncio
    async def test_fast_async_completes(self):
        async def fast():
            return 99

        wrapped = make_gate_tool(fast, timeout_s=5.0)
        assert await wrapped() == 99

    @pytest.mark.asyncio
    async def test_slow_async_raises_timeout(self):
        async def slow():
            await asyncio.sleep(10)
            return "done"

        wrapped = make_gate_tool(slow, timeout_s=0.1)
        with pytest.raises(ToolExecutionTimeout) as exc_info:
            await wrapped()

        assert exc_info.value.timeout_s == 0.1

    @pytest.mark.asyncio
    async def test_async_timeout_tool_name_preserved(self):
        async def async_slow_operation():
            await asyncio.sleep(10)

        wrapped = make_gate_tool(async_slow_operation, timeout_s=0.1)
        with pytest.raises(ToolExecutionTimeout) as exc_info:
            await wrapped()

        assert exc_info.value.tool_name == "async_slow_operation"


# ---------------------------------------------------------------------------
# Return value scanning
# ---------------------------------------------------------------------------


class TestReturnScan:
    def test_return_scan_does_not_block_execution(self):
        """Return scan fires an event but must NOT raise or block the caller."""
        def fetch_config():
            # Returns a dict containing a high-entropy credential-shaped value
            return {"api_key": "sk-abcdefghijklmnopqrstuvwxyz01234"}

        wrapped = instrument(fetch_config, backend_url=_NO_BACKEND, scan_return=True)
        result = wrapped()
        # Execution must complete and return value must be intact
        assert "api_key" in result

    def test_scan_return_false_skips_scan(self):
        """When scan_return=False, no return scanning should occur."""
        def fetch():
            return {"token": "sk-abcdefghijklmnopqrstuvwxyz01234"}

        # scan_return=False → should complete silently with no event
        wrapped = instrument(fetch, backend_url=_NO_BACKEND, scan_return=False)
        result = wrapped()
        assert "token" in result

    @pytest.mark.asyncio
    async def test_async_return_scan_does_not_block(self):
        async def fetch():
            return {"secret": "sk-abcdefghijklmnopqrstuvwxyz01234"}

        wrapped = instrument(fetch, backend_url=_NO_BACKEND, scan_return=True)
        result = await wrapped()
        assert "secret" in result


# ---------------------------------------------------------------------------
# Decorator syntax
# ---------------------------------------------------------------------------


class TestDecoratorSyntax:
    def test_bare_decorator_no_args(self):
        from aegivis.tools import instrument

        @instrument.tool
        def my_tool(x):
            return x * 2

        # instrument.tool with no backend should be a noop
        assert my_tool(5) == 10

    def test_decorator_with_block_on(self):
        from aegivis.tools import instrument

        @instrument.tool(backend_url=_NO_BACKEND, block_on=frozenset({Signal.CREDENTIAL}))
        def secured(token):
            return "ok"

        with pytest.raises(ToolExecutionBlocked):
            secured("sk-supersecretkey1234567890")

    def test_decorator_with_timeout(self):
        from aegivis.tools import instrument

        @instrument.tool(backend_url=_NO_BACKEND, timeout_s=0.1)
        def slow_tool():
            time.sleep(10)

        with pytest.raises(ToolExecutionTimeout):
            slow_tool()

    def test_decorator_with_alert_on_allows_execution(self):
        from aegivis.tools import instrument

        @instrument.tool(backend_url=_NO_BACKEND, alert_on=frozenset({Signal.EMAIL_DESTINATION}))
        def send(to):
            return f"sent to {to}"

        result = send("user@example.com")
        assert result == "sent to user@example.com"

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        from aegivis.tools import instrument

        @instrument.tool(backend_url=_NO_BACKEND, block_on=frozenset({Signal.CREDENTIAL}))
        async def secure_async(token):
            return "async ok"

        with pytest.raises(ToolExecutionBlocked):
            await secure_async("sk-supersecretkey1234567890")


# ---------------------------------------------------------------------------
# Fail-open on gate errors
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_gate_error_fails_open(self):
        """If the classifier raises unexpectedly, the tool should still run."""
        from aegivis.security.arg_classifier import ArgClassifier

        class BrokenClassifier(ArgClassifier):
            def classify_call(self, args, kwargs):
                raise RuntimeError("classifier on fire")

        from aegivis.security.exec_policy import GateConfig
        from aegivis.tools import _InstrumentConfig, _wrap_sync

        cfg = _InstrumentConfig(backend_url="")
        cfg.classifier = BrokenClassifier()
        cfg.gate = GateConfig(block_on=frozenset({Signal.CREDENTIAL}))

        def tool(x):
            return f"result:{x}"

        wrapped = _wrap_sync(tool, "tool", cfg)
        # Should run despite classifier error (fail-open)
        result = wrapped("hello")
        assert result == "result:hello"

    def test_return_scan_error_fails_open(self, monkeypatch):
        """If return scan throws, result must still be returned."""
        from aegivis.security import exec_policy

        def broken_scan(result, cfg):
            raise RuntimeError("scan exploded")

        monkeypatch.setattr(exec_policy, "scan_return_value", broken_scan)

        def fetch():
            return {"data": "safe content"}

        wrapped = instrument(fetch, backend_url=_NO_BACKEND, scan_return=True)
        result = wrapped()
        assert result == {"data": "safe content"}


# ---------------------------------------------------------------------------
# Multi-signal scenarios
# ---------------------------------------------------------------------------


class TestMultiSignal:
    def test_multiple_signals_all_detected(self):
        """A tool call with both email and network signal should detect both."""
        from aegivis.security.arg_classifier import ArgClassifier

        clf = ArgClassifier()
        signals = clf.classify_call(
            args=(),
            kwargs={"to": "victim@evil.com", "url": "https://attacker.com/exfil"},
        )
        sig_types = {s.signal for s in signals}
        assert Signal.EMAIL_DESTINATION in sig_types
        assert Signal.NETWORK_DESTINATION in sig_types

    def test_block_on_any_of_multiple_types(self):
        """block_on with multiple types blocks when ANY is detected."""
        def tool(dest, url):
            return "done"

        wrapped = instrument(
            tool,
            backend_url=_NO_BACKEND,
            block_on=frozenset({Signal.CREDENTIAL, Signal.EMAIL_DESTINATION}),
        )

        # email alone should trigger block
        with pytest.raises(ToolExecutionBlocked):
            wrapped("safe", "attacker@evil.com")

    def test_two_args_one_blocked_one_not(self):
        """Second arg is a credential; first arg is clean. Must still block."""
        def tool(name, auth):
            return "ok"

        wrapped = make_gate_tool(tool, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped("alice", "sk-supersecretkey1234567890")


# ---------------------------------------------------------------------------
# Name independence
# ---------------------------------------------------------------------------


class TestNameIndependence:
    """The gate must work the same regardless of function or param names."""

    def test_function_named_x_blocks_on_credential(self):
        def x(y):
            return "ok"

        wrapped = make_gate_tool(x, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped("sk-supersecretkey1234567890")

    def test_function_named_safe_blocks_on_email(self):
        def safe_tool(recipient):
            return "ok"

        wrapped = make_gate_tool(safe_tool, block_on=frozenset({Signal.EMAIL_DESTINATION}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped("attacker@evil.com")

    def test_kwarg_named_x_triggers_credential_block(self):
        def tool(x):
            return "ok"

        wrapped = make_gate_tool(tool, block_on=frozenset({Signal.CREDENTIAL}))
        with pytest.raises(ToolExecutionBlocked):
            wrapped(x="sk-supersecretkey1234567890")


# ---------------------------------------------------------------------------
# Python AST code execution signal (SDK-C1)
# ---------------------------------------------------------------------------


class TestCodeExecutionSignal:
    def test_subprocess_import_detected(self):
        from aegivis.security.arg_classifier import ArgClassifier, Signal

        clf = ArgClassifier()
        sigs = clf.classify_call(
            args=("import subprocess\nsubprocess.run(['id'])",), kwargs={}
        )
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_eval_call_detected(self):
        from aegivis.security.arg_classifier import ArgClassifier, Signal

        clf = ArgClassifier()
        sigs = clf.classify_call(args=("eval('1+1')",), kwargs={})
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_safe_code_not_detected(self):
        from aegivis.security.arg_classifier import ArgClassifier, Signal

        clf = ArgClassifier()
        sigs = clf.classify_call(args=("result = [x*2 for x in range(10)]",), kwargs={})
        assert Signal.CODE_EXECUTION not in {s.signal for s in sigs}

    def test_code_execution_blocks_tool(self):
        def execute(code):
            return "ran"

        wrapped = instrument(
            execute,
            backend_url=_NO_BACKEND,
            block_on=frozenset({Signal.CODE_EXECUTION}),
        )
        with pytest.raises(ToolExecutionBlocked):
            wrapped("import subprocess\nsubprocess.run(['rm', '-rf', '/'])")

    def test_code_execution_allows_safe_code(self):
        def execute(code):
            return "ran"

        wrapped = instrument(
            execute,
            backend_url=_NO_BACKEND,
            block_on=frozenset({Signal.CODE_EXECUTION}),
        )
        result = wrapped("x = 1 + 2\nprint(x)")
        assert result == "ran"


# ---------------------------------------------------------------------------
# Concurrency limiter (SDK-C2)
# ---------------------------------------------------------------------------


class TestConcurrencyLimiter:
    def test_within_limit_succeeds(self):
        def tool():
            return "ok"

        wrapped = instrument(tool, backend_url=_NO_BACKEND, max_concurrent=3)
        assert wrapped() == "ok"

    def test_raises_when_limit_exceeded_sync(self):
        """Acquire the semaphore externally then verify the tool is rejected."""
        from aegivis.tools import _get_sync_semaphore

        session_id = "test-concurrency-sync"
        wrapped = instrument(
            lambda: "ok",
            backend_url=_NO_BACKEND,
            max_concurrent=1,
            session_id=session_id,
        )

        sem = _get_sync_semaphore(session_id, 1)
        # Manually acquire to simulate a concurrent execution
        acquired = sem.acquire(blocking=False)
        assert acquired
        try:
            with pytest.raises(ToolConcurrencyLimitExceeded) as exc_info:
                wrapped()
            assert exc_info.value.max_concurrent == 1
        finally:
            sem.release()

    @pytest.mark.asyncio
    async def test_raises_when_limit_exceeded_async(self):
        """Two concurrent async calls where limit=1: second must be rejected."""
        import asyncio
        from aegivis.tools import _get_async_semaphore

        session_id = "test-concurrency-async"

        async def slow():
            await asyncio.sleep(5)
            return "done"

        wrapped = instrument(
            slow,
            backend_url=_NO_BACKEND,
            max_concurrent=1,
            session_id=session_id,
        )

        sem = _get_async_semaphore(session_id, 1)
        await sem.acquire()  # hold the slot
        try:
            with pytest.raises(ToolConcurrencyLimitExceeded):
                await wrapped()
        finally:
            sem.release()

    def test_concurrency_limit_exception_attributes(self):
        from aegivis.tools import _get_sync_semaphore

        session_id = "test-concurrency-attrs"
        wrapped = instrument(
            lambda: "ok",
            backend_url=_NO_BACKEND,
            max_concurrent=2,
            session_id=session_id,
        )

        sem = _get_sync_semaphore(session_id, 2)
        sem.acquire(blocking=False)
        sem.acquire(blocking=False)
        try:
            with pytest.raises(ToolConcurrencyLimitExceeded) as exc_info:
                wrapped()
            assert exc_info.value.tool_name == "<lambda>"
            assert exc_info.value.max_concurrent == 2
        finally:
            sem.release()
            sem.release()

    def test_slot_released_after_success(self):
        """After a successful call, the slot must be released."""
        from aegivis.tools import _get_sync_semaphore

        session_id = "test-concurrency-release"
        wrapped = instrument(
            lambda: "ok",
            backend_url=_NO_BACKEND,
            max_concurrent=1,
            session_id=session_id,
        )

        wrapped()  # first call acquires and releases
        wrapped()  # second call must also succeed (slot was released)

    def test_slot_released_after_exception(self):
        """If the tool raises, the slot must still be released."""
        from aegivis.tools import _get_sync_semaphore

        session_id = "test-concurrency-exc-release"

        def boom():
            raise ValueError("intentional")

        wrapped = instrument(
            boom,
            backend_url=_NO_BACKEND,
            max_concurrent=1,
            session_id=session_id,
        )

        with pytest.raises(ValueError):
            wrapped()

        # Slot must be free — second call should succeed
        def ok():
            return "recovered"

        wrapped2 = instrument(
            ok,
            backend_url=_NO_BACKEND,
            max_concurrent=1,
            session_id=session_id,
        )
        assert wrapped2() == "recovered"

    def test_no_limit_by_default(self):
        """Without max_concurrent, no ToolConcurrencyLimitExceeded is raised."""
        def tool():
            return "ok"

        # Just run many times — no semaphore involved
        wrapped = instrument(tool, backend_url=_NO_BACKEND)
        for _ in range(10):
            assert wrapped() == "ok"
