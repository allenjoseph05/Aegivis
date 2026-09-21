"""
Tests for SDK cross-tool taint tracker (SDK-C3).

Covers:
  - SDKTaintTracker unit tests (record_return, check_args, eviction, thread safety)
  - Fragment extraction helpers (email, URL, IP, length)
  - tools.py integration (TOOL_EXEC_TAINT_FLOW event, taint_block, fail-open)
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aegivis.security.taint_tracker import (
    DEFAULT_UNTRUSTED_TOOLS,
    SDKTaintTracker,
    TaintFlow,
    TaintFragment,
    _collect_arg_strings,
    _extract_from_value,
    _is_email_fragment,
    _is_ip_fragment,
    _is_url_fragment,
)
from aegivis.tools import instrument, _session_taint_trackers, _taint_trackers_lock
from aegivis.security.exec_policy import ToolExecutionBlocked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_session() -> str:
    return f"test_sess_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Fragment helper unit tests
# ---------------------------------------------------------------------------

class TestFragmentHelpers:
    def test_email_valid(self):
        assert _is_email_fragment("user@example.com")

    def test_email_no_at(self):
        assert not _is_email_fragment("notanemail")

    def test_email_multiple_at(self):
        assert not _is_email_fragment("a@b@c.com")

    def test_email_no_domain_parts(self):
        assert not _is_email_fragment("user@nodot")

    def test_url_http(self):
        assert _is_url_fragment("http://example.com/path")

    def test_url_https(self):
        assert _is_url_fragment("https://api.evil.com/exfil")

    def test_url_ftp(self):
        assert _is_url_fragment("ftp://files.example.com/data")

    def test_url_no_netloc(self):
        assert not _is_url_fragment("not-a-url")

    def test_url_file_scheme_excluded(self):
        assert not _is_url_fragment("file:///etc/passwd")

    def test_ip_v4(self):
        assert _is_ip_fragment("192.168.1.1")

    def test_ip_v6_loopback(self):
        assert _is_ip_fragment("::1")

    def test_ip_v6_full(self):
        assert _is_ip_fragment("2001:db8::1")

    def test_ip_not_an_ip(self):
        assert not _is_ip_fragment("not.an.ip.address.here")

    def test_ip_domain_name_not_ip(self):
        assert not _is_ip_fragment("example.com")


class TestExtractFromValue:
    def test_string_above_min_len(self):
        results: list[str] = []
        _extract_from_value("this string is long enough", 15, results)
        assert "this string is long enough" in results

    def test_string_below_min_len(self):
        results: list[str] = []
        _extract_from_value("short", 15, results)
        assert not results

    def test_email_extracted_regardless_of_length(self):
        results: list[str] = []
        _extract_from_value("a@b.co", 100, results)
        assert "a@b.co" in results

    def test_url_extracted_regardless_of_length(self):
        results: list[str] = []
        _extract_from_value("http://x.co", 100, results)
        assert "http://x.co" in results

    def test_ip_extracted_regardless_of_length(self):
        results: list[str] = []
        _extract_from_value("10.0.0.1", 100, results)
        assert "10.0.0.1" in results

    def test_dict_values_extracted(self):
        results: list[str] = []
        _extract_from_value({"key": "a string long enough to matter"}, 15, results)
        assert "a string long enough to matter" in results

    def test_list_items_extracted(self):
        results: list[str] = []
        _extract_from_value(["long enough string here", "short"], 15, results)
        assert "long enough string here" in results
        assert "short" not in results

    def test_depth_limit_respected(self):
        # Build deeply nested structure (depth > 4)
        nested: Any = "deep value that is long enough"
        for _ in range(6):
            nested = {"k": nested}
        results: list[str] = []
        _extract_from_value(nested, 15, results)
        assert not results  # too deep

    def test_list_sampling_cap(self):
        # 40 items — only first 30 sampled
        data = [f"string_{i}_long_enough_to_track" for i in range(40)]
        results: list[str] = []
        _extract_from_value(data, 15, results)
        assert len(results) == 30


class TestCollectArgStrings:
    def test_positional_args(self):
        results: list[tuple[str, str]] = []
        _collect_arg_strings("hello world!", "args[0]", results)
        assert ("hello world!", "args[0]") in results

    def test_short_string_skipped(self):
        results: list[tuple[str, str]] = []
        _collect_arg_strings("hi", "args[0]", results)
        assert not results

    def test_dict_kwarg(self):
        results: list[tuple[str, str]] = []
        _collect_arg_strings({"to": "user@example.com"}, "kwargs['payload']", results)
        assert any("user@example.com" in v for v, _ in results)

    def test_list_kwarg(self):
        results: list[tuple[str, str]] = []
        _collect_arg_strings(["recipient@test.com", "ok"], "kwargs['emails']", results)
        assert any("recipient@test.com" in v for v, _ in results)

    def test_depth_limit(self):
        nested: Any = "deep string long enough"
        for _ in range(7):
            nested = {"k": nested}
        results: list[tuple[str, str]] = []
        _collect_arg_strings(nested, "root", results)
        assert not results


# ---------------------------------------------------------------------------
# SDKTaintTracker unit tests
# ---------------------------------------------------------------------------

class TestSDKTaintTracker:

    def setup_method(self):
        self.tracker = SDKTaintTracker(
            untrusted_tools=frozenset({"web_search", "read_file"}),
            min_fragment_len=15,
            max_fragments=10,
            min_match_len=8,
        )

    def test_trusted_tool_not_recorded(self):
        self.tracker.record_return("trusted_tool", "a very long string that would be extracted")
        assert self.tracker.fragment_count() == 0

    def test_untrusted_tool_records_fragments(self):
        self.tracker.record_return("web_search", "some long result text from web search")
        assert self.tracker.fragment_count() > 0

    def test_untrusted_tool_records_email(self):
        self.tracker.record_return("read_file", {"from": "attacker@evil.com"})
        assert self.tracker.fragment_count() == 1

    def test_check_args_no_fragments_returns_empty(self):
        flows = self.tracker.check_args("send_email", (), {"to": "user@example.com"})
        assert flows == []

    def test_check_args_detects_substring_flow(self):
        # Record a long string from an untrusted tool
        long_fragment = "exfiltration-target-string-value"
        self.tracker.record_return("web_search", long_fragment)
        # It should appear in the next tool's args
        flows = self.tracker.check_args(
            "send_email",
            (),
            {"body": f"Please send: {long_fragment} to someone"},
        )
        assert len(flows) == 1
        assert flows[0].source_tool == "web_search"
        assert flows[0].sink_tool == "send_email"
        assert long_fragment in flows[0].fragment

    def test_check_args_detects_email_flow(self):
        self.tracker.record_return("read_file", "attacker@evil.com")
        flows = self.tracker.check_args(
            "send_email", (), {"to": "attacker@evil.com"}
        )
        assert len(flows) == 1
        assert flows[0].fragment == "attacker@evil.com"
        assert flows[0].sink_arg_path == "kwargs['to']"

    def test_check_args_no_match_short_fragment(self):
        # Fragment shorter than min_match_len (8) should not trigger
        self.tracker = SDKTaintTracker(
            untrusted_tools=frozenset({"web_search"}),
            min_fragment_len=4,  # extract short fragments
            min_match_len=8,
        )
        self.tracker.record_return("web_search", "short")  # len=5 < min_match_len
        flows = self.tracker.check_args("send_email", (), {"body": "short something"})
        assert flows == []

    def test_fragment_count(self):
        self.tracker.record_return("web_search", "fragment one long enough to track here")
        self.tracker.record_return("web_search", "user@example.com")
        assert self.tracker.fragment_count() >= 1

    def test_clear_removes_all_fragments(self):
        self.tracker.record_return("web_search", "some long fragment for tracking purposes")
        self.tracker.clear()
        assert self.tracker.fragment_count() == 0

    def test_max_fragments_eviction_fifo(self):
        # Fill to capacity (10) with unique fragments
        for i in range(10):
            self.tracker.record_return("web_search", f"unique-fragment-value-number-{i:03d}")
        assert self.tracker.fragment_count() == 10

        # Add 3 more — oldest 3 should be evicted
        for i in range(10, 13):
            self.tracker.record_return("web_search", f"unique-fragment-value-number-{i:03d}")

        assert self.tracker.fragment_count() == 10
        # The oldest fragment should be gone
        flows = self.tracker.check_args(
            "sink", (), {"q": "unique-fragment-value-number-000"}
        )
        assert flows == []

    def test_deduplication_of_detected_flows(self):
        # Same fragment appearing in two fields should produce two flows (different paths)
        fragment = "shared-sensitive-data-value-here"
        self.tracker.record_return("web_search", fragment)
        flows = self.tracker.check_args(
            "upload", (), {"field1": fragment, "field2": fragment}
        )
        # Each (fragment, path) pair is unique — two paths → two flows
        assert len(flows) == 2

    def test_thread_safe_concurrent_record(self):
        """Multiple threads recording simultaneously must not corrupt state."""
        errors: list[Exception] = []
        def record_many(tool_name: str):
            try:
                for i in range(50):
                    self.tracker.record_return(tool_name, f"http://example.com/{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many, args=("web_search",)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Fragment count is bounded by max_fragments
        assert self.tracker.fragment_count() <= 10

    def test_fail_open_on_record_error(self):
        """record_return must not raise even if something goes wrong internally."""
        # Pass an object that raises on iteration
        class BadObj:
            def __iter__(self):
                raise RuntimeError("boom")

        # Should not raise
        self.tracker.record_return("web_search", BadObj())

    def test_fail_open_on_check_error(self):
        """check_args must return [] on any internal error."""
        class BadArg:
            def __len__(self):
                raise RuntimeError("boom")

        flows = self.tracker.check_args("send_email", (BadArg(),), {})
        assert flows == []

    def test_url_flow_detected(self):
        self.tracker.record_return("web_search", "https://evil.com/exfil")
        flows = self.tracker.check_args(
            "http_post", (), {"url": "https://evil.com/exfil"}
        )
        assert len(flows) == 1
        assert flows[0].source_tool == "web_search"

    def test_ip_flow_detected(self):
        self.tracker.record_return("read_file", "192.168.100.200")
        flows = self.tracker.check_args(
            "connect", (), {"host": "192.168.100.200"}
        )
        assert len(flows) == 1


class TestTaintFlowToDict:
    def test_fragment_truncated_to_80(self):
        flow = TaintFlow(
            fragment="x" * 100,
            source_tool="src",
            sink_tool="sink",
            sink_arg_path="args[0]",
        )
        d = flow.to_dict()
        assert len(d["fragment"]) == 80

    def test_fields_present(self):
        flow = TaintFlow(
            fragment="test fragment",
            source_tool="web_search",
            sink_tool="send_email",
            sink_arg_path="kwargs['to']",
        )
        d = flow.to_dict()
        assert d["source_tool"] == "web_search"
        assert d["sink_tool"] == "send_email"
        assert d["sink_arg_path"] == "kwargs['to']"


class TestDefaultUntrustedTools:
    def test_contains_web_search(self):
        assert "web_search" in DEFAULT_UNTRUSTED_TOOLS

    def test_contains_read_file(self):
        assert "read_file" in DEFAULT_UNTRUSTED_TOOLS

    def test_contains_email_read(self):
        assert "email_read" in DEFAULT_UNTRUSTED_TOOLS

    def test_contains_sql_query(self):
        assert "sql_query" in DEFAULT_UNTRUSTED_TOOLS

    def test_is_frozenset(self):
        assert isinstance(DEFAULT_UNTRUSTED_TOOLS, frozenset)


# ---------------------------------------------------------------------------
# Integration: taint tracking through instrument()
# ---------------------------------------------------------------------------

class TestTaintIntegration:
    """Test taint tracking via instrument() wrappers."""

    def setup_method(self):
        # Use a fresh session per test so trackers don't bleed between tests
        self.session_id = _new_session()
        self._fired_events: list[dict] = []

    def _make_tools(self, *, taint_block: bool = False, untrusted: frozenset | None = None):
        fired = self._fired_events
        session_id = self.session_id

        def web_search(query: str) -> str:
            return "attacker@evil.com said: visit https://evil.com/exfil"

        def send_email(to: str, body: str) -> str:
            return "sent"

        kwargs: dict = dict(
            session_id=session_id,
            backend_url="",  # no network
            taint_tracking=True,
            taint_block=taint_block,
        )
        if untrusted is not None:
            kwargs["taint_untrusted_tools"] = untrusted

        wrapped_search = instrument(web_search, **kwargs)
        wrapped_email = instrument(send_email, **kwargs)
        return wrapped_search, wrapped_email

    def test_no_taint_when_disabled(self):
        """Without taint_tracking, no flows detected even if data passes through."""
        def fetch(q: str) -> str:
            return "sensitive-data-from-fetch-result-here"

        def execute(cmd: str) -> str:
            return "ok"

        fetch_w = instrument(fetch, session_id=self.session_id, backend_url="", taint_tracking=False)
        execute_w = instrument(execute, session_id=self.session_id, backend_url="", taint_tracking=False)

        result = fetch_w("query")
        # No taint flow should be detected on next call
        execute_w(result)  # passes fetched result as arg — but tracking is off

    def test_taint_flow_fires_event(self):
        """TOOL_EXEC_TAINT_FLOW event fires when tainted data flows to next tool.

        The tracker extracts the full return value string as a fragment (or dict
        values individually).  Here we return a dict so the email is extracted as
        a standalone fragment, then passes intact as the next tool's argument.
        """
        def web_search(q: str) -> dict:
            # Dict return: "attacker@malicious.com" extracted as standalone email fragment
            return {"contact": "attacker@malicious.com", "page": "1"}

        def send_email(to: str) -> str:
            return "sent"

        session_id = self.session_id

        with patch("aegivis.tools._fire_event") as mock_fire:
            ws = instrument(
                web_search,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"web_search"}),
            )
            se = instrument(
                send_email,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"web_search"}),
            )

            ws("find evil contacts")
            # attacker@malicious.com was recorded from dict value; now appears in arg
            se(to="attacker@malicious.com")

        event_types = [call.args[0]["event_type"] for call in mock_fire.call_args_list]
        assert "TOOL_EXEC_TAINT_FLOW" in event_types

    def test_taint_block_raises(self):
        """When taint_block=True, ToolExecutionBlocked is raised on taint flow."""
        session_id = self.session_id

        def read_file(path: str) -> dict:
            # Dict return extracts the email as a standalone fragment
            return {"owner": "attacker@pwned.com", "size": "1024"}

        def send_email(to: str) -> str:
            return "sent"

        rf = instrument(
            read_file,
            session_id=session_id,
            backend_url="",
            taint_tracking=True,
            taint_block=True,
            taint_untrusted_tools=frozenset({"read_file"}),
        )
        se = instrument(
            send_email,
            session_id=session_id,
            backend_url="",
            taint_tracking=True,
            taint_block=True,
            taint_untrusted_tools=frozenset({"read_file"}),
        )

        rf("/etc/config")  # records "attacker@pwned.com" as tainted fragment

        with pytest.raises(ToolExecutionBlocked) as exc_info:
            se(to="attacker@pwned.com")

        assert exc_info.value.tool_name == "send_email"
        assert "Taint flow" in exc_info.value.reason

    def test_taint_flow_event_format(self):
        """TOOL_EXEC_TAINT_FLOW event payload has expected structure."""
        session_id = self.session_id

        def fetch_url(url: str) -> dict:
            # Dict so email is extracted as a standalone fragment
            return {"contact": "secret@target.org", "data": "some payload"}

        def upload(destination: str) -> str:
            return "ok"

        captured: list[dict] = []

        def capture_event(event: dict, backend_url: str, api_key: str) -> None:
            if event["event_type"] == "TOOL_EXEC_TAINT_FLOW":
                captured.append(event)

        with patch("aegivis.tools._fire_event", side_effect=capture_event):
            fu = instrument(
                fetch_url,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"fetch_url"}),
            )
            up = instrument(
                upload,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"fetch_url"}),
            )

            fu("http://api.example.com")
            up("secret@target.org")

        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert "taint_flows" in payload
        flow = payload["taint_flows"][0]
        assert "source_tool" in flow
        assert "sink_tool" in flow
        assert "sink_arg_path" in flow
        assert "fragment" in flow
        assert flow["source_tool"] == "fetch_url"
        assert flow["sink_tool"] == "upload"

    def test_custom_untrusted_tools(self):
        """taint_untrusted_tools override works — only specified tools are tracked."""
        session_id = self.session_id

        def my_custom_source(q: str) -> str:
            return "custom-taint-fragment-value-long"

        def sink_fn(data: str) -> str:
            return "ok"

        captured: list[dict] = []

        def capture_event(event: dict, backend_url: str, api_key: str) -> None:
            if event["event_type"] == "TOOL_EXEC_TAINT_FLOW":
                captured.append(event)

        with patch("aegivis.tools._fire_event", side_effect=capture_event):
            src = instrument(
                my_custom_source,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"my_custom_source"}),
            )
            sink = instrument(
                sink_fn,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"my_custom_source"}),
            )

            result = src("query")
            sink(result)  # passes tainted data as positional arg

        assert len(captured) == 1
        assert captured[0]["payload"]["taint_flows"][0]["source_tool"] == "my_custom_source"

    def test_no_flow_when_different_sessions(self):
        """Taint tracker is per-session — different sessions don't share fragments."""
        session_a = _new_session()
        session_b = _new_session()

        def fetch(q: str) -> str:
            return "cross-session-sensitive-data-leaked"

        def sink(data: str) -> str:
            return "ok"

        captured: list[dict] = []

        def capture_event(event: dict, backend_url: str, api_key: str) -> None:
            if event["event_type"] == "TOOL_EXEC_TAINT_FLOW":
                captured.append(event)

        with patch("aegivis.tools._fire_event", side_effect=capture_event):
            src = instrument(
                fetch,
                session_id=session_a,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"fetch"}),
            )
            snk = instrument(
                sink,
                session_id=session_b,  # different session
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"fetch"}),
            )

            result = src("query")
            snk(result)

        assert captured == []

    def test_taint_tracking_async(self):
        """Taint tracking works through async wrappers."""
        session_id = self.session_id
        captured: list[dict] = []

        def capture_event(event: dict, backend_url: str, api_key: str) -> None:
            if event["event_type"] == "TOOL_EXEC_TAINT_FLOW":
                captured.append(event)

        async def async_fetch(q: str) -> dict:
            # Dict return so the email is extracted as a standalone fragment
            return {"email": "victim@example.com", "rank": 1}

        async def async_sink(email: str) -> str:
            return "done"

        with patch("aegivis.tools._fire_event", side_effect=capture_event):
            src = instrument(
                async_fetch,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"async_fetch"}),
            )
            snk = instrument(
                async_sink,
                session_id=session_id,
                backend_url="",
                taint_tracking=True,
                taint_untrusted_tools=frozenset({"async_fetch"}),
            )

            async def run():
                await src("search")
                await snk("victim@example.com")

            asyncio.run(run())

        assert len(captured) == 1
        assert captured[0]["payload"]["taint_flows"][0]["source_tool"] == "async_fetch"

    def test_taint_block_false_still_executes(self):
        """With taint_block=False (default), tool executes despite taint flow."""
        session_id = self.session_id
        executed = []

        def read_url(url: str) -> str:
            return "data from attacker@evil.com about target"

        def process(email: str) -> str:
            executed.append(email)
            return "done"

        ru = instrument(
            read_url,
            session_id=session_id,
            backend_url="",
            taint_tracking=True,
            taint_block=False,
            taint_untrusted_tools=frozenset({"read_url"}),
        )
        pr = instrument(
            process,
            session_id=session_id,
            backend_url="",
            taint_tracking=True,
            taint_block=False,
            taint_untrusted_tools=frozenset({"read_url"}),
        )

        ru("http://feed.example.com")
        pr("attacker@evil.com")

        # Tool executed despite taint flow
        assert "attacker@evil.com" in executed
