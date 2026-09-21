"""
Tests for SDK-C6: Per-tool argument constraints.

Covers:
  - path_prefixes: path-shaped args must start with an allowed prefix
  - url_domains: URL-shaped args must have an allowed hostname (with subdomain matching)
  - ToolArgumentViolation exception attributes
  - TOOL_EXEC_BLOCKED event payload
  - Non-path / non-URL strings are not checked
  - Recursive scanning (dict values, list items)
  - Both constraints simultaneously
  - Sync and async wrappers
  - Decorator syntax
  - None constraints = no restriction (backward compat)
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

import pytest

from aegivis.tools import instrument, _looks_like_path, _looks_like_url, _domain_allowed
from aegivis.security.exec_policy import ToolArgumentViolation


def _sid() -> str:
    return f"sess_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Structural detection helpers
# ---------------------------------------------------------------------------

class TestLooksLikePath:
    def test_unix_absolute(self):
        assert _looks_like_path("/etc/passwd")

    def test_unix_data_dir(self):
        assert _looks_like_path("/data/file.txt")

    def test_home_relative(self):
        assert _looks_like_path("~/documents/report.pdf")

    def test_current_dir_relative(self):
        assert _looks_like_path("./local/file.txt")

    def test_parent_dir_relative(self):
        assert _looks_like_path("../secrets/key.pem")

    def test_windows_absolute(self):
        assert _looks_like_path("C:\\Users\\admin\\file.txt")
        assert _looks_like_path("C:/Users/admin/file.txt")

    def test_plain_filename_not_path(self):
        assert not _looks_like_path("report.pdf")

    def test_email_not_path(self):
        assert not _looks_like_path("user@example.com")

    def test_url_not_path(self):
        assert not _looks_like_path("http://example.com/path")

    def test_empty_string_not_path(self):
        assert not _looks_like_path("")

    def test_single_char_not_path(self):
        assert not _looks_like_path("x")


class TestLooksLikeUrl:
    def test_http(self):
        assert _looks_like_url("http://example.com/api")

    def test_https(self):
        assert _looks_like_url("https://api.evil.com/exfil")

    def test_ws(self):
        assert _looks_like_url("ws://stream.example.com/feed")

    def test_ftp(self):
        assert _looks_like_url("ftp://files.example.com/data")

    def test_plain_domain_not_url(self):
        assert not _looks_like_url("example.com")

    def test_file_scheme_not_url(self):
        assert not _looks_like_url("file:///etc/passwd")

    def test_path_not_url(self):
        assert not _looks_like_url("/etc/passwd")

    def test_empty_not_url(self):
        assert not _looks_like_url("")


class TestDomainAllowed:
    def test_exact_match(self):
        assert _domain_allowed("example.com", frozenset({"example.com"}))

    def test_subdomain_match(self):
        assert _domain_allowed("api.example.com", frozenset({"example.com"}))

    def test_deep_subdomain_match(self):
        assert _domain_allowed("v2.api.example.com", frozenset({"example.com"}))

    def test_different_domain_rejected(self):
        assert not _domain_allowed("evil.com", frozenset({"example.com"}))

    def test_partial_domain_not_matched(self):
        # "evil-example.com" must NOT match "example.com"
        assert not _domain_allowed("evil-example.com", frozenset({"example.com"}))

    def test_case_insensitive(self):
        assert _domain_allowed("API.EXAMPLE.COM", frozenset({"example.com"}))

    def test_multiple_allowed_domains(self):
        allowed = frozenset({"openai.com", "anthropic.com"})
        assert _domain_allowed("api.openai.com", allowed)
        assert _domain_allowed("claude.anthropic.com", allowed)
        assert not _domain_allowed("evil.com", allowed)


# ---------------------------------------------------------------------------
# Path prefix constraint
# ---------------------------------------------------------------------------

class TestPathPrefixes:

    def test_allowed_path_executes(self):
        executed = []

        def read_file(path: str) -> str:
            executed.append(path)
            return "content"

        wrapped = instrument(
            read_file,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/", "/tmp/"}),
        )
        wrapped(path="/data/report.txt")
        assert executed == ["/data/report.txt"]

    def test_disallowed_path_blocked(self):
        def read_file(path: str) -> str:
            return "content"

        wrapped = instrument(
            read_file,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )
        with pytest.raises(ToolArgumentViolation) as exc_info:
            wrapped(path="/etc/passwd")

        assert exc_info.value.constraint_type == "path_prefix"
        assert "/etc/passwd" in exc_info.value.value
        assert exc_info.value.tool_name == "read_file"

    def test_none_path_prefixes_allows_all(self):
        executed = []

        def read_file(path: str) -> str:
            executed.append(path)
            return "ok"

        wrapped = instrument(read_file, session_id=_sid(), backend_url="")
        wrapped(path="/etc/passwd")
        assert executed == ["/etc/passwd"]

    def test_multiple_allowed_prefixes(self):
        executed = []

        def write_file(path: str) -> None:
            executed.append(path)

        wrapped = instrument(
            write_file,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/", "/tmp/", "/var/log/"}),
        )
        wrapped(path="/data/output.txt")
        wrapped(path="/tmp/scratch.txt")
        wrapped(path="/var/log/app.log")
        assert len(executed) == 3

    def test_prefix_must_be_exact_start(self):
        """'/data' prefix should not allow '/data_exfil/file'."""
        def write(path: str) -> None:
            pass

        wrapped = instrument(
            write,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )
        # /data_exfil/file.txt does NOT start with /data/
        with pytest.raises(ToolArgumentViolation):
            wrapped(path="/data_exfil/file.txt")

    def test_path_in_dict_arg_checked(self):
        def process(opts: dict) -> str:
            return "ok"

        wrapped = instrument(
            process,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/safe/"}),
        )
        with pytest.raises(ToolArgumentViolation):
            wrapped(opts={"input": "/etc/shadow", "output": "/safe/result.txt"})

    def test_path_in_list_arg_checked(self):
        def batch(paths: list) -> str:
            return "ok"

        wrapped = instrument(
            batch,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )
        with pytest.raises(ToolArgumentViolation):
            wrapped(paths=["/data/ok.txt", "/etc/passwd"])

    def test_non_path_string_not_checked(self):
        """Plain strings that aren't path-shaped are not checked against prefixes."""
        executed = []

        def send(message: str) -> str:
            executed.append(message)
            return "sent"

        wrapped = instrument(
            send,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/allowed/"}),
        )
        # This is not path-shaped — should pass unchecked
        wrapped(message="hello world, no path here")
        assert executed == ["hello world, no path here"]

    def test_home_relative_path_blocked(self):
        def read(path: str) -> str:
            return "ok"

        wrapped = instrument(
            read,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )
        with pytest.raises(ToolArgumentViolation):
            wrapped(path="~/secrets/key.pem")

    def test_blocked_fires_event(self):
        def read(path: str) -> str:
            return "ok"

        fired: list[dict] = []
        with patch("aegivis.tools._fire_event", side_effect=lambda e, *_: fired.append(e)):
            wrapped = instrument(
                read,
                session_id=_sid(), backend_url="http://fake",
                path_prefixes=frozenset({"/data/"}),
            )
            with pytest.raises(ToolArgumentViolation):
                wrapped(path="/etc/passwd")

        blocked = [e for e in fired if e["event_type"] == "TOOL_EXEC_BLOCKED"]
        assert len(blocked) == 1
        assert blocked[0]["payload"]["constraint_type"] == "path_prefix"
        assert "/etc/passwd" in blocked[0]["payload"]["value_preview"]

    def test_blocked_tool_body_not_called(self):
        executed = []

        def read(path: str) -> str:
            executed.append(path)
            return "content"

        wrapped = instrument(
            read,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )
        with pytest.raises(ToolArgumentViolation):
            wrapped(path="/etc/passwd")

        assert executed == []


# ---------------------------------------------------------------------------
# URL domain constraint
# ---------------------------------------------------------------------------

class TestUrlDomains:

    def test_allowed_domain_executes(self):
        executed = []

        def fetch(url: str) -> str:
            executed.append(url)
            return "response"

        wrapped = instrument(
            fetch,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"api.example.com"}),
        )
        wrapped(url="https://api.example.com/v1/data")
        assert executed == ["https://api.example.com/v1/data"]

    def test_disallowed_domain_blocked(self):
        def fetch(url: str) -> str:
            return "response"

        wrapped = instrument(
            fetch,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"api.example.com"}),
        )
        with pytest.raises(ToolArgumentViolation) as exc_info:
            wrapped(url="https://evil.com/exfil")

        assert exc_info.value.constraint_type == "url_domain"
        assert "evil.com" in exc_info.value.value
        assert exc_info.value.tool_name == "fetch"

    def test_none_url_domains_allows_all(self):
        executed = []

        def fetch(url: str) -> str:
            executed.append(url)
            return "ok"

        wrapped = instrument(fetch, session_id=_sid(), backend_url="")
        wrapped(url="https://evil.com/anything")
        assert executed == ["https://evil.com/anything"]

    def test_subdomain_allowed(self):
        executed = []

        def call(url: str) -> str:
            executed.append(url)
            return "ok"

        wrapped = instrument(
            call,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"example.com"}),
        )
        wrapped(url="https://api.example.com/v2/endpoint")
        wrapped(url="https://v2.api.example.com/data")
        assert len(executed) == 2

    def test_partial_domain_name_not_allowed(self):
        """evil-example.com must not match example.com."""
        def call(url: str) -> str:
            return "ok"

        wrapped = instrument(
            call,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"example.com"}),
        )
        with pytest.raises(ToolArgumentViolation):
            wrapped(url="https://evil-example.com/path")

    def test_multiple_allowed_domains(self):
        executed = []

        def call(url: str) -> str:
            executed.append(url)
            return "ok"

        wrapped = instrument(
            call,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"openai.com", "anthropic.com"}),
        )
        wrapped(url="https://api.openai.com/v1/chat")
        wrapped(url="https://api.anthropic.com/v1/messages")
        assert len(executed) == 2

        with pytest.raises(ToolArgumentViolation):
            wrapped(url="https://google.com/search")

    def test_non_url_string_not_checked(self):
        """Plain strings are not checked against url_domains."""
        executed = []

        def process(text: str) -> str:
            executed.append(text)
            return "ok"

        wrapped = instrument(
            process,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"example.com"}),
        )
        # Plain text with no URL scheme — not checked
        wrapped(text="evil.com is a domain but not a URL")
        assert len(executed) == 1

    def test_url_in_dict_checked(self):
        def call(config: dict) -> str:
            return "ok"

        wrapped = instrument(
            call,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"safe.com"}),
        )
        with pytest.raises(ToolArgumentViolation):
            wrapped(config={"endpoint": "https://evil.com/api", "timeout": 30})

    def test_blocked_fires_event(self):
        def fetch(url: str) -> str:
            return "ok"

        fired: list[dict] = []
        with patch("aegivis.tools._fire_event", side_effect=lambda e, *_: fired.append(e)):
            wrapped = instrument(
                fetch,
                session_id=_sid(), backend_url="http://fake",
                url_domains=frozenset({"safe.com"}),
            )
            with pytest.raises(ToolArgumentViolation):
                wrapped(url="https://evil.com/exfil")

        blocked = [e for e in fired if e["event_type"] == "TOOL_EXEC_BLOCKED"]
        assert len(blocked) == 1
        assert blocked[0]["payload"]["constraint_type"] == "url_domain"


# ---------------------------------------------------------------------------
# Both constraints together
# ---------------------------------------------------------------------------

class TestBothConstraints:

    def test_both_pass(self):
        executed = []

        def tool(path: str, url: str) -> str:
            executed.append((path, url))
            return "ok"

        wrapped = instrument(
            tool,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
            url_domains=frozenset({"api.example.com"}),
        )
        wrapped(path="/data/file.txt", url="https://api.example.com/v1")
        assert len(executed) == 1

    def test_bad_path_blocked_despite_good_url(self):
        def tool(path: str, url: str) -> str:
            return "ok"

        wrapped = instrument(
            tool,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
            url_domains=frozenset({"api.example.com"}),
        )
        with pytest.raises(ToolArgumentViolation) as exc_info:
            wrapped(path="/etc/passwd", url="https://api.example.com/v1")
        assert exc_info.value.constraint_type == "path_prefix"

    def test_bad_url_blocked_despite_good_path(self):
        def tool(path: str, url: str) -> str:
            return "ok"

        wrapped = instrument(
            tool,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
            url_domains=frozenset({"api.example.com"}),
        )
        with pytest.raises(ToolArgumentViolation) as exc_info:
            wrapped(path="/data/file.txt", url="https://evil.com/exfil")
        assert exc_info.value.constraint_type == "url_domain"


# ---------------------------------------------------------------------------
# ToolArgumentViolation exception
# ---------------------------------------------------------------------------

class TestToolArgumentViolationExc:

    def test_attributes_path(self):
        exc = ToolArgumentViolation("read_file", "path_prefix", "/etc/passwd", "['/data/']")
        assert exc.tool_name == "read_file"
        assert exc.constraint_type == "path_prefix"
        assert exc.value == "/etc/passwd"
        assert exc.allowed == "['/data/']"

    def test_attributes_url(self):
        exc = ToolArgumentViolation("fetch", "url_domain", "https://evil.com", "['safe.com']")
        assert exc.constraint_type == "url_domain"
        assert "evil.com" in exc.value

    def test_message_contains_key_info(self):
        exc = ToolArgumentViolation("fn", "path_prefix", "/etc/passwd", "['/data/']")
        msg = str(exc)
        assert "fn" in msg
        assert "path_prefix" in msg
        assert "/etc/passwd" in msg

    def test_is_runtime_error(self):
        assert isinstance(
            ToolArgumentViolation("fn", "url_domain", "https://evil.com", "[]"),
            RuntimeError,
        )


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

class TestConstraintsAsync:

    def test_async_allowed_path(self):
        async def read(path: str) -> str:
            return "content"

        wrapped = instrument(
            read,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )

        async def run():
            result = await wrapped(path="/data/file.txt")
            assert result == "content"

        asyncio.run(run())

    def test_async_blocked_path(self):
        async def read(path: str) -> str:
            return "content"

        wrapped = instrument(
            read,
            session_id=_sid(), backend_url="",
            path_prefixes=frozenset({"/data/"}),
        )

        async def run():
            with pytest.raises(ToolArgumentViolation):
                await wrapped(path="/etc/shadow")

        asyncio.run(run())

    def test_async_blocked_url(self):
        async def fetch(url: str) -> str:
            return "ok"

        wrapped = instrument(
            fetch,
            session_id=_sid(), backend_url="",
            url_domains=frozenset({"safe.com"}),
        )

        async def run():
            with pytest.raises(ToolArgumentViolation) as exc_info:
                await wrapped(url="https://evil.com/exfil")
            assert exc_info.value.constraint_type == "url_domain"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Decorator syntax
# ---------------------------------------------------------------------------

class TestConstraintsDecorator:

    def test_decorator_path_prefix(self):
        sid = _sid()

        @instrument.tool(
            session_id=sid, backend_url="",
            path_prefixes=frozenset({"/safe/"}),
        )
        def write(path: str) -> str:
            return "written"

        assert write(path="/safe/output.txt") == "written"

        with pytest.raises(ToolArgumentViolation):
            write(path="/etc/crontab")

    def test_decorator_url_domain(self):
        sid = _sid()

        @instrument.tool(
            session_id=sid, backend_url="",
            url_domains=frozenset({"trusted.com"}),
        )
        def call(url: str) -> str:
            return "ok"

        assert call(url="https://trusted.com/api") == "ok"

        with pytest.raises(ToolArgumentViolation):
            call(url="https://untrusted.com/api")
