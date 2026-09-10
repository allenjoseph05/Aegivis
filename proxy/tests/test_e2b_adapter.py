"""
Unit tests for E2BSandboxAdapter (Phase 16.2).

_run_in_e2b is mocked — no real E2B API calls or network required.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.security.adapters.e2b_adapter import (
    E2BSandboxAdapter,
    _extract_code,
    _looks_like_interpreter_command,
    _tokenize,
)
from app.security.sandbox import SandboxContext


def ctx(timeout_ms: int = 3000) -> SandboxContext:
    return SandboxContext(session_id="s", agent_id="a", org_id="o", timeout_ms=timeout_ms)


def run(coro):
    return asyncio.run(coro)


ADP = E2BSandboxAdapter()

_OK_OBS  = {"exit_code": 0, "stdout": "hello", "stderr": "", "files_written": 0}
_ERR_OBS = {"exit_code": 1, "stdout": "", "stderr": "NameError: foo", "files_written": 0}
_MASS_OBS = {"exit_code": 0, "stdout": "done", "stderr": "", "files_written": 150}
_WARN_OBS = {"exit_code": 0, "stdout": "done", "stderr": "", "files_written": 30}


def _mock_run(obs: dict):
    return patch(
        "app.security.adapters.e2b_adapter._run_in_e2b",
        new=AsyncMock(return_value=obs),
    )


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_split_underscore(self):
        assert "run" in _tokenize("run_python")
        assert "python" in _tokenize("run_python")

    def test_lowercased(self):
        assert "code" in _tokenize("CODE_EXECUTE")

    def test_single(self):
        assert _tokenize("exec") == frozenset({"exec"})


# ── _extract_code ─────────────────────────────────────────────────────────────

class TestExtractCode:
    def test_code_key(self):
        assert _extract_code({"code": "print(1)"}) == "print(1)"

    def test_script_key(self):
        assert _extract_code({"script": "x = 1"}) == "x = 1"

    def test_python_key(self):
        assert _extract_code({"python": "import os"}) == "import os"

    def test_source_key(self):
        assert _extract_code({"source": "fn main() {}"}) == "fn main() {}"

    def test_empty_string_skipped(self):
        assert _extract_code({"code": "   ", "script": "real code"}) == "real code"

    def test_no_code_key(self):
        assert _extract_code({"message": "hello"}) is None


# ── _looks_like_interpreter_command ───────────────────────────────────────────

class TestLooksLikeInterpreter:
    def test_python3(self):
        assert _looks_like_interpreter_command("python3 script.py")

    def test_bash(self):
        assert _looks_like_interpreter_command("bash run.sh")

    def test_node(self):
        assert _looks_like_interpreter_command("node index.js")

    def test_ls_not_interpreter(self):
        assert not _looks_like_interpreter_command("ls -la")

    def test_git_not_interpreter(self):
        assert not _looks_like_interpreter_command("git push")


# ── can_handle ────────────────────────────────────────────────────────────────

class TestCanHandle:
    def test_execute_tool_name(self):
        assert run(ADP.can_handle("execute_code", {}))

    def test_python_tool_name(self):
        assert run(ADP.can_handle("python_runner", {}))

    def test_run_script_tool(self):
        assert run(ADP.can_handle("run_script", {}))

    def test_shell_tool(self):
        assert run(ADP.can_handle("shell", {}))

    def test_bash_tool(self):
        assert run(ADP.can_handle("bash_exec", {}))

    def test_notebook_tool(self):
        assert run(ADP.can_handle("jupyter_notebook", {}))

    def test_code_arg_key(self):
        assert run(ADP.can_handle("tool", {"code": "print(1)"}))

    def test_script_arg_key(self):
        assert run(ADP.can_handle("tool", {"script": "x = 1"}))

    def test_command_starts_with_python(self):
        assert run(ADP.can_handle("run", {"command": "python3 main.py"}))

    def test_command_bash(self):
        assert run(ADP.can_handle("run", {"command": "bash run.sh"}))

    def test_no_match(self):
        assert not run(ADP.can_handle("send_email", {"to": "a@b.com"}))

    def test_git_not_code(self):
        assert not run(ADP.can_handle("git_push", {}))


# ── dry_run: no code ──────────────────────────────────────────────────────────

class TestNoCode:
    def test_no_code_signal(self):
        r = run(ADP.dry_run("execute_code", {}, ctx()))
        assert "e2b-no-code" in r.signals

    def test_no_code_safe(self):
        r = run(ADP.dry_run("execute_code", {}, ctx()))
        assert r.safe is True

    def test_no_code_not_executed(self):
        r = run(ADP.dry_run("execute_code", {}, ctx()))
        assert r.executed is False


# ── dry_run: E2B unavailable (fail-open) ──────────────────────────────────────

class TestE2BUnavailable:
    def _raise_no_key(self, *a, **kw):
        raise RuntimeError("E2B_API_KEY not set")

    def _raise_not_installed(self, *a, **kw):
        raise RuntimeError("e2b_code_interpreter package not installed")

    def _raise_other(self, *a, **kw):
        raise RuntimeError("connection refused")

    def test_no_api_key_fail_open(self):
        with patch("app.security.adapters.e2b_adapter._run_in_e2b", side_effect=self._raise_no_key):
            r = run(ADP.dry_run("execute_code", {"code": "print(1)"}, ctx()))
        assert r.safe is True
        assert "e2b-no-api-key" in r.signals

    def test_not_installed_fail_open(self):
        with patch("app.security.adapters.e2b_adapter._run_in_e2b", side_effect=self._raise_not_installed):
            r = run(ADP.dry_run("python_run", {"code": "x=1"}, ctx()))
        assert r.safe is True
        assert "e2b-not-installed" in r.signals

    def test_other_error_fail_open(self):
        with patch("app.security.adapters.e2b_adapter._run_in_e2b", side_effect=self._raise_other):
            r = run(ADP.dry_run("execute_code", {"code": "x=1"}, ctx()))
        assert r.safe is True
        assert "e2b-unavailable" in r.signals

    def test_unavailable_not_executed(self):
        with patch("app.security.adapters.e2b_adapter._run_in_e2b", side_effect=self._raise_no_key):
            r = run(ADP.dry_run("execute_code", {"code": "x=1"}, ctx()))
        assert r.executed is False


# ── dry_run: successful execution ─────────────────────────────────────────────

class TestSuccessfulRun:
    def test_clean_run_safe(self):
        with _mock_run(_OK_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "print('hi')"}, ctx()))
        assert r.safe is True
        assert "e2b-ok" in r.signals

    def test_clean_run_executed_true(self):
        with _mock_run(_OK_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "print('hi')"}, ctx()))
        assert r.executed is True

    def test_scope_has_exit_code(self):
        with _mock_run(_OK_OBS):
            r = run(ADP.dry_run("python_runner", {"code": "x=1"}, ctx()))
        assert r.scope_estimate["exit_code"] == 0

    def test_scope_has_files_written(self):
        with _mock_run(_OK_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "x=1"}, ctx()))
        assert "files_written" in r.scope_estimate

    def test_scope_has_stdout(self):
        with _mock_run({**_OK_OBS, "stdout": "hello world"}):
            r = run(ADP.dry_run("execute_code", {"code": "print('hello world')"}, ctx()))
        assert "hello world" in r.scope_estimate.get("stdout", "")

    def test_adapter_name(self):
        with _mock_run(_OK_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "x=1"}, ctx()))
        assert r.adapter == "e2b"


# ── dry_run: exit code error ──────────────────────────────────────────────────

class TestExitError:
    def test_exit_1_unsafe(self):
        with _mock_run(_ERR_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "raise ValueError()"}, ctx()))
        assert r.safe is False

    def test_exit_error_signal(self):
        with _mock_run(_ERR_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "bad"}, ctx()))
        assert any("e2b-exit-error" in s for s in r.signals)

    def test_exit_error_still_executed(self):
        with _mock_run(_ERR_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "bad"}, ctx()))
        assert r.executed is True


# ── dry_run: mass file write ──────────────────────────────────────────────────

class TestMassFileWrite:
    def test_mass_write_unsafe(self):
        with _mock_run(_MASS_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "..."}, ctx()))
        assert r.safe is False
        assert any("e2b-mass-write" in s for s in r.signals)

    def test_warn_write_safe_but_signalled(self):
        with _mock_run(_WARN_OBS):
            r = run(ADP.dry_run("execute_code", {"code": "..."}, ctx()))
        assert r.safe is True
        assert any("e2b-write-warn" in s for s in r.signals)


# ── adapter metadata ──────────────────────────────────────────────────────────

class TestMeta:
    def test_name(self):
        assert ADP.name == "e2b"

    def test_latency_positive(self):
        r = run(ADP.dry_run("execute_code", {}, ctx()))
        assert r.latency_ms >= 0.0

    def test_preview_capped(self):
        with _mock_run({**_OK_OBS, "stdout": "x" * 1000}):
            r = run(ADP.dry_run("execute_code", {"code": "..."}, ctx()))
        assert len(r.preview) <= 500
