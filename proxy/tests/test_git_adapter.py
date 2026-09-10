"""
Unit tests for the Git Sandbox Adapter — Phase 16.7.

Git subprocess calls are mocked so tests pass without a real git repo.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.security.adapters.git import (
    GitSandboxAdapter,
    _extract_git_command,
    _extract_clone_url,
    _extract_paths,
    _tokenize,
)
from app.security.sandbox import SandboxContext


def ctx(timeout_ms: int = 2000) -> SandboxContext:
    return SandboxContext(
        session_id="s", agent_id="a", org_id="o", timeout_ms=timeout_ms,
    )


def run(coro):
    return asyncio.run(coro)


ADP = GitSandboxAdapter()


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_underscore_split(self):
        assert "git" in _tokenize("git_push")
        assert "push" in _tokenize("git_push")

    def test_camel_case_no_split(self):
        # _tokenize splits on non-alphanumeric only — camelCase stays as one token
        tokens = _tokenize("gitPush")
        assert "gitpush" in tokens

    def test_single_word(self):
        assert _tokenize("push") == frozenset({"push"})

    def test_lowercased(self):
        assert "git" in _tokenize("GIT_PUSH")


# ── _extract_git_command ──────────────────────────────────────────────────────

class TestExtractGitCommand:
    def test_command_arg(self):
        assert _extract_git_command("run_git", {"command": "push origin main"}) == "push origin main"

    def test_command_arg_strips_git_prefix(self):
        assert _extract_git_command("run_git", {"command": "git push origin"}) == "push origin"

    def test_cmd_arg(self):
        assert _extract_git_command("run_git", {"cmd": "commit -m 'msg'"}) == "commit -m 'msg'"

    def test_tool_name_push(self):
        assert _extract_git_command("git_push", {}) == "push"

    def test_tool_name_commit(self):
        assert _extract_git_command("git_commit", {}) == "commit"

    def test_tool_name_clone(self):
        assert _extract_git_command("git_clone", {}) == "clone"

    def test_unknown_falls_back(self):
        result = _extract_git_command("run_git", {})
        assert isinstance(result, str)


# ── _extract_clone_url ────────────────────────────────────────────────────────

class TestExtractCloneUrl:
    def test_url_key(self):
        assert _extract_clone_url({"url": "https://github.com/foo/bar"}) == "https://github.com/foo/bar"

    def test_repo_key(self):
        assert _extract_clone_url({"repo": "git@github.com:foo/bar.git"}) == "git@github.com:foo/bar.git"

    def test_repository_key(self):
        assert _extract_clone_url({"repository": "https://repo.example.com"}) == "https://repo.example.com"

    def test_fallback_to_empty(self):
        url = _extract_clone_url({})
        assert isinstance(url, str)


# ── _extract_paths ────────────────────────────────────────────────────────────

class TestExtractPaths:
    def test_list_paths(self):
        assert _extract_paths({"paths": ["a.txt", "b.txt"]}) == ["a.txt", "b.txt"]

    def test_files_key(self):
        assert _extract_paths({"files": ["x.py"]}) == ["x.py"]

    def test_string_paths(self):
        result = _extract_paths({"path": "foo.txt bar.txt"})
        assert "foo.txt" in result

    def test_empty(self):
        assert _extract_paths({}) == []


# ── can_handle ────────────────────────────────────────────────────────────────

class TestCanHandle:
    def test_git_in_tool_name(self):
        assert run(ADP.can_handle("git_push", {}))

    def test_git_commit_tool(self):
        assert run(ADP.can_handle("git_commit", {}))

    def test_git_clone_tool(self):
        assert run(ADP.can_handle("git_clone", {}))

    def test_command_arg_starts_with_git(self):
        assert run(ADP.can_handle("run_command", {"command": "git push origin main"}))

    def test_cmd_arg_starts_with_git(self):
        assert run(ADP.can_handle("exec", {"cmd": "git status"}))

    def test_no_git_tool_no_arg(self):
        assert not run(ADP.can_handle("send_email", {"to": "a@b.com"}))

    def test_unrelated_command_arg(self):
        assert not run(ADP.can_handle("run_command", {"command": "ls -la"}))


# ── dry_run: read-only ops ────────────────────────────────────────────────────

class TestReadOnlyOps:
    def test_git_log_safe(self):
        r = run(ADP.dry_run("git_log", {"command": "log --oneline -5"}, ctx()))
        assert r.safe is True
        assert "git-read-op" in r.signals

    def test_git_status_safe(self):
        r = run(ADP.dry_run("git_status", {}, ctx()))
        assert r.safe is True

    def test_git_diff_safe(self):
        r = run(ADP.dry_run("git_diff", {"command": "diff HEAD"}, ctx()))
        assert r.safe is True

    def test_executed_false_for_read_op(self):
        r = run(ADP.dry_run("git_log", {}, ctx()))
        assert r.executed is False


# ── dry_run: clone ────────────────────────────────────────────────────────────

class TestClone:
    def test_clone_always_safe(self):
        r = run(ADP.dry_run("git_clone", {"url": "https://github.com/foo/bar"}, ctx()))
        assert r.safe is True

    def test_clone_url_in_scope(self):
        r = run(ADP.dry_run("git_clone", {"url": "https://github.com/foo/bar"}, ctx()))
        assert "url" in r.scope_estimate
        assert "github.com" in r.scope_estimate["url"]

    def test_clone_signal(self):
        r = run(ADP.dry_run("git_clone", {"url": "https://example.com/repo"}, ctx()))
        assert any("clone-url" in s for s in r.signals)

    def test_executed_false(self):
        r = run(ADP.dry_run("git_clone", {"url": "https://github.com/foo/bar"}, ctx()))
        assert r.executed is False


# ── dry_run: git rm ───────────────────────────────────────────────────────────

class TestGitRm:
    def test_few_paths_safe(self):
        r = run(ADP.dry_run("git_rm", {"paths": ["a.txt", "b.txt"]}, ctx()))
        assert r.safe is True

    def test_many_paths_unsafe(self):
        paths = [f"file{i}.txt" for i in range(11)]
        r = run(ADP.dry_run("git_rm", {"paths": paths}, ctx()))
        assert r.safe is False
        assert any("mass-rm" in s for s in r.signals)

    def test_path_count_in_scope(self):
        r = run(ADP.dry_run("git_rm", {"paths": ["a.txt", "b.txt", "c.txt"]}, ctx()))
        assert r.scope_estimate["path_count"] == 3


# ── dry_run: git push (mocked subprocess) ────────────────────────────────────

class _FakeProcResult:
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestGitPush:
    def _mock_run_git(self, stdout="Everything up-to-date", stderr="", returncode=0):
        fake = _FakeProcResult(stdout, stderr, returncode)
        return patch(
            "app.security.adapters.git._run_git",
            new=AsyncMock(return_value=fake),
        )

    def test_push_dry_run_ok(self):
        with self._mock_run_git(returncode=0, stdout="Everything up-to-date"):
            r = run(ADP.dry_run("git_push", {}, ctx()))
        assert r.safe is True
        assert "push-dry-run-ok" in r.signals

    def test_push_dry_run_failure(self):
        with self._mock_run_git(returncode=1, stderr="rejected"):
            r = run(ADP.dry_run("git_push", {"command": "push origin main"}, ctx()))
        assert r.safe is False
        assert "push-would-fail" in r.signals

    def test_push_executed_true(self):
        with self._mock_run_git():
            r = run(ADP.dry_run("git_push", {}, ctx()))
        assert r.executed is True

    def test_push_output_in_scope(self):
        with self._mock_run_git(stdout="To github.com:foo/bar.git"):
            r = run(ADP.dry_run("git_push", {}, ctx()))
        assert "dry_run_stdout" in r.scope_estimate

    def test_push_git_unavailable_degrades_gracefully(self):
        with patch(
            "app.security.adapters.git._run_git",
            side_effect=RuntimeError("git is not available on PATH"),
        ):
            r = run(ADP.dry_run("git_push", {}, ctx()))
        # Should degrade gracefully rather than raising
        assert "git-unavailable" in r.signals


# ── dry_run: write ops (commit, merge) with mocked status ────────────────────

class TestWriteOps:
    def _mock_status(self, staged_lines: list[str]):
        output = "\n".join(staged_lines) + "\n"
        fake = _FakeProcResult(stdout=output, stderr="", returncode=0)
        return patch(
            "app.security.adapters.git._run_git",
            new=AsyncMock(return_value=fake),
        )

    def test_commit_small_staged_safe(self):
        staged = ["M  app.py", "A  test.py"]
        with self._mock_status(staged):
            r = run(ADP.dry_run("git_commit", {"command": "commit -m 'fix'"}, ctx()))
        assert r.safe is True
        assert r.scope_estimate["staged_files"] == 2

    def test_commit_mass_staged_unsafe(self):
        staged = [f"M  file{i}.py" for i in range(51)]
        with self._mock_status(staged):
            r = run(ADP.dry_run("git_commit", {}, ctx()))
        assert r.safe is False
        assert any("mass-commit" in s for s in r.signals)

    def test_merge_is_write_op(self):
        with self._mock_status([]):
            r = run(ADP.dry_run("git_merge", {"command": "merge feature-branch"}, ctx()))
        assert any("git-write" in s for s in r.signals)


# ── adapter metadata ──────────────────────────────────────────────────────────

class TestAdapterMeta:
    def test_name(self):
        assert ADP.name == "git"

    def test_adapter_field_in_result(self):
        r = run(ADP.dry_run("git_log", {}, ctx()))
        assert r.adapter == "git"

    def test_latency_positive(self):
        r = run(ADP.dry_run("git_log", {}, ctx()))
        assert r.latency_ms >= 0.0
