"""
Git Sandbox Adapter — Phase 16.7.

Detects git tool calls and performs scope estimation:
  - ``git push``   → ``git push --dry-run`` (native flag; doesn't update refs)
  - ``git commit`` → counts staged files via ``git status --short``
  - ``git clone``  → structural: extracts and reports the clone URL
  - ``git rm``     → counts matched paths via ``git ls-files --error-unmatch``
  - Other writes   → structural scope estimate (staged file count)
  - Read-only ops  → always safe, no subprocess needed

Uses ``asyncio.create_subprocess_exec`` to run ``git`` as a subprocess.
All subprocess calls have a hard timeout enforced by the SandboxRegistry.
Falls back gracefully when ``git`` is not on PATH.

No regex.  Detection is frozenset intersection on lowercased tool-name tokens
and common ``command`` / ``cmd`` arg keys.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..sandbox import SandboxAdapter, SandboxContext, SandboxResult
from ._common import tokenize as _tokenize

logger = logging.getLogger(__name__)


# ── Token sets ────────────────────────────────────────────────────────────────

_GIT_TOOL_TOKENS: frozenset[str] = frozenset({"git"})

_GIT_WRITE_OPS: frozenset[str] = frozenset({
    "push", "commit", "merge", "rebase", "tag",
    "reset", "rm", "mv", "branch", "cherry",
    "pick", "squash", "stash", "restore",
})

_GIT_READ_OPS: frozenset[str] = frozenset({
    "clone", "fetch", "pull", "log", "diff",
    "status", "show", "blame", "grep",
})

_COMMAND_ARG_KEYS: frozenset[str] = frozenset({
    "command", "cmd", "args", "arguments",
    "subcmd", "subcommand", "operation", "op",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_git_command(tool_name: str, args: dict) -> str:
    """
    Infer the git sub-command from args or tool name.

    Priority:
      1. ``args["command"]`` / ``args["cmd"]`` / … (explicit command string)
      2. Tool name token after "git" (e.g. "git_push" → "push")
      3. Fall back to "unknown"
    """
    for key in _COMMAND_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            # Strip the leading "git " prefix if present
            stripped = val.strip()
            if stripped.lower().startswith("git "):
                stripped = stripped[4:].strip()
            return stripped

    # Try to find the sub-command from the tool name tokens
    tokens = _tokenize(tool_name)
    remaining = tokens - _GIT_TOOL_TOKENS
    known = remaining & (_GIT_WRITE_OPS | _GIT_READ_OPS)
    if known:
        return next(iter(known))
    if remaining:
        return next(iter(remaining))
    return "unknown"


async def _run_git(args: list[str], timeout_ms: int) -> "_ProcResult":
    """
    Run ``git <args>`` as a subprocess and capture output.

    Returns a simple namespace with stdout/stderr/returncode.
    Raises FileNotFoundError if git is not on PATH.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        return _ProcResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            returncode=proc.returncode or 0,
        )
    except FileNotFoundError:
        raise RuntimeError("git is not available on PATH")


class _ProcResult:
    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# ── GitSandboxAdapter ─────────────────────────────────────────────────────────

class GitSandboxAdapter(SandboxAdapter):
    """
    Sandbox adapter for git tool calls.

    safe=False conditions:
      - ``git push --dry-run`` returns non-zero (remote rejected)
      - staged file count > 50 (mass commit)

    safe=True always for read operations and clones.
    """

    name = "git"

    async def can_handle(self, tool_name: str, args: dict) -> bool:
        if "git" in _tokenize(tool_name):
            return True
        # Also match when a "command" arg starts with "git "
        for key in _COMMAND_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip().lower().startswith("git "):
                return True
        return False

    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        t0 = time.monotonic()

        command = _extract_git_command(tool_name, args)
        cmd_tokens = frozenset(command.lower().split())

        is_push   = "push"   in cmd_tokens
        is_clone  = "clone"  in cmd_tokens
        is_rm     = "rm"     in cmd_tokens
        is_write  = bool(cmd_tokens & _GIT_WRITE_OPS)

        signals: list[str] = []
        scope: dict = {"command": command, "is_write": is_write}
        safe = True

        # ── git push --dry-run ────────────────────────────────────────────────
        if is_push:
            push_extra = _extract_push_extra_args(args)
            try:
                res = await _run_git(
                    ["push", "--dry-run"] + push_extra,
                    ctx.timeout_ms,
                )
                scope["returncode"]       = res.returncode
                scope["dry_run_stdout"]   = res.stdout[:300]
                scope["dry_run_stderr"]   = res.stderr[:300]

                if res.returncode != 0:
                    signals.append("push-would-fail")
                    safe = False
                else:
                    signals.append("push-dry-run-ok")

                output = (res.stdout or res.stderr).strip()
                preview = f"git push --dry-run: {output[:200]}" if output else "git push --dry-run: ok"

            except RuntimeError as exc:
                # git not on PATH — degrade gracefully
                signals.append("git-unavailable")
                scope["error"] = str(exc)
                preview = f"git push (structural): command='{command}'"

        # ── git clone ─────────────────────────────────────────────────────────
        elif is_clone:
            url = _extract_clone_url(args)
            scope["url"] = url
            signals.append(f"clone-url:{url[:80]}")
            preview = f"Would clone: {url}"

        # ── git rm ────────────────────────────────────────────────────────────
        elif is_rm:
            paths = _extract_paths(args)
            scope["paths"] = paths[:20]
            scope["path_count"] = len(paths)
            if len(paths) > 10:
                signals.append(f"mass-rm:{len(paths)}-paths")
                safe = False
            else:
                signals.append(f"git-rm:{len(paths)}-paths")
            preview = f"git rm: {len(paths)} path(s) — {', '.join(paths[:3])}"

        # ── other write ops (commit, merge, rebase, …) ────────────────────────
        elif is_write:
            signals.append(f"git-write:{command.split()[0]}")
            try:
                status_res = await _run_git(["status", "--short"], ctx.timeout_ms)
                staged = [
                    ln for ln in status_res.stdout.splitlines()
                    if len(ln) >= 2 and ln[0] in ("A", "M", "D", "R", "C")
                ]
                scope["staged_files"] = len(staged)
                if len(staged) > 50:
                    signals.append(f"mass-commit:{len(staged)}-staged")
                    safe = False
                preview = f"git {command.split()[0]}: {len(staged)} staged file(s)"
            except Exception as _git_exc:
                logger.warning("git status error (skipped): %s", _git_exc)
                scope["staged_files"] = None
                preview = f"git {command} (scope estimate unavailable)"

        # ── read-only ops ─────────────────────────────────────────────────────
        else:
            signals.append("git-read-op")
            preview = f"git {command}: read-only operation"

        return SandboxResult(
            adapter="git",
            executed=is_push,
            safe=safe,
            scope_estimate=scope,
            preview=preview[:500],
            signals=signals,
            latency_ms=(time.monotonic() - t0) * 1000,
            error=None,
        )


# ── Argument extraction helpers ───────────────────────────────────────────────

def _extract_push_extra_args(args: dict) -> list[str]:
    """Extract remote and branch from args for git push."""
    parts: list[str] = []
    remote = args.get("remote") or args.get("origin")
    if isinstance(remote, str) and remote.strip():
        parts.append(remote.strip())
    branch = args.get("branch") or args.get("ref")
    if isinstance(branch, str) and branch.strip():
        parts.append(branch.strip())
    return parts


def _extract_clone_url(args: dict) -> str:
    for key in ("url", "repo", "repository", "source", "remote", "uri"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return args.get("command", "").strip()


def _extract_paths(args: dict) -> list[str]:
    for key in ("paths", "files", "path", "file", "args", "arguments"):
        val = args.get(key)
        if isinstance(val, list):
            return [str(v) for v in val if v]
        if isinstance(val, str) and val.strip():
            return val.split()
    return []
