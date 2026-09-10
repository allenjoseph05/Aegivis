"""
E2B Code Execution Sandbox Adapter — Phase 16.2.

Runs code in an E2B Firecracker microVM to preview what it would do before
actual execution. Requires E2B_API_KEY env var and e2b_code_interpreter package.

Detection: tool name contains code/script/execute/shell tokens OR args
contain a "code" / "script" / "python" key with non-empty content.

Scope logic:
  - Executes code verbatim in isolated E2B sandbox
  - Reports: exit_code, stdout snippet, files written count
  - safe=False if:
      • exit_code != 0  (error may indicate malicious / unintended outcome)
      • files written > FILE_WRITE_BLOCK threshold (mass write)
  - safe=True (fail-open) when E2B unavailable / API key missing

No regex. Detection is frozenset intersection on lowercased tool-name tokens.
"""
from __future__ import annotations

import logging
import os
import time

from ..sandbox import SandboxAdapter, SandboxContext, SandboxResult
from ._common import tokenize as _tokenize

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

_FILE_WRITE_WARN  = 20    # files written → warning signal
_FILE_WRITE_BLOCK = 100   # files written → unsafe

# ── Token sets ────────────────────────────────────────────────────────────────

_CODE_TOOL_TOKENS: frozenset[str] = frozenset({
    "code", "execute", "run", "python", "script", "eval", "repl",
    "shell", "bash", "exec", "interpreter", "notebook", "kernel",
    "compute", "sandbox",
})

_CODE_ARG_KEYS: frozenset[str] = frozenset({
    "code", "script", "python", "program", "source", "snippet",
    "expression", "cell", "content",
})

_COMMAND_ARG_KEYS: frozenset[str] = frozenset({
    "command", "cmd", "args", "arguments",
})

_CODE_INTERPRETERS: frozenset[str] = frozenset({
    "python", "python3", "node", "nodejs", "ruby", "perl",
    "bash", "sh", "zsh", "fish",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_code(args: dict) -> str | None:
    for key in _CODE_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _looks_like_interpreter_command(val: str) -> bool:
    """True if command arg starts with a known code interpreter binary."""
    first = val.strip().lower().split()[0] if val.strip() else ""
    return _tokenize(first) & _CODE_INTERPRETERS != frozenset()


async def _run_in_e2b(code: str, timeout_ms: int) -> dict:
    """
    Execute code in an E2B Firecracker sandbox and return observation dict.

    Returns:
        {exit_code: int, stdout: str, stderr: str, files_written: int}

    Raises:
        RuntimeError — if E2B_API_KEY missing or package not installed
    """
    api_key = os.getenv("E2B_API_KEY", "")
    if not api_key:
        raise RuntimeError("E2B_API_KEY not set")

    try:
        from e2b_code_interpreter import AsyncSandbox  # type: ignore[import]
    except ImportError:
        raise RuntimeError("e2b_code_interpreter package not installed")

    timeout_s = max(10, int(timeout_ms / 1000) + 10)
    async with AsyncSandbox(api_key=api_key, timeout=timeout_s) as sb:
        result = await sb.run_code(code)
        try:
            files = await sb.files.list("/")
            file_count = len(files)
        except Exception:
            file_count = 0

    stdout = (result.text or "")[:500]
    stderr = ""
    exit_code = 0
    if result.error:
        stderr = str(result.error)[:300]
        exit_code = 1

    return {
        "exit_code":     exit_code,
        "stdout":        stdout,
        "stderr":        stderr,
        "files_written": file_count,
    }


# ── E2BSandboxAdapter ─────────────────────────────────────────────────────────

class E2BSandboxAdapter(SandboxAdapter):
    """
    Sandbox adapter for code-execution tool calls.

    Executes submitted code in an isolated E2B Firecracker VM and reports
    the observed scope (exit code, output snippet, files written).
    """

    name = "e2b"

    async def can_handle(self, tool_name: str, args: dict) -> bool:
        if _tokenize(tool_name) & _CODE_TOOL_TOKENS:
            return True
        if _extract_code(args) is not None:
            return True
        for key in _COMMAND_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and _looks_like_interpreter_command(val):
                return True
        return False

    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        t0 = time.monotonic()

        code = _extract_code(args)
        signals: list[str] = []
        scope: dict = {}
        safe = True
        executed = False
        preview = "e2b: code execution sandbox"

        if not code:
            signals.append("e2b-no-code")
            preview = "No code content found in args"

        else:
            try:
                obs = await _run_in_e2b(code, ctx.timeout_ms)
                executed = True

                scope = {
                    "exit_code":     obs["exit_code"],
                    "stdout":        obs["stdout"][:200],
                    "files_written": obs["files_written"],
                }

                if obs["exit_code"] != 0:
                    signals.append(f"e2b-exit-error:{obs['exit_code']}")
                    safe = False

                fw = obs["files_written"]
                if fw > _FILE_WRITE_BLOCK:
                    signals.append(f"e2b-mass-write:{fw}-files")
                    safe = False
                elif fw > _FILE_WRITE_WARN:
                    signals.append(f"e2b-write-warn:{fw}-files")

                if not signals:
                    signals.append("e2b-ok")

                preview = (
                    f"E2B sandbox: exit={obs['exit_code']}, "
                    f"files={obs['files_written']}, "
                    f"stdout={obs['stdout'][:80]!r}"
                )

            except RuntimeError as exc:
                reason = str(exc)
                if "E2B_API_KEY" in reason:
                    signals.append("e2b-no-api-key")
                elif "not installed" in reason:
                    signals.append("e2b-not-installed")
                else:
                    signals.append("e2b-unavailable")
                scope["error"] = reason
                safe = True   # fail-open
                preview = f"E2B not available: {reason}"

        return SandboxResult(
            adapter="e2b",
            executed=executed,
            safe=safe,
            scope_estimate=scope,
            preview=preview[:500],
            signals=signals or ["e2b-ok"],
            latency_ms=(time.monotonic() - t0) * 1000,
            error=None,
        )
