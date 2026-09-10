"""
Terraform Plan Sandbox Adapter — Phase 16.8.

Runs ``terraform plan`` in a temp directory to determine the blast radius of
infrastructure changes BEFORE the agent actually applies them.

Detection: tool name contains "terraform" OR a ``command`` arg starts with
"terraform ".

Scope logic:
  - Parses ``terraform show -json`` output for resource_changes.
  - Counts creates / updates / deletes per resource type.
  - safe=False if:
      • any "delete" or "destroy" action detected, OR
      • total resources changed > RESOURCE_CHANGE_THRESHOLD

Subprocess calls use asyncio.create_subprocess_exec with the registry timeout.
Falls back gracefully when ``terraform`` is not on PATH.

No regex. Detection is frozenset intersection on tool-name tokens.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from ..sandbox import SandboxAdapter, SandboxContext, SandboxResult
from ._common import tokenize as _tokenize

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

_RESOURCE_CHANGE_THRESHOLD = 20   # total resources changed; above this → warn
_DESTROY_THRESHOLD         = 0    # any destroy → unsafe (strict)

# ── Token sets ────────────────────────────────────────────────────────────────

_TF_TOOL_TOKENS: frozenset[str] = frozenset({
    "terraform", "tf", "tofu", "opentofu",
})

_COMMAND_ARG_KEYS: frozenset[str] = frozenset({
    "command", "cmd", "args", "arguments", "operation",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run(args: list[str], timeout_ms: int, cwd: str | None = None) -> "_ProcResult":
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_ms / 1000.0
        )
        return _ProcResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            returncode=proc.returncode or 0,
        )
    except FileNotFoundError:
        raise RuntimeError("terraform is not available on PATH")


class _ProcResult:
    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _extract_config(args: dict) -> str | None:
    """Return HCL/JSON config string from args if present."""
    for key in ("config", "hcl", "plan", "content", "body", "template"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _extract_tf_command(tool_name: str, args: dict) -> str:
    for key in _COMMAND_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            s = val.strip()
            for prefix in ("terraform ", "tf "):
                if s.lower().startswith(prefix):
                    s = s[len(prefix):].strip()
            return s
    tokens = _tokenize(tool_name) - _TF_TOOL_TOKENS
    if tokens:
        return next(iter(tokens))
    return "plan"


def _parse_plan_json(plan_json: str) -> dict:
    """Parse terraform show -json output → change summary."""
    try:
        data = json.loads(plan_json)
    except json.JSONDecodeError:
        return {}

    changes: dict[str, list[str]] = {}   # resource_type → list of actions
    for rc in data.get("resource_changes", []):
        rtype = rc.get("type", "unknown")
        actions = rc.get("change", {}).get("actions", [])
        if actions and actions != ["no-op"]:
            changes.setdefault(rtype, []).extend(actions)

    creates  = sum(1 for acts in changes.values() for a in acts if a == "create")
    updates  = sum(1 for acts in changes.values() for a in acts if a == "update")
    deletes  = sum(1 for acts in changes.values() for a in acts if a in {"delete", "destroy"})
    replaces = sum(1 for acts in changes.values() for a in acts if a == "replace")

    return {
        "resource_types_changed": list(changes.keys()),
        "creates":  creates,
        "updates":  updates,
        "deletes":  deletes,
        "replaces": replaces,
        "total":    creates + updates + deletes + replaces,
    }


# ── TerraformPlanAdapter ──────────────────────────────────────────────────────

class TerraformPlanAdapter(SandboxAdapter):
    """
    Sandbox adapter for Terraform/OpenTofu tool calls.

    When a ``config`` / ``hcl`` arg is present the adapter writes it to a
    temp file, runs ``terraform init -input=false`` then
    ``terraform plan -out=tfplan -input=false``, and finally
    ``terraform show -json tfplan`` to get a machine-readable change set.

    When no config is available the adapter runs a structural analysis on the
    command string alone and returns a conservative estimate.
    """

    name = "terraform"

    async def can_handle(self, tool_name: str, args: dict) -> bool:
        if _tokenize(tool_name) & _TF_TOOL_TOKENS:
            return True
        for key in _COMMAND_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str):
                first = val.strip().lower().split()[0] if val.strip() else ""
                if first in {"terraform", "tf", "tofu"}:
                    return True
        return False

    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        t0 = time.monotonic()
        command = _extract_tf_command(tool_name, args)
        cmd_tokens = frozenset(command.lower().split())

        signals: list[str] = []
        scope: dict = {"command": command}
        safe = True

        # ── terraform plan with actual config ─────────────────────────────────
        config = _extract_config(args)
        if "plan" in cmd_tokens or "apply" in cmd_tokens:
            if config:
                result = await self._run_plan(config, ctx.timeout_ms)
                scope.update(result["scope"])
                signals.extend(result["signals"])
                safe = result["safe"]
                preview = result["preview"]
            else:
                # No config available — structural estimate only
                signals.append("tf-plan-no-config")
                if "apply" in cmd_tokens:
                    signals.append("tf-apply-detected")
                    safe = False   # apply without dry-run preview → escalate
                preview = f"terraform {command}: no config provided for preview"

        elif "destroy" in cmd_tokens:
            signals.append("tf-destroy-detected")
            safe = False
            scope["destroy"] = True
            preview = "terraform destroy: all resources will be deleted"

        else:
            # init, validate, fmt, output, etc. — always safe
            signals.append("tf-safe-op")
            preview = f"terraform {command}: safe operation"

        return SandboxResult(
            adapter="terraform",
            executed=bool(config and "plan" in cmd_tokens),
            safe=safe,
            scope_estimate=scope,
            preview=preview[:500],
            signals=signals or ["tf-ok"],
            latency_ms=(time.monotonic() - t0) * 1000,
            error=None,
        )

    async def _run_plan(self, config: str, timeout_ms: int) -> dict:
        """Write config to temp dir, run terraform plan, parse output."""
        with tempfile.TemporaryDirectory(prefix="aegivis_tf_") as tmpdir:
            main_tf = Path(tmpdir) / "main.tf"
            main_tf.write_text(config, encoding="utf-8")
            plan_file = Path(tmpdir) / "tfplan"

            try:
                # init (fast — local providers only; network providers → timeout)
                init_res = await _run(
                    ["terraform", "init", "-input=false", "-backend=false",
                     "-no-color"],
                    timeout_ms, cwd=tmpdir,
                )
                if init_res.returncode != 0:
                    return {
                        "scope": {"init_error": init_res.stderr[:200]},
                        "signals": ["tf-init-failed"],
                        "safe": True,   # fail-open
                        "preview": f"terraform init failed: {init_res.stderr[:100]}",
                    }

                # plan
                plan_res = await _run(
                    ["terraform", "plan", "-input=false", "-no-color",
                     f"-out={plan_file}"],
                    timeout_ms, cwd=tmpdir,
                )
                if plan_res.returncode != 0:
                    return {
                        "scope": {"plan_error": plan_res.stderr[:200]},
                        "signals": ["tf-plan-failed"],
                        "safe": True,
                        "preview": f"terraform plan failed: {plan_res.stderr[:100]}",
                    }

                # show as JSON
                show_res = await _run(
                    ["terraform", "show", "-json", str(plan_file)],
                    timeout_ms, cwd=tmpdir,
                )
                parsed = _parse_plan_json(show_res.stdout)

            except RuntimeError as exc:
                return {
                    "scope": {"error": str(exc)},
                    "signals": ["terraform-unavailable"],
                    "safe": True,
                    "preview": f"terraform not available: {exc}",
                }

        signals: list[str] = []
        safe = True

        if parsed.get("deletes", 0) > _DESTROY_THRESHOLD:
            signals.append(f"tf-deletes:{parsed['deletes']}-resources")
            safe = False
        if parsed.get("replaces", 0) > 0:
            signals.append(f"tf-replaces:{parsed['replaces']}-resources")
            safe = False
        if parsed.get("total", 0) > _RESOURCE_CHANGE_THRESHOLD:
            signals.append(f"tf-large-change:{parsed['total']}-resources")
        if parsed.get("creates", 0) > 0:
            signals.append(f"tf-creates:{parsed['creates']}")
        if parsed.get("updates", 0) > 0:
            signals.append(f"tf-updates:{parsed['updates']}")

        if not signals:
            signals.append("tf-plan-ok")

        types_str = ", ".join(parsed.get("resource_types_changed", [])[:4])
        preview = (
            f"Plan: {parsed.get('creates',0)} add, "
            f"{parsed.get('updates',0)} change, "
            f"{parsed.get('deletes',0)} destroy"
        )
        if types_str:
            preview += f" — {types_str}"

        return {"scope": parsed, "signals": signals, "safe": safe, "preview": preview}
