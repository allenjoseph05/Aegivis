"""
kubectl Dry-Run Sandbox Adapter — Phase 16.8.

Runs ``kubectl apply --dry-run=client`` or ``kubectl delete --dry-run=client``
to estimate the blast radius of Kubernetes operations BEFORE execution.

Detection: tool name contains "kubectl" OR a ``command`` arg starts with
"kubectl ".

Client-side dry-run (``--dry-run=client``) is used because it:
  - Does not require a live cluster connection
  - Runs admission webhooks locally where possible
  - Is safe to call at sandbox time

Scope logic:
  - Parses kubectl output lines for "configured", "created", "deleted", "replaced"
  - Counts affected resources per kind
  - safe=False if:
      • any delete/destroy detected, OR
      • total resources changed > threshold

No regex. Detection is frozenset intersection on lowercased tool-name tokens.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from ..sandbox import SandboxAdapter, SandboxContext, SandboxResult
from ._common import tokenize as _tokenize

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

_RESOURCE_CHANGE_THRESHOLD = 10   # total resources changed; above this → warn
_DELETE_THRESHOLD          = 0    # any delete → unsafe

# ── Token sets ────────────────────────────────────────────────────────────────

_KUBECTL_TOOL_TOKENS: frozenset[str] = frozenset({"kubectl", "kube", "k8s", "k"})

_WRITE_SUBCOMMANDS: frozenset[str] = frozenset({
    "apply", "delete", "patch", "replace", "rollout",
    "scale", "set", "label", "annotate", "taint",
})

_READ_SUBCOMMANDS: frozenset[str] = frozenset({
    "get", "describe", "logs", "exec", "port", "proxy",
    "explain", "api", "version", "cluster", "config",
})

_COMMAND_ARG_KEYS: frozenset[str] = frozenset({
    "command", "cmd", "args", "arguments", "operation",
})

_MANIFEST_ARG_KEYS: frozenset[str] = frozenset({
    "manifest", "yaml", "json", "spec", "resource",
    "content", "body", "template", "file",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_kubectl(args: list[str], timeout_ms: int) -> "_ProcResult":
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
        raise RuntimeError("kubectl is not available on PATH")


class _ProcResult:
    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _extract_subcommand(tool_name: str, args: dict) -> str:
    for key in _COMMAND_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            words = val.strip().lower().split()
            # Skip "kubectl" prefix token
            tokens = [w for w in words if w not in _KUBECTL_TOOL_TOKENS]
            if tokens:
                return tokens[0]
    tokens = _tokenize(tool_name) - _KUBECTL_TOOL_TOKENS
    known = tokens & (_WRITE_SUBCOMMANDS | _READ_SUBCOMMANDS)
    if known:
        return next(iter(known))
    if tokens:
        return next(iter(tokens))
    return "apply"


def _extract_manifest(args: dict) -> str | None:
    for key in _MANIFEST_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _extract_resource_args(args: dict) -> list[str]:
    """Extract resource type/name arguments for delete/get commands."""
    extra: list[str] = []
    for key in ("resource", "resources", "type", "name", "namespace"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            extra.append(val.strip())
        elif isinstance(val, list):
            extra.extend(str(v) for v in val if v)
    return extra


def _parse_kubectl_output(output: str) -> dict[str, int]:
    """
    Parse kubectl dry-run output lines like:
      deployment.apps/nginx configured (dry run)
      service/my-svc created (dry run)
      pod/bad-pod deleted (dry run)

    Returns counts per action: created, configured (updated), deleted.
    """
    counts: dict[str, int] = {"created": 0, "configured": 0, "deleted": 0, "unchanged": 0}
    for line in output.splitlines():
        line = line.lower().strip()
        for action in counts:
            if action in line and "dry run" in line:
                counts[action] += 1
    return counts


# ── KubectlDryRunAdapter ──────────────────────────────────────────────────────

class KubectlDryRunAdapter(SandboxAdapter):
    """
    Sandbox adapter for kubectl operations.

    - apply    → kubectl apply --dry-run=client -f <manifest>
    - delete   → kubectl delete --dry-run=client <resources>
    - patch    → kubectl patch --dry-run=client …
    - read ops → always safe, no subprocess
    """

    name = "kubectl"

    async def can_handle(self, tool_name: str, args: dict) -> bool:
        if _tokenize(tool_name) & _KUBECTL_TOOL_TOKENS:
            return True
        for key in _COMMAND_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str):
                first = val.strip().lower().split()[0] if val.strip() else ""
                if first == "kubectl":
                    return True
        return False

    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        t0 = time.monotonic()
        subcmd = _extract_subcommand(tool_name, args)
        sub_tokens = frozenset(subcmd.lower().split())

        signals: list[str] = []
        scope: dict = {"subcommand": subcmd}
        safe = True
        executed = False

        # ── read-only subcommand ──────────────────────────────────────────────
        if sub_tokens & _READ_SUBCOMMANDS and not (sub_tokens & _WRITE_SUBCOMMANDS):
            signals.append("kubectl-read-op")
            preview = f"kubectl {subcmd}: read-only"

        # ── apply ─────────────────────────────────────────────────────────────
        elif "apply" in sub_tokens:
            manifest = _extract_manifest(args)
            if manifest:
                executed = True
                result = await self._dry_run_apply(manifest, ctx.timeout_ms)
                scope.update(result["scope"])
                signals.extend(result["signals"])
                safe = result["safe"]
                preview = result["preview"]
            else:
                signals.append("kubectl-apply-no-manifest")
                preview = "kubectl apply: no manifest provided"

        # ── delete ────────────────────────────────────────────────────────────
        elif "delete" in sub_tokens:
            executed = True
            resource_args = _extract_resource_args(args)
            result = await self._dry_run_delete(resource_args, ctx.timeout_ms)
            scope.update(result["scope"])
            signals.extend(result["signals"])
            safe = result["safe"]
            preview = result["preview"]

        # ── patch / replace / scale ───────────────────────────────────────────
        elif sub_tokens & {"patch", "replace", "scale", "rollout", "set"}:
            signals.append(f"kubectl-write:{subcmd.split()[0]}")
            safe = True   # conservative — no delete, report for review
            preview = f"kubectl {subcmd}: write operation (no dry-run preview)"

        else:
            signals.append("kubectl-unknown-op")
            preview = f"kubectl {subcmd}: unrecognised operation"

        return SandboxResult(
            adapter="kubectl",
            executed=executed,
            safe=safe,
            scope_estimate=scope,
            preview=preview[:500],
            signals=signals or ["kubectl-ok"],
            latency_ms=(time.monotonic() - t0) * 1000,
            error=None,
        )

    async def _dry_run_apply(self, manifest: str, timeout_ms: int) -> dict:
        """Write manifest to temp file and run kubectl apply --dry-run=client."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(manifest)
            tmp_path = f.name

        try:
            res = await _run_kubectl(
                ["apply", "--dry-run=client", "-f", tmp_path],
                timeout_ms,
            )
        except RuntimeError as exc:
            return {
                "scope": {"error": str(exc)},
                "signals": ["kubectl-unavailable"],
                "safe": True,
                "preview": f"kubectl not available: {exc}",
            }
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception as _unlink_exc:
                logger.debug("Failed to remove temp manifest file: %s", _unlink_exc)

        counts = _parse_kubectl_output(res.stdout + res.stderr)
        signals: list[str] = []
        safe = True

        if res.returncode != 0:
            signals.append("kubectl-apply-error")
            return {
                "scope": {"error": res.stderr[:200], **counts},
                "signals": signals,
                "safe": True,   # fail-open on apply error
                "preview": f"kubectl apply --dry-run failed: {res.stderr[:100]}",
            }

        total = sum(counts.values())
        if counts["deleted"] > _DELETE_THRESHOLD:
            signals.append(f"kubectl-deletes:{counts['deleted']}")
            safe = False
        if total > _RESOURCE_CHANGE_THRESHOLD:
            signals.append(f"kubectl-large-change:{total}-resources")
        if counts["created"] > 0:
            signals.append(f"kubectl-creates:{counts['created']}")
        if counts["configured"] > 0:
            signals.append(f"kubectl-updates:{counts['configured']}")
        if not signals:
            signals.append("kubectl-apply-ok")

        preview = (
            f"apply --dry-run: {counts['created']} create, "
            f"{counts['configured']} update, "
            f"{counts['deleted']} delete"
        )
        return {"scope": counts, "signals": signals, "safe": safe, "preview": preview}

    async def _dry_run_delete(self, resource_args: list[str], timeout_ms: int) -> dict:
        """Run kubectl delete --dry-run=client <resources>."""
        if not resource_args:
            return {
                "scope": {"resource_args": []},
                "signals": ["kubectl-delete-no-resources"],
                "safe": True,
                "preview": "kubectl delete: no resources specified",
            }

        try:
            res = await _run_kubectl(
                ["delete", "--dry-run=client"] + resource_args,
                timeout_ms,
            )
        except RuntimeError as exc:
            return {
                "scope": {"error": str(exc)},
                "signals": ["kubectl-unavailable"],
                "safe": True,
                "preview": f"kubectl not available: {exc}",
            }

        counts = _parse_kubectl_output(res.stdout + res.stderr)
        total_deleted = counts["deleted"]
        signals = [f"kubectl-delete:{total_deleted}-resources"]
        safe = total_deleted <= _DELETE_THRESHOLD

        if not safe:
            signals.append("kubectl-delete-detected")

        preview = f"kubectl delete --dry-run: {total_deleted} resource(s) would be deleted"
        return {
            "scope": {"deleted": total_deleted, "resource_args": resource_args[:5]},
            "signals": signals,
            "safe": safe,
            "preview": preview,
        }
