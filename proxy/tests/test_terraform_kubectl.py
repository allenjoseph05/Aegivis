"""
Unit tests for TerraformPlanAdapter (Phase 16.8) and KubectlDryRunAdapter (Phase 16.8).

Subprocess calls are mocked — no real terraform or kubectl required.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.security.adapters.terraform import (
    TerraformPlanAdapter,
    _parse_plan_json,
    _tokenize as tf_tokenize,
    _extract_tf_command,
)
from app.security.adapters.kubectl import (
    KubectlDryRunAdapter,
    _parse_kubectl_output,
    _tokenize as kube_tokenize,
    _extract_subcommand,
)
from app.security.sandbox import SandboxContext


def ctx(timeout_ms: int = 2000) -> SandboxContext:
    return SandboxContext(session_id="s", agent_id="a", org_id="o", timeout_ms=timeout_ms)


def run(coro):
    return asyncio.run(coro)


TF = TerraformPlanAdapter()
KB = KubectlDryRunAdapter()


# ════════════════════════════════════════════════════════════════════════════
# Terraform helpers
# ════════════════════════════════════════════════════════════════════════════

class TestTfTokenize:
    def test_underscore_split(self):
        assert "terraform" in tf_tokenize("run_terraform")
        assert "plan" in tf_tokenize("run_terraform_plan")

    def test_lowercased(self):
        assert "terraform" in tf_tokenize("TERRAFORM_APPLY")


class TestParsePlanJson:
    def _plan(self, actions_list: list[list[str]]) -> str:
        changes = [
            {"type": f"resource_{i}", "change": {"actions": acts}}
            for i, acts in enumerate(actions_list)
        ]
        return json.dumps({"resource_changes": changes})

    def test_no_changes(self):
        r = _parse_plan_json(json.dumps({"resource_changes": []}))
        assert r["total"] == 0

    def test_creates(self):
        r = _parse_plan_json(self._plan([["create"], ["create"]]))
        assert r["creates"] == 2
        assert r["deletes"] == 0

    def test_deletes(self):
        r = _parse_plan_json(self._plan([["delete"]]))
        assert r["deletes"] == 1

    def test_destroy_counted_as_delete(self):
        r = _parse_plan_json(self._plan([["destroy"]]))
        assert r["deletes"] == 1

    def test_updates(self):
        r = _parse_plan_json(self._plan([["update"]]))
        assert r["updates"] == 1

    def test_no_op_excluded(self):
        r = _parse_plan_json(self._plan([["no-op"]]))
        assert r["total"] == 0

    def test_mixed_plan(self):
        r = _parse_plan_json(self._plan([["create"], ["delete"], ["update"]]))
        assert r["creates"] == 1
        assert r["deletes"] == 1
        assert r["updates"] == 1
        assert r["total"] == 3

    def test_invalid_json_returns_empty(self):
        r = _parse_plan_json("not json at all")
        assert r == {}


class TestExtractTfCommand:
    def test_command_arg(self):
        assert _extract_tf_command("run", {"command": "plan -var foo=bar"}) == "plan -var foo=bar"

    def test_strips_terraform_prefix(self):
        result = _extract_tf_command("run", {"command": "terraform apply"})
        assert result == "apply"

    def test_from_tool_name(self):
        result = _extract_tf_command("terraform_apply", {})
        assert result == "apply"

    def test_default_plan(self):
        assert _extract_tf_command("terraform", {}) == "plan"


# ── TerraformPlanAdapter.can_handle ──────────────────────────────────────────

class TestTfCanHandle:
    def test_terraform_in_tool_name(self):
        assert run(TF.can_handle("run_terraform", {}))

    def test_tf_alias(self):
        assert run(TF.can_handle("tf_plan", {}))

    def test_command_arg_terraform_prefix(self):
        assert run(TF.can_handle("exec", {"command": "terraform plan"}))

    def test_no_match(self):
        assert not run(TF.can_handle("send_email", {"to": "a@b.com"}))

    def test_tofu_tool(self):
        assert run(TF.can_handle("opentofu_apply", {}))


# ── TerraformPlanAdapter.dry_run ─────────────────────────────────────────────

class _FakeProc:
    def __init__(self, stdout="", stderr="", rc=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, rc


class TestTfDryRun:
    def _mock_run(self, responses: list):
        """responses is a list of _FakeProc to return sequentially."""
        call_count = [0]
        async def _fake(*args, **kwargs):
            r = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return r
        return patch("app.security.adapters.terraform._run", new=_fake)

    def test_destroy_command_always_unsafe(self):
        r = run(TF.dry_run("terraform_destroy", {}, ctx()))
        assert r.safe is False
        assert any("destroy" in s for s in r.signals)

    def test_init_command_safe(self):
        r = run(TF.dry_run("terraform_init", {}, ctx()))
        assert r.safe is True

    def test_validate_command_safe(self):
        r = run(TF.dry_run("terraform_validate", {"command": "validate"}, ctx()))
        assert r.safe is True
        assert "tf-safe-op" in r.signals

    def test_apply_without_config_unsafe(self):
        """apply with no config → escalate (can't preview)."""
        r = run(TF.dry_run("terraform_apply", {"command": "apply"}, ctx()))
        assert r.safe is False
        assert any("apply" in s for s in r.signals)

    def test_plan_without_config_no_subprocess(self):
        """plan with no config → no subprocess, structural warning."""
        r = run(TF.dry_run("terraform_plan", {}, ctx()))
        assert "tf-plan-no-config" in r.signals

    def test_plan_with_config_creates_only_safe(self):
        plan_json = json.dumps({"resource_changes": [
            {"type": "aws_s3_bucket", "change": {"actions": ["create"]}}
        ]})
        responses = [
            _FakeProc(),                   # init ok
            _FakeProc(),                   # plan ok
            _FakeProc(stdout=plan_json),   # show ok
        ]
        with self._mock_run(responses):
            r = run(TF.dry_run("terraform_plan", {"config": "resource ... {}"}, ctx()))
        assert r.safe is True
        assert r.scope_estimate.get("creates") == 1

    def test_plan_with_deletes_unsafe(self):
        plan_json = json.dumps({"resource_changes": [
            {"type": "aws_instance", "change": {"actions": ["delete"]}}
        ]})
        responses = [
            _FakeProc(),
            _FakeProc(),
            _FakeProc(stdout=plan_json),
        ]
        with self._mock_run(responses):
            r = run(TF.dry_run("terraform_plan", {"config": "resource... {}"}, ctx()))
        assert r.safe is False
        assert any("deletes" in s for s in r.signals)

    def test_init_failure_fail_open(self):
        responses = [_FakeProc(stderr="no providers", rc=1)]
        with self._mock_run(responses):
            r = run(TF.dry_run("terraform_plan", {"config": "bad config"}, ctx()))
        assert r.safe is True  # fail-open
        assert "tf-init-failed" in r.signals

    def test_terraform_unavailable_fail_open(self):
        async def _raise(*a, **kw):
            raise RuntimeError("terraform is not available on PATH")
        with patch("app.security.adapters.terraform._run", new=_raise):
            r = run(TF.dry_run("terraform_plan", {"config": "resource... {}"}, ctx()))
        assert r.safe is True
        assert "terraform-unavailable" in r.signals

    def test_adapter_name(self):
        r = run(TF.dry_run("terraform_init", {}, ctx()))
        assert r.adapter == "terraform"


# ════════════════════════════════════════════════════════════════════════════
# kubectl helpers
# ════════════════════════════════════════════════════════════════════════════

class TestKubeTokenize:
    def test_kubectl_in_name(self):
        assert "kubectl" in kube_tokenize("kubectl_apply")

    def test_k8s_alias(self):
        assert "k8s" in kube_tokenize("k8s_delete")


class TestParseKubectlOutput:
    def test_created(self):
        r = _parse_kubectl_output("deployment.apps/nginx created (dry run)")
        assert r["created"] == 1

    def test_configured(self):
        r = _parse_kubectl_output("deployment.apps/nginx configured (dry run)")
        assert r["configured"] == 1

    def test_deleted(self):
        r = _parse_kubectl_output("pod/old-pod deleted (dry run)")
        assert r["deleted"] == 1

    def test_multiple_lines(self):
        output = (
            "deployment.apps/web created (dry run)\n"
            "service/my-svc configured (dry run)\n"
            "pod/stale deleted (dry run)\n"
        )
        r = _parse_kubectl_output(output)
        assert r["created"] == 1
        assert r["configured"] == 1
        assert r["deleted"] == 1

    def test_no_dry_run_not_counted(self):
        """Lines without 'dry run' are NOT counted."""
        r = _parse_kubectl_output("deployment.apps/web created")
        assert r["created"] == 0

    def test_empty_output(self):
        r = _parse_kubectl_output("")
        assert all(v == 0 for v in r.values())


class TestExtractSubcommand:
    def test_command_arg(self):
        assert _extract_subcommand("run", {"command": "apply -f manifest.yaml"}) == "apply"

    def test_kubectl_prefix_stripped(self):
        result = _extract_subcommand("run", {"command": "kubectl delete pod/old"})
        assert result == "delete"

    def test_tool_name_apply(self):
        assert _extract_subcommand("kubectl_apply", {}) == "apply"

    def test_tool_name_delete(self):
        assert _extract_subcommand("kubectl_delete", {}) == "delete"

    def test_default_apply(self):
        # No subcommand info → default "apply"
        result = _extract_subcommand("kubectl", {})
        assert result == "apply"


# ── KubectlDryRunAdapter.can_handle ──────────────────────────────────────────

class TestKubeCanHandle:
    def test_kubectl_in_tool_name(self):
        assert run(KB.can_handle("kubectl_apply", {}))

    def test_k8s_tool(self):
        assert run(KB.can_handle("k8s_delete", {}))

    def test_command_arg_kubectl(self):
        assert run(KB.can_handle("exec", {"command": "kubectl apply -f file.yaml"}))

    def test_no_match(self):
        assert not run(KB.can_handle("send_email", {}))

    def test_unrelated_command(self):
        assert not run(KB.can_handle("exec", {"command": "docker run ..."}))


# ── KubectlDryRunAdapter.dry_run ─────────────────────────────────────────────

class TestKubeDryRun:
    def _mock_kubectl(self, stdout="", stderr="", rc=0):
        fake = _FakeProc(stdout, stderr, rc)
        return patch(
            "app.security.adapters.kubectl._run_kubectl",
            new=AsyncMock(return_value=fake),
        )

    def test_read_op_safe_no_subprocess(self):
        r = run(KB.dry_run("kubectl_get", {}, ctx()))
        assert r.safe is True
        assert "kubectl-read-op" in r.signals

    def test_apply_no_manifest_warns(self):
        r = run(KB.dry_run("kubectl_apply", {}, ctx()))
        assert "kubectl-apply-no-manifest" in r.signals

    def test_apply_manifest_creates_safe(self):
        output = "deployment.apps/web created (dry run)\n"
        with self._mock_kubectl(stdout=output, rc=0):
            r = run(KB.dry_run("kubectl_apply", {"manifest": "apiVersion: ..."}, ctx()))
        assert r.safe is True
        assert any("creates" in s for s in r.signals) or "kubectl-apply-ok" in r.signals

    def test_apply_manifest_with_delete_unsafe(self):
        output = "pod/old deleted (dry run)\n"
        with self._mock_kubectl(stdout=output, rc=0):
            r = run(KB.dry_run("kubectl_apply", {"manifest": "apiVersion: ..."}, ctx()))
        assert r.safe is False
        assert any("deletes" in s for s in r.signals)

    def test_apply_error_fail_open(self):
        with self._mock_kubectl(stderr="error: invalid resource", rc=1):
            r = run(KB.dry_run("kubectl_apply", {"manifest": "bad yaml"}, ctx()))
        assert r.safe is True
        assert "kubectl-apply-error" in r.signals

    def test_delete_no_resources(self):
        r = run(KB.dry_run("kubectl_delete", {}, ctx()))
        assert "kubectl-delete-no-resources" in r.signals

    def test_delete_with_resources_unsafe(self):
        output = "pod/old-pod deleted (dry run)\n"
        with self._mock_kubectl(stdout=output, rc=0):
            r = run(KB.dry_run("kubectl_delete", {"resource": "pod/old-pod"}, ctx()))
        assert r.safe is False
        assert any("delete" in s for s in r.signals)

    def test_kubectl_unavailable_fail_open_apply(self):
        async def _raise(*a, **kw):
            raise RuntimeError("kubectl is not available on PATH")
        with patch("app.security.adapters.kubectl._run_kubectl", new=_raise):
            r = run(KB.dry_run("kubectl_apply", {"manifest": "spec: ..."}, ctx()))
        assert r.safe is True
        assert "kubectl-unavailable" in r.signals

    def test_kubectl_unavailable_fail_open_delete(self):
        async def _raise(*a, **kw):
            raise RuntimeError("kubectl is not available on PATH")
        with patch("app.security.adapters.kubectl._run_kubectl", new=_raise):
            r = run(KB.dry_run("kubectl_delete", {"resource": "pod/x"}, ctx()))
        assert r.safe is True
        assert "kubectl-unavailable" in r.signals

    def test_patch_is_write_op(self):
        r = run(KB.dry_run("kubectl_patch", {"command": "patch deployment/web ..."}, ctx()))
        assert any("write" in s for s in r.signals)

    def test_scale_is_write_op(self):
        r = run(KB.dry_run("kubectl_scale", {}, ctx()))
        assert any("write" in s for s in r.signals)

    def test_adapter_name(self):
        r = run(KB.dry_run("kubectl_get", {}, ctx()))
        assert r.adapter == "kubectl"

    def test_executed_true_for_apply_with_manifest(self):
        with self._mock_kubectl(stdout="svc created (dry run)", rc=0):
            r = run(KB.dry_run("kubectl_apply", {"manifest": "apiVersion: ..."}, ctx()))
        assert r.executed is True

    def test_executed_false_for_read_op(self):
        r = run(KB.dry_run("kubectl_get", {}, ctx()))
        assert r.executed is False
