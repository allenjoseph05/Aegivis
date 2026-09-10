"""
Unit tests for AWSDryRunAdapter (Phase 16.3).

boto3 calls are mocked — no real AWS credentials or network required.
_ec2_dry_run and _iam_simulate are patched at the module level.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.security.adapters.aws_adapter import (
    AWSDryRunAdapter,
    _extract_operation,
    _extract_service,
    _is_destructive,
    _tokenize,
)
from app.security.sandbox import SandboxContext


def ctx(timeout_ms: int = 3000) -> SandboxContext:
    return SandboxContext(session_id="s", agent_id="a", org_id="o", timeout_ms=timeout_ms)


def run(coro):
    return asyncio.run(coro)


ADP = AWSDryRunAdapter()

# ── Mock return values ────────────────────────────────────────────────────────

_DRY_RUN_OK    = {"would_succeed": True,  "authorized": True,  "error_code": "DryRunOperation",     "error_message": "ok"}
_DRY_RUN_UNAUTH= {"would_succeed": False, "authorized": False, "error_code": "UnauthorizedOperation","error_message": "denied"}
_DRY_RUN_ERR   = {"would_succeed": None,  "authorized": None,  "error_code": "InvalidParameterValue","error_message": "bad"}

_IAM_ALLOWED   = {"allowed": True,  "decision": "allowed",    "caller_arn": "arn:aws:iam::123:user/bot"}
_IAM_DENIED    = {"allowed": False, "decision": "explicitDeny","caller_arn": "arn:aws:iam::123:user/bot"}
_IAM_UNKNOWN   = {"allowed": None,  "decision": "unknown",     "caller_arn": None}


def _mock_ec2(rv: dict):
    return patch("app.security.adapters.aws_adapter._ec2_dry_run", new=AsyncMock(return_value=rv))


def _mock_iam(rv: dict):
    return patch("app.security.adapters.aws_adapter._iam_simulate", new=AsyncMock(return_value=rv))


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_split_underscore(self):
        assert "terminate" in _tokenize("terminate_instances")

    def test_lowercased(self):
        assert "ec2" in _tokenize("EC2_CLIENT")


# ── _is_destructive ───────────────────────────────────────────────────────────

class TestIsDestructive:
    def test_delete_is_destructive(self):
        assert _is_destructive("delete_objects")

    def test_terminate_is_destructive(self):
        assert _is_destructive("terminate_instances")

    def test_destroy_is_destructive(self):
        assert _is_destructive("destroy_cluster")

    def test_describe_not_destructive(self):
        assert not _is_destructive("describe_instances")

    def test_list_not_destructive(self):
        assert not _is_destructive("list_buckets")

    def test_get_not_destructive(self):
        assert not _is_destructive("get_object")


# ── _extract_service ──────────────────────────────────────────────────────────

class TestExtractService:
    def test_ec2_from_tool_name(self):
        assert _extract_service("ec2_terminate", {}) == "ec2"

    def test_s3_from_tool_name(self):
        assert _extract_service("s3_delete_object", {}) == "s3"

    def test_lambda_from_args(self):
        svc = _extract_service("aws_call", {"service": "lambda"})
        assert svc == "lambda"

    def test_rds_from_tool_name(self):
        assert _extract_service("rds_delete_db", {}) == "rds"


# ── _extract_operation ────────────────────────────────────────────────────────

class TestExtractOperation:
    def test_command_arg(self):
        assert _extract_operation("run", {"command": "terminate_instances"}) == "terminate_instances"

    def test_ec2_prefixed(self):
        result = _extract_operation("run", {"command": "ec2.delete_volume"})
        assert result == "delete_volume"

    def test_operation_arg(self):
        assert _extract_operation("run", {"operation": "list_buckets"}) == "list_buckets"

    def test_from_tool_name_tokens(self):
        result = _extract_operation("aws_delete_snapshot", {})
        # Should contain "delete" or "snapshot"
        assert "delete" in result or "snapshot" in result


# ── can_handle ────────────────────────────────────────────────────────────────

class TestCanHandle:
    def test_ec2_in_tool_name(self):
        assert run(ADP.can_handle("ec2_terminate", {}))

    def test_s3_in_tool_name(self):
        assert run(ADP.can_handle("s3_delete_object", {}))

    def test_lambda_in_tool_name(self):
        assert run(ADP.can_handle("lambda_invoke", {}))

    def test_aws_generic(self):
        assert run(ADP.can_handle("aws_call", {}))

    def test_region_arg(self):
        assert run(ADP.can_handle("cloud_op", {"region": "us-east-1"}))

    def test_bucket_arg(self):
        assert run(ADP.can_handle("delete_file", {"bucket": "my-bucket"}))

    def test_instance_id_arg(self):
        assert run(ADP.can_handle("stop", {"instance_id": "i-1234"}))

    def test_service_arg(self):
        assert run(ADP.can_handle("call", {"service": "ec2"}))

    def test_no_match(self):
        assert not run(ADP.can_handle("send_email", {"to": "a@b.com"}))

    def test_git_no_match(self):
        assert not run(ADP.can_handle("git_push", {}))


# ── dry_run: EC2 DryRun path ──────────────────────────────────────────────────

class TestEC2DryRun:
    def test_terminate_would_succeed_unsafe(self):
        with _mock_ec2(_DRY_RUN_OK):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert r.safe is False
        assert any("ec2-destructive-op" in s for s in r.signals)

    def test_delete_volume_would_succeed_unsafe(self):
        with _mock_ec2(_DRY_RUN_OK):
            r = run(ADP.dry_run("ec2_delete_volume", {"volume_id": "vol-123"}, ctx()))
        assert r.safe is False

    def test_unauthorized_safe(self):
        with _mock_ec2(_DRY_RUN_UNAUTH):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert r.safe is True
        assert "ec2-unauthorized" in r.signals

    def test_dry_run_error_signal(self):
        with _mock_ec2(_DRY_RUN_ERR):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert any("ec2-dry-run-error" in s for s in r.signals)

    def test_executed_true_on_dry_run(self):
        with _mock_ec2(_DRY_RUN_OK):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert r.executed is True

    def test_scope_has_dry_run_result(self):
        with _mock_ec2(_DRY_RUN_OK):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert "dry_run" in r.scope_estimate

    def test_non_destructive_ec2_safe(self):
        with _mock_ec2(_DRY_RUN_OK):
            r = run(ADP.dry_run("ec2_run_instances", {}, ctx()))
        # run_instances is in EC2_DRYRUN_OPS but not destructive
        # would_succeed=True but not destructive → safe
        assert r.safe is True
        assert "ec2-dry-run-ok" in r.signals


# ── dry_run: EC2 unavailable ──────────────────────────────────────────────────

class TestEC2Unavailable:
    def test_no_credentials_fail_open(self):
        with patch("app.security.adapters.aws_adapter._ec2_dry_run",
                   side_effect=RuntimeError("AWS credentials not configured")):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert r.safe is True
        assert "aws-credentials-missing" in r.signals

    def test_boto3_missing_fail_open(self):
        with patch("app.security.adapters.aws_adapter._ec2_dry_run",
                   side_effect=RuntimeError("boto3 not installed")):
            r = run(ADP.dry_run("ec2_terminate_instances", {}, ctx()))
        assert r.safe is True
        assert "aws-sdk-unavailable" in r.signals


# ── dry_run: IAM simulate path (non-EC2 destructive) ─────────────────────────

class TestIAMSimulate:
    def test_s3_delete_allowed_unsafe(self):
        with _mock_iam(_IAM_ALLOWED):
            r = run(ADP.dry_run("s3_delete_object", {"bucket": "b", "key": "k"}, ctx()))
        assert r.safe is False
        assert any("aws-destructive-authorized" in s for s in r.signals)

    def test_s3_delete_denied_safe(self):
        with _mock_iam(_IAM_DENIED):
            r = run(ADP.dry_run("s3_delete_object", {"bucket": "b"}, ctx()))
        assert r.safe is True
        assert "aws-destructive-unauthorized" in r.signals

    def test_s3_delete_unknown_authz_escalates(self):
        with _mock_iam(_IAM_UNKNOWN):
            r = run(ADP.dry_run("s3_delete_object", {"bucket": "b"}, ctx()))
        assert r.safe is False
        assert any("aws-destructive-unknown-authz" in s for s in r.signals)

    def test_executed_true_on_iam_simulate(self):
        with _mock_iam(_IAM_ALLOWED):
            r = run(ADP.dry_run("s3_delete_bucket", {}, ctx()))
        assert r.executed is True

    def test_scope_has_iam_simulate(self):
        with _mock_iam(_IAM_ALLOWED):
            r = run(ADP.dry_run("lambda_delete_function", {}, ctx()))
        assert "iam_simulate" in r.scope_estimate


# ── dry_run: IAM unavailable (destructive non-EC2) ───────────────────────────

class TestIAMUnavailable:
    def test_no_credentials_conservative(self):
        with patch("app.security.adapters.aws_adapter._iam_simulate",
                   side_effect=RuntimeError("AWS credentials not configured")):
            r = run(ADP.dry_run("s3_delete_bucket", {}, ctx()))
        # Destructive + no preview → conservative escalation
        assert r.safe is False
        assert "aws-credentials-missing" in r.signals
        assert any("aws-structural-destructive" in s for s in r.signals)

    def test_boto3_missing_conservative(self):
        with patch("app.security.adapters.aws_adapter._iam_simulate",
                   side_effect=RuntimeError("boto3 not installed")):
            r = run(ADP.dry_run("lambda_delete_function", {}, ctx()))
        assert r.safe is False
        assert "aws-sdk-unavailable" in r.signals


# ── dry_run: non-destructive ops ──────────────────────────────────────────────

class TestNonDestructive:
    def test_list_buckets_safe(self):
        r = run(ADP.dry_run("s3_list_buckets", {}, ctx()))
        assert r.safe is True
        assert any("aws-non-destructive" in s for s in r.signals)

    def test_describe_instances_safe(self):
        r = run(ADP.dry_run("ec2_describe_instances", {}, ctx()))
        assert r.safe is True

    def test_get_object_safe(self):
        r = run(ADP.dry_run("s3_get_object", {"bucket": "b"}, ctx()))
        assert r.safe is True

    def test_non_destructive_not_executed(self):
        r = run(ADP.dry_run("s3_list_objects", {}, ctx()))
        assert r.executed is False


# ── adapter metadata ──────────────────────────────────────────────────────────

class TestMeta:
    def test_name(self):
        assert ADP.name == "aws"

    def test_adapter_field_in_result(self):
        r = run(ADP.dry_run("s3_list_buckets", {}, ctx()))
        assert r.adapter == "aws"

    def test_latency_positive(self):
        r = run(ADP.dry_run("s3_list_buckets", {}, ctx()))
        assert r.latency_ms >= 0.0

    def test_preview_capped(self):
        r = run(ADP.dry_run("s3_list_buckets", {}, ctx()))
        assert len(r.preview) <= 500

    def test_scope_has_service(self):
        r = run(ADP.dry_run("s3_list_buckets", {}, ctx()))
        assert "service" in r.scope_estimate

    def test_scope_has_operation(self):
        r = run(ADP.dry_run("s3_list_buckets", {}, ctx()))
        assert "operation" in r.scope_estimate
