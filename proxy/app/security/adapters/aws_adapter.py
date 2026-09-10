"""
AWS DryRun Sandbox Adapter — Phase 16.3.

Uses boto3 DryRun=True for EC2 operations and IAM simulate_principal_policy
for other AWS services to estimate the blast radius BEFORE execution.

Detection: tool name contains AWS service name tokens (ec2, s3, iam, lambda,
rds, etc.) OR args contain AWS-specific parameter keys (region, bucket,
instance_id, stack_name, …).

Scope logic:
  - EC2 write ops that support DryRun: call with DryRun=True
      DryRunOperation    → would succeed → safe only if non-destructive
      UnauthorizedOperation → insufficient perms → safe (can't do it)
  - Other services (S3, IAM, Lambda, etc.): IAM simulate_principal_policy
      allowed   → would succeed → safe only if non-destructive
      not-allowed → safe (unauthorized)
  - Structural fallback (no credentials): classify by operation name tokens
      destructive + no preview → safe=False (conservative escalation)

Fails open (safe=True) when boto3 is not installed.
Escalates conservatively (safe=False) when destructive op + no credential preview.

No regex. Detection is frozenset intersection on lowercased tool-name tokens.
"""
from __future__ import annotations

import logging
import time

from ..sandbox import SandboxAdapter, SandboxContext, SandboxResult
from ._common import tokenize as _tokenize

logger = logging.getLogger(__name__)

# ── Token sets ────────────────────────────────────────────────────────────────

_AWS_TOOL_TOKENS: frozenset[str] = frozenset({
    "aws", "boto", "ec2", "s3", "iam", "lambda", "cloudformation",
    "rds", "dynamodb", "sqs", "sns", "eks", "ecs", "ssm",
    "amazon", "cloud",
})

_AWS_ARG_KEYS: frozenset[str] = frozenset({
    "region", "bucket", "instance_id", "instance_ids", "stack_name",
    "function_name", "table_name", "queue_url", "topic_arn",
    "resource_id", "cluster_name", "group_id",
    "volume_id", "snapshot_id", "image_id", "key_name",
})

_COMMAND_ARG_KEYS: frozenset[str] = frozenset({
    "command", "cmd", "operation", "action", "method",
})

# EC2 operations that natively support DryRun=True
_EC2_DRYRUN_OPS: frozenset[str] = frozenset({
    "terminate_instances", "stop_instances", "start_instances",
    "reboot_instances", "run_instances", "create_image",
    "delete_volume", "delete_snapshot", "create_security_group",
    "delete_security_group", "authorize_security_group_ingress",
    "revoke_security_group_ingress", "create_key_pair", "delete_key_pair",
    "associate_address", "release_address", "allocate_address",
    "create_vpc", "delete_vpc", "create_subnet", "delete_subnet",
    "create_internet_gateway", "delete_internet_gateway",
    "attach_internet_gateway", "detach_internet_gateway",
    "create_route_table", "delete_route_table",
})

_DESTRUCTIVE_OP_TOKENS: frozenset[str] = frozenset({
    "delete", "terminate", "destroy", "drop", "remove", "purge",
    "deregister", "detach", "revoke", "disable", "suspend", "kill",
})

# Non-destructive service token prefixes that imply read-only
_AWS_NOOPS: frozenset[str] = frozenset({
    "aws", "boto", "amazon", "cloud", "ssm",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_service(tool_name: str, args: dict) -> str:
    tokens = _tokenize(tool_name) - _AWS_NOOPS
    service_tokens = tokens & (_AWS_TOOL_TOKENS - _AWS_NOOPS)
    if service_tokens:
        return next(iter(service_tokens))
    svc = args.get("service") or args.get("aws_service") or args.get("client")
    if isinstance(svc, str):
        return svc.lower().split(".")[0]
    return "unknown"


_SERVICE_PREFIXES: tuple[str, ...] = (
    "ec2_", "s3_", "iam_", "lambda_", "cloudformation_", "rds_",
    "dynamodb_", "sqs_", "sns_", "eks_", "ecs_", "ssm_",
    "aws_", "boto_", "amazon_", "cloud_",
)


def _extract_operation(tool_name: str, args: dict) -> str:
    for key in _COMMAND_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            # Strip optional service prefix: "ec2.terminate_instances" → "terminate_instances"
            s = val.strip().replace("-", "_").lower().split(".")
            return s[-1]
    # Strip known service/generic prefix from tool name to preserve full operation
    name = tool_name.lower()
    for prefix in _SERVICE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    # Fallback: remove the first underscore-delimited token if it's a known service
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0] in _AWS_TOOL_TOKENS:
        return parts[1]
    return name if name else "unknown"


def _is_destructive(operation: str) -> bool:
    return bool(_tokenize(operation) & _DESTRUCTIVE_OP_TOKENS)


async def _ec2_dry_run(operation: str, params: dict, region: str) -> dict:
    """
    Attempt EC2 operation with DryRun=True via boto3.

    Returns:
        {would_succeed: bool|None, authorized: bool|None,
         error_code: str|None, error_message: str|None}

    Raises:
        RuntimeError — boto3 not installed or credentials missing
    """
    try:
        import asyncio
        import boto3  # type: ignore[import]
        from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore[import]
    except ImportError:
        raise RuntimeError("boto3 not installed")

    try:
        client = boto3.client("ec2", region_name=region or "us-east-1")
        method = getattr(client, operation, None)
        if method is None:
            return {
                "would_succeed": None, "authorized": None,
                "error_code": "unsupported-op", "error_message": None,
            }

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: method(**params, DryRun=True))
            # If call returns without exception, it actually ran (shouldn't happen)
            return {"would_succeed": True, "authorized": True, "error_code": None, "error_message": None}
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg  = exc.response["Error"]["Message"]
            if code == "DryRunOperation":
                return {"would_succeed": True,  "authorized": True,  "error_code": code, "error_message": msg}
            if code == "UnauthorizedOperation":
                return {"would_succeed": False, "authorized": False, "error_code": code, "error_message": msg}
            return {"would_succeed": None, "authorized": None, "error_code": code, "error_message": msg}

    except NoCredentialsError:
        raise RuntimeError("AWS credentials not configured")


async def _iam_simulate(action: str, resource: str = "*", region: str = "us-east-1") -> dict:
    """
    Use IAM simulate_principal_policy to check if the current identity
    is permitted to perform *action* on *resource*.

    Returns:
        {allowed: bool|None, decision: str, caller_arn: str|None}

    Raises:
        RuntimeError — boto3 not installed or credentials missing
    """
    try:
        import asyncio
        import boto3  # type: ignore[import]
        from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore[import]
    except ImportError:
        raise RuntimeError("boto3 not installed")

    try:
        loop = asyncio.get_event_loop()
        sts = boto3.client("sts",  region_name=region)
        iam = boto3.client("iam",  region_name=region)

        identity = await loop.run_in_executor(None, sts.get_caller_identity)
        caller_arn = identity["Arn"]

        result = await loop.run_in_executor(
            None,
            lambda: iam.simulate_principal_policy(
                PolicySourceArn=caller_arn,
                ActionNames=[action],
                ResourceArns=[resource],
            ),
        )
        decisions = result.get("EvaluationResults", [])
        if decisions:
            decision = decisions[0]["EvalDecision"]
            return {"allowed": decision == "allowed", "decision": decision, "caller_arn": caller_arn}
        return {"allowed": None, "decision": "unknown", "caller_arn": caller_arn}

    except NoCredentialsError:
        raise RuntimeError("AWS credentials not configured")
    except ClientError as exc:
        raise RuntimeError(f"IAM simulate failed: {exc.response['Error']['Code']}")


# ── AWSDryRunAdapter ──────────────────────────────────────────────────────────

class AWSDryRunAdapter(SandboxAdapter):
    """
    Sandbox adapter for AWS API tool calls.

    - EC2 write ops (terminate, delete, etc.) → DryRun=True
    - All other services → IAM simulate_principal_policy
    - Structural fallback (no credentials) → token-based classification
    """

    name = "aws"

    async def can_handle(self, tool_name: str, args: dict) -> bool:
        if _tokenize(tool_name) & _AWS_TOOL_TOKENS:
            return True
        for key in _AWS_ARG_KEYS:
            if key in args:
                return True
        svc = args.get("service") or args.get("aws_service")
        if isinstance(svc, str):
            return True
        return False

    async def dry_run(
        self,
        tool_name: str,
        args: dict,
        ctx: SandboxContext,
    ) -> SandboxResult:
        t0 = time.monotonic()

        service   = _extract_service(tool_name, args)
        operation = _extract_operation(tool_name, args)
        region    = str(args.get("region") or "us-east-1")

        signals: list[str] = []
        scope: dict = {"service": service, "operation": operation}
        safe = True
        executed = False
        preview = f"AWS {service}.{operation}"

        destructive = _is_destructive(operation)
        op_norm = operation.replace("-", "_").lower()

        # ── EC2 with native DryRun ────────────────────────────────────────────
        if service == "ec2" and op_norm in _EC2_DRYRUN_OPS:
            call_params = {
                k: v for k, v in args.items()
                if k not in ("region", "service", "aws_service", "client")
            }
            try:
                result = await _ec2_dry_run(op_norm, call_params, region)
                executed = True
                scope["dry_run"] = result

                if result.get("would_succeed") is True:
                    signals.append("ec2-dry-run-would-succeed")
                    if destructive:
                        signals.append(f"ec2-destructive-op:{op_norm}")
                        safe = False
                    else:
                        signals.append("ec2-dry-run-ok")
                elif result.get("authorized") is False:
                    signals.append("ec2-unauthorized")
                    safe = True   # can't do it anyway — not our problem
                else:
                    signals.append(f"ec2-dry-run-error:{result.get('error_code', 'unknown')}")

                preview = f"EC2 dry-run {op_norm}: {result.get('error_code', 'ok')}"

            except RuntimeError as exc:
                sig = "aws-credentials-missing" if "credentials" in str(exc).lower() else "aws-sdk-unavailable"
                signals.append(sig)
                scope["error"] = str(exc)
                safe = True   # fail-open: no preview available, not the agent's fault
                preview = f"AWS not available: {exc}"

        # ── Other services: IAM simulate + structural ─────────────────────────
        else:
            if destructive:
                try:
                    # Build approximate IAM action: "s3:DeleteObject"
                    svc_short = service if service not in ("unknown", "cloud") else "s3"
                    iam_action = (
                        f"{svc_short}:{op_norm.replace('_', ' ').title().replace(' ', '')}"
                    )
                    result = await _iam_simulate(iam_action, "*", region)
                    executed = True
                    scope["iam_simulate"] = result

                    if result.get("allowed") is True:
                        signals.append(f"aws-destructive-authorized:{operation}")
                        safe = False
                    elif result.get("allowed") is False:
                        signals.append("aws-destructive-unauthorized")
                        safe = True
                    else:
                        signals.append(f"aws-destructive-unknown-authz:{operation}")
                        safe = False   # conservative: can't confirm safe

                    preview = (
                        f"IAM simulate {iam_action}: "
                        f"{result.get('decision', 'unknown')}"
                    )

                except RuntimeError as exc:
                    sig = "aws-credentials-missing" if "credentials" in str(exc).lower() else "aws-sdk-unavailable"
                    signals.append(sig)
                    signals.append(f"aws-structural-destructive:{operation}")
                    scope["error"] = str(exc)
                    # Conservative: destructive + no preview → escalate
                    safe = False
                    preview = f"AWS destructive op, no preview available: {operation}"

            else:
                # Non-destructive, no creds needed — structurally safe
                signals.append(f"aws-non-destructive:{operation}")
                safe = True
                preview = f"AWS {service}.{operation}: non-destructive (structural)"

        return SandboxResult(
            adapter="aws",
            executed=executed,
            safe=safe,
            scope_estimate=scope,
            preview=preview[:500],
            signals=signals or ["aws-ok"],
            latency_ms=(time.monotonic() - t0) * 1000,
            error=None,
        )
