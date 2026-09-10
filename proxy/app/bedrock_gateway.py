"""
AWS Bedrock boto3 gateway.

Manages a pool of boto3 bedrock-runtime clients (one per region) and provides
async wrappers for Converse API and InvokeModel API calls.

All calls are run in a thread-pool executor because boto3 is synchronous.

Credential resolution order (standard boto3 chain):
  1. AEGIVIS_AWS_ACCESS_KEY_ID / AEGIVIS_AWS_SECRET_ACCESS_KEY env vars
  2. AEGIVIS_BEDROCK_PROFILE (named AWS profile)
  3. Standard boto3 chain: AWS_ACCESS_KEY_ID env → ~/.aws/credentials → IMDSv2

For production Kubernetes: attach an IAM role to the proxy ServiceAccount
via IRSA (IAM Roles for Service Accounts) — zero secrets needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

# ── Client pool ───────────────────────────────────────────────────────────────

# Keyed by (region, profile) so multiple regions work simultaneously.
_client_cache: dict[tuple[str, str], Any] = {}


def get_bedrock_client(region: str | None = None):
    """Return a cached boto3 bedrock-runtime client for the given region."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError(
            "boto3 is not installed. Install with: pip install 'aegivis-proxy[bedrock]'"
        )

    region  = region or settings.bedrock_region
    profile = settings.bedrock_profile
    key     = (region, profile)

    if key not in _client_cache:
        kwargs: dict[str, Any] = {"region_name": region}

        if profile:
            kwargs["profile_name"] = profile
        else:
            # Explicit credentials override (optional — prefer IAM role)
            if settings.aws_access_key_id:
                kwargs["aws_access_key_id"]     = settings.aws_access_key_id
                kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            if settings.aws_session_token:
                kwargs["aws_session_token"] = settings.aws_session_token

        _client_cache[key] = boto3.client("bedrock-runtime", **kwargs)
        logger.info(
            "Created Bedrock client (region=%s, profile=%s)", region, profile or "default"
        )

    return _client_cache[key]


def invalidate_client_cache() -> None:
    """Clear cached boto3 clients (e.g. after credential rotation)."""
    _client_cache.clear()


# ── Request body builder ──────────────────────────────────────────────────────

def _build_converse_kwargs(model_id: str, body: dict) -> dict:
    """
    Build the keyword arguments for boto3.converse() / converse_stream()
    from a raw Bedrock Converse API request body.

    We pass the body fields through rather than the canonical format so the
    proxy doesn't inadvertently lose model-specific parameters.
    """
    kwargs: dict[str, Any] = {"modelId": model_id}

    if "messages" in body:
        kwargs["messages"] = body["messages"]
    if "system" in body:
        kwargs["system"] = body["system"]
    if "inferenceConfig" in body:
        kwargs["inferenceConfig"] = body["inferenceConfig"]
    if "toolConfig" in body:
        kwargs["toolConfig"] = body["toolConfig"]
    if "additionalModelRequestFields" in body:
        kwargs["additionalModelRequestFields"] = body["additionalModelRequestFields"]
    if "guardrailConfig" in body:
        kwargs["guardrailConfig"] = body["guardrailConfig"]

    return kwargs


# ── Converse API ──────────────────────────────────────────────────────────────

async def forward_converse(
    model_id: str,
    body: dict,
    region: str | None = None,
) -> dict:
    """
    Forward a Converse API request to Bedrock and return the raw response dict.

    Returns the Bedrock Converse response body (ResponseMetadata stripped).
    """
    client = get_bedrock_client(region)
    kwargs = _build_converse_kwargs(model_id, body)

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.converse(**kwargs),
        )
    except Exception as exc:
        _handle_boto3_error(exc)

    response.pop("ResponseMetadata", None)
    return response


async def forward_converse_stream(
    model_id: str,
    body: dict,
    region: str | None = None,
) -> dict:
    """
    Forward a Converse Stream request and buffer the full EventStream response.

    Returns a Converse-API-compatible response dict (same shape as converse()).
    Streaming tokens are not delivered incrementally — the response is
    complete before returning.  This is a known limitation of E2 MVP.
    """
    from .providers.bedrock import assemble_converse_stream

    client = get_bedrock_client(region)
    kwargs = _build_converse_kwargs(model_id, body)

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.converse_stream(**kwargs),
        )
    except Exception as exc:
        _handle_boto3_error(exc)

    # response["stream"] is a botocore EventStream — iterate in executor
    event_stream = response.get("stream")
    canonical = await loop.run_in_executor(
        None,
        lambda: assemble_converse_stream(event_stream),
    )

    # Reshape canonical → Converse API response format for transparency
    return _canonical_to_converse_response(canonical)


# ── InvokeModel API ───────────────────────────────────────────────────────────

async def forward_invoke_model(
    model_id: str,
    body_bytes: bytes,
    region: str | None = None,
) -> bytes:
    """
    Forward an InvokeModel request and return the raw response body bytes.
    """
    client = get_bedrock_client(region)
    loop   = asyncio.get_event_loop()

    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model(
                modelId=model_id,
                body=body_bytes,
                contentType="application/json",
                accept="application/json",
            ),
        )
    except Exception as exc:
        _handle_boto3_error(exc)

    return response["body"].read()


async def forward_invoke_model_stream(
    model_id: str,
    body_bytes: bytes,
    region: str | None = None,
) -> bytes:
    """
    Forward an InvokeModel streaming request, buffer all chunks, return as bytes.

    Like converse-stream, streaming tokens are not delivered incrementally.
    """
    client = get_bedrock_client(region)
    loop   = asyncio.get_event_loop()

    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model_with_response_stream(
                modelId=model_id,
                body=body_bytes,
                contentType="application/json",
                accept="application/json",
            ),
        )
    except Exception as exc:
        _handle_boto3_error(exc)

    stream = response.get("body")
    parts: list[bytes] = []

    def _collect():
        for event in stream:
            chunk = event.get("chunk") or {}
            if "bytes" in chunk:
                parts.append(chunk["bytes"])

    await loop.run_in_executor(None, _collect)
    return b"".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _canonical_to_converse_response(canonical: dict) -> dict:
    """Convert canonical Aegivis response dict → Bedrock Converse API response shape."""
    content: list[dict] = []

    if canonical.get("response_text"):
        content.append({"text": canonical["response_text"]})

    for tc in canonical.get("tool_calls") or []:
        try:
            input_obj = json.loads(tc.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_obj = {}
        content.append({
            "toolUse": {
                "toolUseId": tc.get("id", ""),
                "name":      tc.get("name", ""),
                "input":     input_obj,
            }
        })

    usage = canonical.get("token_usage") or {}
    result: dict[str, Any] = {
        "output": {
            "message": {
                "role":    "assistant",
                "content": content,
            }
        },
        "stopReason": canonical.get("finish_reason", "end_turn"),
    }
    if usage:
        result["usage"] = {
            "inputTokens":  usage.get("input_tokens", 0),
            "outputTokens": usage.get("output_tokens", 0),
            "totalTokens":  usage.get("total_tokens", 0),
        }
    return result


def _handle_boto3_error(exc: Exception) -> None:
    """Translate boto3 / botocore errors into cleaner exceptions."""
    try:
        from botocore.exceptions import ClientError, NoCredentialsError, EndpointResolutionError
        if isinstance(exc, NoCredentialsError):
            raise RuntimeError(
                "No AWS credentials found for Bedrock proxy. Set AEGIVIS_AWS_ACCESS_KEY_ID "
                "/ AEGIVIS_AWS_SECRET_ACCESS_KEY, or use an IAM role / AWS profile "
                "(AEGIVIS_BEDROCK_PROFILE). See docs for IRSA setup."
            ) from exc
        if isinstance(exc, ClientError):
            code = exc.response["Error"]["Code"]
            msg  = exc.response["Error"]["Message"]
            raise RuntimeError(f"Bedrock ClientError [{code}]: {msg}") from exc
    except ImportError:
        pass
    raise exc
