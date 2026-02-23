"""
Canonicalization: raw LLM API request/response → canonical AuditEvent dicts.

Each provider handler calls these functions after parsing its own format.
The output is provider-agnostic and matches the shared AuditEvent schema.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ulid import ULID

from .models import (
    AuditEvent,
    EventType,
    InterceptionLayer,
    LLMCallEndPayload,
    LLMCallStartPayload,
    Message,
    Provider,
    ToolCall,
    ToolDefinition,
    TokenUsage,
)


def _now_ns() -> int:
    return time.time_ns()


def _make_run_id() -> str:
    return str(uuid.uuid4())


def make_llm_call_start(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
    extra_params: dict | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    sequence_number: int = 0,
    previous_hash: str = "",
) -> dict:
    """Build a raw dict for LLM_CALL_START (before hash-chain signing)."""
    run_id = run_id or _make_run_id()

    # Normalize messages to Message dicts
    norm_messages = []
    for m in messages:
        norm_messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content"),
            "tool_call_id": m.get("tool_call_id"),
            "name": m.get("name"),
        })

    # Normalize tool definitions — unwrap {"type":"function","function":{...}} if needed
    norm_tools = []
    if tools:
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                fn = t["function"]
            else:
                fn = t
            norm_tools.append({
                "name": fn.get("name", "unknown"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            })

    payload = {
        "messages": norm_messages,
        "tools_available": norm_tools,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "extra_params": extra_params,
    }

    event_id = str(ULID())

    return {
        "event_id": event_id,
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "interception_layer": InterceptionLayer.PROXY,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "event_type": EventType.LLM_CALL_START,
        "payload": payload,
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",  # filled by hash_chain.compute_event_hash
    }


def make_llm_call_end(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    run_id: str,
    response_text: str | None,
    finish_reason: str | None,
    tool_calls: list[dict] | None,
    token_usage: dict | None,
    latency_ms: float | None,
    http_status: int | None = None,
    sequence_number: int = 0,
    previous_hash: str = "",
) -> dict:
    """Build a raw dict for LLM_CALL_END (before hash-chain signing)."""
    norm_tool_calls = []
    if tool_calls:
        for tc in tool_calls:
            norm_tool_calls.append({
                "id": tc.get("id", str(uuid.uuid4())),
                "name": tc.get("name") or tc.get("function", {}).get("name", "unknown"),
                "arguments": tc.get("arguments") or tc.get("function", {}).get("arguments", "{}"),
            })

    norm_usage = None
    if token_usage:
        norm_usage = {
            "prompt_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens"),
            "completion_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
        }

    payload = {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "tool_calls": norm_tool_calls,
        "token_usage": norm_usage,
        "latency_ms": latency_ms,
        "http_status": http_status,
    }

    event_id = str(ULID())

    return {
        "event_id": event_id,
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "interception_layer": InterceptionLayer.PROXY,
        "run_id": run_id,
        "parent_run_id": None,
        "event_type": EventType.LLM_CALL_END,
        "payload": payload,
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",
    }


def make_system_error(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    run_id: str,
    error_message: str,
    error_code: Any = None,
    http_status: int | None = None,
    sequence_number: int = 0,
    previous_hash: str = "",
) -> dict:
    """Build a raw dict for SYSTEM_ERROR."""
    return {
        "event_id": str(ULID()),
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "interception_layer": InterceptionLayer.PROXY,
        "run_id": run_id,
        "parent_run_id": None,
        "event_type": EventType.SYSTEM_ERROR,
        "payload": {
            "error_code": error_code,
            "error_message": error_message,
            "provider": provider,
            "http_status": http_status,
        },
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",
    }


def make_agent_finish(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    run_id: str,
    final_output: str | None,
    total_llm_calls: int,
    total_tool_calls: int,
    session_duration_ms: float | None,
    sequence_number: int = 0,
    previous_hash: str = "",
) -> dict:
    return {
        "event_id": str(ULID()),
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "interception_layer": InterceptionLayer.PROXY,
        "run_id": run_id,
        "parent_run_id": None,
        "event_type": EventType.AGENT_FINISH,
        "payload": {
            "final_output": final_output,
            "total_llm_calls": total_llm_calls,
            "total_tool_calls": total_tool_calls,
            "session_duration_ms": session_duration_ms,
        },
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",
    }


def make_checkpoint(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    run_id: str,
    merkle_root: str,
    events_covered: int,
    from_sequence: int,
    to_sequence: int,
    sequence_number: int,
    previous_hash: str,
) -> dict:
    return {
        "event_id": str(ULID()),
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "interception_layer": InterceptionLayer.PROXY,
        "run_id": run_id,
        "parent_run_id": None,
        "event_type": EventType.CHECKPOINT,
        "payload": {
            "merkle_root": merkle_root,
            "events_covered": events_covered,
            "from_sequence": from_sequence,
            "to_sequence": to_sequence,
        },
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",
    }
