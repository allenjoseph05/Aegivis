"""
Tool call inference from conversation patterns.

The proxy infers TOOL_CALL_START and TOOL_CALL_END events by analyzing
the conversation message history in LLM API requests:

Pattern:
  1. LLM_CALL_END with tool_calls = [{id, name, arguments}]
     → Emit TOOL_CALL_START for each tool_call
  2. Next LLM_CALL_START where messages contain role="tool" with tool_call_id
     → Emit TOOL_CALL_END for each tool result

This reconstruction is session-aware: we track pending tool calls per session.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from ulid import ULID

from .models import EventType, InterceptionLayer

logger = logging.getLogger(__name__)


def _now_ns() -> int:
    return time.time_ns()


def make_tool_call_start(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    parent_run_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_input: Any,
    tool_description: str | None,
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
        "run_id": tool_call_id,
        "parent_run_id": parent_run_id,
        "event_type": EventType.TOOL_CALL_START,
        "payload": {
            "tool_name": tool_name,
            "tool_description": tool_description,
            "tool_input": tool_input,
            "tool_call_id": tool_call_id,
        },
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",
    }


def make_tool_call_end(
    *,
    session_id: str,
    org_id: str,
    agent_id: str,
    provider: str,
    model: str,
    parent_run_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_output: str | None,
    sequence_number: int,
    previous_hash: str,
) -> dict:
    import ulid as _ulid

    output_hash = None
    if tool_output is not None:
        output_hash = hashlib.sha256(tool_output.encode("utf-8")).hexdigest()

    return {
        "event_id": str(_ulid.new()),
        "schema_version": "1.0",
        "org_id": org_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "interception_layer": InterceptionLayer.PROXY,
        "run_id": tool_call_id,
        "parent_run_id": parent_run_id,
        "event_type": EventType.TOOL_CALL_END,
        "payload": {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_output_masked": tool_output,   # PII masking applied by intercept.py
            "tool_output_hash": output_hash,
            "pii_fields_detected": [],           # filled by intercept.py after PII pass
            "success": True,
        },
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": _now_ns(),
        "sequence_number": sequence_number,
        "previous_hash": previous_hash,
        "current_hash": "",
    }


def extract_tool_calls_from_response(llm_end_event: dict) -> list[dict]:
    """Extract normalized tool call dicts from an LLM_CALL_END payload."""
    return llm_end_event.get("payload", {}).get("tool_calls", [])


def extract_tool_results_from_request(messages: list[dict]) -> list[dict]:
    """
    Extract tool result messages from an incoming LLM request's message array.
    Returns list of {"tool_call_id": ..., "content": ..., "name": ...}.
    """
    results = []
    for msg in messages:
        if msg.get("role") == "tool":
            results.append({
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
                "name": msg.get("name"),
            })
    return results
