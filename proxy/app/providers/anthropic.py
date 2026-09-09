"""
Anthropic API format handler.

Handles:
- POST /anthropic/v1/messages (standard + streaming SSE)
- All other /anthropic/... paths pass through unchanged

Anthropic Messages API differs from OpenAI:
- system message is a top-level field, not in messages array
- tool_use is a content block type, not a separate field
- finish_reason is called stop_reason
- token counts use input_tokens/output_tokens
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

PROVIDER_NAME = "anthropic"


def extract_request_params(body: dict) -> dict:
    """
    Extract canonical params from an Anthropic Messages request.
    Normalizes system message into messages array for consistent storage.
    """
    messages = list(body.get("messages", []))

    # Normalize system prompt into messages
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages = [{"role": "system", "content": system}] + messages
        elif isinstance(system, list):
            # System can be a list of content blocks with cache_control etc.
            text_parts = [b.get("text", "") for b in system if isinstance(b, dict)]
            messages = [{"role": "system", "content": " ".join(text_parts)}] + messages

    # Normalize tool definitions
    tools = []
    for t in (body.get("tools") or []):
        tools.append({
            "name": t.get("name", ""),
            "description": t.get("description"),
            "parameters": t.get("input_schema"),  # Anthropic uses input_schema
        })

    return {
        "model": body.get("model", "unknown"),
        "messages": messages,
        "tools": tools,
        "temperature": body.get("temperature"),
        "max_tokens": body.get("max_tokens"),
        "stream": body.get("stream", False),
        "extra_params": {
            k: v for k, v in body.items()
            if k not in {"model", "messages", "system", "tools", "temperature", "max_tokens", "stream"}
        } or None,
    }


def parse_response(body: dict) -> dict:
    """Parse Anthropic Messages response into canonical fields."""
    stop_reason = body.get("stop_reason")

    # Finish reason mapping
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_reason_map.get(stop_reason, stop_reason)

    response_text = None
    tool_calls = []

    thinking_blocks = []
    for block in (body.get("content") or []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            response_text = block.get("text")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })
        elif block.get("type") == "thinking":
            thinking_blocks.append({
                "thinking": block.get("thinking", "")[:settings.reasoning_trace_max_chars],
                "signature": block.get("signature"),
            })

    usage = body.get("usage")
    token_usage = None
    if usage:
        token_usage = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
        }

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "token_usage": token_usage,
        "thinking_blocks": thinking_blocks,
    }


def parse_sse_chunk(chunk_data: str) -> dict | None:
    """
    Parse Anthropic SSE streaming event.

    Anthropic SSE format uses event types:
        event: content_block_start
        data: {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}

        event: content_block_delta
        data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}

        event: message_delta
        data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}
    """
    chunk_data = chunk_data.strip()
    if not chunk_data:
        return None
    try:
        data = json.loads(chunk_data)
    except json.JSONDecodeError:
        return None

    event_type = data.get("type", "")

    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            return {"content": delta.get("text"), "finish_reason": None, "tool_calls_delta": None, "usage": None}
        elif delta.get("type") == "input_json_delta":
            return {"content": None, "finish_reason": None, "tool_input_delta": delta.get("partial_json"), "usage": None}
        elif delta.get("type") == "thinking_delta":
            return {"thinking_delta": delta.get("thinking"), "finish_reason": None, "tool_calls_delta": None, "usage": None}
        elif delta.get("type") == "signature_delta":
            return {"signature_delta": delta.get("signature"), "finish_reason": None, "tool_calls_delta": None, "usage": None}

    elif event_type == "message_delta":
        delta = data.get("delta", {})
        stop_reason = delta.get("stop_reason")
        finish_reason_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
        finish_reason = finish_reason_map.get(stop_reason, stop_reason)
        usage = data.get("usage")
        token_usage = None
        if usage:
            token_usage = {
                "prompt_tokens": None,
                "completion_tokens": usage.get("output_tokens"),
                "total_tokens": None,
            }
        return {"content": None, "finish_reason": finish_reason, "tool_calls_delta": None, "usage": token_usage}

    elif event_type == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "tool_use":
            return {
                "content": None,
                "finish_reason": None,
                "tool_calls_delta": [{
                    "index": data.get("index", 0),
                    "id": block.get("id", ""),
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }],
                "usage": None,
            }
        elif block.get("type") == "thinking":
            return {"thinking_block_start": data.get("index"), "finish_reason": None, "tool_calls_delta": None, "usage": None}

    return None


class SSEAssembler:
    """Reassembles Anthropic streaming response."""

    def __init__(self):
        self._content_parts: list[str] = []
        self._finish_reason: str | None = None
        self._tool_calls_map: dict[int, dict] = {}
        self._usage: dict | None = None
        self._current_tool_idx: int | None = None
        self._thinking_parts: list[str] = []
        self._thinking_signature: str | None = None
        self._in_thinking_block: bool = False

    def feed(self, chunk: dict | None) -> bool:
        if chunk is None:
            return False
        if chunk.get("content"):
            self._content_parts.append(chunk["content"])
        if chunk.get("finish_reason"):
            self._finish_reason = chunk["finish_reason"]
        if chunk.get("usage"):
            self._usage = chunk["usage"]
        if "thinking_block_start" in chunk:
            self._in_thinking_block = True
        if chunk.get("thinking_delta"):
            # Guard: auto-activate block even if thinking_block_start was missed
            if not self._in_thinking_block:
                self._in_thinking_block = True
            self._thinking_parts.append(chunk["thinking_delta"])
        if chunk.get("signature_delta"):
            self._thinking_signature = chunk["signature_delta"]
        for tc_delta in (chunk.get("tool_calls_delta") or []):
            idx = tc_delta.get("index", 0)
            if idx not in self._tool_calls_map:
                self._tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
            self._current_tool_idx = idx  # track most recently active tool block
            entry = self._tool_calls_map[idx]
            if tc_delta.get("id"):
                entry["id"] = tc_delta["id"]
            func = tc_delta.get("function", {})
            if func.get("name"):
                entry["name"] = func["name"]
            if func.get("arguments"):
                entry["arguments"] += func["arguments"]
        if chunk.get("tool_input_delta") and self._current_tool_idx is not None:
            entry = self._tool_calls_map.setdefault(
                self._current_tool_idx, {"id": "", "name": "", "arguments": ""}
            )
            entry["arguments"] += chunk["tool_input_delta"]
        return self._finish_reason is not None

    def build_response(self) -> dict:
        tool_calls = [
            {"id": tc["id"], "name": tc["name"], "arguments": tc["arguments"]}
            for tc in self._tool_calls_map.values()
        ]
        thinking_blocks = []
        if self._thinking_parts:
            thinking_blocks = [{
                "thinking": "".join(self._thinking_parts)[:settings.reasoning_trace_max_chars],
                "signature": self._thinking_signature,
            }]
        return {
            "response_text": "".join(self._content_parts) or None,
            "finish_reason": self._finish_reason,
            "tool_calls": tool_calls,
            "token_usage": self._usage,
            "thinking_blocks": thinking_blocks,
        }


class AnthropicProvider:
    name = PROVIDER_NAME

    @staticmethod
    def extract_request_params(body: dict) -> dict:
        return extract_request_params(body)

    @staticmethod
    def parse_response(body: dict) -> dict:
        return parse_response(body)

    @staticmethod
    def parse_sse_chunk(chunk_data: str) -> dict | None:
        return parse_sse_chunk(chunk_data)

    @staticmethod
    def new_assembler() -> SSEAssembler:
        return SSEAssembler()
