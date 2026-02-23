"""
OpenAI API format handler.

Handles:
- POST /openai/v1/chat/completions (standard + streaming SSE)
- POST /openai/v1/completions (legacy)
- All other /openai/... paths pass through unchanged

Request format: OpenAI Chat Completions API
Response format: OpenAI Chat Completions response OR SSE stream
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai"


def extract_request_params(body: dict) -> dict:
    """
    Extract canonical params from an OpenAI Chat Completions request body.
    Returns dict with: model, messages, tools, temperature, max_tokens, stream, extra_params
    """
    tools = body.get("tools") or []

    # Unwrap {"type": "function", "function": {...}} wrapper format
    normalized_tools = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
            normalized_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            })
        elif isinstance(t, dict):
            normalized_tools.append({
                "name": t.get("name", ""),
                "description": t.get("description"),
                "parameters": t.get("parameters"),
            })
    tools = normalized_tools

    # Legacy function_call format
    functions = body.get("functions") or []
    if functions and not tools:
        tools = [{"name": f["name"], "description": f.get("description"), "parameters": f.get("parameters")} for f in functions]

    return {
        "model": body.get("model", "unknown"),
        "messages": body.get("messages", []),
        "tools": tools,
        "temperature": body.get("temperature"),
        "max_tokens": body.get("max_tokens"),
        "stream": body.get("stream", False),
        "extra_params": {
            k: v for k, v in body.items()
            if k not in {"model", "messages", "tools", "functions", "temperature", "max_tokens", "stream"}
        } or None,
    }


def parse_response(body: dict) -> dict:
    """
    Parse an OpenAI Chat Completions response into canonical fields.
    Returns: {response_text, finish_reason, tool_calls, token_usage}
    """
    choices = body.get("choices", [])
    if not choices:
        return {"response_text": None, "finish_reason": None, "tool_calls": [], "token_usage": None}

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason")

    response_text = message.get("content")

    # Normalize tool calls
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        tool_calls.append({
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "arguments": func.get("arguments", "{}"),
        })

    # Legacy function_call
    fc = message.get("function_call")
    if fc and not tool_calls:
        tool_calls.append({
            "id": f"call_{hash(fc.get('name', ''))&0xFFFFFFFF:08x}",
            "name": fc.get("name", ""),
            "arguments": fc.get("arguments", "{}"),
        })

    usage = body.get("usage")

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "token_usage": usage,
    }


def parse_sse_chunk(chunk_data: str) -> dict | None:
    """
    Parse one SSE data line from an OpenAI streaming response.
    Returns partial response dict or None if not a data chunk.

    SSE format:
        data: {"id": ..., "choices": [{"delta": {...}, "finish_reason": null}]}
        data: [DONE]
    """
    chunk_data = chunk_data.strip()
    if not chunk_data or chunk_data == "[DONE]":
        return None

    try:
        data = json.loads(chunk_data)
    except json.JSONDecodeError:
        return None

    choices = data.get("choices", [])
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    content = delta.get("content")
    tool_calls_delta = delta.get("tool_calls")

    return {
        "content": content,
        "finish_reason": finish_reason,
        "tool_calls_delta": tool_calls_delta,
        "usage": data.get("usage"),
    }


class SSEAssembler:
    """
    Reassembles an OpenAI streaming SSE response into a complete response dict.

    The proxy forwards each chunk immediately to the agent, while buffering
    in this assembler. When finish_reason is detected, emit the full event.
    """

    def __init__(self):
        self._content_parts: list[str] = []
        self._finish_reason: str | None = None
        self._tool_calls_map: dict[int, dict] = {}  # index → partial tool call
        self._usage: dict | None = None

    def feed(self, chunk: dict | None) -> bool:
        """
        Feed one parsed SSE chunk. Returns True if response is complete.
        """
        if chunk is None:
            return False

        if chunk.get("content"):
            self._content_parts.append(chunk["content"])

        if chunk.get("finish_reason"):
            self._finish_reason = chunk["finish_reason"]

        if chunk.get("usage"):
            self._usage = chunk["usage"]

        # Accumulate streaming tool calls
        for tc_delta in (chunk.get("tool_calls_delta") or []):
            idx = tc_delta.get("index", 0)
            if idx not in self._tool_calls_map:
                self._tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
            entry = self._tool_calls_map[idx]
            if tc_delta.get("id"):
                entry["id"] = tc_delta["id"]
            func = tc_delta.get("function", {})
            if func.get("name"):
                entry["name"] = func["name"]
            if func.get("arguments"):
                entry["arguments"] += func["arguments"]

        return self._finish_reason is not None

    def build_response(self) -> dict:
        """Build final canonical response dict from assembled chunks."""
        tool_calls = [
            {"id": tc["id"], "name": tc["name"], "arguments": tc["arguments"]}
            for tc in sorted(self._tool_calls_map.values(), key=lambda x: x["id"])
        ]
        return {
            "response_text": "".join(self._content_parts) or None,
            "finish_reason": self._finish_reason,
            "tool_calls": tool_calls,
            "token_usage": self._usage,
        }


class OpenAIProvider:
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
