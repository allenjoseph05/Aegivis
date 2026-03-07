"""
Google Gemini API format handler.

Handles:
- POST /google/v1/models/{model}:generateContent
- POST /google/v1/models/{model}:streamGenerateContent
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER_NAME = "google"


def extract_request_params(body: dict, model: str = "unknown") -> dict:
    """Extract canonical params from a Google Gemini generateContent request."""
    contents = body.get("contents", [])

    # Normalize to OpenAI-like message format
    messages = []
    for item in contents:
        role_map = {"user": "user", "model": "assistant"}
        role = role_map.get(item.get("role", "user"), "user")
        parts = item.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        messages.append({"role": role, "content": text})

    # System instruction
    sys_instruction = body.get("systemInstruction")
    if sys_instruction:
        parts = sys_instruction.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        messages = [{"role": "system", "content": text}] + messages

    # Tool definitions
    tools = []
    for tool_decl in (body.get("tools") or []):
        for func_decl in (tool_decl.get("functionDeclarations") or []):
            tools.append({
                "name": func_decl.get("name", ""),
                "description": func_decl.get("description"),
                "parameters": func_decl.get("parameters"),
            })

    gen_config = body.get("generationConfig", {})

    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": gen_config.get("temperature"),
        "max_tokens": gen_config.get("maxOutputTokens"),
        "stream": False,
        "extra_params": None,
    }


def parse_response(body: dict) -> dict:
    """Parse Google Gemini generateContent response."""
    candidates = body.get("candidates", [])
    if not candidates:
        return {"response_text": None, "finish_reason": None, "tool_calls": [], "token_usage": None}

    candidate = candidates[0]
    finish_reason_map = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }
    finish_reason = finish_reason_map.get(candidate.get("finishReason", ""), candidate.get("finishReason"))

    content = candidate.get("content", {})
    parts = content.get("parts", [])

    response_text = None
    tool_calls = []
    thinking_blocks: list[dict] = []

    for part in parts:
        if part.get("thought"):
            # Gemini thinking part (2.5 Flash/Pro, 2.0 Flash Thinking)
            text = part.get("text", "")
            if text:
                thinking_blocks.append({"thinking": text, "signature": None})
        elif "text" in part:
            response_text = part["text"]
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "id": f"call_{fc.get('name', 'fn')}",
                "name": fc.get("name", ""),
                "arguments": fc.get("args", {}),
            })

    usage = body.get("usageMetadata")
    token_usage = None
    if usage:
        token_usage = {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        }

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "token_usage": token_usage,
        "thinking_blocks": thinking_blocks,
    }


def parse_sse_chunk(chunk_data: str) -> dict | None:
    """Parse Google Gemini streaming chunk (each chunk is a full JSON object)."""
    if not chunk_data.strip():
        return None
    try:
        data = json.loads(chunk_data)
    except json.JSONDecodeError:
        return None

    # Google streams full response objects per chunk
    parsed = parse_response(data)
    return {
        "content": parsed["response_text"],
        "finish_reason": parsed["finish_reason"],
        "tool_calls_delta": None,
        "usage": parsed["token_usage"],
        "thinking_blocks": parsed.get("thinking_blocks") or [],
    }


class SSEAssembler:
    def __init__(self):
        self._content_parts: list[str] = []
        self._finish_reason: str | None = None
        self._tool_calls: list[dict] = []
        self._usage: dict | None = None
        self._thinking_blocks: list[dict] = []

    def feed(self, chunk: dict | None) -> bool:
        if chunk is None:
            return False
        if chunk.get("content"):
            self._content_parts.append(chunk["content"])
        if chunk.get("finish_reason"):
            self._finish_reason = chunk["finish_reason"]
        if chunk.get("usage"):
            self._usage = chunk["usage"]
        if chunk.get("thinking_blocks"):
            self._thinking_blocks.extend(chunk["thinking_blocks"])
        return self._finish_reason is not None

    def build_response(self) -> dict:
        return {
            "response_text": "".join(self._content_parts) or None,
            "finish_reason": self._finish_reason,
            "tool_calls": self._tool_calls,
            "token_usage": self._usage,
            "thinking_blocks": self._thinking_blocks or None,
        }


class GoogleProvider:
    name = PROVIDER_NAME

    @staticmethod
    def extract_request_params(body: dict, model: str = "unknown") -> dict:
        return extract_request_params(body, model)

    @staticmethod
    def parse_response(body: dict) -> dict:
        return parse_response(body)

    @staticmethod
    def parse_sse_chunk(chunk_data: str) -> dict | None:
        return parse_sse_chunk(chunk_data)

    @staticmethod
    def new_assembler() -> SSEAssembler:
        return SSEAssembler()
