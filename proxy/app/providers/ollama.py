"""
Ollama API format handler.

Ollama exposes an OpenAI-compatible chat endpoint at:
  POST http://localhost:11434/api/chat
  POST http://localhost:11434/v1/chat/completions  (OpenAI compat)

We handle both formats:
- /ollama/v1/... → OpenAI-compatible (reuse OpenAI parser)
- /ollama/api/chat → Ollama native format

**Known limitations (Ollama support is partial):**

1. **Thinking blocks** — Ollama has no concept of reasoning traces. Extended
   thinking is an Anthropic-specific feature. The ``thinking_blocks`` field in
   LLM_CALL_END events will always be ``None`` for Ollama.

2. **Structured tool_calls** — Ollama's native format returns tool calls
   embedded in the ``message.tool_calls`` list. The OpenAI-compat path maps
   them via the OpenAI parser. However, Ollama's tool-call format varies
   across model versions and may not always populate ``function.name`` and
   ``function.arguments`` correctly. Verify tool-call capture with your
   specific model before relying on tool baseline enforcement.

3. **stop_reason mapping** — Ollama uses ``done_reason`` (e.g. ``"stop"``,
   ``"length"``) instead of OpenAI's ``finish_reason``. The native handler
   maps ``done_reason`` → ``finish_reason``, but some Ollama versions return
   ``null`` or omit the field entirely. Treat stop-reason-dependent logic
   (like tool-call loop protection) as best-effort for Ollama.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .openai import (
    SSEAssembler as OpenAIAssembler,
    extract_request_params as openai_extract,
    parse_response as openai_parse_response,
    parse_sse_chunk as openai_parse_sse_chunk,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "ollama"


def extract_request_params_native(body: dict) -> dict:
    """Parse Ollama /api/chat native format."""
    messages = body.get("messages", [])
    options = body.get("options", {})

    return {
        "model": body.get("model", "unknown"),
        "messages": messages,
        "tools": body.get("tools") or [],
        "temperature": options.get("temperature"),
        "max_tokens": options.get("num_predict"),
        "stream": body.get("stream", True),
        "extra_params": None,
    }


def parse_response_native(body: dict) -> dict:
    """Parse Ollama /api/chat native response."""
    message = body.get("message", {})
    done = body.get("done", False)
    done_reason = body.get("done_reason")

    finish_reason = None
    if done:
        finish_reason = "stop" if done_reason in (None, "stop") else done_reason

    response_text = message.get("content")
    tool_calls = []
    for tc in (message.get("tool_calls") or []):
        func = tc.get("function", {})
        tool_calls.append({
            "id": f"call_{func.get('name', 'fn')}_{id(tc) & 0xFFFF:04x}",
            "name": func.get("name", ""),
            "arguments": func.get("arguments", {}),
        })

    usage = None
    if body.get("prompt_eval_count") or body.get("eval_count"):
        usage = {
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
            "total_tokens": (body.get("prompt_eval_count", 0) or 0) + (body.get("eval_count", 0) or 0),
        }

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "token_usage": usage,
    }


def parse_sse_chunk_native(chunk_data: str) -> dict | None:
    """Parse Ollama native streaming chunk (each chunk is a JSON object)."""
    if not chunk_data.strip():
        return None
    try:
        data = json.loads(chunk_data)
    except json.JSONDecodeError:
        return None

    parsed = parse_response_native(data)
    return {
        "content": parsed["response_text"],
        "finish_reason": parsed["finish_reason"],
        "tool_calls_delta": None,
        "usage": parsed["token_usage"],
    }


class OllamaProvider:
    name = PROVIDER_NAME

    @staticmethod
    def extract_request_params(body: dict, path: str = "") -> dict:
        if "/v1/" in path:
            return openai_extract(body)
        return extract_request_params_native(body)

    @staticmethod
    def parse_response(body: dict, path: str = "") -> dict:
        if "/v1/" in path:
            return openai_parse_response(body)
        return parse_response_native(body)

    @staticmethod
    def parse_sse_chunk(chunk_data: str, path: str = "") -> dict | None:
        if "/v1/" in path:
            return openai_parse_sse_chunk(chunk_data)
        return parse_sse_chunk_native(chunk_data)

    @staticmethod
    def new_assembler() -> OpenAIAssembler:
        return OpenAIAssembler()
