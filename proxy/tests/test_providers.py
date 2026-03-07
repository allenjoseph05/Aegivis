"""
Tests for provider parsers (Anthropic, OpenAI, Google, Azure, Ollama).

Covers:
- SSEAssembler: thinking block accumulation (streaming)
- SSEAssembler: thinking_delta without prior thinking_block_start (guard)
- SSEAssembler: tool argument streaming via tool_calls_delta + tool_input_delta
- parse_response: thinking blocks in non-streaming path
- _THINKING_MAX_CHARS: truncation at settings.reasoning_trace_max_chars
- OpenAI parse_response: tool_calls normalization
- Google Gemini parse_response: functionCall format → canonical tool_calls
- Azure parse_response: delegates to OpenAI parser (same format)
- Ollama parse_response (native): message.tool_calls → canonical tool_calls
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from proxy.app.providers.anthropic import (
    SSEAssembler,
    parse_response as anthropic_parse_response,
    parse_sse_chunk,
)
from proxy.app.providers.openai import parse_response as openai_parse_response
from proxy.app.providers.google import parse_response as google_parse_response
from proxy.app.providers.ollama import parse_response_native as ollama_parse_response_native


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_chunks(assembler: SSEAssembler, chunks: list[str | dict]) -> bool:
    """Feed a list of raw JSON strings (as if from SSE stream) or pre-parsed dicts."""
    done = False
    for item in chunks:
        if isinstance(item, str):
            chunk = parse_sse_chunk(item)
        else:
            chunk = item
        result = assembler.feed(chunk)
        if result:
            done = True
    return done


# ---------------------------------------------------------------------------
# SSEAssembler — thinking block tests
# ---------------------------------------------------------------------------

class TestSSEAssemblerThinkingBlocks:
    def test_thinking_block_full_sequence(self):
        """thinking_block_start → N thinking_deltas → signature_delta → stop."""
        asm = SSEAssembler()

        # content_block_start for thinking block
        c1 = json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}})
        # 3 thinking deltas
        c2 = json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "First "}})
        c3 = json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Second "}})
        c4 = json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Third"}})
        # signature
        c5 = json.dumps({"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "sig_abc123"}})
        # stop
        c6 = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}})

        done = _feed_chunks(asm, [c1, c2, c3, c4, c5, c6])
        assert done is True

        response = asm.build_response()
        assert response["finish_reason"] == "stop"
        assert len(response["thinking_blocks"]) == 1
        block = response["thinking_blocks"][0]
        assert block["thinking"] == "First Second Third"
        assert block["signature"] == "sig_abc123"

    def test_thinking_block_without_start_event(self):
        """thinking_delta arriving without prior thinking_block_start — guard activates."""
        asm = SSEAssembler()

        # No content_block_start — delta arrives directly (guard test)
        c1 = json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Orphan thought"}})
        c2 = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}})

        _feed_chunks(asm, [c1, c2])
        response = asm.build_response()

        assert len(response["thinking_blocks"]) == 1
        assert response["thinking_blocks"][0]["thinking"] == "Orphan thought"

    def test_no_thinking_blocks_when_absent(self):
        """Normal text response with no thinking produces empty thinking_blocks."""
        asm = SSEAssembler()

        c1 = json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello!"}})
        c2 = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}})

        _feed_chunks(asm, [c1, c2])
        response = asm.build_response()

        assert response["response_text"] == "Hello!"
        assert response["thinking_blocks"] == []

    def test_thinking_max_chars_truncation_streaming(self):
        """Thinking content longer than reasoning_trace_max_chars is truncated."""
        long_thinking = "X" * 5000  # longer than any reasonable limit

        asm = SSEAssembler()
        c1 = json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": long_thinking}})
        c2 = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}})

        _feed_chunks(asm, [c1, c2])
        response = asm.build_response()

        block = response["thinking_blocks"][0]
        # Must be truncated — never longer than 5000 chars (default 4000)
        assert len(block["thinking"]) <= 5000
        # And must be truncated to exactly reasoning_trace_max_chars
        from proxy.app.config import settings
        assert len(block["thinking"]) == settings.reasoning_trace_max_chars

    def test_thinking_block_signature_none_when_absent(self):
        """If no signature_delta arrives, signature is None."""
        asm = SSEAssembler()

        c1 = json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Some thought"}})
        c2 = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}})

        _feed_chunks(asm, [c1, c2])
        response = asm.build_response()

        assert response["thinking_blocks"][0]["signature"] is None


# ---------------------------------------------------------------------------
# SSEAssembler — tool argument streaming
# ---------------------------------------------------------------------------

class TestSSEAssemblerToolStreaming:
    def test_tool_args_via_tool_calls_delta(self):
        """Tool arguments accumulated via tool_calls_delta (function.arguments field)."""
        asm = SSEAssembler()

        # content_block_start for tool_use block
        c1 = json.dumps({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "tool_use", "id": "call_01", "name": "search"}
        })
        # arguments arrive via tool_calls_delta with function.arguments
        c2 = json.dumps({
            "type": "content_block_delta", "delta": {"type": "text_delta", "text": ""},
            # Note: parse_sse_chunk only recognizes tool_calls_delta from content_block_start,
            # so we feed the pre-parsed dict directly for the arguments chunk
        })
        # Feed pre-parsed tool_calls_delta (what the assembler actually receives)
        arg_chunk = {
            "content": None, "finish_reason": None, "usage": None,
            "tool_calls_delta": [{"index": 0, "id": "call_01", "function": {"name": "search", "arguments": '{"q":'}}],
        }
        arg_chunk2 = {
            "content": None, "finish_reason": None, "usage": None,
            "tool_calls_delta": [{"index": 0, "id": "", "function": {"name": "", "arguments": '"hello"}'}}],
        }
        stop = json.dumps({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}})

        asm.feed(parse_sse_chunk(c1))  # registers tool block
        asm.feed(arg_chunk)
        asm.feed(arg_chunk2)
        asm.feed(parse_sse_chunk(stop))

        response = asm.build_response()
        assert len(response["tool_calls"]) == 1
        tc = response["tool_calls"][0]
        assert tc["name"] == "search"
        assert tc["id"] == "call_01"
        assert tc["arguments"] == '{"q":"hello"}'

    def test_tool_input_delta_accumulation(self):
        """Tool arguments accumulated via tool_input_delta (input_json_delta path)."""
        asm = SSEAssembler()

        # Register the tool block via content_block_start
        start_chunk = {
            "content": None, "finish_reason": None, "usage": None,
            "tool_calls_delta": [{"index": 0, "id": "call_02", "function": {"name": "calc", "arguments": ""}}],
        }
        input_chunk1 = {"tool_input_delta": '{"num":', "content": None, "finish_reason": None, "usage": None}
        input_chunk2 = {"tool_input_delta": "42}", "content": None, "finish_reason": None, "usage": None}
        stop = json.dumps({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}})

        asm.feed(start_chunk)
        asm.feed(input_chunk1)
        asm.feed(input_chunk2)
        asm.feed(parse_sse_chunk(stop))

        response = asm.build_response()
        tc = response["tool_calls"][0]
        assert tc["name"] == "calc"
        assert tc["arguments"] == '{"num":42}'


# ---------------------------------------------------------------------------
# parse_response (non-streaming) — thinking blocks
# ---------------------------------------------------------------------------

class TestAnthropicParseResponse:
    def test_thinking_block_extracted(self):
        body = {
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this.", "signature": "sig_xyz"},
                {"type": "text", "text": "The answer is 42."},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_parse_response(body)
        assert result["response_text"] == "The answer is 42."
        assert len(result["thinking_blocks"]) == 1
        assert result["thinking_blocks"][0]["thinking"] == "Let me reason about this."
        assert result["thinking_blocks"][0]["signature"] == "sig_xyz"

    def test_thinking_max_chars_truncation_non_streaming(self):
        """Long thinking content is truncated in non-streaming parse_response."""
        from proxy.app.config import settings
        long_text = "A" * (settings.reasoning_trace_max_chars + 1000)
        body = {
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": long_text, "signature": None}],
        }
        result = anthropic_parse_response(body)
        assert len(result["thinking_blocks"][0]["thinking"]) == settings.reasoning_trace_max_chars

    def test_no_thinking_blocks_when_absent(self):
        body = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
        }
        result = anthropic_parse_response(body)
        assert result["thinking_blocks"] == []

    def test_tool_use_block_parsed(self):
        body = {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "call_abc", "name": "web_search", "input": {"query": "test"}},
            ],
        }
        result = anthropic_parse_response(body)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "web_search"
        assert tc["arguments"] == {"query": "test"}

    def test_finish_reason_mapping(self):
        for stop_reason, expected in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
        ]:
            body = {"stop_reason": stop_reason, "content": []}
            result = anthropic_parse_response(body)
            assert result["finish_reason"] == expected, f"Failed for {stop_reason}"


# ---------------------------------------------------------------------------
# OpenAI parse_response
# ---------------------------------------------------------------------------

class TestOpenAIParseResponse:
    def test_basic_text_response(self):
        body = {
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
        }
        result = openai_parse_response(body)
        assert result["response_text"] == "Hi!"
        assert result["finish_reason"] == "stop"
        assert result["tool_calls"] == []

    def test_tool_call_normalized(self):
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        result = openai_parse_response(body)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["name"] == "search"
        assert tc["arguments"] == '{"q": "test"}'

    def test_empty_choices_returns_none_fields(self):
        result = openai_parse_response({"choices": []})
        assert result["response_text"] is None
        assert result["finish_reason"] is None
        assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Google Gemini — functionCall format
# ---------------------------------------------------------------------------

class TestGoogleParseResponse:
    """Google Gemini uses parts[].functionCall instead of tool_calls."""

    def test_function_call_parsed(self):
        body = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris", "unit": "celsius"},
                        }
                    }]
                },
                "finishReason": "STOP",
            }]
        }
        result = google_parse_response(body)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert tc["arguments"] == {"city": "Paris", "unit": "celsius"}
        assert tc["id"].startswith("call_")

    def test_multiple_function_calls(self):
        body = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "fn_a", "args": {"x": 1}}},
                        {"functionCall": {"name": "fn_b", "args": {"y": 2}}},
                    ]
                },
                "finishReason": "STOP",
            }]
        }
        result = google_parse_response(body)
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "fn_a"
        assert result["tool_calls"][1]["name"] == "fn_b"

    def test_text_response_no_tool_calls(self):
        body = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello, world!"}]},
                "finishReason": "STOP",
            }]
        }
        result = google_parse_response(body)
        assert result["response_text"] == "Hello, world!"
        assert result["tool_calls"] == []
        assert result["finish_reason"] == "stop"

    def test_empty_candidates_returns_defaults(self):
        result = google_parse_response({"candidates": []})
        assert result["response_text"] is None
        assert result["tool_calls"] == []
        assert result["finish_reason"] is None

    def test_finish_reason_mapping(self):
        for gemini_reason, expected in [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
        ]:
            body = {
                "candidates": [{
                    "content": {"parts": [{"text": "x"}]},
                    "finishReason": gemini_reason,
                }]
            }
            result = google_parse_response(body)
            assert result["finish_reason"] == expected

    def test_token_usage_extracted(self):
        body = {
            "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 5,
                "totalTokenCount": 25,
            },
        }
        result = google_parse_response(body)
        usage = result["token_usage"]
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 25


# ---------------------------------------------------------------------------
# Azure — same format as OpenAI, delegates to OpenAI parser
# ---------------------------------------------------------------------------

class TestAzureParseResponse:
    """Azure uses identical OpenAI message format; AzureProvider delegates to openai_parse_response."""

    def test_azure_uses_openai_parser(self):
        """AzureProvider.parse_response is the same function as openai's."""
        from proxy.app.providers.azure import AzureProvider

        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_az_001",
                        "function": {"name": "list_blobs", "arguments": '{"container": "data"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
        result = AzureProvider.parse_response(body)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "list_blobs"
        assert tc["id"] == "call_az_001"

    def test_azure_text_response(self):
        from proxy.app.providers.azure import AzureProvider

        body = {
            "choices": [{
                "message": {"role": "assistant", "content": "Azure says hi"},
                "finish_reason": "stop",
            }]
        }
        result = AzureProvider.parse_response(body)
        assert result["response_text"] == "Azure says hi"
        assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Ollama native — message.tool_calls format
# ---------------------------------------------------------------------------

class TestOllamaParseResponseNative:
    """Ollama /api/chat native format uses message.tool_calls[].function."""

    def test_tool_call_parsed(self):
        body = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Berlin"},
                    }
                }]
            },
            "done": True,
            "done_reason": "stop",
        }
        result = ollama_parse_response_native(body)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert tc["arguments"] == {"location": "Berlin"}
        assert tc["id"].startswith("call_")

    def test_no_tool_calls(self):
        body = {
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "done_reason": "stop",
        }
        result = ollama_parse_response_native(body)
        assert result["response_text"] == "Hello!"
        assert result["tool_calls"] == []
        assert result["finish_reason"] == "stop"

    def test_done_reason_maps_to_stop(self):
        body = {"message": {"content": "x"}, "done": True, "done_reason": None}
        result = ollama_parse_response_native(body)
        assert result["finish_reason"] == "stop"

    def test_not_done_finish_reason_is_none(self):
        body = {"message": {"content": "partial"}, "done": False}
        result = ollama_parse_response_native(body)
        assert result["finish_reason"] is None

    def test_token_usage_extracted(self):
        body = {
            "message": {"content": "ok"},
            "done": True,
            "prompt_eval_count": 15,
            "eval_count": 8,
        }
        result = ollama_parse_response_native(body)
        usage = result["token_usage"]
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 8
        assert usage["total_tokens"] == 23
