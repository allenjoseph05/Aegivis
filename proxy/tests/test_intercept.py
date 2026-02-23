"""Tests for provider request/response parsing (intercept layer)."""
import pytest
from proxy.app.providers.openai import (
    OpenAIProvider,
    SSEAssembler as OpenAIAssembler,
    parse_response as openai_parse,
    parse_sse_chunk as openai_sse,
)
from proxy.app.providers.anthropic import (
    AnthropicProvider,
    SSEAssembler as AnthropicAssembler,
    extract_request_params as anthropic_extract,
    parse_response as anthropic_parse,
)


class TestOpenAIRequestParsing:
    def test_basic_request(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        params = OpenAIProvider.extract_request_params(body)
        assert params["model"] == "gpt-4o"
        assert len(params["messages"]) == 1
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 512
        assert params["tools"] == []

    def test_request_with_tools(self):
        body = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "search", "description": "Search web", "parameters": {}}}
            ]
        }
        params = OpenAIProvider.extract_request_params(body)
        assert len(params["tools"]) == 1

    def test_streaming_flag(self):
        body = {"model": "gpt-4o", "messages": [], "stream": True}
        params = OpenAIProvider.extract_request_params(body)
        assert params["stream"] is True


class TestOpenAIResponseParsing:
    def test_simple_text_response(self):
        body = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_parse(body)
        assert result["response_text"] == "Hello there!"
        assert result["finish_reason"] == "stop"
        assert result["tool_calls"] == []
        assert result["token_usage"]["total_tokens"] == 15

    def test_tool_call_response(self):
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "weather"}'},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        }
        result = openai_parse(body)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"
        assert result["finish_reason"] == "tool_calls"

    def test_empty_choices(self):
        result = openai_parse({"choices": []})
        assert result["response_text"] is None
        assert result["tool_calls"] == []


class TestOpenAISSEParsing:
    def test_content_chunk(self):
        chunk = openai_sse('{"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}')
        assert chunk is not None
        assert chunk["content"] == "Hello"
        assert chunk["finish_reason"] is None

    def test_finish_chunk(self):
        chunk = openai_sse('{"choices": [{"delta": {}, "finish_reason": "stop"}]}')
        assert chunk is not None
        assert chunk["finish_reason"] == "stop"

    def test_done_sentinel(self):
        chunk = openai_sse("[DONE]")
        assert chunk is None

    def test_invalid_json(self):
        chunk = openai_sse("invalid json")
        assert chunk is None

    def test_empty_string(self):
        chunk = openai_sse("")
        assert chunk is None


class TestOpenAISSEAssembler:
    def test_assemble_multiple_chunks(self):
        assembler = OpenAIAssembler()
        chunks = [
            '{"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            '{"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}',
            '{"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        ]
        for chunk_str in chunks[:-1]:
            done = assembler.feed(openai_sse(chunk_str))
            assert done is False
        done = assembler.feed(openai_sse(chunks[-1]))
        assert done is True

        response = assembler.build_response()
        assert response["response_text"] == "Hello world"
        assert response["finish_reason"] == "stop"

    def test_streaming_tool_calls(self):
        assembler = OpenAIAssembler()
        chunks_data = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_abc", "function": {"name": "search", "arguments": ""}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "test"}'}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        import json
        for d in chunks_data[:-1]:
            assembler.feed(openai_sse(json.dumps(d)))
        assembler.feed(openai_sse(json.dumps(chunks_data[-1])))

        response = assembler.build_response()
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["name"] == "search"
        assert '"q"' in response["tool_calls"][0]["arguments"]


class TestAnthropicRequestParsing:
    def test_system_message_promoted(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
        }
        params = anthropic_extract(body)
        assert params["model"] == "claude-3-5-sonnet-20241022"
        # System message should be first
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][0]["content"] == "You are helpful."
        assert params["messages"][1]["role"] == "user"

    def test_tool_input_schema_normalized(self):
        body = {
            "model": "claude-3",
            "messages": [],
            "tools": [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}],
            "max_tokens": 100,
        }
        params = anthropic_extract(body)
        assert params["tools"][0]["parameters"] == {"type": "object"}


class TestAnthropicResponseParsing:
    def test_text_response(self):
        body = {
            "content": [{"type": "text", "text": "Hello there!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_parse(body)
        assert result["response_text"] == "Hello there!"
        assert result["finish_reason"] == "stop"
        assert result["token_usage"]["prompt_tokens"] == 10

    def test_tool_use_response(self):
        body = {
            "content": [
                {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "weather"}}
            ],
            "stop_reason": "tool_use",
        }
        result = anthropic_parse(body)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"
        assert result["tool_calls"][0]["arguments"] == {"query": "weather"}
