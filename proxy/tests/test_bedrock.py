"""
Tests for AWS Bedrock provider parser (Phase E2).

All tests are pure unit tests — no AWS credentials or boto3 calls needed.
The gateway (bedrock_gateway.py) is tested via mocking in integration tests.
"""
from __future__ import annotations

import json

import pytest

from proxy.app.providers.bedrock import (
    assemble_converse_stream,
    detect_model_family,
    extract_converse_request,
    parse_converse_response,
    parse_invoke_request,
    parse_invoke_response,
    rebuild_converse_body,
    _flatten_content_blocks,
    _parse_llama_prompt,
    _parse_mistral_prompt,
)


# ── detect_model_family ───────────────────────────────────────────────────────

class TestDetectModelFamily:
    def test_anthropic_claude(self):
        assert detect_model_family("anthropic.claude-3-5-sonnet-20241022-v2:0") == "anthropic"

    def test_anthropic_short(self):
        assert detect_model_family("anthropic.claude-instant-v1") == "anthropic"

    def test_meta_llama3(self):
        assert detect_model_family("meta.llama3-8b-instruct-v1:0") == "meta"

    def test_meta_llama2(self):
        assert detect_model_family("meta.llama2-70b-chat-v1") == "meta"

    def test_amazon_titan(self):
        assert detect_model_family("amazon.titan-text-express-v1") == "amazon"

    def test_amazon_nova(self):
        assert detect_model_family("amazon.nova-pro-v1:0") == "amazon"

    def test_mistral(self):
        assert detect_model_family("mistral.mixtral-8x7b-instruct-v0:1") == "mistral"

    def test_cohere(self):
        assert detect_model_family("cohere.command-r-plus-v1:0") == "cohere"

    def test_cross_region_inference_prefix(self):
        # Cross-region profiles: "us.anthropic.claude-..."
        assert detect_model_family("us.anthropic.claude-3-5-sonnet-20241022-v2:0") == "anthropic"

    def test_unknown(self):
        assert detect_model_family("some.unknown-model-v1") == "unknown"


# ── _flatten_content_blocks ───────────────────────────────────────────────────

class TestFlattenContentBlocks:
    def test_plain_string(self):
        assert _flatten_content_blocks("hello") == "hello"

    def test_text_block(self):
        result = _flatten_content_blocks([{"text": "Hello world"}])
        assert result == "Hello world"

    def test_multiple_text_blocks(self):
        blocks = [{"text": "foo"}, {"text": "bar"}]
        assert _flatten_content_blocks(blocks) == "foo bar"

    def test_tool_result_block(self):
        blocks = [{"toolResult": {"content": [{"text": "result text"}]}}]
        assert "result text" in _flatten_content_blocks(blocks)

    def test_image_block(self):
        result = _flatten_content_blocks([{"image": {"format": "png", "source": {}}}])
        assert "[image]" in result

    def test_mixed_blocks(self):
        blocks = [{"text": "Before"}, {"image": {}}, {"text": "After"}]
        result = _flatten_content_blocks(blocks)
        assert "Before" in result
        assert "After" in result


# ── extract_converse_request ──────────────────────────────────────────────────

class TestExtractConverseRequest:
    MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_basic_message(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"text": "Hello"}]}
            ]
        }
        result = extract_converse_request(self.MODEL, body)
        assert result["model"] == self.MODEL
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["stream"] is False

    def test_system_prompt_injected_first(self):
        body = {
            "system": [{"text": "You are helpful."}],
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        }
        result = extract_converse_request(self.MODEL, body)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."
        assert result["messages"][1]["role"] == "user"

    def test_system_string(self):
        body = {
            "system": "Direct string system prompt",
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        }
        result = extract_converse_request(self.MODEL, body)
        assert result["messages"][0]["content"] == "Direct string system prompt"

    def test_tools_normalized(self):
        body = {
            "messages": [{"role": "user", "content": [{"text": "search"}]}],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "web_search",
                            "description": "Search the web",
                            "inputSchema": {"json": {"type": "object", "properties": {"q": {"type": "string"}}}},
                        }
                    }
                ]
            },
        }
        result = extract_converse_request(self.MODEL, body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "web_search"
        assert result["tools"][0]["description"] == "Search the web"
        assert result["tools"][0]["parameters"] is not None

    def test_inference_config(self):
        body = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 1024, "temperature": 0.5},
        }
        result = extract_converse_request(self.MODEL, body)
        assert result["max_tokens"] == 1024
        assert result["temperature"] == 0.5

    def test_no_messages(self):
        result = extract_converse_request(self.MODEL, {})
        assert result["messages"] == []
        assert result["tools"] == []


# ── parse_converse_response ───────────────────────────────────────────────────

class TestParseConverseResponse:
    def test_text_response(self):
        body = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello there!"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        result = parse_converse_response(body)
        assert result["response_text"] == "Hello there!"
        assert result["finish_reason"] == "end_turn"
        assert result["tool_calls"] == []
        assert result["token_usage"]["input_tokens"] == 10
        assert result["token_usage"]["output_tokens"] == 5
        assert result["token_usage"]["total_tokens"] == 15

    def test_tool_use_response(self):
        body = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {
                            "toolUseId": "tu-123",
                            "name": "web_search",
                            "input": {"q": "python"},
                        }}
                    ],
                }
            },
            "stopReason": "tool_use",
        }
        result = parse_converse_response(body)
        assert result["response_text"] is None
        assert result["finish_reason"] == "tool_use"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "tu-123"
        assert tc["name"] == "web_search"
        assert json.loads(tc["arguments"]) == {"q": "python"}

    def test_mixed_text_and_tool(self):
        body = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me search for that."},
                        {"toolUse": {"toolUseId": "tu-1", "name": "search", "input": {}}},
                    ],
                }
            },
            "stopReason": "tool_use",
        }
        result = parse_converse_response(body)
        assert result["response_text"] == "Let me search for that."
        assert len(result["tool_calls"]) == 1

    def test_no_usage(self):
        body = {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
        }
        result = parse_converse_response(body)
        assert result["token_usage"] is None

    def test_empty_body(self):
        result = parse_converse_response({})
        assert result["response_text"] is None
        assert result["finish_reason"] == "end_turn"
        assert result["tool_calls"] == []


# ── assemble_converse_stream ──────────────────────────────────────────────────

class TestAssembleConverseStream:
    """Test EventStream assembly by simulating boto3 EventStream event dicts."""

    def _make_stream(self, events: list[dict]):
        """Simulate a boto3 EventStream as an iterable of event dicts."""
        return iter(events)

    def test_text_stream(self):
        events = [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " world"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}}},
        ]
        result = assemble_converse_stream(self._make_stream(events))
        assert result["response_text"] == "Hello world"
        assert result["finish_reason"] == "end_turn"
        assert result["token_usage"]["total_tokens"] == 7

    def test_tool_use_stream(self):
        events = [
            {"contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tu-1", "name": "search"}},
            }},
            {"contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"q":"py'}},
            }},
            {"contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": 'thon"}'}},
            }},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
        result = assemble_converse_stream(self._make_stream(events))
        assert result["finish_reason"] == "tool_use"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "search"
        assert json.loads(tc["arguments"]) == {"q": "python"}

    def test_empty_stream(self):
        result = assemble_converse_stream(iter([]))
        assert result["response_text"] is None
        assert result["tool_calls"] == []

    def test_mixed_text_and_tool(self):
        events = [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Let me search."}}},
            {"contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "tu-2", "name": "lookup"}},
            }},
            {"contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"id":1}'}},
            }},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
        result = assemble_converse_stream(self._make_stream(events))
        assert result["response_text"] == "Let me search."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "lookup"


# ── parse_invoke_request ──────────────────────────────────────────────────────

class TestParseInvokeRequest:
    def test_claude_delegates_to_anthropic_parser(self):
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 512,
        }
        result = parse_invoke_request(model_id, body)
        assert result["model"] == model_id
        assert result["max_tokens"] == 512
        assert any(m["role"] == "user" for m in result["messages"])

    def test_llama_prompt_parsed(self):
        model_id = "meta.llama3-8b-instruct-v1:0"
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            "What is 2+2?<|eot_id|>"
        )
        body = {"prompt": prompt, "max_gen_len": 256, "temperature": 0.7}
        result = parse_invoke_request(model_id, body)
        assert result["model"] == model_id
        assert result["max_tokens"] == 256
        assert any(m["content"] == "What is 2+2?" for m in result["messages"])

    def test_llama_fallback_on_unparseable_prompt(self):
        model_id = "meta.llama3-8b-instruct-v1:0"
        body = {"prompt": "Plain text prompt", "max_gen_len": 100}
        result = parse_invoke_request(model_id, body)
        assert result["messages"][0]["content"] == "Plain text prompt"

    def test_mistral_prompt_parsed(self):
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        body = {"prompt": "<s>[INST] Hello there [/INST] Hi! [INST] follow up [/INST]", "max_tokens": 200}
        result = parse_invoke_request(model_id, body)
        assert result["model"] == model_id
        assert any(m["role"] == "user" for m in result["messages"])

    def test_amazon_titan(self):
        model_id = "amazon.titan-text-express-v1"
        body = {
            "inputText": "Tell me a joke",
            "textGenerationConfig": {"maxTokenCount": 300, "temperature": 0.8},
        }
        result = parse_invoke_request(model_id, body)
        assert result["messages"][0]["content"] == "Tell me a joke"
        assert result["max_tokens"] == 300
        assert result["temperature"] == 0.8

    def test_unknown_family_returns_raw(self):
        model_id = "some.unknown-model-v1"
        body = {"some_key": "some_value"}
        result = parse_invoke_request(model_id, body)
        assert result["model"] == model_id
        assert len(result["messages"]) == 1


# ── parse_invoke_response ─────────────────────────────────────────────────────

class TestParseInvokeResponse:
    def test_claude_response(self):
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        body = {
            "type": "message",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "The answer is 42."}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = parse_invoke_response(model_id, body)
        assert result["response_text"] == "The answer is 42."
        # Anthropic parser maps end_turn → stop for canonical consistency
        assert result["finish_reason"] == "stop"

    def test_llama_response(self):
        model_id = "meta.llama3-8b-instruct-v1:0"
        body = {
            "generation": "Four.",
            "prompt_token_count": 20,
            "generation_token_count": 5,
            "stop_reason": "stop",
        }
        result = parse_invoke_response(model_id, body)
        assert result["response_text"] == "Four."
        assert result["finish_reason"] == "stop"
        assert result["token_usage"]["total_tokens"] == 25

    def test_mistral_response(self):
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        body = {"outputs": [{"text": "Bonjour!", "stop_reason": "stop"}]}
        result = parse_invoke_response(model_id, body)
        assert result["response_text"] == "Bonjour!"
        assert result["finish_reason"] == "stop"

    def test_amazon_titan_response(self):
        model_id = "amazon.titan-text-express-v1"
        body = {"results": [{"outputText": "Here is the joke.", "completionReason": "FINISH"}]}
        result = parse_invoke_response(model_id, body)
        assert result["response_text"] == "Here is the joke."

    def test_empty_mistral_outputs(self):
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        result = parse_invoke_response(model_id, {"outputs": []})
        assert result["response_text"] is None


# ── rebuild_converse_body ─────────────────────────────────────────────────────

class TestRebuildConverseBody:
    def test_round_trip_basic(self):
        original = {
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            "inferenceConfig": {"maxTokens": 1000},
        }
        canonical = {
            "messages": [{"role": "user", "content": "Hello (modified)"}],
        }
        rebuilt = rebuild_converse_body(canonical, original)
        assert rebuilt["messages"][0]["role"] == "user"
        assert rebuilt["messages"][0]["content"][0]["text"] == "Hello (modified)"
        # Non-message fields preserved
        assert rebuilt["inferenceConfig"] == {"maxTokens": 1000}

    def test_system_prompt_moved_to_top_level(self):
        original = {}
        canonical = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        }
        rebuilt = rebuild_converse_body(canonical, original)
        assert rebuilt["system"] == [{"text": "Be helpful."}]
        assert len(rebuilt["messages"]) == 1
        assert rebuilt["messages"][0]["role"] == "user"


# ── Prompt parsers ────────────────────────────────────────────────────────────

class TestLlamaPromptParser:
    def test_single_user_turn(self):
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            "What is AI?<|eot_id|>"
        )
        messages = _parse_llama_prompt(prompt)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "What is AI?"}

    def test_multi_turn(self):
        prompt = (
            "<|start_header_id|>user<|end_header_id|>\n"
            "Hello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "Hi there<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            "Follow up<|eot_id|>"
        )
        messages = _parse_llama_prompt(prompt)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_empty_returns_empty(self):
        assert _parse_llama_prompt("") == []


class TestMistralPromptParser:
    def test_single_turn(self):
        prompt = "<s>[INST] Tell me a joke [/INST] Why did the chicken... [INST]"
        messages = _parse_mistral_prompt(prompt)
        assert any(m["role"] == "user" and "joke" in m["content"] for m in messages)

    def test_empty_returns_empty(self):
        assert _parse_mistral_prompt("") == []
