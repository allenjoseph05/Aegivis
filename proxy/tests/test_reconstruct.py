"""Tests for tool call inference from conversation patterns."""
import pytest
from proxy.app.reconstruct import (
    extract_tool_calls_from_response,
    extract_tool_results_from_request,
)


class TestExtractToolCalls:
    def test_no_tool_calls(self):
        event = {
            "event_type": "LLM_CALL_END",
            "payload": {
                "response_text": "Hello!",
                "finish_reason": "stop",
                "tool_calls": [],
            }
        }
        result = extract_tool_calls_from_response(event)
        assert result == []

    def test_single_tool_call(self):
        event = {
            "payload": {
                "tool_calls": [
                    {"id": "call_abc", "name": "search", "arguments": '{"query": "weather"}'}
                ]
            }
        }
        result = extract_tool_calls_from_response(event)
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_multiple_tool_calls(self):
        event = {
            "payload": {
                "tool_calls": [
                    {"id": "call_001", "name": "search", "arguments": "{}"},
                    {"id": "call_002", "name": "calculator", "arguments": "{}"},
                ]
            }
        }
        result = extract_tool_calls_from_response(event)
        assert len(result) == 2

    def test_missing_payload(self):
        result = extract_tool_calls_from_response({})
        assert result == []


class TestExtractToolResults:
    def test_no_tool_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the weather?"},
        ]
        result = extract_tool_results_from_request(messages)
        assert result == []

    def test_single_tool_result(self):
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_01", "name": "weather"}]},
            {"role": "tool", "tool_call_id": "call_01", "content": "Sunny, 72°F", "name": "weather"},
        ]
        result = extract_tool_results_from_request(messages)
        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_01"
        assert result[0]["content"] == "Sunny, 72°F"

    def test_multiple_tool_results(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_01", "content": "Result A"},
            {"role": "tool", "tool_call_id": "call_02", "content": "Result B"},
        ]
        result = extract_tool_results_from_request(messages)
        assert len(result) == 2
        ids = {r["tool_call_id"] for r in result}
        assert ids == {"call_01", "call_02"}

    def test_mixed_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_x", "content": "result"},
        ]
        result = extract_tool_results_from_request(messages)
        assert len(result) == 1
