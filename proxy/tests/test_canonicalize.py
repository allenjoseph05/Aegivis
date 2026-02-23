"""Tests for event canonicalization."""
import pytest
from unittest.mock import patch, MagicMock
from proxy.app.canonicalize import (
    make_llm_call_start,
    make_llm_call_end,
    make_system_error,
    make_agent_finish,
    make_checkpoint,
)
from proxy.app.models import EventType

COMMON = dict(
    session_id="sess_test01",
    org_id="org_test",
    agent_id="agent_test",
    provider="openai",
    model="gpt-4o",
    sequence_number=0,
    previous_hash="GENESIS_sess_test01",
)


class TestMakeLLMCallStart:
    def test_required_fields_present(self):
        event = make_llm_call_start(
            **COMMON,
            messages=[{"role": "user", "content": "hello"}],
            run_id="run-001",
        )
        assert event["event_type"] == EventType.LLM_CALL_START
        assert event["schema_version"] == "1.0"
        assert event["session_id"] == "sess_test01"
        assert event["current_hash"] == ""
        assert event["previous_hash"] == "GENESIS_sess_test01"

    def test_event_id_is_ulid_string(self):
        event = make_llm_call_start(
            **COMMON,
            messages=[],
            run_id="run-001",
        )
        assert isinstance(event["event_id"], str)
        assert len(event["event_id"]) == 26  # ULID is 26 chars

    def test_messages_normalized(self):
        event = make_llm_call_start(
            **COMMON,
            messages=[{"role": "user", "content": "hello", "extra": "ignored"}],
            run_id="run-001",
        )
        msg = event["payload"]["messages"][0]
        assert "role" in msg
        assert "content" in msg
        assert "extra" not in msg

    def test_tools_normalized_flat_format(self):
        event = make_llm_call_start(
            **COMMON,
            messages=[],
            tools=[{"name": "search", "description": "Search the web", "parameters": {}}],
            run_id="run-001",
        )
        tools = event["payload"]["tools_available"]
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_tools_normalized_openai_wrapper_format(self):
        """OpenAI sends tools as {"type": "function", "function": {...}}"""
        event = make_llm_call_start(
            **COMMON,
            messages=[],
            tools=[{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object"},
                }
            }],
            run_id="run-001",
        )
        tools = event["payload"]["tools_available"]
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"

    def test_temperature_and_max_tokens(self):
        event = make_llm_call_start(
            **COMMON,
            messages=[],
            temperature=0.7,
            max_tokens=1024,
            run_id="run-001",
        )
        assert event["payload"]["temperature"] == 0.7
        assert event["payload"]["max_tokens"] == 1024

    def test_run_id_auto_generated_if_none(self):
        event = make_llm_call_start(
            **COMMON,
            messages=[],
        )
        assert isinstance(event["run_id"], str)
        assert len(event["run_id"]) > 0


class TestMakeLLMCallEnd:
    def test_tool_calls_normalized(self):
        event = make_llm_call_end(
            **COMMON,
            run_id="run-001",
            response_text=None,
            finish_reason="tool_calls",
            tool_calls=[
                {"id": "call_123", "name": "search", "arguments": '{"query": "weather"}'},
            ],
            token_usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            latency_ms=342.5,
        )
        assert event["event_type"] == EventType.LLM_CALL_END
        tcs = event["payload"]["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["name"] == "search"

    def test_token_usage_normalized_anthropic_style(self):
        """Anthropic uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens."""
        event = make_llm_call_end(
            **COMMON,
            run_id="run-001",
            response_text="hi",
            finish_reason="stop",
            tool_calls=[],
            token_usage={"input_tokens": 30, "output_tokens": 20},
            latency_ms=100,
        )
        usage = event["payload"]["token_usage"]
        assert usage["prompt_tokens"] == 30
        assert usage["completion_tokens"] == 20

    def test_no_tool_calls(self):
        event = make_llm_call_end(
            **COMMON,
            run_id="run-001",
            response_text="Hello!",
            finish_reason="stop",
            tool_calls=[],
            token_usage=None,
            latency_ms=50,
        )
        assert event["payload"]["tool_calls"] == []
        assert event["payload"]["response_text"] == "Hello!"


class TestMakeSystemError:
    def test_fields_present(self):
        event = make_system_error(
            **COMMON,
            run_id="run-001",
            error_message="Rate limit exceeded",
            http_status=429,
        )
        assert event["event_type"] == EventType.SYSTEM_ERROR
        assert event["payload"]["error_message"] == "Rate limit exceeded"
        assert event["payload"]["http_status"] == 429

    def test_error_code_optional(self):
        event = make_system_error(
            **COMMON,
            run_id="run-001",
            error_message="Internal server error",
        )
        assert event["payload"]["error_code"] is None


class TestMakeAgentFinish:
    def test_fields_present(self):
        event = make_agent_finish(
            **COMMON,
            run_id="run-001",
            final_output="Task complete.",
            total_llm_calls=5,
            total_tool_calls=3,
            session_duration_ms=2500.0,
        )
        assert event["event_type"] == EventType.AGENT_FINISH
        assert event["payload"]["total_llm_calls"] == 5
        assert event["payload"]["total_tool_calls"] == 3
        assert event["payload"]["final_output"] == "Task complete."


class TestMakeCheckpoint:
    def test_checkpoint_fields(self):
        event = make_checkpoint(
            **COMMON,
            run_id="run-001",
            merkle_root="a" * 64,
            events_covered=1000,
            from_sequence=0,
            to_sequence=999,
        )
        assert event["event_type"] == EventType.CHECKPOINT
        assert event["payload"]["events_covered"] == 1000
        assert event["payload"]["merkle_root"] == "a" * 64
