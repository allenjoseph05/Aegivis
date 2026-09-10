"""
Tests for proxy.app.security.response_hold — Response-Hold HITL Gate (Phase 19).

Test philosophy:
  - poll_hold_gate: approved, denied, expired, timeout, error-recovery.
  - poll_hold_gate: latency_ms is measured; decision field set correctly.
  - synthesize_denial: structurally valid Anthropic and OpenAI responses.
  - synthesize_denial: no tool_calls/tool_use in either format.
  - synthesize_denial: JSON-serialisable; stop_reason/finish_reason correct.
  - synthesize_denial: custom message overrides default.
  - Edge cases: unreachable backend (all polls error) → timeout decision.
"""
from __future__ import annotations

import json
import pytest
import respx
import httpx

from app.security.response_hold import (
    HoldDecision,
    _DENIAL_TEXT,
    _anthropic_denial,
    _openai_denial,
    poll_hold_gate,
    synthesize_denial,
)

_BACKEND = "http://test-backend:8000"
_API_KEY = "test-key"
_APPROVAL_ID = "appr_abc123"
_URL = f"{_BACKEND}/v1/approvals/{_APPROVAL_ID}"


# ---------------------------------------------------------------------------
# poll_hold_gate — decision outcomes
# ---------------------------------------------------------------------------

class TestPollHoldGateDecisions:
    @pytest.mark.asyncio
    @respx.mock
    async def test_approved_immediately(self):
        respx.get(_URL).mock(
            return_value=httpx.Response(200, json={"status": "approved"})
        )
        decision = await poll_hold_gate(_APPROVAL_ID, 30, _BACKEND, _API_KEY)
        assert decision.approved is True
        assert decision.decision == "approved"
        assert decision.approval_id == _APPROVAL_ID

    @pytest.mark.asyncio
    @respx.mock
    async def test_denied_immediately(self):
        respx.get(_URL).mock(
            return_value=httpx.Response(200, json={"status": "denied"})
        )
        decision = await poll_hold_gate(_APPROVAL_ID, 30, _BACKEND, _API_KEY)
        assert decision.approved is False
        assert decision.decision == "denied"

    @pytest.mark.asyncio
    @respx.mock
    async def test_expired(self):
        respx.get(_URL).mock(
            return_value=httpx.Response(200, json={"status": "expired"})
        )
        decision = await poll_hold_gate(_APPROVAL_ID, 30, _BACKEND, _API_KEY)
        assert decision.approved is False
        assert decision.decision == "expired"

    @pytest.mark.asyncio
    @respx.mock
    async def test_pending_then_approved(self):
        """Polls 'pending' twice, then 'approved'."""
        respx.get(_URL).mock(
            side_effect=[
                httpx.Response(200, json={"status": "pending"}),
                httpx.Response(200, json={"status": "pending"}),
                httpx.Response(200, json={"status": "approved"}),
            ]
        )
        decision = await poll_hold_gate(
            _APPROVAL_ID, 60, _BACKEND, _API_KEY, poll_interval_s=0.01
        )
        assert decision.approved is True
        assert decision.decision == "approved"

    @pytest.mark.asyncio
    @respx.mock
    async def test_pending_then_denied(self):
        respx.get(_URL).mock(
            side_effect=[
                httpx.Response(200, json={"status": "pending"}),
                httpx.Response(200, json={"status": "denied"}),
            ]
        )
        decision = await poll_hold_gate(
            _APPROVAL_ID, 60, _BACKEND, _API_KEY, poll_interval_s=0.01
        )
        assert decision.approved is False
        assert decision.decision == "denied"

    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_when_always_pending(self):
        """Backend never resolves → timeout."""
        respx.get(_URL).mock(
            return_value=httpx.Response(200, json={"status": "pending"})
        )
        decision = await poll_hold_gate(
            _APPROVAL_ID, timeout_s=1, backend_url=_BACKEND,
            api_key=_API_KEY, poll_interval_s=0.05,
        )
        assert decision.approved is False
        assert decision.decision == "timeout"


# ---------------------------------------------------------------------------
# poll_hold_gate — error handling
# ---------------------------------------------------------------------------

class TestPollHoldGateErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error_then_approved(self):
        """First poll raises ConnectionError; second succeeds → approved."""
        respx.get(_URL).mock(
            side_effect=[
                httpx.ConnectError("connection refused"),
                httpx.Response(200, json={"status": "approved"}),
            ]
        )
        decision = await poll_hold_gate(
            _APPROVAL_ID, 60, _BACKEND, _API_KEY, poll_interval_s=0.01
        )
        assert decision.approved is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_polls_error_gives_timeout(self):
        """All polls fail with network error → timeout."""
        respx.get(_URL).mock(side_effect=httpx.ConnectError("unreachable"))
        decision = await poll_hold_gate(
            _APPROVAL_ID, timeout_s=1, backend_url=_BACKEND,
            api_key=_API_KEY, poll_interval_s=0.05,
        )
        assert decision.approved is False
        assert decision.decision == "timeout"

    @pytest.mark.asyncio
    @respx.mock
    async def test_unexpected_status_code_keeps_polling(self):
        """500 response → keeps polling until timeout."""
        respx.get(_URL).mock(
            side_effect=[
                httpx.Response(500, text="internal error"),
                httpx.Response(200, json={"status": "approved"}),
            ]
        )
        decision = await poll_hold_gate(
            _APPROVAL_ID, 30, _BACKEND, _API_KEY, poll_interval_s=0.01
        )
        assert decision.approved is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_status_field_keeps_polling(self):
        """Response has no 'status' field → treated as pending."""
        respx.get(_URL).mock(
            side_effect=[
                httpx.Response(200, json={"id": _APPROVAL_ID}),  # no status
                httpx.Response(200, json={"status": "denied"}),
            ]
        )
        decision = await poll_hold_gate(
            _APPROVAL_ID, 30, _BACKEND, _API_KEY, poll_interval_s=0.01
        )
        assert decision.approved is False
        assert decision.decision == "denied"


# ---------------------------------------------------------------------------
# poll_hold_gate — latency measurement
# ---------------------------------------------------------------------------

class TestPollHoldGateLatency:
    @pytest.mark.asyncio
    @respx.mock
    async def test_latency_ms_is_positive(self):
        respx.get(_URL).mock(
            return_value=httpx.Response(200, json={"status": "approved"})
        )
        decision = await poll_hold_gate(_APPROVAL_ID, 30, _BACKEND, _API_KEY)
        assert decision.latency_ms >= 0.0

    @pytest.mark.asyncio
    @respx.mock
    async def test_latency_ms_type_is_float(self):
        respx.get(_URL).mock(
            return_value=httpx.Response(200, json={"status": "denied"})
        )
        decision = await poll_hold_gate(_APPROVAL_ID, 30, _BACKEND, _API_KEY)
        assert isinstance(decision.latency_ms, float)


# ---------------------------------------------------------------------------
# synthesize_denial — Anthropic format
# ---------------------------------------------------------------------------

class TestSynthesizeDenialAnthropic:
    def test_type_is_message(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        assert r["type"] == "message"

    def test_role_is_assistant(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        assert r["role"] == "assistant"

    def test_content_is_list_with_text(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        assert isinstance(r["content"], list)
        assert r["content"][0]["type"] == "text"
        assert len(r["content"][0]["text"]) > 0

    def test_stop_reason_is_end_turn(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        assert r["stop_reason"] == "end_turn"

    def test_no_tool_use_in_content(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        for block in r["content"]:
            assert block.get("type") != "tool_use"

    def test_model_field_preserved(self):
        model = "claude-3-opus-20240229"
        r = _anthropic_denial(model, _DENIAL_TEXT)
        assert r["model"] == model

    def test_id_field_present(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        assert r["id"].startswith("msg_")

    def test_usage_field_present(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        assert "usage" in r
        assert "input_tokens" in r["usage"]
        assert "output_tokens" in r["usage"]

    def test_json_serialisable(self):
        r = _anthropic_denial("claude-3-5-sonnet-20241022", _DENIAL_TEXT)
        json.dumps(r)  # must not raise

    def test_custom_message(self):
        custom = "Custom denial message."
        r = _anthropic_denial("claude-3-5-sonnet-20241022", custom)
        assert r["content"][0]["text"] == custom


# ---------------------------------------------------------------------------
# synthesize_denial — OpenAI format
# ---------------------------------------------------------------------------

class TestSynthesizeDenialOpenAI:
    def test_object_is_chat_completion(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert r["object"] == "chat.completion"

    def test_choices_is_list(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert isinstance(r["choices"], list)
        assert len(r["choices"]) >= 1

    def test_choice_role_is_assistant(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert r["choices"][0]["message"]["role"] == "assistant"

    def test_finish_reason_is_stop(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert r["choices"][0]["finish_reason"] == "stop"

    def test_no_tool_calls_in_choice(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        msg = r["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert "function_call" not in msg

    def test_content_is_string(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        content = r["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_model_field_preserved(self):
        model = "gpt-4-turbo"
        r = _openai_denial(model, _DENIAL_TEXT)
        assert r["model"] == model

    def test_id_field_starts_with_chatcmpl(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert r["id"].startswith("chatcmpl-")

    def test_usage_field_present(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert "usage" in r
        assert "total_tokens" in r["usage"]

    def test_created_field_is_int(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        assert isinstance(r["created"], int)
        assert r["created"] > 0

    def test_json_serialisable(self):
        r = _openai_denial("gpt-4o", _DENIAL_TEXT)
        json.dumps(r)  # must not raise

    def test_custom_message(self):
        custom = "Custom denial."
        r = _openai_denial("gpt-4o", custom)
        assert r["choices"][0]["message"]["content"] == custom


# ---------------------------------------------------------------------------
# synthesize_denial — provider dispatch
# ---------------------------------------------------------------------------

class TestSynthesizeDenialDispatch:
    def test_anthropic_provider_uses_anthropic_format(self):
        r = synthesize_denial("anthropic", "claude-3-5-sonnet-20241022")
        assert r.get("type") == "message"
        assert r.get("stop_reason") == "end_turn"

    def test_openai_provider_uses_openai_format(self):
        r = synthesize_denial("openai", "gpt-4o")
        assert r.get("object") == "chat.completion"

    def test_groq_uses_openai_format(self):
        r = synthesize_denial("groq", "llama-3.1-70b-versatile")
        assert r.get("object") == "chat.completion"

    def test_together_uses_openai_format(self):
        r = synthesize_denial("together", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        assert r.get("object") == "chat.completion"

    def test_mistral_uses_openai_format(self):
        r = synthesize_denial("mistral", "mistral-large-latest")
        assert r.get("object") == "chat.completion"

    def test_unknown_provider_uses_openai_format(self):
        r = synthesize_denial("some_new_provider", "model-x")
        assert r.get("object") == "chat.completion"

    def test_ids_are_unique_across_calls(self):
        r1 = synthesize_denial("openai", "gpt-4o")
        r2 = synthesize_denial("openai", "gpt-4o")
        assert r1["id"] != r2["id"]

    def test_anthropic_ids_unique(self):
        r1 = synthesize_denial("anthropic", "claude-3-5-sonnet-20241022")
        r2 = synthesize_denial("anthropic", "claude-3-5-sonnet-20241022")
        assert r1["id"] != r2["id"]

    def test_custom_message_passed_through(self):
        msg = "You are not allowed."
        r = synthesize_denial("anthropic", "claude-3-5-sonnet-20241022", message=msg)
        assert r["content"][0]["text"] == msg

    def test_denial_text_contains_approval(self):
        """Default message communicates that approval was required."""
        assert "approval" in _DENIAL_TEXT.lower()
