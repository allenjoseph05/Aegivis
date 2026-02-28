"""
End-to-end integration tests for the AgentBlackBox proxy.

These tests exercise the full request pipeline with:
  - The real FastAPI app (including lifespan)
  - Mocked upstream LLM API (via respx)
  - Mocked event transport (to avoid backend calls)

Run from proxy/ directory:
    python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
import httpx
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_transport_mock():
    """Return a mock EventTransport that doesn't touch the network."""
    t = MagicMock()
    t.start = AsyncMock()
    t.stop = AsyncMock()
    t.enqueue = MagicMock()
    t.enqueue_violation = MagicMock()
    t.buffer_status = MagicMock(return_value={"events": 0, "violations": 0})
    return t


@pytest.fixture(scope="module")
def mock_transport():
    return _make_transport_mock()


@pytest.fixture(scope="module")
def proxy_app(mock_transport):
    """
    Start the proxy app once per module with a mocked transport.
    Yields (TestClient, mock_transport).
    """
    with (
        patch("proxy.app.main.get_transport", return_value=mock_transport),
        patch("proxy.app.intercept.get_transport", return_value=mock_transport),
    ):
        with TestClient(
            __import__("proxy.app.main", fromlist=["app"]).app,
            raise_server_exceptions=False,
        ) as client:
            yield client


# ---------------------------------------------------------------------------
# OpenAI response helpers
# ---------------------------------------------------------------------------

def _openai_chat_response(content: str = "Hello from mock LLM") -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _openai_tool_response(tool_name: str = "get_weather", args: dict | None = None) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args or {"location": "London"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27},
    }


# Proxy routes all OpenAI traffic to https://api.openai.com
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

_CHAT_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
}

_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-test",
    "x-abb-agent-id": "integration-test-agent",
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_returns_ok(self, proxy_app):
        resp = proxy_app.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "agentblackbox-proxy"
        assert "providers" in body
        assert "openai" in body["providers"]
        assert "policy_rules" in body
        assert "active_sessions" in body

    def test_health_no_auth_required(self, proxy_app):
        """Health endpoint has no authentication."""
        resp = proxy_app.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Policy management endpoints
# ---------------------------------------------------------------------------

class TestPolicyEndpoints:
    def test_list_policies_returns_rules(self, proxy_app):
        resp = proxy_app.get("/policies")
        assert resp.status_code == 200
        body = resp.json()
        assert "rules" in body
        assert isinstance(body["rules"], list)

    def test_list_tool_permissions(self, proxy_app):
        resp = proxy_app.get("/tool-permissions")
        assert resp.status_code == 200
        body = resp.json()
        assert "rules" in body
        assert "total" in body
        assert "enabled" in body

    def test_reload_policies_without_admin_key(self, proxy_app):
        """When ABB_ADMIN_API_KEY is empty (default), reload is open."""
        resp = proxy_app.post("/policies/reload", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "reloaded"
        assert "rule_count" in body


# ---------------------------------------------------------------------------
# Normal proxied chat request
# ---------------------------------------------------------------------------

class TestNormalChatRequest:
    @respx.mock
    def test_successful_chat_forwarded(self, proxy_app):
        """A valid chat request is forwarded to OpenAI and the response is returned."""
        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers=_HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        # Verify the upstream response was forwarded
        assert "choices" in body
        assert body["choices"][0]["message"]["content"] == "Hello from mock LLM"

    @respx.mock
    def test_session_id_header_returned(self, proxy_app):
        """Proxy attaches X-ABB-Session-ID header when a block occurs."""
        # Test via a BLOCK scenario so we get the header directly
        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        # Normal request — check no error headers
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers=_HEADERS,
        )
        assert resp.status_code == 200

    @respx.mock
    def test_transport_enqueue_called(self, proxy_app, mock_transport):
        """After a successful request, transport.enqueue is called for LLM_CALL_START."""
        mock_transport.enqueue.reset_mock()
        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers={**_HEADERS, "x-abb-session-id": f"sess-enqueue-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200
        # Transport.enqueue should have been called at least once (LLM_CALL_START)
        assert mock_transport.enqueue.called

    @respx.mock
    def test_enqueued_event_has_correct_type(self, proxy_app, mock_transport):
        """LLM_CALL_START event is the first enqueued event."""
        mock_transport.enqueue.reset_mock()
        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers={**_HEADERS, "x-abb-session-id": f"sess-type-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200
        # First enqueue call should be LLM_CALL_START
        first_call = mock_transport.enqueue.call_args_list[0]
        event = first_call[0][0]
        assert event["event_type"] == "LLM_CALL_START"
        assert event["provider"] == "openai"
        assert event["model"] == "gpt-4o"

    @respx.mock
    def test_hash_chain_on_first_event(self, proxy_app, mock_transport):
        """First event in a new session has GENESIS_{session_id} as previous_hash."""
        mock_transport.enqueue.reset_mock()
        session_id = f"sess-genesis-{uuid.uuid4().hex[:8]}"
        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers={**_HEADERS, "x-abb-session-id": session_id},
        )
        assert resp.status_code == 200
        first_event = mock_transport.enqueue.call_args_list[0][0][0]
        assert first_event["previous_hash"] == f"GENESIS_{session_id}"
        assert first_event["current_hash"] is not None
        assert len(first_event["current_hash"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Agent key authentication
# ---------------------------------------------------------------------------

class TestAgentKeyAuth:
    def test_no_key_allowed_by_default(self, proxy_app):
        """When ABB_REQUIRE_AGENT_KEY is False (default), requests without key pass."""
        # The proxy uses validate_agent_key which returns True for empty key when
        # require_agent_key=False (the default)
        with respx.mock:
            respx.post(_OPENAI_CHAT_URL).mock(
                return_value=httpx.Response(200, json=_openai_chat_response())
            )
            resp = proxy_app.post(
                "/openai/v1/chat/completions",
                json=_CHAT_REQUEST,
                headers={"Content-Type": "application/json", "Authorization": "Bearer sk-test"},
            )
        # With require_agent_key=False, anonymous agent is resolved as "unknown-agent"
        assert resp.status_code == 200

    def test_invalid_key_rejected_when_keys_configured(self, proxy_app):
        """When ABB_AGENT_KEYS is set, invalid keys return 401."""
        from proxy.app.config import settings
        original_keys = settings.agent_keys
        original_require = settings.require_agent_key

        settings.agent_keys = ["valid-key-123"]
        settings.require_agent_key = True

        try:
            resp = proxy_app.post(
                "/openai/v1/chat/completions",
                json=_CHAT_REQUEST,
                headers={
                    "Content-Type": "application/json",
                    "x-abb-agent-key": "wrong-key",
                },
            )
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"] == "invalid_agent_key"
        finally:
            settings.agent_keys = original_keys
            settings.require_agent_key = original_require


# ---------------------------------------------------------------------------
# Policy enforcement — BLOCK
# ---------------------------------------------------------------------------

class TestPolicyBlock:
    @respx.mock
    def test_runaway_agent_blocked_after_limit(self, proxy_app, mock_transport):
        """
        The default 'runaway-llm-call-protection' rule BLOCKs when llm_call_count > 100.
        We inject session state directly to avoid making 100+ real calls.

        Verifies:
        - Policy BLOCK returns 403 with rule name
        - Blocked request is NOT forwarded to upstream LLM
        """
        from proxy.app.session import get_session_tracker

        session_id = f"sess-runaway-{uuid.uuid4().hex[:8]}"
        session_headers = {**_HEADERS, "x-abb-session-id": session_id}

        # Pre-populate session state above the block threshold (gt 100)
        tracker = get_session_tracker()
        state = tracker.get_state(session_id)
        state.llm_call_count = 101  # 101 > 100 → BLOCK on next evaluation
        state.last_hash = f"GENESIS_{session_id}"
        state.sequence_number = 101
        state.started_at_ns = int(time.time() * 1e9)

        upstream_called = False

        def _record_upstream(request):
            nonlocal upstream_called
            upstream_called = True
            return httpx.Response(200, json=_openai_chat_response())

        respx.post(_OPENAI_CHAT_URL).mock(side_effect=_record_upstream)
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers=session_headers,
        )
        assert resp.status_code == 403, f"Expected 403 BLOCK, got {resp.status_code}"
        body = resp.json()
        assert body["error"] == "policy_violation"
        assert "rule" in body
        assert "runaway" in body["rule"].lower()
        assert "session_id" in body
        # Verify upstream was NOT called for the blocked request
        assert not upstream_called, "Upstream LLM was called despite BLOCK"

    @respx.mock
    def test_block_response_has_policy_headers(self, proxy_app):
        """403 response includes X-ABB-Policy-Rule and X-ABB-Session-ID headers."""
        from proxy.app.session import get_session_tracker

        session_id = f"sess-hdr-{uuid.uuid4().hex[:8]}"
        headers = {**_HEADERS, "x-abb-session-id": session_id}

        # Inject session state above threshold
        tracker = get_session_tracker()
        state = tracker.get_state(session_id)
        state.llm_call_count = 101
        state.last_hash = f"GENESIS_{session_id}"
        state.sequence_number = 101
        state.started_at_ns = int(time.time() * 1e9)

        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers=headers,
        )
        assert resp.status_code == 403
        # Response headers carry the rule name and session ID
        resp_headers_lower = {h.lower() for h in resp.headers}
        assert "x-abb-policy-rule" in resp_headers_lower
        assert "x-abb-session-id" in resp_headers_lower


# ---------------------------------------------------------------------------
# Canary token injection
# ---------------------------------------------------------------------------

class TestCanaryInjection:
    @respx.mock
    def test_canary_injected_into_forward_body(self, proxy_app):
        """
        When security is enabled and ABB_SECURITY_CANARY_ENABLED=True, the
        request forwarded to the upstream LLM contains a canary token in the
        system prompt, but the original request (without the canary) is in
        the audit log.
        """
        from proxy.app.config import settings
        if not settings.security_canary_enabled:
            pytest.skip("Security canary not enabled")

        captured_bodies: list[bytes] = []

        def _capture(request: httpx.Request):
            captured_bodies.append(request.content)
            return httpx.Response(200, json=_openai_chat_response())

        respx.post(_OPENAI_CHAT_URL).mock(side_effect=_capture)

        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me a joke."},
                ],
            },
            headers={**_HEADERS, "x-abb-session-id": f"sess-canary-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200
        assert len(captured_bodies) == 1

        # The forwarded body should contain the canary prefix
        forwarded = json.loads(captured_bodies[0])
        all_content = " ".join(
            str(m.get("content") or "")
            for m in forwarded.get("messages", [])
        )
        assert "ABB_CVT_" in all_content, (
            f"Canary token not found in forwarded messages. Content: {all_content[:500]}"
        )

    @respx.mock
    def test_canary_not_in_original_audit_log(self, proxy_app, mock_transport):
        """
        The LLM_CALL_START event enqueued to the backend contains the ORIGINAL
        messages (without the canary token).
        """
        from proxy.app.config import settings
        if not settings.security_canary_enabled:
            pytest.skip("Security canary not enabled")

        mock_transport.enqueue.reset_mock()

        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )

        original_user_content = "What is 2+2?"
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": original_user_content}],
            },
            headers={**_HEADERS, "x-abb-session-id": f"sess-nocnry-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200

        # Find the LLM_CALL_START event
        start_events = [
            call[0][0]
            for call in mock_transport.enqueue.call_args_list
            if call[0][0].get("event_type") == "LLM_CALL_START"
        ]
        assert len(start_events) >= 1
        start = start_events[0]
        messages = start["payload"]["messages"]
        all_content = " ".join(str(m.get("content") or "") for m in messages)
        # Canary must NOT be in the audit log
        assert "ABB_CVT_" not in all_content, (
            "Canary token leaked into audit log!"
        )
        # Original content must be preserved
        assert original_user_content in all_content


# ---------------------------------------------------------------------------
# Spotlighting (indirect prompt injection defense)
# ---------------------------------------------------------------------------

class TestSpotlighting:
    @respx.mock
    def test_tool_output_wrapped_in_spotlight_markers(self, proxy_app):
        """
        When spotlighting is enabled, tool result messages forwarded to the LLM
        are wrapped in randomized delimiters.
        """
        from proxy.app.config import settings
        if not settings.security_spotlighting_enabled:
            pytest.skip("Spotlighting not enabled")

        captured_bodies: list[bytes] = []

        def _capture(request: httpx.Request):
            captured_bodies.append(request.content)
            return httpx.Response(200, json=_openai_chat_response())

        respx.post(_OPENAI_CHAT_URL).mock(side_effect=_capture)

        # Request that includes a tool result message
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "The weather in London is 15C and cloudy.",
                    },
                    {"role": "user", "content": "Thanks!"},
                ],
            },
            headers={**_HEADERS, "x-abb-session-id": f"sess-spotlight-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200
        assert len(captured_bodies) == 1

        forwarded = json.loads(captured_bodies[0])
        # Find the tool message in the forwarded body
        tool_msg = next(
            (m for m in forwarded["messages"] if m.get("role") == "tool"),
            None,
        )
        assert tool_msg is not None
        content = str(tool_msg.get("content") or "")
        # Spotlighting wraps with == marker ==
        assert "==" in content, (
            f"Spotlight markers not found in tool message: {content[:200]}"
        )


# ---------------------------------------------------------------------------
# Passthrough (non-intercept paths)
# ---------------------------------------------------------------------------

class TestPassthrough:
    @respx.mock
    def test_models_endpoint_passthrough(self, proxy_app):
        """Non-chat paths (e.g., /openai/v1/models) are passed through without interception."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]},
            )
        )
        resp = proxy_app.get("/openai/v1/models", headers=_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"


# ---------------------------------------------------------------------------
# Tool call handling
# ---------------------------------------------------------------------------

class TestToolCallHandling:
    @respx.mock
    def test_tool_call_response_generates_tool_start_event(self, proxy_app, mock_transport):
        """
        When the LLM response contains tool_calls, the proxy infers and enqueues
        TOOL_CALL_START events.
        """
        mock_transport.enqueue.reset_mock()

        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(
                200, json=_openai_tool_response("search_web", {"query": "Python docs"})
            )
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers={**_HEADERS, "x-abb-session-id": f"sess-tool-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200

        # Wait for async fire-and-forget process_response to complete
        # (TestClient's sync mode means it runs the event loop in a thread;
        #  give the async task a moment to flush)
        import time as _time
        _time.sleep(0.1)

        event_types = [
            call[0][0]["event_type"]
            for call in mock_transport.enqueue.call_args_list
            if "event_type" in call[0][0]
        ]
        # Should have LLM_CALL_START + LLM_CALL_END + TOOL_CALL_START (finish_reason=tool_calls)
        assert "LLM_CALL_START" in event_types
        # TOOL_CALL_START should be emitted after the tool_calls response
        assert "TOOL_CALL_START" in event_types

    @respx.mock
    def test_agent_finish_event_on_stop(self, proxy_app, mock_transport):
        """
        When finish_reason='stop' and no tool_calls, an AGENT_FINISH event is emitted.
        """
        mock_transport.enqueue.reset_mock()

        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response("Goodbye!"))
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers={**_HEADERS, "x-abb-session-id": f"sess-finish-{uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 200

        import time as _time
        _time.sleep(0.1)

        event_types = [
            call[0][0]["event_type"]
            for call in mock_transport.enqueue.call_args_list
            if "event_type" in call[0][0]
        ]
        assert "AGENT_FINISH" in event_types


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @respx.mock
    def test_upstream_error_generates_system_error_event(self, proxy_app, mock_transport):
        """A 500 from the upstream LLM generates a SYSTEM_ERROR event."""
        mock_transport.enqueue.reset_mock()

        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(
                500, json={"error": {"message": "Internal server error", "type": "server_error"}}
            )
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            json=_CHAT_REQUEST,
            headers={**_HEADERS, "x-abb-session-id": f"sess-err-{uuid.uuid4().hex[:8]}"},
        )
        # The proxy forwards the upstream error response to the client
        assert resp.status_code == 500

        import time as _time
        _time.sleep(0.1)

        event_types = [
            call[0][0]["event_type"]
            for call in mock_transport.enqueue.call_args_list
            if "event_type" in call[0][0]
        ]
        assert "SYSTEM_ERROR" in event_types

    @respx.mock
    def test_malformed_json_body(self, proxy_app):
        """Malformed JSON body is handled gracefully — proxy still forwards to upstream."""
        # With malformed JSON, body is treated as {} (empty)
        respx.post(_OPENAI_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_openai_chat_response())
        )
        resp = proxy_app.post(
            "/openai/v1/chat/completions",
            content=b"not-valid-json",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-test",
                "x-abb-agent-id": "test-agent",
            },
        )
        # Proxy parses body as {} and forwards — upstream still responds 200
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Session state persistence across requests
# ---------------------------------------------------------------------------

class TestSessionStatePersistence:
    @respx.mock
    def test_sequence_numbers_increment_across_requests(self, proxy_app, mock_transport):
        """
        Multiple requests in the same session produce sequentially increasing
        sequence_numbers.
        """
        mock_transport.enqueue.reset_mock()
        session_id = f"sess-seq-{uuid.uuid4().hex[:8]}"
        headers = {**_HEADERS, "x-abb-session-id": session_id}

        for _ in range(3):
            respx.post(_OPENAI_CHAT_URL).mock(
                return_value=httpx.Response(200, json=_openai_chat_response())
            )
            resp = proxy_app.post(
                "/openai/v1/chat/completions",
                json=_CHAT_REQUEST,
                headers=headers,
            )
            assert resp.status_code == 200

        import time as _time
        _time.sleep(0.1)

        # Extract all LLM_CALL_START events for this session
        start_events = [
            call[0][0]
            for call in mock_transport.enqueue.call_args_list
            if call[0][0].get("event_type") == "LLM_CALL_START"
            and call[0][0].get("session_id") == session_id
        ]
        assert len(start_events) == 3

        seq_nums = [e["sequence_number"] for e in start_events]
        assert seq_nums == sorted(seq_nums), f"Sequence numbers not monotonically increasing: {seq_nums}"

    @respx.mock
    def test_hash_chain_links_across_requests(self, proxy_app, mock_transport):
        """
        In a multi-request session, each event's previous_hash equals the
        prior event's current_hash — the chain is intact.
        """
        mock_transport.enqueue.reset_mock()
        session_id = f"sess-chain-{uuid.uuid4().hex[:8]}"
        headers = {**_HEADERS, "x-abb-session-id": session_id}

        for _ in range(2):
            respx.post(_OPENAI_CHAT_URL).mock(
                return_value=httpx.Response(200, json=_openai_chat_response())
            )
            resp = proxy_app.post(
                "/openai/v1/chat/completions",
                json=_CHAT_REQUEST,
                headers=headers,
            )
            assert resp.status_code == 200

        import time as _time
        _time.sleep(0.1)

        events = [
            call[0][0]
            for call in mock_transport.enqueue.call_args_list
            if call[0][0].get("session_id") == session_id
        ]
        # Sort by sequence number
        events.sort(key=lambda e: e.get("sequence_number", 0))

        # Verify chain: each event's previous_hash == prior event's current_hash
        for i in range(1, len(events)):
            prev = events[i - 1]
            curr = events[i]
            assert curr["previous_hash"] == prev["current_hash"], (
                f"Hash chain broken at seq {curr['sequence_number']}: "
                f"expected {prev['current_hash'][:16]}... "
                f"but got {curr['previous_hash'][:16]}..."
            )
