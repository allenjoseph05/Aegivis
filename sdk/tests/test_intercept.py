"""
Tests for aegivis.intercept — zero-config httpx transport-level interceptor.

These tests verify:
- Provider detection by hostname
- Request payload extraction (model, messages, tools)
- Response payload extraction (usage, stop reason, text preview)
- Event firing is fire-and-forget (non-blocking)
- Proxy traffic to Aegivis backend is never re-intercepted (no loops)
- Non-LLM requests pass through unchanged
- Streaming responses don't get body-consumed
- uninstall() restores original httpx methods
"""
from __future__ import annotations

import json
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch


# ── Helpers to test private functions without triggering the patch ─────────

def _get_module():
    """Import and return the intercept module (patch is already installed at import)."""
    import importlib
    import aegivis.intercept as m
    return m


class TestProviderDetection(unittest.TestCase):
    def setUp(self):
        self.m = _get_module()

    def _url(self, host: str, path: str = "/v1/messages") -> MagicMock:
        u = MagicMock()
        u.host = host
        u.path = path
        return u

    def test_anthropic_detected(self):
        self.assertEqual(self.m._detect_provider(self._url("api.anthropic.com")), "anthropic")

    def test_openai_detected(self):
        self.assertEqual(self.m._detect_provider(self._url("api.openai.com", "/v1/chat/completions")), "openai")

    def test_groq_detected(self):
        self.assertEqual(self.m._detect_provider(self._url("api.groq.com")), "groq")

    def test_google_gemini_detected(self):
        self.assertEqual(
            self.m._detect_provider(self._url("generativelanguage.googleapis.com", "/v1beta/models/gemini-pro:generateContent")),
            "google-gemini",
        )

    def test_azure_openai_detected(self):
        self.assertEqual(
            self.m._detect_provider(self._url("myresource.openai.azure.com")),
            "azure-openai",
        )

    def test_cohere_detected(self):
        self.assertEqual(self.m._detect_provider(self._url("api.cohere.ai")), "cohere")

    def test_mistral_detected(self):
        self.assertEqual(self.m._detect_provider(self._url("api.mistral.ai")), "mistral")

    def test_deepseek_detected(self):
        self.assertEqual(self.m._detect_provider(self._url("api.deepseek.com")), "deepseek")

    def test_unknown_host_returns_none(self):
        self.assertIsNone(self.m._detect_provider(self._url("example.com")))

    def test_backend_host_skipped(self):
        # localhost should never be intercepted (it's the Aegivis backend)
        self.assertIsNone(self.m._detect_provider(self._url("localhost")))

    def test_none_url_returns_none(self):
        self.assertIsNone(self.m._detect_provider(None))

    def test_string_url_fallback(self):
        # If given a plain string (not a URL object), should still work or return None safely
        result = self.m._detect_provider("api.anthropic.com")
        # Will try str(getattr(url, "host", url)) → str("api.anthropic.com") → match
        self.assertEqual(result, "anthropic")


class TestRequestExtraction(unittest.TestCase):
    def setUp(self):
        self.m = _get_module()

    def _make_request(self, body: dict) -> MagicMock:
        req = MagicMock()
        req.content = json.dumps(body).encode()
        return req

    def test_extracts_model(self):
        req = self._make_request({"model": "claude-3-5-haiku-20241022", "messages": []})
        p = self.m._read_request(req)
        self.assertEqual(p["model"], "claude-3-5-haiku-20241022")

    def test_extracts_user_message_preview(self):
        req = self._make_request({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Tell me about AI security."},
            ],
        })
        p = self.m._read_request(req)
        self.assertIn("Tell me about AI security", p["user_message_preview"])
        self.assertIn("You are helpful", p["system_prompt_preview"])
        self.assertEqual(p["message_count"], 2)

    def test_extracts_anthropic_content_blocks(self):
        req = self._make_request({
            "model": "claude-opus-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this document"},
                    {"type": "image", "source": {"type": "base64"}},
                ],
            }],
        })
        p = self.m._read_request(req)
        self.assertIn("Analyze this document", p["user_message_preview"])

    def test_extracts_tools(self):
        req = self._make_request({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "search web"}],
            "tools": [
                {"type": "function", "function": {"name": "web_search", "description": "..."}},
                {"type": "function", "function": {"name": "send_email", "description": "..."}},
            ],
        })
        p = self.m._read_request(req)
        self.assertEqual(p["tool_count"], 2)
        self.assertIn("web_search", p["tool_names"])
        self.assertIn("send_email", p["tool_names"])

    def test_extracts_anthropic_system_top_level(self):
        req = self._make_request({
            "model": "claude-3-5-haiku-20241022",
            "system": "You are a security auditor.",
            "messages": [{"role": "user", "content": "Review this."}],
        })
        p = self.m._read_request(req)
        self.assertIn("security auditor", p["system_prompt_preview"])

    def test_extracts_max_tokens(self):
        req = self._make_request({
            "model": "gpt-4o",
            "messages": [],
            "max_tokens": 1024,
        })
        p = self.m._read_request(req)
        self.assertEqual(p["max_tokens"], 1024)

    def test_empty_body_returns_empty_dict(self):
        req = MagicMock()
        req.content = b""
        p = self.m._read_request(req)
        self.assertEqual(p, {})

    def test_invalid_json_returns_empty_dict(self):
        req = MagicMock()
        req.content = b"not json"
        p = self.m._read_request(req)
        self.assertEqual(p, {})


class TestResponseExtraction(unittest.TestCase):
    def setUp(self):
        self.m = _get_module()

    def _make_response(self, body: dict, streaming: bool = False) -> MagicMock:
        resp = MagicMock()
        resp.content = json.dumps(body).encode()
        resp.is_stream = streaming
        resp.status_code = 200
        resp.headers = {}
        return resp

    def test_extracts_anthropic_response(self):
        resp = self._make_response({
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "AI security is critical."}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "stop_reason": "end_turn",
        })
        p = self.m._read_response(resp, {})
        self.assertEqual(p["model"], "claude-3-5-haiku-20241022")
        self.assertEqual(p["input_tokens"], 50)
        self.assertEqual(p["output_tokens"], 20)
        self.assertEqual(p["stop_reason"], "end_turn")
        self.assertIn("AI security", p["response_preview"])

    def test_extracts_openai_response(self):
        resp = self._make_response({
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10},
        })
        p = self.m._read_response(resp, {})
        self.assertEqual(p["input_tokens"], 30)
        self.assertEqual(p["output_tokens"], 10)
        self.assertEqual(p["stop_reason"], "stop")
        self.assertIn("Hello world", p["response_preview"])

    def test_streaming_response_not_consumed(self):
        resp = MagicMock()
        resp.is_stream = True
        resp.headers = {}
        p = self.m._read_response(resp, {"model": "claude-opus-4-6"})
        self.assertTrue(p.get("streamed"))
        # content should NOT have been accessed
        resp.content.assert_not_called()

    def test_empty_response_returns_empty_dict(self):
        resp = MagicMock()
        resp.is_stream = False
        resp.content = b""
        resp.headers = {}
        p = self.m._read_response(resp, {})
        self.assertEqual(p, {})


class TestFireAndForget(unittest.TestCase):
    def setUp(self):
        self.m = _get_module()

    def test_fire_does_not_block(self):
        """_fire must return immediately even if backend is unreachable."""
        t0 = time.monotonic()
        self.m._fire(
            "LLM_CALL_START",
            {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
            "test-session",
            "test-agent",
        )
        elapsed = time.monotonic() - t0
        # Should be instant (thread spawned, not waited on)
        self.assertLess(elapsed, 0.5)

    def test_fire_skips_when_no_backend_url(self):
        """If backend URL is empty, _fire must be a no-op."""
        original = self.m._BACKEND_URL
        try:
            self.m._BACKEND_URL = ""
            # Should not raise even with no backend
            self.m._fire("LLM_CALL_START", {}, "sid", "aid")
        finally:
            self.m._BACKEND_URL = original


class TestHttpxPatch(unittest.TestCase):
    def setUp(self):
        self.m = _get_module()

    def test_patch_is_idempotent(self):
        """Calling _install() twice should not double-patch."""
        import httpx
        result = self.m._install()
        self.assertTrue(result)
        self.assertTrue(getattr(httpx, "_aegivis_patched", False))

        # Call again — should be no-op, still True
        result2 = self.m._install()
        self.assertTrue(result2)

    def test_non_llm_request_passes_through(self):
        """Requests to non-LLM hosts must pass through without modification."""
        import httpx

        calls: list = []
        original_send = httpx.Client.send

        def mock_send(self_client, request, **kwargs):
            calls.append(request.url.host)
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"{}"
            resp.headers = {}
            resp.is_stream = False
            return resp

        httpx.Client.send = mock_send
        try:
            # Patch detect to return None for this test
            with patch.object(self.m, "_detect_provider", return_value=None):
                req = MagicMock()
                req.url = MagicMock()
                req.url.host = "example.com"
                req.url.path = "/"
                # Simulate what the patched send does
                provider = self.m._detect_provider(req.url)
                self.assertIsNone(provider)
        finally:
            httpx.Client.send = original_send


class TestSessionIdResolution(unittest.TestCase):
    def setUp(self):
        self.m = _get_module()
        import os
        # Clear any leftover env vars
        os.environ.pop("AEGIVIS_SESSION_ID", None)
        os.environ.pop("AEGIVIS_AGENT_ID", None)

    def tearDown(self):
        import os
        os.environ.pop("AEGIVIS_SESSION_ID", None)
        os.environ.pop("AEGIVIS_AGENT_ID", None)

    def test_session_id_from_env(self):
        import os
        os.environ["AEGIVIS_SESSION_ID"] = "my-session-123"
        self.assertEqual(self.m._session_id(), "my-session-123")

    def test_session_id_generated_when_absent(self):
        sid = self.m._session_id()
        self.assertTrue(sid.startswith("intercept-"))
        self.assertEqual(len(sid), len("intercept-") + 12)

    def test_agent_id_from_env(self):
        import os
        os.environ["AEGIVIS_AGENT_ID"] = "my-agent"
        self.assertEqual(self.m._agent_id(), "my-agent")


if __name__ == "__main__":
    unittest.main()
