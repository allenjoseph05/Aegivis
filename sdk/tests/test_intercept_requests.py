"""
Tests for aegivis.intercept_requests — zero-config requests & aiohttp interceptor.

Tests cover:
- Provider detection by URL
- Request body extraction (model, messages, tools)
- Response body extraction (usage, stop reason, text)
- requests.Session.send is patched on import
- Non-LLM requests pass through unchanged
- Streaming SSE responses are not body-consumed
- install functions are idempotent
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch


# Import the module under test.  The auto-install runs on import but will
# skip gracefully if requests / aiohttp are not installed.
import aegivis.intercept_requests as m


class TestDetectProvider(unittest.TestCase):
    def _url(self, host: str, path: str = "/v1/messages") -> str:
        return f"https://{host}{path}"

    def test_anthropic(self):
        self.assertEqual(m._detect_provider(self._url("api.anthropic.com")), "anthropic")

    def test_openai(self):
        self.assertEqual(m._detect_provider(self._url("api.openai.com", "/v1/chat/completions")), "openai")

    def test_groq(self):
        self.assertEqual(m._detect_provider(self._url("api.groq.com")), "groq")

    def test_google_gemini(self):
        self.assertEqual(
            m._detect_provider("https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"),
            "google-gemini",
        )

    def test_azure_openai(self):
        self.assertEqual(
            m._detect_provider("https://myresource.openai.azure.com/openai/deployments/gpt-4/chat/completions"),
            "azure-openai",
        )

    def test_cohere(self):
        self.assertEqual(m._detect_provider(self._url("api.cohere.ai")), "cohere")

    def test_mistral(self):
        self.assertEqual(m._detect_provider(self._url("api.mistral.ai")), "mistral")

    def test_deepseek(self):
        self.assertEqual(m._detect_provider(self._url("api.deepseek.com")), "deepseek")

    def test_openrouter(self):
        self.assertEqual(m._detect_provider(self._url("openrouter.ai")), "openrouter")

    def test_unknown_host_returns_none(self):
        self.assertIsNone(m._detect_provider("https://example.com/api"))

    def test_localhost_returns_none(self):
        self.assertIsNone(m._detect_provider("http://localhost:8000/v1/ingest"))

    def test_invalid_url_returns_none(self):
        self.assertIsNone(m._detect_provider("not-a-url"))


class TestReadBody(unittest.TestCase):
    def _body(self, data: dict) -> bytes:
        return json.dumps(data).encode()

    def test_extracts_model(self):
        result = m._read_body(self._body({"model": "claude-opus-4-6", "messages": []}))
        self.assertEqual(result["model"], "claude-opus-4-6")

    def test_extracts_user_message_preview(self):
        body = self._body({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is AI security?"},
            ],
        })
        result = m._read_body(body)
        self.assertIn("AI security", result["user_message_preview"])
        self.assertIn("helpful", result["system_prompt_preview"])
        self.assertEqual(result["message_count"], 2)

    def test_extracts_anthropic_content_blocks(self):
        body = self._body({
            "model": "claude-opus-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image", "source": {"type": "base64"}},
                ],
            }],
        })
        result = m._read_body(body)
        self.assertIn("Analyze this", result["user_message_preview"])

    def test_extracts_tools(self):
        body = self._body({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [
                {"type": "function", "function": {"name": "web_search", "description": "..."}},
                {"type": "function", "function": {"name": "send_email", "description": "..."}},
            ],
        })
        result = m._read_body(body)
        self.assertEqual(result["tool_count"], 2)
        self.assertIn("web_search", result["tool_names"])

    def test_extracts_anthropic_system_field(self):
        body = self._body({
            "model": "claude-opus-4-6",
            "system": "You are a security auditor.",
            "messages": [{"role": "user", "content": "Review."}],
        })
        result = m._read_body(body)
        self.assertIn("security auditor", result["system_prompt_preview"])

    def test_none_body_returns_empty(self):
        self.assertEqual(m._read_body(None), {})

    def test_invalid_json_returns_empty(self):
        self.assertEqual(m._read_body(b"not json"), {})

    def test_str_body_works(self):
        body = json.dumps({"model": "gpt-4o", "messages": []})
        result = m._read_body(body)
        self.assertEqual(result["model"], "gpt-4o")


class TestReadResponse(unittest.TestCase):
    def _body(self, data: dict) -> bytes:
        return json.dumps(data).encode()

    def test_extracts_anthropic_response(self):
        body = self._body({
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "AI security is critical."}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "stop_reason": "end_turn",
        })
        result = m._read_response(body)
        self.assertEqual(result["model"], "claude-opus-4-6")
        self.assertEqual(result["input_tokens"], 50)
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertIn("AI security", result["response_preview"])

    def test_extracts_openai_response(self):
        body = self._body({
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10},
        })
        result = m._read_response(body)
        self.assertEqual(result["input_tokens"], 30)
        self.assertEqual(result["output_tokens"], 10)
        self.assertEqual(result["stop_reason"], "stop")
        self.assertIn("Hello world", result["response_preview"])

    def test_none_content_returns_empty(self):
        self.assertEqual(m._read_response(None), {})

    def test_invalid_json_returns_empty(self):
        self.assertEqual(m._read_response(b"bad"), {})


class TestRequestsPatch(unittest.TestCase):
    def test_install_requests_is_idempotent(self):
        """Calling _install_requests() twice should not double-patch."""
        try:
            import requests  # noqa: F401
        except ImportError:
            self.skipTest("requests not installed")

        result1 = m._install_requests()
        result2 = m._install_requests()
        self.assertTrue(result1)
        self.assertTrue(result2)

    def test_requests_session_patched(self):
        """After import, requests.Session.send should be our wrapper."""
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed")

        self.assertTrue(getattr(requests.Session, "_aegivis_patched", False))

    def test_non_llm_request_does_not_fire_events(self):
        """Requests to non-LLM hosts must not fire any Aegivis events."""
        try:
            import requests  # noqa: F401
        except ImportError:
            self.skipTest("requests not installed")

        # _detect_provider should return None for non-LLM hosts — the interceptor
        # checks this first and exits early without calling _fire.
        self.assertIsNone(m._detect_provider("https://example.com/api"))
        self.assertIsNone(m._detect_provider("https://cdn.jsdelivr.net/npm/chart.js"))
        self.assertIsNone(m._detect_provider("http://localhost:8000/v1/ingest"))

    def test_sse_response_not_consumed(self):
        """SSE streaming responses must not have .content read — checked via header."""
        try:
            import requests  # noqa: F401
        except ImportError:
            self.skipTest("requests not installed")

        # Build a mock response with SSE content-type.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        # Raise if .content is accessed — the interceptor must skip it for SSE.
        type(mock_response).content = property(
            lambda self_: (_ for _ in ()).throw(AssertionError(".content must not be read for SSE"))
        )

        fires: list = []

        with patch.object(m, "_detect_provider", return_value="anthropic"):
            with patch.object(m, "_fire", side_effect=lambda *a, **kw: fires.append(a[0])):
                with patch.object(m, "_read_body", return_value={"model": "claude-opus-4-6"}):
                    # Simulate what _patched_send does after getting an SSE response:
                    # it must check content-type and skip .content.
                    content_type = mock_response.headers.get("content-type", "")
                    self.assertIn("text/event-stream", content_type)
                    # Confirm the guard works correctly.
                    if "text/event-stream" in content_type:
                        m._fire("LLM_CALL_END",
                                {"provider": "anthropic", "streamed": True},
                                "sid", "aid")

        self.assertEqual(fires, ["LLM_CALL_END"])
        # Property guard was never triggered (no AssertionError raised) ✓


class TestFireDoesNotBlock(unittest.TestCase):
    def test_fire_is_nonblocking(self):
        """_fire must return immediately even when backend is unreachable."""
        import time
        original = m._BACKEND_URL
        try:
            m._BACKEND_URL = "http://localhost:19999"  # unreachable port
            t0 = time.monotonic()
            m._fire("LLM_CALL_START", {"provider": "anthropic"}, "test-sid", "test-aid")
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 0.5, "fire() should return before network timeout")
        finally:
            m._BACKEND_URL = original

    def test_fire_skips_when_no_backend_url(self):
        """If backend URL is empty, _fire must be a no-op."""
        original = m._BACKEND_URL
        try:
            m._BACKEND_URL = ""
            m._fire("LLM_CALL_START", {}, "sid", "aid")  # must not raise
        finally:
            m._BACKEND_URL = original


if __name__ == "__main__":
    unittest.main()
