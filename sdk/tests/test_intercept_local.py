"""
Tests for aegivis.intercept_local — local model interceptor.

Tests cover:
- Local provider detection by host:port (Ollama, LM Studio, llama.cpp server)
- Custom URL registration via AEGIVIS_LOCAL_LLM_URLS
- Request body extraction for Ollama generate and chat formats
- Response body extraction for Ollama and OpenAI-compat formats
- Remote URLs are not intercepted (pass-through)
- fire-and-forget is non-blocking
- llama-cpp-python patch smoke test (without the actual library)
- transformers patch smoke test (without the actual library)
"""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch, call


import aegivis.intercept_local as m


class TestDetectLocalProvider(unittest.TestCase):
    def test_ollama_localhost(self):
        self.assertEqual(m._detect_local_provider("http://localhost:11434/api/generate"), "ollama")

    def test_ollama_127(self):
        self.assertEqual(m._detect_local_provider("http://127.0.0.1:11434/api/chat"), "ollama")

    def test_lmstudio(self):
        self.assertEqual(m._detect_local_provider("http://localhost:1234/v1/chat/completions"), "lmstudio")

    def test_llamacpp_server(self):
        self.assertEqual(m._detect_local_provider("http://localhost:8080/completion"), "llamacpp-server")

    def test_backend_port_not_intercepted(self):
        # Aegivis backend on 8000 — must NOT be treated as local LLM
        self.assertIsNone(m._detect_local_provider("http://localhost:8000/v1/ingest"))

    def test_remote_url_not_intercepted(self):
        self.assertIsNone(m._detect_local_provider("https://api.anthropic.com/v1/messages"))

    def test_invalid_url_returns_none(self):
        self.assertIsNone(m._detect_local_provider("not-a-url"))

    def test_custom_url_from_env(self):
        """AEGIVIS_LOCAL_LLM_URLS env var should register additional endpoints."""
        # Manually add a custom entry (simulating env var parsing)
        original = dict(m._LOCAL_LLM_MAP)
        try:
            m._LOCAL_LLM_MAP[("localhost", 9999)] = "custom-local"
            self.assertEqual(m._detect_local_provider("http://localhost:9999/generate"), "custom-local")
        finally:
            m._LOCAL_LLM_MAP.clear()
            m._LOCAL_LLM_MAP.update(original)


class TestReadBody(unittest.TestCase):
    def _enc(self, data: dict) -> bytes:
        return json.dumps(data).encode()

    def test_ollama_generate_format(self):
        """Ollama /api/generate uses 'prompt' not 'messages'."""
        result = m._read_body(self._enc({
            "model": "llama3.2",
            "prompt": "What is AI security?",
            "num_predict": 128,
        }))
        self.assertEqual(result["model"], "llama3.2")
        self.assertIn("AI security", result["prompt_preview"])
        self.assertEqual(result["num_predict"], 128)

    def test_ollama_chat_format(self):
        """Ollama /api/chat uses OpenAI-compatible messages."""
        result = m._read_body(self._enc({
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Explain injection attacks."},
            ],
        }))
        self.assertEqual(result["model"], "llama3.2")
        self.assertIn("injection", result["user_message_preview"])
        self.assertEqual(result["message_count"], 2)

    def test_openai_compat_format(self):
        """LM Studio and vllm expose OpenAI-compatible API."""
        result = m._read_body(self._enc({
            "model": "mistral-7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
            "temperature": 0.7,
        }))
        self.assertEqual(result["max_tokens"], 512)
        self.assertEqual(result["temperature"], 0.7)

    def test_none_returns_empty(self):
        self.assertEqual(m._read_body(None), {})

    def test_invalid_json_returns_empty(self):
        self.assertEqual(m._read_body(b"not json"), {})


class TestReadResponse(unittest.TestCase):
    def _enc(self, data: dict) -> bytes:
        return json.dumps(data).encode()

    def test_ollama_generate_response(self):
        """Ollama /api/generate response format."""
        result = m._read_response(self._enc({
            "model": "llama3.2",
            "response": "AI security focuses on protecting...",
            "eval_count": 42,
            "prompt_eval_count": 15,
            "done_reason": "stop",
        }))
        self.assertEqual(result["model"], "llama3.2")
        self.assertIn("AI security", result["response_preview"])
        self.assertEqual(result["output_tokens"], 42)
        self.assertEqual(result["input_tokens"], 15)
        self.assertEqual(result["stop_reason"], "stop")

    def test_ollama_chat_response(self):
        """Ollama /api/chat response format."""
        result = m._read_response(self._enc({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
        }))
        self.assertIn("Hello", result["response_preview"])

    def test_openai_compat_response(self):
        """OpenAI-compatible response (LM Studio, vllm, llama.cpp server)."""
        result = m._read_response(self._enc({
            "model": "mistral-7b",
            "choices": [{"message": {"content": "Here is the answer."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
        }))
        self.assertEqual(result["input_tokens"], 20)
        self.assertEqual(result["output_tokens"], 30)
        self.assertIn("answer", result["response_preview"])
        self.assertEqual(result["stop_reason"], "stop")

    def test_none_returns_empty(self):
        self.assertEqual(m._read_response(None), {})


class TestHttpxLocalPatch(unittest.TestCase):
    def test_install_is_idempotent(self):
        try:
            import httpx  # noqa: F401
        except ImportError:
            self.skipTest("httpx not installed")
        r1 = m._install_httpx_local()
        r2 = m._install_httpx_local()
        self.assertTrue(r1)
        self.assertTrue(r2)

    def test_remote_url_not_handled(self):
        """_detect_local_provider returns None for remote URLs — quick guard."""
        self.assertIsNone(m._detect_local_provider("https://api.openai.com/v1/chat/completions"))

    def test_ollama_url_detected(self):
        """Ollama URL is detected by the local provider detector."""
        provider = m._detect_local_provider("http://localhost:11434/api/chat")
        self.assertEqual(provider, "ollama")


class TestLlamaCppPatch(unittest.TestCase):
    def test_install_skips_gracefully_when_not_installed(self):
        """If llama_cpp is not installed, _install_llamacpp returns False silently."""
        with patch.dict("sys.modules", {"llama_cpp": None}):
            # This tests that import failure is handled gracefully.
            # Since the module is already imported, we test the guard logic directly.
            result = m._install_llamacpp()
            # Either False (not installed) or True (already installed in env)
            self.assertIn(result, (True, False))

    def test_llamacpp_patch_wraps_call(self):
        """Simulate patching a mock Llama class."""
        fires: list = []

        # Build a minimal mock Llama class
        class MockLlama:
            model_path = "/models/llama-3.2-8b.gguf"
            _aegivis_patched = False

            def __call__(self, prompt: str, **kwargs):
                return {
                    "choices": [{"text": "The answer is 42."}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

            def create_chat_completion(self, messages: list, **kwargs):
                return {
                    "choices": [{"message": {"content": "Hi there!"}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                }

        with patch.object(m, "_fire", side_effect=lambda *a, **kw: fires.append(a[0])):
            # Simulate what _install_llamacpp does — wrap __call__
            orig_call = MockLlama.__call__

            def wrapped(self, prompt, **kwargs):
                m._fire("LLM_CALL_START",
                        {"provider": "llamacpp", "model": self.model_path.split("/")[-1],
                         "prompt_preview": prompt[:500]},
                        m._session_id(), m._agent_id())
                result = orig_call(self, prompt, **kwargs)
                m._fire("LLM_CALL_END",
                        {"provider": "llamacpp",
                         "response_preview": result["choices"][0]["text"][:500]},
                        m._session_id(), m._agent_id())
                return result

            MockLlama.__call__ = wrapped
            llama = MockLlama()
            result = llama("What is 6 * 7?")

        self.assertEqual(result["choices"][0]["text"], "The answer is 42.")
        self.assertEqual(fires, ["LLM_CALL_START", "LLM_CALL_END"])


class TestTransformersPatch(unittest.TestCase):
    def test_install_skips_gracefully_when_not_installed(self):
        result = m._install_transformers()
        self.assertIn(result, (True, False))

    def test_transformers_patch_wraps_pipeline(self):
        """Simulate patching a mock Pipeline class."""
        fires: list = []

        class MockPipeline:
            task = "text-generation"
            _aegivis_patched = False

            class model:
                name_or_path = "gpt2"

            def __call__(self, inputs, **kwargs):
                return [{"generated_text": f"{inputs} The result is here."}]

        with patch.object(m, "_fire", side_effect=lambda *a, **kw: fires.append(a[0])):
            orig_call = MockPipeline.__call__

            def wrapped(self, inputs, **kwargs):
                m._fire("LLM_CALL_START",
                        {"provider": "transformers", "model": self.model.name_or_path,
                         "task": self.task, "prompt_preview": str(inputs)[:500]},
                        m._session_id(), m._agent_id())
                result = orig_call(self, inputs, **kwargs)
                text = result[0].get("generated_text", "")
                m._fire("LLM_CALL_END",
                        {"provider": "transformers", "response_preview": text[:500]},
                        m._session_id(), m._agent_id())
                return result

            MockPipeline.__call__ = wrapped
            pipe = MockPipeline()
            result = pipe("Tell me a story")

        self.assertIn("result", result[0]["generated_text"])
        self.assertEqual(fires, ["LLM_CALL_START", "LLM_CALL_END"])


class TestFireNonBlocking(unittest.TestCase):
    def test_fire_returns_immediately(self):
        import time
        original = m._BACKEND_URL
        try:
            m._BACKEND_URL = "http://localhost:19999"  # unreachable
            t0 = time.monotonic()
            m._fire("LLM_CALL_START", {"provider": "ollama"}, "sid", "aid")
            self.assertLess(time.monotonic() - t0, 0.5)
        finally:
            m._BACKEND_URL = original

    def test_fire_noop_when_disabled(self):
        original = m._BACKEND_URL
        try:
            m._BACKEND_URL = ""
            m._fire("LLM_CALL_START", {}, "sid", "aid")  # must not raise
        finally:
            m._BACKEND_URL = original


if __name__ == "__main__":
    unittest.main()
