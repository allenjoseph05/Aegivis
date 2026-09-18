"""
LiteLLM callback adapter for Aegivis.

LiteLLM is a universal LLM router used internally by CrewAI, Agno,
LlamaIndex (llm= parameter), Haystack, and dozens of other frameworks.
Installing this callback gives coverage across all of them without any
per-framework configuration.

Install::

    pip install 'aegivis[litellm]'

Usage::

    import litellm
    from aegivis.adapters.litellm import AegivisLiteLLMCallback

    litellm.callbacks = [AegivisLiteLLMCallback(agent_id="my-agent")]

    # Now every litellm.completion() call is captured, including calls made
    # internally by CrewAI, LlamaIndex, Agno, etc.

Or with aegivis.session for correlated session tracking::

    import aegivis
    from aegivis.adapters.litellm import AegivisLiteLLMCallback

    cb = AegivisLiteLLMCallback(agent_id="research-crew")
    litellm.callbacks = [cb]

    with aegivis.session(agent_id="research-crew") as s:
        crew.kickoff()   # all LLM calls automatically correlated to this session

Events emitted
--------------
- ``LLM_CALL_START``  — before every API call (model, messages preview, tools)
- ``LLM_CALL_END``    — after successful response (model, tokens, latency, response preview)
- ``LLM_CALL_ERROR``  — on API failure (error message, model)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("AEGIVIS_DEBUG", "") == "1"


class AegivisLiteLLMCallback:
    """
    LiteLLM ``CustomLogger``-compatible callback that emits Aegivis audit events.

    Parameters
    ----------
    agent_id    Identifier attached to every emitted event.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL`` env var.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY`` env var.
    """

    def __init__(
        self,
        *,
        agent_id: str = "litellm-agent",
        backend_url: str = "",
        api_key: str = "",
    ) -> None:
        self._agent_id    = agent_id
        self._backend_url = (backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000")).rstrip("/")
        self._api_key     = api_key or os.environ.get("AEGIVIS_API_KEY", "dev-dashboard-key")

    # ── Sync hooks ────────────────────────────────────────────────────────

    def log_pre_api_call(
        self,
        model: str,
        messages: list,
        kwargs: dict,
    ) -> None:
        """Called before every LiteLLM API call."""
        payload = self._build_request_payload(model, messages, kwargs)
        self._fire("LLM_CALL_START", payload)

    def log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after a successful LiteLLM API call."""
        payload = self._build_response_payload(kwargs, response_obj, start_time, end_time)
        self._fire("LLM_CALL_END", payload)

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after a failed LiteLLM API call."""
        payload = {
            "model":      kwargs.get("model", "unknown"),
            "provider":   _extract_provider(kwargs.get("model", "")),
            "error":      str(response_obj)[:300] if response_obj else "unknown error",
            "latency_ms": round((end_time - start_time) * 1000, 1) if start_time and end_time else None,
        }
        self._fire("LLM_CALL_ERROR", payload)

    # ── Async hooks (same logic, LiteLLM calls these in async contexts) ──

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: list,
        kwargs: dict,
    ) -> None:
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    # ── Payload builders ──────────────────────────────────────────────────

    def _build_request_payload(
        self, model: str, messages: list, kwargs: dict,
    ) -> dict:
        p: dict = {
            "model":    model,
            "provider": _extract_provider(model),
        }

        if messages:
            p["message_count"] = len(messages)
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content", "")
                    p["user_message_preview"] = str(content)[:500]
                    break
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    p["system_prompt_preview"] = str(m.get("content", ""))[:300]
                    break

        tools = kwargs.get("tools") or kwargs.get("functions") or []
        if tools:
            p["tool_count"] = len(tools)
            p["tool_names"] = [
                t.get("name") or (t.get("function") or {}).get("name") or "?"
                for t in tools[:10]
            ]

        for k in ("max_tokens", "temperature", "stream"):
            if k in kwargs:
                p[k] = kwargs[k]

        return p

    def _build_response_payload(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> dict:
        model   = kwargs.get("model", "unknown")
        p: dict = {
            "model":      model,
            "provider":   _extract_provider(model),
            "latency_ms": round((end_time - start_time) * 1000, 1) if start_time and end_time else None,
        }

        if response_obj is not None:
            # ModelResponse object (litellm standard)
            usage = getattr(response_obj, "usage", None)
            if usage:
                p["input_tokens"]  = getattr(usage, "prompt_tokens", None)
                p["output_tokens"] = getattr(usage, "completion_tokens", None)
                # Remove None values
                p = {k: v for k, v in p.items() if v is not None}

            choices = getattr(response_obj, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg:
                    content = getattr(msg, "content", None)
                    if content:
                        p["response_preview"] = str(content)[:500]
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        p["tool_calls_count"] = len(tool_calls)

                finish_reason = getattr(choices[0], "finish_reason", None)
                if finish_reason:
                    p["stop_reason"] = str(finish_reason)

        return p

    # ── Fire-and-forget ───────────────────────────────────────────────────

    def _fire(self, event_type: str, payload: dict) -> None:
        if not self._backend_url:
            return

        session_id = os.environ.get("AEGIVIS_SESSION_ID") or "litellm-session"
        agent_id   = os.environ.get("AEGIVIS_AGENT_ID")  or self._agent_id

        event = {
            "event_type":         event_type,
            "agent_id":           agent_id,
            "session_id":         session_id,
            "timestamp_ns":       time.time_ns(),
            "interception_layer": "sdk-litellm",
            "provider":           payload.get("provider", "unknown"),
            "model":              payload.get("model", ""),
            "payload":            payload,
        }
        body    = json.dumps(event, default=str).encode()
        headers = {"Content-Type": "application/json", "X-API-Key": self._api_key}
        url     = self._backend_url + "/v1/ingest"

        if _DEBUG:
            logger.debug("aegivis.litellm: %s %s", event_type, payload.get("model", ""))

        def _post() -> None:
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=3):
                    pass
            except Exception as exc:
                if _DEBUG:
                    logger.debug("aegivis.litellm: post failed: %s", exc)

        threading.Thread(target=_post, daemon=True).start()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_provider(model: str) -> str:
    """
    Infer provider from LiteLLM model string.
    LiteLLM uses prefixes: "anthropic/claude-3", "openai/gpt-4o", "groq/llama3", etc.
    Falls back to model name heuristics when no prefix is present.
    """
    if "/" in model:
        prefix = model.split("/")[0].lower()
        # LiteLLM prefix → clean name
        _PREFIX_MAP = {
            "anthropic":    "anthropic",
            "openai":       "openai",
            "azure":        "azure-openai",
            "azure_ai":     "azure-inference",
            "google":       "google-gemini",
            "vertex_ai":    "google-vertex",
            "cohere":       "cohere",
            "mistral":      "mistral",
            "groq":         "groq",
            "together_ai":  "together",
            "perplexity":   "perplexity",
            "deepseek":     "deepseek",
            "xai":          "xai-grok",
            "bedrock":      "aws-bedrock",
            "fireworks_ai": "fireworks",
            "sambanova":    "sambanova",
            "cerebras":     "cerebras",
            "openrouter":   "openrouter",
            "replicate":    "replicate",
        }
        if prefix in _PREFIX_MAP:
            return _PREFIX_MAP[prefix]
        return prefix

    # No prefix — guess from model name
    m = model.lower()
    if "claude" in m:     return "anthropic"
    if "gpt" in m:        return "openai"
    if "gemini" in m:     return "google-gemini"
    if "llama" in m:      return "meta"
    if "mixtral" in m:    return "mistral"
    if "command" in m:    return "cohere"
    return "unknown"
