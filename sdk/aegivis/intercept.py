"""
Aegivis Zero-Config Interceptor
================================
Patches ``httpx`` at the transport layer so every outbound LLM API call from
every Python AI framework is captured automatically — no ``base_url`` config
required and no proxy URL to set.

Import once in your agent entrypoint::

    import aegivis.intercept   # that's it

Covered automatically:
    anthropic SDK, openai SDK, LangChain, LangGraph, AutoGen, CrewAI,
    LlamaIndex, LiteLLM, Pydantic AI, smolagents, Haystack, DSPy,
    Google Generative AI SDK, Cohere SDK — anything that uses httpx internally.

Supported providers (auto-detected by hostname):
    anthropic · openai · azure-openai · google-gemini · google-vertex
    cohere · mistral · groq · together · perplexity · deepseek · xai-grok
    aws-bedrock · fireworks · sambanova · cerebras · openrouter

Environment variables::

    AEGIVIS_BACKEND_URL   Where to ship events (default: http://localhost:8000)
    AEGIVIS_API_KEY       API key (default: dev-dashboard-key)
    AEGIVIS_AGENT_ID      Agent label attached to events (default: intercepted-agent)
    AEGIVIS_INTERCEPT     Set "false" to disable without removing the import
    AEGIVIS_DEBUG         Set "1" to log intercept activity to stderr

Works alongside aegivis.session:
    with aegivis.session(agent_id="my-agent") as s:
        ...   # session_id/agent_id from the context flow through automatically
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — read at import time; user can set env vars before importing this.
# ---------------------------------------------------------------------------

_BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY     = os.environ.get("AEGIVIS_API_KEY", "dev-dashboard-key")
_AGENT_ID    = os.environ.get("AEGIVIS_AGENT_ID", "intercepted-agent")
_DEBUG       = os.environ.get("AEGIVIS_DEBUG", "") == "1"
_ENABLED     = os.environ.get("AEGIVIS_INTERCEPT", "true").lower() not in ("false", "0", "no")

# Skip hosts that are the Aegivis backend/proxy to avoid infinite loops.
def _skip_hosts() -> frozenset[str]:
    hosts = {"localhost", "127.0.0.1", "::1"}
    for url in (_BACKEND_URL, os.environ.get("AEGIVIS_PROXY_URL", "")):
        if url:
            # Extract hostname from URL, ignore port
            host = url.split("//")[-1].split("/")[0].split(":")[0]
            if host:
                hosts.add(host)
    return frozenset(hosts)

_SKIP_HOSTS = _skip_hosts()

# ---------------------------------------------------------------------------
# Provider detection
# (hostname_fragment, provider_name) — checked in order, first match wins
# ---------------------------------------------------------------------------

_PROVIDER_MAP: list[tuple[str, str]] = [
    ("api.anthropic.com",                 "anthropic"),
    ("api.openai.com",                    "openai"),
    (".openai.azure.com",                 "azure-openai"),
    ("inference.ai.azure.com",            "azure-inference"),
    ("generativelanguage.googleapis.com", "google-gemini"),
    ("aiplatform.googleapis.com",         "google-vertex"),
    ("api.cohere.ai",                     "cohere"),
    ("api.cohere.com",                    "cohere"),
    ("api.mistral.ai",                    "mistral"),
    ("api.groq.com",                      "groq"),
    ("api.together.xyz",                  "together"),
    ("api.perplexity.ai",                 "perplexity"),
    ("api.deepseek.com",                  "deepseek"),
    ("api.x.ai",                          "xai-grok"),
    ("bedrock-runtime.amazonaws.com",     "aws-bedrock"),
    ("api.fireworks.ai",                  "fireworks"),
    ("api.sambanova.ai",                  "sambanova"),
    ("api.cerebras.ai",                   "cerebras"),
    ("openrouter.ai",                     "openrouter"),
    ("api.replicate.com",                 "replicate"),
    ("api.nvidia.com",                    "nvidia"),
    ("integrate.api.nvidia.com",          "nvidia"),
]


def _detect_provider(url: Any) -> str | None:
    """Return provider name if URL is a known LLM API, else None."""
    try:
        host = str(getattr(url, "host", url)).lower()
    except Exception:
        return None

    # Never intercept our own backend/proxy traffic.
    if any(skip in host for skip in _SKIP_HOSTS if skip):
        return None

    for fragment, provider in _PROVIDER_MAP:
        if fragment in host:
            return provider
    return None


# ---------------------------------------------------------------------------
# Request payload extraction
# ---------------------------------------------------------------------------

def _read_request(request: Any) -> dict:
    """Extract model, messages, tools from request body. Never raises."""
    try:
        raw = getattr(request, "content", None) or b""
        if not raw:
            return {}
        body = json.loads(raw)
    except Exception:
        return {}

    p: dict = {}

    if model := body.get("model"):
        p["model"] = str(model)

    messages: list = body.get("messages") or []
    if messages:
        p["message_count"] = len(messages)
        # Extract last user message preview
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                raw_content = m.get("content", "")
                if isinstance(raw_content, list):  # Anthropic content-block format
                    text = " ".join(
                        b.get("text", "") for b in raw_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = str(raw_content)
                p["user_message_preview"] = text[:500]
                break

    # System prompt — Anthropic top-level or OpenAI system message
    sys: Any = body.get("system") or ""
    if not sys:
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                sys = m.get("content", "")
                break
    if sys:
        sys_text = (
            " ".join(b.get("text", "") for b in sys if isinstance(b, dict))
            if isinstance(sys, list) else str(sys)
        )
        p["system_prompt_preview"] = sys_text[:300]

    # Tools / functions
    tools: list = body.get("tools") or body.get("functions") or []
    if tools:
        p["tool_count"] = len(tools)
        p["tool_names"] = [
            t.get("name") or (t.get("function") or {}).get("name") or "?"
            for t in tools[:10]
        ]

    for k in ("max_tokens", "max_completion_tokens", "temperature", "stream"):
        if k in body:
            p[k] = body[k]

    return p


def _read_response(response: Any, req_payload: dict) -> dict:
    """Extract completion text, usage, stop reason from response body. Never raises."""
    try:
        # Don't consume streaming responses — that would break the caller.
        if getattr(response, "is_stream", False) or getattr(response, "headers", {}).get(
            "content-type", ""
        ).startswith("text/event-stream"):
            return {"streamed": True, "model": req_payload.get("model", "")}

        raw = getattr(response, "content", None) or b""
        if not raw:
            return {}
        body = json.loads(raw)
    except Exception:
        return {}

    p: dict = {}

    if model := body.get("model"):
        p["model"] = str(model)

    # Token usage (Anthropic and OpenAI formats)
    usage = body.get("usage") or {}
    input_tok  = usage.get("input_tokens")  or usage.get("prompt_tokens")
    output_tok = usage.get("output_tokens") or usage.get("completion_tokens")
    if input_tok  is not None:
        p["input_tokens"]  = input_tok
    if output_tok is not None:
        p["output_tokens"] = output_tok

    # Stop reason
    stop = (
        body.get("stop_reason")
        or ((body.get("choices") or [{}])[0].get("finish_reason"))
    )
    if stop:
        p["stop_reason"] = str(stop)

    # Response text — Anthropic content blocks or OpenAI choices
    text = ""
    for block in (body.get("content") or []):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            break
    if not text:
        choices = body.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
    if text:
        p["response_preview"] = str(text)[:500]

    # Tool use in response
    tool_use = body.get("tool_use") or (
        ((body.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    )
    if tool_use:
        p["tool_calls_count"] = len(tool_use)

    return p


# ---------------------------------------------------------------------------
# Event shipping — uses urllib to avoid httpx recursion
# ---------------------------------------------------------------------------

def _fire(
    event_type: str,
    payload: dict,
    session_id: str,
    agent_id: str,
) -> None:
    """Fire-and-forget: POST event to Aegivis backend via urllib (no httpx recursion)."""
    if not _BACKEND_URL or not _ENABLED:
        return

    event = {
        "event_type":          event_type,
        "agent_id":            agent_id,
        "session_id":          session_id,
        "timestamp_ns":        time.time_ns(),
        "interception_layer":  "sdk-intercept",
        "provider":            payload.get("provider", "unknown"),
        "model":               payload.get("model", ""),
        "payload":             payload,
    }
    body = json.dumps(event, default=str).encode()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key":    _API_KEY,
    }

    def _post() -> None:
        try:
            req = urllib.request.Request(
                _BACKEND_URL + "/v1/ingest",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception as exc:
            if _DEBUG:
                logger.debug("aegivis.intercept: post failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


# ---------------------------------------------------------------------------
# Session / agent ID resolution (reads env set by aegivis.session context)
# ---------------------------------------------------------------------------

def _session_id() -> str:
    """Return active session ID or generate a stable one for this thread."""
    return os.environ.get("AEGIVIS_SESSION_ID") or f"intercept-{uuid.uuid4().hex[:12]}"


def _agent_id() -> str:
    return os.environ.get("AEGIVIS_AGENT_ID") or _AGENT_ID


# ---------------------------------------------------------------------------
# httpx patching
# ---------------------------------------------------------------------------

def _install() -> bool:
    """
    Monkey-patch httpx.Client.send and httpx.AsyncClient.send.

    Both sync and async paths are patched so coverage is complete regardless
    of whether the calling framework uses asyncio or threading.

    Returns True on success, False if httpx is not installed.
    """
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        if _DEBUG:
            logger.debug("aegivis.intercept: httpx not installed — patch skipped")
        return False

    if getattr(httpx, "_aegivis_patched", False):
        return True  # idempotent

    _orig_sync  = httpx.Client.send
    _orig_async = httpx.AsyncClient.send

    # ── Sync patch ────────────────────────────────────────────────────────
    def _sync_send(self: Any, request: Any, **kwargs: Any) -> Any:
        provider = _detect_provider(request.url)
        if not provider:
            return _orig_sync(self, request, **kwargs)

        sid = _session_id()
        aid = _agent_id()

        req_payload = _read_request(request)
        req_payload["provider"] = provider
        if _DEBUG:
            logger.debug(
                "aegivis.intercept: %s %s session=%s",
                provider, getattr(request.url, "path", ""), sid,
            )
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = _orig_sync(self, request, **kwargs)
        except Exception as exc:
            _fire(
                "LLM_CALL_ERROR",
                {"provider": provider, "error": str(exc)[:300], "model": req_payload.get("model", "")},
                sid, aid,
            )
            raise

        resp_payload = _read_response(response, req_payload)
        resp_payload["provider"]     = provider
        resp_payload["latency_ms"]   = round((time.monotonic() - t0) * 1000, 1)
        resp_payload["status_code"]  = response.status_code
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    # ── Async patch ───────────────────────────────────────────────────────
    async def _async_send(self: Any, request: Any, **kwargs: Any) -> Any:
        provider = _detect_provider(request.url)
        if not provider:
            return await _orig_async(self, request, **kwargs)

        sid = _session_id()
        aid = _agent_id()

        req_payload = _read_request(request)
        req_payload["provider"] = provider
        if _DEBUG:
            logger.debug(
                "aegivis.intercept: async %s %s session=%s",
                provider, getattr(request.url, "path", ""), sid,
            )
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = await _orig_async(self, request, **kwargs)
        except Exception as exc:
            _fire(
                "LLM_CALL_ERROR",
                {"provider": provider, "error": str(exc)[:300], "model": req_payload.get("model", "")},
                sid, aid,
            )
            raise

        resp_payload = _read_response(response, req_payload)
        resp_payload["provider"]    = provider
        resp_payload["latency_ms"]  = round((time.monotonic() - t0) * 1000, 1)
        resp_payload["status_code"] = response.status_code
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    httpx.Client.send      = _sync_send      # type: ignore[method-assign]
    httpx.AsyncClient.send = _async_send     # type: ignore[method-assign]
    httpx._aegivis_patched = True            # type: ignore[attr-defined]

    if _DEBUG:
        logger.debug("aegivis.intercept: httpx patched (sync + async)")

    return True


def uninstall() -> None:
    """
    Remove the httpx patches (useful in tests or when switching to proxy mode).
    After calling this, LLM calls are no longer intercepted.
    """
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return

    if not getattr(httpx, "_aegivis_patched", False):
        return

    # Retrieve originals stored in closure via __wrapped__ attribute
    orig_sync  = getattr(httpx.Client.send,      "__aegivis_orig__", None)
    orig_async = getattr(httpx.AsyncClient.send, "__aegivis_orig__", None)
    if orig_sync:
        httpx.Client.send = orig_sync          # type: ignore[method-assign]
    if orig_async:
        httpx.AsyncClient.send = orig_async    # type: ignore[method-assign]
    httpx._aegivis_patched = False             # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Auto-install on import
# ---------------------------------------------------------------------------

if _ENABLED:
    _install()
