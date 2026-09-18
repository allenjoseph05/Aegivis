"""
Aegivis Zero-Config Interceptor — requests & aiohttp
=====================================================
Patches ``requests.Session.send`` and ``aiohttp.ClientSession._request``
so every outbound LLM API call from Python agents using these libraries is
captured automatically — no proxy URL required.

Import once alongside (or instead of) ``aegivis.intercept``::

    import aegivis.intercept_requests   # patches requests + aiohttp

Why this exists:
    ``aegivis.intercept`` patches httpx — most modern AI SDKs use httpx.
    But many agents (older LangChain, custom integrations, raw REST clients)
    use ``requests`` or ``aiohttp`` directly.  Both interceptors can be active
    at the same time without conflict.

Covered by this module:
    Any Python agent using ``requests`` or ``aiohttp`` to call LLM providers,
    including older LangChain (≤ 0.1), raw OpenAI API scripts, boto3 Bedrock
    calls via requests, and custom framework integrations.

Same environment variables as ``aegivis.intercept``::

    AEGIVIS_BACKEND_URL   Where to ship events (default: http://localhost:8000)
    AEGIVIS_API_KEY       API key (default: dev-dashboard-key)
    AEGIVIS_AGENT_ID      Agent label attached to events
    AEGIVIS_INTERCEPT     Set "false" to disable without removing the import
    AEGIVIS_DEBUG         Set "1" to log intercept activity to stderr
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
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — same env vars as intercept.py
# ---------------------------------------------------------------------------

_BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY     = os.environ.get("AEGIVIS_API_KEY", "dev-dashboard-key")
_AGENT_ID    = os.environ.get("AEGIVIS_AGENT_ID", "intercepted-agent")
_DEBUG       = os.environ.get("AEGIVIS_DEBUG", "") == "1"
_ENABLED     = os.environ.get("AEGIVIS_INTERCEPT", "true").lower() not in ("false", "0", "no")


def _skip_hosts() -> frozenset[str]:
    hosts = {"localhost", "127.0.0.1", "::1"}
    for url in (_BACKEND_URL, os.environ.get("AEGIVIS_PROXY_URL", "")):
        if url:
            host = url.split("//")[-1].split("/")[0].split(":")[0]
            if host:
                hosts.add(host)
    return frozenset(hosts)


_SKIP_HOSTS = _skip_hosts()

# Same provider map as intercept.py — must stay in sync.
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


def _detect_provider(url: str) -> str | None:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if not host:
        return None
    if any(skip in host for skip in _SKIP_HOSTS if skip):
        return None
    for fragment, provider in _PROVIDER_MAP:
        if fragment in host:
            return provider
    return None


# ---------------------------------------------------------------------------
# Request / response payload extraction
# ---------------------------------------------------------------------------

def _read_body(body: Any) -> dict:
    """Parse JSON body from bytes, str, or any serialisable type. Never raises."""
    try:
        if body is None:
            return {}
        if isinstance(body, (bytes, bytearray)):
            data = json.loads(body)
        elif isinstance(body, str):
            data = json.loads(body)
        else:
            return {}
    except Exception:
        return {}

    p: dict = {}
    if model := data.get("model"):
        p["model"] = str(model)

    messages: list = data.get("messages") or []
    if messages:
        p["message_count"] = len(messages)
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                raw = m.get("content", "")
                text = (
                    " ".join(b.get("text", "") for b in raw
                             if isinstance(b, dict) and b.get("type") == "text")
                    if isinstance(raw, list) else str(raw)
                )
                p["user_message_preview"] = text[:500]
                break

    sys: Any = data.get("system") or ""
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

    tools: list = data.get("tools") or data.get("functions") or []
    if tools:
        p["tool_count"] = len(tools)
        p["tool_names"] = [
            t.get("name") or (t.get("function") or {}).get("name") or "?"
            for t in tools[:10]
        ]

    for k in ("max_tokens", "temperature", "stream"):
        if k in data:
            p[k] = data[k]

    return p


def _read_response(content: bytes | None) -> dict:
    """Parse JSON response body. Never raises."""
    try:
        if not content:
            return {}
        data = json.loads(content)
    except Exception:
        return {}

    p: dict = {}
    if model := data.get("model"):
        p["model"] = str(model)

    usage = data.get("usage") or {}
    input_tok  = usage.get("input_tokens")  or usage.get("prompt_tokens")
    output_tok = usage.get("output_tokens") or usage.get("completion_tokens")
    if input_tok  is not None:
        p["input_tokens"]  = input_tok
    if output_tok is not None:
        p["output_tokens"] = output_tok

    stop = (
        data.get("stop_reason")
        or ((data.get("choices") or [{}])[0].get("finish_reason"))
    )
    if stop:
        p["stop_reason"] = str(stop)

    text = ""
    for block in (data.get("content") or []):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            break
    if not text:
        choices = data.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
    if text:
        p["response_preview"] = str(text)[:500]

    return p


# ---------------------------------------------------------------------------
# Event shipping — urllib to avoid requests/aiohttp recursion
# ---------------------------------------------------------------------------

def _fire(event_type: str, payload: dict, session_id: str, agent_id: str) -> None:
    if not _BACKEND_URL or not _ENABLED:
        return
    event = {
        "event_type":         event_type,
        "agent_id":           agent_id,
        "session_id":         session_id,
        "timestamp_ns":       time.time_ns(),
        "interception_layer": "sdk-intercept-requests",
        "provider":           payload.get("provider", "unknown"),
        "model":              payload.get("model", ""),
        "payload":            payload,
    }
    body = json.dumps(event, default=str).encode()
    headers = {"Content-Type": "application/json", "X-API-Key": _API_KEY}

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
                logger.debug("aegivis.intercept_requests: post failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


def _session_id() -> str:
    return os.environ.get("AEGIVIS_SESSION_ID") or f"intercept-{uuid.uuid4().hex[:12]}"


def _agent_id() -> str:
    return os.environ.get("AEGIVIS_AGENT_ID") or _AGENT_ID


# ---------------------------------------------------------------------------
# requests patch — patches Session.send (lowest-level, all requests flow here)
# ---------------------------------------------------------------------------

def _install_requests() -> bool:
    """Patch requests.Session.send. Returns True if patched, False if requests not installed."""
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        if _DEBUG:
            logger.debug("aegivis.intercept_requests: requests not installed — skipped")
        return False

    if getattr(requests.Session, "_aegivis_patched", False):
        return True

    _orig_send = requests.Session.send

    def _patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
        url = getattr(request, "url", "") or ""
        provider = _detect_provider(url)
        if not provider:
            return _orig_send(self, request, **kwargs)

        sid = _session_id()
        aid = _agent_id()

        req_payload = _read_body(getattr(request, "body", None))
        req_payload["provider"] = provider
        if _DEBUG:
            logger.debug("aegivis.intercept_requests: requests %s %s session=%s", provider, url[:80], sid)
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = _orig_send(self, request, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": provider, "error": str(exc)[:300], "model": req_payload.get("model", "")},
                  sid, aid)
            raise

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            _fire("LLM_CALL_END",
                  {"provider": provider, "streamed": True, "latency_ms": latency_ms,
                   "status_code": response.status_code},
                  sid, aid)
            return response

        resp_payload = _read_response(response.content)
        resp_payload["provider"]     = provider
        resp_payload["latency_ms"]   = latency_ms
        resp_payload["status_code"]  = response.status_code
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    _patched_send._aegivis_orig_ = _orig_send        # type: ignore[attr-defined]
    requests.Session.send = _patched_send             # type: ignore[method-assign]
    requests.Session._aegivis_patched = True          # type: ignore[attr-defined]

    if _DEBUG:
        logger.debug("aegivis.intercept_requests: requests.Session.send patched")
    return True


# ---------------------------------------------------------------------------
# aiohttp patch — patches ClientSession._request (internal, all requests flow here)
# ---------------------------------------------------------------------------

def _install_aiohttp() -> bool:
    """Patch aiohttp.ClientSession._request. Returns True if patched, False if aiohttp not installed."""
    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        if _DEBUG:
            logger.debug("aegivis.intercept_requests: aiohttp not installed — skipped")
        return False

    if getattr(aiohttp.ClientSession, "_aegivis_patched", False):
        return True

    _orig_request = aiohttp.ClientSession._request

    async def _patched_request(self: Any, method: str, str_or_url: Any, **kwargs: Any) -> Any:
        url = str(str_or_url)
        provider = _detect_provider(url)
        if not provider:
            return await _orig_request(self, method, str_or_url, **kwargs)

        sid = _session_id()
        aid = _agent_id()

        # LLM calls typically use json= kwarg (not data=)
        json_body = kwargs.get("json")
        raw_body  = kwargs.get("data")
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode()
        elif isinstance(raw_body, (bytes, bytearray, str)):
            body_bytes = raw_body if isinstance(raw_body, (bytes, bytearray)) else raw_body.encode()
        else:
            body_bytes = None

        req_payload = _read_body(body_bytes)
        req_payload["provider"] = provider
        if _DEBUG:
            logger.debug("aegivis.intercept_requests: aiohttp %s %s session=%s", provider, url[:80], sid)
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = await _orig_request(self, method, str_or_url, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": provider, "error": str(exc)[:300], "model": req_payload.get("model", "")},
                  sid, aid)
            raise

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            _fire("LLM_CALL_END",
                  {"provider": provider, "streamed": True, "latency_ms": latency_ms,
                   "status_code": response.status},
                  sid, aid)
            return response

        try:
            content = await response.read()
        except Exception:
            content = b""

        resp_payload = _read_response(content)
        resp_payload["provider"]    = provider
        resp_payload["latency_ms"]  = latency_ms
        resp_payload["status_code"] = response.status
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    _patched_request._aegivis_orig_ = _orig_request      # type: ignore[attr-defined]
    aiohttp.ClientSession._request = _patched_request     # type: ignore[method-assign]
    aiohttp.ClientSession._aegivis_patched = True         # type: ignore[attr-defined]

    if _DEBUG:
        logger.debug("aegivis.intercept_requests: aiohttp.ClientSession._request patched")
    return True


# ---------------------------------------------------------------------------
# Auto-install on import
# ---------------------------------------------------------------------------

if _ENABLED:
    _install_requests()
    _install_aiohttp()
