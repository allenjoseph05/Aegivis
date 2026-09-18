"""
Aegivis Local Model Interceptor
================================
Captures LLM calls to locally-running model servers and Python-library-based
models that make no HTTP calls at all.

Import once (alongside or instead of ``aegivis.intercept``)::

    import aegivis.intercept_local

Covered automatically:

HTTP-server local models (detected by host:port):
    Ollama          — http://localhost:11434
    LM Studio       — http://localhost:1234
    vllm / llama.cpp server — any port listed in AEGIVIS_LOCAL_LLM_URLS

Python-library local models (patched directly):
    llama-cpp-python  — ``pip install llama-cpp-python``
    HuggingFace Transformers — ``pip install transformers``

How it co-exists with ``aegivis.intercept``:
    This module installs its own httpx/requests wrapper that runs *before*
    the general interceptor.  For local LLM endpoints (e.g. localhost:11434)
    it fires an event then lets the request pass through — the general
    interceptor skips localhost so no double-counting occurs.
    For non-local URLs it passes control directly to the next wrapper.

Environment variables::

    AEGIVIS_LOCAL_LLM_URLS  Comma-separated additional local server URLs to
                             intercept, e.g. "http://localhost:8080,http://10.0.0.5:11434"
    AEGIVIS_BACKEND_URL     Where to ship events (default: http://localhost:8000)
    AEGIVIS_API_KEY         API key (default: dev-dashboard-key)
    AEGIVIS_AGENT_ID        Agent label attached to events
    AEGIVIS_INTERCEPT       Set "false" to disable
    AEGIVIS_DEBUG           Set "1" to log to stderr
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
# Config
# ---------------------------------------------------------------------------

_BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY     = os.environ.get("AEGIVIS_API_KEY", "dev-dashboard-key")
_AGENT_ID    = os.environ.get("AEGIVIS_AGENT_ID", "intercepted-agent")
_DEBUG       = os.environ.get("AEGIVIS_DEBUG", "") == "1"
_ENABLED     = os.environ.get("AEGIVIS_INTERCEPT", "true").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# Local LLM endpoint registry
# (host, port) → provider name
# ---------------------------------------------------------------------------

_LOCAL_LLM_MAP: dict[tuple[str, int], str] = {
    ("localhost",  11434): "ollama",
    ("127.0.0.1",  11434): "ollama",
    ("0.0.0.0",    11434): "ollama",
    ("localhost",   1234): "lmstudio",
    ("127.0.0.1",   1234): "lmstudio",
    ("localhost",   8080): "llamacpp-server",
    ("127.0.0.1",   8080): "llamacpp-server",
}


def _load_custom_urls() -> None:
    """Parse AEGIVIS_LOCAL_LLM_URLS and add to the registry."""
    raw = os.environ.get("AEGIVIS_LOCAL_LLM_URLS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            parsed = urlparse(entry)
            host = parsed.hostname or "localhost"
            port = parsed.port or 80
            _LOCAL_LLM_MAP[(host, port)] = "local-llm"
            if host == "localhost":
                _LOCAL_LLM_MAP[("127.0.0.1", port)] = "local-llm"
        except Exception:
            pass


_load_custom_urls()


def _detect_local_provider(url: str) -> str | None:
    """Return local provider name if URL is a known local LLM endpoint, else None."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return _LOCAL_LLM_MAP.get((host, port))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared body extraction (same as intercept_requests.py)
# ---------------------------------------------------------------------------

def _read_body(body: Any) -> dict:
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

    # Ollama uses "prompt" for generate endpoint (non-chat)
    if prompt := data.get("prompt"):
        p["prompt_preview"] = str(prompt)[:500]

    tools: list = data.get("tools") or []
    if tools:
        p["tool_count"] = len(tools)
        p["tool_names"] = [
            t.get("name") or (t.get("function") or {}).get("name") or "?"
            for t in tools[:10]
        ]

    for k in ("max_tokens", "temperature", "stream", "num_predict", "num_ctx"):
        if k in data:
            p[k] = data[k]

    return p


def _read_response(content: bytes | None) -> dict:
    try:
        if not content:
            return {}
        data = json.loads(content)
    except Exception:
        return {}

    p: dict = {}
    if model := data.get("model"):
        p["model"] = str(model)

    # Ollama generate response: {"response": "...", "eval_count": N, "prompt_eval_count": N}
    if response_text := data.get("response"):
        p["response_preview"] = str(response_text)[:500]
        if eval_count := data.get("eval_count"):
            p["output_tokens"] = eval_count
        if prompt_eval := data.get("prompt_eval_count"):
            p["input_tokens"] = prompt_eval

    # Ollama chat response: {"message": {"role": "assistant", "content": "..."}}
    if message := data.get("message"):
        p["response_preview"] = str(message.get("content", ""))[:500]

    # OpenAI-compatible (used by LM Studio, vllm, llama.cpp server)
    usage = data.get("usage") or {}
    if not p.get("input_tokens"):
        if tok := usage.get("prompt_tokens") or usage.get("input_tokens"):
            p["input_tokens"] = tok
    if not p.get("output_tokens"):
        if tok := usage.get("completion_tokens") or usage.get("output_tokens"):
            p["output_tokens"] = tok

    if not p.get("response_preview"):
        choices = data.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
            if text:
                p["response_preview"] = str(text)[:500]
            if fr := choices[0].get("finish_reason"):
                p["stop_reason"] = str(fr)

    if done_reason := data.get("done_reason"):
        p["stop_reason"] = str(done_reason)

    return p


# ---------------------------------------------------------------------------
# Event shipping
# ---------------------------------------------------------------------------

def _fire(event_type: str, payload: dict, session_id: str, agent_id: str) -> None:
    if not _BACKEND_URL or not _ENABLED:
        return
    event = {
        "event_type":         event_type,
        "agent_id":           agent_id,
        "session_id":         session_id,
        "timestamp_ns":       time.time_ns(),
        "interception_layer": "sdk-intercept-local",
        "provider":           payload.get("provider", "local"),
        "model":              payload.get("model", ""),
        "payload":            payload,
    }
    body = json.dumps(event, default=str).encode()
    headers = {"Content-Type": "application/json", "X-API-Key": _API_KEY}

    def _post() -> None:
        try:
            req = urllib.request.Request(
                _BACKEND_URL + "/v1/ingest", data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception as exc:
            if _DEBUG:
                logger.debug("aegivis.intercept_local: post failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


def _session_id() -> str:
    return os.environ.get("AEGIVIS_SESSION_ID") or f"local-{uuid.uuid4().hex[:12]}"


def _agent_id() -> str:
    return os.environ.get("AEGIVIS_AGENT_ID") or _AGENT_ID


# ---------------------------------------------------------------------------
# httpx patch — local endpoints only
# ---------------------------------------------------------------------------

def _install_httpx_local() -> bool:
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return False

    if getattr(httpx, "_aegivis_local_patched", False):
        return True

    # Wrap whatever send is current (may already be wrapped by aegivis.intercept).
    _next_sync  = httpx.Client.send
    _next_async = httpx.AsyncClient.send

    def _sync_send(self: Any, request: Any, **kwargs: Any) -> Any:
        url = str(getattr(request.url, "raw", None) or request.url)
        provider = _detect_local_provider(url)
        if not provider:
            return _next_sync(self, request, **kwargs)

        sid = _session_id()
        aid = _agent_id()
        req_payload = _read_body(getattr(request, "content", None))
        req_payload["provider"] = provider
        if _DEBUG:
            logger.debug("aegivis.intercept_local: httpx %s %s", provider, url[:80])
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = _next_sync(self, request, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": provider, "error": str(exc)[:300]}, sid, aid)
            raise

        content_type = response.headers.get("content-type", "")
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        if "text/event-stream" in content_type:
            _fire("LLM_CALL_END",
                  {"provider": provider, "streamed": True, "latency_ms": latency_ms}, sid, aid)
            return response

        resp_payload = _read_response(getattr(response, "content", None))
        resp_payload["provider"]   = provider
        resp_payload["latency_ms"] = latency_ms
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    async def _async_send(self: Any, request: Any, **kwargs: Any) -> Any:
        url = str(getattr(request.url, "raw", None) or request.url)
        provider = _detect_local_provider(url)
        if not provider:
            return await _next_async(self, request, **kwargs)

        sid = _session_id()
        aid = _agent_id()
        req_payload = _read_body(getattr(request, "content", None))
        req_payload["provider"] = provider
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = await _next_async(self, request, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": provider, "error": str(exc)[:300]}, sid, aid)
            raise

        content_type = response.headers.get("content-type", "")
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        if "text/event-stream" in content_type:
            _fire("LLM_CALL_END",
                  {"provider": provider, "streamed": True, "latency_ms": latency_ms}, sid, aid)
            return response

        resp_payload = _read_response(getattr(response, "content", None))
        resp_payload["provider"]   = provider
        resp_payload["latency_ms"] = latency_ms
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    httpx.Client.send      = _sync_send       # type: ignore[method-assign]
    httpx.AsyncClient.send = _async_send      # type: ignore[method-assign]
    httpx._aegivis_local_patched = True       # type: ignore[attr-defined]
    if _DEBUG:
        logger.debug("aegivis.intercept_local: httpx local patch installed")
    return True


# ---------------------------------------------------------------------------
# requests patch — local endpoints only
# ---------------------------------------------------------------------------

def _install_requests_local() -> bool:
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        return False

    if getattr(requests.Session, "_aegivis_local_patched", False):
        return True

    _next_send = requests.Session.send

    def _patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
        url = getattr(request, "url", "") or ""
        provider = _detect_local_provider(url)
        if not provider:
            return _next_send(self, request, **kwargs)

        sid = _session_id()
        aid = _agent_id()
        req_payload = _read_body(getattr(request, "body", None))
        req_payload["provider"] = provider
        _fire("LLM_CALL_START", req_payload, sid, aid)
        t0 = time.monotonic()

        try:
            response = _next_send(self, request, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": provider, "error": str(exc)[:300]}, sid, aid)
            raise

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            _fire("LLM_CALL_END",
                  {"provider": provider, "streamed": True, "latency_ms": latency_ms}, sid, aid)
            return response

        resp_payload = _read_response(response.content)
        resp_payload["provider"]   = provider
        resp_payload["latency_ms"] = latency_ms
        _fire("LLM_CALL_END", resp_payload, sid, aid)
        return response

    requests.Session.send = _patched_send               # type: ignore[method-assign]
    requests.Session._aegivis_local_patched = True      # type: ignore[attr-defined]
    if _DEBUG:
        logger.debug("aegivis.intercept_local: requests local patch installed")
    return True


# ---------------------------------------------------------------------------
# llama-cpp-python patch
# ---------------------------------------------------------------------------

def _install_llamacpp() -> bool:
    """
    Patch ``llama_cpp.Llama.__call__`` and ``Llama.create_chat_completion``.

    llama-cpp-python runs the model in-process — there are no HTTP calls.
    We wrap at the Python method level to capture prompt/response/tokens.
    """
    try:
        import llama_cpp  # noqa: PLC0415
    except ImportError:
        return False

    if getattr(llama_cpp.Llama, "_aegivis_patched", False):
        return True

    _orig_call        = llama_cpp.Llama.__call__
    _orig_chat        = llama_cpp.Llama.create_chat_completion

    def _patched_call(self: Any, prompt: str, **kwargs: Any) -> Any:
        sid = _session_id()
        aid = _agent_id()
        model_name = getattr(self, "model_path", "llama-cpp").split("/")[-1]
        payload = {
            "provider":       "llamacpp",
            "model":          model_name,
            "prompt_preview": str(prompt)[:500],
        }
        for k in ("max_tokens", "temperature", "top_p", "stop"):
            if k in kwargs:
                payload[k] = kwargs[k]
        _fire("LLM_CALL_START", payload, sid, aid)
        t0 = time.monotonic()

        try:
            result = _orig_call(self, prompt, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": "llamacpp", "model": model_name, "error": str(exc)[:300]},
                  sid, aid)
            raise

        text = ""
        choices = result.get("choices") or []
        if choices:
            text = choices[0].get("text", "")
        usage = result.get("usage") or {}

        _fire("LLM_CALL_END", {
            "provider":         "llamacpp",
            "model":            model_name,
            "response_preview": text[:500],
            "input_tokens":     usage.get("prompt_tokens"),
            "output_tokens":    usage.get("completion_tokens"),
            "latency_ms":       round((time.monotonic() - t0) * 1000, 1),
        }, sid, aid)
        return result

    def _patched_chat(self: Any, messages: list, **kwargs: Any) -> Any:
        sid = _session_id()
        aid = _agent_id()
        model_name = getattr(self, "model_path", "llama-cpp").split("/")[-1]

        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        payload = {
            "provider":             "llamacpp",
            "model":                model_name,
            "user_message_preview": str(last_user)[:500],
            "message_count":        len(messages),
        }
        _fire("LLM_CALL_START", payload, sid, aid)
        t0 = time.monotonic()

        try:
            result = _orig_chat(self, messages, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": "llamacpp", "model": model_name, "error": str(exc)[:300]},
                  sid, aid)
            raise

        choices = result.get("choices") or []
        text = (choices[0].get("message") or {}).get("content", "") if choices else ""
        usage = result.get("usage") or {}

        _fire("LLM_CALL_END", {
            "provider":         "llamacpp",
            "model":            model_name,
            "response_preview": text[:500],
            "input_tokens":     usage.get("prompt_tokens"),
            "output_tokens":    usage.get("completion_tokens"),
            "latency_ms":       round((time.monotonic() - t0) * 1000, 1),
        }, sid, aid)
        return result

    llama_cpp.Llama.__call__                 = _patched_call  # type: ignore[method-assign]
    llama_cpp.Llama.create_chat_completion   = _patched_chat  # type: ignore[method-assign]
    llama_cpp.Llama._aegivis_patched         = True           # type: ignore[attr-defined]

    if _DEBUG:
        logger.debug("aegivis.intercept_local: llama_cpp patched")
    return True


# ---------------------------------------------------------------------------
# HuggingFace Transformers patch
# ---------------------------------------------------------------------------

def _install_transformers() -> bool:
    """
    Patch ``transformers.Pipeline.__call__``.

    Covers all pipeline types: text-generation, text2text-generation,
    question-answering, summarization, translation, etc.
    Token counts are not available without a separate tokenizer call so
    we record the prompt/response preview and task name only.
    """
    try:
        from transformers.pipelines.base import Pipeline  # noqa: PLC0415
    except ImportError:
        return False

    if getattr(Pipeline, "_aegivis_patched", False):
        return True

    _orig_call = Pipeline.__call__

    def _patched_pipeline_call(self: Any, inputs: Any, **kwargs: Any) -> Any:
        sid = _session_id()
        aid = _agent_id()

        task = getattr(self, "task", "unknown")
        model_name = getattr(getattr(self, "model", None), "name_or_path", "transformers")

        # Normalise inputs to a string preview
        if isinstance(inputs, str):
            prompt_preview = inputs[:500]
        elif isinstance(inputs, list):
            prompt_preview = str(inputs[0])[:500] if inputs else ""
        elif isinstance(inputs, dict):
            # question-answering: {"question": ..., "context": ...}
            prompt_preview = str(inputs.get("question") or inputs.get("inputs", ""))[:500]
        else:
            prompt_preview = str(inputs)[:500]

        payload = {
            "provider":       "transformers",
            "model":          model_name,
            "task":           task,
            "prompt_preview": prompt_preview,
        }
        _fire("LLM_CALL_START", payload, sid, aid)
        t0 = time.monotonic()

        try:
            result = _orig_call(self, inputs, **kwargs)
        except Exception as exc:
            _fire("LLM_CALL_ERROR",
                  {"provider": "transformers", "model": model_name,
                   "task": task, "error": str(exc)[:300]},
                  sid, aid)
            raise

        # Normalise output to a text preview
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                text = (
                    first.get("generated_text")
                    or first.get("translation_text")
                    or first.get("summary_text")
                    or first.get("answer")
                    or str(first)
                )
            else:
                text = str(first)
        elif isinstance(result, dict):
            text = str(result.get("answer") or result.get("generated_text") or result)
        else:
            text = str(result)

        _fire("LLM_CALL_END", {
            "provider":         "transformers",
            "model":            model_name,
            "task":             task,
            "response_preview": text[:500],
            "latency_ms":       round((time.monotonic() - t0) * 1000, 1),
        }, sid, aid)
        return result

    Pipeline.__call__         = _patched_pipeline_call  # type: ignore[method-assign]
    Pipeline._aegivis_patched = True                    # type: ignore[attr-defined]

    if _DEBUG:
        logger.debug("aegivis.intercept_local: transformers.Pipeline patched")
    return True


# ---------------------------------------------------------------------------
# Auto-install on import
# ---------------------------------------------------------------------------

if _ENABLED:
    _install_httpx_local()
    _install_requests_local()
    _install_llamacpp()
    _install_transformers()
