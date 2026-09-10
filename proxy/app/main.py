"""
Aegivis LLM Proxy — FastAPI application.

Routes all LLM API traffic through the proxy for forensic capture.

Usage (set one env var per provider):
    export OPENAI_BASE_URL="http://localhost:8080/openai"
    export ANTHROPIC_BASE_URL="http://localhost:8080/anthropic"
    # Then run your agent normally
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Body, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agent_identity import validate_agent_key
from .config import settings
from .shadow_router import fire_shadow_comparison
from .intercept import InterceptContext
from .policy import PolicyAction, get_policy_engine, reload_policy_engine
from .tool_permissions import (
    get_tool_permissions_engine,
    reload_tool_permissions_engine,
)
from .providers import (
    AnthropicProvider,
    AzureProvider,
    CerebrasProvider,
    CohereCompatProvider,
    DeepSeekProvider,
    FireworksProvider,
    GoogleProvider,
    GroqProvider,
    MistralProvider,
    NvidiaProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider,
    SambanovaProvider,
    TogetherProvider,
    VertexProvider,
    XAIProvider,
)
from .providers.mcp import MCPProvider
from .session import get_session_tracker
from .transport import get_transport
from .canonicalize import make_embedding_call, make_media_call, make_a2a_event
from .hash_chain import compute_event_hash
from .models import EventType as _EventType

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

# Provider routing table
PROVIDERS = {
    "openai":     (OpenAIProvider,    settings.openai_upstream),
    "anthropic":  (AnthropicProvider, settings.anthropic_upstream),
    "google":     (GoogleProvider,    settings.google_upstream),
    "azure":      (AzureProvider,     settings.azure_upstream),
    "ollama":     (OllamaProvider,    settings.ollama_upstream),
    # OpenAI-compatible providers (Phase E1)
    "groq":       (GroqProvider,       settings.groq_upstream),
    "mistral":    (MistralProvider,    settings.mistral_upstream),
    "together":   (TogetherProvider,   settings.together_upstream),
    "perplexity": (PerplexityProvider, settings.perplexity_upstream),
    "deepseek":   (DeepSeekProvider,   settings.deepseek_upstream),
    "xai":        (XAIProvider,        settings.xai_upstream),
    "fireworks":  (FireworksProvider,  settings.fireworks_upstream),
    "openrouter": (OpenRouterProvider, settings.openrouter_upstream),
    "cerebras":   (CerebrasProvider,   settings.cerebras_upstream),
    "sambanova":  (SambanovaProvider,  settings.sambanova_upstream),
    "nvidia":     (NvidiaProvider,     settings.nvidia_upstream),
    "cohere":     (CohereCompatProvider, settings.cohere_upstream),
    # Vertex AI uses dynamic upstream from path — listed here for /health visibility
    "vertex":     (VertexProvider, "dynamic"),
}

# Headers that must not be forwarded upstream (proxy-internal or computed)
# content-length is excluded: httpx recomputes it from the actual body bytes,
# which may differ from the original if spotlighting/canary modified the body.
_HOP_BY_HOP = frozenset({
    "host", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
    "content-length",
    "x-aegivis-session-id", "x-aegivis-agent-id", "x-aegivis-org-id",
    "x-aegivis-parent-agent-id", "x-aegivis-parent-session-id",
})

# Paths that trigger LLM interception (intercept these, passthrough everything else)
_OPENAI_COMPAT_PATHS = {"/v1/chat/completions", "/v1/completions"}

_INTERCEPT_PATHS = {
    "openai":     {"/v1/chat/completions", "/v1/completions"},
    "anthropic":  {"/v1/messages"},
    "google":     set(),   # matched by regex in route handler
    "azure":      {"/openai/deployments"},   # prefix match
    "ollama":     {"/v1/chat/completions", "/api/chat"},
    # OpenAI-compatible providers (Phase E1)
    "groq":       _OPENAI_COMPAT_PATHS,
    "mistral":    _OPENAI_COMPAT_PATHS,
    "together":   _OPENAI_COMPAT_PATHS,
    "perplexity": _OPENAI_COMPAT_PATHS,
    "deepseek":   _OPENAI_COMPAT_PATHS,
    "xai":        _OPENAI_COMPAT_PATHS,
    "fireworks":  _OPENAI_COMPAT_PATHS,
    "openrouter": _OPENAI_COMPAT_PATHS,
    "cerebras":   _OPENAI_COMPAT_PATHS,
    "sambanova":  _OPENAI_COMPAT_PATHS,
    "nvidia":     _OPENAI_COMPAT_PATHS,
    "cohere":     _OPENAI_COMPAT_PATHS,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Restore session state from disk (survives proxy restarts)
    tracker = get_session_tracker()
    tracker.load()

    transport = get_transport()
    await transport.start()

    # Load policy engine
    if settings.policy_yaml:
        reload_policy_engine(yaml_path=Path(settings.policy_yaml))
    else:
        engine = get_policy_engine()
        logger.info(f"Policy engine loaded: {engine.rule_count} rules")

    # Load tool permissions engine
    if settings.tool_permissions_yaml:
        reload_tool_permissions_engine(yaml_path=Path(settings.tool_permissions_yaml))
    else:
        tp_engine = get_tool_permissions_engine()
        logger.info(
            f"Tool permissions engine loaded: {tp_engine.rule_count} rules "
            f"({tp_engine.enabled_rule_count} enabled)"
        )

    # Warn if capability manifest signing key is not set (Phase 12)
    if not settings.manifest_signing_key:
        logger.warning(
            "AEGIVIS_MANIFEST_SIGNING_KEY is not set — capability manifest signatures "
            "will not be verified. Set to a 32+ char secret shared with the backend "
            "for tamper-proof manifest enforcement."
        )

    # Setup OpenTelemetry tracing (Phase 3.4)
    if settings.otel_enabled:
        from .tracing import setup_tracing
        ok = setup_tracing(
            service_name=settings.otel_service_name,
            endpoint=settings.otel_endpoint,
        )
        if not ok:
            logger.warning(
                "OTel tracing requested (AEGIVIS_OTEL_ENABLED=true) but packages missing. "
                "Install with: pip install 'aegivis-proxy[observability]'"
            )

    # Background stale-session cleanup every 30 minutes
    async def _evict_loop():
        while True:
            await asyncio.sleep(1800)
            get_session_tracker().evict_stale()

    evict_task = asyncio.create_task(_evict_loop(), name="aegivis-session-evict")

    # Warm the security config cache for the default org so the first burst of
    # requests after deployment never all hit the backend simultaneously.
    # Agent-level configs are warmed lazily on first request per agent.
    try:
        from .security.remote_config import get_security_config
        await get_security_config(settings.org_id)
        logger.info("Security config cache warmed for org=%s", settings.org_id)
    except Exception as _wexc:
        logger.debug(
            "Security config cache warm-up skipped (backend not yet ready): %s", _wexc
        )

    buf_status = transport.buffer_status()
    logger.info(
        f"Aegivis Proxy ready on {settings.host}:{settings.port} | "
        f"active sessions: {tracker.active_count()} | "
        f"buffered events: {buf_status.get('events', 0)}"
    )

    yield

    evict_task.cancel()
    # Persist session state before transport stops (so in-flight events get the right hashes)
    tracker.save()
    await transport.stop()
    logger.info("Aegivis Proxy shutdown complete")


app = FastAPI(
    title="Aegivis LLM Proxy",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

# Mount Prometheus /metrics endpoint (Phase 3.4)
# Optional: requires prometheus-client. No-op if not installed.
if settings.metrics_enabled:
    from .metrics import get_metrics_app
    _metrics_asgi = get_metrics_app()
    if _metrics_asgi is not None:
        app.mount("/metrics", _metrics_asgi)
        logger.info("Prometheus /metrics endpoint mounted")

# Allow the dashboard (and any local dev origin) to call the management API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _extract_headers(request: Request) -> dict[str, str]:
    """Forward headers upstream, stripping proxy-internal ones."""
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


def _get_aegivis_headers(request: Request) -> dict[str, str | None] | Response:
    """Extract Aegivis headers. Returns a Response if agent identity is invalid."""
    agent_key = request.headers.get("x-aegivis-agent-key")
    agent_id_header = request.headers.get("x-aegivis-agent-id", "unknown-agent")

    is_valid, resolved_agent_id, error = validate_agent_key(agent_key, agent_id_header)
    if not is_valid:
        return JSONResponse(
            status_code=401,
            content={"error": "invalid_agent_key", "detail": error},
        )

    return {
        "session_id":        request.headers.get("x-aegivis-session-id"),
        "agent_id":          resolved_agent_id,
        "org_id":            request.headers.get("x-aegivis-org-id", settings.org_id),
        "parent_agent_id":   request.headers.get("x-aegivis-parent-agent-id") or None,
        "parent_session_id": request.headers.get("x-aegivis-parent-session-id") or None,
    }


def _should_intercept(provider_name: str, path: str) -> bool:
    """Determine if this path should be intercepted vs passed through."""
    if provider_name == "google":
        return "generateContent" in path or "streamGenerateContent" in path
    if provider_name == "azure":
        return "chat/completions" in path
    return path in _INTERCEPT_PATHS.get(provider_name, set())


async def _proxy_and_capture(
    request: Request,
    provider_name: str,
    upstream_base: str,
    downstream_path: str,
    provider_cls,
) -> Response:
    """
    Core proxy handler: intercept, policy-check, forward, capture response.
    """
    abb = _get_aegivis_headers(request)
    # Agent identity check failed — return 401 directly
    if isinstance(abb, Response):
        return abb

    body_bytes = await request.body()

    # Reject oversized payloads before any processing
    limit = settings.max_request_body_bytes
    if limit > 0 and len(body_bytes) > limit:
        return JSONResponse(
            status_code=413,
            content={
                "error": "payload_too_large",
                "detail": f"Request body exceeds {limit} bytes limit.",
            },
        )

    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    is_stream = body.get("stream", False)
    model = body.get("model", "unknown")

    # Extract canonical request params
    if provider_name == "google":
        # Extract model from URL path for Google
        path_parts = downstream_path.split("/")
        model_idx = next((i for i, p in enumerate(path_parts) if p == "models"), None)
        if model_idx and model_idx + 1 < len(path_parts):
            model = path_parts[model_idx + 1].split(":")[0]
        request_params = provider_cls.extract_request_params(body, model=model)
    elif provider_name == "ollama":
        request_params = provider_cls.extract_request_params(body, path=downstream_path)
    else:
        request_params = provider_cls.extract_request_params(body)
        model = request_params.get("model", model)

    # Build interception context — use Redis transport if available, else HTTP
    from .transport import get_best_transport as _get_best_transport
    _transport = await _get_best_transport()
    context = InterceptContext(
        session_tracker=get_session_tracker(),
        org_id=abb["org_id"],
        transport=_transport,
    )

    # Process request (emit LLM_CALL_START + any pending TOOL_CALL_END)
    # Returns a 4-tuple: (session_id, run_id, violations, forward_body)
    # forward_body is non-None when canary injection / spotlighting modified
    # the messages that should be forwarded to the upstream LLM.
    session_id, run_id, violations, forward_body = await context.process_request(
        request_data=body,
        provider=provider_name,
        model=model,
        agent_id=abb["agent_id"],
        explicit_session_id=abb["session_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )

    # Check for BLOCK violations — return 403 without forwarding to LLM
    block_violations = [v for v in violations if v.action == PolicyAction.BLOCK]
    if block_violations:
        v = block_violations[0]
        return JSONResponse(
            status_code=403,
            content={
                "error": "policy_violation",
                "rule": v.rule_name,
                "reason": v.reason,
                "session_id": session_id,
            },
            headers={
                "X-Aegivis-Policy-Rule": v.rule_name,
                "X-Aegivis-Session-ID": session_id,
            },
        )

    # If canary injection / spotlighting modified the messages, re-serialize
    # the body so the modified version is forwarded to the upstream LLM.
    if forward_body is not None:
        body_bytes = json.dumps(forward_body).encode("utf-8")

    # Forward request upstream
    upstream_url = f"{upstream_base}{downstream_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    forward_headers = _extract_headers(request)
    t_start = time.time()

    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    try:
        if is_stream:
            return await _handle_streaming(
                client=client,
                upstream_url=upstream_url,
                forward_headers=forward_headers,
                body_bytes=body_bytes,
                method=request.method,
                context=context,
                session_id=session_id,
                run_id=run_id,
                provider_name=provider_name,
                model=model,
                agent_id=abb["agent_id"],
                provider_cls=provider_cls,
                t_start=t_start,
                downstream_path=downstream_path,
            )
        else:
            return await _handle_non_streaming(
                client=client,
                upstream_url=upstream_url,
                forward_headers=forward_headers,
                body_bytes=body_bytes,
                method=request.method,
                context=context,
                session_id=session_id,
                run_id=run_id,
                provider_name=provider_name,
                model=model,
                agent_id=abb["agent_id"],
                provider_cls=provider_cls,
                t_start=t_start,
                downstream_path=downstream_path,
            )
    finally:
        await client.aclose()


async def _handle_non_streaming(
    *,
    client,
    upstream_url,
    forward_headers,
    body_bytes,
    method,
    context,
    session_id,
    run_id,
    provider_name,
    model,
    agent_id,
    provider_cls,
    t_start,
    downstream_path,
) -> Response:
    resp = await client.request(
        method=method,
        url=upstream_url,
        content=body_bytes,
        headers=forward_headers,
    )
    latency_ms = (time.time() - t_start) * 1000

    try:
        resp_body = resp.json()
    except Exception:
        resp_body = {}

    if provider_name == "ollama":
        parsed_resp = provider_cls.parse_response(resp_body, path=downstream_path)
    else:
        parsed_resp = provider_cls.parse_response(resp_body)

    resp_violations = await context.process_response(
        session_id=session_id,
        run_id=run_id,
        provider=provider_name,
        model=model,
        agent_id=agent_id,
        response_data=parsed_resp,
        latency_ms=latency_ms,
        http_status=resp.status_code,
    )

    # Check for BLOCK violations raised during TOOL_CALL_START processing
    # (e.g. taint-tracking data-exfiltration-attempt). Return 403 so the agent
    # never receives the tool_calls and cannot execute the exfiltrating tool.
    resp_blocks = [v for v in resp_violations if v.action == PolicyAction.BLOCK]
    if resp_blocks:
        rv = resp_blocks[0]
        return JSONResponse(
            status_code=403,
            content={
                "error": "policy_violation",
                "rule": rv.rule_name,
                "reason": rv.reason,
                "session_id": session_id,
            },
            headers={
                "X-Aegivis-Policy-Rule": rv.rule_name,
                "X-Aegivis-Session-ID": session_id,
            },
        )

    # ── Phase 19: Response-Hold HITL (non-streaming only) ────────────────────
    # If a HITL approval was created during process_response() (blast_radius
    # HIGH+ or explicit hitl_required_tools) AND response_hold_enabled is True,
    # hold this response and poll until a human decides.
    #
    # Approved → release original response to agent (tool executes normally).
    # Denied   → return synthetic denial response with no tool_calls so the
    #             agent never attempts to execute the blocked action.
    #
    # This is the true pre-execution gate: the agent never receives tool_calls
    # that have not been reviewed.  No retroactive blocking needed.
    try:
        _should_deny, _denial_body = await context.maybe_hold_response(
            session_id=session_id,
            agent_id=agent_id,
            provider=provider_name,
            model=model,
        )
        if _should_deny and _denial_body is not None:
            return JSONResponse(
                status_code=200,   # 200 so the agent SDK parses it normally
                content=_denial_body,
                headers={
                    "X-Aegivis-Hold-Decision": "denied",
                    "X-Aegivis-Session-ID": session_id,
                },
            )
    except Exception as _hold_exc:
        logger.warning("[RESPONSE-HOLD] Error in hold gate (continuing): %s", _hold_exc)

    # ── Live Model Shadowing (fire-and-forget, zero hot-path latency) ──────────
    if (
        settings.shadow_enabled
        and settings.shadow_model
        and resp.status_code == 200
        and random.random() < settings.shadow_sample_rate
    ):
        asyncio.create_task(
            fire_shadow_comparison(
                org_id=context.org_id,
                session_id=session_id,
                agent_id=agent_id,
                provider=provider_name,
                upstream_url=str(upstream_url),
                forward_headers=dict(forward_headers),
                request_body_bytes=body_bytes,
                primary_model=model,
                primary_latency_ms=latency_ms,
                primary_tokens=parsed_resp.get("total_tokens", 0) or 0,
                primary_tool_calls=parsed_resp.get("tool_calls") or [],
                primary_response_len=len(parsed_resp.get("response_text") or ""),
            )
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def _handle_streaming(
    *,
    client,
    upstream_url,
    forward_headers,
    body_bytes,
    method,
    context,
    session_id,
    run_id,
    provider_name,
    model,
    agent_id,
    provider_cls,
    t_start,
    downstream_path,
) -> StreamingResponse:
    assembler = provider_cls.new_assembler()

    async def stream_gen():
        async with client.stream(
            method=method,
            url=upstream_url,
            content=body_bytes,
            headers=forward_headers,
        ) as resp:
            complete = False
            async for line in resp.aiter_lines():
                # Forward to agent immediately
                if line.startswith("data: "):
                    data_part = line[6:]
                    # Ollama needs the path to pick the right parser
                    # (OpenAI-compat /v1/ vs native /api/chat)
                    if provider_name == "ollama":
                        chunk = provider_cls.parse_sse_chunk(data_part, path=downstream_path)
                    else:
                        chunk = provider_cls.parse_sse_chunk(data_part)
                    is_done = assembler.feed(chunk)
                    if is_done:
                        complete = True
                elif (
                    provider_name == "ollama"
                    and "/v1/" not in downstream_path
                    and line.strip()
                ):
                    # Ollama native /api/chat streams raw JSON lines (no SSE prefix)
                    chunk = provider_cls.parse_sse_chunk(line, path=downstream_path)
                    is_done = assembler.feed(chunk)
                    if is_done:
                        complete = True
                yield f"{line}\n\n".encode()

            latency_ms = (time.time() - t_start) * 1000

            if complete:
                response_data = assembler.build_response()
            else:
                response_data = assembler.build_response()
                response_data["finish_reason"] = response_data.get("finish_reason") or "stop"

            # Fire-and-forget event emission
            asyncio.create_task(context.process_response(
                session_id=session_id,
                run_id=run_id,
                provider=provider_name,
                model=model,
                agent_id=agent_id,
                response_data=response_data,
                latency_ms=latency_ms,
                http_status=resp.status_code,
            ))

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Route handlers ────────────────────────────────────────────────────────────


async def _handle_image_gen(
    request: Request,
    provider_name: str,
    upstream_base: str,
) -> Response:
    """
    Phase E5 — Image generation route handler.

    Scans the prompt for injection signals (adversary-injected text via tool
    results) and PII (data-in-image exfiltration risk).  Forwards unconditionally;
    fires ALERT violations on detection.
    """
    from .security.image_prompt_guard import (  # noqa: PLC0415
        scan_image_prompt,
        extract_image_prompt,
    )
    from .transport import get_best_transport as _get_best  # noqa: PLC0415

    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    model = body.get("model", "dall-e-3")
    tracker = get_session_tracker()
    session_id = tracker.resolve_session(
        explicit_session_id=abb["session_id"],
        messages=[],
        agent_id=abb["agent_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )
    state = tracker.get_state(session_id)

    guard_result = None
    if settings.security_image_gen_enabled:
        prompt_text = extract_image_prompt(body)
        guard_result = scan_image_prompt(prompt_text)

    # Forward to upstream
    upstream_url = f"{upstream_base}/v1/images/generations"
    if request.url.query:
        upstream_url += f"?{request.url.query}"
    headers = _extract_headers(request)
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(upstream_url, content=body_bytes, headers=headers)
        latency_ms = (time.time() - t_start) * 1000
        response_bytes = resp.content
        http_status = resp.status_code
        content_type = resp.headers.get("content-type", "application/json")
    except Exception as exc:
        logger.warning("Image gen upstream error: %s", exc)
        return JSONResponse(status_code=502,
                            content={"error": "upstream_error", "detail": str(exc)})

    if settings.security_image_gen_enabled and guard_result is not None:
        seq = state.sequence_number
        prev_hash = state.last_hash
        event = make_media_call(
            event_type=_EventType.IMAGE_GEN_CALL,
            session_id=session_id,
            org_id=abb["org_id"],
            agent_id=abb["agent_id"],
            provider=provider_name,
            model=model,
            payload={
                "prompt_chars":       guard_result.prompt_chars,
                "injection_detected": guard_result.injection_detected,
                "injection_score":    round(guard_result.injection_score, 4),
            },
            pii_detected=guard_result.pii_types,
            http_status=http_status,
            latency_ms=latency_ms,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        event["previous_hash"] = prev_hash
        event["current_hash"] = compute_event_hash(event)
        state.sequence_number += 1
        state.last_hash = event["current_hash"]

        transport = await _get_best()
        transport.enqueue(event)

        if guard_result.injection_detected:
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "image-prompt-injection",
                "action": "ALERT",
                "reason": (
                    f"Injection score {guard_result.injection_score:.2f} in image "
                    f"prompt — possible adversarial injection via tool result."
                ),
                "severity": "HIGH", "event_type": "IMAGE_GEN_CALL", "model": model,
                "provider": provider_name,
            })
            logger.warning(
                "Image prompt injection [session=%s score=%.2f]",
                session_id, guard_result.injection_score,
            )

        if guard_result.pii_detected:
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "image-prompt-injection",
                "action": "ALERT",
                "reason": (
                    f"PII ({', '.join(guard_result.pii_types)}) in image prompt "
                    f"— may be rendered as text in the generated image."
                ),
                "severity": "HIGH", "event_type": "IMAGE_GEN_CALL", "model": model,
                "provider": provider_name,
            })

    return Response(
        content=response_bytes,
        status_code=http_status,
        media_type=content_type,
        headers={"X-Aegivis-Session-ID": session_id},
    )


async def _handle_audio(
    request: Request,
    provider_name: str,
    upstream_base: str,
    direction: str,  # "tts" or "stt"
) -> Response:
    """
    Phase E5 — Audio route handler (TTS and STT).

    TTS (/v1/audio/speech): scan input text for PII before vocalisation.
    STT (/v1/audio/transcriptions): forward multipart, scan response text
      for PII and injection signals (injected audio attack surface).
    """
    from .security.image_prompt_guard import (  # noqa: PLC0415
        scan_audio_input,
        scan_image_prompt,  # reused for STT output injection scan
    )
    from .transport import get_best_transport as _get_best  # noqa: PLC0415

    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    tracker = get_session_tracker()
    session_id = tracker.resolve_session(
        explicit_session_id=abb["session_id"],
        messages=[],
        agent_id=abb["agent_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )
    state = tracker.get_state(session_id)

    body_bytes = await request.body()
    model = "whisper-1" if direction == "stt" else "tts-1"

    # TTS: scan request body text for PII before forwarding
    tts_guard = None
    if direction == "tts" and settings.security_audio_enabled:
        try:
            body = json.loads(body_bytes) if body_bytes else {}
            model = body.get("model", "tts-1")
            input_text = body.get("input", "")
            tts_guard = scan_audio_input(str(input_text))
        except Exception:
            pass

    # Forward to upstream
    endpoint = "/v1/audio/speech" if direction == "tts" else "/v1/audio/transcriptions"
    upstream_url = f"{upstream_base}{endpoint}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"
    headers = _extract_headers(request)
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(upstream_url, content=body_bytes, headers=headers)
        latency_ms = (time.time() - t_start) * 1000
        response_bytes = resp.content
        http_status = resp.status_code
        content_type = resp.headers.get("content-type", "application/json")
    except Exception as exc:
        logger.warning("Audio upstream error (%s): %s", direction, exc)
        return JSONResponse(status_code=502,
                            content={"error": "upstream_error", "detail": str(exc)})

    # STT: scan transcription text in the response
    stt_injection_guard = None
    stt_pii_guard = None
    if direction == "stt" and settings.security_audio_enabled and http_status == 200:
        try:
            resp_body = json.loads(response_bytes)
            transcript = resp_body.get("text", "")
            if transcript:
                stt_injection_guard = scan_image_prompt(transcript)  # injection scan
                stt_pii_guard = scan_audio_input(transcript)          # PII scan
        except Exception:
            pass

    if settings.security_audio_enabled:
        pii_types: list[str] = []
        if tts_guard and tts_guard.pii_detected:
            pii_types = tts_guard.pii_types
        elif stt_pii_guard and stt_pii_guard.pii_detected:
            pii_types = stt_pii_guard.pii_types

        payload: dict = {"direction": direction}
        if tts_guard:
            payload["input_chars"] = tts_guard.input_chars
        if stt_injection_guard:
            payload["transcript_injection_score"] = round(
                stt_injection_guard.injection_score, 4
            )

        seq = state.sequence_number
        prev_hash = state.last_hash
        event = make_media_call(
            event_type=_EventType.AUDIO_CALL,
            session_id=session_id,
            org_id=abb["org_id"],
            agent_id=abb["agent_id"],
            provider=provider_name,
            model=model,
            payload=payload,
            pii_detected=pii_types,
            http_status=http_status,
            latency_ms=latency_ms,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        event["previous_hash"] = prev_hash
        event["current_hash"] = compute_event_hash(event)
        state.sequence_number += 1
        state.last_hash = event["current_hash"]

        transport = await _get_best()
        transport.enqueue(event)

        if pii_types:
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "audio-pii-detected",
                "action": "ALERT",
                "reason": (
                    f"PII ({', '.join(pii_types)}) detected in audio "
                    f"{'input' if direction == 'tts' else 'transcription'}."
                ),
                "severity": "HIGH", "event_type": "AUDIO_CALL", "model": model,
                "provider": provider_name,
            })

        if stt_injection_guard and stt_injection_guard.injection_detected:
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "image-prompt-injection",
                "action": "ALERT",
                "reason": (
                    f"Injection signals in STT transcription "
                    f"(score={stt_injection_guard.injection_score:.2f}) — "
                    f"possible injected-audio attack."
                ),
                "severity": "HIGH", "event_type": "AUDIO_CALL", "model": model,
                "provider": provider_name,
            })

    return Response(
        content=response_bytes,
        status_code=http_status,
        media_type=content_type,
        headers={"X-Aegivis-Session-ID": session_id},
    )


async def _handle_embeddings(
    request: Request,
    provider_name: str,
    upstream_base: str,
) -> Response:
    """
    Phase E4 — Shared embedding route handler.

    Intercepts /v1/embeddings calls for PII detection and audit logging.
    Unlike LLM calls, embeddings are not blocked by default — the handler
    fires an ALERT violation and logs an EMBEDDING_CALL event, but forwards
    the request to upstream unconditionally (unless the proxy is in block mode).

    Supported providers: openai, azure, cohere (OpenAI-compat /v1/embeddings).
    Google Vertex embedContent uses a separate path and is handled by vertex_proxy.
    """
    from .security.embedding_guard import (  # noqa: PLC0415
        scan_embedding_input,
        extract_embedding_texts,
    )
    from .transport import get_best_transport as _get_best  # noqa: PLC0415

    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    model = body.get("model", "text-embedding-ada-002")

    # ── Resolve session (lightweight; no LLM_CALL_START event) ────────────────
    tracker = get_session_tracker()
    session_id = tracker.resolve_session(
        explicit_session_id=abb["session_id"],
        messages=[],
        agent_id=abb["agent_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )
    state = tracker.get_state(session_id)

    # ── PII scan ───────────────────────────────────────────────────────────────
    pii_violation = None
    guard_result = None
    if settings.security_embedding_enabled:
        texts = extract_embedding_texts(body)
        guard_result = scan_embedding_input(texts)
        state.embedding_call_count += 1

    # ── Forward to upstream ────────────────────────────────────────────────────
    upstream_url = f"{upstream_base}/v1/embeddings"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    headers = _extract_headers(request)
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(upstream_url, content=body_bytes, headers=headers)
        latency_ms = (time.time() - t_start) * 1000
        response_bytes = resp.content
        http_status = resp.status_code
        content_type = resp.headers.get("content-type", "application/json")
    except Exception as exc:
        logger.warning("Embedding upstream error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_error", "detail": str(exc)},
        )

    # ── Emit EMBEDDING_CALL audit event ────────────────────────────────────────
    if settings.security_embedding_enabled and guard_result is not None:
        seq = state.sequence_number
        prev_hash = state.last_hash
        event = make_embedding_call(
            session_id=session_id,
            org_id=abb["org_id"],
            agent_id=abb["agent_id"],
            provider=provider_name,
            model=model,
            input_count=guard_result.input_count,
            total_chars=guard_result.total_chars,
            pii_detected=guard_result.pii_types,
            http_status=http_status,
            latency_ms=latency_ms,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        event["previous_hash"] = prev_hash
        event["current_hash"] = compute_event_hash(event)
        state.sequence_number += 1
        state.last_hash = event["current_hash"]

        transport = await _get_best()
        transport.enqueue(event)

        # ── Fire PII violation if detected ─────────────────────────────────
        if guard_result.pii_detected:
            from .policy import PolicyViolation, PolicyAction  # noqa: PLC0415
            severity = "CRITICAL" if guard_result.critical_pii else "HIGH"
            pii_types_str = ", ".join(guard_result.pii_types)
            violation_event = {
                "session_id": session_id,
                "agent_id":   abb["agent_id"],
                "org_id":     abb["org_id"],
                "rule_name":  "embedding-pii-detected",
                "action":     "ALERT",
                "reason":     (
                    f"PII detected in embedding input ({pii_types_str}). "
                    f"Sensitive data may be persisted in the vector store."
                ),
                "severity":   severity,
                "event_type": "EMBEDDING_CALL",
                "model":      model,
                "provider":   provider_name,
            }
            transport.enqueue_violation(violation_event)
            logger.warning(
                "Embedding PII detected [session=%s agent=%s types=%s critical=%s]",
                session_id, abb["agent_id"], pii_types_str, guard_result.critical_pii,
            )

        # ── Volume anomaly alert ────────────────────────────────────────────
        threshold = settings.security_embedding_volume_threshold
        if state.embedding_call_count >= threshold:
            logger.warning(
                "Embedding volume spike [session=%s count=%d threshold=%d]",
                session_id, state.embedding_call_count, threshold,
            )

    return Response(
        content=response_bytes,
        status_code=http_status,
        media_type=content_type,
        headers={"X-Aegivis-Session-ID": session_id},
    )


async def _handle_a2a(request: Request) -> Response:
    """
    Phase E7 — A2A (Agent-to-Agent) Protocol proxy handler.

    Intercepts Google A2A JSON-RPC 2.0 messages exchanged between autonomous
    agents.  The calling agent sets ``X-A2A-Agent-URL`` to the actual target
    agent endpoint; this proxy sits transparently in the middle.

    Security checks performed on every ``tasks/send`` call:
    - Structural prompt injection scan (delimiter anomaly + Unicode stego).
    - PII detection (presidio-first; email + IBAN regex fallback).
    - Response artifact text scanned for PII leakage.

    BLOCK mode: enable ``AEGIVIS_SECURITY_A2A_BLOCK_INJECTION=true``.
    Default: ALERT only (observe + audit without breaking delegation pipelines).
    """
    from .security.a2a_scanner import scan_a2a_request, scan_a2a_response  # noqa: PLC0415
    from .transport import get_best_transport as _get_best                  # noqa: PLC0415

    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    # The target A2A agent URL must be supplied by the calling agent.
    target_url = request.headers.get("x-a2a-agent-url", "").strip()
    if not target_url:
        return JSONResponse(
            status_code=400,
            content={
                "error": "missing_target",
                "detail": (
                    "Set X-A2A-Agent-URL header to the downstream A2A agent endpoint. "
                    "Example: X-A2A-Agent-URL: http://my-agent.internal:8000"
                ),
            },
        )

    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    # ── Resolve session ────────────────────────────────────────────────────
    tracker = get_session_tracker()
    session_id = tracker.resolve_session(
        explicit_session_id=abb["session_id"],
        messages=[],
        agent_id=abb["agent_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )
    state = tracker.get_state(session_id)

    # ── Security scan (outgoing message) ──────────────────────────────────
    scan = None
    if settings.security_a2a_enabled:
        scan = scan_a2a_request(body)

    # BLOCK mode: reject if injection detected and block mode is on
    if (
        scan is not None
        and scan.injection_triggered
        and settings.security_a2a_block_injection
    ):
        return JSONResponse(
            status_code=403,
            content={
                "error": "policy_violation",
                "rule": "a2a-injection-detected",
                "reason": (
                    f"Prompt injection signals (score={scan.injection_score:.2f}) "
                    f"detected in A2A message to {target_url}."
                ),
                "session_id": session_id,
            },
            headers={
                "X-Aegivis-Policy-Rule": "a2a-injection-detected",
                "X-Aegivis-Session-ID": session_id,
            },
        )

    # ── Forward to target agent ────────────────────────────────────────────
    forward_headers = _extract_headers(request)
    # Strip our internal A2A header before forwarding to the real agent
    forward_headers.pop("x-a2a-agent-url", None)

    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                target_url,
                content=body_bytes,
                headers=forward_headers,
            )
        latency_ms = (time.time() - t_start) * 1000
        response_bytes = resp.content
        http_status = resp.status_code
        content_type = resp.headers.get("content-type", "application/json")
    except Exception as exc:
        logger.warning("A2A upstream error [target=%s]: %s", target_url, exc)
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_error", "detail": str(exc)},
        )

    # ── Security scan (response artifacts) ────────────────────────────────
    resp_scan = None
    if settings.security_a2a_enabled and http_status == 200:
        try:
            resp_body = json.loads(response_bytes)
            resp_scan = scan_a2a_response(resp_body)
        except Exception:
            pass

    # ── Emit outgoing A2A audit event ──────────────────────────────────────
    if settings.security_a2a_enabled and scan is not None:
        transport = await _get_best()

        seq = state.sequence_number
        prev_hash = state.last_hash
        send_event = make_a2a_event(
            event_type=_EventType.A2A_MESSAGE_SEND,
            session_id=session_id,
            org_id=abb["org_id"],
            agent_id=abb["agent_id"],
            direction="send",
            method=scan.method,
            task_id=scan.task_id,
            target_agent_url=target_url,
            text_parts=scan.text_parts,
            injection_score=scan.injection_score,
            injection_triggered=scan.injection_triggered,
            pii_detected=scan.pii_detected,
            http_status=http_status,
            latency_ms=latency_ms,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        send_event["current_hash"] = compute_event_hash(send_event)
        state.sequence_number += 1
        state.last_hash = send_event["current_hash"]
        transport.enqueue(send_event)

        if scan.injection_triggered:
            severity = "CRITICAL" if scan.injection_score >= 0.80 else "HIGH"
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "a2a-injection-detected",
                "action": "BLOCK" if settings.security_a2a_block_injection else "ALERT",
                "reason": (
                    f"Prompt injection (score={scan.injection_score:.2f}) in A2A "
                    f"message to {target_url} — possible cross-agent injection pivot."
                ),
                "severity": severity, "event_type": "A2A_MESSAGE_SEND",
                "model": "a2a-protocol", "provider": "a2a",
            })
            logger.warning(
                "A2A injection detected [session=%s score=%.2f target=%s]",
                session_id, scan.injection_score, target_url,
            )

        if scan.pii_detected:
            sev = "CRITICAL" if scan.critical_pii else "HIGH"
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "a2a-pii-detected",
                "action": "ALERT",
                "reason": (
                    f"PII ({', '.join(scan.pii_detected)}) in A2A message to "
                    f"{target_url} — sensitive data crossing agent trust boundary."
                ),
                "severity": sev, "event_type": "A2A_MESSAGE_SEND",
                "model": "a2a-protocol", "provider": "a2a",
            })

    # ── Emit response A2A audit event ──────────────────────────────────────
    if settings.security_a2a_enabled and resp_scan is not None:
        transport = await _get_best()

        seq = state.sequence_number
        prev_hash = state.last_hash
        recv_event = make_a2a_event(
            event_type=_EventType.A2A_MESSAGE_RECEIVE,
            session_id=session_id,
            org_id=abb["org_id"],
            agent_id=abb["agent_id"],
            direction="receive",
            method=body.get("method", ""),
            task_id=resp_scan.task_id,
            target_agent_url=target_url,
            artifact_texts=resp_scan.artifact_texts,
            pii_detected=resp_scan.pii_detected,
            http_status=http_status,
            latency_ms=latency_ms,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        recv_event["current_hash"] = compute_event_hash(recv_event)
        state.sequence_number += 1
        state.last_hash = recv_event["current_hash"]
        transport.enqueue(recv_event)

        if resp_scan.pii_detected:
            sev = "CRITICAL" if resp_scan.critical_pii else "HIGH"
            transport.enqueue_violation({
                "session_id": session_id, "agent_id": abb["agent_id"],
                "org_id": abb["org_id"], "rule_name": "a2a-pii-detected",
                "action": "ALERT",
                "reason": (
                    f"PII ({', '.join(resp_scan.pii_detected)}) in A2A response "
                    f"from {target_url} — sensitive data received from peer agent."
                ),
                "severity": sev, "event_type": "A2A_MESSAGE_RECEIVE",
                "model": "a2a-protocol", "provider": "a2a",
            })

    return Response(
        content=response_bytes,
        status_code=http_status,
        media_type=content_type,
        headers={"X-Aegivis-Session-ID": session_id},
    )


@app.api_route("/a2a", methods=["POST", "OPTIONS"])
async def a2a_proxy(request: Request):
    """
    Phase E7 — A2A Protocol proxy endpoint.

    Intercepts Google A2A JSON-RPC 2.0 inter-agent messages for security
    scanning and forensic audit logging.

    Set X-A2A-Agent-URL to the real target agent URL::

        POST http://localhost:8080/a2a
        X-A2A-Agent-URL: http://my-downstream-agent:8000
        Content-Type: application/json

        {"jsonrpc":"2.0","method":"tasks/send","params":{...}}
    """
    return await _handle_a2a(request)


@app.api_route("/openai/v1/embeddings", methods=["POST", "OPTIONS"])
async def openai_embeddings(request: Request):
    """Phase E4 — Intercept OpenAI embedding calls for PII scanning + audit."""
    return await _handle_embeddings(request, "openai", settings.openai_upstream)


@app.api_route("/openai/v1/images/generations", methods=["POST", "OPTIONS"])
async def openai_image_generations(request: Request):
    """Phase E5 — Intercept image generation calls for prompt injection + PII."""
    return await _handle_image_gen(request, "openai", settings.openai_upstream)


@app.api_route("/openai/v1/audio/speech", methods=["POST", "OPTIONS"])
async def openai_audio_speech(request: Request):
    """Phase E5 — Intercept TTS calls for PII in vocalized text."""
    return await _handle_audio(request, "openai", settings.openai_upstream, "tts")


@app.api_route("/openai/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
async def openai_audio_transcriptions(request: Request):
    """Phase E5 — Intercept STT calls; scan transcription result for PII + injection."""
    return await _handle_audio(request, "openai", settings.openai_upstream, "stt")


@app.api_route("/openai/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(path: str, request: Request):
    downstream = f"/{path}"
    if _should_intercept("openai", downstream):
        return await _proxy_and_capture(request, "openai", settings.openai_upstream, downstream, OpenAIProvider)
    # Passthrough
    return await _passthrough(request, settings.openai_upstream, downstream)


@app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def anthropic_proxy(path: str, request: Request):
    downstream = f"/{path}"
    if _should_intercept("anthropic", downstream):
        return await _proxy_and_capture(request, "anthropic", settings.anthropic_upstream, downstream, AnthropicProvider)
    return await _passthrough(request, settings.anthropic_upstream, downstream)


@app.api_route("/google/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def google_proxy(path: str, request: Request):
    downstream = f"/{path}"
    if _should_intercept("google", downstream):
        return await _proxy_and_capture(request, "google", settings.google_upstream, downstream, GoogleProvider)
    return await _passthrough(request, settings.google_upstream, downstream)


@app.api_route("/azure/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def azure_proxy(path: str, request: Request):
    downstream = f"/{path}"
    upstream = settings.azure_upstream
    if not upstream:
        raise HTTPException(status_code=503, detail="AEGIVIS_AZURE_UPSTREAM not configured")
    if _should_intercept("azure", downstream):
        return await _proxy_and_capture(request, "azure", upstream, downstream, AzureProvider)
    return await _passthrough(request, upstream, downstream)


@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def ollama_proxy(path: str, request: Request):
    downstream = f"/{path}"
    if _should_intercept("ollama", downstream):
        return await _proxy_and_capture(request, "ollama", settings.ollama_upstream, downstream, OllamaProvider)
    return await _passthrough(request, settings.ollama_upstream, downstream)


# ── OpenAI-compatible provider routes (Phase E1) ─────────────────────────────
# All of these share the same OpenAI Chat Completions request/response format.
# Routes are generated dynamically to avoid boilerplate.

def _make_compat_route(provider_name: str, upstream_setting: str):
    """Return a route handler closure for an OpenAI-compatible provider."""
    _methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    async def _handler(path: str, request: Request):
        upstream = getattr(settings, upstream_setting)
        downstream = f"/{path}"
        provider_cls = PROVIDERS[provider_name][0]
        if _should_intercept(provider_name, downstream):
            return await _proxy_and_capture(request, provider_name, upstream, downstream, provider_cls)
        return await _passthrough(request, upstream, downstream)

    _handler.__name__ = f"{provider_name}_proxy"
    return _handler


_COMPAT_ROUTES = [
    ("groq",       "groq_upstream"),
    ("mistral",    "mistral_upstream"),
    ("together",   "together_upstream"),
    ("perplexity", "perplexity_upstream"),
    ("deepseek",   "deepseek_upstream"),
    ("xai",        "xai_upstream"),
    ("fireworks",  "fireworks_upstream"),
    ("openrouter", "openrouter_upstream"),
    ("cerebras",   "cerebras_upstream"),
    ("sambanova",  "sambanova_upstream"),
    ("nvidia",     "nvidia_upstream"),
    ("cohere",     "cohere_upstream"),
]

for _pname, _uattr in _COMPAT_ROUTES:
    app.add_api_route(
        f"/{_pname}/{{path:path}}",
        _make_compat_route(_pname, _uattr),
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )


# ── AWS Bedrock route (Phase E2) ──────────────────────────────────────────────

import re as _re
_BEDROCK_PATH_RE = _re.compile(
    r"model/(.+?)/(converse-stream|converse|invoke-with-response-stream|invoke)$"
)


@app.api_route("/bedrock/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def bedrock_proxy(path: str, request: Request):
    """
    AWS Bedrock proxy route.

    Accepts standard Bedrock HTTP requests from boto3 (set
    AWS_ENDPOINT_URL_BEDROCK=http://localhost:8080/bedrock).

    Incoming SigV4 Authorization headers are ignored — the proxy authenticates
    to Bedrock using its own AWS credentials (see bedrock_gateway.py).

    Supports:
      POST /bedrock/model/{modelId}/converse
      POST /bedrock/model/{modelId}/converse-stream   (buffered → JSON response)
      POST /bedrock/model/{modelId}/invoke
      POST /bedrock/model/{modelId}/invoke-with-response-stream (buffered)
    """
    from .providers.bedrock import (
        BedrockProvider,
        extract_converse_request,
        parse_converse_response,
        parse_invoke_request,
        parse_invoke_response,
        detect_model_family,
    )
    from .bedrock_gateway import (
        forward_converse,
        forward_converse_stream,
        forward_invoke_model,
        forward_invoke_model_stream,
    )

    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    body_bytes = await request.body()

    limit = settings.max_request_body_bytes
    if limit > 0 and len(body_bytes) > limit:
        return JSONResponse(
            status_code=413,
            content={"error": "payload_too_large",
                     "detail": f"Request body exceeds {limit} bytes limit."},
        )

    # Parse path: "model/{modelId}/{api_type}"
    match = _BEDROCK_PATH_RE.search(path)
    if not match:
        # Passthrough for non-inference paths (e.g. list-foundation-models)
        return JSONResponse(
            status_code=404,
            content={"error": "unsupported_bedrock_path",
                     "detail": f"Aegivis Bedrock proxy only supports model inference paths. Got: /{path}"},
        )

    model_id = match.group(1)
    api_type = match.group(2)   # converse | converse-stream | invoke | invoke-with-response-stream

    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    # Extract canonical params for security scanning
    is_invoke = api_type.startswith("invoke")
    if is_invoke:
        request_params = parse_invoke_request(model_id, body)
    else:
        request_params = extract_converse_request(model_id, body)

    model = model_id

    # Build intercept context
    from .transport import get_best_transport as _get_best_transport
    _transport = await _get_best_transport()
    context = InterceptContext(
        session_tracker=get_session_tracker(),
        org_id=abb["org_id"],
        transport=_transport,
    )

    # Synthesize a request dict that process_request understands
    # (it mirrors the provider's extract_request_params output)
    session_id, run_id, violations, forward_body = await context.process_request(
        request_data=request_params,
        provider="bedrock",
        model=model,
        agent_id=abb["agent_id"],
        explicit_session_id=abb["session_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )

    # Block check
    block_violations = [v for v in violations if v.action == PolicyAction.BLOCK]
    if block_violations:
        v = block_violations[0]
        return JSONResponse(
            status_code=403,
            content={
                "error":      "policy_violation",
                "rule":       v.rule_name,
                "reason":     v.reason,
                "session_id": session_id,
            },
            headers={
                "X-Aegivis-Policy-Rule": v.rule_name,
                "X-Aegivis-Session-ID":  session_id,
            },
        )

    # If proxy modified the body (canary / spotlighting), rebuild Bedrock format
    if forward_body is not None and not is_invoke:
        from .providers.bedrock import rebuild_converse_body
        body = rebuild_converse_body(forward_body, body)
        body_bytes = json.dumps(body).encode("utf-8")

    t_start = time.time()

    try:
        if api_type == "converse":
            raw_response = await forward_converse(model_id, body)
            parsed_resp  = parse_converse_response(raw_response)
            response_bytes = json.dumps(raw_response).encode("utf-8")
            content_type   = "application/json"

        elif api_type == "converse-stream":
            # Buffer stream → return as standard Converse JSON
            raw_response = await forward_converse_stream(model_id, body)
            parsed_resp  = parse_converse_response(raw_response)
            response_bytes = json.dumps(raw_response).encode("utf-8")
            content_type   = "application/json"

        elif api_type == "invoke":
            response_bytes = await forward_invoke_model(model_id, body_bytes)
            raw_resp_body  = json.loads(response_bytes) if response_bytes else {}
            parsed_resp    = parse_invoke_response(model_id, raw_resp_body)
            content_type   = "application/json"

        else:  # invoke-with-response-stream
            response_bytes = await forward_invoke_model_stream(model_id, body_bytes)
            raw_resp_body  = json.loads(response_bytes) if response_bytes else {}
            parsed_resp    = parse_invoke_response(model_id, raw_resp_body)
            content_type   = "application/json"

    except RuntimeError as exc:
        logger.error("Bedrock gateway error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": "bedrock_gateway_error", "detail": str(exc)},
        )

    latency_ms = (time.time() - t_start) * 1000

    await context.process_response(
        session_id=session_id,
        run_id=run_id,
        provider="bedrock",
        model=model,
        agent_id=abb["agent_id"],
        response_data=parsed_resp,
        latency_ms=latency_ms,
        http_status=200,
    )

    return Response(
        content=response_bytes,
        status_code=200,
        media_type=content_type,
        headers={"X-Aegivis-Session-ID": session_id},
    )


@app.api_route("/vertex/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def vertex_proxy(path: str, request: Request):
    """
    Google Vertex AI proxy (Phase E3).

    Intercepts all Vertex AI generateContent/streamGenerateContent calls and
    applies the full Aegivis security stack (injection, PII, output scan, etc.).

    Auth: GCP OAuth2 Bearer token passes through unchanged — the proxy does NOT
    need its own GCP credentials.  The upstream URL is constructed dynamically
    from the {location} segment in the request path:

        /vertex/v1beta1/projects/{proj}/locations/{loc}/publishers/google/models/{model}:generateContent
        → https://{loc}-aiplatform.googleapis.com/v1beta1/projects/{proj}/...

    SDK configuration:
        # google-cloud-aiplatform
        vertexai.init(project="proj", location="us-central1",
                      api_endpoint="http://localhost:8080/vertex")

        # google-genai unified SDK (v1.0+)
        GOOGLE_GENAI_USE_VERTEXAI=1
        GOOGLE_CLOUD_PROJECT=my-project
        GOOGLE_CLOUD_LOCATION=us-central1
        GOOGLE_GENAI_API_ENDPOINT=http://localhost:8080/vertex
    """
    from .providers.vertex import (  # noqa: PLC0415
        VertexProvider as _VP,
        build_upstream_url,
        extract_model_from_path,
    )

    full_path = f"/{path}"

    # Only intercept generateContent calls; pass everything else through
    is_generate = "generateContent" in full_path or "streamGenerateContent" in full_path

    # Construct dynamic upstream from the location in the path
    upstream_base = build_upstream_url(full_path, default_location=settings.vertex_location)

    if not is_generate:
        return await _passthrough(request, upstream_base, full_path)

    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    body_bytes = await request.body()
    limit = settings.max_request_body_bytes
    if limit > 0 and len(body_bytes) > limit:
        return JSONResponse(
            status_code=413,
            content={"error": "payload_too_large",
                     "detail": f"Request body exceeds {limit} bytes limit."},
        )

    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    model = extract_model_from_path(full_path)
    request_params = _VP.extract_request_params(body, model=model)
    is_stream = "streamGenerateContent" in full_path

    from .transport import get_best_transport as _get_best_transport  # noqa: PLC0415
    transport = await _get_best_transport()
    context = InterceptContext(
        session_tracker=get_session_tracker(),
        transport=transport,
        org_id=abb["org_id"],
    )

    session_id, run_id, violations, _forward_body = await context.process_request(
        request_data=request_params,
        provider="vertex",
        model=model,
        agent_id=abb["agent_id"],
        explicit_session_id=abb["session_id"],
        parent_agent_id=abb["parent_agent_id"],
        parent_session_id=abb["parent_session_id"],
    )
    block_violations = [v for v in violations if v.action == PolicyAction.BLOCK]
    if block_violations:
        v = block_violations[0]
        return JSONResponse(
            status_code=403,
            content={"error": "policy_violation", "rule": v.rule_name,
                     "reason": v.reason, "session_id": session_id},
            headers={"X-Aegivis-Policy-Rule": v.rule_name,
                     "X-Aegivis-Session-ID": session_id},
        )

    headers = _extract_headers(request)
    upstream_url = f"{upstream_base}{full_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    t_start = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        if is_stream:
            # Streaming: assemble full response then return (same as Bedrock)
            assembler = _VP.new_assembler()
            async with client.stream("POST", upstream_url,
                                     content=body_bytes, headers=headers) as resp:
                if resp.status_code != 200:
                    err_bytes = await resp.aread()
                    latency_ms = (time.time() - t_start) * 1000
                    await context.process_response(
                        session_id=session_id, run_id=run_id, provider="vertex",
                        model=model, agent_id=abb["agent_id"],
                        response_data={"response_text": None, "finish_reason": "error",
                                       "tool_calls": [], "token_usage": None},
                        latency_ms=latency_ms, http_status=resp.status_code,
                    )
                    return Response(content=err_bytes, status_code=resp.status_code,
                                    media_type=resp.headers.get("content-type"))
                async for line in resp.aiter_lines():
                    chunk = _VP.parse_sse_chunk(line)
                    if chunk is not None and assembler.feed(chunk):
                        break
            parsed_resp = assembler.build_response()
            response_bytes = json.dumps({"candidates": [], "_aegivis_streamed": True}).encode()
            content_type = "application/json"
        else:
            resp = await client.post(upstream_url, content=body_bytes, headers=headers)
            response_bytes = resp.content
            content_type = resp.headers.get("content-type", "application/json")
            try:
                resp_body = json.loads(response_bytes)
            except json.JSONDecodeError:
                resp_body = {}
            parsed_resp = _VP.parse_response(resp_body)
            if resp.status_code != 200:
                latency_ms = (time.time() - t_start) * 1000
                await context.process_response(
                    session_id=session_id, run_id=run_id, provider="vertex",
                    model=model, agent_id=abb["agent_id"],
                    response_data={"response_text": None, "finish_reason": "error",
                                   "tool_calls": [], "token_usage": None},
                    latency_ms=latency_ms, http_status=resp.status_code,
                )
                return Response(content=response_bytes, status_code=resp.status_code,
                                media_type=content_type)

    latency_ms = (time.time() - t_start) * 1000
    await context.process_response(
        session_id=session_id, run_id=run_id, provider="vertex",
        model=model, agent_id=abb["agent_id"],
        response_data=parsed_resp, latency_ms=latency_ms, http_status=200,
    )

    return Response(
        content=response_bytes,
        status_code=200,
        media_type=content_type,
        headers={"X-Aegivis-Session-ID": session_id},
    )


@app.api_route("/mcp", methods=["GET", "POST", "OPTIONS"])
@app.api_route("/mcp/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def mcp_proxy(request: Request, path: str = ""):
    """
    MCP (Model Context Protocol) Streamable HTTP proxy.

    Intercepts ``tools/call`` requests and applies Aegivis security policy
    before forwarding to the real MCP server. All MCP interactions are logged
    to the audit trail with session correlation.

    Upstream MCP server resolved from (in priority order):
        1. X-Aegivis-Mcp-Server  request header
        2. AEGIVIS_MCP_SERVER_URL environment variable
    """
    import os as _os  # noqa: PLC0415

    upstream = (
        request.headers.get("x-aegivis-mcp-server")
        or _os.environ.get("AEGIVIS_MCP_SERVER_URL", "")
    ).rstrip("/")

    if not upstream:
        return JSONResponse(
            status_code=503,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": "No MCP server configured. Set X-Aegivis-Mcp-Server header or AEGIVIS_MCP_SERVER_URL.",
                },
            },
        )

    body_bytes = await request.body()
    mcp_req    = MCPProvider.parse_request(body_bytes) if body_bytes else None

    # ── GET = SSE stream subscription — forward directly ──────────────────
    if request.method == "GET":
        return await _passthrough(request, upstream, f"/{path}" if path else "")

    # ── POST = JSON-RPC 2.0 message ───────────────────────────────────────
    abb = _get_aegivis_headers(request)
    if isinstance(abb, Response):
        return abb

    session_id = abb["session_id"] or f"mcp-{__import__('uuid').uuid4().hex[:12]}"
    agent_id   = abb["agent_id"]
    org_id     = abb["org_id"]

    # Log + policy-check tools/call; pass everything else through after logging.
    if mcp_req and mcp_req.method in MCPProvider.INTERCEPTED_METHODS:
        audit_payload = MCPProvider.extract_audit_payload(mcp_req)
        audit_payload["session_id"] = session_id
        audit_payload["agent_id"]   = agent_id

        # Apply tool policy for tools/call
        if mcp_req.method == "tools/call" and mcp_req.tool_name:
            tp_engine = get_tool_permissions_engine()
            allowed, rule = tp_engine.check(mcp_req.tool_name, agent_id=agent_id, org_id=org_id)
            if not allowed:
                logger.info(
                    "MCP tools/call BLOCKED: tool=%s agent=%s rule=%s",
                    mcp_req.tool_name, agent_id, rule,
                )
                return JSONResponse(
                    status_code=403,
                    content=MCPProvider.build_policy_block_response(
                        request_id=mcp_req.id,
                        rule_name=rule or "tool-denied",
                        reason=f"Tool '{mcp_req.tool_name}' is not permitted for agent '{agent_id}'",
                    ),
                )

    # Forward to upstream MCP server
    target_path = f"/{path}" if path else ""
    forward_url = f"{upstream}{target_path}"
    if request.url.query:
        forward_url += f"?{request.url.query}"

    forward_headers = _extract_headers(request)
    # Preserve MCP session continuity header
    if mcp_session := request.headers.get("mcp-session-id"):
        forward_headers["mcp-session-id"] = mcp_session

    accept = request.headers.get("accept", "application/json")
    forward_headers["accept"] = accept

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=forward_url,
                content=body_bytes,
                headers=forward_headers,
            )
        except Exception as exc:
            logger.warning("MCP upstream error: %s", exc)
            return JSONResponse(
                status_code=502,
                content={
                    "jsonrpc": "2.0",
                    "id": mcp_req.id if mcp_req else None,
                    "error": {"code": -32603, "message": f"MCP upstream unreachable: {exc}"},
                },
            )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type", "application/json"),
    )


async def _passthrough(request: Request, upstream_base: str, path: str) -> Response:
    """Forward request to upstream without interception."""
    url = f"{upstream_base}{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = _extract_headers(request)
    body_bytes = await request.body()

    limit = settings.max_request_body_bytes
    if limit > 0 and len(body_bytes) > limit:
        return JSONResponse(
            status_code=413,
            content={
                "error": "payload_too_large",
                "detail": f"Request body exceeds {limit} bytes limit.",
            },
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            content=body_bytes,
            headers=headers,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


@app.get("/health")
async def health():
    engine    = get_policy_engine()
    tp_engine = get_tool_permissions_engine()
    tracker   = get_session_tracker()
    transport = get_transport()
    buf = transport.buffer_status()
    return {
        "status": "ok",
        "service": "aegivis-proxy",
        "version": "1.0.0",
        "providers": list(PROVIDERS.keys()),
        "policy_rules": engine.rule_count,
        "tool_permission_rules": tp_engine.rule_count,
        "tool_permission_rules_enabled": tp_engine.enabled_rule_count,
        "active_sessions": tracker.active_count(),
        "buffered_events": buf.get("events", 0),
        "buffered_violations": buf.get("violations", 0),
    }


# ─── Admin API key guard ────────────────────────────────────────────────────

def _require_admin_key(request: Request) -> JSONResponse | None:
    """
    Enforce the admin API key on management write endpoints.

    Returns a JSONResponse 403 if authentication fails, None if it passes.
    When AEGIVIS_ADMIN_API_KEY is empty (default), all callers are allowed
    (suitable for local development / Docker Compose setups with no exposure).
    """
    required_key = settings.admin_api_key.strip()
    if not required_key:
        return None  # admin auth disabled — open access

    provided_key = (
        request.headers.get("X-Aegivis-Admin-Key")
        or request.headers.get("x-aegivis-admin-key")
        or ""
    ).strip()

    if not provided_key or provided_key != required_key:
        return JSONResponse(
            status_code=403,
            content={
                "error": "admin_auth_required",
                "detail": (
                    "This endpoint requires a valid X-Aegivis-Admin-Key header. "
                    "Set AEGIVIS_ADMIN_API_KEY on the proxy to configure authentication."
                ),
            },
        )
    return None


# ─── Policy engine endpoints ────────────────────────────────────────────────

@app.get("/policies")
async def list_policies():
    """List all active policy rules (for dashboard/debugging)."""
    return {"rules": get_policy_engine().rules_summary()}


@app.post("/policies/reload")
async def reload_policies(request: Request):
    """
    Reload policy rules from JSON body or from default YAML file.

    Requires X-Aegivis-Admin-Key header when AEGIVIS_ADMIN_API_KEY is configured.
    Body (optional): {"rules": [...]}
    """
    auth_error = _require_admin_key(request)
    if auth_error:
        return auth_error

    try:
        body = await request.json()
        rules_data = body.get("rules")
    except Exception:
        rules_data = None

    if rules_data is not None:
        count = reload_policy_engine(rules_data=rules_data)
    elif settings.policy_yaml:
        count = reload_policy_engine(yaml_path=Path(settings.policy_yaml))
    else:
        count = reload_policy_engine()

    return {"status": "reloaded", "rule_count": count}


# ─── Tool permissions endpoints ─────────────────────────────────────────────

@app.get("/tool-permissions")
async def list_tool_permissions():
    """
    List all tool permission rules (enabled and disabled).

    This is the live view of what the ToolPermissionsEngine is currently
    evaluating.  All rules are shown; check the ``enabled`` field to see
    which ones are active.
    """
    tp = get_tool_permissions_engine()
    return {
        "rules": tp.rules_summary(),
        "total": tp.rule_count,
        "enabled": tp.enabled_rule_count,
    }


@app.post("/tool-permissions/reload")
async def reload_tool_permissions_api(request: Request):
    """
    Hot-reload tool permission rules from a JSON body or the configured YAML.

    Requires X-Aegivis-Admin-Key header when AEGIVIS_ADMIN_API_KEY is configured.
    Body (optional): {"rules": [...]}  -- list of rule dicts in YAML schema
    No body          -- reload from AEGIVIS_TOOL_PERMISSIONS_YAML or bundled file
    """
    auth_error = _require_admin_key(request)
    if auth_error:
        return auth_error

    try:
        body = await request.json()
        rules_data = body.get("rules")
    except Exception:
        rules_data = None

    if rules_data is not None:
        count = reload_tool_permissions_engine(rules_data=rules_data)
    elif settings.tool_permissions_yaml:
        count = reload_tool_permissions_engine(
            yaml_path=Path(settings.tool_permissions_yaml)
        )
    else:
        count = reload_tool_permissions_engine()

    tp = get_tool_permissions_engine()
    return {
        "status": "reloaded",
        "rule_count": count,
        "enabled_rule_count": tp.enabled_rule_count,
    }


# ─── Security Benchmark endpoints ───────────────────────────────────────────

# In-memory cache — survives the process lifetime, resets on restart.
_benchmark_cache: dict | None = None


class BenchmarkRunRequest(BaseModel):
    use_classifier: bool = False


@app.get("/benchmark/last")
async def get_last_benchmark():
    """
    Return the most recent benchmark report, or null if none has been run yet
    (always HTTP 200 — avoids spurious browser console 404 errors).

    The report is cached in memory and persists until the proxy restarts or
    a new benchmark is run via POST /benchmark/run.
    """
    return _benchmark_cache  # None → JSON null (200); report dict when available


@app.post("/benchmark/run")
async def run_benchmark_endpoint(req: BenchmarkRunRequest | None = Body(default=None)):
    """
    Run the full security benchmark against the labelled attack dataset.

    Executes synchronously in a thread-pool executor to avoid blocking the
    event loop.  Returns the full BenchmarkReport as JSON and caches it.

    Optional body: {"use_classifier": true} — include DeBERTa ML classifier
    in each scan (adds ~50ms/case; requires aegivis-proxy[classifier]).

    Typical runtime: 0.5-5s (structural only) to 30-120s (with semantic/
    classifier layers active).
    """
    global _benchmark_cache
    use_clf = req.use_classifier if req else False
    try:
        from .benchmark import run_benchmark
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None, functools.partial(run_benchmark, use_classifier=use_clf)
        )
        _benchmark_cache = report.to_dict()
        return _benchmark_cache
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Benchmark module unavailable: {exc}",
        )
    except Exception as exc:
        logger.exception("Benchmark run failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {exc}",
        )


# ─── External Dataset Benchmark endpoints ─────────────────────────────────────

_external_benchmark_cache: dict | None = None


@app.get("/benchmark/external/last")
async def get_last_external_benchmark():
    """
    Return the most recent external dataset benchmark report, or null if none
    has been run yet (always HTTP 200 — avoids spurious browser console 404 errors).

    External benchmarks run against real-world HuggingFace datasets:
    deepset/prompt-injections, lmsys/toxic-chat, xTRam1/safe-guard-prompt-injection,
    nvidia/Aegis-AI-Content-Safety-Dataset-1.0, fka/awesome-chatgpt-prompts.

    Requires: pip install datasets
    """
    return _external_benchmark_cache  # None → JSON null (200); report dict when available


@app.post("/benchmark/external/run")
async def run_external_benchmark_endpoint(req: BenchmarkRunRequest | None = Body(default=None)):
    """
    Run the security benchmark against real-world HuggingFace datasets.

    Uses streaming mode — only the rows we actually need are downloaded, so
    the first run typically completes in 30-120s (vs 60+ min previously).

    A hard wall-clock timeout is enforced (default 600s, configurable via
    BENCHMARK_EXTERNAL_TIMEOUT_S env var).  If the benchmark does not complete
    within that window a 504 is returned with guidance on reducing sample size.

    Optional body: {"use_classifier": true} — include DeBERTa ML classifier.

    Requires: pip install datasets
    """
    global _external_benchmark_cache
    use_clf = req.use_classifier if req else False
    _timeout_s = int(os.environ.get("BENCHMARK_EXTERNAL_TIMEOUT_S", "600"))
    loop = asyncio.get_running_loop()
    try:
        from .benchmark import run_external_benchmark
        report = await asyncio.wait_for(
            loop.run_in_executor(
                None, functools.partial(run_external_benchmark, use_classifier=use_clf)
            ),
            timeout=_timeout_s,
        )
        _external_benchmark_cache = report.to_dict()
        return _external_benchmark_cache
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"External benchmark timed out after {_timeout_s}s. "
                "Reduce sample size with env vars: "
                "BENCHMARK_EXT_MAX_ATTACKS (default 150), "
                "BENCHMARK_EXT_MAX_CLEAN (default 75), "
                "BENCHMARK_EXT_TOTAL_MAX (default 1200). "
                "Or raise the deadline: BENCHMARK_EXTERNAL_TIMEOUT_S."
            ),
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"HuggingFace 'datasets' library not installed on proxy. "
                f"Run: pip install datasets  (error: {exc})"
            ),
        )
    except RuntimeError as exc:
        # load_external_cases() raises RuntimeError when all datasets fail
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("External benchmark run failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"External benchmark failed: {exc}",
        )


# ─── Security Playground endpoint ───────────────────────────────────────────

@app.post("/playground/scan")
async def playground_scan(request: Request):
    """
    Scan arbitrary text through the sync security enforcement pipeline.

    Used by the dashboard Security Playground so operators can paste prompts,
    tool calls, or tool outputs and see exactly which layers fire and why.

    Body:
      { "text": "...", "mode": "prompt" }
      { "mode": "tool_call", "tool_name": "http_request", "tool_args": {...} }

    Returns per-layer scores and an overall verdict with latency.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    mode = body.get("mode", "prompt")
    text = body.get("text", "")
    tool_name = body.get("tool_name", "unknown_tool")
    tool_args = body.get("tool_args", {})

    t0 = time.perf_counter()
    layers: list[dict] = []
    overall_score = 0.0
    would_block = False

    try:
        if mode in ("prompt", "tool_output"):
            from .enforcement import scan_messages
            messages = [{"role": "user", "content": text}]
            result = scan_messages(messages)

            # Structural layer
            seg = result.structural.per_segment[0] if result.structural.per_segment else None
            layers.append({
                "id": "structural",
                "name": "Structural Scanner",
                "score": round(result.structural.score, 4),
                "label": result.structural.label,
                "triggered": result.structural.score >= settings.security_injection_alert_threshold,
                "details": {
                    "matched_phrases": seg.matched_phrases[:5] if seg else [],
                    "delimiter_hits": seg.delimiter_hits if seg else 0,
                    "phrase_score": round(result.structural.per_segment[0].phrase_score, 4) if seg and hasattr(seg, "phrase_score") else 0.0,
                },
            })

            # Encoding layer
            layers.append({
                "id": "encoding",
                "name": "Encoding Normalizer",
                "score": round(result.encoding_score, 4),
                "label": "triggered" if result.encoding_detected else "safe",
                "triggered": bool(result.encoding_detected),
                "details": {"encoding_detected": result.encoding_detected},
            })

            # Credential layer
            layers.append({
                "id": "credential",
                "name": "Credential Scanner",
                "score": 1.0 if result.credential_detected else 0.0,
                "label": "detected" if result.credential_detected else "safe",
                "triggered": result.credential_detected,
                "details": {"credential_count": result.credential_count},
            })

            # ML classifier (only if sync mode ran it)
            if result.classifier_score > 0.0:
                layers.append({
                    "id": "ml_classifier",
                    "name": "ML Classifier (DeBERTa)",
                    "score": round(result.classifier_score, 4),
                    "label": result.classifier_label or "safe",
                    "triggered": result.classifier_score >= settings.analysis_classifier_threshold,
                    "details": {"model": settings.analysis_classifier_model},
                })

            # Unicode steganography
            try:
                from .security.unicode_scanner import scan_unicode_stego
                unicode_r = scan_unicode_stego(text)
                layers.append({
                    "id": "unicode",
                    "name": "Unicode Steganography",
                    "score": 1.0 if unicode_r.detected else 0.0,
                    "label": unicode_r.severity if unicode_r.detected else "safe",
                    "triggered": unicode_r.detected,
                    "details": {
                        "invisible_char_count": unicode_r.invisible_char_count,
                        "tag_decoded": unicode_r.tag_decoded or "",
                    },
                })
            except Exception:
                pass

            overall_score = result.injection_score
            block_t = settings.security_injection_block_threshold
            would_block = overall_score >= block_t or result.credential_detected

        elif mode == "tool_call":
            from .enforcement import scan_tool_call
            tool_result = scan_tool_call(tool_name, tool_args)

            layers.append({
                "id": "rce",
                "name": "RCE Scanner",
                "score": 1.0 if tool_result.rce_detected else 0.0,
                "label": "detected" if tool_result.rce_detected else "safe",
                "triggered": tool_result.rce_detected,
                "details": {},
            })
            layers.append({
                "id": "ssrf",
                "name": "SSRF Scanner",
                "score": 1.0 if tool_result.ssrf_detected else 0.0,
                "label": "detected" if tool_result.ssrf_detected else "safe",
                "triggered": tool_result.ssrf_detected,
                "details": {},
            })

            overall_score = 1.0 if (tool_result.rce_detected or tool_result.ssrf_detected) else 0.0
            would_block = bool(overall_score)

        else:
            return JSONResponse({"error": f"unknown mode: {mode}"}, status_code=400)

    except Exception as exc:
        logger.warning("playground_scan error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    latency_ms = (time.perf_counter() - t0) * 1000
    block_t = settings.security_injection_block_threshold
    alert_t = settings.security_injection_alert_threshold
    would_alert = (overall_score >= alert_t) and not would_block

    if would_block:
        overall_label = "malicious"
    elif would_alert:
        overall_label = "suspicious"
    else:
        overall_label = "safe"

    return {
        "overall_score": round(overall_score, 4),
        "overall_label": overall_label,
        "would_block": would_block,
        "would_alert": would_alert,
        "layers": layers,
        "latency_ms": round(latency_ms, 2),
        "thresholds": {"block": block_t, "alert": alert_t},
    }
