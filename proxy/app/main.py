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
from .intercept import InterceptContext
from .policy import PolicyAction, get_policy_engine, reload_policy_engine
from .tool_permissions import (
    get_tool_permissions_engine,
    reload_tool_permissions_engine,
)
from .providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
)
from .session import get_session_tracker
from .transport import get_transport

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

# Provider routing table
PROVIDERS = {
    "openai": (OpenAIProvider, settings.openai_upstream),
    "anthropic": (AnthropicProvider, settings.anthropic_upstream),
    "google": (GoogleProvider, settings.google_upstream),
    "azure": (AzureProvider, settings.azure_upstream),
    "ollama": (OllamaProvider, settings.ollama_upstream),
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
_INTERCEPT_PATHS = {
    "openai": {"/v1/chat/completions", "/v1/completions"},
    "anthropic": {"/v1/messages"},
    "google": set(),   # matched by regex in route handler
    "azure": {"/openai/deployments"},   # prefix match
    "ollama": {"/v1/chat/completions", "/api/chat"},
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
