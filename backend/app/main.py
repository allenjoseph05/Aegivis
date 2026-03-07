"""
Aegivis Backend — FastAPI application.

Receives event batches from the proxy, persists to PostgreSQL,
and serves the REST API for the dashboard.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.health import router as health_router
from .api.v1.agents import router as agents_router
from .api.v1.anomalies import router as anomalies_router
from .api.v1.baselines import router as baselines_router
from .api.v1.metrics import router as metrics_router
from .api.v1.compliance import router as compliance_router
from .api.v1.policy import router as policy_router
from .api.v1.topology import router as topology_router
from .api.v1.events import router as events_router
from .api.v1.ingest import router as ingest_router
from .api.v1.replay import router as replay_router
from .api.v1.sessions import router as sessions_router
from .api.v1.verify import router as verify_router
from .api.v1.violations import router as violations_router
from .api.v1.ws import router as ws_router

# ── Enterprise plugin ────────────────────────────────────────────────────────
# If the aegivis-enterprise package is installed, it registers additional API
# endpoints for: org settings, security features, per-agent profiles, approvals,
# manifests, egress rules, security posture, analytics, export, and admin.
#
# Community builds omit these routes entirely — they require the enterprise
# package and its DB migrations.
#
# Dev fallback: when aegivis-enterprise is not installed as a package but the
# source files are present locally (monorepo development), we fall back to
# direct local imports so the full feature set remains available.
# REMOVE the _dev_fallback block when publishing community-only builds.

_enterprise_loaded = False
try:
    from aegivis_enterprise.backend_ext.routes import (  # type: ignore[import]
        register_enterprise_routes as _register_enterprise,
    )
    _enterprise_loaded = True
except ImportError:
    pass

from .config import settings
from .db.connection import close_engine, get_engine

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from sqlalchemy import text

    # Verify DB connection on startup
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified")
    except Exception as e:
        logger.warning(f"Database not reachable on startup: {e}. Will retry on first request.")

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
                "Install with: pip install 'aegivis-backend[observability]'"
            )

    # Redis Streams consumers (Phase D) — only when AEGIVIS_REDIS_URL is set
    _stream_tasks: list[asyncio.Task] = []
    _redis_client = None
    if settings.redis_url:
        try:
            from redis.asyncio import Redis as AsyncRedis
            from .db.connection import get_session_factory
            from .workers.stream_consumer import run_event_consumer, run_violation_consumer

            _redis_client = AsyncRedis.from_url(
                settings.redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            await _redis_client.ping()
            logger.info("Redis connected: %s", settings.redis_url)

            sf = get_session_factory()
            _stream_tasks.append(
                asyncio.create_task(
                    run_event_consumer(_redis_client, sf),
                    name="aegivis-stream-events",
                )
            )
            _stream_tasks.append(
                asyncio.create_task(
                    run_violation_consumer(_redis_client, sf),
                    name="aegivis-stream-violations",
                )
            )
            logger.info("Redis Streams consumers started")
        except ImportError:
            logger.warning(
                "AEGIVIS_REDIS_URL is set but redis package is missing. "
                "Install with: pip install 'aegivis-backend[redis]'"
            )
        except Exception as e:
            logger.warning("Redis unavailable (%s) — stream consumers not started", e)

    logger.info(f"Aegivis Backend ready on {settings.host}:{settings.port}")
    yield

    # Cancel stream consumer tasks on shutdown
    for task in _stream_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if _redis_client:
        await _redis_client.aclose()
        logger.info("Redis client closed")

    await close_engine()
    logger.info("Aegivis Backend shutdown complete")


app = FastAPI(
    title="Aegivis Backend API",
    version="1.0.0",
    description="Forensic audit backend for AI agents",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Mount Prometheus /metrics endpoint (Phase 3.4)
if settings.metrics_enabled:
    from .metrics import get_metrics_app
    _metrics_asgi = get_metrics_app()
    if _metrics_asgi is not None:
        app.mount("/metrics", _metrics_asgi)
        logger.info("Prometheus /metrics endpoint mounted (backend)")

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (added after CORS; Starlette processes in reverse-add order,
# so rate limiting executes before CORS on the inbound path)
from .middleware.rate_limit import RateLimitMiddleware  # noqa: E402
app.add_middleware(RateLimitMiddleware, settings=settings)

# Health check (no auth required)
app.include_router(health_router, tags=["health"])

# V1 API (auth required)
app.include_router(ingest_router, prefix="/v1", tags=["ingest"])
app.include_router(sessions_router, prefix="/v1", tags=["sessions"])
app.include_router(events_router, prefix="/v1", tags=["events"])
app.include_router(verify_router, prefix="/v1", tags=["verification"])
app.include_router(replay_router, prefix="/v1", tags=["forensics"])
app.include_router(compliance_router, prefix="/v1", tags=["compliance"])
app.include_router(violations_router, prefix="/v1", tags=["policy"])
app.include_router(baselines_router, prefix="/v1", tags=["baselines"])
app.include_router(anomalies_router, prefix="/v1", tags=["anomalies"])
app.include_router(metrics_router, prefix="/v1", tags=["metrics"])
app.include_router(policy_router, prefix="/v1", tags=["policy"])
app.include_router(agents_router, prefix="/v1", tags=["agents"])
app.include_router(topology_router, prefix="/v1", tags=["topology"])
app.include_router(ws_router, tags=["websocket"])  # no /v1 prefix - ws path is /ws/alerts

# ── Enterprise routes ────────────────────────────────────────────────────────
if _enterprise_loaded:
    _register_enterprise(app)
    logger.info("Enterprise features: enabled (aegivis-enterprise package)")
else:
    # Dev fallback — load enterprise routes from local source tree.
    # This block keeps the monorepo dev environment fully functional without
    # installing the enterprise package separately.
    # REMOVE this block when publishing the community-only build.
    def _load_dev_enterprise_routes() -> None:
        try:
            from .api.v1.analytics import router as _analytics
            from .api.v1.admin import router as _admin
            from .api.v1.org_settings import router as _org_settings
            from .api.v1.approvals import router as _approvals
            from .api.v1.manifests import router as _manifests
            from .api.v1.security_posture import router as _security_posture
            from .api.v1.egress_rules import router as _egress_rules
            from .api.v1.security_features import router as _security_features
            from .api.v1.agent_security_features import router as _agent_security_features
            from .api.v1.export import router as _export

            app.include_router(_analytics, prefix="/v1", tags=["analytics"])
            app.include_router(_admin, prefix="/v1", tags=["admin"])
            app.include_router(_org_settings, prefix="/v1", tags=["settings"])
            app.include_router(_approvals, prefix="/v1", tags=["approvals"])
            app.include_router(_manifests, prefix="/v1", tags=["manifests"])
            app.include_router(_security_posture, prefix="/v1", tags=["security-posture"])
            app.include_router(_egress_rules, prefix="/v1", tags=["egress-rules"])
            app.include_router(_security_features, prefix="/v1", tags=["security-features"])
            app.include_router(_agent_security_features, prefix="/v1", tags=["security-features"])
            app.include_router(_export, prefix="/v1", tags=["export"])
            logger.info("Enterprise features: dev mode (local imports)")
        except ImportError as exc:
            logger.warning("Enterprise dev fallback failed: %s — some routes unavailable", exc)

    _load_dev_enterprise_routes()
