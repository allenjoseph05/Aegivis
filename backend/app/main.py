"""
AgentBlackBox Backend — FastAPI application.

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
from .api.v1.export import router as export_router
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
                "OTel tracing requested (ABB_OTEL_ENABLED=true) but packages missing. "
                "Install with: pip install 'agentblackbox-backend[observability]'"
            )

    # Redis Streams consumers (Phase D) — only when ABB_REDIS_URL is set
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
                    name="abb-stream-events",
                )
            )
            _stream_tasks.append(
                asyncio.create_task(
                    run_violation_consumer(_redis_client, sf),
                    name="abb-stream-violations",
                )
            )
            logger.info("Redis Streams consumers started")
        except ImportError:
            logger.warning(
                "ABB_REDIS_URL is set but redis package is missing. "
                "Install with: pip install 'agentblackbox-backend[redis]'"
            )
        except Exception as e:
            logger.warning("Redis unavailable (%s) — stream consumers not started", e)

    logger.info(f"AgentBlackBox Backend ready on {settings.host}:{settings.port}")
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
    logger.info("AgentBlackBox Backend shutdown complete")


app = FastAPI(
    title="AgentBlackBox Backend API",
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
app.include_router(export_router, prefix="/v1", tags=["export"])
app.include_router(ws_router, tags=["websocket"])  # no /v1 prefix - ws path is /ws/alerts
