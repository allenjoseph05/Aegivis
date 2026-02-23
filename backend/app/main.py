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
from .api.v1.anomalies import router as anomalies_router
from .api.v1.baselines import router as baselines_router
from .api.v1.compliance import router as compliance_router
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
    # Verify DB connection on startup
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    engine = get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified")
    except Exception as e:
        logger.warning(f"Database not reachable on startup: {e}. Will retry on first request.")

    logger.info(f"AgentBlackBox Backend ready on {settings.host}:{settings.port}")
    yield

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
app.include_router(ws_router, tags=["websocket"])  # no /v1 prefix - ws path is /ws/alerts
