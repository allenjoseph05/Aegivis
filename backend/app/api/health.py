"""Health check endpoint."""
from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from ..db.connection import get_session

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "aegivis-backend", "version": "1.0.0"}


@router.get("/health/db", summary="Database health check")
async def health_db(db: AsyncSession = Depends(get_session)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}
