"""API key authentication middleware.

Returns an OrgContext so every endpoint knows which org the caller
belongs to — without trusting a client-supplied org_id parameter.

Resolution order:
1. DB lookup in api_keys table (key_hash → org_id)
2. Fallback: key in settings.api_keys list → "default-org"
3. 403 if neither matches

Process-level LRU cache (60 s TTL) avoids a DB round-trip on every
request.  Revoked keys are visible within 60 s of revocation.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.connection import get_session

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Process-level cache: sha256(key) → (OrgContext, expiry_monotonic)
_cache: dict[str, tuple["OrgContext", float]] = {}


@dataclass
class OrgContext:
    """Resolved identity for the authenticated caller."""
    org_id: str
    key_name: str | None = None


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
    db: AsyncSession = Depends(get_session),
) -> OrgContext:
    """FastAPI dependency: validates X-API-Key header, returns OrgContext."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    now = time.monotonic()

    # Fast path: process-level cache (60 s TTL)
    if key_hash in _cache:
        ctx, exp = _cache[key_hash]
        if now < exp:
            return ctx

    # DB lookup
    db_error: Exception | None = None
    try:
        row = (
            await db.execute(
                text(
                    "SELECT org_id, name FROM api_keys "
                    "WHERE key_hash = :h AND revoked_at IS NULL"
                ),
                {"h": key_hash},
            )
        ).fetchone()
    except Exception as e:
        logger.warning("DB unavailable during API key lookup: %s", e)
        db_error = e
        row = None

    if row:
        ctx = OrgContext(org_id=row.org_id, key_name=row.name)
    elif api_key in settings.api_keys:
        # Backward-compat: keys declared in config belong to default-org.
        # Remove from config list once the api_keys table is fully provisioned.
        ctx = OrgContext(org_id="default-org", key_name="config-key")
    elif db_error is not None:
        # DB is down and the key isn't in the config fallback list.
        # Return 503 so callers know this is transient, not an auth failure.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    _cache[key_hash] = (ctx, now + 60.0)
    return ctx
