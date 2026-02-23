"""
Test configuration for backend tests.

Overrides the `get_session` FastAPI dependency with an AsyncMock by default so
that auth, schema-validation, and pure-logic tests can run without a real
PostgreSQL instance.

Tests that require a real database should be marked with `@pytest.mark.require_db`
and are automatically skipped when PostgreSQL / asyncpg is unavailable.
"""
from __future__ import annotations

import sys
import os

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock

import pytest


def _db_available() -> bool:
    try:
        import asyncpg  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Autouse fixture: replace get_session with an AsyncMock for every test so
# that engine creation is never attempted unless explicitly requested.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_db_session(request):
    """
    Replace the `get_session` dependency with a no-op AsyncMock.
    Tests decorated with @pytest.mark.require_db are skipped instead when
    the database driver is not available.
    """
    if request.node.get_closest_marker("require_db"):
        if not _db_available():
            pytest.skip("PostgreSQL / asyncpg not available")
        yield  # real DB -- no override
        return

    # Patch for non-DB tests
    from app.main import app
    from app.db.connection import get_session

    # Build a result mock with sync fetchall/fetchone (matches SQLAlchemy CursorResult API)
    result_mock = MagicMock()
    result_mock.fetchall.return_value = []
    result_mock.fetchone.return_value = None
    result_mock.scalars.return_value.all.return_value = []

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.execute = AsyncMock(return_value=result_mock)

    async def _mock_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = _mock_get_session
    yield mock_session
    app.dependency_overrides.pop(get_session, None)
