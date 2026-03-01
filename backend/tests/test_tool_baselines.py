"""
Unit tests for /v1/baselines tool baseline endpoint functions.

Tests call the async endpoint handler functions directly with a mocked
SQLAlchemy AsyncSession — no database, no Docker, no network required.

Run: cd backend && python -m pytest tests/test_tool_baselines.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

import pytest

from app.api.v1.baselines import (
    list_baselines,
    list_agent_tools,
    get_approved_tools,
    observe_tools,
    approve_tool,
    deny_tool,
    approve_all_tools,
    ObserveRequest,
)
from app.middleware.auth import OrgContext

TEST_ORG   = "test-org"
TEST_AGENT = "agent-1"
ORG_CTX    = OrgContext(org_id=TEST_ORG, key_name="test-key")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build mock DB results
# ─────────────────────────────────────────────────────────────────────────────

def _row(**fields):
    """Build a dict that supports both dict-style and attribute access."""
    m = MagicMock()
    m.__getitem__ = lambda self, k: fields[k]
    m.get = lambda k, d=None: fields.get(k, d)
    for k, v in fields.items():
        setattr(m, k, v)
    return m


def _mapping_result(rows: list[dict]):
    """
    Return a mock CursorResult whose .mappings() yields the given rows as
    dict-like objects, and whose direct iteration yields (value,) tuples
    (for single-column SELECT queries).
    """
    result = MagicMock()
    result.rowcount = len(rows)

    map_rows = [_row(**r) for r in rows]
    mapping_obj = MagicMock()
    mapping_obj.__iter__ = MagicMock(return_value=iter(map_rows))
    result.mappings = MagicMock(return_value=mapping_obj)

    # Direct iteration: yield first value of each row (tool_name queries)
    first_vals = [[list(r.values())[0]] for r in rows] if rows else []
    result.__iter__ = MagicMock(return_value=iter(first_vals))
    return result


def _db(rows: list[dict] | None = None, rowcount: int | None = None):
    """Return a mock AsyncSession."""
    result = _mapping_result(rows or [])
    if rowcount is not None:
        result.rowcount = rowcount
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    db.commit  = AsyncMock()
    return db


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/baselines
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_baselines_empty():
    resp = await list_baselines(db=_db([]), org_ctx=ORG_CTX)
    assert resp["count"] == 0
    assert resp["baselines"] == []


@pytest.mark.asyncio
async def test_list_baselines_shows_agent():
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    rows = [{"agent_id": TEST_AGENT, "pending": 2, "approved": 1,
             "denied": 0, "total": 3, "last_seen_at": now}]
    resp = await list_baselines(db=_db(rows), org_ctx=ORG_CTX)
    assert resp["count"] == 1
    b = resp["baselines"][0]
    assert b["agent_id"] == TEST_AGENT
    assert b["pending"] == 2
    assert b["approved"] == 1
    assert b["enforcement_active"] is True   # approved > 0


@pytest.mark.asyncio
async def test_list_baselines_enforcement_false_when_no_approved():
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    rows = [{"agent_id": TEST_AGENT, "pending": 3, "approved": 0,
             "denied": 0, "total": 3, "last_seen_at": now}]
    resp = await list_baselines(db=_db(rows), org_ctx=ORG_CTX)
    assert resp["baselines"][0]["enforcement_active"] is False


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/baselines/{agent_id}/tools/observe
# ─────────────────────────────="────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observe_creates_tools():
    req = ObserveRequest(
        tools=[{"name": "search_kb", "description": "Search", "schema": {}},
               {"name": "create_ticket", "description": "Ticket", "schema": {}}],
        session_id="sess-1",
    )
    resp = await observe_tools(TEST_AGENT, body=req, db=_db([]), org_ctx=ORG_CTX)
    assert resp["upserted"] == 2


@pytest.mark.asyncio
async def test_observe_skips_tools_without_name():
    req = ObserveRequest(
        tools=[{"description": "no name", "schema": {}},
               {"name": "valid_tool", "description": "ok", "schema": {}}],
        session_id="sess-1",
    )
    resp = await observe_tools(TEST_AGENT, body=req, db=_db([]), org_ctx=ORG_CTX)
    assert resp["upserted"] == 1


@pytest.mark.asyncio
async def test_observe_empty_list():
    req = ObserveRequest(tools=[], session_id="sess-1")
    resp = await observe_tools(TEST_AGENT, body=req, db=_db([]), org_ctx=ORG_CTX)
    assert resp["upserted"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/baselines/{agent_id}/tools/approved
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approved_returns_404_when_empty():
    from fastapi import HTTPException
    db = _db([])
    # Direct iteration returns no rows
    db.execute.return_value.__iter__ = MagicMock(return_value=iter([]))
    with pytest.raises(HTTPException) as exc:
        await get_approved_tools(TEST_AGENT, db=db, org_ctx=ORG_CTX)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_approved_returns_tool_names():
    db = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.__iter__ = MagicMock(return_value=iter([["search_kb"], ["create_ticket"]]))
    db.execute = AsyncMock(return_value=result)

    resp = await get_approved_tools(TEST_AGENT, db=db, org_ctx=ORG_CTX)
    assert set(resp["approved_tools"]) == {"search_kb", "create_ticket"}
    assert resp["count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# POST approve / deny
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_tool_success():
    resp = await approve_tool(TEST_AGENT, "search_kb", db=_db([], rowcount=1), org_ctx=ORG_CTX)
    assert resp["status"] == "approved"
    assert resp["tool_name"] == "search_kb"


@pytest.mark.asyncio
async def test_deny_tool_success():
    resp = await deny_tool(TEST_AGENT, "create_ticket", db=_db([], rowcount=1), org_ctx=ORG_CTX)
    assert resp["status"] == "denied"


@pytest.mark.asyncio
async def test_approve_unknown_tool_raises_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await approve_tool(TEST_AGENT, "unknown", db=_db([], rowcount=0), org_ctx=ORG_CTX)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_deny_unknown_tool_raises_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await deny_tool(TEST_AGENT, "unknown", db=_db([], rowcount=0), org_ctx=ORG_CTX)
    assert exc.value.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# POST approve-all
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_all_returns_approved_list():
    db = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.rowcount = 2
    result.__iter__ = MagicMock(return_value=iter([["search_kb"], ["create_ticket"]]))
    db.execute = AsyncMock(return_value=result)

    resp = await approve_all_tools(TEST_AGENT, db=db, org_ctx=ORG_CTX)
    assert resp["approved_count"] == 2
    assert set(resp["approved_tools"]) == {"search_kb", "create_ticket"}


@pytest.mark.asyncio
async def test_approve_all_zero_pending_returns_empty():
    db = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.rowcount = 0
    result.__iter__ = MagicMock(return_value=iter([]))
    db.execute = AsyncMock(return_value=result)

    resp = await approve_all_tools(TEST_AGENT, db=db, org_ctx=ORG_CTX)
    assert resp["approved_count"] == 0
    assert resp["approved_tools"] == []
