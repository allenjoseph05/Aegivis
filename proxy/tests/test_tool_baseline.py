"""
Unit tests for proxy/app/security/tool_baseline.py

Tests cover:
  - Tool extraction from OpenAI and Anthropic request formats
  - Stable, order-independent hashing
  - Mid-session hash-change detection helpers
  - Approved-baseline cache behaviour (hit / miss / TTL / fail-open)
  - report_tools_observed (fire-and-forget, error suppression)
  - Edge cases: empty arrays, tools with no name, mixed formats

No network calls, no database, no Docker required.
Run: cd proxy && python -m pytest tests/test_tool_baseline.py -v
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.security.tool_baseline import (
    _normalize_tool,
    extract_tools,
    extract_tool_names,
    hash_tool_set,
    normalize_tools_for_storage,
    invalidate_cache,
    get_approved_baseline,
    report_tools_observed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search knowledge base",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket",
            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
    },
]

ANTHROPIC_TOOLS = [
    {
        "name": "search_kb",
        "description": "Search knowledge base",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket",
        "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_tool
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeTool:
    def test_openai_format(self):
        n = _normalize_tool(OPENAI_TOOLS[0])
        assert n is not None
        assert n["name"] == "search_kb"
        assert n["description"] == "Search knowledge base"
        assert "query" in n["schema"]["properties"]

    def test_anthropic_format(self):
        n = _normalize_tool(ANTHROPIC_TOOLS[0])
        assert n is not None
        assert n["name"] == "search_kb"

    def test_missing_name_returns_none(self):
        assert _normalize_tool({"type": "function", "function": {"description": "x"}}) is None

    def test_non_dict_returns_none(self):
        assert _normalize_tool("not-a-dict") is None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# extract_tools / extract_tool_names
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTools:
    def test_openai_body(self):
        body = {"model": "gpt-4o", "tools": OPENAI_TOOLS, "messages": []}
        assert extract_tools(body) == OPENAI_TOOLS

    def test_anthropic_body(self):
        body = {"model": "claude-3-5-sonnet", "tools": ANTHROPIC_TOOLS, "messages": []}
        assert extract_tools(body) == ANTHROPIC_TOOLS

    def test_no_tools_key(self):
        assert extract_tools({"model": "gpt-4o", "messages": []}) == []

    def test_tools_not_list(self):
        assert extract_tools({"tools": "search"}) == []

    def test_extract_names_openai(self):
        names = extract_tool_names(OPENAI_TOOLS)
        assert names == {"search_kb", "create_ticket"}

    def test_extract_names_anthropic(self):
        names = extract_tool_names(ANTHROPIC_TOOLS)
        assert names == {"search_kb", "create_ticket"}

    def test_extract_names_empty(self):
        assert extract_tool_names([]) == set()


# ─────────────────────────────────────────────────────────────────────────────
# hash_tool_set
# ─────────────────────────────────────────────────────────────────────────────

class TestHashToolSet:
    def test_stable_across_calls(self):
        assert hash_tool_set(OPENAI_TOOLS) == hash_tool_set(OPENAI_TOOLS)

    def test_order_independent(self):
        reversed_tools = list(reversed(OPENAI_TOOLS))
        assert hash_tool_set(OPENAI_TOOLS) == hash_tool_set(reversed_tools)

    def test_openai_and_anthropic_same_tools_same_hash(self):
        # Same tool set expressed in different provider formats should
        # produce the same fingerprint (normalized before hashing).
        assert hash_tool_set(OPENAI_TOOLS) == hash_tool_set(ANTHROPIC_TOOLS)

    def test_adding_tool_changes_hash(self):
        h1 = hash_tool_set(OPENAI_TOOLS[:1])
        h2 = hash_tool_set(OPENAI_TOOLS)
        assert h1 != h2

    def test_schema_change_changes_hash(self):
        import copy
        modified = copy.deepcopy(OPENAI_TOOLS)
        # Add an extra parameter to the first tool
        modified[0]["function"]["parameters"]["properties"]["extra"] = {"type": "boolean"}
        assert hash_tool_set(OPENAI_TOOLS) != hash_tool_set(modified)

    def test_empty_list(self):
        h = hash_tool_set([])
        assert isinstance(h, str) and len(h) == 16

    def test_returns_16_char_hex(self):
        h = hash_tool_set(OPENAI_TOOLS)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# normalize_tools_for_storage
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeForStorage:
    def test_returns_list_of_dicts(self):
        result = normalize_tools_for_storage(OPENAI_TOOLS)
        assert len(result) == 2
        assert all("name" in r and "description" in r and "schema" in r for r in result)

    def test_skips_nameless_entries(self):
        tools = [{"type": "function", "function": {}}]
        assert normalize_tools_for_storage(tools) == []


# ─────────────────────────────────────────────────────────────────────────────
# get_approved_baseline (cache + HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetApprovedBaseline:
    def setup_method(self):
        # Clear the module-level cache before each test
        from app.security import tool_baseline
        tool_baseline._baseline_cache.clear()

    @pytest.mark.asyncio
    async def test_returns_approved_set(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"approved_tools": ["search_kb", "create_ticket"]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            result = await get_approved_baseline(
                "agent-1", "org-1", "http://backend", "key", cache_ttl_s=60.0
            )
        assert result == frozenset({"search_kb", "create_ticket"})

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            result = await get_approved_baseline(
                "agent-new", "org-1", "http://backend", "key"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_backend_error(self):
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=Exception("connection refused"))
            MockClient.return_value = instance

            # Must NOT raise — fails open
            result = await get_approved_baseline(
                "agent-1", "org-1", "http://bad-host", "key"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_second_request(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"approved_tools": ["search_kb"]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            await get_approved_baseline("ag", "org", "http://b", "k", cache_ttl_s=60.0)
            await get_approved_baseline("ag", "org", "http://b", "k", cache_ttl_s=60.0)

            # HTTP should only be called once
            assert instance.get.call_count == 1

    @pytest.mark.asyncio
    async def test_expired_cache_re_fetches(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"approved_tools": ["search_kb"]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            await get_approved_baseline("ag2", "org", "http://b", "k", cache_ttl_s=0.0)
            await get_approved_baseline("ag2", "org", "http://b", "k", cache_ttl_s=0.0)

            # TTL=0 → both calls should fetch
            assert instance.get.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# report_tools_observed (fire-and-forget)
# ─────────────────────────────────────────────────────────────────────────────

class TestReportToolsObserved:
    @pytest.mark.asyncio
    async def test_posts_to_backend(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            await report_tools_observed(
                "agent-1", "org-1", OPENAI_TOOLS, "sess-1", "http://backend", "key"
            )
            assert instance.post.called

    @pytest.mark.asyncio
    async def test_suppresses_errors(self):
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.post = AsyncMock(side_effect=Exception("network down"))
            MockClient.return_value = instance

            # Must NOT raise
            await report_tools_observed(
                "agent-1", "org-1", OPENAI_TOOLS, "sess-1", "http://bad", "key"
            )

    @pytest.mark.asyncio
    async def test_empty_tools_skips_post(self):
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.post = AsyncMock()
            MockClient.return_value = instance

            await report_tools_observed(
                "agent-1", "org-1", [], "sess-1", "http://backend", "key"
            )
            assert not instance.post.called
