"""
Tests for proxy/app/providers/mcp.py — MCP protocol parser and handler.

Tests cover:
- JSON-RPC 2.0 request parsing (all MCP methods)
- Response parsing
- Error response building
- Policy block response
- Audit payload extraction for each method type
- Argument truncation
- Edge cases: empty body, invalid JSON, non-MCP messages
"""
from __future__ import annotations

import json
import pytest

from proxy.app.providers.mcp import MCPProvider, MCPRequest, MCPResponse, _truncate_args


# ---------------------------------------------------------------------------
# MCPProvider.parse_request
# ---------------------------------------------------------------------------

class TestParseRequest:
    def test_tools_call(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {"query": "AI security 2026"},
            },
        }).encode()
        req = MCPProvider.parse_request(body)
        assert req is not None
        assert req.method == "tools/call"
        assert req.id == "1"
        assert req.tool_name == "web_search"
        assert req.tool_arguments == {"query": "AI security 2026"}
        assert not req.is_notification

    def test_tools_list(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/list",
            "params": {},
        }).encode()
        req = MCPProvider.parse_request(body)
        assert req is not None
        assert req.method == "tools/list"
        assert req.id == 42

    def test_initialize(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "claude-desktop", "version": "1.0.0"},
                "capabilities": {"roots": {}, "sampling": {}},
            },
        }).encode()
        req = MCPProvider.parse_request(body)
        assert req is not None
        assert req.method == "initialize"
        assert req.params["clientInfo"]["name"] == "claude-desktop"

    def test_resources_read(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "resources/read",
            "params": {"uri": "file:///etc/config.yaml"},
        }).encode()
        req = MCPProvider.parse_request(body)
        assert req is not None
        assert req.params["uri"] == "file:///etc/config.yaml"

    def test_notification_has_no_id(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed",
            "params": {},
        }).encode()
        req = MCPProvider.parse_request(body)
        assert req is not None
        assert req.is_notification
        assert req.id is None

    def test_accepts_dict(self):
        data = {"jsonrpc": "2.0", "id": "1", "method": "ping", "params": {}}
        req = MCPProvider.parse_request(data)
        assert req is not None
        assert req.method == "ping"

    def test_accepts_str(self):
        body = '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'
        req = MCPProvider.parse_request(body)
        assert req is not None

    def test_returns_none_for_empty(self):
        assert MCPProvider.parse_request(b"") is None
        assert MCPProvider.parse_request("") is None

    def test_returns_none_for_invalid_json(self):
        assert MCPProvider.parse_request(b"not json") is None

    def test_returns_none_for_wrong_jsonrpc_version(self):
        body = json.dumps({"jsonrpc": "1.0", "id": "1", "method": "tools/call"}).encode()
        assert MCPProvider.parse_request(body) is None

    def test_returns_none_for_missing_method(self):
        body = json.dumps({"jsonrpc": "2.0", "id": "1"}).encode()
        assert MCPProvider.parse_request(body) is None


# ---------------------------------------------------------------------------
# MCPProvider.parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_success_response(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "content": [{"type": "text", "text": "Search results here."}],
            },
        }).encode()
        resp = MCPProvider.parse_response(body)
        assert resp is not None
        assert not resp.is_error
        assert resp.tool_result_text == "Search results here."

    def test_error_response(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": "1",
            "error": {"code": -32603, "message": "Internal error"},
        }).encode()
        resp = MCPProvider.parse_response(body)
        assert resp is not None
        assert resp.is_error
        assert resp.error["code"] == -32603

    def test_tools_list_response(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": "2",
            "result": {
                "tools": [
                    {"name": "web_search", "description": "..."},
                    {"name": "send_email", "description": "..."},
                ],
            },
        }).encode()
        resp = MCPProvider.parse_response(body)
        assert resp is not None
        assert len(resp.listed_tools) == 2
        assert resp.listed_tools[0]["name"] == "web_search"

    def test_returns_none_for_invalid(self):
        assert MCPProvider.parse_response(b"bad") is None


# ---------------------------------------------------------------------------
# MCPProvider.build_error_response / build_policy_block_response
# ---------------------------------------------------------------------------

class TestErrorBuilding:
    def test_build_error_response(self):
        resp = MCPProvider.build_error_response("req-1", -32600, "Invalid Request")
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-1"
        assert resp["error"]["code"] == -32600
        assert resp["error"]["message"] == "Invalid Request"

    def test_build_error_response_with_data(self):
        resp = MCPProvider.build_error_response("req-1", -32603, "Oops", data={"detail": "x"})
        assert resp["error"]["data"] == {"detail": "x"}

    def test_build_policy_block_response(self):
        resp = MCPProvider.build_policy_block_response("req-2", "tool-denied", "not allowed")
        assert resp["error"]["code"] == -32600
        assert resp["error"]["data"]["rule"] == "tool-denied"
        assert "not allowed" in resp["error"]["data"]["reason"]


# ---------------------------------------------------------------------------
# MCPProvider.extract_audit_payload
# ---------------------------------------------------------------------------

class TestAuditPayload:
    def _req(self, method: str, params: dict, req_id: str = "1") -> MCPRequest:
        return MCPRequest(jsonrpc="2.0", method=method, id=req_id, params=params)

    def test_tools_call_payload(self):
        req = self._req("tools/call", {"name": "send_email", "arguments": {"to": "x@y.com", "body": "hi"}})
        p = MCPProvider.extract_audit_payload(req)
        assert p["mcp_method"] == "tools/call"
        assert p["tool_name"] == "send_email"
        assert p["tool_arguments"]["to"] == "x@y.com"

    def test_initialize_payload(self):
        req = self._req("initialize", {
            "clientInfo": {"name": "my-agent", "version": "0.1"},
            "capabilities": {"roots": {}, "sampling": {}},
        })
        p = MCPProvider.extract_audit_payload(req)
        assert p["client_name"] == "my-agent"
        assert "roots" in p["client_capabilities"]

    def test_resources_read_payload(self):
        req = self._req("resources/read", {"uri": "file:///secrets.txt"})
        p = MCPProvider.extract_audit_payload(req)
        assert "secrets.txt" in p["resource_uri"]

    def test_prompts_get_payload(self):
        req = self._req("prompts/get", {"name": "summarise", "arguments": {"text": "hello"}})
        p = MCPProvider.extract_audit_payload(req)
        assert p["prompt_name"] == "summarise"

    def test_notification_payload(self):
        req = MCPRequest(jsonrpc="2.0", method="notifications/message", id=None, params={})
        p = MCPProvider.extract_audit_payload(req)
        assert p["is_notification"] is True


# ---------------------------------------------------------------------------
# MCPProvider.extract_response_payload
# ---------------------------------------------------------------------------

class TestResponsePayload:
    def test_tools_call_result(self):
        resp = MCPResponse(
            id="1",
            result={"content": [{"type": "text", "text": "The answer is 42."}]},
        )
        p = MCPProvider.extract_response_payload("tools/call", resp)
        assert "42" in p["result_preview"]
        assert p["is_error"] is False

    def test_tools_list_result(self):
        resp = MCPResponse(
            id="2",
            result={"tools": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        )
        p = MCPProvider.extract_response_payload("tools/list", resp)
        assert p["tool_count"] == 3
        assert "a" in p["tool_names"]

    def test_error_response_recorded(self):
        resp = MCPResponse(id="3", error={"code": -32603, "message": "Internal"})
        p = MCPProvider.extract_response_payload("tools/call", resp)
        assert p["rpc_error_code"] == -32603


# ---------------------------------------------------------------------------
# _truncate_args
# ---------------------------------------------------------------------------

class TestTruncateArgs:
    def test_truncates_long_strings(self):
        args = {"text": "x" * 1000}
        result = _truncate_args(args, max_str_len=300)
        assert len(result["text"]) == 301  # 300 chars + "…"

    def test_short_strings_unchanged(self):
        args = {"key": "short value"}
        result = _truncate_args(args, max_str_len=300)
        assert result["key"] == "short value"

    def test_nested_dict_truncated(self):
        args = {"inner": {"text": "y" * 500}}
        result = _truncate_args(args, max_str_len=300)
        assert len(result["inner"]["text"]) == 301

    def test_list_strings_truncated(self):
        args = {"items": ["a" * 400, "short"]}
        result = _truncate_args(args, max_str_len=300)
        assert len(result["items"][0]) == 301
        assert result["items"][1] == "short"

    def test_non_string_values_unchanged(self):
        args = {"count": 42, "flag": True, "ratio": 0.5}
        result = _truncate_args(args)
        assert result == args


# ---------------------------------------------------------------------------
# MCPRequest properties
# ---------------------------------------------------------------------------

class TestMCPRequestProperties:
    def test_tool_name_only_for_tools_call(self):
        req = MCPRequest(jsonrpc="2.0", method="tools/list", id="1", params={"name": "foo"})
        assert req.tool_name is None  # only tools/call returns tool_name

    def test_tool_arguments_only_for_tools_call(self):
        req = MCPRequest(jsonrpc="2.0", method="initialize", id="1", params={"arguments": {"x": 1}})
        assert req.tool_arguments == {}

    def test_is_notification_true_when_no_id(self):
        req = MCPRequest(jsonrpc="2.0", method="notifications/cancelled", id=None, params={})
        assert req.is_notification

    def test_is_notification_false_with_id(self):
        req = MCPRequest(jsonrpc="2.0", method="tools/call", id="1", params={})
        assert not req.is_notification


# ---------------------------------------------------------------------------
# Intercepted / passthrough method sets
# ---------------------------------------------------------------------------

class TestMethodSets:
    def test_intercepted_methods_present(self):
        for method in ("tools/call", "tools/list", "resources/read", "prompts/get", "initialize"):
            assert method in MCPProvider.INTERCEPTED_METHODS

    def test_passthrough_methods_present(self):
        assert "ping" in MCPProvider.PASSTHROUGH_METHODS
