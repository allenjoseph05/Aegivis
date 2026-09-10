"""
Model Context Protocol (MCP) provider for Aegivis proxy.

Handles the MCP Streamable HTTP transport (spec 2025-03-26):
  https://spec.modelcontextprotocol.io/specification/2025-03-26/

The proxy intercepts MCP JSON-RPC 2.0 messages, applies security policy to
``tools/call`` requests (tool name allowlist, argument scanning, PDG tracking),
logs all MCP interactions to the audit trail, then forwards to the real MCP server.

Supported MCP methods intercepted:
    tools/call          — Tool invocation (security-scanned, policy-enforced)
    tools/list          — Tool discovery (logged, allowlist filtering optional)
    resources/read      — Resource access (logged)
    prompts/get         — Prompt retrieval (logged)
    initialize          — Session establishment (logged)
    notifications/*     — All notifications (passed through, logged)

MCP methods passed through without interception:
    ping                — Health check
    sampling/*          — LLM sampling requests (forwarded as-is)

Route: POST /mcp   (Streamable HTTP transport)
       GET  /mcp   (SSE stream for server-initiated messages)

The upstream MCP server URL is resolved in this priority order:
    1. X-Aegivis-Mcp-Server  request header (per-call override)
    2. AEGIVIS_MCP_SERVER_URL environment variable (global default)
    3. 503 if neither is set
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP message types
# ---------------------------------------------------------------------------

@dataclass
class MCPRequest:
    """Parsed MCP JSON-RPC 2.0 request."""
    jsonrpc: str
    method: str
    id: Any | None          = None
    params: dict            = field(default_factory=dict)

    @property
    def is_notification(self) -> bool:
        """Notifications have no id field."""
        return self.id is None

    @property
    def tool_name(self) -> str | None:
        """Return tool name if this is a tools/call request."""
        if self.method == "tools/call":
            return self.params.get("name")
        return None

    @property
    def tool_arguments(self) -> dict:
        """Return tool arguments if this is a tools/call request."""
        if self.method == "tools/call":
            return self.params.get("arguments") or {}
        return {}


@dataclass
class MCPResponse:
    """Parsed MCP JSON-RPC 2.0 response."""
    id: Any | None = None
    result: Any    = None
    error: dict | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def tool_result_content(self) -> list[dict]:
        """Extract content blocks from a tools/call result."""
        if not self.result:
            return []
        return self.result.get("content") or []

    @property
    def tool_result_text(self) -> str:
        """Concatenate all text content blocks from a tools/call result."""
        return " ".join(
            block.get("text", "")
            for block in self.tool_result_content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    @property
    def listed_tools(self) -> list[dict]:
        """Extract tool definitions from a tools/list response."""
        if not self.result:
            return []
        return self.result.get("tools") or []


class MCPProvider:
    """
    Parser and serialiser for the MCP Streamable HTTP transport.

    Stateless — all methods are class-level or static.
    """

    # MCP methods that Aegivis actively scans/enforces
    INTERCEPTED_METHODS = frozenset({
        "tools/call",
        "tools/list",
        "resources/read",
        "prompts/get",
        "initialize",
    })

    # MCP methods passed through without interception
    PASSTHROUGH_METHODS = frozenset({
        "ping",
        "sampling/createMessage",
        "sampling/createMessageStream",
    })

    @staticmethod
    def parse_request(body: bytes | str | dict) -> MCPRequest | None:
        """
        Parse a raw MCP request body into an MCPRequest.

        Accepts bytes, str (JSON), or an already-parsed dict.
        Returns None if the body is not a valid MCP JSON-RPC message.
        """
        try:
            if isinstance(body, (bytes, bytearray)):
                data = json.loads(body)
            elif isinstance(body, str):
                data = json.loads(body)
            else:
                data = body

            if not isinstance(data, dict):
                return None
            if data.get("jsonrpc") != "2.0":
                return None
            method = data.get("method")
            if not method or not isinstance(method, str):
                return None

            return MCPRequest(
                jsonrpc="2.0",
                method=method,
                id=data.get("id"),
                params=data.get("params") or {},
            )
        except Exception:
            return None

    @staticmethod
    def parse_response(body: bytes | str | dict) -> MCPResponse | None:
        """Parse a raw MCP response body into an MCPResponse."""
        try:
            if isinstance(body, (bytes, bytearray)):
                data = json.loads(body)
            elif isinstance(body, str):
                data = json.loads(body)
            else:
                data = body

            if not isinstance(data, dict):
                return None

            return MCPResponse(
                id=data.get("id"),
                result=data.get("result"),
                error=data.get("error"),
            )
        except Exception:
            return None

    @staticmethod
    def build_error_response(
        request_id: Any,
        code: int,
        message: str,
        data: dict | None = None,
    ) -> dict:
        """Build a JSON-RPC 2.0 error response body."""
        err: dict = {"code": code, "message": message}
        if data:
            err["data"] = data
        return {
            "jsonrpc": "2.0",
            "id":      request_id,
            "error":   err,
        }

    @staticmethod
    def build_policy_block_response(
        request_id: Any,
        rule_name: str,
        reason: str,
    ) -> dict:
        """
        Build the MCP error response returned when Aegivis blocks a tools/call.

        Uses JSON-RPC error code -32600 (Invalid Request) to signal a policy
        violation rather than a transport or parsing error.
        """
        return MCPProvider.build_error_response(
            request_id=request_id,
            code=-32600,
            message="Policy violation — tool call blocked by Aegivis",
            data={"rule": rule_name, "reason": reason},
        )

    @staticmethod
    def extract_audit_payload(request: MCPRequest) -> dict:
        """
        Build the audit event payload for an MCP request.

        This is the dict stored in the Aegivis backend's audit_events table
        under the ``payload`` JSONB column.
        """
        p: dict = {
            "mcp_method":  request.method,
            "is_notification": request.is_notification,
        }

        if request.method == "tools/call":
            p["tool_name"]      = request.tool_name
            p["tool_arguments"] = _truncate_args(request.tool_arguments)

        elif request.method == "tools/list":
            cursor = request.params.get("cursor")
            if cursor:
                p["cursor"] = str(cursor)[:100]

        elif request.method == "resources/read":
            p["resource_uri"] = str(request.params.get("uri", ""))[:500]

        elif request.method == "prompts/get":
            p["prompt_name"]      = str(request.params.get("name", ""))
            p["prompt_arguments"] = request.params.get("arguments") or {}

        elif request.method == "initialize":
            client_info = request.params.get("clientInfo") or {}
            p["client_name"]    = client_info.get("name", "unknown")
            p["client_version"] = client_info.get("version", "unknown")
            caps = request.params.get("capabilities") or {}
            p["client_capabilities"] = list(caps.keys())

        return p

    @staticmethod
    def extract_response_payload(
        mcp_method: str,
        response: MCPResponse,
    ) -> dict:
        """Build the audit payload for an MCP response."""
        p: dict = {"mcp_method": mcp_method}

        if mcp_method == "tools/call":
            text = response.tool_result_text
            if text:
                p["result_preview"] = text[:500]
            p["is_error"] = bool(
                any(
                    b.get("type") == "error"
                    for b in response.tool_result_content
                    if isinstance(b, dict)
                )
            )

        elif mcp_method == "tools/list":
            tools = response.listed_tools
            p["tool_count"] = len(tools)
            p["tool_names"] = [t.get("name", "?") for t in tools[:20]]

        elif mcp_method == "initialize":
            server_info = (response.result or {}).get("serverInfo") or {}
            p["server_name"]    = server_info.get("name", "unknown")
            p["server_version"] = server_info.get("version", "unknown")

        if response.is_error:
            p["rpc_error_code"]    = (response.error or {}).get("code")
            p["rpc_error_message"] = str((response.error or {}).get("message", ""))[:200]

        return p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate_args(args: dict, max_str_len: int = 300) -> dict:
    """Recursively truncate long string values in an argument dict."""
    result: dict = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_str_len:
            result[k] = v[:max_str_len] + "…"
        elif isinstance(v, dict):
            result[k] = _truncate_args(v, max_str_len)
        elif isinstance(v, list):
            result[k] = [
                (item[:max_str_len] + "…" if isinstance(item, str) and len(item) > max_str_len else item)
                for item in v[:50]
            ]
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# MCP JSON-RPC error codes (standard)
# ---------------------------------------------------------------------------

class MCPErrorCode:
    PARSE_ERROR      = -32700
    INVALID_REQUEST  = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS   = -32602
    INTERNAL_ERROR   = -32603
    # MCP-specific
    RESOURCE_NOT_FOUND = -32002
    TOOL_NOT_FOUND     = -32003
    POLICY_BLOCKED     = -32600  # reuse Invalid Request for policy blocks
