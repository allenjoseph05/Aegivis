"""
Enterprise plugin hooks for the Aegivis proxy.

This module is always importable in the community edition. It attempts to load
the real EnterpriseHooks implementation from the ``aegivis-enterprise`` package.
If the package is not installed, it falls back to the no-op implementation —
all community security features continue to work unchanged.

Usage in intercept.py:
    from .enterprise_hooks import hooks

    result = await hooks.on_request_scan(session, messages, cfg)
    result = await hooks.on_tool_call_scan(session, tool_name, args, cfg)
    result = await hooks.on_response_scan(session, response_text, cfg)
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


class _NoOpHooks:
    """Community no-op implementation of the enterprise hook interface."""

    async def on_request_scan(self, session, messages, cfg) -> None:
        return None

    async def on_tool_call_scan(self, session, tool_name, args, cfg) -> None:
        return None

    async def on_response_scan(self, session, response_text, cfg) -> None:
        return None


# Attempt to load enterprise implementation
try:
    from aegivis_enterprise.proxy_ext.hooks import EnterpriseHooks as _EnterpriseHooks
    hooks = _EnterpriseHooks()
    _logger.debug("Enterprise proxy hooks loaded (aegivis-enterprise)")
except ImportError:
    hooks = _NoOpHooks()  # type: ignore[assignment]
    _logger.debug("Enterprise proxy hooks not available — community mode")
