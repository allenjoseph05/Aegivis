"""
Default ScanConfig builder from environment variables.

Environment variables
---------------------
AEGIVIS_BACKEND_URL   Backend URL for event reporting (optional).
AEGIVIS_API_KEY       Backend API key (optional).
AEGIVIS_AGENT_ID      Agent identifier for reported events (optional).
AEGIVIS_SESSION_ID    Session identifier for reported events (optional).
"""
from __future__ import annotations

import os


def default_config():
    """
    Build a :class:`~aegivis.memory.ScanConfig` from environment variables.

    Import at call time to avoid circular import.
    """
    from aegivis.memory import ScanConfig  # noqa: PLC0415

    return ScanConfig(
        backend_url=os.environ.get("AEGIVIS_BACKEND_URL", ""),
        api_key=os.environ.get("AEGIVIS_API_KEY", ""),
        agent_id=os.environ.get("AEGIVIS_AGENT_ID", ""),
        session_id=os.environ.get("AEGIVIS_SESSION_ID", ""),
    )
