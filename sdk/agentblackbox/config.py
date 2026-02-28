"""
Default ScanConfig builder from environment variables.

Environment variables
---------------------
ABB_BACKEND_URL   Backend URL for event reporting (optional).
ABB_API_KEY       Backend API key (optional).
ABB_AGENT_ID      Agent identifier for reported events (optional).
ABB_SESSION_ID    Session identifier for reported events (optional).
"""
from __future__ import annotations

import os


def default_config():
    """
    Build a :class:`~agentblackbox.memory.ScanConfig` from environment variables.

    Import at call time to avoid circular import.
    """
    from agentblackbox.memory import ScanConfig  # noqa: PLC0415

    return ScanConfig(
        backend_url=os.environ.get("ABB_BACKEND_URL", ""),
        api_key=os.environ.get("ABB_API_KEY", ""),
        agent_id=os.environ.get("ABB_AGENT_ID", ""),
        session_id=os.environ.get("ABB_SESSION_ID", ""),
    )
