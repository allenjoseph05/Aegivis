"""
Non-blocking event reporter for memory commit events.

Reports MEMORY_WRITE_BLOCKED and MEMORY_WRITE_SCANNED events to the
Aegivis backend. Fires httpx.post in a daemon thread — never
blocks the caller. Silent on network failure (logs to stderr only if
AEGIVIS_DEBUG=1 is set).
"""
from __future__ import annotations

import os
import threading
import time
import logging

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("AEGIVIS_DEBUG", "") == "1"

MEMORY_WRITE_BLOCKED = "MEMORY_WRITE_BLOCKED"
MEMORY_WRITE_SCANNED = "MEMORY_WRITE_SCANNED"


class MemoryEventReporter:
    """
    Async-safe, non-blocking event reporter.

    Each :meth:`report` call spawns a daemon thread that POSTs to the
    backend's ``/v1/ingest`` endpoint. The thread is daemon so it never
    prevents interpreter shutdown.
    """

    def __init__(self, backend_url: str = "", api_key: str = "") -> None:
        self.backend_url = backend_url.rstrip("/")
        self.api_key = api_key

    def report(
        self,
        event_type: str,
        text_preview: str,
        score: float,
        matched_phrases: list[str],
        agent_id: str = "",
        session_id: str = "",
    ) -> None:
        """
        Fire-and-forget: report an event to the backend.

        Does nothing if ``backend_url`` is empty.
        """
        if not self.backend_url:
            return

        event = {
            "event_type": event_type,
            "timestamp_ns": int(time.time() * 1_000_000_000),
            "session_id": session_id or "sdk-session",
            "agent_id": agent_id or "sdk-agent",
            "org_id": "default-org",
            "payload": {
                "text_preview": text_preview[:200],
                "score": score,
                "matched_phrases": matched_phrases[:5],
                "source": "memory_commit_validator",
            },
        }

        url = self.backend_url
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        def _post() -> None:
            try:
                import httpx  # noqa: PLC0415

                with httpx.Client(timeout=5.0) as client:
                    client.post(
                        f"{url}/v1/ingest",
                        json={"events": [event]},
                        headers=headers,
                    )
            except Exception as exc:
                if _DEBUG:
                    logger.warning("MemoryEventReporter: failed to post event: %s", exc)

        t = threading.Thread(target=_post, daemon=True)
        t.start()
