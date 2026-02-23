"""
AgentBlackBox Python SDK - session context manager.

Provides optional enrichment for agents that want to annotate
sessions with custom events and metadata. Works alongside the proxy:
- Proxy captures all LLM calls automatically
- SDK adds application-level context (custom events, metadata, labels)

Usage:
    import agentblackbox as abb

    with abb.session(agent_id="contract-review-agent") as s:
        s.annotate("Starting contract review for client Acme Corp")
        result = await agent.run("Review contract #4821")
        s.annotate("Review complete", metadata={"contract_id": "4821"})
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BACKEND_URL = os.environ.get("ABB_BACKEND_URL", "http://localhost:8000")
_API_KEY = os.environ.get("ABB_BACKEND_API_KEY", "dev-proxy-key")
_ORG_ID = os.environ.get("ABB_ORG_ID", "default-org")


class Session:
    """Context for a single agent session. Provides annotation API."""

    def __init__(
        self,
        agent_id: str,
        session_id: str | None = None,
        metadata: dict | None = None,
    ):
        self.agent_id = agent_id
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.metadata = metadata or {}
        self._events: list[dict] = []
        self._started_at = time.time_ns()
        self._seq = 0
        self._last_hash = f"SDK_GENESIS_{self.session_id}"

        # Set env vars so the proxy picks up this session/agent ID
        os.environ["ABB_SESSION_ID"] = self.session_id
        os.environ["ABB_AGENT_ID"] = self.agent_id

        logger.debug(f"ABB session started: {self.session_id}")

    def _build_event(self, event_type: str, payload: dict) -> dict:
        """Build a properly chained event and append it to the internal buffer."""
        self._seq += 1
        event_without_hash = {
            "event_id": f"sdk_{uuid.uuid4().hex[:20]}",
            "schema_version": "1.0",
            "org_id": _ORG_ID,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "provider": "sdk",
            "model": "sdk",
            "interception_layer": "sdk",
            "run_id": str(uuid.uuid4()),
            "parent_run_id": None,
            "event_type": event_type,
            "payload": payload,
            "payload_hash": None,
            "pii_detected": [],
            "timestamp_ns": time.time_ns(),
            "sequence_number": self._seq,
            "previous_hash": self._last_hash,
        }
        current_hash = hashlib.sha256(
            json.dumps(event_without_hash, sort_keys=True, default=str).encode()
        ).hexdigest()
        event = {**event_without_hash, "current_hash": current_hash}
        self._last_hash = current_hash
        self._events.append(event)
        logger.debug(f"ABB event built: {event_type} seq={self._seq}")
        return event

    def annotate(self, message: str, metadata: dict | None = None, level: str = "info") -> dict:
        """
        Add a custom annotation event to this session.
        Stored as AGENT_THOUGHT event type, visible in the dashboard timeline.
        """
        return self._build_event("AGENT_THOUGHT", {
            "message": message,
            "level": level,
            "metadata": metadata or {},
        })

    def tag(self, key: str, value: str) -> dict:
        """
        Attach a key-value tag to this session for filtering/searching.

        Example:
            s.tag("environment", "production")
            s.tag("customer_tier", "enterprise")
        """
        return self._build_event("AGENT_THOUGHT", {
            "type": "tag",
            "tag_key": key,
            "tag_value": value,
        })

    def error(self, message: str, exc: Exception | None = None) -> dict:
        """
        Record an application-level error in the audit trail.

        Example:
            try:
                result = risky_operation()
            except ValueError as e:
                s.error("Risky operation failed", exc=e)
        """
        return self._build_event("SYSTEM_ERROR", {
            "error_message": message,
            "exception_type": type(exc).__name__ if exc else None,
            "interception_layer": "sdk",
            "provider": "sdk",
        })

    def _flush(self):
        """Send buffered events to backend (sync, called on context exit)."""
        if not self._events:
            return
        try:
            payload = {
                "events": self._events,
                "batch_id": str(uuid.uuid4()),
                "sent_at_ns": time.time_ns(),
            }
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    f"{_BACKEND_URL}/v1/events",
                    content=json.dumps(payload, default=str),
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": _API_KEY,
                    },
                )
            logger.debug(f"ABB flushed {len(self._events)} SDK events")
        except Exception as e:
            logger.warning(f"ABB SDK flush failed (non-critical): {e}")
        finally:
            self._events.clear()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_):
        self._flush()
        # Clean up env vars
        os.environ.pop("ABB_SESSION_ID", None)
        os.environ.pop("ABB_AGENT_ID", None)

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *_):
        self._flush()
        os.environ.pop("ABB_SESSION_ID", None)
        os.environ.pop("ABB_AGENT_ID", None)


def session(
    agent_id: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> Session:
    """
    Create a new AgentBlackBox session context.

    Args:
        agent_id: Human-readable identifier for this agent
        session_id: Optional explicit session ID (auto-generated if not provided)
        metadata: Optional key-value metadata attached to this session

    Returns:
        Session context manager (use with `with` or `async with`)

    Example:
        with abb.session(agent_id="my-agent") as s:
            s.annotate("Starting task")
            result = run_agent()
    """
    return Session(agent_id=agent_id, session_id=session_id, metadata=metadata)
