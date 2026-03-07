"""
AutoGen adapter for Aegivis.

Instruments AutoGen ``ConversableAgent`` instances (v0.2.x and v0.4.x) to emit
``AGENT_THOUGHT`` audit events for every message exchange.  Works by
monkey-patching ``generate_reply()`` (v0.2/v0.3) and, when present,
``on_messages()`` (v0.4 async API).

Install::

    pip install 'aegivis[autogen]'

Usage::

    import autogen
    from aegivis.adapters.autogen import instrument_agent, instrument_group_chat

    assistant = autogen.AssistantAgent("assistant", llm_config={...})
    instrument_agent(assistant, agent_id="assistant")

    # For GroupChat, instrument all agents at once:
    groupchat = autogen.GroupChat(agents=[assistant, user_proxy], messages=[])
    manager   = autogen.GroupChatManager(groupchat=groupchat, ...)
    instrument_group_chat(groupchat, manager, agent_id="myteam")
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import types
from typing import Any

logger = logging.getLogger(__name__)


def instrument_agent(
    agent: Any,
    *,
    agent_id: str = "",
    backend_url: str = "",
    api_key: str = "",
) -> Any:
    """
    Instrument an AutoGen ``ConversableAgent`` to emit Aegivis audit events.

    Patches ``agent.generate_reply()`` (v0.2/v0.3) and ``agent.on_messages()``
    (v0.4 async) to record every message exchange as an ``AGENT_THOUGHT`` event.
    The agent object is mutated in-place and returned for chaining.

    Parameters
    ----------
    agent       Any AutoGen ConversableAgent (v0.2, v0.3, or v0.4).
    agent_id    Event label. Defaults to ``agent.name`` when available.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.

    Returns
    -------
    The same agent (instrumented in-place).
    """
    _id = agent_id or getattr(agent, "name", "autogen-agent")
    _url = backend_url or os.environ.get("AEGIVIS_BACKEND_URL", "")
    _key = api_key or os.environ.get("AEGIVIS_API_KEY", "")

    # ── v0.2 / v0.3: sync generate_reply ──────────────────────────────────
    if hasattr(agent, "generate_reply") and callable(agent.generate_reply):
        _orig_sync = agent.generate_reply

        def _hooked_generate_reply(messages=None, sender=None, **kwargs):
            reply = _orig_sync(messages=messages, sender=sender, **kwargs)
            if reply is not None:
                _fire(_url, _key, "AGENT_THOUGHT", _id, {
                    "event": "generate_reply",
                    "sender": getattr(sender, "name", str(sender)) if sender else None,
                    "reply_preview": str(reply)[:500],
                    "message_count": len(messages) if messages else 0,
                })
            return reply

        agent.generate_reply = _hooked_generate_reply

    # ── v0.4: async on_messages ────────────────────────────────────────────
    _orig_async_fn = None
    if hasattr(agent, "on_messages"):
        raw = getattr(type(agent), "on_messages", None)
        if raw is not None and callable(raw):
            _orig_async_fn = raw

    if _orig_async_fn is not None:
        async def _hooked_on_messages(self_agent, messages, cancellation_token=None):
            result = await _orig_async_fn(self_agent, messages, cancellation_token)
            content = getattr(result, "chat_message", None)
            _fire(_url, _key, "AGENT_THOUGHT", _id, {
                "event": "on_messages",
                "message_count": len(messages) if messages else 0,
                "reply_preview": str(
                    getattr(content, "content", "")
                )[:500] if content else None,
            })
            return result

        agent.on_messages = types.MethodType(_hooked_on_messages, agent)

    return agent


def instrument_group_chat(
    group_chat: Any,
    manager: Any,
    *,
    agent_id: str = "groupchat",
    backend_url: str = "",
    api_key: str = "",
) -> None:
    """
    Instrument every agent in an AutoGen ``GroupChat`` plus its manager.

    Calls :func:`instrument_agent` on each member of ``group_chat.agents``
    and on the ``manager``.  Each agent gets an ID of ``{agent_id}/{agent.name}``
    and the manager gets ``{agent_id}/manager``.

    Parameters
    ----------
    group_chat  AutoGen ``GroupChat`` with a ``.agents`` list.
    manager     ``GroupChatManager`` instance (also instrumented).
    agent_id    ID prefix for all agents in this chat.
    backend_url Aegivis backend URL. Falls back to ``AEGIVIS_BACKEND_URL``.
    api_key     Aegivis API key. Falls back to ``AEGIVIS_API_KEY``.
    """
    for ag in getattr(group_chat, "agents", []):
        instrument_agent(
            ag,
            agent_id=f"{agent_id}/{getattr(ag, 'name', 'agent')}",
            backend_url=backend_url,
            api_key=api_key,
        )
    instrument_agent(
        manager,
        agent_id=f"{agent_id}/manager",
        backend_url=backend_url,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fire(url: str, key: str, event_type: str, agent_id: str, payload: dict) -> None:
    if not url:
        return
    event = {
        "event_type": event_type,
        "agent_id": agent_id,
        "timestamp_ns": time.time_ns(),
        "payload": payload,
    }
    headers = {"Content-Type": "application/json", "X-API-Key": key}

    def _post() -> None:
        try:
            body = json.dumps(event).encode()
            req = urllib.request.Request(
                url.rstrip("/") + "/v1/ingest",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("Aegivis fire-and-forget failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()
