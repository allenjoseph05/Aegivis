"""
LangGraph adapter for Aegivis.

Extends :class:`AegivisLangChain` with LangGraph-specific callback hooks:

- ``on_custom_event`` — captures events dispatched from graph nodes via
  ``adispatch_custom_event`` (LangGraph streaming events API, v2).
- ``on_chat_model_start`` — records chat model invocations within nodes,
  including the model name.

All standard LangChain hooks (tool start/end, chain start/end, agent actions)
are inherited from :class:`AegivisLangChain` and work without modification in
LangGraph graphs, because LangGraph uses the same callback system.

Requires::

    pip install 'aegivis[langchain]'   # same extra — no additional deps

Usage::

    import aegivis as abb
    from aegivis.adapters.langgraph import AegivisLangGraph

    with abb.session(agent_id="my-graph") as s:
        handler = AegivisLangGraph(s)
        # Sync or async invocation:
        result = graph.invoke(inputs, config={"callbacks": [handler]})
        # Streaming:
        async for chunk in graph.astream(inputs, config={"callbacks": [handler]}):
            ...

Note
----
For ``graph.astream_events(inputs, version="v2")`` the handler can be passed
as a regular callback in the run config.  ``on_custom_event`` will fire for
every ``dispatch_custom_event`` call inside a node.
"""
from __future__ import annotations

from typing import Any

try:
    from aegivis.adapters.langchain import AegivisLangChain
except ImportError:
    raise ImportError(
        "langchain-core is not installed. Install with: pip install 'aegivis[langchain]'"
    )


class AegivisLangGraph(AegivisLangChain):
    """
    LangGraph callback handler that records graph node events to Aegivis.

    Inherits all tool, chain, and agent hooks from
    :class:`~aegivis.adapters.langchain.AegivisLangChain` and adds:

    - **Custom events** dispatched via ``adispatch_custom_event`` inside
      graph nodes → ``AGENT_THOUGHT`` with ``event_name`` and ``data``.
    - **Chat model starts** within nodes → ``AGENT_THOUGHT`` with model name
      and message count.

    Compatible with compiled ``StateGraph``, ``MessageGraph``, and any
    LangGraph graph that uses the standard LCEL callback interface.
    """

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Any = None,  # noqa: ARG002
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """
        Handle a LangGraph custom event dispatched via ``adispatch_custom_event``.

        Records the event as an ``AGENT_THOUGHT`` annotation including the
        event name, serialised data (truncated to 500 chars), and any tags.
        """
        self._session.annotate(
            f"langgraph:{name}",
            {
                "data": str(data)[:500] if data is not None else None,
                "tags": tags or [],
            },
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        *,
        run_id: Any = None,  # noqa: ARG002
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """
        Record a chat model invocation within a graph node.

        Extracts the model name from the serialised config and counts the
        total messages passed to the model.
        """
        model_name = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("name", "unknown_model")
        )
        # messages is a list of lists (one per run) in LCEL callbacks
        msg_count = sum(
            len(turn) if isinstance(turn, list) else 1
            for turn in messages
        )
        self._session.annotate(
            f"chat_model_start:{model_name}",
            {"message_count": msg_count, "tags": tags or []},
        )
