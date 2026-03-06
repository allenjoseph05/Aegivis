"""
LangChain callback adapter for Aegivis.

Automatically records tool calls, chain events, and agent actions
as AGENT_THOUGHT events in the Aegivis audit trail.

Install:
    pip install 'aegivis[langchain]'

Usage:
    import aegivis as abb
    from aegivis.adapters.langchain import AegivisLangChain

    with abb.session(agent_id="my-agent") as s:
        handler = AegivisLangChain(s)
        chain.invoke({"input": "..."}, config={"callbacks": [handler]})
"""
from __future__ import annotations

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "LangChain is not installed. Install with: pip install 'aegivis[langchain]'"
    )

from typing import Any, Union


class AegivisLangChain(BaseCallbackHandler):
    """
    LangChain BaseCallbackHandler that records events to Aegivis.

    Pass this as a callback to any LangChain chain, agent, or tool to
    automatically capture a structured audit trail.
    """

    def __init__(self, session: Any):
        """
        Args:
            session: An aegivis.Session instance
        """
        super().__init__()
        self._session = session

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        self._session.annotate(
            f"tool_start:{serialized.get('name', '?')}",
            {"input": str(input_str)[:200]},
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self._session.annotate(
            "tool_end",
            {"output": str(output)[:200]},
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        self._session.error(
            f"Tool error: {str(error)[:200]}",
            exc=error if isinstance(error, Exception) else None,
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self._session.annotate(
            f"chain_start:{serialized.get('name', '?')}",
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self._session.annotate("chain_end")

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        self._session.error(
            f"Chain error: {str(error)[:200]}",
            exc=error if isinstance(error, Exception) else None,
        )

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        self._session.annotate(
            f"agent_action:{action.tool}",
            {"log": str(action.log)[:500]},
        )

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        self._session.annotate(
            "agent_finish",
            {"output": str(finish.return_values)[:200]},
        )
