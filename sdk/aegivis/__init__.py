"""
Aegivis Python SDK

Provides optional session enrichment for AI agents.
The proxy handles LLM call capture automatically -- the SDK adds
custom annotations and metadata.

Quick start::

    import aegivis as abb

    with abb.session(agent_id="my-agent") as s:
        s.annotate("Starting task")
        result = run_my_agent()

Tool instrumentation (captures actual tool execution, not just LLM intent)::

    from aegivis import instrument

    tools = instrument(tools, agent_id="my-agent")  # wrap a list

    @instrument.tool                                  # or a decorator
    def send_email(to: str, body: str) -> str: ...

Framework adapters
------------------
LangChain / LangGraph (pip install 'aegivis[langchain]')::

    from aegivis.adapters.langchain import AegivisLangChain
    from aegivis.adapters.langgraph import AegivisLangGraph

OpenAI Agents SDK (pip install 'aegivis[openai-agents]')::

    from agents import set_trace_processors
    from aegivis.adapters.openai_agents import AegivisTracingProcessor
    set_trace_processors([AegivisTracingProcessor(agent_id="my-agent")])

AutoGen (pip install 'aegivis[autogen]')::

    from aegivis.adapters.autogen import instrument_agent
    instrument_agent(assistant, agent_id="assistant")

CrewAI (pip install 'aegivis[crewai]')::

    from aegivis.adapters.crewai import AegivisCrewAICallback
    crew = Crew(..., step_callback=AegivisCrewAICallback(agent_id="crew"))
"""
from .session import Session, session
from .tools import instrument

__version__ = "2.0.0"
__all__ = ["Session", "session", "instrument"]

# Lazy import of LangChain adapter -- only available if langchain-core is installed
try:
    from .adapters.langchain import AegivisLangChain
    __all__ = ["Session", "session", "instrument", "AegivisLangChain"]
except ImportError:
    pass
