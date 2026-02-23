"""
AgentBlackBox Python SDK

Provides optional session enrichment for AI agents.
The proxy handles LLM call capture automatically -- the SDK adds
custom annotations and metadata.

Quick start:
    import agentblackbox as abb

    with abb.session(agent_id="my-agent") as s:
        s.annotate("Starting task")
        result = run_my_agent()

LangChain adapter (requires pip install 'agentblackbox[langchain]'):
    from agentblackbox.adapters.langchain import AgentBlackBoxLangChain
"""
from .session import Session, session

__version__ = "2.0.0"
__all__ = ["Session", "session"]

# Lazy import of LangChain adapter -- only available if langchain-core is installed
try:
    from .adapters.langchain import AgentBlackBoxLangChain
    __all__ = ["Session", "session", "AgentBlackBoxLangChain"]
except ImportError:
    pass
