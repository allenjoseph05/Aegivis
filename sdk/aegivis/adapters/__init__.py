"""
Aegivis SDK adapters for popular AI frameworks.

Available adapters
------------------
langchain
    ``AegivisLangChain`` — LangChain ``BaseCallbackHandler``.
    Records tool calls, chain events, and agent actions.
    Install: ``pip install 'aegivis[langchain]'``

langgraph
    ``AegivisLangGraph`` — extends ``AegivisLangChain`` with LangGraph
    custom-event and chat-model-start hooks.
    Install: ``pip install 'aegivis[langchain]'`` (same extra)

openai_agents
    ``AegivisTracingProcessor`` — OpenAI Agents SDK ``TracingProcessor``.
    Captures tool calls, LLM generations, and agent handoffs.
    Install: ``pip install 'aegivis[openai-agents]'``

autogen
    ``instrument_agent`` / ``instrument_group_chat`` — patches AutoGen
    ``ConversableAgent.generate_reply()`` (v0.2/v0.3) and
    ``on_messages()`` (v0.4) to emit ``AGENT_THOUGHT`` events.
    Install: ``pip install 'aegivis[autogen]'``

crewai
    ``AegivisCrewAICallback`` — CrewAI ``step_callback`` that emits
    ``AGENT_THOUGHT`` events for every agent step and task completion.
    Install: ``pip install 'aegivis[crewai]'``
"""
