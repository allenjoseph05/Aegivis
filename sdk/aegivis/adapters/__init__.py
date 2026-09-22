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

litellm
    ``AegivisLiteLLMCallback`` — LiteLLM ``CustomLogger`` that emits
    ``LLM_CALL_START`` / ``LLM_CALL_END`` / ``LLM_CALL_ERROR`` events.
    Covers CrewAI, LlamaIndex, Agno, Haystack and any litellm-based framework.
    Install: ``pip install 'aegivis[litellm]'``

pydantic_ai (Phase E6)
    ``instrument_agent`` — wraps PydanticAI ``Agent.run()`` / ``run_sync()``
    to emit ``TOOL_EXEC_END`` + ``AGENT_THOUGHT`` events after each run.
    Install: ``pip install 'aegivis[pydantic-ai]'``

google_adk (Phase E6)
    ``instrument_runner`` — wraps Google ADK ``Runner.run_async()`` to
    intercept ``FunctionCall`` / ``FunctionResponse`` events in real time.
    Install: ``pip install 'aegivis[google-adk]'``

smolagents (Phase E6)
    ``instrument_agent`` — wraps each tool in ``agent.tools`` and
    ``agent.run()`` to emit ``TOOL_EXEC_END`` / ``AGENT_THOUGHT`` events.
    Install: ``pip install 'aegivis[smolagents]'``

haystack (Phase E6)
    ``instrument_pipeline`` — wraps every Haystack v2 pipeline component's
    ``.run()`` to emit ``TOOL_EXEC_END`` events for each component step.
    Install: ``pip install 'aegivis[haystack]'``
"""
