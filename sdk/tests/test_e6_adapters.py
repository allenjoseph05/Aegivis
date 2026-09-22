"""
Tests for Phase E6 framework adapters.

All tests run without the target frameworks installed — they use plain Python
objects / MagicMocks to verify the adapter's monkey-patching logic, event
firing, and graceful error handling.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch


# ─── PydanticAI adapter ────────────────────────────────────────────────────────

class TestPydanticAIAdapter:
    def _make_agent(self, sync_result=None, async_result=None):
        """Build a minimal mock PydanticAI-like Agent."""
        class MockPart:
            pass

        class ToolCallPart(MockPart):
            def __init__(self, name, args, call_id):
                self.tool_name = name
                self.args = args
                self.tool_call_id = call_id

        class ToolReturnPart(MockPart):
            def __init__(self, content, call_id):
                self.content = content
                self.tool_call_id = call_id

        class TextPart(MockPart):
            def __init__(self, text):
                self.content = text

        class MockMessage:
            def __init__(self, parts):
                self.parts = parts

        class MockResult:
            def all_messages(self):
                return [
                    MockMessage([ToolCallPart("search", {"q": "Paris"}, "c1")]),
                    MockMessage([ToolReturnPart("Paris is the capital.", "c1")]),
                    MockMessage([TextPart("France's capital is Paris.")]),
                ]

        class MockAgent:
            model = "gpt-4o"

            def run_sync(self, prompt, **kwargs):
                return MockResult()

            async def run(self, prompt, **kwargs):
                return MockResult()

        return MockAgent(), MockResult

    def test_instrument_returns_agent(self):
        from aegivis.adapters.pydantic_ai import instrument_agent
        agent, _ = self._make_agent()
        result = instrument_agent(agent, agent_id="test-pa")
        assert result is agent

    def test_sync_run_fires_events(self):
        from aegivis.adapters.pydantic_ai import instrument_agent, _fire
        fired = []

        agent, _ = self._make_agent()
        with patch("aegivis.adapters.pydantic_ai._fire", side_effect=lambda *a, **k: fired.append(a)):
            instrument_agent(agent, agent_id="pa-agent", backend_url="http://localhost")
            agent.run_sync("What is the capital of France?")

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_END" in event_types
        assert "AGENT_THOUGHT" in event_types

    def test_async_run_fires_events(self):
        from aegivis.adapters.pydantic_ai import instrument_agent
        fired = []

        agent, _ = self._make_agent()
        with patch("aegivis.adapters.pydantic_ai._fire", side_effect=lambda *a, **k: fired.append(a)):
            instrument_agent(agent, agent_id="pa-agent", backend_url="http://localhost")
            asyncio.run(agent.run("What is the capital of France?"))

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_END" in event_types

    def test_no_url_no_http_call(self):
        from aegivis.adapters.pydantic_ai import instrument_agent
        import urllib.request

        agent, _ = self._make_agent()
        instrument_agent(agent, agent_id="pa-agent", backend_url="")

        # _fire is called but must not open any URL connection when url is empty
        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            agent.run_sync("hello")
            # Give daemon threads a moment to run
            time.sleep(0.05)
            mock_urlopen.assert_not_called()

    def test_fire_does_not_block_without_url(self):
        from aegivis.adapters.pydantic_ai import _fire
        # Should return without error when url is empty
        _fire("", "key", "TOOL_EXEC_END", "agent", {"x": 1})


# ─── Google ADK adapter ────────────────────────────────────────────────────────

class TestGoogleADKAdapter:
    def _make_runner(self, events):
        """Build a minimal mock ADK Runner."""
        class MockPart:
            def __init__(self, fn_call=None, fn_response=None, text=None):
                self.function_call = fn_call
                self.function_response = fn_response
                self.text = text

        class MockFunctionCall:
            def __init__(self, name, args):
                self.name = name
                self.args = args

        class MockFunctionResponse:
            def __init__(self, name, response):
                self.name = name
                self.response = response

        class MockContent:
            def __init__(self, parts):
                self.parts = parts

        class MockEvent:
            def __init__(self, content):
                self.content = content

        class MockRunner:
            app_name = "test-app"

            async def run_async(self, *args, **kwargs):
                for ev in events:
                    yield ev

        call_event = MockEvent(MockContent([
            MockPart(fn_call=MockFunctionCall("web_search", {"q": "Paris"}))
        ]))
        return_event = MockEvent(MockContent([
            MockPart(fn_response=MockFunctionResponse("web_search", {"result": "Paris"}))
        ]))
        text_event = MockEvent(MockContent([
            MockPart(text="Paris is the capital of France.")
        ]))

        runner = MockRunner()
        return runner, [call_event, return_event, text_event]

    def test_instrument_returns_runner(self):
        from aegivis.adapters.google_adk import instrument_runner
        runner, events = self._make_runner([])
        result = instrument_runner(runner, agent_id="adk-agent")
        assert result is runner

    def test_run_async_fires_tool_exec_end(self):
        from aegivis.adapters.google_adk import instrument_runner
        fired = []

        runner, events = self._make_runner(events=[])

        class FakeRunner:
            app_name = "test-app"

            async def run_async(self2, *args, **kwargs):
                for ev in events:
                    yield ev

        runner = FakeRunner()
        with patch("aegivis.adapters.google_adk._fire",
                   side_effect=lambda *a, **k: fired.append(a)):
            instrument_runner(runner, agent_id="adk", backend_url="http://localhost")

            async def _collect():
                results = []
                async for ev in runner.run_async():
                    results.append(ev)
                return results

            asyncio.run(_collect())

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_END" in event_types
        assert "AGENT_THOUGHT" in event_types

    def test_no_url_skips_fire(self):
        from aegivis.adapters.google_adk import _fire
        _fire("", "key", "TOOL_EXEC_END", "adk", {})  # must not raise


# ─── Smolagents adapter ────────────────────────────────────────────────────────

class TestSmolagentsAdapter:
    def _make_agent(self):
        class MockTool:
            def __call__(self, query):
                return f"result for {query}"

        class MockAgent:
            def __init__(self):
                self.tools = {"web_search": MockTool()}

            def run(self, task):
                # Simulate internal tool call
                self.tools["web_search"](task)
                return f"Answer about {task}"

        return MockAgent()

    def test_instrument_returns_agent(self):
        from aegivis.adapters.smolagents import instrument_agent
        agent = self._make_agent()
        result = instrument_agent(agent, agent_id="smol")
        assert result is agent

    def test_tool_call_fires_event(self):
        from aegivis.adapters.smolagents import instrument_agent
        fired = []

        agent = self._make_agent()
        with patch("aegivis.adapters.smolagents._fire",
                   side_effect=lambda *a, **k: fired.append(a)):
            instrument_agent(agent, agent_id="smol", backend_url="http://localhost")
            agent.run("weather in Paris")

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_END" in event_types
        assert "AGENT_THOUGHT" in event_types

    def test_tool_error_fires_exec_error(self):
        from aegivis.adapters.smolagents import instrument_agent
        fired = []

        class FailingTool:
            def __call__(self, q):
                raise RuntimeError("network down")

        class MockAgent:
            def __init__(self):
                self.tools = {"failing_tool": FailingTool()}
            def run(self, task):
                return "done"

        agent = MockAgent()
        with patch("aegivis.adapters.smolagents._fire",
                   side_effect=lambda *a, **k: fired.append(a)):
            instrument_agent(agent, agent_id="smol", backend_url="http://localhost")
            try:
                agent.tools["failing_tool"]("test")
            except RuntimeError:
                pass

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_ERROR" in event_types

    def test_empty_toolbox_no_error(self):
        from aegivis.adapters.smolagents import instrument_agent

        class AgentNoTools:
            tools = {}
            def run(self, t): return "ok"

        agent = AgentNoTools()
        instrument_agent(agent, agent_id="smol")  # must not raise

    def test_no_url_skips_fire(self):
        from aegivis.adapters.smolagents import _fire
        _fire("", "key", "TOOL_EXEC_END", "smol", {})


# ─── Haystack adapter ─────────────────────────────────────────────────────────

class TestHaystackAdapter:
    def _make_pipeline(self):
        class MockComponent:
            name = "retriever"
            _aegivis_instrumented = False

            def run(self, query=""):
                return {"documents": ["doc1", "doc2"]}

        class MockPipeline:
            def __init__(self):
                comp = MockComponent()
                self.components = {"retriever": comp}
                self.graph = None  # no networkx

            def run(self, inputs):
                results = {}
                for name, comp in self.components.items():
                    results[name] = comp.run(**inputs.get(name, {}))
                return results

        return MockPipeline()

    def test_instrument_returns_pipeline(self):
        from aegivis.adapters.haystack import instrument_pipeline
        pipeline = self._make_pipeline()
        result = instrument_pipeline(pipeline, agent_id="hay")
        assert result is pipeline

    def test_component_run_fires_event(self):
        from aegivis.adapters.haystack import instrument_pipeline
        fired = []

        pipeline = self._make_pipeline()
        with patch("aegivis.adapters.haystack._fire",
                   side_effect=lambda *a, **k: fired.append(a)):
            instrument_pipeline(pipeline, agent_id="hay", backend_url="http://localhost")
            pipeline.run({"retriever": {"query": "Paris"}})

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_END" in event_types
        assert "AGENT_THOUGHT" in event_types

    def test_component_error_fires_exec_error(self):
        from aegivis.adapters.haystack import instrument_pipeline
        fired = []

        class FailingComp:
            _aegivis_instrumented = False
            def run(self, **kwargs):
                raise RuntimeError("store offline")

        class MockPipeline:
            components = {"failing": FailingComp()}
            graph = None
            def run(self, inputs): return {}

        pipeline = MockPipeline()
        with patch("aegivis.adapters.haystack._fire",
                   side_effect=lambda *a, **k: fired.append(a)):
            instrument_pipeline(pipeline, agent_id="hay", backend_url="http://localhost")
            try:
                pipeline.components["failing"].run()
            except RuntimeError:
                pass

        event_types = [f[2] for f in fired]
        assert "TOOL_EXEC_ERROR" in event_types

    def test_idempotent_instrumentation(self):
        """Instrumenting the same pipeline twice should not double-wrap components."""
        from aegivis.adapters.haystack import instrument_pipeline
        fired = []

        pipeline = self._make_pipeline()
        with patch("aegivis.adapters.haystack._fire",
                   side_effect=lambda *a, **k: fired.append(a)):
            instrument_pipeline(pipeline, agent_id="hay", backend_url="http://localhost")
            instrument_pipeline(pipeline, agent_id="hay", backend_url="http://localhost")
            pipeline.run({"retriever": {"query": "test"}})

        # Should fire exactly once per component, not twice
        tool_exec_ends = [f for f in fired if f[2] == "TOOL_EXEC_END"]
        assert len(tool_exec_ends) == 1

    def test_no_url_skips_fire(self):
        from aegivis.adapters.haystack import _fire
        _fire("", "key", "TOOL_EXEC_END", "hay", {})
