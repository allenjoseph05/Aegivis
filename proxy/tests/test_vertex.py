"""Tests for Phase E3 — Vertex AI provider."""
import json
import pytest
import respx
import httpx
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.app.providers.vertex import (
    VertexProvider,
    extract_location_from_path,
    extract_model_from_path,
    build_upstream_url,
)
from proxy.app.main import app


# ─── Transport mock (mirrors test_integration.py pattern) ─────────────────────

def _make_mock_transport():
    t = MagicMock()
    t.enqueue = MagicMock()
    t.enqueue_violation = MagicMock()
    t.buffer_status = MagicMock(return_value={"events": 0, "violations": 0})
    t.start = AsyncMock()
    t.stop = AsyncMock()
    return t


@pytest.fixture()
def vertex_client():
    """TestClient with mocked transport — same pattern as test_integration.py."""
    mock_transport = _make_mock_transport()

    async def _mock_get_best(*, force_http=False):  # noqa: ARG001
        return mock_transport

    with (
        patch("proxy.app.main.get_transport", return_value=mock_transport),
        patch("proxy.app.intercept.get_transport", return_value=mock_transport),
        patch("proxy.app.transport.get_best_transport", side_effect=_mock_get_best),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ─── Path parsing ─────────────────────────────────────────────────────────────

class TestExtractLocation:
    def test_standard_path(self):
        path = "/v1beta1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
        assert extract_location_from_path(path) == "us-central1"

    def test_eu_region(self):
        path = "/v1/projects/proj/locations/europe-west4/publishers/google/models/gemini-1.0-pro:generateContent"
        assert extract_location_from_path(path) == "europe-west4"

    def test_asia_region(self):
        path = "/v1beta1/projects/proj/locations/asia-northeast1/publishers/google/models/gemini-pro:generateContent"
        assert extract_location_from_path(path) == "asia-northeast1"

    def test_no_location_returns_none(self):
        assert extract_location_from_path("/v1beta1/projects/proj") is None

    def test_empty_path(self):
        assert extract_location_from_path("") is None


class TestExtractModel:
    def test_standard_model(self):
        path = "/v1beta1/projects/proj/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
        assert extract_model_from_path(path) == "gemini-1.5-pro"

    def test_model_with_version(self):
        path = ".../models/gemini-2.0-flash-exp:generateContent"
        assert extract_model_from_path(path) == "gemini-2.0-flash-exp"

    def test_no_model_returns_unknown(self):
        assert extract_model_from_path("/v1/health") == "unknown"


class TestBuildUpstreamUrl:
    def test_extracts_location_from_path(self):
        path = "/v1beta1/projects/proj/locations/us-east4/publishers/google/models/gemini:generateContent"
        url = build_upstream_url(path)
        assert url == "https://us-east4-aiplatform.googleapis.com"

    def test_falls_back_to_default_location(self):
        url = build_upstream_url("/v1/health", default_location="eu-west1")
        assert url == "https://eu-west1-aiplatform.googleapis.com"

    def test_default_location_is_us_central1(self):
        url = build_upstream_url("/v1/no-location")
        assert "us-central1" in url


# ─── Provider parsing — delegates to Google ──────────────────────────────────

class TestVertexProviderParsing:
    SAMPLE_REQUEST = {
        "contents": [{"role": "user", "parts": [{"text": "Hello from Vertex AI"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
    }

    SAMPLE_RESPONSE = {
        "candidates": [{
            "content": {"parts": [{"text": "Hello from Gemini on Vertex"}], "role": "model"},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 8,
            "totalTokenCount": 13,
        },
    }

    def test_extract_request_params(self):
        params = VertexProvider.extract_request_params(self.SAMPLE_REQUEST, model="gemini-1.5-pro")
        assert params["model"] == "gemini-1.5-pro"
        assert any(m["content"] == "Hello from Vertex AI" for m in params["messages"])

    def test_parse_response(self):
        parsed = VertexProvider.parse_response(self.SAMPLE_RESPONSE)
        assert parsed["response_text"] == "Hello from Gemini on Vertex"
        assert parsed["finish_reason"] == "stop"
        assert parsed["token_usage"]["total_tokens"] == 13

    def test_parse_response_with_tool_call(self):
        resp = {
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
        }
        parsed = VertexProvider.parse_response(resp)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"

    def test_parse_empty_response(self):
        parsed = VertexProvider.parse_response({})
        assert parsed["response_text"] is None
        assert parsed["finish_reason"] is None

    def test_parse_sse_chunk(self):
        chunk = json.dumps({
            "candidates": [{
                "content": {"parts": [{"text": "streaming chunk"}], "role": "model"},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"totalTokenCount": 10},
        })
        result = VertexProvider.parse_sse_chunk(chunk)
        assert result is not None
        assert result["content"] == "streaming chunk"

    def test_parse_empty_chunk(self):
        assert VertexProvider.parse_sse_chunk("") is None
        assert VertexProvider.parse_sse_chunk("  ") is None

    def test_new_assembler(self):
        assembler = VertexProvider.new_assembler()
        assert assembler is not None

    def test_assembler_builds_response(self):
        assembler = VertexProvider.new_assembler()
        chunk1 = {"content": "Hello ", "finish_reason": None, "usage": None, "thinking_blocks": []}
        chunk2 = {"content": "world", "finish_reason": "stop", "usage": {"total_tokens": 5}, "thinking_blocks": []}
        assembler.feed(chunk1)
        done = assembler.feed(chunk2)
        assert done is True
        resp = assembler.build_response()
        assert resp["response_text"] == "Hello world"
        assert resp["finish_reason"] == "stop"

    def test_provider_name(self):
        assert VertexProvider.name == "vertex"


# ─── HTTP route integration ───────────────────────────────────────────────────

class TestVertexRoute:
    """Integration tests for the /vertex/{path:path} FastAPI route."""

    VERTEX_PATH = "/v1beta1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
    UPSTREAM_URL = "https://us-central1-aiplatform.googleapis.com"

    FAKE_RESPONSE = {
        "candidates": [{
            "content": {"parts": [{"text": "4"}], "role": "model"},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
    }

    REQUEST_BODY = {
        "contents": [{"role": "user", "parts": [{"text": "What is 2+2?"}]}],
    }

    @respx.mock
    def test_generate_content_intercepted(self, vertex_client):
        respx.post(f"{self.UPSTREAM_URL}{self.VERTEX_PATH}").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        resp = vertex_client.post(
            f"/vertex{self.VERTEX_PATH}",
            json=self.REQUEST_BODY,
            headers={"Authorization": "Bearer fake-gcp-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "candidates" in body

    @respx.mock
    def test_non_generate_path_passes_through(self, vertex_client):
        """Non-generateContent paths (e.g. model listing) are passed through without interception."""
        list_path = "/v1beta1/projects/my-project/locations/us-central1/publishers/google/models"
        respx.get(f"{self.UPSTREAM_URL}{list_path}").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        resp = vertex_client.get(
            f"/vertex{list_path}",
            headers={"Authorization": "Bearer fake-gcp-token"},
        )
        assert resp.status_code == 200

    @respx.mock
    def test_session_id_returned_in_header(self, vertex_client):
        respx.post(f"{self.UPSTREAM_URL}{self.VERTEX_PATH}").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        resp = vertex_client.post(
            f"/vertex{self.VERTEX_PATH}",
            json=self.REQUEST_BODY,
        )
        assert resp.status_code == 200
        assert "x-aegivis-session-id" in resp.headers

    @respx.mock
    def test_upstream_5xx_forwarded(self, vertex_client):
        respx.post(f"{self.UPSTREAM_URL}{self.VERTEX_PATH}").mock(
            return_value=httpx.Response(503, json={"error": {"message": "Service Unavailable"}})
        )
        resp = vertex_client.post(
            f"/vertex{self.VERTEX_PATH}",
            json=self.REQUEST_BODY,
        )
        assert resp.status_code == 503

    @respx.mock
    def test_europe_region_routes_correctly(self, vertex_client):
        """Verify dynamic location extraction: europe-west4 → europe-west4-aiplatform.googleapis.com"""
        eu_path = "/v1beta1/projects/proj/locations/europe-west4/publishers/google/models/gemini-1.5-flash:generateContent"
        eu_upstream = "https://europe-west4-aiplatform.googleapis.com"
        respx.post(f"{eu_upstream}{eu_path}").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        resp = vertex_client.post(
            f"/vertex{eu_path}",
            json=self.REQUEST_BODY,
        )
        assert resp.status_code == 200
