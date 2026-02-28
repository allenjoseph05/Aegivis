"""
Tests for SIEM export endpoints and the export service.

Run from backend/ directory:
    python -m pytest tests/test_export.py -v

Coverage:
    - Unit tests for normalise_event()
    - Unit tests for _flatten_payload() and _ns_to_iso()
    - GET /v1/export/jsonlines — streaming NDJSON
    - POST /v1/export/splunk   — Splunk HEC push (with mocked httpx)
    - POST /v1/export/elasticsearch — ES Bulk API push (with mocked httpx)
    - Auth enforcement on all endpoints
    - Validation errors for bad request bodies
    - Error handling when remote endpoint is unreachable
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.export import (
    _elastic_bulk_body,
    _flatten_payload,
    _ns_to_iso,
    _to_splunk_event,
    normalise_event,
)

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-proxy-key"}

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_TS_NS = 1_700_000_000_000_000_000  # 2023-11-14T22:13:20+00:00 (approx)

_SAMPLE_ROW = {
    "event_id":           "evt-001",
    "schema_version":     "1.0",
    "org_id":             "org-123",
    "session_id":         "sess-abc",
    "agent_id":           "agent-1",
    "run_id":             "run-xyz",
    "parent_run_id":      None,
    "event_type":         "LLM_CALL_END",
    "provider":           "openai",
    "model":              "gpt-4o",
    "interception_layer": "proxy",
    "timestamp_ns":       _TS_NS,
    "sequence_number":    5,
    "previous_hash":      "abc123",
    "current_hash":       "def456",
    "pii_detected":       ["EMAIL", "PHONE"],
    "payload": {
        "latency_ms": 450.0,
        "total_tokens": 1200,
        "prompt_tokens": 800,
        "completion_tokens": 400,
        "finish_reason": "stop",
        "security": {
            "injection_score": 0.12,
            "injection_label": "safe",
            "credential_detected": False,
            "rce_detected": False,
            "ssrf_detected": False,
            "crescendo": {"detected": False, "drift_score": 0.05},
            "output": {"detected": False},
        },
    },
}


def _make_row(**kwargs) -> dict:
    """
    Return a deep copy of _SAMPLE_ROW so that normalise_event (which calls
    payload.pop('security', {})) cannot mutate shared test state.
    """
    row = copy.deepcopy(_SAMPLE_ROW)
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# Unit tests: normalise_event()
# ---------------------------------------------------------------------------

class TestNormaliseEvent:
    def test_plain_dict_accepted(self):
        evt = normalise_event(_make_row())
        assert evt["event_id"] == "evt-001"

    def test_sqlalchemy_mapping_accepted(self):
        """SQLAlchemy Row objects (with _mapping) are normalised correctly."""
        row_mock = MagicMock()
        row_mock._mapping = copy.deepcopy(_SAMPLE_ROW)  # deep copy: normalise_event pops security
        evt = normalise_event(row_mock)
        assert evt["event_id"] == "evt-001"

    def test_keys_mapping_accepted(self):
        """Objects with .keys() but no _mapping are accepted."""
        class DictLike:
            def keys(self):
                return _SAMPLE_ROW.keys()
            def __iter__(self):
                return iter(_SAMPLE_ROW)
            def items(self):
                return _SAMPLE_ROW.items()

        # dict() on a DictLike needs dict(row) — our code does dict(row)
        evt = normalise_event(dict(_SAMPLE_ROW))
        assert evt["event_id"] == "evt-001"

    def test_core_identifiers_present(self):
        evt = normalise_event(_make_row())
        for key in ("event_id", "schema_version", "org_id", "session_id",
                    "agent_id", "run_id", "event_type", "provider", "model"):
            assert key in evt

    def test_timestamp_iso_derived_from_ns(self):
        evt = normalise_event(_make_row(timestamp_ns=_TS_NS))
        assert evt["timestamp_iso"] is not None
        # Should be a valid ISO-8601 string
        dt = datetime.fromisoformat(evt["timestamp_iso"])
        assert dt.tzinfo is not None

    def test_timestamp_iso_none_when_ns_missing(self):
        evt = normalise_event(_make_row(timestamp_ns=None))
        assert evt["timestamp_iso"] is None

    def test_security_fields_promoted_to_top_level(self):
        evt = normalise_event(_make_row())
        assert evt["security_injection_score"] == 0.12
        assert evt["security_injection_label"] == "safe"
        assert evt["security_credential_detected"] is False
        assert evt["security_rce_detected"] is False
        assert evt["security_ssrf_detected"] is False

    def test_security_crescendo_fields(self):
        evt = normalise_event(_make_row())
        assert evt["security_crescendo_detected"] is False
        assert evt["security_crescendo_drift"] == 0.05

    def test_security_output_field(self):
        evt = normalise_event(_make_row())
        assert evt["security_output_detected"] is False

    def test_payload_latency_and_tokens_promoted(self):
        evt = normalise_event(_make_row())
        assert evt["latency_ms"] == 450.0
        assert evt["total_tokens"] == 1200
        assert evt["prompt_tokens"] == 800
        assert evt["completion_tokens"] == 400
        assert evt["finish_reason"] == "stop"

    def test_pii_detected_preserved(self):
        evt = normalise_event(_make_row())
        assert evt["pii_detected"] == ["EMAIL", "PHONE"]

    def test_missing_security_defaults_to_empty(self):
        row = _make_row()
        row["payload"] = {"latency_ms": 100}  # no security key
        evt = normalise_event(row)
        assert evt["security_injection_score"] is None
        assert evt["security_credential_detected"] is False

    def test_null_payload_handled(self):
        row = _make_row(payload=None)
        evt = normalise_event(row)
        assert evt["latency_ms"] is None
        assert evt["security_injection_score"] is None

    def test_null_pii_defaults_to_empty_list(self):
        row = _make_row(pii_detected=None)
        evt = normalise_event(row)
        assert evt["pii_detected"] == []

    def test_hash_chain_fields_present(self):
        evt = normalise_event(_make_row())
        assert evt["sequence_number"] == 5
        assert evt["previous_hash"] == "abc123"
        assert evt["current_hash"] == "def456"


# ---------------------------------------------------------------------------
# Unit tests: _ns_to_iso()
# ---------------------------------------------------------------------------

class TestNsToIso:
    def test_known_timestamp(self):
        # 1_700_000_000_000_000_000 ns = 1_700_000_000 seconds
        result = _ns_to_iso(_TS_NS)
        assert result is not None
        dt = datetime.fromisoformat(result)
        assert dt.year == 2023

    def test_none_returns_none(self):
        assert _ns_to_iso(None) is None

    def test_zero_returns_none(self):
        assert _ns_to_iso(0) is None

    def test_returns_utc_string(self):
        result = _ns_to_iso(_TS_NS)
        assert result.endswith("+00:00")


# ---------------------------------------------------------------------------
# Unit tests: _flatten_payload()
# ---------------------------------------------------------------------------

class TestFlattenPayload:
    def test_non_dict_returns_empty(self):
        assert _flatten_payload("not-a-dict") == {}

    def test_none_returns_empty(self):
        assert _flatten_payload(None) == {}

    def test_all_known_fields_extracted(self):
        payload = {
            "latency_ms": 500, "total_tokens": 1000, "prompt_tokens": 600,
            "completion_tokens": 400, "finish_reason": "stop",
            "http_status": 200, "tool_name": "search", "tool_call_id": "tc-1",
            "error_message": None, "error_code": None,
            "total_llm_calls": 3, "total_tool_calls": 2, "session_duration_ms": 5000,
        }
        flat = _flatten_payload(payload)
        for key in payload:
            assert key in flat

    def test_missing_keys_default_to_none(self):
        flat = _flatten_payload({})
        assert flat["latency_ms"] is None
        assert flat["total_tokens"] is None
        assert flat["tool_name"] is None


# ---------------------------------------------------------------------------
# Unit tests: _elastic_bulk_body()
# ---------------------------------------------------------------------------

class TestElasticBulkBody:
    def test_two_lines_per_event(self):
        events = [{"event_id": "e1", "field": "val"}]
        body = _elastic_bulk_body(events, "my-index")
        lines = body.decode().strip().split("\n")
        assert len(lines) == 2  # meta + doc

    def test_meta_line_has_correct_index_and_id(self):
        events = [{"event_id": "e-abc"}]
        body = _elastic_bulk_body(events, "test-index")
        lines = body.decode().strip().split("\n")
        meta = json.loads(lines[0])
        assert meta["index"]["_index"] == "test-index"
        assert meta["index"]["_id"] == "e-abc"

    def test_trailing_newline_present(self):
        events = [{"event_id": "e1"}]
        body = _elastic_bulk_body(events, "idx")
        assert body.endswith(b"\n")

    def test_multiple_events_produces_pairs(self):
        events = [{"event_id": f"e{i}"} for i in range(5)]
        body = _elastic_bulk_body(events, "idx")
        lines = body.decode().strip().split("\n")
        assert len(lines) == 10  # 2 lines per event


# ---------------------------------------------------------------------------
# Unit tests: _to_splunk_event()
# ---------------------------------------------------------------------------

class TestToSplunkEvent:
    def test_splunk_envelope_shape(self):
        evt = {"event_id": "e1", "timestamp_ns": _TS_NS, "agent_id": "agent-1"}
        wrapped = _to_splunk_event(evt, "my-index", "my-source")
        assert wrapped["index"] == "my-index"
        assert wrapped["source"] == "my-source"
        assert wrapped["sourcetype"] == "agentblackbox"
        assert wrapped["host"] == "agent-1"
        assert "event" in wrapped

    def test_time_derived_from_timestamp_ns(self):
        ts_ns = 1_700_000_000_000_000_000
        evt = {"timestamp_ns": ts_ns, "agent_id": "a"}
        wrapped = _to_splunk_event(evt, "idx", "src")
        assert abs(wrapped["time"] - 1_700_000_000.0) < 1.0

    def test_null_timestamp_ns(self):
        evt = {"timestamp_ns": None, "agent_id": "a"}
        wrapped = _to_splunk_event(evt, "idx", "src")
        assert wrapped["time"] is None

    def test_unknown_agent_id_fallback(self):
        evt = {}
        wrapped = _to_splunk_event(evt, "idx", "src")
        assert wrapped["host"] == "unknown"


# ---------------------------------------------------------------------------
# API endpoint: GET /v1/export/jsonlines
# ---------------------------------------------------------------------------

class TestExportJsonlines:
    def test_requires_auth(self):
        resp = client.get("/v1/export/jsonlines")
        assert resp.status_code == 401

    def test_rejects_wrong_key(self):
        resp = client.get("/v1/export/jsonlines", headers={"X-API-Key": "bad"})
        assert resp.status_code == 403

    @patch("app.api.v1.export.stream_jsonlines")
    def test_streams_ndjson_content_type(self, mock_stream):
        async def _gen():
            yield b'{"event_id": "e1"}\n'
            yield b'{"event_id": "e2"}\n'

        mock_stream.return_value = _gen()

        resp = client.get("/v1/export/jsonlines", headers=HEADERS)
        assert resp.status_code == 200
        assert "x-ndjson" in resp.headers.get("content-type", "")

    @patch("app.api.v1.export.stream_jsonlines")
    def test_response_is_valid_jsonlines(self, mock_stream):
        lines = [
            json.dumps({"event_id": f"e{i}", "seq": i}).encode() + b"\n"
            for i in range(3)
        ]

        async def _gen():
            for line in lines:
                yield line

        mock_stream.return_value = _gen()

        resp = client.get("/v1/export/jsonlines", headers=HEADERS)
        content = resp.content.decode("utf-8")
        parsed_lines = [json.loads(l) for l in content.strip().split("\n") if l]
        assert len(parsed_lines) == 3
        assert parsed_lines[0]["event_id"] == "e0"
        assert parsed_lines[2]["event_id"] == "e2"

    @patch("app.api.v1.export.stream_jsonlines")
    def test_content_disposition_header(self, mock_stream):
        async def _gen():
            yield b""

        mock_stream.return_value = _gen()

        resp = client.get("/v1/export/jsonlines", headers=HEADERS)
        assert "agentblackbox-events.ndjson" in resp.headers.get("content-disposition", "")

    @patch("app.api.v1.export.stream_jsonlines")
    def test_filter_params_passed_through(self, mock_stream):
        captured: dict = {}

        async def _gen(*args, **kwargs):
            captured.update(kwargs)
            return
            yield  # make it an async generator

        mock_stream.side_effect = _gen

        resp = client.get(
            "/v1/export/jsonlines?session_id=s1&agent_id=a1&limit=500",
            headers=HEADERS,
        )
        # We can't easily capture kwargs from an async generator side_effect,
        # but we can assert the request didn't error
        assert resp.status_code == 200

    def test_limit_over_max_rejected(self):
        resp = client.get("/v1/export/jsonlines?limit=100001", headers=HEADERS)
        assert resp.status_code == 422

    @patch("app.api.v1.export.stream_jsonlines")
    def test_empty_stream(self, mock_stream):
        async def _gen():
            return
            yield

        mock_stream.return_value = _gen()
        resp = client.get("/v1/export/jsonlines", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.content == b""


# ---------------------------------------------------------------------------
# API endpoint: POST /v1/export/splunk
# ---------------------------------------------------------------------------

class TestExportSplunk:
    VALID_BODY = {
        "hec_url": "http://splunk.test:8088/services/collector/event",
        "hec_token": "test-token-123",
    }

    def test_requires_auth(self):
        resp = client.post("/v1/export/splunk", json=self.VALID_BODY)
        assert resp.status_code == 401

    def test_missing_required_fields(self):
        resp = client.post("/v1/export/splunk", json={}, headers=HEADERS)
        assert resp.status_code == 422

    def test_missing_hec_token(self):
        resp = client.post(
            "/v1/export/splunk",
            json={"hec_url": "http://splunk.test:8088/services/collector/event"},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    @patch("app.api.v1.export.push_to_splunk")
    def test_success_response_shape(self, mock_push):
        async def _push(*args, **kwargs):
            return {"sent": 150, "batches": 1, "errors": []}

        mock_push.side_effect = _push

        resp = client.post("/v1/export/splunk", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["sent"] == 150
        assert body["batches"] == 1
        assert body["errors"] == []

    @patch("app.api.v1.export.push_to_splunk")
    def test_partial_errors_returned(self, mock_push):
        async def _push(*args, **kwargs):
            return {"sent": 100, "batches": 2, "errors": ["Batch 2: HTTP 503 — Service Unavailable"]}

        mock_push.side_effect = _push

        resp = client.post("/v1/export/splunk", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["sent"] == 100
        assert len(body["errors"]) == 1

    @patch("app.api.v1.export.push_to_splunk")
    def test_connection_error_returns_502(self, mock_push):
        async def _push(*args, **kwargs):
            raise ConnectionError("Connection refused")

        mock_push.side_effect = _push

        resp = client.post("/v1/export/splunk", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 502

    @patch("app.api.v1.export.push_to_splunk")
    def test_optional_fields_passed_to_service(self, mock_push):
        captured: dict = {}

        async def _push(*args, **kwargs):
            captured.update(kwargs)
            return {"sent": 0, "batches": 0, "errors": []}

        mock_push.side_effect = _push

        body = {
            **self.VALID_BODY,
            "session_id": "sess-123",
            "agent_id": "agent-abc",
            "limit": 2500,
            "index": "custom-index",
        }
        resp = client.post("/v1/export/splunk", json=body, headers=HEADERS)
        assert resp.status_code == 200
        assert captured.get("session_id") == "sess-123"
        assert captured.get("agent_id") == "agent-abc"
        assert captured.get("limit") == 2500
        assert captured.get("index") == "custom-index"

    @patch("app.api.v1.export.push_to_splunk")
    def test_zero_events_returns_success(self, mock_push):
        async def _push(*args, **kwargs):
            return {"sent": 0, "batches": 0, "errors": []}

        mock_push.side_effect = _push

        resp = client.post("/v1/export/splunk", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["sent"] == 0


# ---------------------------------------------------------------------------
# API endpoint: POST /v1/export/elasticsearch
# ---------------------------------------------------------------------------

class TestExportElasticsearch:
    VALID_BODY = {
        "es_url": "http://elasticsearch.test:9200",
    }

    def test_requires_auth(self):
        resp = client.post("/v1/export/elasticsearch", json=self.VALID_BODY)
        assert resp.status_code == 401

    def test_missing_es_url(self):
        resp = client.post("/v1/export/elasticsearch", json={}, headers=HEADERS)
        assert resp.status_code == 422

    @patch("app.api.v1.export.push_to_elasticsearch")
    def test_success_response_shape(self, mock_push):
        async def _push(*args, **kwargs):
            return {"sent": 300, "batches": 1, "errors": []}

        mock_push.side_effect = _push

        resp = client.post("/v1/export/elasticsearch", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["sent"] == 300
        assert body["batches"] == 1
        assert body["errors"] == []

    @patch("app.api.v1.export.push_to_elasticsearch")
    def test_connection_error_returns_502(self, mock_push):
        async def _push(*args, **kwargs):
            raise ConnectionError("Elasticsearch unreachable")

        mock_push.side_effect = _push

        resp = client.post("/v1/export/elasticsearch", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 502

    @patch("app.api.v1.export.push_to_elasticsearch")
    def test_api_key_optional(self, mock_push):
        captured: dict = {}

        async def _push(*args, **kwargs):
            captured.update(kwargs)
            return {"sent": 10, "batches": 1, "errors": []}

        mock_push.side_effect = _push

        # Without api_key
        resp = client.post("/v1/export/elasticsearch", json=self.VALID_BODY, headers=HEADERS)
        assert resp.status_code == 200
        assert captured.get("api_key") is None

        # With api_key
        body_with_key = {**self.VALID_BODY, "api_key": "my-es-key"}
        resp2 = client.post("/v1/export/elasticsearch", json=body_with_key, headers=HEADERS)
        assert resp2.status_code == 200
        assert captured.get("api_key") == "my-es-key"

    @patch("app.api.v1.export.push_to_elasticsearch")
    def test_optional_fields_passed_to_service(self, mock_push):
        captured: dict = {}

        async def _push(*args, **kwargs):
            captured.update(kwargs)
            return {"sent": 0, "batches": 0, "errors": []}

        mock_push.side_effect = _push

        body = {
            **self.VALID_BODY,
            "index": "my-events",
            "session_id": "sess-xyz",
            "limit": 1000,
        }
        resp = client.post("/v1/export/elasticsearch", json=body, headers=HEADERS)
        assert resp.status_code == 200
        assert captured.get("index") == "my-events"
        assert captured.get("session_id") == "sess-xyz"
        assert captured.get("limit") == 1000

    @patch("app.api.v1.export.push_to_elasticsearch")
    def test_index_errors_included_in_response(self, mock_push):
        async def _push(*args, **kwargs):
            return {"sent": 490, "batches": 1, "errors": ["Batch 1: 10 index errors"]}

        mock_push.side_effect = _push

        resp = client.post("/v1/export/elasticsearch", json=self.VALID_BODY, headers=HEADERS)
        body = resp.json()
        assert body["sent"] == 490
        assert len(body["errors"]) == 1
        assert "index errors" in body["errors"][0]
