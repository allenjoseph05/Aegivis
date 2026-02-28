"""
Tests for GET /v1/topology endpoint and the topology service.

Run from backend/ directory:
    python -m pytest tests/test_topology.py -v

Coverage:
    - Unit tests for _compute_risk()
    - API auth enforcement
    - Full topology response shape and content
    - include_isolated and min_edge_calls query params
    - Risk level boundaries
    - Empty database (no agents, no edges)
    - Isolated agent filtering
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.topology import (
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
    _compute_risk,
)

client = TestClient(app)
HEADERS = {"X-API-Key": "dev-proxy-key"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_node(
    agent_id: str = "agent-1",
    session_count: int = 5,
    llm_call_count: int = 20,
    tool_call_count: int = 10,
    error_count: int = 1,
    violation_count: int = 0,
    anomaly_count: int = 0,
    pii_event_count: int = 0,
    injection_score_max: float = 0.0,
    risk_score: float = 0.0,
    risk_level: str = "low",
    providers: list[str] | None = None,
    models: list[str] | None = None,
) -> TopologyNode:
    return TopologyNode(
        agent_id            = agent_id,
        session_count       = session_count,
        llm_call_count      = llm_call_count,
        tool_call_count     = tool_call_count,
        error_count         = error_count,
        violation_count     = violation_count,
        anomaly_count       = anomaly_count,
        pii_event_count     = pii_event_count,
        injection_score_max = injection_score_max,
        risk_score          = risk_score,
        risk_level          = risk_level,
        first_seen          = _NOW.isoformat(),
        last_seen           = _NOW.isoformat(),
        providers           = providers or ["openai"],
        models              = models or ["gpt-4o"],
    )


def _make_edge(source: str = "agent-1", target: str = "agent-2", call_count: int = 5) -> TopologyEdge:
    return TopologyEdge(
        source         = source,
        target         = target,
        call_count     = call_count,
        avg_latency_ms = 123.45,
        first_seen     = _NOW.isoformat(),
        last_seen      = _NOW.isoformat(),
    )


def _make_graph(
    nodes: list[TopologyNode] | None = None,
    edges: list[TopologyEdge] | None = None,
) -> TopologyGraph:
    return TopologyGraph(
        nodes       = nodes or [],
        edges       = edges or [],
        computed_at = _NOW.isoformat(),
    )


# ---------------------------------------------------------------------------
# Unit tests: _compute_risk()
# ---------------------------------------------------------------------------

class TestComputeRisk:
    """Unit tests for the risk scoring formula — no DB or HTTP involved."""

    def test_zero_signals_is_low(self):
        score, level = _compute_risk(0, 0, 0.0)
        assert score == 0.0
        assert level == "low"

    def test_low_boundary(self):
        # 1 violation (0.05) → below 0.20 threshold → low
        score, level = _compute_risk(1, 0, 0.0)
        assert score == 0.05
        assert level == "low"

    def test_medium_boundary(self):
        # 4 violations → 0.20 → medium
        score, level = _compute_risk(4, 0, 0.0)
        assert score == 0.20
        assert level == "medium"

    def test_high_boundary(self):
        # 8 violations → 0.40 → high
        score, level = _compute_risk(8, 0, 0.0)
        assert score == 0.40
        assert level == "high"

    def test_critical_boundary(self):
        # 8 violations + 3 anomalies → 0.40 + 0.30 = 0.70 → critical
        score, level = _compute_risk(8, 3, 0.0)
        assert score == 0.70
        assert level == "critical"

    def test_violation_cap_at_0_4(self):
        # 100 violations: min(0.4, 100*0.05) = 0.4
        score, _ = _compute_risk(100, 0, 0.0)
        assert score == 0.40

    def test_anomaly_cap_at_0_3(self):
        # 100 anomalies: min(0.3, 100*0.10) = 0.3
        score, _ = _compute_risk(0, 100, 0.0)
        assert score == 0.30

    def test_injection_below_threshold_no_contribution(self):
        # injection_score = 0.39 < 0.4 — no contribution
        score, level = _compute_risk(0, 0, 0.39)
        assert score == 0.0
        assert level == "low"

    def test_injection_at_threshold_no_contribution(self):
        # injection_score = 0.40 → (0.40-0.40)/0.6*0.3 = 0.0
        score, _ = _compute_risk(0, 0, 0.40)
        assert score == 0.0

    def test_injection_above_threshold_contributes(self):
        # injection_score = 1.0 → (1.0-0.4)/0.6*0.3 = 0.3
        score, _ = _compute_risk(0, 0, 1.0)
        assert abs(score - 0.30) < 0.001

    def test_all_three_signals_combined(self):
        # 8 violations (0.4) + 3 anomalies (0.3) + injection 1.0 (0.3) → 1.0 capped
        score, level = _compute_risk(8, 3, 1.0)
        assert score == 1.0
        assert level == "critical"

    def test_score_is_rounded_to_3_decimals(self):
        # 1 violation (0.05) — already exact; test rounding with odd values
        score, _ = _compute_risk(1, 1, 0.5)
        # score = 0.05 + 0.10 + (0.5-0.4)/0.6*0.3 = 0.15 + 0.05 = 0.2
        str_score = str(score)
        # No more than 3 decimal places
        if "." in str_score:
            assert len(str_score.split(".")[1]) <= 3


# ---------------------------------------------------------------------------
# Unit tests: TopologyNode.to_dict() / TopologyEdge.to_dict()
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_node_to_dict_keys(self):
        node = _make_node()
        d = node.to_dict()
        expected_keys = {
            "agent_id", "session_count", "llm_call_count", "tool_call_count",
            "error_count", "violation_count", "anomaly_count", "pii_event_count",
            "injection_score_max", "risk_score", "risk_level",
            "first_seen", "last_seen", "providers", "models",
        }
        assert expected_keys == set(d.keys())

    def test_node_to_dict_values(self):
        node = _make_node(agent_id="test", session_count=3, risk_level="high")
        d = node.to_dict()
        assert d["agent_id"] == "test"
        assert d["session_count"] == 3
        assert d["risk_level"] == "high"

    def test_edge_to_dict_keys(self):
        edge = _make_edge()
        d = edge.to_dict()
        expected_keys = {"source", "target", "call_count", "avg_latency_ms", "first_seen", "last_seen"}
        assert expected_keys == set(d.keys())

    def test_graph_to_dict_shape(self):
        graph = _make_graph(
            nodes=[_make_node("a"), _make_node("b")],
            edges=[_make_edge("a", "b")],
        )
        d = graph.to_dict()
        assert d["total_nodes"] == 2
        assert d["total_edges"] == 1
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert "computed_at" in d


# ---------------------------------------------------------------------------
# API endpoint tests — mock compute_topology
# ---------------------------------------------------------------------------

class TestTopologyEndpoint:
    """API-level tests; mock `compute_topology` so no DB is needed."""

    def test_requires_auth(self):
        resp = client.get("/v1/topology")
        assert resp.status_code == 401

    def test_rejects_wrong_key(self):
        resp = client.get("/v1/topology", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    @patch("app.api.v1.topology.compute_topology")
    def test_empty_graph(self, mock_compute):
        mock_compute.return_value = _make_graph()
        mock_compute.__call__ = AsyncMock(return_value=_make_graph())
        # Use AsyncMock for async function
        import asyncio
        mock_compute.side_effect = None
        mock_compute.return_value = None

        async def _async_return(*args, **kwargs):
            return _make_graph()

        mock_compute.side_effect = _async_return

        resp = client.get("/v1/topology", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_nodes"] == 0
        assert body["total_edges"] == 0
        assert body["nodes"] == []
        assert body["edges"] == []
        assert "computed_at" in body

    @patch("app.api.v1.topology.compute_topology")
    def test_single_node(self, mock_compute):
        node = _make_node("orchestrator", session_count=10, risk_level="medium")
        graph = _make_graph(nodes=[node])

        async def _return(*args, **kwargs):
            return graph

        mock_compute.side_effect = _return

        resp = client.get("/v1/topology", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_nodes"] == 1
        assert body["total_edges"] == 0
        n = body["nodes"][0]
        assert n["agent_id"] == "orchestrator"
        assert n["risk_level"] == "medium"

    @patch("app.api.v1.topology.compute_topology")
    def test_graph_with_edges(self, mock_compute):
        nodes = [_make_node("orchestrator"), _make_node("worker")]
        edges = [_make_edge("orchestrator", "worker", call_count=42)]
        graph = _make_graph(nodes=nodes, edges=edges)

        async def _return(*args, **kwargs):
            return graph

        mock_compute.side_effect = _return

        resp = client.get("/v1/topology", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_nodes"] == 2
        assert body["total_edges"] == 1
        edge = body["edges"][0]
        assert edge["source"] == "orchestrator"
        assert edge["target"] == "worker"
        assert edge["call_count"] == 42

    @patch("app.api.v1.topology.compute_topology")
    def test_include_isolated_param_passed(self, mock_compute):
        captured: dict = {}

        async def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _make_graph()

        mock_compute.side_effect = _capture

        resp = client.get("/v1/topology?include_isolated=false", headers=HEADERS)
        assert resp.status_code == 200
        assert captured.get("include_isolated") is False

    @patch("app.api.v1.topology.compute_topology")
    def test_min_edge_calls_param_passed(self, mock_compute):
        captured: dict = {}

        async def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _make_graph()

        mock_compute.side_effect = _capture

        resp = client.get("/v1/topology?min_edge_calls=3", headers=HEADERS)
        assert resp.status_code == 200
        assert captured.get("min_edge_calls") == 3

    def test_min_edge_calls_lt_one_rejected(self):
        resp = client.get("/v1/topology?min_edge_calls=0", headers=HEADERS)
        assert resp.status_code == 422

    @patch("app.api.v1.topology.compute_topology")
    def test_default_params_are_applied(self, mock_compute):
        captured: dict = {}

        async def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _make_graph()

        mock_compute.side_effect = _capture

        resp = client.get("/v1/topology", headers=HEADERS)
        assert resp.status_code == 200
        assert captured.get("include_isolated") is True
        assert captured.get("min_edge_calls") == 1

    @patch("app.api.v1.topology.compute_topology")
    def test_all_node_fields_present(self, mock_compute):
        node = _make_node(
            agent_id="agent-x",
            violation_count=2,
            anomaly_count=1,
            injection_score_max=0.55,
            providers=["openai", "anthropic"],
            models=["gpt-4o", "claude-3-sonnet"],
        )
        score, level = _compute_risk(2, 1, 0.55)
        node.risk_score = score
        node.risk_level = level

        async def _return(*args, **kwargs):
            return _make_graph(nodes=[node])

        mock_compute.side_effect = _return

        resp = client.get("/v1/topology", headers=HEADERS)
        n = resp.json()["nodes"][0]
        assert n["violation_count"] == 2
        assert n["anomaly_count"] == 1
        assert n["injection_score_max"] == 0.55
        assert "openai" in n["providers"]
        assert "gpt-4o" in n["models"]
        assert isinstance(n["risk_score"], float)
        assert n["risk_level"] in ("low", "medium", "high", "critical")

    @patch("app.api.v1.topology.compute_topology")
    def test_null_latency_in_edge_serialized_as_null(self, mock_compute):
        edge = TopologyEdge(
            source="a", target="b", call_count=1,
            avg_latency_ms=None,
            first_seen=None, last_seen=None,
        )

        async def _return(*args, **kwargs):
            return _make_graph(edges=[edge])

        mock_compute.side_effect = _return

        resp = client.get("/v1/topology", headers=HEADERS)
        e = resp.json()["edges"][0]
        assert e["avg_latency_ms"] is None

    @patch("app.api.v1.topology.compute_topology")
    def test_multiple_edges_returned(self, mock_compute):
        edges = [
            _make_edge("a", "b", 10),
            _make_edge("b", "c", 5),
            _make_edge("a", "c", 2),
        ]

        async def _return(*args, **kwargs):
            return _make_graph(edges=edges)

        mock_compute.side_effect = _return

        resp = client.get("/v1/topology", headers=HEADERS)
        body = resp.json()
        assert body["total_edges"] == 3
        assert len(body["edges"]) == 3


# ---------------------------------------------------------------------------
# Risk level boundary table test
# ---------------------------------------------------------------------------

class TestRiskLevelBoundaries:
    """Verify exact boundaries between risk levels."""

    CASES = [
        # (violations, anomalies, injection, expected_level)
        (0,  0, 0.00, "low"),
        (3,  0, 0.00, "low"),    # 0.15 < 0.20
        (4,  0, 0.00, "medium"), # 0.20
        (7,  0, 0.00, "medium"), # 0.35 < 0.40
        (8,  0, 0.00, "high"),   # 0.40
        (8,  3, 0.00, "critical"), # 0.70
        (0,  3, 0.00, "medium"), # 0.30
        (0,  4, 0.00, "medium"), # 0.30 (anomaly cap is 0.3, not 0.4)
        (0,  0, 1.00, "medium"), # 0.30
        (4,  2, 1.00, "critical"), # 0.20+0.20+0.30=0.70
    ]

    @pytest.mark.parametrize("violations,anomalies,injection,expected", CASES)
    def test_level(self, violations, anomalies, injection, expected):
        _, level = _compute_risk(violations, anomalies, injection)
        assert level == expected, (
            f"Expected {expected!r} for v={violations} a={anomalies} i={injection}"
        )
