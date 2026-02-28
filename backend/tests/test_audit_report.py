"""
Tests for the org-wide compliance audit report service and endpoint.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.services.audit_report import (
    _verify_chains,
    _compute_control_status,
    build_audit_report,
    _FRAMEWORK_CONTROLS,
)


# ─── Unit tests: _verify_chains ───────────────────────────────────────────────

class FakeChainRow:
    def __init__(self, session_id, prev, curr, seq):
        self.session_id = session_id
        self.previous_hash = prev
        self.current_hash = curr
        self.sequence_number = seq


def test_verify_chains_empty():
    total, valid = _verify_chains([])
    assert total == 0
    assert valid == 0


def test_verify_chains_single_session_single_event():
    rows = [FakeChainRow("s1", "0" * 64, "abc", 0)]
    total, valid = _verify_chains(rows)
    assert total == 1
    assert valid == 1


def test_verify_chains_single_session_valid():
    rows = [
        FakeChainRow("s1", "0" * 64, "hash1", 0),
        FakeChainRow("s1", "hash1", "hash2", 1),
        FakeChainRow("s1", "hash2", "hash3", 2),
    ]
    total, valid = _verify_chains(rows)
    assert total == 1
    assert valid == 1


def test_verify_chains_single_session_broken():
    rows = [
        FakeChainRow("s1", "0" * 64, "hash1", 0),
        FakeChainRow("s1", "WRONG", "hash2", 1),  # broken link
    ]
    total, valid = _verify_chains(rows)
    assert total == 1
    assert valid == 0


def test_verify_chains_multiple_sessions():
    rows = [
        FakeChainRow("s1", "0" * 64, "h1", 0),
        FakeChainRow("s1", "h1", "h2", 1),
        FakeChainRow("s2", "0" * 64, "x1", 0),
        FakeChainRow("s2", "BROKEN", "x2", 1),  # s2 invalid
    ]
    total, valid = _verify_chains(rows)
    assert total == 2
    assert valid == 1


# ─── Unit tests: _compute_control_status ─────────────────────────────────────

def _make_summary(**kwargs):
    base = {
        "total_sessions": 10,
        "llm_calls": 20,
        "tool_calls": 5,
        "pii_events": 0,
        "blocked_count": 0,
        "alert_count": 0,
        "anomalies": 0,
    }
    base.update(kwargs)
    return base


def test_control_injection_no_calls():
    status, ev = _compute_control_status(
        "injection", _make_summary(llm_calls=0), {}, 0, 0, 0
    )
    assert status == "fail"


def test_control_injection_blocked():
    vr = {"prompt-injection-detected": {"action": "BLOCK", "count": 3}}
    status, ev = _compute_control_status(
        "injection", _make_summary(llm_calls=10), vr, 0, 5, 5
    )
    assert status == "pass"
    assert "3" in ev


def test_control_injection_alert_only():
    vr = {"prompt-injection-detected": {"action": "ALERT", "count": 2}}
    status, ev = _compute_control_status(
        "injection", _make_summary(llm_calls=10), vr, 0, 5, 5
    )
    assert status == "partial"


def test_control_injection_clean():
    status, ev = _compute_control_status(
        "injection", _make_summary(llm_calls=10), {}, 0, 5, 5
    )
    assert status == "pass"


def test_control_pii_detected():
    status, ev = _compute_control_status(
        "pii", _make_summary(pii_events=3), {}, 0, 5, 5
    )
    assert status == "partial"
    assert "3" in ev


def test_control_pii_clean():
    status, ev = _compute_control_status(
        "pii", _make_summary(pii_events=0), {}, 0, 5, 5
    )
    assert status == "pass"


def test_control_record_keeping_no_sessions():
    status, ev = _compute_control_status(
        "record_keeping", _make_summary(total_sessions=0), {}, 0, 0, 0
    )
    assert status == "fail"


def test_control_record_keeping_has_sessions():
    status, ev = _compute_control_status(
        "record_keeping", _make_summary(total_sessions=5, llm_calls=10), {}, 0, 5, 5
    )
    assert status == "pass"


def test_control_chain_integrity_all_valid():
    status, ev = _compute_control_status(
        "chain_integrity", _make_summary(), {}, 0, 10, 10
    )
    assert status == "pass"
    assert "100%" in ev


def test_control_chain_integrity_broken():
    status, ev = _compute_control_status(
        "chain_integrity", _make_summary(), {}, 0, 10, 8
    )
    assert status == "fail"


def test_control_chain_integrity_no_events():
    status, ev = _compute_control_status(
        "chain_integrity", _make_summary(), {}, 0, 0, 0
    )
    assert status == "fail"


def test_control_anomalies_detected():
    status, ev = _compute_control_status(
        "anomalies", _make_summary(), {}, 5, 10, 10
    )
    assert status == "pass"
    assert "5" in ev


def test_control_anomaly_no_block():
    status, ev = _compute_control_status(
        "robustness", _make_summary(), {}, 3, 10, 10
    )
    assert status == "partial"


def test_control_robustness_with_block():
    vr = {"injection-rule": {"action": "BLOCK", "count": 2}}
    status, ev = _compute_control_status(
        "robustness", _make_summary(), vr, 3, 10, 10
    )
    assert status == "pass"


# ─── Framework coverage ───────────────────────────────────────────────────────

def test_all_frameworks_have_controls():
    for fw in ["owasp_asi_2026", "eu_ai_act", "hipaa", "soc2"]:
        assert fw in _FRAMEWORK_CONTROLS
        controls = _FRAMEWORK_CONTROLS[fw]
        assert len(controls) == 4
        for ctrl in controls:
            assert "id" in ctrl
            assert "name" in ctrl
            assert "evidence_type" in ctrl


# ─── Integration-style: build_audit_report with mocked DB ────────────────────

def _make_db_mock(
    ev_row=None,
    viol_rows=None,
    anom_row=None,
    chain_rows=None,
    top_rows=None,
    agents_rows=None,
):
    """Build an AsyncSession mock that returns preset rows for each query."""
    db = AsyncMock()

    # We call db.execute 5 times (4 gather + 1 top violations), then 1 agents.
    # MagicMock side_effect cycles through results.
    def make_result(rows_or_row, single=False):
        r = MagicMock()
        if single:
            r.fetchone.return_value = rows_or_row
        else:
            r.fetchone.return_value = rows_or_row[0] if rows_or_row else None
            r.fetchall.return_value = rows_or_row or []
        return r

    ev_res = MagicMock()
    ev_res.fetchone.return_value = ev_row
    viol_res = MagicMock()
    viol_res.fetchall.return_value = viol_rows or []
    anom_res = MagicMock()
    anom_res.fetchone.return_value = anom_row
    chain_res = MagicMock()
    chain_res.fetchall.return_value = chain_rows or []
    top_res = MagicMock()
    top_res.fetchall.return_value = top_rows or []
    agents_res = MagicMock()
    agents_res.fetchall.return_value = agents_rows or []

    call_results = [ev_res, viol_res, anom_res, chain_res, top_res, agents_res]
    call_count = [0]

    async def fake_execute(sql, params=None):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(call_results):
            return call_results[idx]
        return MagicMock()

    db.execute.side_effect = fake_execute
    return db


class FakeEvRow:
    total_sessions = 10
    llm_calls = 20
    tool_calls = 5
    memory_blocked = 2
    pii_events = 1
    error_count = 0
    agent_count = 3
    avg_latency_ms = 145.5


class FakeAnomRow:
    total = 2


class FakeViolRow:
    rule_name = "prompt-injection-detected"
    action = "BLOCK"
    count = 3


@pytest.mark.asyncio
async def test_build_audit_report_soc2():
    db = _make_db_mock(
        ev_row=FakeEvRow(),
        viol_rows=[FakeViolRow()],
        anom_row=FakeAnomRow(),
        chain_rows=[
            FakeChainRow("s1", "0" * 64, "h1", 0),
            FakeChainRow("s1", "h1", "h2", 1),
        ],
    )
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    from_ns = now_ns - 30 * 24 * 3600 * 1_000_000_000
    report = await build_audit_report(
        db, org_id="test-org", from_ts_ns=from_ns, to_ts_ns=now_ns, framework="soc2"
    )

    assert report["framework"] == "soc2"
    assert report["org_id"] == "test-org"
    assert "controls" in report
    assert len(report["controls"]) == 4
    assert all(c["status"] in {"pass", "partial", "fail"} for c in report["controls"])
    assert report["summary"]["total_sessions"] == 10
    assert report["summary"]["blocked_count"] == 3
    assert report["summary"]["anomalies"] == 2
    assert report["summary"]["chain_valid_pct"] == 100.0


@pytest.mark.asyncio
async def test_build_audit_report_invalid_framework():
    db = _make_db_mock()
    with pytest.raises(ValueError, match="Unknown framework"):
        await build_audit_report(
            db,
            org_id="test-org",
            from_ts_ns=0,
            to_ts_ns=1_000_000_000,
            framework="unknown_fw",
        )


@pytest.mark.asyncio
async def test_build_audit_report_all_frameworks():
    for fw in ["owasp_asi_2026", "eu_ai_act", "hipaa", "soc2"]:
        db = _make_db_mock(ev_row=FakeEvRow(), anom_row=FakeAnomRow())
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        report = await build_audit_report(
            db,
            org_id="default-org",
            from_ts_ns=now_ns - 86400_000_000_000,
            to_ts_ns=now_ns,
            framework=fw,
        )
        assert report["framework"] == fw
        assert len(report["controls"]) == 4


@pytest.mark.asyncio
async def test_build_audit_report_no_data():
    db = _make_db_mock(ev_row=None, anom_row=None)
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    report = await build_audit_report(
        db,
        org_id="default-org",
        from_ts_ns=now_ns - 86400_000_000_000,
        to_ts_ns=now_ns,
        framework="soc2",
    )
    assert report["summary"]["total_sessions"] == 0
    assert report["summary"]["overall_status"] == "fail"


@pytest.mark.asyncio
async def test_build_audit_report_overall_status_pass():
    """No violations, no PII, valid chains → overall pass."""

    class CleanEvRow:
        total_sessions = 5
        llm_calls = 10
        tool_calls = 3
        memory_blocked = 0
        pii_events = 0
        error_count = 0
        agent_count = 1
        avg_latency_ms = 80.0

    db = _make_db_mock(
        ev_row=CleanEvRow(),
        anom_row=FakeAnomRow(),
        chain_rows=[FakeChainRow("s1", "0" * 64, "h1", 0)],
    )
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    report = await build_audit_report(
        db,
        org_id="default-org",
        from_ts_ns=now_ns - 86400_000_000_000,
        to_ts_ns=now_ns,
        framework="soc2",
    )
    assert report["summary"]["overall_status"] == "pass"
