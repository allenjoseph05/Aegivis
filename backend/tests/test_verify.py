"""Tests for hash chain verification service."""
import pytest
import time
import hashlib
import json
from app.services.hash_verifier import recompute_hash, verify_session_chain, VerificationResult


def _canonical_json(obj: dict, exclude_keys=None) -> bytes:
    if exclude_keys:
        obj = {k: v for k, v in obj.items() if k not in exclude_keys}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _make_event(seq: int, session_id: str, prev_hash: str) -> dict:
    event = {
        "event_id": f"evt_{seq:04d}",
        "schema_version": "1.0",
        "org_id": "test",
        "session_id": session_id,
        "agent_id": "agent",
        "provider": "openai",
        "model": "gpt-4o",
        "interception_layer": "proxy",
        "run_id": f"run_{seq}",
        "parent_run_id": None,
        "event_type": "LLM_CALL_START",
        "payload": {"messages": [], "seq": seq},
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": time.time_ns() + seq,
        "sequence_number": seq,
        "previous_hash": prev_hash,
        "current_hash": "",
    }
    event["current_hash"] = hashlib.sha256(
        _canonical_json(event, exclude_keys={"current_hash", "received_at"})
    ).hexdigest()
    return event


def _build_chain(n: int, session_id="test_sess") -> list[dict]:
    events = []
    prev = f"GENESIS_{session_id}"
    for i in range(n):
        e = _make_event(i, session_id, prev)
        prev = e["current_hash"]
        events.append(e)
    return events


class TestRecomputeHash:
    def test_matches_stored(self):
        events = _build_chain(1)
        recomputed = recompute_hash(events[0])
        assert recomputed == events[0]["current_hash"]

    def test_detects_payload_change(self):
        events = _build_chain(1)
        events[0]["payload"]["seq"] = 999  # tamper
        recomputed = recompute_hash(events[0])
        assert recomputed != events[0]["current_hash"]


class TestVerifySessionChain:
    def test_valid_chain(self):
        events = _build_chain(10)
        result = verify_session_chain("test_sess", events)
        assert result.valid is True
        assert result.first_failed_sequence is None
        assert result.total_events == 10

    def test_empty_chain(self):
        result = verify_session_chain("test_sess", [])
        assert result.valid is True
        assert result.total_events == 0

    def test_single_event(self):
        events = _build_chain(1)
        result = verify_session_chain("test_sess", events)
        assert result.valid is True

    def test_tampered_payload_detected(self):
        events = _build_chain(5)
        events[3]["payload"]["messages"] = [{"role": "user", "content": "INJECTED"}]
        # Do NOT recompute hash — simulating real attack
        result = verify_session_chain("test_sess", events)
        assert result.valid is False
        assert result.first_failed_sequence == 3

    def test_chain_break_detected(self):
        events = _build_chain(5)
        # Break the chain: event[2] has wrong previous_hash
        events[2]["previous_hash"] = "wrong" * 10 + "0000"
        # Also need to recompute event[2].current_hash for the break to be at seq 2
        events[2]["current_hash"] = recompute_hash(events[2])
        result = verify_session_chain("test_sess", events)
        assert result.valid is False
        # Chain break at event 2 (wrong previous_hash)
        assert result.first_failed_sequence == 2

    def test_result_has_timestamp(self):
        events = _build_chain(3)
        result = verify_session_chain("s", events)
        assert result.checked_at is not None
        assert "T" in result.checked_at  # ISO format

    def test_to_dict(self):
        events = _build_chain(3)
        result = verify_session_chain("s", events)
        d = result.to_dict()
        assert "valid" in d
        assert "total_events" in d
        assert "session_id" in d
