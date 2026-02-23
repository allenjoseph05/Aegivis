"""Tests for the SHA-256 hash chain implementation."""
import pytest
import copy
from proxy.app.hash_chain import (
    compute_event_hash,
    genesis_hash,
    merkle_root,
    verify_chain,
)


def _make_event(session_id: str, seq: int, prev_hash: str, payload: dict | None = None) -> dict:
    """Create a minimal event dict for testing."""
    import time
    event = {
        "event_id": f"event_{seq:04d}",
        "schema_version": "1.0",
        "org_id": "test-org",
        "session_id": session_id,
        "agent_id": "test-agent",
        "provider": "openai",
        "model": "gpt-4o",
        "interception_layer": "proxy",
        "run_id": f"run_{seq:04d}",
        "parent_run_id": None,
        "event_type": "LLM_CALL_START",
        "payload": payload or {"messages": [{"role": "user", "content": "hello"}]},
        "payload_hash": None,
        "pii_detected": [],
        "timestamp_ns": time.time_ns() + seq,
        "sequence_number": seq,
        "previous_hash": prev_hash,
        "current_hash": "",
    }
    event["current_hash"] = compute_event_hash(event)
    return event


def _build_chain(n: int, session_id: str = "sess_abc123") -> list[dict]:
    """Build a valid chain of n events."""
    events = []
    prev = genesis_hash(session_id)
    for i in range(n):
        event = _make_event(session_id, i, prev)
        prev = event["current_hash"]
        events.append(event)
    return events


class TestHashComputation:
    def test_hash_is_64_hex_chars(self):
        event = _make_event("s1", 0, genesis_hash("s1"))
        assert len(event["current_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in event["current_hash"])

    def test_hash_is_deterministic(self):
        event = _make_event("s1", 0, genesis_hash("s1"))
        h1 = compute_event_hash(event)
        h2 = compute_event_hash(event)
        assert h1 == h2

    def test_hash_excludes_current_hash_field(self):
        event = _make_event("s1", 0, genesis_hash("s1"))
        event["current_hash"] = "deadbeef" * 8
        recomputed = compute_event_hash(event)
        # Should produce same hash regardless of current_hash value
        event2 = copy.deepcopy(event)
        event2["current_hash"] = "cafebabe" * 8
        assert compute_event_hash(event2) == recomputed

    def test_any_payload_change_changes_hash(self):
        event1 = _make_event("s1", 0, genesis_hash("s1"), {"messages": [{"role": "user", "content": "hello"}]})
        event2 = _make_event("s1", 0, genesis_hash("s1"), {"messages": [{"role": "user", "content": "MODIFIED"}]})
        assert event1["current_hash"] != event2["current_hash"]

    def test_genesis_hash_format(self):
        gh = genesis_hash("sess_abc123")
        assert gh == "GENESIS_sess_abc123"


class TestChainVerification:
    def test_valid_chain_passes(self):
        events = _build_chain(5)
        valid, failed_seq, msg = verify_chain(events)
        assert valid is True
        assert failed_seq is None
        assert msg is None

    def test_empty_chain_passes(self):
        valid, failed_seq, msg = verify_chain([])
        assert valid is True

    def test_single_event_chain(self):
        events = _build_chain(1)
        valid, _, _ = verify_chain(events)
        assert valid is True

    def test_tampered_payload_detected(self):
        events = _build_chain(5)
        # Tamper with event at sequence 2
        events[2]["payload"]["messages"][0]["content"] = "TAMPERED"
        # Do NOT recompute hash — simulating a real attack

        valid, failed_seq, msg = verify_chain(events)
        assert valid is False
        assert failed_seq == 2
        assert "mismatch" in msg.lower()

    def test_tampered_hash_chain_detected(self):
        events = _build_chain(5)
        # Tamper: modify event[2].current_hash to break the chain at event[3]
        events[2]["current_hash"] = "a" * 64
        # event[3].previous_hash should no longer match

        valid, failed_seq, msg = verify_chain(events)
        assert valid is False
        # Either event[2] hash mismatch or event[3] chain break
        assert failed_seq is not None

    def test_reordered_events_detected(self):
        events = _build_chain(5)
        # Swap events 1 and 2 — breaks previous_hash linkage
        events[1], events[2] = events[2], events[1]
        valid, failed_seq, msg = verify_chain(events)
        assert valid is False

    def test_100_event_chain(self):
        events = _build_chain(100)
        valid, _, _ = verify_chain(events)
        assert valid is True


class TestMerkleRoot:
    def test_single_hash(self):
        h = "a" * 64
        root = merkle_root([h])
        assert len(root) == 64

    def test_two_hashes(self):
        h1 = "a" * 64
        h2 = "b" * 64
        root = merkle_root([h1, h2])
        assert len(root) == 64

    def test_odd_count(self):
        hashes = ["a" * 64, "b" * 64, "c" * 64]
        root = merkle_root(hashes)
        assert len(root) == 64

    def test_different_orders_give_different_roots(self):
        h1, h2 = "a" * 64, "b" * 64
        r1 = merkle_root([h1, h2])
        r2 = merkle_root([h2, h1])
        assert r1 != r2

    def test_empty_returns_hash(self):
        root = merkle_root([])
        assert len(root) == 64

    def test_1000_hashes(self):
        import hashlib
        hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(1000)]
        root = merkle_root(hashes)
        assert len(root) == 64
