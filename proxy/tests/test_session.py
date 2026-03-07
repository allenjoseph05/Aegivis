"""
Tests for SessionState.to_dict() / from_dict() round-trip.

These tests act as a regression guard: if a new persisted field is added to
to_dict() without a matching entry in from_dict() (or vice versa), the
key-set assertion in test_to_dict_from_dict_all_persisted_fields will fail,
prompting the developer to update both methods together.
"""
from __future__ import annotations

import json

import pytest

from app.session import SessionState


def test_to_dict_from_dict_defaults():
    """Default SessionState round-trips correctly through to_dict() / from_dict()."""
    orig = SessionState("sess_abc123", "agent-1")
    data = orig.to_dict()
    restored = SessionState.from_dict(data)

    assert restored.session_id == orig.session_id
    assert restored.agent_id == orig.agent_id
    assert restored.sequence_number == orig.sequence_number
    assert restored.last_hash == orig.last_hash
    assert restored.llm_call_count == orig.llm_call_count
    assert restored.tool_call_count == orig.tool_call_count
    assert restored.pending_tool_calls == orig.pending_tool_calls
    assert restored.started_at_ns == orig.started_at_ns
    assert restored.last_seen_ns == orig.last_seen_ns
    assert restored.first_user_hash == orig.first_user_hash
    assert restored.max_injection_score == orig.max_injection_score
    assert restored.total_tokens == orig.total_tokens
    assert restored.error_count == orig.error_count
    assert restored.system_prompt_hash == orig.system_prompt_hash
    assert restored.parent_agent_id == orig.parent_agent_id
    assert restored.parent_session_id == orig.parent_session_id
    assert restored.spawn_depth == orig.spawn_depth


def test_to_dict_from_dict_all_persisted_fields():
    """
    All persisted fields survive a round-trip and to_dict() key set is exact.

    If this test fails after adding a new field, update both to_dict() and
    from_dict() and add the key to expected_keys below.
    """
    orig = SessionState(
        "sess_test01",
        "agent-finance",
        first_user_hash="abc123",
        parent_agent_id="agent-root",
        parent_session_id="sess_parent01",
        spawn_depth=2,
    )
    orig.sequence_number = 42
    orig.last_hash = "sha256_abc"
    orig.llm_call_count = 7
    orig.tool_call_count = 3
    orig.pending_tool_calls = {"run_abc": {"tool_name": "bash", "args": {"cmd": "ls"}}}
    orig.max_injection_score = 0.73
    orig.total_tokens = 1500
    orig.error_count = 1
    orig.system_prompt_hash = "deadbeef"

    data = orig.to_dict()

    # Exact key set — new persisted fields must be added here too
    expected_keys = {
        "session_id", "agent_id", "sequence_number", "last_hash",
        "llm_call_count", "tool_call_count", "pending_tool_calls",
        "started_at_ns", "last_seen_ns", "first_user_hash",
        "max_injection_score", "total_tokens", "error_count",
        "system_prompt_hash", "parent_agent_id", "parent_session_id",
        "spawn_depth",
    }
    assert set(data.keys()) == expected_keys, (
        f"to_dict() key mismatch — add/remove fields without updating tests?\n"
        f"  extra={set(data.keys()) - expected_keys}\n"
        f"  missing={expected_keys - set(data.keys())}"
    )

    restored = SessionState.from_dict(data)

    assert restored.session_id == "sess_test01"
    assert restored.agent_id == "agent-finance"
    assert restored.first_user_hash == "abc123"
    assert restored.parent_agent_id == "agent-root"
    assert restored.parent_session_id == "sess_parent01"
    assert restored.spawn_depth == 2
    assert restored.sequence_number == 42
    assert restored.last_hash == "sha256_abc"
    assert restored.llm_call_count == 7
    assert restored.tool_call_count == 3
    assert restored.pending_tool_calls == {"run_abc": {"tool_name": "bash", "args": {"cmd": "ls"}}}
    assert restored.max_injection_score == pytest.approx(0.73)
    assert restored.total_tokens == 1500
    assert restored.error_count == 1
    assert restored.system_prompt_hash == "deadbeef"


def test_from_dict_resets_non_persisted_fields():
    """Non-persisted runtime state is always reset to safe defaults on load."""
    orig = SessionState("sess_xyz", "agent-test")
    data = orig.to_dict()

    # Manually set non-persisted fields before calling from_dict to confirm
    # they are NOT read from the dict (they aren't in it, but confirm defaults)
    restored = SessionState.from_dict(data)

    assert restored.ml_injection_flag is False
    assert restored.ml_injection_score == 0.0
    assert restored.trust_score == 1.0          # starts fully trusted
    assert restored.hitl_pending_approval_id is None
    assert restored.taint_tracker is None
    assert restored.pdg is None
    assert restored.ifc_context is None
    assert restored.tools_hash is None
    assert restored.tools_set is None
    assert restored.active_canaries == {}
    assert restored.event_type_sequence == []
    assert restored.injection_score_history == []


def test_to_dict_is_json_serializable():
    """to_dict() output must be JSON-serializable (used for file and Redis persistence)."""
    orig = SessionState("sess_json", "agent-json")
    orig.pending_tool_calls = {"run_1": {"tool_name": "bash", "args": {"cmd": "echo hi"}}}
    data = orig.to_dict()
    serialized = json.dumps(data)   # must not raise
    assert "sess_json" in serialized
    assert "agent-json" in serialized


def test_from_dict_missing_optional_fields_use_defaults():
    """from_dict() handles partial dicts (e.g. old format from disk) without crashing."""
    minimal = {"session_id": "sess_minimal"}
    restored = SessionState.from_dict(minimal)
    assert restored.session_id == "sess_minimal"
    assert restored.agent_id == "unknown"
    assert restored.sequence_number == 0
    assert restored.spawn_depth == 0
    assert restored.system_prompt_hash is None
    assert restored.parent_agent_id is None


def test_spawn_depth_persists():
    """spawn_depth is a critical field for multi-agent loop detection — must survive round-trip."""
    orig = SessionState("sess_child", "agent-child", spawn_depth=3)
    restored = SessionState.from_dict(orig.to_dict())
    assert restored.spawn_depth == 3
