"""
Phase 2 SDK tests: hash chain correctness, new methods, LangChain adapter.

Run from sdk/ directory:
    python -m pytest tests/test_sdk_phase2.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
import os
import importlib

import pytest

# Make the SDK importable without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegivis.session import Session


# ---------------------------------------------------------------------------
# Hash chain tests
# ---------------------------------------------------------------------------

def test_hash_chain_sequential():
    """Each event's previous_hash equals the prior event's current_hash."""
    s = Session(agent_id="test-agent", session_id="test-sess-001")
    e1 = s.annotate("first")
    e2 = s.annotate("second")
    e3 = s.annotate("third")

    assert e1["previous_hash"] == f"SDK_GENESIS_{s.session_id}"
    assert e2["previous_hash"] == e1["current_hash"]
    assert e3["previous_hash"] == e2["current_hash"]


def test_hash_chain_integrity():
    """current_hash is correctly computed from the event body."""
    s = Session(agent_id="test-agent", session_id="test-sess-002")
    e = s.annotate("hello world")

    # Recompute: current_hash should not be in the dict used for hashing
    event_for_hash = {k: v for k, v in e.items() if k != "current_hash"}
    expected = hashlib.sha256(
        json.dumps(event_for_hash, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert e["current_hash"] == expected


def test_sequence_numbers_increment():
    """Sequence numbers start at 1 and increment correctly."""
    s = Session(agent_id="test-agent", session_id="test-sess-003")
    e1 = s.annotate("first")
    e2 = s.annotate("second")
    e3 = s.annotate("third")

    assert e1["sequence_number"] == 1
    assert e2["sequence_number"] == 2
    assert e3["sequence_number"] == 3


def test_hash_differs_per_event():
    """Every event produces a unique hash (no constant collision)."""
    s = Session(agent_id="test-agent", session_id="test-sess-004")
    hashes = [s.annotate(f"msg-{i}")["current_hash"] for i in range(5)]
    assert len(set(hashes)) == 5, "Expected all hashes to be unique"


# ---------------------------------------------------------------------------
# tag() method tests
# ---------------------------------------------------------------------------

def test_tag_event_payload():
    """tag() creates correct AGENT_THOUGHT payload with type=tag."""
    s = Session(agent_id="test-agent", session_id="test-sess-005")
    e = s.tag("env", "prod")

    assert e["event_type"] == "AGENT_THOUGHT"
    assert e["payload"]["type"] == "tag"
    assert e["payload"]["tag_key"] == "env"
    assert e["payload"]["tag_value"] == "prod"


def test_tag_appears_in_event_buffer():
    """tag() event is recorded in the session's internal buffer."""
    s = Session(agent_id="test-agent", session_id="test-sess-006")
    s.tag("region", "us-east-1")
    assert len(s._events) == 1
    assert s._events[0]["payload"]["tag_key"] == "region"


# ---------------------------------------------------------------------------
# error() method tests
# ---------------------------------------------------------------------------

def test_error_event_payload():
    """error() creates SYSTEM_ERROR event with correct fields."""
    s = Session(agent_id="test-agent", session_id="test-sess-007")
    exc = ValueError("something went wrong")
    e = s.error("fail", exc=exc)

    assert e["event_type"] == "SYSTEM_ERROR"
    assert e["payload"]["error_message"] == "fail"
    assert e["payload"]["exception_type"] == "ValueError"
    assert e["payload"]["interception_layer"] == "sdk"


def test_error_without_exception():
    """error() without exc arg sets exception_type to None."""
    s = Session(agent_id="test-agent", session_id="test-sess-008")
    e = s.error("something failed")

    assert e["event_type"] == "SYSTEM_ERROR"
    assert e["payload"]["exception_type"] is None
    assert e["payload"]["error_message"] == "something failed"


def test_mixed_method_chain():
    """annotate, tag, and error all chain hashes correctly."""
    s = Session(agent_id="test-agent", session_id="test-sess-009")
    e1 = s.annotate("starting")
    e2 = s.tag("env", "test")
    e3 = s.error("boom", exc=RuntimeError("oops"))

    assert e2["previous_hash"] == e1["current_hash"]
    assert e3["previous_hash"] == e2["current_hash"]
    assert e3["sequence_number"] == 3


# ---------------------------------------------------------------------------
# LangChain adapter import guard
# ---------------------------------------------------------------------------

def test_langchain_import_guard():
    """Importing the LangChain adapter without langchain-core raises ImportError."""
    # Force the adapter module to be reimported without langchain available
    # by temporarily hiding it from sys.modules
    adapter_mod_name = "aegivis.adapters.langchain"

    # Remove cached import if present
    sys.modules.pop(adapter_mod_name, None)
    sys.modules.pop("langchain_core", None)
    sys.modules.pop("langchain_core.callbacks", None)

    # Simulate langchain_core not being installed
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"langchain_core": None, "langchain_core.callbacks": None}):
        with pytest.raises(ImportError, match="aegivis\\[langchain\\]"):
            # Re-import to trigger the ImportError path
            if adapter_mod_name in sys.modules:
                del sys.modules[adapter_mod_name]
            import aegivis.adapters.langchain  # noqa: F401
