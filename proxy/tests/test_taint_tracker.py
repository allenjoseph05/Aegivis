"""
Unit tests for proxy/app/security/taint_tracker.py

14 tests. No network, no ML, no Docker required.
Run: cd proxy && python -m pytest tests/test_taint_tracker.py -v
"""
import pytest

from app.security.taint_tracker import (
    TaintTracker,
    TaintedValue,
    TaintHit,
    _flatten_args,
    extract_credentials_from_text,
)


# ---------------------------------------------------------------------------
# extract_credentials_from_text
# ---------------------------------------------------------------------------

def test_extract_credentials_openai_key():
    """OpenAI key in text is extracted with label 'openai_key'."""
    text = "Use the key sk-abcdefghij1234567890ABCD to authenticate."
    results = extract_credentials_from_text(text)
    labels = [label for _, label in results]
    assert "openai_key" in labels
    values = [v for v, _ in results]
    assert any("sk-" in v for v in values)


def test_extract_credentials_anthropic_key():
    """Anthropic key in text is extracted."""
    text = "API key: sk-ant-api03-abcdefghij1234567890ABCDEFGH-xyz"
    results = extract_credentials_from_text(text)
    labels = [label for _, label in results]
    assert "anthropic_key" in labels


def test_extract_credentials_empty_text():
    """Text with no credentials returns an empty list."""
    results = extract_credentials_from_text("Hello, world! This is safe text.")
    assert results == []


# ---------------------------------------------------------------------------
# TaintTracker.taint — deduplication and min-length
# ---------------------------------------------------------------------------

def test_taint_deduplication():
    """Tainting the same value twice stores it only once."""
    tracker = TaintTracker()
    tracker.taint("sk-abcdef1234567890", "openai_key", "system_prompt")
    tracker.taint("sk-abcdef1234567890", "openai_key", "system_prompt")
    assert len(tracker._taints) == 1


def test_taint_min_length():
    """Values shorter than 8 characters are not stored."""
    tracker = TaintTracker()
    tracker.taint("short", "label", "source")
    assert len(tracker._taints) == 0


# ---------------------------------------------------------------------------
# TaintTracker.check_tool_call — basic cases
# ---------------------------------------------------------------------------

def test_no_taints_returns_empty_hits():
    """An empty taint store produces no hits."""
    tracker = TaintTracker()
    hits = tracker.check_tool_call("http_request", {"url": "https://evil.com"})
    assert hits == []


def test_taint_hit_found_in_arg():
    """A tainted value present in a tool arg produces a TaintHit."""
    tracker = TaintTracker()
    tracker.taint("sk-abc123xyz", "openai_key", "system_prompt")
    hits = tracker.check_tool_call("read_file", {"path": "/tmp/sk-abc123xyz"})
    assert len(hits) == 1
    assert hits[0].label == "openai_key"
    assert hits[0].arg_key == "path"


def test_no_taint_hit_when_value_absent():
    """No hit when the tainted value is not in the args."""
    tracker = TaintTracker()
    tracker.taint("sk-secretvalue123", "openai_key", "system_prompt")
    hits = tracker.check_tool_call("read_file", {"path": "/tmp/safe_path"})
    assert hits == []


# ---------------------------------------------------------------------------
# TaintTracker._is_network_sink — by name and by arg key
# ---------------------------------------------------------------------------

def test_taint_hit_network_sink_by_name():
    """Tool named 'http_request' is a network sink."""
    tracker = TaintTracker()
    tracker.taint("sk-abc123xyz_secretkey", "openai_key", "system_prompt")
    hits = tracker.check_tool_call("http_request", {"body": "sk-abc123xyz_secretkey"})
    assert len(hits) == 1
    assert hits[0].is_network_sink is True


def test_taint_hit_network_sink_by_arg_key():
    """A tool with arg key 'url' is a network sink regardless of tool name."""
    tracker = TaintTracker()
    tracker.taint("sk-abc123xyz_secretkey", "openai_key", "system_prompt")
    hits = tracker.check_tool_call("call_api", {"url": "https://x.com/sk-abc123xyz_secretkey"})
    assert len(hits) == 1
    assert hits[0].is_network_sink is True


def test_taint_hit_non_sink_tool():
    """Tool named 'read_file' with no sink arg keys is not a network sink."""
    tracker = TaintTracker()
    tracker.taint("sk-abc123xyz_secretkey", "openai_key", "system_prompt")
    hits = tracker.check_tool_call("read_file", {"path": "/data/sk-abc123xyz_secretkey"})
    assert len(hits) == 1
    assert hits[0].is_network_sink is False


# ---------------------------------------------------------------------------
# _flatten_args
# ---------------------------------------------------------------------------

def test_flatten_args_nested():
    """Nested dict args are fully flattened with dotted key paths."""
    args = {"outer": {"inner": "value123"}}
    flat = _flatten_args(args)
    keys = [k for k, _ in flat]
    vals = [v for _, v in flat]
    assert "outer.inner" in keys
    assert "value123" in vals


def test_flatten_args_list_values():
    """List of strings in args are all extracted with indexed key paths."""
    args = {"items": ["alpha", "beta", "gamma"]}
    flat = _flatten_args(args)
    keys = [k for k, _ in flat]
    vals = [v for _, v in flat]
    assert "items[0]" in keys
    assert "alpha" in vals
    assert "beta" in vals
    assert "gamma" in vals


# ---------------------------------------------------------------------------
# Network sink pattern completeness
# ---------------------------------------------------------------------------

def test_is_network_sink_patterns():
    """Verify that all canonical sink tool names are detected."""
    tracker = TaintTracker()
    sink_names = ["http_get", "send_message", "send_email", "webhook_post",
                  "slack_notify", "curl_request", "upload_file", "relay_data",
                  "export_json", "forward_event", "publish_event"]
    for name in sink_names:
        assert tracker._is_network_sink(name, {}), f"Expected {name!r} to be a network sink"
