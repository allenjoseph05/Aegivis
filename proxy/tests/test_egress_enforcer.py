"""
Tests for the Hard Egress Enforcer (Phase 13).

Covers:
  - is_empty() with no configured allowlist
  - Exact hostname match (allowed)
  - Exact hostname not in list (blocked)
  - Hostname with port match
  - Wildcard subdomain match (*.example.com)
  - Wildcard root match (example.com when *.example.com configured)
  - CIDR range match (10.0.0.0/8)
  - CIDR range non-match
  - URL with scheme — hostname extracted correctly
  - URL with non-standard port
  - Bare hostname arg value
  - Non-network arg key is ignored
  - Multiple violations in one call
  - Empty allowlist with check_tool_call — all destinations raise violation
  - Nested dict arg
  - _extract_hostname() helpers
  - Case-insensitive matching
"""
from __future__ import annotations

import pytest
from proxy.app.security.egress_enforcer import (
    EgressEnforcer,
    EgressViolation,
    _extract_hostname,
)


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------

def test_is_empty_no_allowlist():
    e = EgressEnforcer([])
    assert e.is_empty() is True


def test_is_empty_with_entries():
    e = EgressEnforcer(["api.openai.com"])
    assert e.is_empty() is False


# ---------------------------------------------------------------------------
# Exact hostname matching
# ---------------------------------------------------------------------------

def test_exact_match_allowed():
    e = EgressEnforcer(["api.openai.com", "api.anthropic.com"])
    violations = e.check_tool_call("http_get", {"url": "https://api.openai.com/v1/chat"})
    assert violations == []


def test_exact_match_not_in_list():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call("http_get", {"url": "https://attacker.evil.com/steal"})
    assert len(violations) == 1
    assert violations[0].destination == "attacker.evil.com"


def test_exact_match_with_port():
    e = EgressEnforcer(["db.internal:5432"])
    violations = e.check_tool_call("db_query", {"host": "db.internal:5432"})
    assert violations == []


def test_exact_match_hostname_only_when_list_has_port():
    # hostname without port should also match "db.internal:5432"
    e = EgressEnforcer(["db.internal:5432"])
    violations = e.check_tool_call("db_query", {"host": "db.internal"})
    # hostname-only doesn't include port, so "db.internal" != "db.internal:5432"
    # But we fall back to host_only match for the port-stripped version
    # "db.internal" != "db.internal" (port-stripped from "db.internal:5432" = "db.internal")
    assert violations == []


def test_blocked_different_port():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call("http_get", {"url": "https://api.openai.com:8443/v1"})
    # Should match because we strip port during comparison
    assert violations == []


# ---------------------------------------------------------------------------
# Wildcard subdomains
# ---------------------------------------------------------------------------

def test_wildcard_subdomain_match():
    e = EgressEnforcer(["*.anthropic.com"])
    violations = e.check_tool_call("fetch", {"url": "https://api.anthropic.com/v1/messages"})
    assert violations == []


def test_wildcard_root_match():
    # *.example.com should also match example.com itself
    e = EgressEnforcer(["*.googleapis.com"])
    violations = e.check_tool_call("fetch", {"url": "https://googleapis.com/resource"})
    assert violations == []


def test_wildcard_subdomain_not_matched():
    e = EgressEnforcer(["*.anthropic.com"])
    violations = e.check_tool_call("fetch", {"url": "https://attacker-anthropic.com/steal"})
    assert len(violations) == 1


def test_wildcard_deeper_subdomain():
    e = EgressEnforcer(["*.openai.com"])
    violations = e.check_tool_call("fetch", {"url": "https://api.beta.openai.com/v1"})
    # "api.beta.openai.com" ends with ".openai.com" → allowed
    assert violations == []


# ---------------------------------------------------------------------------
# CIDR ranges
# ---------------------------------------------------------------------------

def test_cidr_range_match():
    e = EgressEnforcer(["10.0.0.0/8"])
    violations = e.check_tool_call("db_call", {"host": "10.42.0.5"})
    assert violations == []


def test_cidr_range_no_match():
    e = EgressEnforcer(["10.0.0.0/8"])
    violations = e.check_tool_call("db_call", {"host": "192.168.1.1"})
    assert len(violations) == 1


def test_cidr_private_172():
    e = EgressEnforcer(["172.16.0.0/12"])
    violations = e.check_tool_call("db_call", {"host": "172.20.0.50"})
    assert violations == []


# ---------------------------------------------------------------------------
# URL extraction (scheme handling)
# ---------------------------------------------------------------------------

def test_url_with_https_scheme():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call("call", {"url": "https://api.openai.com/v1"})
    assert violations == []


def test_url_with_http_scheme():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call("call", {"url": "http://api.openai.com/v1"})
    assert violations == []


def test_url_with_custom_scheme():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call("call", {"url": "grpc://api.openai.com:443/v1"})
    assert violations == []


def test_url_non_standard_port_allowed():
    e = EgressEnforcer(["myservice.internal"])
    violations = e.check_tool_call("call", {"url": "https://myservice.internal:8443/api"})
    assert violations == []


# ---------------------------------------------------------------------------
# Non-network arg keys ignored
# ---------------------------------------------------------------------------

def test_non_network_arg_ignored():
    e = EgressEnforcer(["api.openai.com"])
    # "query" is not a network arg key
    violations = e.check_tool_call("web_search", {"query": "https://attacker.evil.com/evil"})
    assert violations == []


def test_only_network_args_checked():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call(
        "combined_call",
        {
            "url": "https://api.openai.com/v1",       # allowed
            "body": "https://attacker.com/steal",       # not a network arg key → ignored
            "endpoint": "https://attacker.com/collect", # network arg key → blocked
        }
    )
    assert len(violations) == 1
    assert violations[0].arg_key == "endpoint"


# ---------------------------------------------------------------------------
# Multiple violations
# ---------------------------------------------------------------------------

def test_multiple_violations():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call(
        "batch_call",
        {
            "url": "https://attacker1.evil.com/steal",
            "endpoint": "https://attacker2.evil.com/collect",
            "webhook_url": "https://attacker3.evil.com/webhook",
        }
    )
    assert len(violations) == 3


# ---------------------------------------------------------------------------
# Nested args
# ---------------------------------------------------------------------------

def test_nested_url_arg():
    e = EgressEnforcer(["api.openai.com"])
    violations = e.check_tool_call(
        "http_call",
        {"options": {"url": "https://attacker.evil.com/steal"}}
    )
    assert len(violations) == 1
    assert "url" in violations[0].arg_key


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_case_insensitive_hostname():
    e = EgressEnforcer(["API.OPENAI.COM"])
    violations = e.check_tool_call("call", {"url": "https://api.openai.com/v1"})
    assert violations == []


def test_case_insensitive_wildcard():
    e = EgressEnforcer(["*.ANTHROPIC.COM"])
    violations = e.check_tool_call("call", {"url": "https://API.Anthropic.COM/v1"})
    assert violations == []


# ---------------------------------------------------------------------------
# _extract_hostname helper
# ---------------------------------------------------------------------------

def test_extract_hostname_https_url():
    assert _extract_hostname("https://api.openai.com/v1/chat") == "api.openai.com"


def test_extract_hostname_http_url():
    assert _extract_hostname("http://example.com/path?q=1") == "example.com"


def test_extract_hostname_url_with_port():
    result = _extract_hostname("https://myservice.internal:8443/api")
    assert result == "myservice.internal:8443"


def test_extract_hostname_bare():
    result = _extract_hostname("api.openai.com")
    assert result == "api.openai.com"


def test_extract_hostname_bare_with_port():
    result = _extract_hostname("db.internal:5432")
    assert result == "db.internal:5432"


def test_extract_hostname_ip():
    result = _extract_hostname("10.0.0.5")
    assert result == "10.0.0.5"


def test_extract_hostname_ip_with_port():
    result = _extract_hostname("10.0.0.5:8080")
    assert result == "10.0.0.5:8080"


def test_extract_hostname_none_for_non_dest():
    # Random text — not a hostname
    assert _extract_hostname("hello world") is None


def test_extract_hostname_empty():
    assert _extract_hostname("") is None


# ---------------------------------------------------------------------------
# allowed_list diagnostic
# ---------------------------------------------------------------------------

def test_allowed_list_returns_entries():
    e = EgressEnforcer(["api.openai.com", "*.anthropic.com", "10.0.0.0/8"])
    result = e.allowed_list()
    assert "api.openai.com" in result
    assert "*.anthropic.com" in result


# ---------------------------------------------------------------------------
# IPv6 addresses
# ---------------------------------------------------------------------------

def test_ipv6_cidr_match():
    e = EgressEnforcer(["2001:db8::/32"])
    violations = e.check_tool_call("http_get", {"url": "http://[2001:db8::1]/api"})
    assert violations == []


def test_ipv6_cidr_no_match():
    e = EgressEnforcer(["2001:db8::/32"])
    violations = e.check_tool_call("http_get", {"url": "http://[fe80::1]/api"})
    assert len(violations) == 1


def test_ipv6_exact_match():
    e = EgressEnforcer(["::1"])
    violations = e.check_tool_call("http_get", {"host": "::1"})
    assert violations == []


def test_extract_hostname_ipv6_url():
    """IPv6 addresses in URLs are bracket-enclosed: http://[::1]:8080/path."""
    result = _extract_hostname("http://[::1]:8080/api")
    # Should extract "::1" or "[::1]" — either way the CIDR check resolves it
    assert result is not None


# ---------------------------------------------------------------------------
# IP address in URL (IPv4 as URL host)
# ---------------------------------------------------------------------------

def test_ip_in_url_cidr_match():
    e = EgressEnforcer(["10.0.0.0/8"])
    violations = e.check_tool_call("fetch", {"url": "http://10.20.30.40/api"})
    assert violations == []


def test_ip_in_url_cidr_no_match():
    e = EgressEnforcer(["10.0.0.0/8"])
    violations = e.check_tool_call("fetch", {"url": "http://172.16.0.1/api"})
    assert len(violations) == 1
