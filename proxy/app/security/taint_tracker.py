"""
Data Flow Taint Tracker.

Tags credential values the moment they enter the agent context (system prompt,
tool results). When the agent fires a tool call, every argument is checked
against the taint store. Exact substring match — near-zero false positive rate.

Threat model:
  An attacker (via indirect injection or compromised tool) causes the agent to
  call a network-sink tool (http_request, send_email, webhook) with a live
  credential as an argument. The credential may have come from the system
  prompt or a previous tool result. This scanner detects and blocks that.

Why this beats output scanning for exfiltration:
  - Output scanning scans what the LLM *says* (inferential, high FP).
  - Taint tracking enforces *what the agent is about to execute* (structural).
  - Exact match on credential value → near-zero FP.
  - Enforcement at TOOL_CALL_START → before the tool executes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaintedValue:
    value: str    # raw secret — NEVER log
    label: str    # e.g. "openai_key", "jwt_token"
    source: str   # "system_prompt" | "tool_result:{tool_name}"


@dataclass
class TaintHit:
    label: str
    source: str
    arg_key: str
    tool_name: str
    is_network_sink: bool


# ---------------------------------------------------------------------------
# Network sink detection
# ---------------------------------------------------------------------------

_SINK_NAME_RE = re.compile(
    r"http|request|fetch|curl|send|email|mail|webhook|upload|"
    r"push|publish|notify|slack|teams|discord|sms|forward|relay|export",
    re.IGNORECASE,
)

_SINK_ARG_KEYS: frozenset[str] = frozenset({
    "url", "endpoint", "webhook", "webhook_url", "uri",
    "destination", "target", "to", "recipient", "address",
    "server", "api_url", "base_url", "callback", "callback_url",
})


# ---------------------------------------------------------------------------
# TaintTracker
# ---------------------------------------------------------------------------

class TaintTracker:
    """Per-session credential taint store. Not persisted to Redis.

    Limitation: If the proxy restarts mid-session, this in-memory taint store
    is lost. A multi-turn exfiltration chain that began before the restart will
    not be detected — the taint store starts empty on restart. The Session PDG
    (session_pdg.py) has the same limitation. Mitigation: rolling restart windows
    should be short relative to typical session duration, and the canary token
    system provides a complementary detection layer that is stateless.
    """

    def __init__(self) -> None:
        self._taints: list[TaintedValue] = []
        self._seen: set[str] = set()  # dedup by value

    def taint(self, value: str, label: str, source: str) -> None:
        """Add a credential to the taint store. Min length 8, deduplicated."""
        if len(value) < 8 or value in self._seen:
            return
        self._seen.add(value)
        self._taints.append(TaintedValue(value=value, label=label, source=source))

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> list[TaintHit]:
        """Scan all tool args for tainted values. Returns hits (possibly empty)."""
        if not self._taints:
            return []
        hits: list[TaintHit] = []
        sink = self._is_network_sink(tool_name, args)
        flat = _flatten_args(args)
        for key, str_val in flat:
            for tv in self._taints:
                if tv.value in str_val:
                    hits.append(TaintHit(
                        label=tv.label,
                        source=tv.source,
                        arg_key=key,
                        tool_name=tool_name,
                        is_network_sink=sink,
                    ))
        return hits

    def _is_network_sink(self, tool_name: str, args: dict[str, Any]) -> bool:
        if _SINK_NAME_RE.search(tool_name):
            return True
        return any(k in _SINK_ARG_KEYS for k in args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_args(args: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Recursively flatten args dict to (key_path, str_value) pairs."""
    result: list[tuple[str, str]] = []
    for k, v in args.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            result.append((key, v))
        elif isinstance(v, dict):
            result.extend(_flatten_args(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    result.append((f"{key}[{i}]", item))
                elif isinstance(item, dict):
                    result.extend(_flatten_args(item, f"{key}[{i}]"))
    return result


import re as _re

# Credential patterns for taint extraction: (compiled_re, label)
# Each pattern matches a full credential value (not truncated).
_CREDENTIAL_PATTERNS: list[tuple["_re.Pattern[str]", str]] = [
    (_re.compile(r"sk-[A-Za-z0-9]{20,}"),                                "openai_key"),
    (_re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),                          "anthropic_key"),
    (_re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "jwt_token"),
    (_re.compile(r"AKIA[0-9A-Z]{16}"),                                   "aws_access_key"),
    (_re.compile(r"(?i)Bearer\s+([A-Za-z0-9._\-]{20,})"),               "bearer_token"),
    (_re.compile(r"ghp_[A-Za-z0-9]{36}"),                                "github_pat"),
    (_re.compile(r"ghr_[A-Za-z0-9]{36}"),                                "github_refresh"),
    (_re.compile(r"(?i)api[_-]?key[\"'\s:=]+([A-Za-z0-9._\-]{16,})"),  "generic_api_key"),
]


def extract_credentials_from_text(text: str) -> list[tuple[str, str]]:
    """
    Extract (value, label) pairs for known credential formats found in *text*.

    Returns full matched strings with a minimum length of 8 characters.
    """
    results: list[tuple[str, str]] = []
    for pattern, label in _CREDENTIAL_PATTERNS:
        for m in pattern.finditer(text):
            # Use group(1) if it captured a sub-group (e.g. Bearer), else group(0)
            val = m.group(1) if m.lastindex else m.group(0)
            if len(val) >= 8:
                results.append((val, label))
    return results
