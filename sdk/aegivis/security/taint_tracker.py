"""
SDK Cross-Tool Taint Tracker.

Tracks data flowing between tool calls within a session. When a tool
returns data from an untrusted source (e.g. web search, file read,
external API), fragments of that data are stored. If those fragments
later appear as arguments to a subsequent tool call, a taint flow is
detected and reported.

This closes the gap the proxy Session PDG cannot reach: tool-to-tool
data flows that bypass the LLM round-trip (direct chaining), and flows
involving actual Python object values rather than JSON message strings.

Design:
  - No regex. Fragment extraction uses the same structural checks as
    ArgClassifier (email shape, URL scheme, IP address, length threshold).
  - Fail-open: all errors return empty results, never raise to caller.
  - Bounded memory: fragments capped at ``max_fragments`` per session.
    Oldest fragments are evicted on overflow (FIFO).
  - Matching: substring containment with minimum length guard to prevent
    trivial matches on short common strings.
"""
from __future__ import annotations

import ipaddress
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Default untrusted tool name set
# ---------------------------------------------------------------------------

#: Tools whose return values are automatically tracked as untrusted.
#: Developers can override via ``taint_untrusted_tools`` in ``GateConfig``.
#: This is an explicit declaration, not name-inference — the developer
#: decides which tools interact with external / untrusted data sources.
DEFAULT_UNTRUSTED_TOOLS: frozenset[str] = frozenset({
    # Web / HTTP
    "web_search", "search", "fetch", "http_get", "http_post",
    "browse", "web_browse", "web_fetch", "scrape", "crawl",
    "read_url", "get_url", "retrieve", "request",
    # Email
    "email_read", "read_email", "get_email", "fetch_email",
    # File system (can contain injected instructions)
    "read_file", "open_file", "cat_file", "load_file",
    # Database results (can contain injected content)
    "query_db", "sql_query", "db_query", "run_query",
})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaintFragment:
    """
    A string fragment extracted from an untrusted tool's return value.

    Attributes:
        value:       The extracted string fragment.
        source_tool: Name of the tool that returned it.
        recorded_ns: Monotonic nanoseconds when recorded (for FIFO eviction).
    """
    value:       str
    source_tool: str
    recorded_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class TaintFlow:
    """
    A detected cross-tool data flow: an untrusted fragment appeared in
    a subsequent tool's arguments.

    Attributes:
        fragment:      The matching fragment string.
        source_tool:   Tool that originally returned the fragment.
        sink_tool:     Tool that received it as an argument.
        sink_arg_path: Argument location, e.g. ``kwargs['to']``.
    """
    fragment:      str
    source_tool:   str
    sink_tool:     str
    sink_arg_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "fragment":      self.fragment[:80],  # preview only
            "source_tool":   self.source_tool,
            "sink_tool":     self.sink_tool,
            "sink_arg_path": self.sink_arg_path,
        }


# ---------------------------------------------------------------------------
# Fragment extraction helpers
# ---------------------------------------------------------------------------

def _is_email_fragment(value: str) -> bool:
    """True if value structurally looks like an email address."""
    if value.count("@") != 1:
        return False
    local, domain = value.split("@", 1)
    if not local or not domain:
        return False
    parts = domain.split(".")
    return len(parts) >= 2 and all(parts)


def _is_url_fragment(value: str) -> bool:
    """True if value is a URL with a network scheme and a netloc."""
    _NETWORK_SCHEMES = frozenset({"http", "https", "ftp", "ws", "wss"})
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme.lower() in _NETWORK_SCHEMES and bool(parsed.netloc)


def _is_ip_fragment(value: str) -> bool:
    """True if value is a bare IPv4 or IPv6 address."""
    for candidate in (value, value.split(":")[0]):
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            pass
    return False


def _extract_from_string(
    value: str,
    min_len: int,
    results: list[str],
) -> None:
    """
    Extract significant fragments from a single string value.

    A fragment is extracted if it is:
      - An email-shaped string (always, regardless of length).
      - A URL string (always).
      - A bare IP address (always).
      - A string of length >= ``min_len`` (general significant string).

    Duplicates within a single extraction pass are allowed — the caller
    deduplicates across the full session via the fragment set.
    """
    if not value or len(value) < 4:
        return

    # Structural checks — extract even if short
    if _is_email_fragment(value) or _is_url_fragment(value) or _is_ip_fragment(value):
        results.append(value)
        return

    # General length threshold
    if len(value) >= min_len:
        results.append(value)


def _extract_from_value(
    value: Any,
    min_len: int,
    results: list[str],
    depth: int = 0,
) -> None:
    """Recursively extract fragments from a return value (dict, list, str)."""
    if depth > 4:
        return

    if isinstance(value, str):
        _extract_from_string(value, min_len, results)

    elif isinstance(value, dict):
        for v in value.values():
            _extract_from_value(v, min_len, results, depth + 1)

    elif isinstance(value, (list, tuple)):
        for item in value[:30]:  # sample up to 30 items
            _extract_from_value(item, min_len, results, depth + 1)


# ---------------------------------------------------------------------------
# Argument string collection
# ---------------------------------------------------------------------------

def _collect_arg_strings(
    value: Any,
    path: str,
    results: list[tuple[str, str]],
    depth: int = 0,
) -> None:
    """
    Collect all string values from tool arguments with their paths.

    Returns list of ``(string_value, path)`` tuples for matching.
    Ignores strings shorter than 4 chars (too short to match meaningfully).
    """
    if depth > 5:
        return

    if isinstance(value, str):
        if len(value) >= 4:
            results.append((value, path))

    elif isinstance(value, dict):
        for k, v in value.items():
            _collect_arg_strings(v, f"{path}[{k!r}]", results, depth + 1)

    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value[:20]):
            _collect_arg_strings(item, f"{path}[{i}]", results, depth + 1)


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class SDKTaintTracker:
    """
    Per-session cross-tool taint tracker.

    Thread-safe via a single lock — acquisition is brief (list operations).

    Args:
        untrusted_tools:  Tool names whose return values are tracked.
                          Only fragments from these tools are stored.
        min_fragment_len: Minimum string length for non-structural fragments
                          (emails, URLs, IPs are always tracked regardless).
        max_fragments:    Maximum stored fragments per session.
                          Oldest are evicted (FIFO) when limit is reached.
        min_match_len:    Minimum length of a fragment for it to trigger a
                          taint flow detection (prevents trivial matches).
    """

    def __init__(
        self,
        untrusted_tools: frozenset[str] = DEFAULT_UNTRUSTED_TOOLS,
        min_fragment_len: int = 15,
        max_fragments: int = 500,
        min_match_len: int = 8,
    ) -> None:
        self._untrusted = untrusted_tools
        self._min_fragment_len = min_fragment_len
        self._max_fragments = max_fragments
        self._min_match_len = min_match_len
        self._fragments: list[TaintFragment] = []
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def record_return(self, tool_name: str, result: Any) -> None:
        """
        Extract fragments from a tool's return value and store them.

        Only records fragments if ``tool_name`` is in ``untrusted_tools``.
        Silently no-ops on any extraction error.
        """
        if tool_name not in self._untrusted:
            return
        try:
            raw: list[str] = []
            _extract_from_value(result, self._min_fragment_len, raw)
            if not raw:
                return
            now = time.monotonic_ns()
            new_frags = [
                TaintFragment(value=v, source_tool=tool_name, recorded_ns=now)
                for v in raw
            ]
            with self._lock:
                self._fragments.extend(new_frags)
                if len(self._fragments) > self._max_fragments:
                    # Evict oldest fragments (FIFO)
                    self._fragments = self._fragments[-self._max_fragments:]
        except Exception:
            pass  # fail-open: taint tracking must not break tool execution

    def check_args(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict,
    ) -> list[TaintFlow]:
        """
        Check whether any stored untrusted fragment appears in the arguments.

        Returns a list of ``TaintFlow`` instances — one per unique
        (fragment, arg_path) pair detected. Empty list means no taint flows.
        Silently returns empty list on any error.
        """
        try:
            with self._lock:
                fragments = list(self._fragments)  # snapshot under lock

            if not fragments:
                return []

            # Collect all string values from args
            arg_strings: list[tuple[str, str]] = []
            for i, v in enumerate(args):
                _collect_arg_strings(v, f"args[{i}]", arg_strings)
            for k, v in kwargs.items():
                _collect_arg_strings(v, f"kwargs[{k!r}]", arg_strings)

            if not arg_strings:
                return []

            flows: list[TaintFlow] = []
            seen: set[tuple[str, str]] = set()  # deduplicate (fragment, path)

            for frag in fragments:
                if len(frag.value) < self._min_match_len:
                    continue
                for arg_val, arg_path in arg_strings:
                    if frag.value in arg_val:
                        key = (frag.value, arg_path)
                        if key not in seen:
                            seen.add(key)
                            flows.append(TaintFlow(
                                fragment=frag.value,
                                source_tool=frag.source_tool,
                                sink_tool=tool_name,
                                sink_arg_path=arg_path,
                            ))

            return flows
        except Exception:
            return []  # fail-open

    def fragment_count(self) -> int:
        """Current number of stored fragments (for testing / metrics)."""
        with self._lock:
            return len(self._fragments)

    def clear(self) -> None:
        """Clear all stored fragments (for testing)."""
        with self._lock:
            self._fragments.clear()
