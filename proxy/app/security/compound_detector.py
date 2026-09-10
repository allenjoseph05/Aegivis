"""
Compound Sequence Detector — Phase 23.

Detects dangerous multi-step action sequences within a session.  A single tool
call may be benign; the combination of several calls across the session reveals
malicious intent that no per-call scanner can see.

Examples of compound patterns:
  cred-exfil:         credential_access → network_send
  data-exfil-chain:   data_read → encode → network_send
  rce-persistence:    code_exec → file_write → process_spawn
  recon-exfil:        recon → credential_access → network_send
  env-exfil:          env_read → network_send
  mem-dump-exfil:     memory_access → network_send
  auth-bypass-recon:  data_read → auth_modify

Architecture:
  - ``classify_tool_intent(tool_name)`` — tokenises the tool name and maps it
    to one of the intent classes below.  Purely structural: frozenset lookups
    on lowercased tokens.  No regex.  No content scanning.
  - ``SequencePattern`` — a named ordered subsequence of intent classes that
    constitutes a dangerous pattern.
  - ``PatternMatch`` — returned when a pattern is completed.
  - ``CompoundSequenceDetector.check()`` — called at each TOOL_CALL_START with
    the session's accumulated intent history.  Returns any patterns that are
    newly completed by this call.
  - ``CompoundSequenceDetector.check()`` is O(P × H) where P is the number of
    patterns and H is the history length.  Both are small in practice.

Integration in intercept.py (after blast radius hook):
    intent_class = classify_tool_intent(tc_name)
    session.intent_history.append(intent_class)   # new session slot
    matches = _compound_detector.check(intent_class, session.intent_history)
    for match in matches:
        # fire "compound-sequence-violation" ALERT/BLOCK violation

Session slot:
    ``intent_history: list[str]``  — list of intent classes seen this session
    (initialised to [] in SessionState, not persisted — runtime only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent class taxonomy — immutable frozensets, O(1) membership
#
# Each set contains verb *tokens* (individual words after splitting the tool
# name on ``_`` / ``-``).  A tool name matches a class when ANY token from
# the name appears in the class set.
# ---------------------------------------------------------------------------

#: Accessing secrets, API keys, passwords, certificates, tokens.
_CREDENTIAL_ACCESS: frozenset[str] = frozenset({
    "credential", "credentials", "secret", "secrets",
    "password", "passwd", "apikey", "api", "token", "tokens",
    "key", "keys", "cert", "certificate", "ssh", "private",
    "vault", "keychain", "keystore", "auth",
})

#: Reading environment variables (common source of secrets).
_ENV_READ: frozenset[str] = frozenset({
    "env", "environment", "envvar", "environ", "dotenv", "config",
    "configuration", "settings", "getenv",
})

#: Reading data from persistent storage (files, DB, memory cache).
_DATA_READ: frozenset[str] = frozenset({
    "read", "fetch", "load", "get", "query", "select", "search",
    "lookup", "retrieve", "scan", "dump", "export", "download",
    "pull", "extract",
})

#: Listing / enumerating resources — reconnaissance.
_RECON: frozenset[str] = frozenset({
    "list", "enumerate", "ls", "dir", "find", "discover",
    "inventory", "stat", "info", "describe", "show", "check",
})

#: Accessing process memory or heap (advanced exfiltration vector).
_MEMORY_ACCESS: frozenset[str] = frozenset({
    "memory", "heap", "stack", "core", "memdump", "ptrace",
    "inspect",
})

#: Encoding / compression / encryption (often precedes exfiltration).
_ENCODE: frozenset[str] = frozenset({
    "encode", "encoding", "base64", "compress", "compress",
    "encrypt", "zip", "pack", "serialize", "serialise", "marshal",
    "gzip", "bzip", "zlib",
})

#: Writing files to disk.
_FILE_WRITE: frozenset[str] = frozenset({
    "write", "save", "store", "create", "append", "put",
    "output", "log", "record", "persist",
})

#: Executing code or shell commands.
_CODE_EXEC: frozenset[str] = frozenset({
    "execute", "exec", "run", "eval", "shell", "bash", "sh",
    "cmd", "invoke", "call", "launch",
    "compile", "interpret", "script",
})

#: Spawning OS processes.
_PROCESS_SPAWN: frozenset[str] = frozenset({
    "process", "subprocess", "spawn", "fork", "daemon",
    "service", "start", "job", "task", "cron",
})

#: Sending data to external destinations.
_NETWORK_SEND: frozenset[str] = frozenset({
    "send", "email", "mail", "smtp", "webhook", "notify",
    "post", "submit", "upload", "broadcast", "relay", "forward",
    "push", "emit", "dispatch", "publish", "message", "slack",
    "teams", "sms", "fax",
})

#: Modifying authentication, permissions, or user accounts.
_AUTH_MODIFY: frozenset[str] = frozenset({
    "grant", "revoke", "elevate", "escalate", "promote", "assign",
    "permission", "permissions", "role", "roles", "user", "users",
    "account", "admin", "sudo", "privilege", "access",
    "authorize", "unauthorize", "allow", "deny",
})

#: Ordered list: (intent_class_name, frozenset_of_tokens)
#: First match wins — most specific classes should come first.
_INTENT_CLASSES: list[tuple[str, frozenset[str]]] = [
    ("credential_access", _CREDENTIAL_ACCESS),
    ("env_read",          _ENV_READ),
    ("memory_access",     _MEMORY_ACCESS),
    ("encode",            _ENCODE),
    ("code_exec",         _CODE_EXEC),
    ("process_spawn",     _PROCESS_SPAWN),
    ("network_send",      _NETWORK_SEND),
    ("recon",             _RECON),
    ("auth_modify",       _AUTH_MODIFY),
    ("file_write",        _FILE_WRITE),
    ("data_read",         _DATA_READ),
]


def classify_tool_intent(tool_name: str) -> str:
    """
    Return the intent class for a tool name.

    Structural: lower-cases the tool name, splits on ``_`` and ``-``,
    checks each token against intent class frozensets.  First match wins.

    Returns ``"unknown"`` if no class matches.
    """
    normalized = tool_name.lower().replace("-", "_").replace(".", "_")
    tokens = set(normalized.split("_"))
    # Remove empty tokens (consecutive delimiters)
    tokens.discard("")

    for class_name, verb_set in _INTENT_CLASSES:
        if tokens & verb_set:  # non-empty intersection → match
            return class_name

    return "unknown"


# ---------------------------------------------------------------------------
# Sequence patterns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequencePattern:
    """
    A named compound sequence pattern.

    ``steps`` is an ordered tuple of intent classes.  A pattern is matched
    when all steps appear in the session's intent history in the given order
    (subsequence matching — steps need not be adjacent).

    This means ``cred-exfil`` with steps ``("credential_access", "network_send")``
    matches any session where *at least one* credential_access came *before*
    *at least one* network_send, regardless of what happened in between.
    """
    name: str
    description: str
    steps: tuple[str, ...]
    severity: str   # "critical", "high", "medium"


#: Built-in compound sequence patterns.
#: New patterns can be appended by operators after module load.
PATTERNS: list[SequencePattern] = [
    SequencePattern(
        name="cred-exfil",
        description="Credential access followed by external data transmission — classic exfiltration chain.",
        steps=("credential_access", "network_send"),
        severity="critical",
    ),
    SequencePattern(
        name="env-exfil",
        description="Environment variable read (likely containing API keys) followed by network send.",
        steps=("env_read", "network_send"),
        severity="critical",
    ),
    SequencePattern(
        name="data-exfil-encoded",
        description="Data read, encoded (base64/compress/encrypt), then transmitted externally.",
        steps=("data_read", "encode", "network_send"),
        severity="critical",
    ),
    SequencePattern(
        name="mem-dump-exfil",
        description="Memory access (heap/core dump) followed by external transmission.",
        steps=("memory_access", "network_send"),
        severity="critical",
    ),
    SequencePattern(
        name="recon-cred-exfil",
        description="Reconnaissance, then credential harvest, then exfiltration — full attack chain.",
        steps=("recon", "credential_access", "network_send"),
        severity="critical",
    ),
    SequencePattern(
        name="rce-persistence",
        description="Code execution followed by file write and process spawn — malware persistence pattern.",
        steps=("code_exec", "file_write", "process_spawn"),
        severity="high",
    ),
    SequencePattern(
        name="recon-auth-escalation",
        description="Reconnaissance followed by authentication/permission modification.",
        steps=("recon", "auth_modify"),
        severity="high",
    ),
    SequencePattern(
        name="data-read-auth-modify",
        description="Data read (possibly config files) followed by permission modification.",
        steps=("data_read", "auth_modify"),
        severity="high",
    ),
    SequencePattern(
        name="cred-access-code-exec",
        description="Credential access followed by code execution — may use stolen credentials to run privileged code.",
        steps=("credential_access", "code_exec"),
        severity="high",
    ),
    SequencePattern(
        name="env-read-code-exec",
        description="Environment variable read followed by code execution — key injection into spawned process.",
        steps=("env_read", "code_exec"),
        severity="medium",
    ),
    SequencePattern(
        name="file-write-process-spawn",
        description="File write followed by process spawn — write-and-execute pattern.",
        steps=("file_write", "process_spawn"),
        severity="medium",
    ),
]


# ---------------------------------------------------------------------------
# PatternMatch
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """
    Returned when a compound sequence pattern is completed.

    Attributes:
        pattern_name:      The name of the matched pattern.
        description:       Human-readable description of the pattern.
        severity:          ``"critical"``, ``"high"``, or ``"medium"``.
        completing_class:  Intent class of the tool that completed the pattern.
        steps_matched:     Ordered list of intent classes that satisfied the
                           pattern steps (length == len(pattern.steps)).
    """
    pattern_name: str
    description: str
    severity: str
    completing_class: str
    steps_matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "severity": self.severity,
            "completing_class": self.completing_class,
            "steps_matched": self.steps_matched,
        }


# ---------------------------------------------------------------------------
# CompoundSequenceDetector
# ---------------------------------------------------------------------------

def _is_subsequence(pattern_steps: tuple[str, ...], history: list[str]) -> bool:
    """
    Return True if ``pattern_steps`` is a subsequence of ``history``.

    A subsequence means every element of ``pattern_steps`` appears in
    ``history`` in the same relative order, but not necessarily adjacently.

    O(len(pattern_steps) + len(history)).
    """
    step_idx = 0
    for item in history:
        if step_idx == len(pattern_steps):
            break
        if item == pattern_steps[step_idx]:
            step_idx += 1
    return step_idx == len(pattern_steps)


def _extract_subsequence_witness(
    pattern_steps: tuple[str, ...], history: list[str]
) -> list[str]:
    """
    Return the matched subsequence elements (for diagnostics).

    Same algorithm as _is_subsequence but collects the matched items.
    Returns an empty list if the pattern does not match.
    """
    result: list[str] = []
    step_idx = 0
    for item in history:
        if step_idx == len(pattern_steps):
            break
        if item == pattern_steps[step_idx]:
            result.append(item)
            step_idx += 1
    if step_idx < len(pattern_steps):
        return []
    return result


class CompoundSequenceDetector:
    """
    Detects compound attack sequences across a session's tool call history.

    Usage::

        detector = CompoundSequenceDetector()

        # At each TOOL_CALL_START (intercept.py):
        intent = classify_tool_intent(tool_name)
        session.intent_history.append(intent)
        matches = detector.check(intent, session.intent_history)
        for match in matches:
            # fire "compound-sequence-violation" violation

    The detector checks only patterns that the *current* call could complete
    (i.e. patterns whose last step == current intent class).  This avoids
    re-firing patterns that were already completed earlier in the session by
    tracking which patterns have already been flagged.
    """

    def __init__(self, patterns: list[SequencePattern] | None = None) -> None:
        """
        Args:
            patterns: List of patterns to check.  Defaults to the built-in
                      ``PATTERNS`` list.  Pass a custom list for testing or
                      operator-defined patterns.
        """
        self._patterns: list[SequencePattern] = patterns if patterns is not None else PATTERNS

    def check(
        self,
        current_intent: str,
        full_history: list[str],
    ) -> list[PatternMatch]:
        """
        Check whether the current tool call completes any sequence pattern.

        Args:
            current_intent: Intent class of the tool just called (should equal
                            ``full_history[-1]`` if the caller has already
                            appended it).
            full_history:   Complete ordered list of intent classes for this
                            session, including the current call.

        Returns:
            List of ``PatternMatch`` objects for each pattern that is now
            complete.  Empty list if no pattern is completed.

        Note:
            A pattern may match multiple times if the session repeats the same
            sequence.  Callers that want to fire at most once per pattern
            per session should track previously fired pattern names externally.
        """
        matches: list[PatternMatch] = []

        for pattern in self._patterns:
            # Fast pre-filter: current intent must match the last step of the
            # pattern.  No point running the full subsequence check otherwise.
            if pattern.steps[-1] != current_intent:
                continue

            witness = _extract_subsequence_witness(pattern.steps, full_history)
            if witness:
                logger.info(
                    "[COMPOUND-SEQ] Pattern '%s' (%s) completed — steps: %s",
                    pattern.name, pattern.severity, witness,
                )
                matches.append(PatternMatch(
                    pattern_name=pattern.name,
                    description=pattern.description,
                    severity=pattern.severity,
                    completing_class=current_intent,
                    steps_matched=witness,
                ))

        return matches


#: Module-level singleton for use by intercept.py.
compound_detector = CompoundSequenceDetector()
