"""
Argument Value Classifier — SDK Execution Gate.

Classifies actual argument values passed to tool functions into capability
signals. Classification is purely structural:

  - Shannon entropy (mathematical) for credential detection
  - Frozenset membership for prefix / metachar detection
  - stdlib parsers (urllib.parse, ipaddress) for URL / IP detection
  - sqlglot AST for SQL mutation detection (optional; skipped if not installed)
  - stdlib ast for Python code execution detection (dangerous imports / builtins)

No regex on content. No ML. No tool-name inference. Signals are emitted
based solely on argument values regardless of which key or tool name they
appear under — a credential is a credential whether it is in ``token``,
``api_key``, ``x``, or the third positional argument.

Usage::

    from aegivis.security.arg_classifier import ArgClassifier, ClassifierConfig

    cfg   = ClassifierConfig()
    clf   = ArgClassifier(cfg)
    sigs  = clf.classify_call(args=("hello@example.com",), kwargs={})
    # → [ArgSignal(signal="email_destination", path="args[0]", preview="hello@...")]
"""
from __future__ import annotations

import ast
import ipaddress
import math
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Signal type constants
# ---------------------------------------------------------------------------

class Signal:
    """Canonical signal-type identifiers."""
    EMAIL_DESTINATION   = "email_destination"
    CREDENTIAL          = "credential"
    NETWORK_DESTINATION = "network_destination"
    SHELL_METACHAR      = "shell_metachar"
    SQL_MUTATION        = "sql_mutation"
    FILE_PATH           = "file_path"
    BULK_TARGET         = "bulk_target"
    LARGE_PAYLOAD       = "large_payload"
    CODE_EXECUTION      = "code_execution"


ALL_SIGNALS: frozenset[str] = frozenset({
    Signal.EMAIL_DESTINATION,
    Signal.CREDENTIAL,
    Signal.NETWORK_DESTINATION,
    Signal.SHELL_METACHAR,
    Signal.SQL_MUTATION,
    Signal.FILE_PATH,
    Signal.BULK_TARGET,
    Signal.LARGE_PAYLOAD,
    Signal.CODE_EXECUTION,
})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """
    All thresholds and sets used by the classifier.

    Every parameter has a sensible default but can be overridden per-tool.
    No value here is hardcoded in the detection logic — all checks read from
    the config instance passed at construction time.
    """

    # ── Credential detection ──────────────────────────────────────────────

    #: Minimum string length before entropy check is applied.
    #: Shorter strings have inherently low entropy even if random.
    credential_min_length: int = 24

    #: Shannon entropy threshold in bits-per-character.
    #: Human-readable text ≈ 3.0–4.0 bpc; base-64 secrets ≈ 5.5–6.0 bpc.
    #: 4.5 gives a comfortable margin that avoids flagging UUIDs (~3.9 bpc).
    credential_entropy_threshold: float = 4.5

    #: Exact string prefixes that unambiguously indicate a bearer / API token.
    #: Applied before entropy to catch short-but-structured keys.
    credential_prefixes: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "Bearer ",   # HTTP Authorization header value
            "sk-",       # OpenAI / Anthropic secret key
            "AKIA",      # AWS access key ID prefix
            "ghp_",      # GitHub personal access token
            "ghs_",      # GitHub server-to-server token
            "ghr_",      # GitHub refresh token
            "xoxb-",     # Slack bot token
            "xoxp-",     # Slack user token
            "xoxa-",     # Slack app-level token
            "ya29.",     # Google OAuth2 access token
            "eyJ",       # JWT (base64url-encoded JSON object: {"…")
        })
    )

    # ── Shell metacharacter detection ─────────────────────────────────────

    #: Multi-character sequences whose presence in a string argument
    #: indicates possible shell injection or command chaining.
    #: Each entry is checked with the ``in`` operator — no regex.
    shell_danger_sequences: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "$(",    # command substitution
            "`",     # backtick command substitution
            " && ",  # shell AND operator
            " || ",  # shell OR operator
            " | ",   # pipe
            "; ",    # command separator
        })
    )

    # ── Network destination detection ─────────────────────────────────────

    #: URL schemes that indicate outbound network communication.
    network_schemes: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "http", "https", "ftp", "ftps",
            "ws", "wss", "grpc", "grpcs",
        })
    )

    # ── File path detection ───────────────────────────────────────────────

    #: String prefixes that unambiguously indicate a file system path.
    path_prefixes: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "/",      # Unix absolute path
            "~/",     # Unix home-relative path
            "./",     # relative path
            "../",    # parent-relative path
            "C:\\",   # Windows drive path
            "D:\\",   # Windows drive path
            "\\\\",   # UNC network path
        })
    )

    # ── Bulk target detection ─────────────────────────────────────────────

    #: Minimum number of items in a list/tuple to trigger bulk_target check.
    bulk_target_min_items: int = 20

    #: Maximum number of items sampled from a bulk list when checking content.
    bulk_target_sample_size: int = 10

    # ── Large payload detection ───────────────────────────────────────────

    #: String length (characters) above which large_payload is signalled.
    large_payload_chars: int = 100_000

    # ── Python code execution detection ──────────────────────────────────

    #: Module names whose import in a string argument signals dangerous code.
    #: Applied via stdlib ``ast`` — no regex.
    code_danger_modules: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "subprocess",   # arbitrary process execution
            "os",           # os.system, os.popen, os.execv …
            "socket",       # raw network connections
            "ctypes",       # FFI / memory manipulation
            "shutil",       # mass file operations
            "importlib",    # dynamic import bypass
            "pty",          # pseudo-terminal (shell spawn)
        })
    )

    #: Builtin / function names whose call in a string argument signals
    #: dangerous code execution.
    code_danger_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "eval",       # arbitrary expression execution
            "exec",       # arbitrary statement execution
            "compile",    # bytecode compilation
            "__import__", # dynamic import
        })
    )

    # ── Traversal safety ─────────────────────────────────────────────────

    #: Maximum recursion depth when traversing nested dicts / lists.
    max_traversal_depth: int = 6


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgSignal:
    """
    A single capability signal detected in an argument value.

    Attributes:
        signal:  One of the ``Signal.*`` constants.
        path:    Human-readable location string, e.g. ``args[0]`` or
                 ``kwargs['token']``.
        preview: A short, safe preview of the triggering value (never the
                 full value — credentials are truncated after 6 chars).
    """
    signal:  str
    path:    str
    preview: str

    def to_dict(self) -> dict[str, str]:
        return {"signal": self.signal, "path": self.path, "preview": self.preview}


# ---------------------------------------------------------------------------
# Internal pure functions — each returns True/False for one check
# ---------------------------------------------------------------------------

def _shannon_entropy(value: str) -> float:
    """
    Compute Shannon entropy in bits per character.

    H = -Σ p(c) * log2(p(c))   for each unique character c in value.
    Returns 0.0 for empty strings.
    """
    if not value:
        return 0.0
    length = len(value)
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _is_email_shaped(value: str) -> bool:
    """
    Return True if value structurally resembles an e-mail address.

    Purely structural: split on ``@``, verify non-empty local part and a
    domain with at least two dot-separated segments — no regex.
    """
    at_count = value.count("@")
    if at_count != 1:
        return False
    local, domain = value.split("@", 1)
    if not local or not domain:
        return False
    domain_parts = domain.split(".")
    return len(domain_parts) >= 2 and all(domain_parts)


def _is_network_destination(value: str, schemes: frozenset[str]) -> bool:
    """
    Return True if value is a URL with a known network scheme, or a bare
    IP address (v4 or v6), optionally with a port.

    Uses stdlib only: urllib.parse and ipaddress.
    """
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme.lower() in schemes and parsed.netloc:
        return True
    # Bare IP (v4, v6) or IPv4:port — try full string first (handles IPv6),
    # then first colon-segment (handles "1.2.3.4:port").
    for candidate in (value, value.split(":")[0]):
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            pass
    return False


def _has_shell_danger(value: str, sequences: frozenset[str]) -> bool:
    """Return True if value contains any shell danger sequence."""
    return any(seq in value for seq in sequences)


def _is_code_execution(
    value: str,
    danger_modules: frozenset[str],
    danger_builtins: frozenset[str],
) -> bool:
    """
    Return True if value parses as Python code containing dangerous patterns.

    Uses stdlib ``ast`` only — no imports, no regex.

    Two checks:
      1. ``import <danger_module>`` or ``from <danger_module> import …``
      2. Direct call to a danger builtin: ``eval(…)``, ``exec(…)``, etc.

    Returns False immediately if the string is not valid Python syntax,
    so non-code strings (emails, URLs, prose) are always fast-rejected.
    """
    try:
        tree = ast.parse(value)
    except SyntaxError:
        return False
    except Exception:
        return False

    for node in ast.walk(tree):
        # import subprocess  /  import os.path
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in danger_modules:
                    return True

        # from subprocess import run  /  from os import system
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in danger_modules:
                return True

        # eval(...)  /  exec(...)  /  __import__(...)
        elif isinstance(node, ast.Call):
            func = node.func
            # Direct name call: eval(x)
            if isinstance(func, ast.Name) and func.id in danger_builtins:
                return True
            # Attribute call: builtins.eval(x)
            if isinstance(func, ast.Attribute) and func.attr in danger_builtins:
                return True

    return False


def _is_sql_mutation(value: str) -> bool:
    """
    Return True if value parses as a SQL statement with side-effects
    (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE).

    sqlglot is optional; returns False if not installed.
    """
    try:
        import sqlglot               # noqa: PLC0415
        import sqlglot.expressions as e  # noqa: PLC0415
    except ImportError:
        return False

    # Truncate was renamed to TruncateTable in sqlglot ≥ 26; support both.
    _truncate_cls = getattr(e, "TruncateTable", None) or getattr(e, "Truncate", None)
    _MUTATION_NODE_TYPES = tuple(filter(None, (
        e.Insert, e.Update, e.Delete,
        e.Drop, _truncate_cls, e.Alter, e.Create,
    )))
    try:
        statements = sqlglot.parse(value)
        return any(
            isinstance(stmt, _MUTATION_NODE_TYPES)
            for stmt in statements
            if stmt is not None
        )
    except Exception:
        return False


def _is_file_path(value: str, prefixes: frozenset[str]) -> bool:
    """Return True if value starts with a known filesystem path prefix."""
    return any(value.startswith(prefix) for prefix in prefixes)


def _is_credential(
    value: str,
    min_length: int,
    entropy_threshold: float,
    prefixes: frozenset[str],
) -> bool:
    """
    Return True if value looks like a secret credential.

    Two independent checks:
      1. Starts with a known credential prefix.
      2. Long enough AND high Shannon entropy.
    """
    if any(value.startswith(prefix) for prefix in prefixes):
        return True
    return len(value) >= min_length and _shannon_entropy(value) >= entropy_threshold


def _safe_preview(value: str, max_chars: int = 40) -> str:
    """Return a truncated preview safe for logging (never the full value)."""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "…"


def _credential_preview(value: str) -> str:
    """Return a credential-safe preview: first 6 chars + length."""
    return f"{value[:6]}… ({len(value)} chars)"


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class ArgClassifier:
    """
    Classifies argument values into capability signals.

    Thread-safe: all state is in the immutable ``config``; no mutable
    instance state is modified after construction.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._cfg = config or ClassifierConfig()

    # ── Public entry point ────────────────────────────────────────────────

    def classify_call(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> list[ArgSignal]:
        """
        Classify all positional and keyword arguments.

        Returns a flat list of ``ArgSignal`` instances.  Empty list means
        no capability signals were detected.
        """
        signals: list[ArgSignal] = []
        for i, value in enumerate(args):
            self._traverse(value, f"args[{i}]", signals, depth=0)
        for key, value in kwargs.items():
            self._traverse(value, f"kwargs[{key!r}]", signals, depth=0)
        return signals

    # ── Recursive traversal ───────────────────────────────────────────────

    def _traverse(
        self,
        value: Any,
        path: str,
        signals: list[ArgSignal],
        depth: int,
    ) -> None:
        if depth > self._cfg.max_traversal_depth:
            return

        if isinstance(value, str):
            self._classify_string(value, path, signals)

        elif isinstance(value, (list, tuple)):
            self._check_bulk_target(value, path, signals)
            for i, item in enumerate(value):
                self._traverse(item, f"{path}[{i}]", signals, depth + 1)

        elif isinstance(value, dict):
            for k, v in value.items():
                self._traverse(v, f"{path}[{k!r}]", signals, depth + 1)

        # int / float / bool / None → no signals

    def _classify_string(
        self,
        value: str,
        path: str,
        signals: list[ArgSignal],
    ) -> None:
        cfg = self._cfg

        if not value:
            return

        # Large payload — check first (cheap, short-circuits deep scanning
        # of enormous strings)
        if len(value) >= cfg.large_payload_chars:
            signals.append(ArgSignal(
                signal=Signal.LARGE_PAYLOAD,
                path=path,
                preview=f"{len(value):,} characters",
            ))
            return  # don't apply further checks to a massive blob

        # Credential (prefix check before entropy — cheaper)
        if _is_credential(
            value,
            cfg.credential_min_length,
            cfg.credential_entropy_threshold,
            cfg.credential_prefixes,
        ):
            signals.append(ArgSignal(
                signal=Signal.CREDENTIAL,
                path=path,
                preview=_credential_preview(value),
            ))

        # Email address
        if _is_email_shaped(value):
            signals.append(ArgSignal(
                signal=Signal.EMAIL_DESTINATION,
                path=path,
                preview=_safe_preview(value),
            ))

        # Network destination (URL or bare IP)
        if _is_network_destination(value, cfg.network_schemes):
            signals.append(ArgSignal(
                signal=Signal.NETWORK_DESTINATION,
                path=path,
                preview=_safe_preview(value),
            ))

        # Shell metacharacters
        if _has_shell_danger(value, cfg.shell_danger_sequences):
            signals.append(ArgSignal(
                signal=Signal.SHELL_METACHAR,
                path=path,
                preview=_safe_preview(value),
            ))

        # File path
        if _is_file_path(value, cfg.path_prefixes):
            signals.append(ArgSignal(
                signal=Signal.FILE_PATH,
                path=path,
                preview=_safe_preview(value),
            ))

        # SQL mutation (optional; skips if sqlglot not installed)
        if _is_sql_mutation(value):
            signals.append(ArgSignal(
                signal=Signal.SQL_MUTATION,
                path=path,
                preview=_safe_preview(value, max_chars=60),
            ))

        # Python code execution (dangerous imports or builtins)
        if _is_code_execution(value, cfg.code_danger_modules, cfg.code_danger_builtins):
            signals.append(ArgSignal(
                signal=Signal.CODE_EXECUTION,
                path=path,
                preview=_safe_preview(value, max_chars=60),
            ))

    def _check_bulk_target(
        self,
        value: list | tuple,
        path: str,
        signals: list[ArgSignal],
    ) -> None:
        cfg = self._cfg
        if len(value) < cfg.bulk_target_min_items:
            return
        # Sample up to bulk_target_sample_size items to check content
        sample = value[: cfg.bulk_target_sample_size]
        email_hits = sum(
            1 for item in sample
            if isinstance(item, str) and _is_email_shaped(item)
        )
        net_hits = sum(
            1 for item in sample
            if isinstance(item, str)
            and _is_network_destination(item, cfg.network_schemes)
        )
        if email_hits > 0 or net_hits > 0:
            signals.append(ArgSignal(
                signal=Signal.BULK_TARGET,
                path=path,
                preview=f"{len(value)} items ({email_hits} email, {net_hits} network in sample)",
            ))
