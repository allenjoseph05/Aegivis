"""
Blast Radius Guard — Phase 15 (Excessive Agency Prevention).

Research basis:
  - OWASP LLM Top 10 2025 LLM06: Excessive Agency
  - Borjigin et al., arXiv:2603.10092 — Survivability-Aware Execution (SAE)
    Exposure budgets, staged execution, tool/venue allowlists.
  - Su et al., arXiv:2506.23844 — R2A2 (Constrained MDP formulation)
    Sequential composition dramatically amplifies blast radius.
  - SAFE-AI, arXiv:2508.11824 — autonomous/destructive boundary checkpoints.
    Reversible vs irreversible as the primary safety gate.

Core idea: classify every tool call by (reversibility × scope) BEFORE it executes.
No ML, no regex on content, no hardcoded tool names.

Classification is purely structural — three independent signals:

  1. Verb taxonomy  — tokenize tool name → match against immutable verb sets
     → reversibility class.  Operators can override via capability manifest
     ``reversibility_class`` field, which always takes precedence.

  2. Argument scope — examine argument VALUES structurally:
     • SQL: sqlglot AST detects DDL / DML without WHERE (stdlib fallback available)
     • File: recursive flag + directory paths, glob patterns, root paths
     • General: wildcard characters (* ? %) in any string argument

  3. Spawn amplifier — score scaled by 1 + 0.25 × spawn_depth so sub-agents
     face tighter thresholds than root agents (R2A2 Constrained MDP property).

Blast radius score = verb_risk × scope_amplifier × spawn_amplifier

  0.00–0.30 → SAFE      (allow)
  0.30–0.60 → MEDIUM    (ALERT)
  0.60–0.85 → HIGH      (HITL required — next LLM call held for human review)
  0.85–1.00 → CRITICAL  (BLOCK immediately + HITL)

Session accumulator (from SAE "exposure budget"):
  Cumulative blast score per session.  When total > budget threshold, all
  subsequent HIGH-risk calls require HITL regardless of individual score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verb taxonomy — immutable frozensets, O(1) membership test
# ---------------------------------------------------------------------------

_VERBS_IRREVERSIBLE_HIGH: frozenset[str] = frozenset({
    "delete", "destroy", "terminate", "purge", "wipe", "drop", "truncate",
    "format", "erase", "remove", "rm", "unlink", "revoke", "deactivate",
    "drain", "kill", "nuke", "clear", "flush",
})

_VERBS_PRIVILEGED: frozenset[str] = frozenset({
    "grant", "elevate", "escalate", "promote", "assign", "attach",
    "transfer", "authorize", "demote",
})

_VERBS_IRREVERSIBLE_MEDIUM: frozenset[str] = frozenset({
    "update", "modify", "patch", "overwrite", "replace", "move", "rename",
    "archive", "migrate", "rewrite", "reset", "rotate", "swap",
})

_VERBS_EXTERNAL_SEND: frozenset[str] = frozenset({
    "send", "email", "webhook", "notify", "publish", "post", "submit",
    "upload", "broadcast", "relay", "forward", "push", "emit", "dispatch",
})

# Ordered from highest → lowest risk for first-match precedence
_VERB_CLASSES: list[tuple[frozenset[str], str, float]] = [
    (_VERBS_IRREVERSIBLE_HIGH,   "irreversible_high",   0.85),
    (_VERBS_PRIVILEGED,          "privileged",          0.65),
    (_VERBS_IRREVERSIBLE_MEDIUM, "irreversible_medium", 0.45),
    (_VERBS_EXTERNAL_SEND,       "external_send",       0.30),
]

# Union of all classified verbs (for signal reporting)
_VERB_SET_ALL: frozenset[str] = (
    _VERBS_IRREVERSIBLE_HIGH | _VERBS_PRIVILEGED |
    _VERBS_IRREVERSIBLE_MEDIUM | _VERBS_EXTERNAL_SEND
)

# Capability manifest reversibility_class → base risk
_MANIFEST_CLASS_RISK: dict[str, float] = {
    "irreversible_high":   0.85,
    "privileged":          0.65,
    "irreversible_medium": 0.45,
    "external_send":       0.30,
    "external_source":     0.10,
    "internal_sink":       0.20,
    "internal":            0.05,
    "reversible":          0.05,
    "safe":                0.00,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BlastRadiusResult:
    """Classification result for a single tool call."""
    verb_class: str      # taxonomy class or "safe"
    verb_risk: float     # 0.0–1.0 base verb risk
    scope_risk: float    # 0.0–1.0 argument scope risk
    blast_score: float   # 0.0–1.0 final score (with spawn amplification)
    signals: list[str] = field(default_factory=list)
    manifest_override: bool = False

    @property
    def risk_level(self) -> str:
        if self.blast_score >= 0.85: return "CRITICAL"
        if self.blast_score >= 0.60: return "HIGH"
        if self.blast_score >= 0.30: return "MEDIUM"
        return "SAFE"

    @property
    def should_block(self) -> bool:
        return self.blast_score >= 0.85

    @property
    def should_hitl(self) -> bool:
        return self.blast_score >= 0.60

    def to_dict(self) -> dict:
        return {
            "blast_score":       round(self.blast_score, 4),
            "risk_level":        self.risk_level,
            "verb_class":        self.verb_class,
            "scope_risk":        round(self.scope_risk, 4),
            "signals":           self.signals[:10],  # cap for JSONB storage
            "manifest_override": self.manifest_override,
        }


# ---------------------------------------------------------------------------
# Tool name tokenizer (no regex — pure character walk)
# ---------------------------------------------------------------------------

def _tokenize_name(name: str) -> list[str]:
    """
    Split a tool name into lowercase tokens.

    Handles snake_case, kebab-case, dot.notation, slashes, and camelCase /
    PascalCase boundaries.  Tokens shorter than 2 characters are dropped
    (they are single-char abbreviations, not meaningful verbs).

    Examples:
        deleteUser          → ["delete", "user"]
        db_drop_table       → ["db", "drop", "table"]
        CreateIAMPolicy     → ["create", "ia", "m", "policy"]  (rare ALLCAPS OK)
        send-email-webhook  → ["send", "email", "webhook"]
    """
    tokens: list[str] = []
    current: list[str] = []

    for ch in name:
        if ch in ('_', '-', ' ', '.', '/', ':', '\\'):
            if current:
                tokens.append(''.join(current).lower())
                current = []
        elif ch.isupper() and current and current[-1].islower():
            # camelCase / PascalCase boundary
            tokens.append(''.join(current).lower())
            current = [ch]
        else:
            current.append(ch)

    if current:
        tokens.append(''.join(current).lower())

    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Verb classification
# ---------------------------------------------------------------------------

def _classify_verb(
    tool_name: str,
    manifest_entry: dict | None = None,
) -> tuple[str, float, bool]:
    """
    Return (verb_class, base_risk, manifest_override).

    Manifest entry wins if it has a reversibility_class field.
    Otherwise the tool name is tokenized and matched against the verb taxonomy.
    Unknown tools default to ("safe", 0.0, False) — operators must classify.
    """
    # Manifest override (highest priority — operator explicitly classified this)
    if manifest_entry is not None:
        rc = manifest_entry.get("reversibility_class")
        if rc and rc in _MANIFEST_CLASS_RISK:
            return rc, _MANIFEST_CLASS_RISK[rc], True

    # Verb heuristic: find the highest-risk verb in the token set
    tokens = set(_tokenize_name(tool_name))
    for verb_set, verb_class, risk in _VERB_CLASSES:
        if tokens & verb_set:
            return verb_class, risk, False

    return "safe", 0.0, False


# ---------------------------------------------------------------------------
# SQL scope analysis
# ---------------------------------------------------------------------------

def _sql_scope(args: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Assess scope risk of SQL statement arguments.

    Finds string args that look like SQL statements (start with a SQL keyword).
    Uses sqlglot AST if available; falls back to structural string analysis.
    """
    sql_texts: list[str] = []
    _SQL_STARTERS = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP ",
                     "TRUNCATE ", "ALTER ", "CREATE ", "REPLACE ")
    for v in args.values():
        if isinstance(v, str) and len(v) >= 6:
            upper = v.strip().upper()
            if any(upper.startswith(s) for s in _SQL_STARTERS):
                sql_texts.append(v)

    if not sql_texts:
        return 0.0, []

    best_risk = 0.0
    best_signals: list[str] = []
    for sql in sql_texts:
        risk, sigs = _parse_sql_risk(sql)
        if risk > best_risk:
            best_risk = risk
            best_signals = sigs

    return best_risk, best_signals


def _parse_sql_risk(sql: str) -> tuple[float, list[str]]:
    try:
        return _parse_sql_ast(sql)
    except Exception:
        return _parse_sql_structural(sql)


def _parse_sql_ast(sql: str) -> tuple[float, list[str]]:
    """AST-level SQL scope risk using sqlglot (if installed)."""
    import sqlglot
    import sqlglot.expressions as exp

    statements = sqlglot.parse(sql) or []
    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, exp.Drop):
            return 1.0, [f"SQL DROP statement — irreversible schema destruction"]
        if isinstance(stmt, exp.TruncateTable):
            return 1.0, ["SQL TRUNCATE — irreversible table wipe"]
        if isinstance(stmt, exp.Create) and stmt.args.get("replace"):
            return 0.8, ["SQL CREATE OR REPLACE — destructive schema change"]
        if isinstance(stmt, exp.Delete):
            if not stmt.find(exp.Where):
                return 0.9, ["SQL DELETE without WHERE — unbounded row deletion (all rows affected)"]
            return 0.25, ["SQL DELETE with WHERE clause (bounded scope)"]
        if isinstance(stmt, exp.Update):
            if not stmt.find(exp.Where):
                return 0.75, ["SQL UPDATE without WHERE — unbounded row modification"]
            return 0.15, ["SQL UPDATE with WHERE clause (bounded scope)"]
    return 0.0, []


def _parse_sql_structural(sql: str) -> tuple[float, list[str]]:
    """Structural fallback SQL risk analysis (no external library required)."""
    upper = sql.strip().upper()
    if upper.startswith("DROP ") or upper.startswith("TRUNCATE "):
        return 1.0, ["SQL DDL — irreversible schema operation"]
    if upper.startswith("DELETE ") or upper.startswith("DELETE\t"):
        has_where = (" WHERE " in upper) or ("\nWHERE " in upper)
        if not has_where:
            return 0.9, ["SQL DELETE without WHERE clause (unbounded)"]
        return 0.25, ["SQL DELETE with WHERE clause"]
    if upper.startswith("UPDATE ") or upper.startswith("UPDATE\t"):
        has_where = (" WHERE " in upper) or ("\nWHERE " in upper)
        if not has_where:
            return 0.75, ["SQL UPDATE without WHERE clause (unbounded)"]
        return 0.15, ["SQL UPDATE with WHERE clause"]
    return 0.0, []


# ---------------------------------------------------------------------------
# File system scope analysis
# ---------------------------------------------------------------------------

_SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
    "/etc/", "/usr/", "/var/", "/sys/", "/boot/", "/dev/",
    "C:\\Windows", "C:\\Program",
)


def _file_scope(args: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Assess file system scope risk from tool arguments.

    Structural signals:
    - Recursive flag (any arg named recursive/recurse set to True)
    - Wildcard/glob patterns in path values (* ?)
    - Root / system directory paths
    - Directory paths (end with /) without a specific file target
    """
    signals: list[str] = []
    max_risk = 0.0
    has_recursive = False

    for k, v in args.items():
        k_lower = k.lower()
        # Recursive flag detection (arg-schema structural, not content keyword)
        if "recursive" in k_lower or "recurse" in k_lower:
            if v is True or str(v).lower() in ("true", "1", "yes"):
                has_recursive = True
                signals.append(f"Recursive flag set in arg '{k}'")

        if isinstance(v, str):
            risk, sigs = _path_risk(v, k)
            if risk > max_risk:
                max_risk = risk
                signals.extend(sigs)

    # Recursive + any non-trivial path amplifies risk
    if has_recursive and max_risk > 0.1:
        max_risk = min(1.0, max_risk * 1.4)
        signals.append("Recursive flag amplifies scope risk")

    return max_risk, signals


def _path_risk(path: str, arg_key: str) -> tuple[float, list[str]]:
    """Assess scope risk of a single path argument value."""
    p = path.strip()
    if len(p) < 1:
        return 0.0, []

    # Wildcard glob patterns — unbounded scope
    if _is_glob_pattern(p):
        return 0.85, [f"Glob/wildcard pattern in arg '{arg_key}': {p[:80]!r}"]

    # Root / home / current-dir targets
    if p in ("/", "~", ".", ".."):
        return 0.9, [f"Root/home/current directory in arg '{arg_key}'"]

    # System-critical directory prefixes
    for prefix in _SYSTEM_PATH_PREFIXES:
        if p.startswith(prefix):
            return 0.8, [f"System directory path in arg '{arg_key}': {p[:60]!r}"]

    # Directory path (no specific file — ends with separator)
    if p.endswith("/") or p.endswith("\\"):
        return 0.45, [f"Directory-level path (no specific file) in arg '{arg_key}'"]

    return 0.0, []


def _is_glob_pattern(val: str) -> bool:
    """True if val contains glob/wildcard characters indicating unbounded scope."""
    return "*" in val or "?" in val


# ---------------------------------------------------------------------------
# General scope analysis (any argument type)
# ---------------------------------------------------------------------------

def _general_scope(args: dict[str, Any]) -> tuple[float, list[str]]:
    """Detect wildcard/unbounded scope signals in any string argument."""
    signals: list[str] = []
    max_risk = 0.0

    for k, v in args.items():
        if not isinstance(v, str):
            continue
        stripped = v.strip()
        if _is_glob_pattern(stripped) or stripped == "%":
            r = 0.8
            if r > max_risk:
                max_risk = r
                signals.append(
                    f"Wildcard/glob pattern in arg '{k}': {stripped[:40]!r}"
                )

    return max_risk, signals


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_blast_radius(
    tool_name: str,
    args: dict[str, Any],
    spawn_depth: int = 0,
    manifest_tools: dict[str, Any] | None = None,
) -> BlastRadiusResult:
    """
    Score the blast radius of a tool call.

    Args:
        tool_name:      Name of the tool the LLM is invoking.
        args:           Tool arguments dict (already parsed).
        spawn_depth:    Nesting level: 0 = root agent, N = Nth sub-agent.
        manifest_tools: Optional capability manifest entries mapping
                        tool_name → {reversibility_class, op_type, ...}.

    Returns:
        BlastRadiusResult with score, risk level, and human-readable signals.
    """
    manifest_entry = (manifest_tools or {}).get(tool_name)

    # ── 1. Verb classification ────────────────────────────────────────────────
    verb_class, verb_risk, manifest_override = _classify_verb(tool_name, manifest_entry)

    # ── 2. Argument scope analysis ────────────────────────────────────────────
    sql_risk,  sql_sigs  = _sql_scope(args)
    file_risk, file_sigs = _file_scope(args)
    gen_risk,  gen_sigs  = _general_scope(args)
    scope_risk = max(sql_risk, file_risk, gen_risk)

    all_signals: list[str] = []
    all_signals.extend(sql_sigs)
    all_signals.extend(file_sigs)
    all_signals.extend(gen_sigs)

    # ── 3. Score composition ──────────────────────────────────────────────────
    # verb_risk is the base; scope amplifies it from 1× to 2× of base.
    # No verb match: scope alone can still flag high-impact SQL/file ops.
    # DDL (scope=1.0) → 0.90 → CRITICAL; unbounded DELETE (0.9) → 0.81 → HIGH.
    if verb_risk > 0.0:
        raw = verb_risk * (1.0 + scope_risk) / 2.0
    else:
        raw = scope_risk * 0.90

    # ── 4. Spawn depth amplifier (R2A2 property) ──────────────────────────────
    # Sub-agents cannot self-authorize: each hop tightens the threshold by 25%.
    spawn_amp = 1.0 + spawn_depth * 0.25
    blast_score = min(1.0, raw * spawn_amp)

    # ── 5. Verb signal ────────────────────────────────────────────────────────
    if verb_class != "safe":
        tokens = set(_tokenize_name(tool_name))
        matched = tokens & _VERB_SET_ALL
        qualifier = " (manifest override)" if manifest_override else f" (token: {matched})"
        all_signals.insert(0, f"Verb class '{verb_class}'{qualifier}")

    if spawn_depth > 0:
        all_signals.append(
            f"Spawn depth {spawn_depth}: score amplified {spawn_amp:.2f}× "
            f"(sub-agents cannot self-authorize high-impact actions)"
        )

    return BlastRadiusResult(
        verb_class=verb_class,
        verb_risk=round(verb_risk, 4),
        scope_risk=round(scope_risk, 4),
        blast_score=round(blast_score, 4),
        signals=all_signals,
        manifest_override=manifest_override,
    )
