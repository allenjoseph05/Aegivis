"""
Counterfactual Scope Estimator — Phase 17 (Pre-Execution Scope Measurement).

Core idea: transform a destructive SQL statement into SELECT COUNT(*) to
measure the *actual* number of rows that would be affected BEFORE any
mutation happens.  This gives the sandbox (Phase 16.1 SQL adapter) a
precise scope number — not a heuristic — to feed into the safety decision.

Supported transforms:
    DELETE FROM <table> [WHERE <cond>]
      → SELECT COUNT(*) FROM <table> [WHERE <cond>]

    UPDATE <table> SET ... [WHERE <cond>]
      → SELECT COUNT(*) FROM <table> [WHERE <cond>]

    DROP TABLE / TRUNCATE TABLE
      → None (count_sql is None — the operation destroys the table itself)
        TransformResult.is_ddl = True; rows_affected sentinel = -1 ("entire table")

Not supported (returns count_sql=None, type=UNKNOWN):
    INSERT — scope depends on VALUES / SELECT subquery (complex; deferred)
    ALTER, MERGE, CREATE — structural changes; deferred

Primary path: sqlglot AST transform (sqlglot is a hard dependency ≥25.0.0).
Fallback path: structural string analysis — used only when sqlglot raises an
  exception on unusual SQL dialects or edge-case syntax.  Covers the common
  DELETE/UPDATE/DROP/TRUNCATE patterns via pure string operations (no regex).

Fallback limitations (documented, not bugs):
  - Table aliases in structural fallback may not be stripped correctly.
  - WHERE conditions containing the literal text " SET " (rare) may confuse
    the UPDATE structural parser.  The AST path handles this correctly.
  - Subquery tables appear in the `tables` list from the AST path but not
    from the structural fallback (structural gives primary table only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """
    Result of transforming a write SQL statement to SELECT COUNT(*).

    Attributes:
        count_sql:       The COUNT(*) SQL string to execute for scope estimation.
                         None for DDL (DROP/TRUNCATE) — entire table is affected.
        statement_type:  Uppercase keyword: "DELETE", "UPDATE", "DROP", "TRUNCATE",
                         "UNKNOWN".
        tables:          All table names referenced (primary + subquery tables).
                         From AST path: complete.  From structural fallback: primary only.
        is_bounded:      True when the original statement had a WHERE clause
                         (scope is limited to matching rows).
        is_ddl:          True for DROP/TRUNCATE — these destroy the table entirely.
                         Always implies count_sql=None and is_bounded=False.
        via_fallback:    True when the structural fallback was used (AST path failed).
    """
    count_sql: str | None
    statement_type: str
    tables: list[str] = field(default_factory=list)
    is_bounded: bool = False
    is_ddl: bool = False
    via_fallback: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transform_to_count(sql: str) -> TransformResult:
    """
    Transform a write SQL statement to SELECT COUNT(*).

    Tries the sqlglot AST path first; falls back to structural parsing
    on any exception.  Always returns a TransformResult — never raises.

    Args:
        sql: The SQL statement to transform (any case; leading/trailing
             whitespace is stripped internally).

    Returns:
        TransformResult with count_sql set for DELETE/UPDATE, None for DDL.
    """
    try:
        return _transform_ast(sql)
    except Exception as exc:
        logger.debug(
            "scope_estimator: AST transform failed (%s), using structural fallback", exc
        )
        return _transform_structural(sql)


# ---------------------------------------------------------------------------
# AST transform (primary path — sqlglot)
# ---------------------------------------------------------------------------

def _transform_ast(sql: str) -> TransformResult:
    """Transform via sqlglot AST — preserves complex WHERE, aliases, subqueries."""
    import sqlglot
    import sqlglot.expressions as exp

    stmt = sqlglot.parse_one(sql.strip())
    if stmt is None:
        return TransformResult(count_sql=None, statement_type="UNKNOWN")

    if isinstance(stmt, exp.Drop):
        tables = [t.name for t in stmt.find_all(exp.Table)]
        return TransformResult(
            count_sql=None, statement_type="DROP",
            tables=tables, is_bounded=False, is_ddl=True,
        )

    if isinstance(stmt, exp.TruncateTable):
        tables = [t.name for t in stmt.find_all(exp.Table)]
        return TransformResult(
            count_sql=None, statement_type="TRUNCATE",
            tables=tables, is_bounded=False, is_ddl=True,
        )

    if isinstance(stmt, exp.Delete):
        table_node = stmt.args["this"]
        tables = [t.name for t in stmt.find_all(exp.Table)]
        where = stmt.args.get("where")
        count = exp.select(exp.Count(this=exp.Star())).from_(table_node)
        if where:
            count = count.where(where.this)
        return TransformResult(
            count_sql=count.sql(), statement_type="DELETE",
            tables=tables, is_bounded=bool(where),
        )

    if isinstance(stmt, exp.Update):
        table_node = stmt.args["this"]
        tables = [t.name for t in stmt.find_all(exp.Table)]
        where = stmt.args.get("where")
        count = exp.select(exp.Count(this=exp.Star())).from_(table_node)
        if where:
            count = count.where(where.this)
        return TransformResult(
            count_sql=count.sql(), statement_type="UPDATE",
            tables=tables, is_bounded=bool(where),
        )

    # SELECT, INSERT, MERGE, ALTER, CREATE, etc. — not a scope-estimable write
    return TransformResult(
        count_sql=None,
        statement_type=type(stmt).__name__.upper(),
    )


# ---------------------------------------------------------------------------
# Structural fallback (pure string operations — no regex)
# ---------------------------------------------------------------------------

def _transform_structural(sql: str) -> TransformResult:
    """
    Structural string fallback for when sqlglot raises.

    Handles the common patterns DELETE/UPDATE/DROP/TRUNCATE using simple
    string operations.  Returns TransformResult(via_fallback=True).

    Limitations:
      - Does not handle multi-table DELETE/UPDATE JOINs.
      - WHERE parsing may mis-identify the clause if the literal text " SET "
        appears in the WHERE condition of an UPDATE statement (exceedingly rare).
    """
    upper = sql.strip().upper()

    if upper.startswith("DELETE"):
        return _structural_delete(sql)

    if upper.startswith("UPDATE"):
        return _structural_update(sql)

    if upper.startswith("DROP"):
        table = _structural_extract_object_name(sql, skip_words=("DROP", "TABLE", "DATABASE", "SCHEMA", "INDEX", "VIEW"))
        return TransformResult(
            count_sql=None, statement_type="DROP",
            tables=[table] if table else [], is_bounded=False, is_ddl=True,
            via_fallback=True,
        )

    if upper.startswith("TRUNCATE"):
        table = _structural_extract_object_name(sql, skip_words=("TRUNCATE", "TABLE"))
        return TransformResult(
            count_sql=None, statement_type="TRUNCATE",
            tables=[table] if table else [], is_bounded=False, is_ddl=True,
            via_fallback=True,
        )

    return TransformResult(count_sql=None, statement_type="UNKNOWN", via_fallback=True)


def _structural_delete(sql: str) -> TransformResult:
    """Structural parse for DELETE [FROM] <table> [WHERE <cond>]."""
    rest = sql.strip()

    # Strip DELETE keyword
    rest = rest[len("DELETE"):].lstrip()
    # Strip optional FROM
    if rest.upper().startswith("FROM ") or rest.upper().startswith("FROM\t"):
        rest = rest[4:].lstrip()

    upper_rest = rest.upper()
    where_idx = upper_rest.find(" WHERE ")

    if where_idx >= 0:
        table = rest[:where_idx].split()[0]  # first token before WHERE
        where_clause = rest[where_idx + 1:]  # "WHERE ..."
        count_sql = f"SELECT COUNT(*) FROM {table} {where_clause}"
        return TransformResult(
            count_sql=count_sql, statement_type="DELETE",
            tables=[table], is_bounded=True, via_fallback=True,
        )
    else:
        tokens = rest.split()
        table = tokens[0] if tokens else "unknown"
        return TransformResult(
            count_sql=f"SELECT COUNT(*) FROM {table}",
            statement_type="DELETE",
            tables=[table], is_bounded=False, via_fallback=True,
        )


def _structural_update(sql: str) -> TransformResult:
    """Structural parse for UPDATE <table> SET ... [WHERE <cond>]."""
    rest = sql.strip()
    rest = rest[len("UPDATE"):].lstrip()  # strip UPDATE
    upper_rest = rest.upper()

    # Find SET keyword
    set_idx = upper_rest.find(" SET ")
    if set_idx < 0:
        tokens = rest.split()
        table = tokens[0] if tokens else "unknown"
        return TransformResult(
            count_sql=f"SELECT COUNT(*) FROM {table}",
            statement_type="UPDATE",
            tables=[table], is_bounded=False, via_fallback=True,
        )

    table = rest[:set_idx].split()[0]  # first token before SET
    after_set = rest[set_idx + 5:]     # everything after SET
    upper_after = after_set.upper()

    where_idx = upper_after.find(" WHERE ")
    if where_idx >= 0:
        where_clause = after_set[where_idx + 1:]  # "WHERE ..."
        count_sql = f"SELECT COUNT(*) FROM {table} {where_clause}"
        return TransformResult(
            count_sql=count_sql, statement_type="UPDATE",
            tables=[table], is_bounded=True, via_fallback=True,
        )
    else:
        return TransformResult(
            count_sql=f"SELECT COUNT(*) FROM {table}",
            statement_type="UPDATE",
            tables=[table], is_bounded=False, via_fallback=True,
        )


def _structural_extract_object_name(sql: str, skip_words: tuple[str, ...]) -> str:
    """Extract the object name after skipping known DDL keywords."""
    tokens = sql.strip().split()
    for tok in tokens:
        if tok.upper() not in skip_words:
            return tok
    return "unknown"
