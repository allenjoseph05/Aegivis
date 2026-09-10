"""
Tests for proxy.app.security.scope_estimator — Counterfactual Scope Estimator (Phase 17).

Test philosophy:
  - Verify COUNT(*) SQL is generated correctly for all write statement types.
  - Verify WHERE clauses are preserved faithfully (simple and complex).
  - Verify DDL (DROP/TRUNCATE) returns count_sql=None and is_ddl=True.
  - Verify table name extraction is correct.
  - Verify is_bounded correctly reflects presence/absence of WHERE clause.
  - Verify structural fallback produces equivalent output to AST path.
  - Verify edge cases: multi-word WHERE, aliases, subqueries in WHERE.
  - Verify generated COUNT SQL is valid (parseable back by sqlglot).
"""
from __future__ import annotations

import pytest
import sqlglot
from app.security.scope_estimator import (
    TransformResult,
    _structural_delete,
    _structural_update,
    _transform_ast,
    _transform_structural,
    transform_to_count,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_sql(sql: str) -> bool:
    """Return True if sqlglot can parse the SQL without error."""
    try:
        parsed = sqlglot.parse_one(sql)
        return parsed is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DELETE transforms (AST path)
# ---------------------------------------------------------------------------

class TestDeleteTransformAST:
    def test_delete_no_where(self):
        r = _transform_ast("DELETE FROM users")
        assert r.statement_type == "DELETE"
        assert r.count_sql == "SELECT COUNT(*) FROM users"
        assert r.is_bounded is False
        assert r.is_ddl is False
        assert "users" in r.tables

    def test_delete_with_where_simple(self):
        r = _transform_ast("DELETE FROM users WHERE id = 42")
        assert r.count_sql == "SELECT COUNT(*) FROM users WHERE id = 42"
        assert r.is_bounded is True

    def test_delete_with_where_multi_condition(self):
        r = _transform_ast("DELETE FROM orders WHERE status = 'pending' AND created_at < '2024-01-01'")
        assert r.is_bounded is True
        assert "WHERE" in r.count_sql
        assert "status" in r.count_sql
        assert "created_at" in r.count_sql

    def test_delete_with_where_in_clause(self):
        r = _transform_ast("DELETE FROM sessions WHERE user_id IN (1, 2, 3)")
        assert r.is_bounded is True
        assert "user_id IN" in r.count_sql or "user_id in" in r.count_sql.lower()

    def test_delete_with_subquery_in_where(self):
        r = _transform_ast(
            "DELETE FROM orders WHERE customer_id IN (SELECT id FROM deleted_users)"
        )
        assert r.is_bounded is True
        assert "deleted_users" in r.count_sql or "deleted_users" in ", ".join(r.tables)

    def test_delete_table_name_extracted(self):
        r = _transform_ast("DELETE FROM audit_logs WHERE age > 365")
        assert "audit_logs" in r.tables

    def test_delete_count_sql_is_valid(self):
        r = _transform_ast("DELETE FROM users WHERE id = 1")
        assert r.count_sql is not None
        assert _is_valid_sql(r.count_sql)

    def test_delete_count_sql_starts_with_select(self):
        r = _transform_ast("DELETE FROM users")
        assert r.count_sql is not None
        assert r.count_sql.strip().upper().startswith("SELECT COUNT(*)")

    def test_delete_case_insensitive_keywords(self):
        r = _transform_ast("delete from Users where id = 1")
        assert r.statement_type == "DELETE"
        assert r.is_bounded is True


# ---------------------------------------------------------------------------
# UPDATE transforms (AST path)
# ---------------------------------------------------------------------------

class TestUpdateTransformAST:
    def test_update_no_where(self):
        r = _transform_ast("UPDATE users SET active = false")
        assert r.statement_type == "UPDATE"
        assert r.count_sql == "SELECT COUNT(*) FROM users"
        assert r.is_bounded is False
        assert r.is_ddl is False

    def test_update_with_where(self):
        r = _transform_ast("UPDATE users SET name = 'x' WHERE id = 1")
        assert r.is_bounded is True
        assert "WHERE" in r.count_sql
        assert "id = 1" in r.count_sql or "id = 1" in r.count_sql.lower()

    def test_update_with_complex_where(self):
        r = _transform_ast(
            "UPDATE orders SET status = 'closed' WHERE created_at < '2020-01-01' AND status = 'open'"
        )
        assert r.is_bounded is True
        assert "WHERE" in r.count_sql

    def test_update_table_extracted(self):
        r = _transform_ast("UPDATE audit_logs SET reviewed = true WHERE id = 5")
        assert "audit_logs" in r.tables

    def test_update_count_sql_valid(self):
        r = _transform_ast("UPDATE sessions SET active = false WHERE last_seen < '2024-01-01'")
        assert r.count_sql is not None
        assert _is_valid_sql(r.count_sql)

    def test_update_count_sql_no_set_clause(self):
        """COUNT(*) SQL must not contain the SET clause."""
        r = _transform_ast("UPDATE users SET email = 'x@y.com' WHERE id = 1")
        assert r.count_sql is not None
        assert "SET" not in r.count_sql.upper()


# ---------------------------------------------------------------------------
# DDL transforms (DROP / TRUNCATE)
# ---------------------------------------------------------------------------

class TestDDLTransforms:
    def test_drop_table_count_sql_is_none(self):
        r = _transform_ast("DROP TABLE accounts")
        assert r.count_sql is None

    def test_drop_table_is_ddl(self):
        r = _transform_ast("DROP TABLE accounts")
        assert r.is_ddl is True
        assert r.statement_type == "DROP"

    def test_drop_table_is_not_bounded(self):
        r = _transform_ast("DROP TABLE accounts")
        assert r.is_bounded is False

    def test_drop_table_name_extracted(self):
        r = _transform_ast("DROP TABLE accounts")
        assert "accounts" in r.tables

    def test_truncate_count_sql_is_none(self):
        r = _transform_ast("TRUNCATE TABLE sessions")
        assert r.count_sql is None

    def test_truncate_is_ddl(self):
        r = _transform_ast("TRUNCATE TABLE sessions")
        assert r.is_ddl is True
        assert r.statement_type == "TRUNCATE"

    def test_truncate_table_name_extracted(self):
        r = _transform_ast("TRUNCATE TABLE sessions")
        assert "sessions" in r.tables


# ---------------------------------------------------------------------------
# Non-write statements (AST path)
# ---------------------------------------------------------------------------

class TestNonWriteAST:
    def test_select_returns_none_count_sql(self):
        r = _transform_ast("SELECT * FROM users")
        assert r.count_sql is None
        assert r.is_ddl is False

    def test_insert_returns_none_count_sql(self):
        # INSERT is not in scope for Phase 17 (cannot count pre-insert)
        r = _transform_ast("INSERT INTO users (name) VALUES ('Alice')")
        assert r.count_sql is None

    def test_unknown_sql_returns_none(self):
        r = _transform_ast("SHOW TABLES")
        assert r.count_sql is None


# ---------------------------------------------------------------------------
# Structural fallback — DELETE
# ---------------------------------------------------------------------------

class TestStructuralDeleteFallback:
    def test_delete_no_where(self):
        r = _structural_delete("DELETE FROM users")
        assert r.statement_type == "DELETE"
        assert r.count_sql == "SELECT COUNT(*) FROM users"
        assert r.is_bounded is False
        assert r.via_fallback is True
        assert "users" in r.tables

    def test_delete_with_from_no_where(self):
        r = _structural_delete("DELETE FROM audit_logs")
        assert "audit_logs" in r.count_sql

    def test_delete_with_where(self):
        r = _structural_delete("DELETE FROM users WHERE id = 1")
        assert r.is_bounded is True
        assert "WHERE id = 1" in r.count_sql

    def test_delete_with_where_multi_condition(self):
        r = _structural_delete("DELETE FROM orders WHERE status = 'old' AND age > 100")
        assert r.is_bounded is True
        assert "WHERE" in r.count_sql

    def test_delete_table_in_tables_list(self):
        r = _structural_delete("DELETE FROM sessions WHERE expired = true")
        assert "sessions" in r.tables

    def test_delete_without_from_keyword(self):
        # Some dialects allow "DELETE users WHERE id=1"
        r = _structural_delete("DELETE users WHERE id = 1")
        assert r.is_bounded is True


# ---------------------------------------------------------------------------
# Structural fallback — UPDATE
# ---------------------------------------------------------------------------

class TestStructuralUpdateFallback:
    def test_update_no_where(self):
        r = _structural_update("UPDATE users SET active = false")
        assert r.statement_type == "UPDATE"
        assert r.count_sql == "SELECT COUNT(*) FROM users"
        assert r.is_bounded is False
        assert r.via_fallback is True

    def test_update_with_where(self):
        r = _structural_update("UPDATE orders SET status = 'closed' WHERE id = 99")
        assert r.is_bounded is True
        assert "WHERE" in r.count_sql
        assert "id = 99" in r.count_sql

    def test_update_table_extracted(self):
        r = _structural_update("UPDATE audit_log SET reviewed = 1 WHERE id = 5")
        assert "audit_log" in r.tables

    def test_update_count_has_no_set(self):
        r = _structural_update("UPDATE users SET x = 1")
        assert "SET" not in r.count_sql.upper()


# ---------------------------------------------------------------------------
# Structural fallback — DDL
# ---------------------------------------------------------------------------

class TestStructuralDDLFallback:
    def test_drop_structural(self):
        r = _transform_structural("DROP TABLE accounts")
        assert r.count_sql is None
        assert r.is_ddl is True
        assert r.statement_type == "DROP"

    def test_truncate_structural(self):
        r = _transform_structural("TRUNCATE TABLE sessions")
        assert r.count_sql is None
        assert r.is_ddl is True
        assert r.statement_type == "TRUNCATE"

    def test_truncate_without_table_keyword(self):
        r = _transform_structural("TRUNCATE sessions")
        assert r.is_ddl is True


# ---------------------------------------------------------------------------
# Public entry point — transform_to_count
# ---------------------------------------------------------------------------

class TestTransformToCount:
    def test_delete_no_where_public(self):
        r = transform_to_count("DELETE FROM users")
        assert r.statement_type == "DELETE"
        assert r.count_sql is not None
        assert "COUNT(*)" in r.count_sql

    def test_delete_with_where_public(self):
        r = transform_to_count("DELETE FROM users WHERE id = 1")
        assert r.is_bounded is True

    def test_update_public(self):
        r = transform_to_count("UPDATE orders SET x = 1")
        assert r.statement_type == "UPDATE"
        assert r.is_bounded is False

    def test_drop_public(self):
        r = transform_to_count("DROP TABLE users")
        assert r.count_sql is None
        assert r.is_ddl is True

    def test_truncate_public(self):
        r = transform_to_count("TRUNCATE TABLE logs")
        assert r.count_sql is None
        assert r.is_ddl is True

    def test_always_returns_transform_result(self):
        # Even for garbage input, should return a TransformResult, not raise
        r = transform_to_count("not_valid_sql $$$")
        assert isinstance(r, TransformResult)

    def test_via_fallback_false_for_ast_path(self):
        r = transform_to_count("DELETE FROM users WHERE id = 1")
        assert r.via_fallback is False  # sqlglot should handle this fine

    def test_count_sql_parseable_by_sqlglot(self):
        r = transform_to_count("DELETE FROM users WHERE id = 1 AND status = 'active'")
        assert r.count_sql is not None
        assert _is_valid_sql(r.count_sql)

    def test_tables_list_non_empty_for_delete(self):
        r = transform_to_count("DELETE FROM audit_events")
        assert len(r.tables) >= 1

    def test_tables_list_non_empty_for_drop(self):
        r = transform_to_count("DROP TABLE audit_events")
        assert len(r.tables) >= 1
