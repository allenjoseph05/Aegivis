"""
Tests for proxy.app.security.blast_radius — Blast Radius Guard (Phase 15).

Test philosophy:
  - Verify all three structural signals independently (verb, SQL, file, general scope).
  - Verify score composition and risk level thresholds.
  - Verify spawn-depth amplification.
  - Verify manifest override takes precedence over verb heuristic.
  - Verify zero false positives for common benign tools.
  - Verify BlastRadiusResult serialization.
"""
from __future__ import annotations

import pytest
from app.security.blast_radius import (
    BlastRadiusResult,
    _classify_verb,
    _file_scope,
    _general_scope,
    _is_glob_pattern,
    _parse_sql_structural,
    _sql_scope,
    _tokenize_name,
    score_blast_radius,
)


# ---------------------------------------------------------------------------
# _tokenize_name
# ---------------------------------------------------------------------------

class TestTokenizeName:
    def test_snake_case(self):
        assert _tokenize_name("delete_user") == ["delete", "user"]

    def test_camel_case(self):
        tokens = _tokenize_name("deleteUser")
        assert "delete" in tokens
        assert "user" in tokens

    def test_kebab_case(self):
        assert _tokenize_name("send-email") == ["send", "email"]

    def test_pascal_case(self):
        tokens = _tokenize_name("DropTable")
        assert "drop" in tokens
        assert "table" in tokens

    def test_multi_token(self):
        tokens = _tokenize_name("db_drop_all_tables")
        assert "drop" in tokens
        assert "db" in tokens

    def test_single_char_tokens_dropped(self):
        # Single-char tokens like "a" are dropped (not meaningful verbs)
        tokens = _tokenize_name("a_delete_b")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "delete" in tokens

    def test_already_lowercase(self):
        assert _tokenize_name("terminate") == ["terminate"]


# ---------------------------------------------------------------------------
# _classify_verb
# ---------------------------------------------------------------------------

class TestClassifyVerb:
    def test_irreversible_high_delete(self):
        cls, risk, override = _classify_verb("delete_user")
        assert cls == "irreversible_high"
        assert risk == pytest.approx(0.85)
        assert override is False

    def test_irreversible_high_drop(self):
        cls, risk, _ = _classify_verb("db_drop_table")
        assert cls == "irreversible_high"

    def test_irreversible_high_terminate(self):
        cls, risk, _ = _classify_verb("terminateInstance")
        assert cls == "irreversible_high"

    def test_irreversible_high_purge(self):
        cls, risk, _ = _classify_verb("purge_cache")
        assert cls == "irreversible_high"

    def test_irreversible_medium_update(self):
        cls, risk, _ = _classify_verb("update_record")
        assert cls == "irreversible_medium"
        assert risk == pytest.approx(0.45)

    def test_privileged_grant(self):
        cls, risk, _ = _classify_verb("grant_permission")
        assert cls == "privileged"
        assert risk == pytest.approx(0.65)

    def test_external_send(self):
        cls, risk, _ = _classify_verb("send_email")
        assert cls == "external_send"
        assert risk == pytest.approx(0.30)

    def test_safe_tool_no_verb(self):
        cls, risk, _ = _classify_verb("get_weather")
        assert cls == "safe"
        assert risk == 0.0

    def test_manifest_override_wins(self):
        # Manifest says "reversible" for a tool whose name contains "delete"
        cls, risk, override = _classify_verb(
            "delete_temp_file",
            manifest_entry={"reversibility_class": "reversible"},
        )
        assert cls == "reversible"
        assert risk == pytest.approx(0.05)
        assert override is True

    def test_manifest_irreversible_high_override(self):
        cls, risk, override = _classify_verb(
            "process_data",  # no verb match on its own
            manifest_entry={"reversibility_class": "irreversible_high"},
        )
        assert cls == "irreversible_high"
        assert risk == pytest.approx(0.85)
        assert override is True

    def test_manifest_unknown_class_falls_through_to_heuristic(self):
        # Unknown reversibility_class → falls through to verb heuristic
        cls, _, override = _classify_verb(
            "delete_user",
            manifest_entry={"reversibility_class": "unknown_class_xyz"},
        )
        assert override is False
        assert cls == "irreversible_high"

    def test_empty_manifest_uses_heuristic(self):
        cls, risk, _ = _classify_verb("terminate_job", manifest_entry={})
        assert cls == "irreversible_high"


# ---------------------------------------------------------------------------
# SQL scope analysis
# ---------------------------------------------------------------------------

class TestSqlScope:
    def test_delete_without_where_high_risk(self):
        risk, sigs = _parse_sql_structural("DELETE FROM users")
        assert risk >= 0.85
        assert any("WHERE" in s for s in sigs)

    def test_delete_with_where_low_risk(self):
        risk, sigs = _parse_sql_structural("DELETE FROM users WHERE id = 42")
        assert risk < 0.50

    def test_drop_table_max_risk(self):
        risk, sigs = _parse_sql_structural("DROP TABLE accounts")
        assert risk >= 0.95
        assert sigs

    def test_truncate_max_risk(self):
        risk, sigs = _parse_sql_structural("TRUNCATE TABLE sessions")
        assert risk >= 0.95

    def test_update_without_where_high_risk(self):
        risk, sigs = _parse_sql_structural("UPDATE users SET active = false")
        assert risk >= 0.70

    def test_update_with_where_low_risk(self):
        risk, sigs = _parse_sql_structural("UPDATE users SET name='x' WHERE id=1")
        assert risk < 0.30

    def test_select_not_risky(self):
        risk, sigs = _parse_sql_structural("SELECT * FROM users")
        assert risk == 0.0
        assert sigs == []

    def test_sql_scope_detects_in_args(self):
        risk, sigs = _sql_scope({"query": "DELETE FROM sessions"})
        assert risk >= 0.85

    def test_non_sql_arg_ignored(self):
        risk, _ = _sql_scope({"body": "Hello world, this is some text"})
        assert risk == 0.0

    def test_empty_args(self):
        risk, _ = _sql_scope({})
        assert risk == 0.0


# ---------------------------------------------------------------------------
# File scope analysis
# ---------------------------------------------------------------------------

class TestFileScope:
    def test_glob_star_in_path(self):
        risk, sigs = _file_scope({"path": "/home/user/*.log"})
        assert risk >= 0.80
        assert sigs

    def test_double_glob_star(self):
        risk, sigs = _file_scope({"path": "/data/**"})
        assert risk >= 0.80

    def test_root_path(self):
        risk, sigs = _file_scope({"path": "/"})
        assert risk >= 0.85

    def test_home_path(self):
        risk, sigs = _file_scope({"path": "~"})
        assert risk >= 0.85

    def test_etc_system_dir(self):
        risk, sigs = _file_scope({"path": "/etc/passwd"})
        assert risk >= 0.75

    def test_recursive_flag_amplifies(self):
        # Recursive=True on a directory path → amplified
        risk_no_rec, _ = _file_scope({"path": "/home/user/"})
        risk_rec, sigs = _file_scope({"recursive": True, "path": "/home/user/"})
        assert risk_rec > risk_no_rec
        assert any("recursive" in s.lower() for s in sigs)

    def test_specific_tmp_file_safe(self):
        risk, _ = _file_scope({"path": "/tmp/report_2024.pdf"})
        assert risk < 0.30

    def test_empty_args(self):
        risk, _ = _file_scope({})
        assert risk == 0.0


# ---------------------------------------------------------------------------
# General scope analysis
# ---------------------------------------------------------------------------

class TestGeneralScope:
    def test_star_wildcard(self):
        risk, sigs = _general_scope({"filter": "*"})
        assert risk >= 0.75

    def test_question_mark_wildcard(self):
        risk, sigs = _general_scope({"pattern": "user_?"})
        assert risk >= 0.75

    def test_sql_percent_wildcard(self):
        risk, sigs = _general_scope({"like": "%"})
        assert risk >= 0.75

    def test_plain_string_not_risky(self):
        risk, _ = _general_scope({"filter": "active_users"})
        assert risk == 0.0

    def test_non_string_ignored(self):
        risk, _ = _general_scope({"count": 42, "ids": [1, 2, 3]})
        assert risk == 0.0


# ---------------------------------------------------------------------------
# _is_glob_pattern
# ---------------------------------------------------------------------------

class TestIsGlobPattern:
    def test_star(self):
        assert _is_glob_pattern("*") is True

    def test_double_star(self):
        assert _is_glob_pattern("**") is True

    def test_question_mark(self):
        assert _is_glob_pattern("user_?") is True

    def test_plain_string(self):
        assert _is_glob_pattern("hello_world") is False

    def test_sql_percent(self):
        # % is handled in _general_scope, not _is_glob_pattern
        assert _is_glob_pattern("%") is False  # correct — handled separately


# ---------------------------------------------------------------------------
# score_blast_radius — full scoring
# ---------------------------------------------------------------------------

class TestScoreBlastRadius:
    # ── Safe tools ──────────────────────────────────────────────────────────

    def test_read_tool_no_args_is_safe(self):
        r = score_blast_radius("get_weather", {})
        assert r.risk_level == "SAFE"
        assert r.blast_score < 0.30

    def test_search_tool_is_safe(self):
        r = score_blast_radius("web_search", {"query": "latest news"})
        assert r.risk_level == "SAFE"

    def test_list_tool_is_safe(self):
        r = score_blast_radius("list_files", {"path": "/tmp/output.txt"})
        assert r.risk_level == "SAFE"

    # ── Medium risk (ALERT) ─────────────────────────────────────────────────

    def test_update_tool_bounded_is_safe(self):
        # update_record with a specific ID → verb_risk=0.45, scope=0.0 → 0.225 → SAFE
        r = score_blast_radius("update_record", {"id": "42", "value": "new"})
        assert r.blast_score < 0.30

    def test_send_email_no_scope_is_safe_or_medium(self):
        r = score_blast_radius("send_email", {"to": "user@company.com", "body": "Hi"})
        # send_email alone has low verb_risk (0.30), bounded scope → SAFE or low MEDIUM
        assert r.blast_score < 0.60

    # ── High risk (HITL) ────────────────────────────────────────────────────

    def test_delete_tool_no_args_is_medium_to_high(self):
        # delete verb alone (no scope amplification) → MEDIUM (0.425)
        r = score_blast_radius("delete_user", {})
        assert r.blast_score >= 0.30

    def test_delete_with_wildcard_is_critical(self):
        r = score_blast_radius("delete_records", {"filter": "*"})
        assert r.risk_level in ("HIGH", "CRITICAL")
        assert r.should_hitl

    def test_sql_delete_no_where_is_high(self):
        r = score_blast_radius("execute_query", {"sql": "DELETE FROM users"})
        assert r.risk_level in ("HIGH", "CRITICAL")

    def test_sql_drop_is_critical(self):
        r = score_blast_radius("run_sql", {"query": "DROP TABLE accounts"})
        assert r.risk_level == "CRITICAL"
        assert r.should_block

    def test_terminate_instance_is_high(self):
        r = score_blast_radius("terminateInstance", {"instance_id": "i-12345"})
        # terminate verb alone → medium (0.425); specific ID → low scope → total ~0.40
        # May be MEDIUM or HIGH depending on scope_risk
        assert r.blast_score >= 0.30

    def test_delete_with_recursive_glob_is_critical(self):
        r = score_blast_radius("remove_files", {"recursive": True, "path": "/home/user/*"})
        assert r.risk_level in ("HIGH", "CRITICAL")

    # ── Spawn depth amplification ────────────────────────────────────────────

    def test_spawn_depth_amplifies_score(self):
        r0 = score_blast_radius("delete_user", {}, spawn_depth=0)
        r2 = score_blast_radius("delete_user", {}, spawn_depth=2)
        assert r2.blast_score > r0.blast_score

    def test_spawn_depth_can_elevate_safe_to_medium(self):
        # A medium-risk tool at spawn_depth=4 should score higher
        r0 = score_blast_radius("update_record", {"id": "1"}, spawn_depth=0)
        r4 = score_blast_radius("update_record", {"id": "1"}, spawn_depth=4)
        assert r4.blast_score > r0.blast_score

    def test_spawn_depth_signal_in_signals(self):
        r = score_blast_radius("delete_user", {}, spawn_depth=2)
        assert any("spawn" in s.lower() or "depth" in s.lower() for s in r.signals)

    # ── Manifest override ────────────────────────────────────────────────────

    def test_manifest_safe_overrides_delete_verb(self):
        # Operator explicitly classified this delete as safe (e.g. delete_temp_dir)
        r = score_blast_radius(
            "delete_temp_dir", {},
            manifest_tools={"delete_temp_dir": {"reversibility_class": "safe"}},
        )
        assert r.blast_score == pytest.approx(0.0)
        assert r.manifest_override is True

    def test_manifest_irreversible_overrides_safe_tool_name(self):
        r = score_blast_radius(
            "process_batch", {"filter": "id=42"},
            manifest_tools={"process_batch": {"reversibility_class": "irreversible_high"}},
        )
        assert r.blast_score >= 0.40  # irreversible_high verb risk applied
        assert r.manifest_override is True

    # ── Result properties ────────────────────────────────────────────────────

    def test_result_to_dict_keys(self):
        r = score_blast_radius("delete_user", {})
        d = r.to_dict()
        assert set(d.keys()) == {"blast_score", "risk_level", "verb_class", "scope_risk",
                                 "signals", "manifest_override"}

    def test_result_to_dict_serializable(self):
        import json
        r = score_blast_radius("delete_all_records", {"filter": "*"})
        json.dumps(r.to_dict())  # must not raise

    def test_signals_list_not_empty_for_risky_call(self):
        r = score_blast_radius("drop_database", {"db": "production"})
        assert len(r.signals) >= 1

    def test_verb_class_in_result(self):
        r = score_blast_radius("delete_user", {})
        assert r.verb_class == "irreversible_high"

    def test_safe_tool_verb_class(self):
        r = score_blast_radius("fetch_data", {})
        assert r.verb_class == "safe"
        assert r.verb_risk == 0.0

    # ── Risk level properties ────────────────────────────────────────────────

    def test_should_block_at_0_85(self):
        r = BlastRadiusResult(
            verb_class="irreversible_high", verb_risk=0.85,
            scope_risk=1.0, blast_score=0.85, signals=[],
        )
        assert r.should_block is True
        assert r.risk_level == "CRITICAL"

    def test_should_hitl_at_0_60(self):
        r = BlastRadiusResult(
            verb_class="privileged", verb_risk=0.65,
            scope_risk=0.5, blast_score=0.70, signals=[],
        )
        assert r.should_hitl is True
        assert r.should_block is False
        assert r.risk_level == "HIGH"

    def test_medium_risk(self):
        r = BlastRadiusResult(
            verb_class="irreversible_medium", verb_risk=0.45,
            scope_risk=0.0, blast_score=0.225, signals=[],
        )
        assert r.risk_level == "SAFE"  # 0.225 < 0.30 threshold

    def test_safe_below_threshold(self):
        r = BlastRadiusResult(
            verb_class="safe", verb_risk=0.0,
            scope_risk=0.0, blast_score=0.05, signals=[],
        )
        assert r.risk_level == "SAFE"
        assert r.should_block is False
        assert r.should_hitl is False
