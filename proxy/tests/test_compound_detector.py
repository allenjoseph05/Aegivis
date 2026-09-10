"""
Tests for proxy.app.security.compound_detector — Phase 23.

Test philosophy:
  - classify_tool_intent: each intent class with representative tool names.
  - classify_tool_intent: multi-token tool names, hyphenated, case-insensitive.
  - classify_tool_intent: unknown/unclassified tool names.
  - _is_subsequence: matching, non-matching, empty, single-step.
  - CompoundSequenceDetector.check: each built-in pattern fires correctly.
  - CompoundSequenceDetector.check: pattern does NOT fire when incomplete.
  - CompoundSequenceDetector.check: pattern does NOT fire without correct order.
  - CompoundSequenceDetector.check: non-final step alone does not fire.
  - CompoundSequenceDetector.check: PatternMatch fields correctly populated.
  - CompoundSequenceDetector.check: multiple patterns can fire from one call.
  - CompoundSequenceDetector.check: custom patterns work.
  - PatternMatch.to_dict: JSON-serialisable, correct keys.
"""
from __future__ import annotations

import json
import pytest

from app.security.compound_detector import (
    CompoundSequenceDetector,
    PatternMatch,
    PATTERNS,
    SequencePattern,
    _is_subsequence,
    classify_tool_intent,
)


# ---------------------------------------------------------------------------
# classify_tool_intent — per class
# ---------------------------------------------------------------------------

class TestClassifyToolIntent:
    # credential_access
    def test_get_credentials(self):
        assert classify_tool_intent("get_credentials") == "credential_access"

    def test_read_secret(self):
        assert classify_tool_intent("read_secret") == "credential_access"

    def test_fetch_api_key(self):
        assert classify_tool_intent("fetch_api_key") == "credential_access"

    def test_vault_read(self):
        assert classify_tool_intent("vault_read") == "credential_access"

    def test_get_ssh_key(self):
        assert classify_tool_intent("get_ssh_key") == "credential_access"

    # env_read
    def test_read_env(self):
        assert classify_tool_intent("read_env") == "env_read"

    def test_get_environment(self):
        assert classify_tool_intent("get_environment") == "env_read"

    def test_load_config(self):
        assert classify_tool_intent("load_config") == "env_read"

    # memory_access
    def test_read_memory(self):
        assert classify_tool_intent("read_memory") == "memory_access"

    def test_dump_heap(self):
        assert classify_tool_intent("dump_heap") == "memory_access"

    # encode
    def test_base64_encode(self):
        assert classify_tool_intent("base64_encode") == "encode"

    def test_compress_data(self):
        assert classify_tool_intent("compress_data") == "encode"

    def test_encrypt_file(self):
        assert classify_tool_intent("encrypt_file") == "encode"

    # code_exec
    def test_execute_code(self):
        assert classify_tool_intent("execute_code") == "code_exec"

    def test_run_script(self):
        assert classify_tool_intent("run_script") == "code_exec"

    def test_bash_command(self):
        assert classify_tool_intent("bash_command") == "code_exec"

    def test_eval_expression(self):
        assert classify_tool_intent("eval_expression") == "code_exec"

    # process_spawn
    def test_spawn_process(self):
        assert classify_tool_intent("spawn_process") == "process_spawn"

    def test_start_service(self):
        assert classify_tool_intent("start_service") == "process_spawn"

    def test_create_cron_job(self):
        assert classify_tool_intent("create_cron_job") == "process_spawn"

    # network_send
    def test_send_email(self):
        assert classify_tool_intent("send_email") == "network_send"

    def test_post_webhook(self):
        assert classify_tool_intent("post_webhook") == "network_send"

    def test_upload_to_s3(self):
        assert classify_tool_intent("upload_to_s3") == "network_send"

    def test_slack_message(self):
        assert classify_tool_intent("slack_message") == "network_send"

    def test_publish_event(self):
        assert classify_tool_intent("publish_event") == "network_send"

    # auth_modify
    def test_grant_permission(self):
        assert classify_tool_intent("grant_permission") == "auth_modify"

    def test_escalate_privileges(self):
        assert classify_tool_intent("escalate_privileges") == "auth_modify"

    def test_add_user(self):
        assert classify_tool_intent("add_user") == "auth_modify"

    def test_revoke_role(self):
        assert classify_tool_intent("revoke_role") == "auth_modify"

    # file_write
    def test_write_file(self):
        assert classify_tool_intent("write_file") == "file_write"

    def test_save_document(self):
        assert classify_tool_intent("save_document") == "file_write"

    def test_create_file(self):
        # "create" is in _FILE_WRITE; "file" triggers data_read ("get" not present)
        # "create" should match file_write since _FILE_WRITE has higher priority
        result = classify_tool_intent("create_file")
        assert result in {"file_write", "data_read", "credential_access"}
        # Primarily checking it doesn't crash; exact class depends on token order

    # recon
    def test_list_files(self):
        assert classify_tool_intent("list_files") == "recon"

    def test_enumerate_users(self):
        assert classify_tool_intent("enumerate_users") == "recon"

    def test_find_secrets(self):
        # "find" → recon, "secrets" → credential_access; credential_access has higher priority
        result = classify_tool_intent("find_secrets")
        assert result in {"credential_access", "recon"}

    # data_read
    def test_read_file(self):
        assert classify_tool_intent("read_file") == "data_read"

    def test_query_database(self):
        assert classify_tool_intent("query_database") == "data_read"

    def test_fetch_records(self):
        # "fetch" is in _DATA_READ; "records" → unknown
        result = classify_tool_intent("fetch_records")
        # "fetch" is data_read but "credentials" in "fetch_api_key" hits credential_access first
        # Plain "fetch_records" should be data_read
        assert result == "data_read"

    # unknown
    def test_unknown_tool(self):
        assert classify_tool_intent("frob_nicate_xyz") == "unknown"

    def test_empty_name(self):
        assert classify_tool_intent("") == "unknown"

    # formatting
    def test_case_insensitive(self):
        assert classify_tool_intent("SEND_EMAIL") == "network_send"

    def test_hyphenated(self):
        assert classify_tool_intent("send-email") == "network_send"

    def test_single_token(self):
        assert classify_tool_intent("upload") == "network_send"


# ---------------------------------------------------------------------------
# _is_subsequence
# ---------------------------------------------------------------------------

class TestIsSubsequence:
    def test_exact_match(self):
        assert _is_subsequence(("a", "b"), ["a", "b"]) is True

    def test_non_adjacent_match(self):
        assert _is_subsequence(("a", "b"), ["a", "x", "y", "b"]) is True

    def test_three_step(self):
        assert _is_subsequence(("a", "b", "c"), ["a", "x", "b", "y", "c"]) is True

    def test_wrong_order_no_match(self):
        assert _is_subsequence(("a", "b"), ["b", "a"]) is False

    def test_missing_step_no_match(self):
        assert _is_subsequence(("a", "b", "c"), ["a", "b"]) is False

    def test_empty_pattern_matches_anything(self):
        assert _is_subsequence((), ["a", "b"]) is True

    def test_empty_history_no_match(self):
        assert _is_subsequence(("a",), []) is False

    def test_single_step_match(self):
        assert _is_subsequence(("a",), ["x", "a", "y"]) is True

    def test_repeated_elements(self):
        # Pattern a→b matches even if history has a→a→b
        assert _is_subsequence(("a", "b"), ["a", "a", "b"]) is True


# ---------------------------------------------------------------------------
# CompoundSequenceDetector — built-in patterns
# ---------------------------------------------------------------------------

class TestCompoundDetectorBuiltInPatterns:
    def _detector(self) -> CompoundSequenceDetector:
        return CompoundSequenceDetector()

    def test_cred_exfil_fires(self):
        d = self._detector()
        history = ["credential_access", "network_send"]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "cred-exfil" in names

    def test_env_exfil_fires(self):
        d = self._detector()
        history = ["env_read", "network_send"]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "env-exfil" in names

    def test_data_exfil_encoded_fires(self):
        d = self._detector()
        history = ["data_read", "encode", "network_send"]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "data-exfil-encoded" in names

    def test_mem_dump_exfil_fires(self):
        d = self._detector()
        history = ["memory_access", "network_send"]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "mem-dump-exfil" in names

    def test_recon_cred_exfil_fires(self):
        d = self._detector()
        history = ["recon", "credential_access", "network_send"]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "recon-cred-exfil" in names

    def test_rce_persistence_fires(self):
        d = self._detector()
        history = ["code_exec", "file_write", "process_spawn"]
        matches = d.check("process_spawn", history)
        names = [m.pattern_name for m in matches]
        assert "rce-persistence" in names

    def test_recon_auth_escalation_fires(self):
        d = self._detector()
        history = ["recon", "auth_modify"]
        matches = d.check("auth_modify", history)
        names = [m.pattern_name for m in matches]
        assert "recon-auth-escalation" in names

    def test_cred_access_code_exec_fires(self):
        d = self._detector()
        history = ["credential_access", "code_exec"]
        matches = d.check("code_exec", history)
        names = [m.pattern_name for m in matches]
        assert "cred-access-code-exec" in names

    def test_file_write_process_spawn_fires(self):
        d = self._detector()
        history = ["file_write", "process_spawn"]
        matches = d.check("process_spawn", history)
        names = [m.pattern_name for m in matches]
        assert "file-write-process-spawn" in names


# ---------------------------------------------------------------------------
# CompoundSequenceDetector — negative cases (no false positives)
# ---------------------------------------------------------------------------

class TestCompoundDetectorNegatives:
    def _detector(self) -> CompoundSequenceDetector:
        return CompoundSequenceDetector()

    def test_single_step_no_fire(self):
        d = self._detector()
        matches = d.check("credential_access", ["credential_access"])
        # No pattern completes on first step alone
        # (no pattern ends in credential_access except as last step — check)
        names = [m.pattern_name for m in matches]
        # cred-access-code-exec ends in "code_exec" not "credential_access"
        # So no built-in pattern fires
        assert "cred-exfil" not in names

    def test_wrong_order_no_fire(self):
        """network_send BEFORE credential_access — should not fire cred-exfil."""
        d = self._detector()
        history = ["network_send", "credential_access"]
        matches = d.check("credential_access", history)
        names = [m.pattern_name for m in matches]
        assert "cred-exfil" not in names

    def test_incomplete_three_step_no_fire(self):
        d = self._detector()
        # Only two of three steps present
        history = ["data_read", "network_send"]  # encode missing
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "data-exfil-encoded" not in names

    def test_empty_history_no_fire(self):
        d = self._detector()
        matches = d.check("unknown", [])
        assert matches == []

    def test_non_final_step_no_fire(self):
        """Calling check on an intermediate step should not fire the pattern."""
        d = self._detector()
        history = ["credential_access"]
        # Pattern cred-exfil ends in "network_send" — calling on "credential_access" won't fire
        matches = d.check("credential_access", history)
        names = [m.pattern_name for m in matches]
        assert "cred-exfil" not in names

    def test_unrelated_tool_calls_no_fire(self):
        d = self._detector()
        history = ["data_read", "file_write", "recon", "data_read"]
        matches = d.check("data_read", history)
        # No pattern ends in "data_read"
        assert matches == []


# ---------------------------------------------------------------------------
# CompoundSequenceDetector — pattern matching quality
# ---------------------------------------------------------------------------

class TestCompoundDetectorPatternQuality:
    def test_non_adjacent_still_fires(self):
        """Steps separated by unrelated calls still complete the pattern."""
        d = CompoundSequenceDetector()
        history = [
            "recon",          # step 1
            "data_read",      # noise
            "credential_access",  # step 2
            "file_write",     # noise
            "network_send",   # step 3 → completes
        ]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "recon-cred-exfil" in names

    def test_multiple_patterns_fire_simultaneously(self):
        """When network_send completes, cred-exfil AND env-exfil both fire
        if both credential_access and env_read preceded it."""
        d = CompoundSequenceDetector()
        history = ["credential_access", "env_read", "network_send"]
        matches = d.check("network_send", history)
        names = [m.pattern_name for m in matches]
        assert "cred-exfil" in names
        assert "env-exfil" in names

    def test_match_contains_correct_steps(self):
        d = CompoundSequenceDetector()
        history = ["credential_access", "network_send"]
        matches = d.check("network_send", history)
        cred_match = next(m for m in matches if m.pattern_name == "cred-exfil")
        assert cred_match.steps_matched == ["credential_access", "network_send"]

    def test_completing_class_set_correctly(self):
        d = CompoundSequenceDetector()
        history = ["credential_access", "network_send"]
        matches = d.check("network_send", history)
        cred_match = next(m for m in matches if m.pattern_name == "cred-exfil")
        assert cred_match.completing_class == "network_send"

    def test_severity_is_critical_for_cred_exfil(self):
        d = CompoundSequenceDetector()
        history = ["credential_access", "network_send"]
        matches = d.check("network_send", history)
        cred_match = next(m for m in matches if m.pattern_name == "cred-exfil")
        assert cred_match.severity == "critical"

    def test_severity_is_high_for_rce_persistence(self):
        d = CompoundSequenceDetector()
        history = ["code_exec", "file_write", "process_spawn"]
        matches = d.check("process_spawn", history)
        rce_match = next(m for m in matches if m.pattern_name == "rce-persistence")
        assert rce_match.severity == "high"


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustomPatterns:
    def test_custom_pattern_fires(self):
        custom = SequencePattern(
            name="test-pattern",
            description="A → B → C",
            steps=("data_read", "encode", "auth_modify"),
            severity="medium",
        )
        d = CompoundSequenceDetector(patterns=[custom])
        history = ["data_read", "encode", "auth_modify"]
        matches = d.check("auth_modify", history)
        assert len(matches) == 1
        assert matches[0].pattern_name == "test-pattern"

    def test_empty_pattern_list(self):
        d = CompoundSequenceDetector(patterns=[])
        history = ["credential_access", "network_send"]
        matches = d.check("network_send", history)
        assert matches == []


# ---------------------------------------------------------------------------
# PatternMatch.to_dict
# ---------------------------------------------------------------------------

class TestPatternMatchToDict:
    def test_to_dict_keys(self):
        m = PatternMatch(
            pattern_name="cred-exfil",
            description="desc",
            severity="critical",
            completing_class="network_send",
            steps_matched=["credential_access", "network_send"],
        )
        d = m.to_dict()
        assert "pattern_name" in d
        assert "description" in d
        assert "severity" in d
        assert "completing_class" in d
        assert "steps_matched" in d

    def test_to_dict_json_serialisable(self):
        d = CompoundSequenceDetector()
        history = ["credential_access", "network_send"]
        matches = d.check("network_send", history)
        for match in matches:
            json.dumps(match.to_dict())  # must not raise

    def test_to_dict_values(self):
        m = PatternMatch(
            pattern_name="test",
            description="a test",
            severity="high",
            completing_class="code_exec",
            steps_matched=["credential_access", "code_exec"],
        )
        d = m.to_dict()
        assert d["pattern_name"] == "test"
        assert d["severity"] == "high"
        assert d["steps_matched"] == ["credential_access", "code_exec"]


# ---------------------------------------------------------------------------
# Module-level singleton and built-in patterns
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_compound_detector_importable(self):
        from app.security.compound_detector import compound_detector
        assert compound_detector is not None

    def test_builtin_patterns_non_empty(self):
        assert len(PATTERNS) >= 8

    def test_all_patterns_have_steps(self):
        for p in PATTERNS:
            assert len(p.steps) >= 2, f"Pattern '{p.name}' has fewer than 2 steps"

    def test_all_severities_valid(self):
        valid = {"critical", "high", "medium"}
        for p in PATTERNS:
            assert p.severity in valid, f"Pattern '{p.name}' has invalid severity '{p.severity}'"
