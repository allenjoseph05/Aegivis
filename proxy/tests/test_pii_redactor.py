"""Tests for PII Redaction & Tokenization Engine."""
from __future__ import annotations

import pytest
from proxy.app.security.pii_redactor import (
    PIIVault, RedactionConfig, CustomPattern, PIIBlockError,
    apply_redaction, redact_messages,
    _fallback_findings, _custom_findings, _merge_findings,
    DEFAULT_PII_MODES,
)


# ---------------------------------------------------------------------------
# PIIVault
# ---------------------------------------------------------------------------

class TestPIIVault:
    def test_store_and_restore(self):
        vault = PIIVault()
        token = vault.store("alice@example.com", "EMAIL_ADDRESS")
        assert token == "<PII-EMAIL_ADDRESS-1>"
        restored = vault.restore(f"Contact {token} for details")
        assert "alice@example.com" in restored

    def test_deduplication(self):
        vault = PIIVault()
        t1 = vault.store("alice@example.com", "EMAIL_ADDRESS")
        t2 = vault.store("alice@example.com", "EMAIL_ADDRESS")
        assert t1 == t2

    def test_different_values_different_tokens(self):
        vault = PIIVault()
        t1 = vault.store("alice@example.com", "EMAIL_ADDRESS")
        t2 = vault.store("bob@example.com", "EMAIL_ADDRESS")
        assert t1 != t2
        assert t1 == "<PII-EMAIL_ADDRESS-1>"
        assert t2 == "<PII-EMAIL_ADDRESS-2>"

    def test_counter_per_type(self):
        vault = PIIVault()
        vault.store("alice@example.com", "EMAIL_ADDRESS")
        vault.store("555-1234", "PHONE_NUMBER")
        assert vault._counters["EMAIL_ADDRESS"] == 1
        assert vault._counters["PHONE_NUMBER"] == 1

    def test_restore_multiple_tokens(self):
        vault = PIIVault()
        t1 = vault.store("alice@example.com", "EMAIL_ADDRESS")
        t2 = vault.store("555-555-5555", "PHONE_NUMBER")
        text = f"Email: {t1}, Phone: {t2}"
        restored = vault.restore(text)
        assert "alice@example.com" in restored
        assert "555-555-5555" in restored

    def test_restore_unknown_token_unchanged(self):
        vault = PIIVault()
        text = "Contact <PII-EMAIL_ADDRESS-99> for info"
        assert vault.restore(text) == text

    def test_size(self):
        vault = PIIVault()
        assert vault.size() == 0
        vault.store("a@b.com", "EMAIL_ADDRESS")
        vault.store("a@b.com", "EMAIL_ADDRESS")   # dedup
        vault.store("c@d.com", "EMAIL_ADDRESS")
        assert vault.size() == 2

    def test_summary(self):
        vault = PIIVault()
        vault.store("a@b.com", "EMAIL_ADDRESS")
        vault.store("c@d.com", "EMAIL_ADDRESS")
        vault.store("555-1234", "PHONE_NUMBER")
        s = vault.summary()
        assert s["total_tokens"] == 3
        assert s["by_type"]["EMAIL_ADDRESS"] == 2
        assert s["by_type"]["PHONE_NUMBER"] == 1

    def test_restore_empty_vault(self):
        vault = PIIVault()
        text = "No PII here"
        assert vault.restore(text) == text


# ---------------------------------------------------------------------------
# CustomPattern
# ---------------------------------------------------------------------------

class TestCustomPattern:
    def test_valid_pattern_compiles(self):
        cp = CustomPattern(id="1", name="EmpID", pattern=r"EMP-\d{6}", type_name="EMPLOYEE_ID", mode="redact")
        assert cp._compiled is not None

    def test_invalid_pattern_degrades(self):
        cp = CustomPattern(id="1", name="Bad", pattern="[invalid", type_name="BAD", mode="redact")
        assert cp._compiled is None

    def test_from_dict(self):
        d = {"name": "Employee ID", "pattern": r"EMP-\d{6}", "type_name": "EMPLOYEE_ID", "mode": "tokenize"}
        cp = CustomPattern.from_dict(d)
        assert cp.name == "Employee ID"
        assert cp.mode == "tokenize"
        assert cp.id  # auto-generated

    def test_to_dict_roundtrip(self):
        cp = CustomPattern(id="abc", name="EmpID", pattern=r"EMP-\d+", type_name="EMPLOYEE_ID", mode="redact")
        d = cp.to_dict()
        assert d["id"] == "abc"
        assert d["mode"] == "redact"


# ---------------------------------------------------------------------------
# RedactionConfig
# ---------------------------------------------------------------------------

class TestRedactionConfig:
    def test_default_modes(self):
        cfg = RedactionConfig()
        assert cfg.get_mode("EMAIL_ADDRESS") == "tokenize"
        assert cfg.get_mode("US_SSN") == "block"
        assert cfg.get_mode("CREDIT_CARD") == "block"
        assert cfg.get_mode("PERSON") == "allow"

    def test_unknown_type_defaults_to_allow(self):
        cfg = RedactionConfig()
        assert cfg.get_mode("UNKNOWN_TYPE") == "allow"

    def test_custom_mode_override(self):
        cfg = RedactionConfig(pii_type_modes={"EMAIL_ADDRESS": "block"})
        assert cfg.get_mode("EMAIL_ADDRESS") == "block"


# ---------------------------------------------------------------------------
# Fallback regex findings
# ---------------------------------------------------------------------------

class TestFallbackFindings:
    def test_detects_email(self):
        cfg = RedactionConfig()
        findings = _fallback_findings("Send to alice@example.com today", cfg)
        types = {f.pii_type for f in findings}
        assert "EMAIL_ADDRESS" in types

    def test_detects_phone(self):
        cfg = RedactionConfig()
        findings = _fallback_findings("Call 555-123-4567", cfg)
        types = {f.pii_type for f in findings}
        assert "PHONE_NUMBER" in types

    def test_detects_ssn(self):
        cfg = RedactionConfig()
        findings = _fallback_findings("SSN: 123-45-6789", cfg)
        types = {f.pii_type for f in findings}
        assert "US_SSN" in types

    def test_allow_mode_skipped(self):
        cfg = RedactionConfig(pii_type_modes={**DEFAULT_PII_MODES, "EMAIL_ADDRESS": "allow"})
        findings = _fallback_findings("alice@example.com", cfg)
        assert not any(f.pii_type == "EMAIL_ADDRESS" for f in findings)

    def test_clean_text_no_findings(self):
        cfg = RedactionConfig()
        findings = _fallback_findings("The weather is nice today.", cfg)
        assert findings == []


# ---------------------------------------------------------------------------
# Custom pattern findings
# ---------------------------------------------------------------------------

class TestCustomFindings:
    def test_detects_custom_employee_id(self):
        cp = CustomPattern(id="1", name="EmpID", pattern=r"EMP-\d{6}", type_name="EMPLOYEE_ID", mode="redact")
        cfg = RedactionConfig(custom_patterns=[cp])
        findings = _custom_findings("Employee EMP-123456 is assigned", cfg)
        assert len(findings) == 1
        assert findings[0].pii_type == "EMPLOYEE_ID"
        assert findings[0].mode == "redact"
        assert findings[0].original == "EMP-123456"

    def test_allow_mode_skipped(self):
        cp = CustomPattern(id="1", name="Safe", pattern=r"SAFE-\d+", type_name="SAFE", mode="allow")
        cfg = RedactionConfig(custom_patterns=[cp])
        findings = _custom_findings("SAFE-999", cfg)
        assert findings == []

    def test_invalid_compiled_skipped(self):
        cp = CustomPattern(id="1", name="Bad", pattern="[bad", type_name="BAD", mode="redact")
        cfg = RedactionConfig(custom_patterns=[cp])
        findings = _custom_findings("anything", cfg)
        assert findings == []

    def test_custom_score_is_1(self):
        cp = CustomPattern(id="1", name="EmpID", pattern=r"EMP-\d+", type_name="EMPLOYEE_ID", mode="tokenize")
        cfg = RedactionConfig(custom_patterns=[cp])
        findings = _custom_findings("EMP-001", cfg)
        assert findings[0].score == 1.0


# ---------------------------------------------------------------------------
# Merge findings
# ---------------------------------------------------------------------------

class TestMergeFindings:
    def _f(self, start, end, pii_type="EMAIL_ADDRESS", score=0.9, mode="tokenize"):
        from proxy.app.security.pii_redactor import PIIFinding
        return PIIFinding(start=start, end=end, pii_type=pii_type, mode=mode, score=score, original="x")

    def test_non_overlapping_kept(self):
        findings = [self._f(0, 5), self._f(10, 15)]
        merged = _merge_findings(findings)
        assert len(merged) == 2

    def test_overlapping_higher_score_kept(self):
        # Same span, different scores — first (higher) kept
        f1 = self._f(0, 10, score=0.9)
        f2 = self._f(0, 8, score=0.7)
        merged = _merge_findings([f1, f2])
        # After sort by start, f1 is added, f2 overlaps and is skipped
        assert len(merged) == 1

    def test_empty_returns_empty(self):
        assert _merge_findings([]) == []

    def test_single_kept(self):
        assert len(_merge_findings([self._f(5, 10)])) == 1


# ---------------------------------------------------------------------------
# apply_redaction
# ---------------------------------------------------------------------------

class TestApplyRedaction:
    def test_email_tokenized(self):
        vault = PIIVault()
        cfg = RedactionConfig()
        text = "Contact alice@example.com for info"
        result, findings = apply_redaction(text, vault, cfg)
        assert "alice@example.com" not in result
        assert "<PII-EMAIL_ADDRESS-1>" in result
        assert len(findings) >= 1

    def test_ssn_blocks(self):
        vault = PIIVault()
        cfg = RedactionConfig()
        with pytest.raises(PIIBlockError) as exc_info:
            apply_redaction("SSN: 123-45-6789", vault, cfg)
        assert exc_info.value.pii_type == "US_SSN"

    def test_redact_mode_irreversible(self):
        vault = PIIVault()
        cfg = RedactionConfig(pii_type_modes={**DEFAULT_PII_MODES, "EMAIL_ADDRESS": "redact"})
        result, _ = apply_redaction("alice@example.com", vault, cfg)
        assert "[REDACTED-EMAIL_ADDRESS]" in result
        assert vault.size() == 0   # nothing in vault for redact mode

    def test_allow_mode_passes_through(self):
        vault = PIIVault()
        cfg = RedactionConfig(pii_type_modes={**DEFAULT_PII_MODES, "EMAIL_ADDRESS": "allow"})
        result, findings = apply_redaction("alice@example.com", vault, cfg)
        assert result == "alice@example.com"

    def test_disabled_config_passthrough(self):
        vault = PIIVault()
        cfg = RedactionConfig(enabled=False)
        result, findings = apply_redaction("alice@example.com", vault, cfg)
        assert result == "alice@example.com"
        assert findings == []

    def test_custom_pattern_tokenized(self):
        vault = PIIVault()
        cp = CustomPattern(id="1", name="EmpID", pattern=r"EMP-\d{6}", type_name="EMPLOYEE_ID", mode="tokenize")
        cfg = RedactionConfig(custom_patterns=[cp])
        result, findings = apply_redaction("Employee EMP-123456 processed", vault, cfg)
        assert "EMP-123456" not in result
        assert "<PII-EMPLOYEE_ID-1>" in result

    def test_token_restore(self):
        vault = PIIVault()
        cfg = RedactionConfig()
        text = "Email alice@example.com for details"
        redacted, _ = apply_redaction(text, vault, cfg)
        restored = vault.restore(redacted)
        assert "alice@example.com" in restored

    def test_empty_text(self):
        vault = PIIVault()
        cfg = RedactionConfig()
        result, findings = apply_redaction("", vault, cfg)
        assert result == ""
        assert findings == []


# ---------------------------------------------------------------------------
# redact_messages
# ---------------------------------------------------------------------------

class TestRedactMessages:
    def _cfg(self):
        return RedactionConfig()

    def test_simple_string_content(self):
        vault = PIIVault()
        messages = [{"role": "user", "content": "My email is user@test.com"}]
        result, summary = redact_messages(messages, vault, self._cfg())
        assert "user@test.com" not in result[0]["content"]
        assert summary["total_findings"] >= 1

    def test_multipart_content(self):
        vault = PIIVault()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Email: user@test.com"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]}]
        result, summary = redact_messages(messages, vault, self._cfg())
        text_part = result[0]["content"][0]["text"]
        assert "user@test.com" not in text_part

    def test_tool_result_content(self):
        vault = PIIVault()
        messages = [{"role": "user", "content": [
            {"type": "tool_result", "content": "Customer email: customer@corp.com"},
        ]}]
        result, summary = redact_messages(messages, vault, self._cfg())
        assert "customer@corp.com" not in result[0]["content"][0]["content"]

    def test_block_propagates(self):
        vault = PIIVault()
        cfg = RedactionConfig()
        messages = [{"role": "user", "content": "SSN: 123-45-6789"}]
        with pytest.raises(PIIBlockError):
            redact_messages(messages, vault, cfg)

    def test_disabled_passthrough(self):
        vault = PIIVault()
        cfg = RedactionConfig(enabled=False)
        messages = [{"role": "user", "content": "My email is user@test.com"}]
        result, summary = redact_messages(messages, vault, cfg)
        assert result[0]["content"] == "My email is user@test.com"
        assert summary["total_findings"] == 0

    def test_original_not_mutated(self):
        vault = PIIVault()
        messages = [{"role": "user", "content": "Email: user@test.com"}]
        original_content = messages[0]["content"]
        redact_messages(messages, vault, self._cfg())
        # Original list should be unchanged (deep copy)
        assert messages[0]["content"] == original_content

    def test_clean_messages_no_findings(self):
        vault = PIIVault()
        messages = [{"role": "user", "content": "What is the weather today?"}]
        result, summary = redact_messages(messages, vault, self._cfg())
        assert summary["total_findings"] == 0
        assert result[0]["content"] == messages[0]["content"]
