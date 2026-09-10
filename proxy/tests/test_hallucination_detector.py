"""Tests for SourceAccuracyDetector (formerly HallucinationDetector).

MiniCheck is optional — without it the detector fails open (checked=False).
Logprob-based scoring was removed: weak signal, arbitrary threshold, high FP rate.
"""
from __future__ import annotations

import pytest
from proxy.app.security.hallucination_detector import (
    SourceAccuracyConfig,
    SourceAccuracyDetector,
    SourceAccuracyFinding,
    SourceAccuracyResult,
    # backward-compat aliases (also tested to ensure they still resolve)
    HallucinationConfig,
    HallucinationDetector,
    HallucinationFinding,
    HallucinationResult,
    extract_factual_sentences,
    _serialize_tool_result,
)


# ── extract_factual_sentences ──────────────────────────────────────────────────

class TestExtractFactualSentences:
    def test_sentence_with_number_kept(self):
        sentences = extract_factual_sentences("The temperature is 18°C today.")
        assert len(sentences) == 1
        assert "18" in sentences[0]

    def test_short_sentence_filtered(self):
        sentences = extract_factual_sentences("OK.")
        assert sentences == []

    def test_short_sentences_filtered_by_length(self):
        # Filtering is length-only (no prefix/keyword heuristic).
        # Short sentences (< 20 chars) are excluded; longer ones pass through.
        sentences = extract_factual_sentences("OK. Sure. Yes.")
        assert sentences == []

    def test_proper_noun_kept(self):
        sentences = extract_factual_sentences("The weather in Paris is currently sunny.")
        assert len(sentences) == 1

    def test_multiple_sentences_mixed(self):
        text = (
            "Let me summarize what I found. "
            "The account balance is $1,234.56. "
            "You can review it later."
        )
        sentences = extract_factual_sentences(text)
        # Only the balance sentence is factual (contains $)
        assert any("1,234" in s for s in sentences)

    def test_status_word_kept(self):
        sentences = extract_factual_sentences("The API request returns an error response.")
        assert len(sentences) == 1

    def test_empty_text_returns_empty(self):
        assert extract_factual_sentences("") == []

    def test_only_whitespace_returns_empty(self):
        assert extract_factual_sentences("   ") == []


# ── _serialize_tool_result ─────────────────────────────────────────────────────

class TestSerializeToolResult:
    def test_string_passthrough(self):
        assert _serialize_tool_result("hello") == "hello"

    def test_dict_serialized(self):
        out = _serialize_tool_result({"key": "value"})
        assert "key" in out
        assert "value" in out

    def test_list_serialized(self):
        out = _serialize_tool_result([1, 2, 3])
        assert "1" in out

    def test_none_returns_empty(self):
        assert _serialize_tool_result(None) == ""

    def test_truncation(self):
        long_str = "x" * 2000
        result = _serialize_tool_result(long_str)
        assert len(result) <= 1500


# ── SourceAccuracyFinding ─────────────────────────────────────────────────────

class TestSourceAccuracyFinding:
    def _make(self, score=0.1, severity="HIGH"):
        return SourceAccuracyFinding(
            sentence="The balance is $100.",
            tool_name="get_balance",
            tool_snippet='{"error": "not_found"}',
            score=score,
            method="minicheck",
            severity=severity,
        )

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        assert set(d.keys()) == {"sentence", "tool_name", "tool_snippet", "score", "method", "severity"}

    def test_score_rounded(self):
        f = SourceAccuracyFinding(
            sentence="x" * 25, tool_name="t", tool_snippet="s",
            score=0.123456789, method="minicheck", severity="LOW",
        )
        assert f.to_dict()["score"] == 0.1235

    def test_sentence_truncated(self):
        f = SourceAccuracyFinding(
            sentence="x" * 400, tool_name="t", tool_snippet="s",
            score=0.1, method="minicheck", severity="HIGH",
        )
        assert len(f.to_dict()["sentence"]) == 300


# ── SourceAccuracyResult ──────────────────────────────────────────────────────

class TestSourceAccuracyResult:
    def test_detected_false_when_no_findings(self):
        r = SourceAccuracyResult(findings=[], checked=True)
        assert r.detected is False

    def test_detected_true_when_findings(self):
        f = SourceAccuracyFinding("s" * 25, "t", "doc", 0.1, "minicheck", "HIGH")
        r = SourceAccuracyResult(findings=[f], checked=True)
        assert r.detected is True

    def test_severity_none_when_no_findings(self):
        r = SourceAccuracyResult(findings=[], checked=True)
        assert r.severity == "NONE"

    def test_severity_high(self):
        f = SourceAccuracyFinding("s" * 25, "t", "doc", 0.05, "minicheck", "HIGH")
        r = SourceAccuracyResult(findings=[f], checked=True)
        assert r.severity == "HIGH"

    def test_severity_medium(self):
        f = SourceAccuracyFinding("s" * 25, "t", "doc", 0.2, "minicheck", "MEDIUM")
        r = SourceAccuracyResult(findings=[f], checked=True)
        assert r.severity == "MEDIUM"

    def test_to_dict_keys(self):
        r = SourceAccuracyResult(findings=[], checked=True)
        d = r.to_dict()
        assert "checked" in d
        assert "detected" in d
        assert "finding_count" in d
        assert "severity" in d
        assert "findings" in d


# ── SourceAccuracyDetector (no MiniCheck) ────────────────────────────────────

class TestSourceAccuracyDetectorNoMiniCheck:
    """Without MiniCheck installed the detector fails open (checked=False, no findings)."""

    def _cfg(self, **kw):
        defaults = dict(
            enabled=True,
            action="alert",
            threshold=0.35,
            use_minicheck=False,
        )
        defaults.update(kw)
        return SourceAccuracyConfig(**defaults)

    def _detector(self):
        return SourceAccuracyDetector()

    def test_disabled_returns_unchecked(self):
        cfg = self._cfg(enabled=False)
        r = self._detector().check(
            tool_results=[{"tool_name": "t", "content": "x"}],
            llm_response="The answer is 42.",
            config=cfg,
        )
        assert r.checked is False

    def test_no_tool_results_skipped(self):
        r = self._detector().check(
            tool_results=[],
            llm_response="The answer is 42.",
            config=self._cfg(),
        )
        assert r.checked is False

    def test_no_response_skipped(self):
        r = self._detector().check(
            tool_results=[{"tool_name": "t", "content": "some data"}],
            llm_response=None,
            config=self._cfg(),
        )
        assert r.checked is False

    def test_empty_response_skipped(self):
        r = self._detector().check(
            tool_results=[{"tool_name": "t", "content": "some data"}],
            llm_response="",
            config=self._cfg(),
        )
        assert r.checked is False

    def test_minicheck_not_available_returns_unchecked(self):
        """Without MiniCheck the detector fails open to avoid FPs."""
        r = self._detector().check(
            tool_results=[{"tool_name": "t", "content": "data"}],
            llm_response="The value is 100.",
            config=self._cfg(use_minicheck=False),
        )
        assert r.checked is False
        assert r.minicheck_available is False
        assert r.detected is False


# ── SourceAccuracyConfig ──────────────────────────────────────────────────────

class TestSourceAccuracyConfig:
    def test_defaults(self):
        cfg = SourceAccuracyConfig()
        assert cfg.enabled is True
        assert cfg.action == "alert"
        assert cfg.threshold == pytest.approx(0.35)
        assert cfg.use_minicheck is True

    def test_custom_threshold(self):
        cfg = SourceAccuracyConfig(threshold=0.5)
        assert cfg.threshold == 0.5

    def test_block_action(self):
        cfg = SourceAccuracyConfig(action="block")
        assert cfg.action == "block"


# ── Backward-compatibility aliases ────────────────────────────────────────────

class TestBackwardCompatAliases:
    """Verify the old Hallucination* names still resolve to the new types."""

    def test_hallucination_finding_is_source_accuracy_finding(self):
        assert HallucinationFinding is SourceAccuracyFinding

    def test_hallucination_result_is_source_accuracy_result(self):
        assert HallucinationResult is SourceAccuracyResult

    def test_hallucination_config_is_source_accuracy_config(self):
        assert HallucinationConfig is SourceAccuracyConfig

    def test_hallucination_detector_is_source_accuracy_detector(self):
        assert HallucinationDetector is SourceAccuracyDetector
