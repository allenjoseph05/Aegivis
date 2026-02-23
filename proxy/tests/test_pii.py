"""Tests for PII detection and masking (Presidio integration)."""
import json
import pytest
from unittest.mock import patch
from proxy.app.pii import analyze_and_mask, hash_original, mask_dict, PIIResult


# MUST be defined before any @pytest.mark.skipif that calls it
def _presidio_available() -> bool:
    try:
        import presidio_analyzer  # noqa
        return True
    except ImportError:
        return False


class TestHashOriginal:
    def test_produces_64_char_hex(self):
        h = hash_original("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert hash_original("test") == hash_original("test")

    def test_different_inputs_different_hashes(self):
        assert hash_original("foo") != hash_original("bar")


class TestAnalyzeAndMask:
    def test_no_pii_text_returns_pii_result(self):
        result = analyze_and_mask("The weather is nice today.")
        assert isinstance(result, PIIResult)
        assert result.original_hash is not None
        assert len(result.original_hash) == 64

    def test_original_hash_computed_regardless(self):
        text = "Hello, my name is John Doe"
        result = analyze_and_mask(text)
        assert result.original_hash == hash_original(text)

    def test_empty_string(self):
        result = analyze_and_mask("")
        assert result.masked_text == ""
        assert result.pii_detected is False

    def test_presidio_not_available_fallback(self):
        """When Presidio is not installed, returns original text unchanged."""
        with patch("proxy.app.pii._get_engines", return_value=(None, None)):
            result = analyze_and_mask("Email me at test@example.com")
            assert result.masked_text == "Email me at test@example.com"
            assert result.entity_types == []
            assert result.pii_detected is False

    @pytest.mark.skipif(
        not _presidio_available(),
        reason="Presidio not installed — run: pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg"
    )
    def test_email_detection(self):
        result = analyze_and_mask("Contact us at john.doe@example.com for support.")
        assert result.pii_detected is True
        assert "EMAIL_ADDRESS" in result.entity_types
        assert "john.doe@example.com" not in result.masked_text
        assert "<EMAIL_ADDRESS>" in result.masked_text

    @pytest.mark.skipif(
        not _presidio_available(),
        reason="Presidio not installed"
    )
    def test_person_name_detection(self):
        result = analyze_and_mask("Hello, my name is John Smith and I need help.")
        assert result.pii_detected is True
        assert "PERSON" in result.entity_types

    @pytest.mark.skipif(
        not _presidio_available(),
        reason="Presidio not installed"
    )
    def test_masked_text_replaces_pii(self):
        result = analyze_and_mask("My SSN is 123-45-6789.")
        assert "123-45-6789" not in result.masked_text


class TestMaskDict:
    def test_simple_string_value(self):
        with patch("proxy.app.pii.analyze_and_mask") as mock_analyze:
            mock_analyze.return_value = PIIResult(
                masked_text="Hello <PERSON>",
                entity_types=["PERSON"],
                original_hash="a" * 64,
                pii_detected=True,
            )
            masked, entities, orig_hash = mask_dict({"content": "Hello John"})
            assert "PERSON" in entities
            assert len(orig_hash) == 64

    def test_nested_dict(self):
        obj = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        masked, entities, orig_hash = mask_dict(obj)
        assert isinstance(masked, dict)
        assert isinstance(entities, list)
        assert len(orig_hash) == 64

    def test_original_hash_from_full_object(self):
        import hashlib
        obj = {"key": "value", "number": 42}
        _, _, orig_hash = mask_dict(obj)
        expected = hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        assert orig_hash == expected

    def test_non_string_values_passthrough(self):
        obj = {"count": 42, "active": True, "ratio": 3.14}
        masked, entities, _ = mask_dict(obj)
        assert masked["count"] == 42
        assert masked["active"] is True
        assert masked["ratio"] == 3.14

    def test_empty_dict(self):
        masked, entities, orig_hash = mask_dict({})
        assert masked == {}
        assert entities == []
        assert len(orig_hash) == 64

    def test_list_values_recursed(self):
        obj = {"items": ["hello", "world"]}
        masked, entities, orig_hash = mask_dict(obj)
        assert isinstance(masked["items"], list)
        assert len(masked["items"]) == 2
