"""Tests for Phase E4 — Embedding call security guard."""
import pytest
import httpx
import respx
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.app.security.embedding_guard import (
    scan_embedding_input,
    extract_embedding_texts,
    EmbeddingGuardResult,
    _detect_regex,
    _detect_presidio,
)
from proxy.app.main import app


# ─── Transport mock ────────────────────────────────────────────────────────────

def _make_mock_transport():
    t = MagicMock()
    t.enqueue = MagicMock()
    t.enqueue_violation = MagicMock()
    t.buffer_status = MagicMock(return_value={"events": 0, "violations": 0})
    t.start = AsyncMock()
    t.stop = AsyncMock()
    return t


@pytest.fixture()
def emb_client():
    """TestClient with mocked transport."""
    mock_transport = _make_mock_transport()

    async def _mock_get_best(*, force_http=False):
        return mock_transport

    with (
        patch("proxy.app.main.get_transport", return_value=mock_transport),
        patch("proxy.app.intercept.get_transport", return_value=mock_transport),
        patch("proxy.app.transport.get_best_transport", side_effect=_mock_get_best),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_transport


# ─── extract_embedding_texts ───────────────────────────────────────────────────

class TestExtractEmbeddingTexts:
    def test_string_input(self):
        assert extract_embedding_texts({"input": "hello world"}) == ["hello world"]

    def test_list_of_strings(self):
        assert extract_embedding_texts({"input": ["hello", "world"]}) == ["hello", "world"]

    def test_pre_tokenized_list_of_lists(self):
        result = extract_embedding_texts({"input": [[1, 2, 3], [4, 5, 6]]})
        assert len(result) == 2
        assert "pre-tokenised" in result[0]

    def test_pre_tokenized_single_array(self):
        result = extract_embedding_texts({"input": [1, 2, 3, 4]})
        # Each int becomes str("1") etc.
        assert len(result) > 0

    def test_empty_input_key(self):
        assert extract_embedding_texts({}) == [""]

    def test_non_string_scalar(self):
        assert extract_embedding_texts({"input": 42}) == ["42"]


# ─── Regex fallback — _detect_regex ────────────────────────────────────────────
# Tests the low-FPR regex-only path (no presidio).
# Only email and IBAN are detected via regex; SSN/CC/phone require presidio.

class TestDetectRegex:
    def test_email_detected(self):
        types, critical, used_presidio = _detect_regex("contact alice@example.com")
        assert "EMAIL_ADDRESS" in types
        assert critical is False
        assert used_presidio is False

    def test_iban_detected_as_critical(self):
        types, critical, _ = _detect_regex("IBAN: GB29NWBK60161331926819")
        assert "IBAN_CODE" in types
        assert critical is True

    def test_clean_text_no_match(self):
        types, critical, _ = _detect_regex("The weather is nice today")
        assert types == set()
        assert critical is False

    def test_ssn_not_in_regex_fallback(self):
        # SSN is intentionally NOT in regex fallback (high FPR)
        types, _, _ = _detect_regex("SSN: 123-45-6789")
        assert "US_SSN" not in types

    def test_phone_not_in_regex_fallback(self):
        # Phone is intentionally NOT in regex fallback (high FPR)
        types, _, _ = _detect_regex("Call (555) 867-5309")
        assert "PHONE_NUMBER" not in types

    def test_credit_card_not_in_regex_fallback(self):
        # CC requires Luhn validation — regex alone has too many FPs
        types, _, _ = _detect_regex("Card: 4111111111111111")
        assert "CREDIT_CARD" not in types


# ─── scan_embedding_input — integration ───────────────────────────────────────

class TestScanEmbeddingInput:
    def test_returns_dataclass(self):
        result = scan_embedding_input(["hello"])
        assert isinstance(result, EmbeddingGuardResult)

    def test_stats_computed(self):
        result = scan_embedding_input(["hello", "world"])
        assert result.total_chars == 10
        assert result.input_count == 2

    def test_clean_text_no_pii(self):
        result = scan_embedding_input(["The Fibonacci sequence: 1,1,2,3,5,8,13"])
        # May use presidio or regex — either way should not fire on this
        assert result.pii_detected is False

    def test_email_detected_via_any_backend(self):
        result = scan_embedding_input(["Contact alice@example.com for support"])
        assert result.pii_detected is True
        assert "EMAIL_ADDRESS" in result.pii_types

    def test_iban_detected_as_critical(self):
        result = scan_embedding_input(["Transfer to IBAN GB29NWBK60161331926819"])
        assert result.pii_detected is True
        assert result.critical_pii is True

    def test_pii_types_are_sorted(self):
        result = scan_embedding_input(["alice@example.com", "world"])
        assert result.pii_types == sorted(result.pii_types)

    def test_used_presidio_flag_is_bool(self):
        result = scan_embedding_input(["text"])
        assert isinstance(result.used_presidio, bool)

    def test_to_dict_has_all_keys(self):
        d = scan_embedding_input(["text"]).to_dict()
        for key in ("pii_detected", "critical_pii", "pii_types",
                    "total_chars", "input_count", "used_presidio"):
            assert key in d

    def test_empty_list(self):
        result = scan_embedding_input([])
        assert result.input_count == 0
        assert result.total_chars == 0
        assert result.pii_detected is False


# ─── Route integration ─────────────────────────────────────────────────────────

class TestEmbeddingRoute:
    UPSTREAM_URL = "https://api.openai.com"
    FAKE_RESPONSE = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    CLEAN_BODY = {"model": "text-embedding-ada-002", "input": "The sky is blue."}

    @respx.mock
    def test_clean_text_passes_through(self, emb_client):
        client, _ = emb_client
        respx.post(f"{self.UPSTREAM_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        resp = client.post("/openai/v1/embeddings", json=self.CLEAN_BODY)
        assert resp.status_code == 200
        assert "data" in resp.json()

    @respx.mock
    def test_session_id_returned_in_header(self, emb_client):
        client, _ = emb_client
        respx.post(f"{self.UPSTREAM_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        resp = client.post("/openai/v1/embeddings", json=self.CLEAN_BODY)
        assert "x-aegivis-session-id" in resp.headers

    @respx.mock
    def test_embedding_event_enqueued(self, emb_client):
        client, transport = emb_client
        respx.post(f"{self.UPSTREAM_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        client.post("/openai/v1/embeddings", json=self.CLEAN_BODY)
        assert transport.enqueue.called
        event = transport.enqueue.call_args[0][0]
        assert event["event_type"] == "EMBEDDING_CALL"

    @respx.mock
    def test_pii_triggers_violation(self, emb_client):
        client, transport = emb_client
        respx.post(f"{self.UPSTREAM_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        pii_body = {
            "model": "text-embedding-ada-002",
            "input": "Please email alice@example.com or wire to GB29NWBK60161331926819",
        }
        resp = client.post("/openai/v1/embeddings", json=pii_body)
        assert resp.status_code == 200   # ALERT, not blocked
        assert transport.enqueue_violation.called

    @respx.mock
    def test_upstream_5xx_forwarded(self, emb_client):
        client, _ = emb_client
        respx.post(f"{self.UPSTREAM_URL}/v1/embeddings").mock(
            return_value=httpx.Response(503, json={"error": {"message": "overloaded"}})
        )
        resp = client.post("/openai/v1/embeddings", json=self.CLEAN_BODY)
        assert resp.status_code == 503

    @respx.mock
    def test_batch_input_list(self, emb_client):
        client, transport = emb_client
        respx.post(f"{self.UPSTREAM_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=self.FAKE_RESPONSE)
        )
        batch_body = {
            "model": "text-embedding-ada-002",
            "input": ["text one", "text two", "text three"],
        }
        resp = client.post("/openai/v1/embeddings", json=batch_body)
        assert resp.status_code == 200
        event = transport.enqueue.call_args[0][0]
        assert event["payload"]["input_count"] == 3
