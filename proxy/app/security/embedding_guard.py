"""
Phase E4 — Embedding call security guard.

Scans text passed to embedding APIs (OpenAI /v1/embeddings, etc.) for:

  1. PII leakage — detected via presidio-analyzer (ML + NLP, low false-positive
     rate). Falls back to a minimal high-precision regex set when presidio is
     not installed.
  2. Volume anomaly — per-session embedding call counter tracked in session state.

Why presidio over pure regex
-----------------------------
Regex is structural — it fires on *shape*, not *meaning*.  The SSN pattern
``\\d{3}-\\d{2}-\\d{4}`` matches ZIP+4 suffixes, order numbers, and serial
codes.  A 16-digit Visa-prefix number passes a card regex but may not pass
the Luhn checksum.  Presidio uses spaCy NER + context heuristics to
distinguish "patient SSN: 123-45-6789" from "product code: 123-45-6789",
cutting the false-positive rate dramatically.

The regex fallback (email + IBAN) is kept because:
- Email has very low FPR (RFC 5321 format is highly specific)
- IBAN has a 2-letter country prefix + mod-97 checksum structure that is
  rarely matched accidentally

All other types (SSN, credit card, phone) are presidio-only to avoid FPs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# High-precision regex fallback (low FPR only)
# These run ONLY when presidio is not installed.
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
# IBAN: 2-letter country + 2 check digits + up to 30 alphanumeric chars
# The structure is highly specific; accidental matches are rare.
_IBAN_RE  = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b")

_REGEX_PATTERNS: list[tuple[re.Pattern, str, bool]] = [
    (_EMAIL_RE, "EMAIL_ADDRESS", False),
    (_IBAN_RE,  "IBAN_CODE",     True),
]

# Presidio entity types considered critical (high-severity PII)
_CRITICAL_TYPES = frozenset({
    "US_SSN", "CREDIT_CARD", "IBAN_CODE",
    "MEDICAL_LICENSE", "UK_NHS", "US_PASSPORT",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingGuardResult:
    """Result from scanning one embedding API call."""

    pii_detected: bool
    """True if any PII was found in the input texts."""

    critical_pii: bool
    """True if any critical-severity PII type was detected (SSN, CC, IBAN, etc.)."""

    pii_types: list[str]
    """Deduplicated, sorted list of detected PII entity types."""

    total_chars: int
    """Total character count across all input strings."""

    input_count: int
    """Number of strings in the embedding batch."""

    used_presidio: bool
    """Whether presidio was available and used (False = regex-only fallback)."""

    def to_dict(self) -> dict:
        return {
            "pii_detected":  self.pii_detected,
            "critical_pii":  self.critical_pii,
            "pii_types":     self.pii_types,
            "total_chars":   self.total_chars,
            "input_count":   self.input_count,
            "used_presidio": self.used_presidio,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_embedding_input(texts: list[str]) -> EmbeddingGuardResult:
    """
    Scan a batch of strings to be embedded for PII.

    Args:
        texts: The ``input`` field from an OpenAI-compatible embeddings request,
               already normalised to list[str] by ``extract_embedding_texts()``.

    Returns:
        EmbeddingGuardResult with PII findings and stats.
    """
    total_chars = sum(len(t) for t in texts)
    combined = "\n".join(texts)

    found_types, critical, used_presidio = _detect_pii(combined)

    pii_types = sorted(found_types)
    return EmbeddingGuardResult(
        pii_detected=bool(found_types),
        critical_pii=critical,
        pii_types=pii_types,
        total_chars=total_chars,
        input_count=len(texts),
        used_presidio=used_presidio,
    )


def extract_embedding_texts(body: dict) -> list[str]:
    """
    Normalise the ``input`` field of an OpenAI-compatible embeddings request
    to a flat list of strings.

    The field can be:
    - str: single text
    - list[str]: batch of texts
    - list[list[int]]: pre-tokenised batch — return placeholder strings
    - list[int]: single pre-tokenised input — each int becomes a string
    """
    raw = body.get("input", "")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        result: list[str] = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, list):
                # Pre-tokenized token array — can't scan tokens, use placeholder
                result.append(f"<{len(item)} pre-tokenised tokens>")
            else:
                result.append(str(item))
        return result or [""]
    return [str(raw)]


# ---------------------------------------------------------------------------
# Internal detection logic
# ---------------------------------------------------------------------------

def _detect_pii(text: str) -> tuple[set[str], bool, bool]:
    """
    Detect PII in text using presidio (preferred) or regex fallback.

    Returns:
        (entity_types, any_critical, used_presidio)
    """
    # ── Attempt presidio (ML + NLP, low FPR) ──────────────────────────────
    engine = _get_presidio_engine()
    if engine is not None:
        return _detect_presidio(text, engine)

    # ── Fallback: high-precision regex only ────────────────────────────────
    logger.debug(
        "presidio not installed — embedding PII scan using regex fallback "
        "(email + IBAN only; SSN/CC/phone require presidio for low FPR)"
    )
    return _detect_regex(text)


def _detect_presidio(text: str, engine) -> tuple[set[str], bool, bool]:
    try:
        found: set[str] = set()
        critical = False
        for result in engine.analyze(text=text, language="en"):
            found.add(result.entity_type)
            if result.entity_type in _CRITICAL_TYPES:
                critical = True
        return found, critical, True
    except Exception as exc:
        logger.warning("presidio analysis failed, falling back to regex: %s", exc)
        types, crit, _ = _detect_regex(text)
        return types, crit, False


def _detect_regex(text: str) -> tuple[set[str], bool, bool]:
    found: set[str] = set()
    critical = False
    for pattern, entity_type, is_critical in _REGEX_PATTERNS:
        if pattern.search(text):
            found.add(entity_type)
            if is_critical:
                critical = True
    return found, critical, False


# ---------------------------------------------------------------------------
# Presidio engine singleton (lazy — expensive to construct)
# ---------------------------------------------------------------------------

_presidio_engine = None
_presidio_checked = False


def _get_presidio_engine():
    """Return the presidio AnalyzerEngine, or None if not installed."""
    global _presidio_engine, _presidio_checked
    if _presidio_checked:
        return _presidio_engine
    _presidio_checked = True
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
        _presidio_engine = AnalyzerEngine()
        logger.debug("presidio AnalyzerEngine loaded for embedding PII scanning")
    except ImportError:
        _presidio_engine = None
        logger.debug("presidio not installed; embedding guard using regex fallback")
    return _presidio_engine
