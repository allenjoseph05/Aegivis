"""
PII detection and masking using Microsoft Presidio (local NER, no cloud).

Presidio uses spaCy's en_core_web_lg model for context-aware named entity recognition.
Supports 50+ entity types: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,
US_SSN, IP_ADDRESS, MEDICAL_LICENSE, DATE_OF_BIRTH, IBAN_CODE, etc.

Setup (one-time):
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_PRESIDIO_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _PRESIDIO_AVAILABLE = True
except ImportError:
    logger.warning(
        "presidio-analyzer/presidio-anonymizer not installed. "
        "PII detection will be disabled. "
        "Install: pip install presidio-analyzer presidio-anonymizer && "
        "python -m spacy download en_core_web_lg"
    )


@lru_cache(maxsize=1)
def _get_engines() -> tuple[Any, Any] | tuple[None, None]:
    """Lazily initialize Presidio engines (expensive — load once)."""
    if not _PRESIDIO_AVAILABLE:
        return None, None
    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        logger.info("Presidio engines initialized (spaCy NER ready)")
        return analyzer, anonymizer
    except Exception as e:
        logger.error(f"Failed to initialize Presidio: {e}")
        return None, None


class PIIResult:
    """Result of PII analysis and anonymization."""

    __slots__ = ("masked_text", "entity_types", "original_hash", "pii_detected")

    def __init__(
        self,
        masked_text: str,
        entity_types: list[str],
        original_hash: str,
        pii_detected: bool,
    ):
        self.masked_text = masked_text
        self.entity_types = entity_types
        self.original_hash = original_hash
        self.pii_detected = pii_detected


def hash_original(text: str) -> str:
    """SHA-256 hash of original text before masking (forensic proof of content)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analyze_and_mask(text: str, language: str = "en") -> PIIResult:
    """
    Detect and mask PII in text using Presidio NER.

    Returns PIIResult with:
    - masked_text: original text with PII replaced by <ENTITY_TYPE>
    - entity_types: list of detected PII types (e.g. ["PERSON", "EMAIL_ADDRESS"])
    - original_hash: SHA-256 of original text (stored separately for forensics)
    - pii_detected: True if any PII was found

    Falls back to returning original text unchanged if Presidio not available.
    """
    original_hash = hash_original(text)

    analyzer, anonymizer = _get_engines()

    if analyzer is None or not text.strip():
        return PIIResult(
            masked_text=text,
            entity_types=[],
            original_hash=original_hash,
            pii_detected=False,
        )

    try:
        results = analyzer.analyze(text=text, language=language)
        entity_types = sorted(set(r.entity_type for r in results))

        if not results:
            return PIIResult(
                masked_text=text,
                entity_types=[],
                original_hash=original_hash,
                pii_detected=False,
            )

        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in entity_types
        }

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        return PIIResult(
            masked_text=anonymized.text,
            entity_types=entity_types,
            original_hash=original_hash,
            pii_detected=True,
        )

    except Exception as e:
        logger.error(f"PII analysis failed: {e}")
        return PIIResult(
            masked_text=text,
            entity_types=[],
            original_hash=original_hash,
            pii_detected=False,
        )


def mask_dict(obj: Any, language: str = "en") -> tuple[Any, list[str], str]:
    """
    Recursively mask PII in a dict/list/str.

    Returns:
        (masked_obj, all_entity_types_found, original_json_hash)
    """
    original_json = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    original_hash = hash_original(original_json)
    all_entities: set[str] = set()

    def _recurse(node: Any) -> Any:
        if isinstance(node, str):
            result = analyze_and_mask(node, language)
            all_entities.update(result.entity_types)
            return result.masked_text
        elif isinstance(node, dict):
            return {k: _recurse(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_recurse(item) for item in node]
        else:
            return node

    masked = _recurse(obj)
    return masked, sorted(all_entities), original_hash
