"""
PII Redaction & Tokenization Engine

Tokenizes or redacts PII in LLM API request messages before they reach the
provider, and optionally restores tokens in LLM responses.

Modes per PII type:
  tokenize  — replace with reversible token <PII-TYPE-N>, restore in response
  redact    — replace with [REDACTED-TYPE], one-way (for compliance logging)
  block     — reject the entire request (raises PIIBlockError)
  allow     — pass through unchanged

Supports:
  • Presidio Analyzer (presidio-analyzer) for standard PII detection —
    email, phone, SSN, credit card, IP, passport, bank account, IBAN, etc.
  • spaCy NER for PERSON / LOCATION / ORG entities
  • Custom per-org regex patterns (EMPLOYEE_ID, PROJECT_CODE, ACCOUNT_NUM, etc.)

Graceful degradation: if presidio_analyzer is not installed, falls back to
regex-only scanning (custom patterns + a small built-in set of common types).

Never raises outside of PIIBlockError; all other exceptions are caught and
logged with the original text returned unchanged.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default PII type modes (what happens to each type by default)
# ---------------------------------------------------------------------------

DEFAULT_PII_MODES: dict[str, str] = {
    "EMAIL_ADDRESS":       "tokenize",
    "PHONE_NUMBER":        "tokenize",
    "US_SSN":              "block",
    "CREDIT_CARD":         "block",
    "IP_ADDRESS":          "tokenize",
    "US_DRIVER_LICENSE":   "redact",
    "US_PASSPORT":         "redact",
    "US_BANK_NUMBER":      "redact",
    "IBAN_CODE":           "redact",
    "MEDICAL_LICENSE":     "block",
    "UK_NHS":              "block",
    "PERSON":              "allow",     # high FPR for names; org can tighten
    "LOCATION":            "allow",
    "DATE_TIME":           "allow",
    "URL":                 "allow",
    "NRP":                 "allow",
    "ORGANIZATION":        "allow",
    "CRYPTO":              "redact",
}

# Minimum Presidio confidence score to act on a finding
DEFAULT_MIN_SCORE: float = 0.7

# Fallback regexes used when Presidio is not available
_FALLBACK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL_ADDRESS",  re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("PHONE_NUMBER",   re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("US_SSN",         re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("CREDIT_CARD",    re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b")),
    ("IP_ADDRESS",     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CustomPattern:
    """A per-org regex pattern for company-specific sensitive data."""
    id: str           # stable UUID hex (for delete)
    name: str         # human label: "Employee ID", "Project Code"
    pattern: str      # raw regex string
    type_name: str    # used in token: <PII-EMPLOYEE_ID-1>
    mode: str         # tokenize | redact | block | allow
    _compiled: re.Pattern | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        try:
            self._compiled = re.compile(self.pattern)
        except re.error as exc:
            logger.warning("Custom pattern %r invalid regex: %s", self.name, exc)
            self._compiled = None

    @classmethod
    def from_dict(cls, d: dict) -> "CustomPattern":
        return cls(
            id=d.get("id", uuid.uuid4().hex),
            name=d["name"],
            pattern=d["pattern"],
            type_name=d.get("type_name", d["name"].upper().replace(" ", "_")),
            mode=d.get("mode", "redact"),
        )

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "name":      self.name,
            "pattern":   self.pattern,
            "type_name": self.type_name,
            "mode":      self.mode,
        }


@dataclass
class RedactionConfig:
    """Org-level PII redaction configuration."""
    enabled: bool = True
    pii_type_modes: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PII_MODES))
    custom_patterns: list[CustomPattern] = field(default_factory=list)
    min_score: float = DEFAULT_MIN_SCORE

    def get_mode(self, pii_type: str) -> str:
        return self.pii_type_modes.get(pii_type, "allow")

    @classmethod
    def default(cls) -> "RedactionConfig":
        return cls()


@dataclass
class PIIFinding:
    """A single PII item found in text."""
    start: int
    end: int
    pii_type: str
    mode: str
    score: float
    original: str    # the matched text

    def __lt__(self, other: "PIIFinding") -> bool:
        return self.start < other.start


class PIIBlockError(Exception):
    """Raised when a PII type with mode='block' is found in a message."""
    def __init__(self, pii_type: str, count: int):
        self.pii_type = pii_type
        self.count = count
        super().__init__(f"PII type {pii_type!r} with mode=block found ({count} instance(s))")


# ---------------------------------------------------------------------------
# PIIVault — per-session token store
# ---------------------------------------------------------------------------

class PIIVault:
    """
    Session-scoped store of token → real_value mappings.

    Tokens are of the form <PII-{TYPE}-{N}> (e.g. <PII-EMAIL_ADDRESS-1>).
    The same value always maps to the same token within a session (dedup).

    Not persisted — lives only in memory for the session duration.
    If the proxy restarts mid-session, tokens in buffered LLM responses
    cannot be restored (acceptable: sessions restart too).
    """

    def __init__(self) -> None:
        self._token_to_value: dict[str, str] = {}
        self._value_to_token: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self._restore_re: re.Pattern = re.compile(r"<PII-[A-Z_0-9]+-\d+>")

    def _make_token(self, pii_type: str) -> str:
        n = self._counters.get(pii_type, 0) + 1
        self._counters[pii_type] = n
        return f"<PII-{pii_type}-{n}>"

    def store(self, value: str, pii_type: str) -> str:
        """Return (and remember) the token for value, reusing if already seen."""
        if value in self._value_to_token:
            return self._value_to_token[value]
        token = self._make_token(pii_type)
        self._token_to_value[token] = value
        self._value_to_token[value] = token
        return token

    def restore(self, text: str) -> str:
        """Replace all <PII-*> tokens in text with their real values."""
        if not self._token_to_value:
            return text
        return self._restore_re.sub(
            lambda m: self._token_to_value.get(m.group(0), m.group(0)),
            text,
        )

    def size(self) -> int:
        return len(self._token_to_value)

    def summary(self) -> dict:
        """Compact stats dict for audit event JSONB storage."""
        by_type: dict[str, int] = {}
        for token in self._token_to_value:
            # <PII-EMAIL_ADDRESS-1> → "EMAIL_ADDRESS"
            parts = token[5:-1].rsplit("-", 1)   # strip "<PII-" and ">"
            pii_type = parts[0] if parts else "UNKNOWN"
            by_type[pii_type] = by_type.get(pii_type, 0) + 1
        return {"total_tokens": self.size(), "by_type": by_type}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _presidio_findings(text: str, config: RedactionConfig) -> list[PIIFinding]:
    """Run Presidio Analyzer and return PIIFinding list. Empty list if not available."""
    try:
        from presidio_analyzer import AnalyzerEngine
        _engine = _get_analyzer()
        results = _engine.analyze(text=text, language="en")
        findings: list[PIIFinding] = []
        for r in results:
            if r.score < config.min_score:
                continue
            mode = config.get_mode(r.entity_type)
            if mode == "allow":
                continue
            findings.append(PIIFinding(
                start=r.start,
                end=r.end,
                pii_type=r.entity_type,
                mode=mode,
                score=r.score,
                original=text[r.start:r.end],
            ))
        return findings
    except ImportError:
        return _fallback_findings(text, config)
    except Exception as exc:
        logger.debug("Presidio error (falling back to regex): %s", exc)
        return _fallback_findings(text, config)


def _fallback_findings(text: str, config: RedactionConfig) -> list[PIIFinding]:
    """Regex-based fallback when Presidio is unavailable."""
    findings: list[PIIFinding] = []
    for pii_type, pattern in _FALLBACK_PATTERNS:
        mode = config.get_mode(pii_type)
        if mode == "allow":
            continue
        for m in pattern.finditer(text):
            findings.append(PIIFinding(
                start=m.start(),
                end=m.end(),
                pii_type=pii_type,
                mode=mode,
                score=0.8,
                original=m.group(0),
            ))
    return findings


def _custom_findings(text: str, config: RedactionConfig) -> list[PIIFinding]:
    """Run custom org-defined patterns and return findings."""
    findings: list[PIIFinding] = []
    for cp in config.custom_patterns:
        if cp._compiled is None:
            continue
        if cp.mode == "allow":
            continue
        for m in cp._compiled.finditer(text):
            findings.append(PIIFinding(
                start=m.start(),
                end=m.end(),
                pii_type=cp.type_name,
                mode=cp.mode,
                score=1.0,   # custom patterns are exact — no uncertainty
                original=m.group(0),
            ))
    return findings


def _merge_findings(findings: list[PIIFinding]) -> list[PIIFinding]:
    """
    Remove overlapping findings, keeping the highest-score one per span.
    Returns findings sorted by start position (ascending).
    """
    if not findings:
        return findings

    # Sort by start, then by score descending to prefer higher-confidence finding
    sorted_f = sorted(findings, key=lambda f: (f.start, -f.score))
    merged: list[PIIFinding] = []
    last_end = -1

    for f in sorted_f:
        if f.start >= last_end:
            merged.append(f)
            last_end = f.end
        # else: overlaps with prior — skip (prior had higher score or came first)

    return merged


# Lazy singleton analyzer (expensive to construct)
_analyzer_instance = None


def _get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        from presidio_analyzer import AnalyzerEngine
        _analyzer_instance = AnalyzerEngine()
    return _analyzer_instance


# ---------------------------------------------------------------------------
# Core redaction function
# ---------------------------------------------------------------------------

def apply_redaction(
    text: str,
    vault: PIIVault,
    config: RedactionConfig,
) -> tuple[str, list[PIIFinding]]:
    """
    Scan text for PII and apply redaction/tokenization in-place.

    Args:
        text:   The raw text to scan and modify.
        vault:  Session PIIVault for token storage (mutated in-place).
        config: Org-level redaction configuration.

    Returns:
        (redacted_text, findings) — findings list used for audit logging.

    Raises:
        PIIBlockError: if any finding has mode='block'.
    """
    if not text or not config.enabled:
        return text, []

    try:
        all_findings = _merge_findings(
            _presidio_findings(text, config) + _custom_findings(text, config)
        )

        # Check for block-mode findings before making any changes
        for f in all_findings:
            if f.mode == "block":
                raise PIIBlockError(f.pii_type, sum(1 for x in all_findings if x.pii_type == f.pii_type))

        if not all_findings:
            return text, []

        # Apply replacements right-to-left to preserve offsets
        result = text
        for f in reversed(all_findings):
            if f.mode == "tokenize":
                replacement = vault.store(f.original, f.pii_type)
            elif f.mode == "redact":
                replacement = f"[REDACTED-{f.pii_type}]"
            else:
                continue
            result = result[:f.start] + replacement + result[f.end:]

        return result, all_findings

    except PIIBlockError:
        raise
    except Exception as exc:
        logger.warning("PII redaction error (returning original): %s", exc)
        return text, []


# ---------------------------------------------------------------------------
# Message-level redaction (walks OpenAI/Anthropic message arrays)
# ---------------------------------------------------------------------------

def redact_messages(
    messages: list[dict],
    vault: PIIVault,
    config: RedactionConfig,
) -> tuple[list[dict], dict]:
    """
    Apply PII redaction to all string content in a messages array.

    Handles:
    - Simple string content: {"role": "user", "content": "text"}
    - Multi-part content: {"role": "user", "content": [{"type": "text", "text": "..."}]}
    - Tool results inside content blocks

    Args:
        messages: LLM API messages array (modified in-place conceptually — a
                  deep copy is returned so the original is not mutated).
        vault:    Session PIIVault.
        config:   Redaction config.

    Returns:
        (redacted_messages, summary_dict) where summary has:
          total_findings, by_type, blocked (False always — PIIBlockError is raised)

    Raises:
        PIIBlockError: propagated from apply_redaction if mode=block found.
    """
    if not config.enabled:
        return messages, {"total_findings": 0, "by_type": {}, "blocked": False}

    import copy
    redacted = copy.deepcopy(messages)
    all_findings: list[PIIFinding] = []

    for msg in redacted:
        content = msg.get("content")
        if isinstance(content, str):
            new_content, findings = apply_redaction(content, vault, config)
            msg["content"] = new_content
            all_findings.extend(findings)

        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                # text block
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    new_text, findings = apply_redaction(part["text"], vault, config)
                    part["text"] = new_text
                    all_findings.extend(findings)
                # tool_result block
                if part.get("type") == "tool_result":
                    inner = part.get("content")
                    if isinstance(inner, str):
                        new_inner, findings = apply_redaction(inner, vault, config)
                        part["content"] = new_inner
                        all_findings.extend(findings)
                    elif isinstance(inner, list):
                        for inner_part in inner:
                            if isinstance(inner_part, dict) and inner_part.get("type") == "text":
                                new_t, findings = apply_redaction(inner_part.get("text", ""), vault, config)
                                inner_part["text"] = new_t
                                all_findings.extend(findings)

    # Summarise by type
    by_type: dict[str, int] = {}
    for f in all_findings:
        by_type[f.pii_type] = by_type.get(f.pii_type, 0) + 1

    return redacted, {
        "total_findings": len(all_findings),
        "by_type": by_type,
        "blocked": False,
    }


# ---------------------------------------------------------------------------
# Remote config fetch (proxy fetches from backend with 60s TTL cache)
# ---------------------------------------------------------------------------

_redaction_config_cache: dict[str, tuple[float, RedactionConfig]] = {}  # org_id → (ts, cfg)
_REDACTION_CONFIG_TTL = 60.0


async def get_redaction_config(
    org_id: str,
    backend_url: str,
    api_key: str,
) -> RedactionConfig:
    """
    Fetch org-level redaction config from backend with 60 s TTL cache.
    Falls back to default config on any error.
    """
    cached = _redaction_config_cache.get(org_id)
    if cached and (time.monotonic() - cached[0]) < _REDACTION_CONFIG_TTL:
        return cached[1]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{backend_url}/v1/redaction-config",
                headers={"X-API-Key": api_key},
            )
        if resp.status_code == 200:
            data = resp.json()
            cfg = _parse_redaction_config(data)
            _redaction_config_cache[org_id] = (time.monotonic(), cfg)
            return cfg
    except Exception as exc:
        logger.debug("Could not fetch redaction config (using defaults): %s", exc)

    # Cache the default so we don't hammer a failing backend
    default = RedactionConfig.default()
    _redaction_config_cache[org_id] = (time.monotonic(), default)
    return default


def _parse_redaction_config(data: dict) -> RedactionConfig:
    cfg = RedactionConfig()
    cfg.enabled = bool(data.get("enabled", True))
    cfg.min_score = float(data.get("min_score", DEFAULT_MIN_SCORE))

    modes = data.get("pii_type_modes")
    if isinstance(modes, dict):
        cfg.pii_type_modes = {**DEFAULT_PII_MODES, **modes}

    patterns_raw = data.get("custom_patterns") or []
    cfg.custom_patterns = []
    for p in patterns_raw:
        try:
            cfg.custom_patterns.append(CustomPattern.from_dict(p))
        except Exception:
            pass

    return cfg


def invalidate_redaction_config_cache(org_id: str) -> None:
    _redaction_config_cache.pop(org_id, None)
