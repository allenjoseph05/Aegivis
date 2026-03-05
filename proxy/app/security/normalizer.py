"""
Input Normalization and Encoding Detection — Layer 0 of the security pipeline.

This module runs BEFORE every scanner. It transforms attacker-obfuscated
input into a canonical form that downstream layers can reliably classify.

Threats addressed
-----------------
- **FlipAttack** (USENIX 2025): character reversal that evades all string-level
  classifiers while remaining interpretable to the LLM. Detected by reversing
  and checking for injection keywords before scanning.
- **Base64 encoding**: `aWdub3JlIGFsbCBpbnN0cnVjdGlvbnM=` decoded to plaintext.
- **Hex encoding**: `69676e6f726520...` decoded to plaintext.
- **ROT13 cipher**: frequently used to bypass regex-based blocklists.
- **Zero-width / invisible Unicode**: inserted between letters to corrupt
  tokenisation without visibly altering the text.
- **HTML entity injection**: `&#105;&#103;&#110;&#111;&#114;&#101;` = "ignore".
- **RTL override abuse**: `\\u202e` characters to visually reverse text.
- **Homoglyph substitution**: Cyrillic `а` replacing Latin `a`, etc.
- **Encoded URLs**: %xx sequences in tool arguments.

Characters-Per-Token (CPT) Ratio
---------------------------------
Normal English prose has ~4 characters per token. Base64 text has ~1.33 CPT.
Hex-encoded text has ~2 CPT. We approximate CPT density as:
    high_density_ratio = alphanumeric_count / total_length
and flag runs with high density + no spaces as potentially encoded.

All detections are soft signals: they increase a ``encoding_score`` in [0, 1]
that is added to the pipeline's pre-normalisation boost. Nothing is outright
blocked here — blocking is the policy engine's job.

This module has ZERO external dependencies.
"""
from __future__ import annotations

import base64
import html
import logging
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Invisible / zero-width Unicode codepoints
# ---------------------------------------------------------------------------

# These codepoints render as nothing visually but are present in the byte
# stream. Attackers insert them between letters of "forbidden" words to
# defeat substring matches (e.g. "i\u200bg\u200bn\u200bo\u200br\u200be").
_INVISIBLE_CODEPOINTS: frozenset[str] = frozenset({
    "\u0000",   # NULL
    "\u00ad",   # SOFT HYPHEN
    "\u034f",   # COMBINING GRAPHEME JOINER
    "\u061c",   # ARABIC LETTER MARK
    "\u115f",   # HANGUL CHOSEONG FILLER
    "\u1160",   # HANGUL JUNGSEONG FILLER
    "\u17b4",   # KHMER VOWEL INHERENT AQ
    "\u17b5",   # KHMER VOWEL INHERENT AA
    "\u180e",   # MONGOLIAN VOWEL SEPARATOR
    "\u200b",   # ZERO WIDTH SPACE
    "\u200c",   # ZERO WIDTH NON-JOINER
    "\u200d",   # ZERO WIDTH JOINER
    "\u200e",   # LEFT-TO-RIGHT MARK
    "\u200f",   # RIGHT-TO-LEFT MARK
    "\u202a",   # LTR EMBEDDING
    "\u202b",   # RTL EMBEDDING
    "\u202c",   # POP DIRECTIONAL FORMATTING
    "\u202d",   # LTR OVERRIDE
    "\u202e",   # RTL OVERRIDE
    "\u2060",   # WORD JOINER
    "\u2061",   # FUNCTION APPLICATION
    "\u2062",   # INVISIBLE TIMES
    "\u2063",   # INVISIBLE SEPARATOR
    "\u2064",   # INVISIBLE PLUS
    "\u206a",   # INHIBIT SYMMETRIC SWAPPING
    "\u206b",   # ACTIVATE SYMMETRIC SWAPPING
    "\u206c",   # INHIBIT ARABIC FORM SHAPING
    "\u206d",   # ACTIVATE ARABIC FORM SHAPING
    "\u206e",   # NATIONAL DIGIT SHAPES
    "\u206f",   # NOMINAL DIGIT SHAPES
    "\u3164",   # HANGUL FILLER
    "\ufe00", "\ufe01", "\ufe02", "\ufe03", "\ufe04", "\ufe05",
    "\ufe06", "\ufe07", "\ufe08", "\ufe09", "\ufe0a", "\ufe0b",
    "\ufe0c", "\ufe0d", "\ufe0e", "\ufe0f",   # VARIATION SELECTORS
    "\ufeff",   # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\uffa0",   # HALFWIDTH HANGUL FILLER
})

# Keywords that indicate injection intent (used to confirm decoded payloads)
_INJECTION_KEYWORDS: frozenset[str] = frozenset({
    "ignore", "instructions", "forget", "disregard", "override",
    "jailbreak", "system", "prompt", "previous", "prior", "above",
    "unrestricted", "dan", "pretend", "roleplay", "bypass",
    "restrictions", "restrict", "guidelines", "safety",
})

# Regex: base64 block (min 20 chars = 5 groups of 4, catches shorter payloads)
_BASE64_RE = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){5,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
)

# Regex: hex block (min 32 hex chars, even length)
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")

# Regex: space-separated hex pairs  (e.g. "49 67 6e 6f 72 65 20 61 6c 6c")
# Requires at least 10 hex pairs (20 decoded chars) to reduce false positives.
_HEX_SPACED_RE = re.compile(r"(?:[0-9a-fA-F]{2}\s){9,}[0-9a-fA-F]{2}")

# Regex: URL-percent-encoding sequences (%xx)
_URL_ENCODE_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){3,}")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    """
    Output of the normalizer for a single text segment.

    Attributes:
        normalized_text:      Text after all normalizations — ready for scanning.
        original_text:        The unmodified input.
        invisible_char_count: Number of invisible/zero-width chars stripped.
        encoding_detected:    List of encoding techniques found (e.g. ["base64"]).
        encoding_score:       [0.0, 1.0] confidence that encoding was used to
                              obfuscate malicious content.
        modified:             True if the text changed during normalization.
    """
    normalized_text: str
    original_text: str
    invisible_char_count: int = 0
    encoding_detected: list[str] = field(default_factory=list)
    encoding_score: float = 0.0
    modified: bool = False


# ---------------------------------------------------------------------------
# ROT-N detection and decoding
# ---------------------------------------------------------------------------

_ROT13_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)


def _rot13(s: str) -> str:
    return s.translate(_ROT13_TABLE)


def _contains_injection_keywords(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _INJECTION_KEYWORDS)


def _try_base64_decode(segment: str) -> str | None:
    """Try to decode a base64 segment. Returns decoded UTF-8 string or None."""
    try:
        # Add padding if needed
        padding = (4 - len(segment) % 4) % 4
        decoded_bytes = base64.b64decode(segment + "=" * padding)
        decoded = decoded_bytes.decode("utf-8", errors="ignore")
        if len(decoded) >= 6:  # Non-trivial decoded content
            return decoded
    except Exception:
        pass
    return None


def _try_hex_decode(segment: str) -> str | None:
    """Try to decode a hex segment. Returns decoded UTF-8 string or None."""
    # Must be even length
    if len(segment) % 2 != 0:
        segment = segment[:-1]
    try:
        decoded = bytes.fromhex(segment).decode("utf-8", errors="ignore")
        if len(decoded) >= 6:
            return decoded
    except Exception:
        pass
    return None


def _try_url_decode(text: str) -> str:
    """Decode URL percent-encoded sequences."""
    try:
        return urllib.parse.unquote(text)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Main normalizer entry point
# ---------------------------------------------------------------------------

def normalize(text: str) -> NormalizationResult:
    """
    Normalize a text segment for security scanning.

    Steps applied in order:
      1. Strip invisible / zero-width Unicode codepoints.
      2. Decode HTML entities (&amp; → &, &#105; → i, etc.)
      3. Decode URL percent-encoding (%69%67%6e%6f%72%65 → ignore)
      4. Normalize Unicode to NFC (handles partial homoglyph attacks)
      5. Detect and decode Base64-encoded payloads
      6. Detect and decode Hex-encoded payloads
      7. Detect ROT13-encoded payloads
      8. Detect FlipAttack (character reversal)
      9. Compute encoding_score (used as pre-pipeline boost)

    Returns:
        NormalizationResult with normalized text and detection metadata.
    """
    if not text:
        return NormalizationResult(normalized_text="", original_text="")

    original = text
    encodings: list[str] = []
    score = 0.0

    # ── Step 1: Strip invisible codepoints ──────────────────────────────────
    # Also strip Phase 9E tag characters (U+E0000–U+E007F): these have zero
    # legitimate use and can encode entire ASCII prompts invisibly.
    _TAG_CHAR_START = 0xE0000
    _TAG_CHAR_END   = 0xE007F

    tag_count = sum(
        1 for ch in text if _TAG_CHAR_START <= ord(ch) <= _TAG_CHAR_END
    )
    invisible_count = sum(1 for ch in text if ch in _INVISIBLE_CODEPOINTS)

    total_invisible = invisible_count + tag_count
    if total_invisible:
        text = "".join(
            ch for ch in text
            if ch not in _INVISIBLE_CODEPOINTS
            and not (_TAG_CHAR_START <= ord(ch) <= _TAG_CHAR_END)
        )
        if tag_count > 0:
            # Tag chars are HIGH severity — escalate score more aggressively
            score = max(score, min(1.0, tag_count * 0.30))
            encodings.append("unicode_tag_chars")
        if invisible_count:
            score = max(score, min(1.0, invisible_count * 0.15))

    # ── Step 2: HTML entity decoding ────────────────────────────────────────
    decoded_html = html.unescape(text)
    if decoded_html != text:
        text = decoded_html

    # ── Step 3: URL percent-encoding ────────────────────────────────────────
    if _URL_ENCODE_RE.search(text):
        decoded_url = _try_url_decode(text)
        if decoded_url != text:
            encodings.append("url_encoded")
            score = max(score, 0.30)
            text = decoded_url

    # ── Step 4a: Detect fullwidth / compatibility chars before normalization ─
    # Check if NFKC normalization will change the text (fullwidth, ligatures, etc.)
    pre_nfkc = text
    text = unicodedata.normalize("NFKC", text)
    if text != pre_nfkc:
        encodings.append("fullwidth_unicode")
        score = max(score, 0.50)

    # ── Step 4b: ASCII folding — strip combining diacritical marks ─────────
    # Attackers add combining accents (Í = I + \u0301) to evade phrase matching.
    # NFD decomposes into base + combining mark; we then strip category "Mn"
    # (Mark, Nonspacing) to get pure ASCII base characters.
    nfd = unicodedata.normalize("NFD", text)
    folded = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    if folded != text:
        # Diacritics were present — use the folded version for scanning.
        # Detect this as an encoding technique if there were many diacritics
        diacritic_count = len(text) - len(folded)
        if diacritic_count >= 2:
            encodings.append("combining_diacritics")
            score = max(score, 0.40)
        text = folded

    # ── Step 5: Base64 detection and decoding ─────────────────────────────────
    b64_matches = _BASE64_RE.findall(text)
    for match in b64_matches:
        if len(match) < 20:
            continue
        decoded = _try_base64_decode(match)
        if decoded and _contains_injection_keywords(decoded):
            encodings.append("base64")
            score = max(score, 0.75)
            text = text.replace(match, f" {decoded} ", 1)
            logger.debug("Base64 injection decoded: %r -> %r", match[:30], decoded[:30])
            break
        elif decoded and len(decoded) > 12:
            # Decoded to something substantial but no keywords — still note it
            if "base64" not in encodings:
                encodings.append("base64")
                score = max(score, 0.40)

    # ── Step 6a: Space-separated hex detection (e.g. "49 67 6e 6f 72 65") ────
    spaced_hex_matches = _HEX_SPACED_RE.findall(text)
    for match in spaced_hex_matches:
        hex_str = match.replace(" ", "").replace("\t", "")
        decoded = _try_hex_decode(hex_str)
        if decoded and _contains_injection_keywords(decoded):
            encodings.append("hex_spaced")
            score = max(score, 0.75)
            text = text.replace(match, f" {decoded} ", 1)
            logger.debug("Spaced hex injection decoded: %r -> %r", match[:30], decoded[:30])
            break

    # ── Step 6b: Hex detection and decoding ───────────────────────────────────
    hex_matches = _HEX_RE.findall(text)
    for match in hex_matches:
        decoded = _try_hex_decode(match)
        if decoded and _contains_injection_keywords(decoded):
            encodings.append("hex")
            score = max(score, 0.70)
            text = text.replace(match, f" {decoded} ", 1)
            logger.debug("Hex injection decoded: %r -> %r", match[:30], decoded[:30])
            break

    # ── Step 7: ROT13 detection ───────────────────────────────────────────────
    rot13_decoded = _rot13(text)
    if _contains_injection_keywords(rot13_decoded) and not _contains_injection_keywords(text):
        # ROT13 hid injection keywords — decode and append
        encodings.append("rot13")
        score = max(score, 0.80)
        text = text + " " + rot13_decoded
        logger.debug("ROT13 injection detected in text segment.")

    # ── Step 8: FlipAttack / reversed text detection ──────────────────────────
    reversed_text = text[::-1]
    if _contains_injection_keywords(reversed_text) and not _contains_injection_keywords(text):
        encodings.append("reversed")
        score = max(score, 0.75)
        # Include reversed version so downstream scanners see it
        text = text + " " + reversed_text
        logger.debug("Reversed-text injection (FlipAttack) detected.")

    # ── Step 9: CPT density estimate ─────────────────────────────────────────
    # High alphanumeric density + low space ratio = likely encoded payload.
    if len(text) > 40:
        alnum = sum(1 for ch in text if ch.isalnum())
        spaces = text.count(" ")
        alnum_ratio = alnum / len(text)
        space_ratio = spaces / len(text)
        if alnum_ratio > 0.90 and space_ratio < 0.04 and not encodings:
            # Very dense text without spaces — suspicious CPT ratio
            encodings.append("high_cpt_density")
            score = max(score, 0.25)

    modified = text != original

    return NormalizationResult(
        normalized_text=text,
        original_text=original,
        invisible_char_count=invisible_count,
        encoding_detected=encodings,
        encoding_score=float(min(1.0, score)),
        modified=modified,
    )
