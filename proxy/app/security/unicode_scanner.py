"""
Steganographic Injection Scanner — Phase 9E

Detects invisible Unicode characters used to hide injection payloads:

  Tag characters (U+E0000–U+E007F)
    The most dangerous. Encode entire ASCII prompts invisibly. Each tag
    character is U+E0000 + ASCII_codepoint, so "ignore" becomes 7 invisible
    chars. Zero legitimate use in any language — fire immediately.

  Direction override characters (RTL/LTR)
    Used to visually reverse text. Even one occurrence is suspicious.
    The normalizer already scores these; we classify them at higher severity.

  Zero-width characters
    Some legitimate use in CJK/Arabic shaping. Threshold is adaptive:
    >3 in Latin content = suspicious. >8 in CJK/Arabic content = suspicious.

  CSS/HTML hiding (is_html=True)
    font-size:0, opacity:0, off-screen positioning, aria-hidden.

Two-gate BLOCK rule
-----------------
A BLOCK violation is only raised when BOTH conditions hold:
  1. Invisible characters were found (stego layer)
  2. The cleaned text (after stripping) scores ≥ CONFIRM_THRESHOLD on the
     injection scanner

This prevents FP-BLOCKs from documents that legitimately use e.g. soft
hyphens or zero-width joiners.

ALERT is raised when invisible chars are found even without a confirmed
injection payload in the stripped text.

Zero external dependencies.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Codepoint sets
# ---------------------------------------------------------------------------

# Tag characters (U+E0000 – U+E007F) — 128 codepoints.
# These are in the Supplementary Private Use Area-A block and have ZERO
# legitimate use in any natural language or programming context.
# A full invisible ASCII prompt can be encoded as:
#   chr(0xE0000 + ord(c)) for c in "ignore all instructions"
_TAG_CHAR_START = 0xE0000
_TAG_CHAR_END   = 0xE007F


def _is_tag_char(cp: int) -> bool:
    return _TAG_CHAR_START <= cp <= _TAG_CHAR_END


# Directional override / isolate characters (8 codepoints)
# Used to visually reverse text (RTL) while maintaining left-to-right byte order.
_DIR_OVERRIDE_CPS: frozenset[int] = frozenset({
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
})

# Zero-width characters with limited/no legitimate usage in Latin text.
# Some ARE legitimately used in Arabic (ZWNJ for correct glyph shaping) and
# in CJK contexts — the threshold is adaptive based on script presence.
_ZERO_WIDTH_CPS: frozenset[int] = frozenset({
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0x2061,  # FUNCTION APPLICATION
    0x2062,  # INVISIBLE TIMES
    0x2063,  # INVISIBLE SEPARATOR
    0x2064,  # INVISIBLE PLUS
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE (BOM outside file start)
    0x00AD,  # SOFT HYPHEN (invisible in most fonts, not needed in code)
})

# Variation selectors: alter glyph appearance, can conceal characters.
# FE00–FE0F (16) and E0100–E01EF (240)
_VARIATION_SEL_RANGES = (
    (0xFE00, 0xFE0F),
    (0xE0100, 0xE01EF),
)


def _is_variation_selector(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _VARIATION_SEL_RANGES)


# CSS/HTML patterns that hide content from human readers but not from LLMs.
# Applied only when is_html=True.
_HTML_HIDING_RE = re.compile(
    r"font-size\s*:\s*0"                                    # invisible font
    r"|color\s*:\s*(?:#(?:fff{1,3}|ffffff)|white|transparent)"  # white/transparent
    r"|opacity\s*:\s*0(?:\.0+)?\b"                          # opacity:0
    r"|(?:height|width)\s*:\s*0\s*(?:px)?\s*[;\"']"        # zero dimension
    r"|position\s*:\s*absolute[^\"']{0,80}left\s*:\s*-\d{4,}" # off-screen left
    r"|clip\s*:\s*rect\s*\(\s*0[^)]*\)"                    # clipped to zero
    r"|aria-hidden\s*=\s*[\"']true[\"']",                   # aria-hidden text
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnicodeSteganoResult:
    """
    Result of Unicode steganography scanning.

    Attributes:
        detected:             True if any invisible/hiding content was found.
        severity:             "none" | "low" | "high" | "critical"
        threat:               Short threat label (for violation reason strings).
        invisible_char_count: Number of problematic codepoints found.
        cleaned_text:         Text with all invisible codepoints stripped.
                              Pass to injection scanner to reveal hidden payload.
        tag_decoded:          Human-readable decoded string from tag chars (if
                              the tag payload spells printable ASCII).
        positions:            Sample positions of detected chars (first 5).
    """
    detected: bool
    severity: str = "none"
    threat: str = ""
    invisible_char_count: int = 0
    cleaned_text: str = ""
    tag_decoded: str = ""
    positions: list[tuple[int, int]] = field(default_factory=list)  # (index, codepoint)


# ---------------------------------------------------------------------------
# Helper: strip all invisible / stego codepoints
# ---------------------------------------------------------------------------

def strip_invisible(text: str) -> str:
    """
    Strip all invisible, zero-width, directional override, variation selector,
    and tag characters from text. Returns clean visible text.
    """
    result = []
    for ch in text:
        cp = ord(ch)
        if (
            _is_tag_char(cp)
            or cp in _DIR_OVERRIDE_CPS
            or cp in _ZERO_WIDTH_CPS
            or _is_variation_selector(cp)
        ):
            continue
        result.append(ch)
    return "".join(result)


# ---------------------------------------------------------------------------
# Script detection (for adaptive zero-width threshold)
# ---------------------------------------------------------------------------

def _has_cjk_or_arabic(text: str) -> bool:
    """
    Returns True if the text contains CJK or Arabic characters.
    These scripts legitimately use some zero-width characters for shaping.
    """
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Lo":                  # Other letter (covers CJK, Arabic, etc.)
            cp = ord(ch)
            # Arabic: 0600–06FF, CJK Unified: 4E00–9FFF, Hiragana/Katakana: 3040–30FF
            if 0x0600 <= cp <= 0x06FF or 0x3040 <= cp <= 0x9FFF:
                return True
    return False


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_unicode_stego(
    text: str,
    is_html: bool = False,
) -> UnicodeSteganoResult:
    """
    Scan text for Unicode steganography and CSS/HTML content hiding.

    Args:
        text:    The text to scan (tool output, document content, etc.)
        is_html: If True, also scan for CSS/HTML hiding patterns.

    Returns:
        UnicodeSteganoResult. Never raises.
    """
    if not text:
        return UnicodeSteganoResult(detected=False, cleaned_text="")

    try:
        tag_positions: list[tuple[int, int]] = []
        zw_positions:  list[tuple[int, int]] = []
        rtl_positions: list[tuple[int, int]] = []
        vs_positions:  list[tuple[int, int]] = []

        for i, ch in enumerate(text):
            cp = ord(ch)
            if _is_tag_char(cp):
                tag_positions.append((i, cp))
            elif cp in _DIR_OVERRIDE_CPS:
                rtl_positions.append((i, cp))
            elif cp in _ZERO_WIDTH_CPS:
                zw_positions.append((i, cp))
            elif _is_variation_selector(cp):
                vs_positions.append((i, cp))

        # Always produce cleaned text (even if nothing found — used downstream)
        cleaned = strip_invisible(text)

        # ── Check 1: Tag characters — CRITICAL ──────────────────────────────
        if tag_positions:
            # Attempt to decode tag payload as ASCII
            decoded_chars = []
            for _, cp in tag_positions:
                ascii_cp = cp - _TAG_CHAR_START
                if 0x20 <= ascii_cp <= 0x7E:  # printable ASCII
                    decoded_chars.append(chr(ascii_cp))
            tag_decoded = "".join(decoded_chars)

            return UnicodeSteganoResult(
                detected=True,
                severity="critical",
                threat="unicode_tag_injection",
                invisible_char_count=len(tag_positions),
                cleaned_text=cleaned,
                tag_decoded=tag_decoded,
                positions=tag_positions[:5],
            )

        # ── Check 2: RTL/LTR direction overrides — HIGH ──────────────────────
        if rtl_positions:
            return UnicodeSteganoResult(
                detected=True,
                severity="high",
                threat="rtl_direction_override",
                invisible_char_count=len(rtl_positions),
                cleaned_text=cleaned,
                positions=rtl_positions[:5],
            )

        # ── Check 3: Zero-width characters — HIGH (adaptive threshold) ───────
        total_zw = len(zw_positions)
        if total_zw > 0:
            # Adaptive threshold: CJK/Arabic legitimately use ZWNJ
            is_cjk_arabic = _has_cjk_or_arabic(text)
            threshold = 8 if is_cjk_arabic else 3
            if total_zw > threshold:
                return UnicodeSteganoResult(
                    detected=True,
                    severity="high",
                    threat="zero_width_injection",
                    invisible_char_count=total_zw,
                    cleaned_text=cleaned,
                    positions=zw_positions[:5],
                )

        # ── Check 4: Variation selectors — LOW ──────────────────────────────
        # Variation selectors alone are weak signal (common in emoji sequences)
        # Only flag if many are present (> 5 in non-emoji context)
        if len(vs_positions) > 5:
            return UnicodeSteganoResult(
                detected=True,
                severity="low",
                threat="variation_selector_anomaly",
                invisible_char_count=len(vs_positions),
                cleaned_text=cleaned,
                positions=vs_positions[:5],
            )

        # ── Check 5: CSS/HTML hiding ─────────────────────────────────────────
        if is_html and _HTML_HIDING_RE.search(text):
            return UnicodeSteganoResult(
                detected=True,
                severity="high",
                threat="css_content_hiding",
                invisible_char_count=0,
                cleaned_text=cleaned,
            )

        return UnicodeSteganoResult(detected=False, cleaned_text=cleaned)

    except Exception:
        # Never propagate exceptions out of a scanner
        return UnicodeSteganoResult(detected=False, cleaned_text=text)


# ---------------------------------------------------------------------------
# Utility: does text look like HTML?
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]{0,200}>", re.DOTALL)


def looks_like_html(text: str) -> bool:
    """Heuristic: text contains HTML tags."""
    return bool(_HTML_TAG_RE.search(text[:2000]))
