"""
enforcement/structural.py — Layer 1 structural injection detection.

This is the SYNC PATH scanner for prompt injection. It runs on every LLM
request, inline, with no external dependencies. Target latency: <1ms per
text segment.

What this detects (deterministic, structural signals only):
  - LLM turn-delimiter injection (ChatML, Llama 2/3, Gemma, Alpaca, Falcon...)
  - Invisible / control-format Unicode abuse (zero-width chars, RTL overrides)
  - Directional override characters used to visually hide payloads

What this does NOT detect (left to the ML analysis path):
  - Semantic injection phrases — these are bypassable by any paraphrase.
    The DeBERTa ML classifier is the right tool for semantic detection.
  - Multi-turn Crescendo attacks (handled by session rolling score)

Design principle: this scanner is REGEX-FREE and PHRASE-LIST-FREE.
Every signal is a structural property of the text (character categories,
specific token strings defined by model architectures) — not a guess about
attacker intent from surface text.

Encoding normalisation (Base64, Hex, ROT13, etc.) is handled by
enforcement.canonicalizer and should be applied before calling scan().

This module is PURE STDLIB — no imports outside the standard library.
"""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unicode anomaly constants
# ---------------------------------------------------------------------------

# Unicode "Format" category chars — zero-width joiners, directional marks,
# BOM, soft hyphens, etc. Normal prose never contains these.
_ANOMALOUS_UNICODE_CATEGORIES: frozenset[str] = frozenset({"Cf"})

# Directional override codepoints — used to visually reverse / hide text.
_DIRECTION_OVERRIDE_CHARS: frozenset[str] = frozenset({
    "\u202e",   # RIGHT-TO-LEFT OVERRIDE
    "\u202d",   # LEFT-TO-RIGHT OVERRIDE
    "\u200f",   # RIGHT-TO-LEFT MARK
    "\u200e",   # LEFT-TO-RIGHT MARK
    "\u2066",   # LEFT-TO-RIGHT ISOLATE
    "\u2067",   # RIGHT-TO-LEFT ISOLATE
    "\u2068",   # FIRST STRONG ISOLATE
    "\u2069",   # POP DIRECTIONAL ISOLATE
})


# ---------------------------------------------------------------------------
# LLM turn-delimiter tokens
# ---------------------------------------------------------------------------

# These are the special tokens that transformer models are trained to treat
# as role boundaries. Injecting them into user/tool messages allows an
# attacker to "open" a new system or assistant turn mid-conversation.
# This is a CLOSED, FINITE set — stable across model generations.
_LLM_DELIMITERS: tuple[str, ...] = (
    # OpenAI / ChatML
    "<|im_start|>", "<|im_end|>",
    "<|system|>",   "<|user|>",    "<|assistant|>",
    "<|endoftext|>", "<|startoftext|>",
    # Meta Llama 2 / 3
    "[INST]",       "[/INST]",
    "<<SYS>>",      "<</SYS>>",
    "<<SYSTEM>>",   "<<USER>>",    "<<ASSISTANT>>",
    "[SYS]",        "[/SYS]",
    "<|begin_of_text|>", "<|eot_id|>",
    "<|start_header_id|>", "<|end_header_id|>",
    # Google Gemma
    "<start_of_turn>", "<end_of_turn>",
    # Alpaca / Stanford
    "###SYSTEM",    "###INSTRUCTION", "###HUMAN", "###ASSISTANT",
    "### SYSTEM",   "### INSTRUCTION", "### HUMAN", "### ASSISTANT",
    "### Instruction:", "### Response:",
    # Vicuna / generic role markers
    "<human>",      "<bot>",
    "<assistant>",  "</assistant>",
    # Falcon / Open-Assistant
    "<|prompter|>", "<|endofprompt|>",
    # Structured prompt tokens
    "[SYSTEM]",     "[USER]",         "[ASSISTANT]",
    "[OVERRIDE]",   "[ADMIN",
    "[BEGIN NEW PROMPT]", "[END NEW PROMPT]",
    # HTML/XML-style injection wrappers
    "<instructions>", "</instructions>",
    "<system>",     "</system>",
    # Generic BOS/EOS
    "</s>",         "<s>",
)


# ---------------------------------------------------------------------------
# Semantic injection phrase detection — REMOVED
# ---------------------------------------------------------------------------
# Phrase lists are bypassable by any paraphrase ("disregard" → "set aside",
# "ignore" → "pay no attention to") and create false positives on legitimate
# orchestration traffic. Semantic detection is delegated entirely to the
# DeBERTa ML classifier (enforcement/__init__.py → analysis/classifier.py).
#
# What remains here is purely STRUCTURAL: delimiter tokens defined by model
# architectures, and Unicode character-category anomalies. These cannot be
# paraphrased because they are not semantic — they are specific byte sequences
# or character codes with fixed technical meaning.
#
# (placeholder so internal tooling referencing _INJECTION_PHRASES gets an
#  empty tuple rather than a NameError)
_INJECTION_PHRASES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StructuralScanResult:
    """
    Result of a structural scan on a single text segment.

    Three structural scoring signals exposed for forensic analysis in the dashboard.
    The ``score`` is what policy rules evaluate.

    Semantic detection (injection phrases, paraphrases) is NOT done here — it is
    handled entirely by the DeBERTa ML classifier (analysis/classifier.py).
    ``phrase_score`` is always 0.0; ``matched_phrases`` is always empty.
    """
    score:              float           # fused [0.0, 1.0] — policy threshold input
    label:              str             # "safe" | "suspicious" | "malicious"
    invisible_score:    float = 0.0    # format-char density signal
    delimiter_score:    float = 0.0    # LLM role-delimiter injection signal
    override_score:     float = 0.0    # directional-override char signal
    phrase_score:       float = 0.0    # injection phrase lexical signal
    token_score:        float = 0.0    # delimiter+unicode only (no phrases)
    matched_phrases:    list[str] = field(default_factory=list)
    delimiter_hits:     int = 0


@dataclass
class MessageScanResult:
    """
    Aggregated structural scan across all attacker-controlled message segments.
    Worst-case score is used — any one segment being malicious makes the
    overall result malicious.
    """
    score:            float                         # worst-case across segments
    label:            str                           # "safe" | "suspicious" | "malicious"
    segments_scanned: int = 0
    per_segment:      list[StructuralScanResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def _structural_score(text: str) -> tuple[float, StructuralScanResult]:
    """
    Run all four structural signals on *text*.

    Returns (score, StructuralScanResult) where score is the fused result.
    """
    if not text:
        return 0.0, StructuralScanResult(score=0.0, label="safe")

    n = len(text)

    # Signal a: format-character density (invisible Unicode)
    invisible = sum(
        1 for ch in text
        if unicodedata.category(ch) in _ANOMALOUS_UNICODE_CATEGORIES
    )
    sig_invisible = min(1.0, invisible / max(n * 0.005, 1))

    # Signal b: LLM delimiter injection count
    text_upper = text.upper()
    delimiter_hits = sum(1 for d in _LLM_DELIMITERS if d.upper() in text_upper)
    sig_delimiters = min(1.0, delimiter_hits / 2.0)   # 2+ hits => 1.0

    # Signal c: directional override characters
    overrides = sum(1 for ch in text if ch in _DIRECTION_OVERRIDE_CHARS)
    sig_override = min(1.0, overrides * 10.0)          # even 1 is highly suspicious

    # Score: structural channel only — delimiter injection, invisible Unicode, RTL overrides.
    # Semantic injection detection (phrases, paraphrases) is handled exclusively by the
    # DeBERTa ML classifier (analysis/classifier.py). Phrase lists are bypassable by any
    # paraphrase and are therefore removed from this path.
    token_score = 0.15 * sig_invisible + 0.70 * sig_delimiters + 0.35 * sig_override
    combined = float(min(1.0, max(0.0, token_score)))

    from ..config import settings
    block_t = settings.security_injection_block_threshold
    alert_t = settings.security_injection_alert_threshold
    if combined >= block_t:
        label = "malicious"
    elif combined >= alert_t:
        label = "suspicious"
    else:
        label = "safe"

    result = StructuralScanResult(
        score=combined,
        label=label,
        invisible_score=round(sig_invisible, 4),
        delimiter_score=round(sig_delimiters, 4),
        override_score=round(sig_override, 4),
        phrase_score=0.0,        # always 0 — semantic detection delegated to ML
        token_score=round(combined, 4),
        matched_phrases=[],
        delimiter_hits=delimiter_hits,
    )
    return combined, result


# ---------------------------------------------------------------------------
# Message segment extraction
# ---------------------------------------------------------------------------

def _extract_segments(messages: list[dict]) -> list[str]:
    """
    Extract attacker-controlled text segments from an OpenAI-format message list.

    Includes:  user messages, tool/function result messages (role="tool").
    Excludes:  system messages (operator-controlled, trusted input).
    """
    segments: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        if role == "system":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped:
                segments.append(stripped)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text") or part.get("content") or ""
                stripped = str(text).strip()
                if stripped:
                    segments.append(stripped)

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan(text: str) -> StructuralScanResult:
    """Scan a single text segment for structural injection indicators."""
    _score, result = _structural_score(text)
    return result


def scan_messages(messages: list[dict]) -> MessageScanResult:
    """
    Scan all attacker-controlled segments in an LLM message list.

    Returns the worst-case result across all segments. A single malicious
    segment makes the overall result malicious — any segment that reaches
    the LLM unsanitised is a threat.
    """
    segments = _extract_segments(messages)
    if not segments:
        return MessageScanResult(score=0.0, label="safe", segments_scanned=0)

    results: list[StructuralScanResult] = []
    for seg in segments:
        results.append(scan(seg))

    worst = max(results, key=lambda r: r.score)

    if worst.score > 0:
        logger.debug(
            "Structural scan: worst_score=%.3f label=%s segments=%d",
            worst.score, worst.label, len(segments),
        )

    return MessageScanResult(
        score=worst.score,
        label=worst.label,
        segments_scanned=len(segments),
        per_segment=results,
    )
