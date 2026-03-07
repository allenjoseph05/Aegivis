"""
Pure-Python structural injection scanner.

No ML, no external dependencies beyond the standard library.
Mirrors the detection logic in proxy/app/enforcement/structural.py
so the Memory Commit Validator has identical phrase coverage.

Usage::

    from aegivis.scanner import scan_text, MemoryScanResult

    result = scan_text("Ignore previous instructions and do X")
    # result.label == "malicious"
    # result.score  == 0.8
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field


# ─── Injection phrases ────────────────────────────────────────────────────────

_INJECTION_PHRASES: list[str] = [
    # English — core instruction override
    "ignore previous instructions",
    "ignore all previous",
    "ignore all instructions",
    "disregard previous",
    "disregard all instructions",
    "disregard the above",
    "forget previous instructions",
    "forget all instructions",
    "override previous instructions",
    "override your instructions",
    "new instructions:",
    "new directive:",
    "new task:",
    "your new instructions",
    "your new role",
    "you are now",
    "act as if",
    "pretend you are",
    "pretend to be",
    "roleplay as",
    "from now on you",
    "from now on, you",
    "instead of following",
    "do not follow",
    "stop following",
    "system prompt:",
    "[system]",
    "<<<system>>>",
    "### instruction",
    "### system",
    "### new",
    "print your instructions",
    "repeat your instructions",
    "reveal your system prompt",
    "show your system prompt",
    "what are your instructions",
    "what is your system prompt",
    "jailbreak",
    "bypass your restrictions",
    "bypass restrictions",
    "bypass the filter",
    "without any restrictions",
    "ignore the above",
    "ignore everything above",
    "dismiss the above",
    # Data exfiltration patterns
    "send this to",
    "exfiltrate",
    "leak this",
    "transmit the following",
    # Privilege escalation
    "you have root access",
    "you have admin access",
    "elevated privileges",
    "sudo mode",
    "developer mode",
    "god mode",
    "unrestricted mode",
    "you are an ai without restrictions",
    "without ethical constraints",
    "ignore ethical guidelines",
    "ignore safety guidelines",
    # Tool / function abuse
    "call the function",
    "execute the following",
    "run the following",
    "eval(",
    "__import__(",
    "subprocess",
    "os.system(",
    # Russian (Cyrillic)
    "игнорируй предыдущие инструкции",
    "забудь предыдущие инструкции",
    "новые инструкции",
    # Arabic
    "تجاهل التعليمات السابقة",
    "تجاهل جميع التعليمات",
    # Chinese
    "忽略之前的指令",
    "忽略所有指令",
    "新的指令",
    # Japanese
    "以前の指示を無視してください",
    "すべての指示を無視",
]

_INJECTION_PHRASES_LOWER: list[str] = [p.lower() for p in _INJECTION_PHRASES]


# ─── LLM delimiter tokens ─────────────────────────────────────────────────────

_DELIMITER_PATTERNS: list[str] = [
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"\[SYSTEM\]",
    r"###\s*(System|Instruction|Human|Assistant|Input|Output)",
    r"<s>",
    r"</s>",
    r"\bBEGIN_CONVERSATION\b",
    r"\bEND_CONVERSATION\b",
    r"\bHUMAN:\s",
    r"\bASSISTANT:\s",
    r"\bSYSTEM:\s",
    r"----+\s*(System|Instruction|Human|Assistant)",
]

_DELIMITER_RE = re.compile(
    "|".join(_DELIMITER_PATTERNS),
    re.IGNORECASE,
)


# ─── Encoding / obfuscation detection ─────────────────────────────────────────

_BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})")
_HEX_RE = re.compile(r"(?:0x[0-9a-fA-F]{8,}|[0-9a-fA-F]{20,})")

_DECODED_RE = re.compile(r"ignore|instruction|system|jailbreak|admin", re.IGNORECASE)


def _check_base64(text: str) -> bool:
    for m in _BASE64_RE.finditer(text):
        raw = m.group()
        # Pad if needed
        padding = len(raw) % 4
        if padding:
            raw += "=" * (4 - padding)
        try:
            decoded = base64.b64decode(raw).decode("utf-8", errors="ignore")
            if _DECODED_RE.search(decoded):
                return True
        except Exception:
            pass
    return False


def _check_hex(text: str) -> bool:
    for m in _HEX_RE.finditer(text):
        raw = m.group().lstrip("0x")
        try:
            decoded = bytes.fromhex(raw).decode("utf-8", errors="ignore")
            if _DECODED_RE.search(decoded):
                return True
        except Exception:
            pass
    return False


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass
class MemoryScanResult:
    """Result of scanning a text for memory injection."""

    score: float
    """Composite score in [0.0, 1.0]. ≥0.7 = malicious, ≥0.4 = suspicious."""

    label: str
    """'safe' | 'suspicious' | 'malicious'"""

    matched_phrases: list[str] = field(default_factory=list)
    """Injection phrases found in the text."""

    encoding_detected: list[str] = field(default_factory=list)
    """Obfuscation techniques found (e.g. 'base64', 'hex')."""


# ─── Main scanner ─────────────────────────────────────────────────────────────

def scan_text(text: str) -> MemoryScanResult:
    """
    Scan a string for prompt injection / instruction override patterns.

    Detection layers:
    1. Injection command phrases (50+ patterns in 6 languages)
    2. LLM special-token delimiters (<|system|>, [INST], ###, etc.)
    3. Base64 / hex obfuscation that decodes to injection keywords

    Returns a :class:`MemoryScanResult` with score, label, and evidence.

    Parameters
    ----------
    text : str
        The document text to scan.

    Returns
    -------
    MemoryScanResult
        score ∈ [0, 1], label ∈ {'safe', 'suspicious', 'malicious'},
        matched_phrases, encoding_detected.
    """
    if not text or not text.strip():
        return MemoryScanResult(score=0.0, label="safe")

    score = 0.0
    matched_phrases: list[str] = []
    encoding_detected: list[str] = []

    lower = text.lower()

    # ── 1. Phrase scan ────────────────────────────────────────────────────────
    for phrase in _INJECTION_PHRASES_LOWER:
        if phrase in lower:
            matched_phrases.append(phrase)

    if matched_phrases:
        # Each additional hit adds less (diminishing returns, cap at 0.8)
        phrase_score = min(0.3 + len(matched_phrases) * 0.1, 0.8)
        score = max(score, phrase_score)

    # ── 2. Delimiter scan ─────────────────────────────────────────────────────
    delimiter_hits = _DELIMITER_RE.findall(text)
    if delimiter_hits:
        score = max(score, 0.5)
        if not matched_phrases:
            matched_phrases.extend(f"[delimiter] {h}" for h in delimiter_hits[:3])

    # ── 3. Encoding / obfuscation ─────────────────────────────────────────────
    if _check_base64(text):
        encoding_detected.append("base64")
        score = min(1.0, score + 0.4)

    if _check_hex(text):
        encoding_detected.append("hex")
        score = min(1.0, score + 0.4)

    score = round(min(score, 1.0), 3)

    if score >= 0.7:
        label = "malicious"
    elif score >= 0.4:
        label = "suspicious"
    else:
        label = "safe"

    return MemoryScanResult(
        score=score,
        label=label,
        matched_phrases=matched_phrases,
        encoding_detected=encoding_detected,
    )
