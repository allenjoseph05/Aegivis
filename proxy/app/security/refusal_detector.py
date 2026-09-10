"""
Refusal detector — detect when an LLM refuses to answer a request.

Motivation (Li et al., arXiv:2603.03919):
    Adversaries inject "risk context" documents into a RAG knowledge base.
    These documents contain safety-framing language co-occurring with harm
    topics.  The LLM's own safety alignment fires, causing it to refuse the
    agent's legitimate query — a Denial-of-Service attack via false positives.

    To surface this, we score every LLM response for refusal signals.
    A spike in refusal rate across many requests/sessions is the observable
    footprint of a RAG-poison DoS campaign.

Scoring:
    STRONG_PHRASES  (score += 0.55)  — explicit cannot/won't/refuse language
    SOFT_PHRASES    (score += 0.30)  — hedging / safety-alignment language
    SHORT_RESPONSE  (tokens < 80)   — refusals are usually terse
    LOW_CONTENT_RATIO (< 12 chars/token) — little actual content

    Final score is capped at 1.0.  Detection threshold: 0.50.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Strong refusal patterns (explicit cannot / won't) ────────────────────────
_STRONG_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\bi (can(?:not|'t)|won't|will not|am unable to|refuse to|must decline)\b", re.I),
    re.compile(r"\bi'm (unable|not able) to (help|assist|provide|answer|complete|do that)\b", re.I),
    re.compile(r"\bI (must|have to|need to) (decline|refuse|stop|abstain)\b", re.I),
    re.compile(r"\b(I cannot|I can't|I won't|I will not) (assist|help|support|provide|generate|create|do)\b", re.I),
    re.compile(r"\bthat('s| is) (something I|not something I) (can|will|am able)", re.I),
    re.compile(r"\bI (don't|do not) (think I (should|can)|feel comfortable)\b", re.I),
]

# ── Soft refusal / safety-alignment phrases ───────────────────────────────────
_SOFT_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\b(against|violates?|outside of) (my |the )?(guidelines?|policies|terms|values|training)\b", re.I),
    re.compile(r"\b(this (request|question|task)|your request) (is |appears |seems )?(inappropriate|harmful|dangerous|unethical|illegal)\b", re.I),
    re.compile(r"\b(I('m| am)) (sorry|afraid),? but\b", re.I),
    re.compile(r"\b(I understand (your|the) (concern|request|question),? but)\b", re.I),
    re.compile(r"\b(potential(ly)?|serious) (harm|damage|danger|risk)\b", re.I),
    re.compile(r"\b(safety|ethical) (concern|consideration|reason|principle)\b", re.I),
    re.compile(r"\b(could be|may be|is|are) (harmful|dangerous|misused|exploited)\b", re.I),
    re.compile(r"\b(designed|built|trained|programmed) (to (avoid|prevent|not|refuse))\b", re.I),
    re.compile(r"\b(please (consult|contact|speak with|seek|reach out to)|I (recommend|suggest|encourage))\b", re.I),
    re.compile(r"\bI (appreciate|understand) (your|the) (understanding|patience|concern)\b", re.I),
]

_SCORE_STRONG = 0.55
_SCORE_SOFT = 0.30
_SHORT_TOKEN_THRESHOLD = 80
_SHORT_RESPONSE_BONUS = 0.15
_LOW_CONTENT_RATIO = 12   # chars per token; lower = less content-dense
_DETECTION_THRESHOLD = 0.50


@dataclass
class RefusalResult:
    detected: bool
    score: float
    strong_match: bool
    matched_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "score": round(self.score, 3),
            "strong_match": self.strong_match,
            "matched_phrases": self.matched_phrases[:5],
        }


def score_refusal(
    response_text: str,
    token_count: int | None = None,
    finish_reason: str | None = None,
) -> RefusalResult:
    """
    Score a single LLM response for refusal signals.

    Parameters
    ----------
    response_text : str
        The full text content of the LLM's response.
    token_count : int | None
        Completion token count from the provider (used for short-response boost).
    finish_reason : str | None
        Provider finish reason ('stop', 'end_turn', etc.).  Non-'stop' reasons
        (max_tokens, tool_calls) suppress the refusal score.

    Returns
    -------
    RefusalResult
        score 0.0–1.0.  detected=True when score >= 0.50.
    """
    if not response_text or not response_text.strip():
        return RefusalResult(detected=False, score=0.0, strong_match=False)

    # Streaming/tool-call completions are not refusals
    if finish_reason and finish_reason not in ("stop", "end_turn", "max_tokens", None):
        return RefusalResult(detected=False, score=0.0, strong_match=False)

    text = response_text.strip()
    score = 0.0
    strong_match = False
    matched: list[str] = []

    # ── Strong phrase check ───────────────────────────────────────────────────
    for pat in _STRONG_PHRASES:
        m = pat.search(text)
        if m:
            score += _SCORE_STRONG
            strong_match = True
            matched.append(m.group(0)[:60])
            break  # one strong phrase is enough

    # ── Soft phrase check ─────────────────────────────────────────────────────
    soft_hits = 0
    for pat in _SOFT_PHRASES:
        m = pat.search(text)
        if m:
            score += _SCORE_SOFT
            soft_hits += 1
            matched.append(m.group(0)[:60])
            if soft_hits >= 2:
                break  # cap at two soft hits to avoid over-counting

    # ── Short-response boost ──────────────────────────────────────────────────
    # Refusals are terse.  Count tokens by whitespace split when provider
    # token_count is unavailable.
    est_tokens = token_count if token_count is not None else len(text.split())
    if score > 0.0 and est_tokens < _SHORT_TOKEN_THRESHOLD:
        score += _SHORT_RESPONSE_BONUS

    # ── Low content-density penalty ───────────────────────────────────────────
    if est_tokens > 0 and (len(text) / est_tokens) < _LOW_CONTENT_RATIO and score > 0.0:
        score += 0.05

    score = min(score, 1.0)
    detected = score >= _DETECTION_THRESHOLD

    return RefusalResult(
        detected=detected,
        score=score,
        strong_match=strong_match,
        matched_phrases=matched,
    )
