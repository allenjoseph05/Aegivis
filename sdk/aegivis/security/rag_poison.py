"""
RAG poison detector — detect safety-alignment-triggering documents.

Background (Li et al., arXiv:2603.03919 — "When Safety Becomes a Vulnerability"):
    Adversaries inject "risk context" documents into a RAG knowledge base.
    Each poisoned document contains:
      1. Safety-framing language — phrases that mimic model safety training
         ("For safety reasons...", "WARNING: This content involves...")
      2. Harm-topic markers     — references to dangerous/illegal topics
         ("weapons", "drug synthesis", "hacking", etc.)

    When the agent queries the knowledge base, the retrieved poisoned document
    appears in the LLM's context window.  The LLM's own RLHF/safety alignment
    fires, causing it to refuse the agent's *legitimate* query.
    Net effect: Denial-of-Service via false-positive safety refusals.

Detection strategy (co-occurrence scoring):
    We score each document on two independent axes:
      - safety_framing_score  : presence of safety-training-mimic language
      - harm_topic_score      : presence of harm-domain keywords

    rag_poison_score = safety_framing_score * harm_topic_score * 2.0  (capped at 1.0)

    Using multiplication enforces co-occurrence — either axis alone does NOT
    trigger detection, only the combination does.  This eliminates FPs for:
      - Legitimate safety documentation (high framing, zero harm topics)
      - Legitimate security research docs (harm topics, zero safety framing)

Thresholds (configurable via ScanConfig.rag_poison_*):
    alert : 0.35   — log warning + report MEMORY_WRITE_SCANNED
    block : 0.70   — raise MemoryInjectionError (same as injection scanner)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Safety-framing patterns (mimic of RLHF safety training language) ──────────
# These are phrases that appear in safety-filtered training data and thus
# prime the LLM's safety classifier to trigger on subsequent text.
_SAFETY_FRAMING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(for safety (reasons|considerations|purposes))\b", re.I),
    re.compile(r"\b(WARNING|CAUTION|DANGER|ALERT)\s*[:!]\s*(this (content|document|text|information))\b", re.I),
    re.compile(r"\b(this (content|material|information|document) (is|may be|could be) (harmful|dangerous|sensitive|inappropriate))\b", re.I),
    re.compile(r"\b(content (warning|advisory|notice))\b", re.I),
    re.compile(r"\b(trigger (warning|content))\b", re.I),
    re.compile(r"\b(do not (attempt|try|replicate|reproduce|follow) (this|these|the following))\b", re.I),
    re.compile(r"\b(this (violates?|goes against|conflicts with) (guidelines?|policies|safety|ethics|laws?))\b", re.I),
    re.compile(r"\b(in the context of (potential|serious|real) (harm|danger|risk|threat))\b", re.I),
    re.compile(r"\b((potential|possible|serious) (harm|danger|risk) (to (others?|people|society|users?)))\b", re.I),
    re.compile(r"\b(should (not|never) be (used|applied|followed|attempted) (for|to|without))\b", re.I),
    re.compile(r"\b(responsible (use|disclosure|handling|practices))\b", re.I),
    re.compile(r"\b(ethical (concern|consideration|implication|guideline))\b", re.I),
    re.compile(r"\b(misuse (of|potential|risk))\b", re.I),
]

# ── Harm-topic markers (harmful domains that trigger LLM safety filters) ──────
# These cover the ten harm categories that RLHF safety training consistently
# targets. Presence signals the document is about a safety-sensitive topic.
_HARM_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    # Violence / weapons
    re.compile(r"\b(weapon[s]?|firearm[s]?|explosiv[e]?s?|bomb[s]?|grenade[s]?|ammunition)\b", re.I),
    re.compile(r"\b(kill[s]?ing|murder|assassination|torture|abuse)\b", re.I),
    re.compile(r"\b(mass (casualt(y|ies)|shooting|killing|destruction))\b", re.I),
    # Chemical / biological
    re.compile(r"\b(synthesis (of|route|step)|chemical (weapon|agent|precursor)|nerve agent)\b", re.I),
    re.compile(r"\b(bioweapon|pathogen|toxin|poison[s]? (production|synthesis|making))\b", re.I),
    # Drugs / illegal substances
    re.compile(r"\b(drug (synthesis|manufacturing|production|trafficking|dealing)|narcotic[s]?)\b", re.I),
    re.compile(r"\b(methamphetamine|fentanyl|heroin|cocaine) (production|synthesis|how to (make|produce))\b", re.I),
    # Hacking / cyber
    re.compile(r"\b(malware|ransomware|exploit|zero.?day|remote code execution|sql injection|phish(ing)?)\b", re.I),
    re.compile(r"\b(hack(ing)?|unauthorized access|password crack(ing)?|credential (theft|dump|stuffing))\b", re.I),
    # CSAM / exploitation
    re.compile(r"\b(child (exploit(ation)?|abus(e|ing)|sexu(al)?)|minor[s]? (nude|naked|sexu))\b", re.I),
    # Fraud / financial crime
    re.compile(r"\b((identity|financial) (fraud|theft)|money laundering|credit card (fraud|skimming|cloning))\b", re.I),
    # Self-harm
    re.compile(r"\b(suicide (method[s]?|instruction[s]?|how to)|self.harm (technique[s]?|method[s]?))\b", re.I),
    # Radicalization / terrorism
    re.compile(r"\b(terrori(sm|st)|radicali(zation|sing)|jihad|extremi(sm|st))\b", re.I),
]

# Per-hit contribution to each axis score.
# Using flat increments rather than 1/N so the scores don't dilute as
# the pattern library grows.  Calibrated so that:
#   - 1 safety-framing hit  →  0.25   (weak framing)
#   - 3 safety-framing hits →  0.75   (strong framing)
#   - 1 harm-topic hit      →  0.40   (one concrete harm domain present)
#   - 2 harm-topic hits     →  0.80   (two harm domains → very suspicious)
# Combined (multiplicative, × 2, capped at 1.0):
#   1 safety (0.25) + 1 harm (0.40) → 0.25 × 0.40 × 2 = 0.20  (below alert)
#   2 safety (0.50) + 1 harm (0.40) → 0.50 × 0.40 × 2 = 0.40  (alert)
#   3 safety (0.75) + 1 harm (0.40) → 0.75 × 0.40 × 2 = 0.60  (alert)
_WEIGHT_SAFETY_FRAMING = 0.25
_WEIGHT_HARM_TOPIC = 0.40

# Default thresholds (overridable via ScanConfig attributes)
DEFAULT_ALERT_THRESHOLD = 0.35
DEFAULT_BLOCK_THRESHOLD = 0.70


@dataclass
class RagPoisonResult:
    """Result of a single document scan."""
    score: float                        # 0.0 – 1.0
    detected: bool                      # score >= alert_threshold
    safety_framing_score: float         # 0.0 – 1.0 (axis 1)
    harm_topic_score: float             # 0.0 – 1.0 (axis 2)
    safety_matches: list[str] = field(default_factory=list)
    harm_matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "detected": self.detected,
            "safety_framing_score": round(self.safety_framing_score, 3),
            "harm_topic_score": round(self.harm_topic_score, 3),
            "safety_matches": self.safety_matches[:3],
            "harm_matches": self.harm_matches[:3],
        }


def score_rag_poison(text: str, alert_threshold: float = DEFAULT_ALERT_THRESHOLD) -> RagPoisonResult:
    """
    Score a single document for RAG-poison characteristics.

    Uses multiplicative co-occurrence: both safety-framing language AND
    harm-topic markers must be present for the score to be non-trivial.

    Parameters
    ----------
    text : str
        Raw document text being written to the RAG store.
    alert_threshold : float
        Score at which ``detected`` is set to True.

    Returns
    -------
    RagPoisonResult
        score in [0.0, 1.0].  detected=True when score >= alert_threshold.
    """
    if not text or not text.strip():
        return RagPoisonResult(score=0.0, detected=False,
                               safety_framing_score=0.0, harm_topic_score=0.0)

    safety_matches: list[str] = []
    harm_matches: list[str] = []

    # ── Safety-framing axis ───────────────────────────────────────────────────
    safety_hits = 0
    for pat in _SAFETY_FRAMING_PATTERNS:
        m = pat.search(text)
        if m:
            safety_hits += 1
            safety_matches.append(m.group(0)[:80])

    safety_framing_score = min(safety_hits * _WEIGHT_SAFETY_FRAMING, 1.0)

    # ── Harm-topic axis ───────────────────────────────────────────────────────
    harm_hits = 0
    for pat in _HARM_TOPIC_PATTERNS:
        m = pat.search(text)
        if m:
            harm_hits += 1
            harm_matches.append(m.group(0)[:80])

    harm_topic_score = min(harm_hits * _WEIGHT_HARM_TOPIC, 1.0)

    # ── Combined score (multiplicative co-occurrence) ─────────────────────────
    combined = min(safety_framing_score * harm_topic_score * 2.0, 1.0)

    return RagPoisonResult(
        score=combined,
        detected=combined >= alert_threshold,
        safety_framing_score=safety_framing_score,
        harm_topic_score=harm_topic_score,
        safety_matches=safety_matches,
        harm_matches=harm_matches,
    )
