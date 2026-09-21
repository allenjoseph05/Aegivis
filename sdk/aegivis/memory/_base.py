"""
Shared core logic used by all memory wrapper implementations.

Every wrapper calls :func:`_scan_and_report` before committing texts to
the vector store.  This keeps the actual scan/block/report logic in one
place so wrappers stay thin.

Two independent scanners run on every document:

1. **Injection scanner** (``scan_text``): detects prompt-injection payloads
   embedded in documents intended to hijack the agent at retrieval time.

2. **RAG-poison scanner** (``score_rag_poison``): detects safety-alignment-
   triggering documents (Li et al., arXiv:2603.03919) — documents that contain
   safety-framing language co-occurring with harm-topic markers, causing the
   LLM's RLHF safety filter to fire false-positive refusals and deny service
   to legitimate agent queries.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aegivis.scanner import scan_text
from aegivis.client import (
    MemoryEventReporter,
    MEMORY_WRITE_BLOCKED,
    MEMORY_WRITE_SCANNED,
)

if TYPE_CHECKING:
    from aegivis.memory import ScanConfig

logger = logging.getLogger(__name__)

# Event type emitted when a RAG-poison document is detected.
# Sent to the backend via the same MemoryEventReporter channel.
MEMORY_RAG_POISON_DETECTED = "MEMORY_RAG_POISON_DETECTED"


def _scan_and_report(texts: list[str], config: "ScanConfig") -> None:
    """
    Scan *texts* for injection content and RAG-poison content.

    **Injection scanning:**

    - Score ≥ ``config.block_threshold``:  raises :class:`MemoryInjectionError`.
    - Score ≥ ``config.alert_threshold``:  logs a warning and reports MEMORY_WRITE_SCANNED.

    **RAG-poison scanning** (Li et al., arXiv:2603.03919):

    - Detects documents with safety-framing language co-occurring with harm
      topics — the pattern used to trigger false-positive LLM refusals.
    - Score ≥ ``config.rag_poison_block_threshold`` (default 0.70): raises
      :class:`MemoryInjectionError` (document rejected at ingestion time).
    - Score ≥ ``config.rag_poison_alert_threshold`` (default 0.35): logs a
      warning and reports MEMORY_RAG_POISON_DETECTED (document allowed but
      flagged for operator review).

    Parameters
    ----------
    texts  : list[str]
        Document strings to scan.  Empty / non-string items are skipped.
    config : ScanConfig
        Thresholds and optional backend reporting configuration.

    Raises
    ------
    MemoryInjectionError
        When any text's injection score or RAG-poison score exceeds the
        respective block threshold.
    """
    from aegivis.memory import MemoryInjectionError  # noqa: PLC0415
    from aegivis.security.rag_poison import score_rag_poison  # noqa: PLC0415

    reporter = MemoryEventReporter(
        backend_url=config.backend_url,
        api_key=config.api_key,
    )

    # Pull RAG-poison thresholds from config (with backwards-compat defaults)
    rag_alert_threshold = getattr(config, "rag_poison_alert_threshold", 0.35)
    rag_block_threshold = getattr(config, "rag_poison_block_threshold", 0.70)

    for text in texts:
        if not isinstance(text, str):
            continue
        if not text.strip():
            continue

        # ── Injection scanner ─────────────────────────────────────────────────
        result = scan_text(text)

        if result.score >= config.block_threshold:
            reporter.report(
                MEMORY_WRITE_BLOCKED,
                text_preview=text[:200],
                score=result.score,
                matched_phrases=result.matched_phrases,
                agent_id=config.agent_id,
                session_id=config.session_id,
            )
            raise MemoryInjectionError(
                f"Memory write blocked (score={result.score:.2f}): "
                f"{result.matched_phrases[:3]}"
            )

        if result.score >= config.alert_threshold:
            logger.warning(
                "MemoryGuard ALERT: suspicious content (score=%.2f) phrases=%s",
                result.score,
                result.matched_phrases[:3],
            )
            reporter.report(
                MEMORY_WRITE_SCANNED,
                text_preview=text[:200],
                score=result.score,
                matched_phrases=result.matched_phrases,
                agent_id=config.agent_id,
                session_id=config.session_id,
            )

        # ── RAG-poison scanner ────────────────────────────────────────────────
        # Runs independently of the injection scanner so a clean injection score
        # does not suppress RAG-poison detection.
        try:
            rp = score_rag_poison(text, alert_threshold=rag_alert_threshold)
        except Exception as _rp_exc:  # pragma: no cover
            logger.debug("RAG-poison scan error (skipped): %s", _rp_exc)
            continue

        if rp.score >= rag_block_threshold:
            logger.warning(
                "MemoryGuard RAG-POISON BLOCK: safety-framing+harm co-occurrence "
                "(score=%.2f sf=%.2f ht=%.2f) — rejecting document at ingestion",
                rp.score, rp.safety_framing_score, rp.harm_topic_score,
            )
            reporter.report(
                MEMORY_WRITE_BLOCKED,
                text_preview=text[:200],
                score=rp.score,
                matched_phrases=rp.safety_matches + rp.harm_matches,
                rag_poison=True,
                rag_poison_score=rp.score,
                agent_id=config.agent_id,
                session_id=config.session_id,
            )
            raise MemoryInjectionError(
                f"Memory write blocked — RAG-poison document detected "
                f"(score={rp.score:.2f}, safety_framing={rp.safety_framing_score:.2f}, "
                f"harm_topics={rp.harm_topic_score:.2f})"
            )

        if rp.detected:  # score >= rag_alert_threshold
            logger.warning(
                "MemoryGuard RAG-POISON ALERT: safety-framing+harm co-occurrence "
                "(score=%.2f sf=%.2f ht=%.2f)",
                rp.score, rp.safety_framing_score, rp.harm_topic_score,
            )
            reporter.report(
                MEMORY_RAG_POISON_DETECTED,
                text_preview=text[:200],
                score=rp.score,
                rag_poison_score=rp.score,
                safety_framing_score=rp.safety_framing_score,
                harm_topic_score=rp.harm_topic_score,
                matched_phrases=rp.safety_matches + rp.harm_matches,
                agent_id=config.agent_id,
                session_id=config.session_id,
            )
