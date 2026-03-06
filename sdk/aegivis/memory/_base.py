"""
Shared core logic used by all memory wrapper implementations.

Every wrapper calls :func:`_scan_and_report` before committing texts to
the vector store.  This keeps the actual scan/block/report logic in one
place so wrappers stay thin.
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


def _scan_and_report(texts: list[str], config: "ScanConfig") -> None:
    """
    Scan *texts* for injection content.

    - Score ≥ ``config.block_threshold``:  raises :class:`MemoryInjectionError`.
    - Score ≥ ``config.alert_threshold``:  logs a warning and reports MEMORY_WRITE_SCANNED.
    - Below alert threshold: no action.

    Parameters
    ----------
    texts  : list[str]
        Document strings to scan.  Empty / non-string items are skipped.
    config : ScanConfig
        Thresholds and optional backend reporting configuration.

    Raises
    ------
    MemoryInjectionError
        When any text's injection score exceeds ``config.block_threshold``.
    """
    from aegivis.memory import MemoryInjectionError  # noqa: PLC0415

    reporter = MemoryEventReporter(
        backend_url=config.backend_url,
        api_key=config.api_key,
    )

    for text in texts:
        if not isinstance(text, str):
            continue
        if not text.strip():
            continue

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
