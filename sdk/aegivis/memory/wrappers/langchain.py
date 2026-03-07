"""
LangChain memory wrapper for the Memory Commit Validator.

Monkey-patches ``save_context()`` on any LangChain BaseMemory subclass
so texts are scanned before being committed.

Compatible with::
    ConversationBufferMemory, ConversationSummaryMemory,
    VectorStoreRetrieverMemory, ConversationKGMemory, and custom memories.
"""
from __future__ import annotations

from typing import Any

from aegivis.memory._base import _scan_and_report
from aegivis.memory import ScanConfig


def wrap(memory: Any, *, config: ScanConfig) -> Any:
    """
    Patch ``memory.save_context()`` in-place and return *memory*.

    The patch extracts all string values from the ``outputs`` dict passed
    to ``save_context`` and scans them before forwarding to the original
    implementation.
    """
    original_save_context = memory.save_context

    def _guarded_save_context(
        inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> Any:
        # Scan string values from both inputs and outputs
        texts: list[str] = []
        for val in list((inputs or {}).values()) + list((outputs or {}).values()):
            if isinstance(val, str):
                texts.append(val)
        if texts:
            _scan_and_report(texts, config)
        return original_save_context(inputs, outputs)

    memory.save_context = _guarded_save_context
    return memory
