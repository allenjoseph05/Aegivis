"""
Pinecone index wrapper for the Memory Commit Validator.

Intercepts ``index.upsert(vectors=[...])`` to scan the ``text`` /
``content`` / ``chunk_text`` field in each vector's metadata dict.

Example vector format::

    index.upsert(vectors=[
        {"id": "1", "values": [...], "metadata": {"text": "some document text"}}
    ])
"""
from __future__ import annotations

from typing import Any

from aegivis.memory._base import _scan_and_report
from aegivis.memory import ScanConfig

# Metadata keys that typically hold the raw document text in Pinecone
_TEXT_KEYS = ("text", "content", "chunk_text", "document", "body", "page_content")


def _extract_texts(vectors: list[Any]) -> list[str]:
    texts: list[str] = []
    for vec in vectors:
        if not isinstance(vec, dict):
            continue
        meta = vec.get("metadata") or {}
        if isinstance(meta, dict):
            for key in _TEXT_KEYS:
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append(val)
                    break  # one text per vector
    return texts


def wrap(index: Any, *, config: ScanConfig) -> Any:
    """Monkey-patch ``index.upsert`` in-place and return *index*."""
    original_upsert = index.upsert

    def _guarded_upsert(vectors: list[Any] | None = None, **kwargs: Any) -> Any:
        if vectors:
            texts = _extract_texts(vectors)
            if texts:
                _scan_and_report(texts, config)
        return original_upsert(vectors=vectors, **kwargs)

    index.upsert = _guarded_upsert
    return index
