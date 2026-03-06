"""
ChromaDB wrapper for the Memory Commit Validator.

Intercepts ``client.get_or_create_collection()``, ``client.create_collection()``,
and ``client.get_collection()`` to return a wrapped Collection whose ``.add()``
and ``.upsert()`` methods are guarded by the injection scanner.

**Compatibility:** Verified compatible with ``chromadb`` **0.4.x and 0.5.x**.
The wrapper uses duck typing (``Any``) — no direct ``chromadb`` import — so it
works with any version that exposes the standard Collection API.  The three
client methods and the ``documents=`` parameter on ``.add()`` / ``.upsert()``
are stable across both release lines.  ChromaDB 0.5's breaking changes
(``EmbeddingFunction`` interface, ``Settings`` restructure) do not affect this
wrapper because collection data operations are unchanged.
"""
from __future__ import annotations

from typing import Any

from aegivis.memory._base import _scan_and_report
from aegivis.memory import ScanConfig


class _WrappedCollection:
    """Thin proxy around a real ChromaDB Collection."""

    def __init__(self, real_collection: Any, config: ScanConfig) -> None:
        self._real = real_collection
        self._config = config

    def add(self, documents: list[str] | None = None, **kwargs: Any) -> Any:
        if documents:
            _scan_and_report(documents, self._config)
        return self._real.add(documents=documents, **kwargs)

    def upsert(self, documents: list[str] | None = None, **kwargs: Any) -> Any:
        if documents:
            _scan_and_report(documents, self._config)
        return self._real.upsert(documents=documents, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedChromaClient:
    """Thin proxy around a ChromaDB Client that returns wrapped Collections."""

    def __init__(self, real_client: Any, config: ScanConfig) -> None:
        self._real = real_client
        self._config = config

    def get_or_create_collection(self, *args: Any, **kwargs: Any) -> _WrappedCollection:
        col = self._real.get_or_create_collection(*args, **kwargs)
        return _WrappedCollection(col, self._config)

    def create_collection(self, *args: Any, **kwargs: Any) -> _WrappedCollection:
        col = self._real.create_collection(*args, **kwargs)
        return _WrappedCollection(col, self._config)

    def get_collection(self, *args: Any, **kwargs: Any) -> _WrappedCollection:
        col = self._real.get_collection(*args, **kwargs)
        return _WrappedCollection(col, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap(client: Any, *, config: ScanConfig) -> _WrappedChromaClient:
    """Wrap *client* (a ChromaDB Client) and return a guarded proxy."""
    return _WrappedChromaClient(client, config)
