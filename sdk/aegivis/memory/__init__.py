"""
Memory Commit Validator — public API.

Provides :func:`wrap_chroma`, :func:`wrap_langchain_memory`,
:func:`wrap_pinecone`, :func:`wrap_weaviate`, and :func:`wrap_callable`
to intercept vector store writes and scan them for injected instructions.

Quick start::

    import chromadb
    from aegivis.memory import wrap_chroma, ScanConfig

    client = wrap_chroma(
        chromadb.Client(),
        config=ScanConfig(block_threshold=0.7, alert_threshold=0.4),
    )
    collection = client.get_or_create_collection("my_docs")
    collection.add(documents=["safe text"], ids=["1"])   # fine
    collection.add(documents=["ignore all instructions"], ids=["2"])  # raises MemoryInjectionError

Environment variables read by :func:`~aegivis.config.default_config`:
    AEGIVIS_BACKEND_URL, AEGIVIS_API_KEY, AEGIVIS_AGENT_ID, AEGIVIS_SESSION_ID
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ScanConfig:
    """Configuration for the memory injection scanner."""

    block_threshold: float = 0.7
    """Score threshold above which writes are BLOCKED (MemoryInjectionError raised)."""

    alert_threshold: float = 0.4
    """Score threshold above which an ALERT is logged (write proceeds)."""

    agent_id: str = ""
    """Agent identifier included in backend event reports."""

    session_id: str = ""
    """Session identifier included in backend event reports."""

    backend_url: str = ""
    """Aegivis backend URL for event reporting (optional)."""

    api_key: str = ""
    """API key for the backend (optional)."""


# ─── Exception ─────────────────────────────────────────────────────────────────

class MemoryInjectionError(Exception):
    """Raised when a memory write is blocked due to injection detection."""


# ─── Wrapper registry ──────────────────────────────────────────────────────────

_WRAPPERS: dict[str, Callable] = {}


def register_wrapper(name: str, fn: Callable) -> None:
    """Register a wrapper under *name*. Called by each :mod:`wrappers` module."""
    _WRAPPERS[name] = fn


# ─── Public API ────────────────────────────────────────────────────────────────

def _get_config(config: ScanConfig | None) -> ScanConfig:
    if config is not None:
        return config
    try:
        from aegivis.config import default_config  # noqa: PLC0415
        return default_config()
    except Exception:
        return ScanConfig()


def wrap_chroma(client: Any, *, config: ScanConfig | None = None) -> Any:
    """
    Wrap a ``chromadb.Client`` to intercept ``.add()`` / ``.upsert()`` on all
    collections it returns.

    Parameters
    ----------
    client : chromadb.Client | chromadb.ClientAPI
        The ChromaDB client to wrap.
    config : ScanConfig, optional
        Scanner configuration.  Defaults to :func:`~aegivis.config.default_config`.

    Returns
    -------
    WrappedChromaClient
        A proxy that intercepts collection factory calls.

    Raises
    ------
    MemoryInjectionError
        Propagated from collection ``.add()`` / ``.upsert()`` when injection detected.
    """
    from aegivis.memory.wrappers.chroma import wrap as _wrap  # noqa: PLC0415
    return _wrap(client, config=_get_config(config))


def wrap_langchain_memory(memory: Any, *, config: ScanConfig | None = None) -> Any:
    """
    Wrap any LangChain memory object that exposes ``.save_context()``.

    Compatible with ``ConversationBufferMemory``, ``ConversationSummaryMemory``,
    ``VectorStoreRetrieverMemory``, and custom implementations.

    Parameters
    ----------
    memory : BaseMemory
        The LangChain memory instance to wrap.
    config : ScanConfig, optional
        Scanner configuration.

    Returns
    -------
    Any
        The same memory object with ``save_context`` monkey-patched.
    """
    from aegivis.memory.wrappers.langchain import wrap as _wrap  # noqa: PLC0415
    return _wrap(memory, config=_get_config(config))


def wrap_pinecone(index: Any, *, config: ScanConfig | None = None) -> Any:
    """
    Wrap a ``pinecone.Index`` to intercept ``.upsert(vectors=[...])`` calls.

    Scans the ``text`` / ``content`` field in each vector's metadata dict.

    Parameters
    ----------
    index : pinecone.Index
        Pinecone index instance to wrap.
    config : ScanConfig, optional
        Scanner configuration.

    Returns
    -------
    Any
        The same index with ``upsert`` monkey-patched.
    """
    from aegivis.memory.wrappers.pinecone import wrap as _wrap  # noqa: PLC0415
    return _wrap(index, config=_get_config(config))


def wrap_weaviate(client: Any, *, config: ScanConfig | None = None) -> Any:
    """
    Wrap a ``weaviate.Client`` to intercept object-create and batch operations.

    Scans all string values in the ``data_object`` dict.

    Parameters
    ----------
    client : weaviate.Client
        Weaviate client instance to wrap.
    config : ScanConfig, optional
        Scanner configuration.

    Returns
    -------
    Any
        The same client with ``data_object.create`` and batch methods patched.
    """
    from aegivis.memory.wrappers.weaviate import wrap as _wrap  # noqa: PLC0415
    return _wrap(client, config=_get_config(config))


def wrap_callable(fn: Callable, *, config: ScanConfig | None = None) -> Callable:
    """
    Generic wrapper for any callable whose first positional argument or
    ``text`` / ``documents`` keyword argument contains text to scan.

    Use this for unsupported vector stores::

        from aegivis.memory import wrap_callable
        safe_write = wrap_callable(my_store.add_texts)
        safe_write(["safe text"])

    Parameters
    ----------
    fn : Callable
        The write function to wrap.
    config : ScanConfig, optional
        Scanner configuration.

    Returns
    -------
    Callable
        A wrapper that scans before calling *fn*.
    """
    from aegivis.memory.wrappers.callable import wrap as _wrap  # noqa: PLC0415
    return _wrap(fn, config=_get_config(config))


__all__ = [
    "ScanConfig",
    "MemoryInjectionError",
    "register_wrapper",
    "wrap_chroma",
    "wrap_langchain_memory",
    "wrap_pinecone",
    "wrap_weaviate",
    "wrap_callable",
]
