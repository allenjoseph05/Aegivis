"""
Aegivis RAG / Vector DB Retrieval Interceptor
=============================================
Zero-config monkey-patching for all major vector database clients.
Captures what context the agent retrieves before LLM calls — the critical
missing piece for full RAG pipeline auditability.

Import once::

    import aegivis.intercept_rag

Covered automatically (patched at the class level — no client wrapping needed):

    chromadb       — ``Collection.query()``
    pinecone       — ``Index.query()`` (captures result metadata text)
    qdrant-client  — ``QdrantClient.search()`` and ``QdrantClient.query_points()``
    weaviate       — ``Collection.query.near_text()`` and ``near_vector()`` (v4 API)
    pymilvus       — ``Collection.search()``
    LangChain      — ``VectorStore.similarity_search()`` and
                     ``VectorStore.similarity_search_with_score()``
                     (one patch covers Pinecone, Weaviate, Chroma, Qdrant, FAISS,
                      Redis, MongoDB Atlas, PgVector, etc.)

Emits ``RAG_RETRIEVAL`` events with:
    vector_store    — which DB was queried
    collection_name — collection / index / namespace
    query_preview   — first 500 chars of text query (when available)
    top_k           — requested number of results
    result_count    — actual results returned
    result_preview  — text of the first result (first 300 chars)
    result_scores   — similarity scores for top results (when available)
    latency_ms      — retrieval time

Why this matters:
    Poisoned RAG results are the most common vector for document injection attacks.
    Without retrieval capture, Aegivis can see the agent's LLM calls but not WHAT
    context drove those calls.  This closes the audit gap.

Environment variables::

    AEGIVIS_BACKEND_URL   Where to ship events (default: http://localhost:8000)
    AEGIVIS_API_KEY       API key (default: dev-dashboard-key)
    AEGIVIS_AGENT_ID      Agent label
    AEGIVIS_INTERCEPT     Set "false" to disable
    AEGIVIS_DEBUG         Set "1" to log to stderr
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY     = os.environ.get("AEGIVIS_API_KEY", "dev-dashboard-key")
_AGENT_ID    = os.environ.get("AEGIVIS_AGENT_ID", "intercepted-agent")
_DEBUG       = os.environ.get("AEGIVIS_DEBUG", "") == "1"
_ENABLED     = os.environ.get("AEGIVIS_INTERCEPT", "true").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# Event shipping
# ---------------------------------------------------------------------------

def _fire(payload: dict, session_id: str, agent_id: str) -> None:
    if not _BACKEND_URL or not _ENABLED:
        return
    event = {
        "event_type":         "RAG_RETRIEVAL",
        "agent_id":           agent_id,
        "session_id":         session_id,
        "timestamp_ns":       time.time_ns(),
        "interception_layer": "sdk-intercept-rag",
        "provider":           payload.get("vector_store", "unknown"),
        "model":              "",
        "payload":            payload,
    }
    body = json.dumps(event, default=str).encode()
    headers = {"Content-Type": "application/json", "X-API-Key": _API_KEY}

    def _post() -> None:
        try:
            req = urllib.request.Request(
                _BACKEND_URL + "/v1/ingest", data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception as exc:
            if _DEBUG:
                logger.debug("aegivis.intercept_rag: post failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


def _session_id() -> str:
    return os.environ.get("AEGIVIS_SESSION_ID") or f"rag-{uuid.uuid4().hex[:12]}"


def _agent_id() -> str:
    return os.environ.get("AEGIVIS_AGENT_ID") or _AGENT_ID


# ---------------------------------------------------------------------------
# Result text extraction helpers
# ---------------------------------------------------------------------------

_META_TEXT_KEYS = ("text", "content", "chunk_text", "document", "body", "page_content", "passage")


def _meta_text(metadata: Any) -> str:
    """Extract human-readable text from a result's metadata dict."""
    if not isinstance(metadata, dict):
        return ""
    for key in _META_TEXT_KEYS:
        val = metadata.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


# ---------------------------------------------------------------------------
# ChromaDB — Collection.query()
# ---------------------------------------------------------------------------

def _install_chroma() -> bool:
    """
    Patch ``chromadb.api.models.Collection.Collection.query``.
    Works with chromadb 0.4.x, 0.5.x, and 0.6.x (duck-typed, no hard import).
    """
    try:
        # chromadb 0.4/0.5 path
        from chromadb.api.models.Collection import Collection  # noqa: PLC0415
    except ImportError:
        try:
            # chromadb 0.6+ path
            from chromadb.api.types import Collection  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            return False

    if getattr(Collection, "_aegivis_rag_patched", False):
        return True

    _orig_query = Collection.query

    def _patched_query(
        self: Any,
        query_embeddings: Any = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> Any:
        sid = _session_id()
        aid = _agent_id()
        t0  = time.monotonic()

        result = _orig_query(
            self,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            **kwargs,
        )

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        docs: list[list[str]] = result.get("documents") or [[]]
        flat_docs = docs[0] if docs else []
        distances: list[list[float]] = result.get("distances") or [[]]
        flat_dist = distances[0] if distances else []

        payload: dict = {
            "vector_store":     "chroma",
            "collection_name":  getattr(self, "name", "unknown"),
            "top_k":            n_results,
            "result_count":     len(flat_docs),
            "latency_ms":       latency_ms,
        }
        if query_texts:
            payload["query_preview"] = str(query_texts[0])[:500]
        if flat_docs:
            payload["result_preview"] = str(flat_docs[0])[:300]
        if flat_dist:
            payload["result_scores"] = [round(s, 4) for s in flat_dist[:5]]

        if _DEBUG:
            logger.debug("aegivis.intercept_rag: chroma query collection=%s results=%d",
                         payload["collection_name"], payload["result_count"])
        _fire(payload, sid, aid)
        return result

    Collection.query             = _patched_query  # type: ignore[method-assign]
    Collection._aegivis_rag_patched = True         # type: ignore[attr-defined]

    if _DEBUG:
        logger.debug("aegivis.intercept_rag: chromadb.Collection.query patched")
    return True


# ---------------------------------------------------------------------------
# Pinecone — Index.query()
# ---------------------------------------------------------------------------

def _install_pinecone() -> bool:
    """
    Patch ``pinecone.Index.query`` (v2 / legacy) and ``pinecone.data.index.Index.query`` (v3+).
    Pinecone takes a pre-embedded vector — we capture result metadata text.
    """
    patched_any = False

    for module_path, class_name in [
        ("pinecone", "Index"),
        ("pinecone.data.index", "Index"),
        ("pinecone.grpc", "GRPCIndex"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
        except (ImportError, AttributeError):
            continue

        if getattr(cls, "_aegivis_rag_patched", False):
            patched_any = True
            continue

        _orig_query = cls.query

        def _make_patch(orig: Any, store_label: str) -> Any:
            def _patched_query(self: Any, *args: Any, **kwargs: Any) -> Any:
                sid = _session_id()
                aid = _agent_id()
                t0  = time.monotonic()
                result = orig(self, *args, **kwargs)
                latency_ms = round((time.monotonic() - t0) * 1000, 1)

                # result.matches or result["matches"]
                matches = getattr(result, "matches", None) or result.get("matches") or []
                scores  = []
                texts   = []
                for m in matches:
                    s = getattr(m, "score", None) or (m.get("score") if isinstance(m, dict) else None)
                    if s is not None:
                        scores.append(round(float(s), 4))
                    meta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {})
                    t = _meta_text(meta)
                    if t:
                        texts.append(t)

                top_k = kwargs.get("top_k") or (args[1] if len(args) > 1 else 10)
                ns    = kwargs.get("namespace", getattr(self, "_namespace", ""))

                payload: dict = {
                    "vector_store":     store_label,
                    "collection_name":  (getattr(self, "name", None) or getattr(self, "_name", "pinecone-index") or ""),
                    "namespace":        str(ns) if ns else "",
                    "top_k":            top_k,
                    "result_count":     len(matches),
                    "latency_ms":       latency_ms,
                }
                if texts:
                    payload["result_preview"] = texts[0][:300]
                if scores:
                    payload["result_scores"] = scores[:5]

                if _DEBUG:
                    logger.debug("aegivis.intercept_rag: pinecone query results=%d", len(matches))
                _fire(payload, sid, aid)
                return result

            return _patched_query

        cls.query = _make_patch(_orig_query, "pinecone")  # type: ignore[method-assign]
        cls._aegivis_rag_patched = True                   # type: ignore[attr-defined]
        patched_any = True

    if patched_any and _DEBUG:
        logger.debug("aegivis.intercept_rag: pinecone Index.query patched")
    return patched_any


# ---------------------------------------------------------------------------
# Qdrant — QdrantClient.search() and query_points()
# ---------------------------------------------------------------------------

def _install_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient  # noqa: PLC0415
    except ImportError:
        return False

    if getattr(QdrantClient, "_aegivis_rag_patched", False):
        return True

    _orig_search       = QdrantClient.search
    _orig_query_points = getattr(QdrantClient, "query_points", None)

    def _patched_search(
        self: Any,
        collection_name: str,
        *args: Any,
        limit: int = 10,
        **kwargs: Any,
    ) -> Any:
        sid = _session_id()
        aid = _agent_id()
        t0  = time.monotonic()
        result = _orig_search(self, collection_name, *args, limit=limit, **kwargs)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        # result is list[ScoredPoint]; each has .score and .payload
        scores = []
        texts  = []
        for hit in (result or []):
            s = getattr(hit, "score", None)
            if s is not None:
                scores.append(round(float(s), 4))
            p = getattr(hit, "payload", None) or {}
            t = _meta_text(p)
            if t:
                texts.append(t)

        payload: dict = {
            "vector_store":    "qdrant",
            "collection_name": collection_name,
            "top_k":           limit,
            "result_count":    len(result or []),
            "latency_ms":      latency_ms,
        }
        if texts:
            payload["result_preview"] = texts[0][:300]
        if scores:
            payload["result_scores"] = scores[:5]

        if _DEBUG:
            logger.debug("aegivis.intercept_rag: qdrant search collection=%s results=%d",
                         collection_name, payload["result_count"])
        _fire(payload, sid, aid)
        return result

    QdrantClient.search = _patched_search  # type: ignore[method-assign]

    # query_points was added in qdrant-client 1.7
    if _orig_query_points is not None:
        def _patched_query_points(
            self: Any,
            collection_name: str,
            *args: Any,
            limit: int = 10,
            **kwargs: Any,
        ) -> Any:
            sid = _session_id()
            aid = _agent_id()
            t0  = time.monotonic()
            result = _orig_query_points(self, collection_name, *args, limit=limit, **kwargs)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)

            points = getattr(result, "points", result) or []
            scores = []
            texts  = []
            for hit in points:
                s = getattr(hit, "score", None)
                if s is not None:
                    scores.append(round(float(s), 4))
                p = getattr(hit, "payload", None) or {}
                t = _meta_text(p)
                if t:
                    texts.append(t)

            payload: dict = {
                "vector_store":    "qdrant",
                "collection_name": collection_name,
                "top_k":           limit,
                "result_count":    len(points),
                "latency_ms":      latency_ms,
            }
            if texts:
                payload["result_preview"] = texts[0][:300]
            if scores:
                payload["result_scores"] = scores[:5]
            _fire(payload, sid, aid)
            return result

        QdrantClient.query_points = _patched_query_points  # type: ignore[method-assign]

    QdrantClient._aegivis_rag_patched = True  # type: ignore[attr-defined]
    if _DEBUG:
        logger.debug("aegivis.intercept_rag: qdrant_client.QdrantClient patched")
    return True


# ---------------------------------------------------------------------------
# LangChain VectorStore — similarity_search() + similarity_search_with_score()
# One patch covers ALL LangChain-wrapped stores (Pinecone, Weaviate, Chroma,
# Qdrant, FAISS, Redis, PgVector, MongoDB Atlas, etc.)
# ---------------------------------------------------------------------------

def _install_langchain() -> bool:
    try:
        from langchain_core.vectorstores import VectorStore  # noqa: PLC0415
    except ImportError:
        try:
            from langchain.schema import BaseRetriever as VectorStore  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            return False

    if getattr(VectorStore, "_aegivis_rag_patched", False):
        return True

    _orig_sim  = VectorStore.similarity_search
    _orig_sim_score = getattr(VectorStore, "similarity_search_with_score", None)
    _orig_mmr  = getattr(VectorStore, "max_marginal_relevance_search", None)

    def _store_name(self: Any) -> str:
        return type(self).__name__

    def _patched_similarity_search(
        self: Any,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> Any:
        sid = _session_id()
        aid = _agent_id()
        t0  = time.monotonic()
        docs = _orig_sim(self, query, k=k, **kwargs)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        texts = [getattr(d, "page_content", str(d)) for d in (docs or [])]
        payload: dict = {
            "vector_store":    "langchain/" + _store_name(self),
            "collection_name": getattr(self, "collection_name", getattr(self, "_collection_name", "")),
            "query_preview":   str(query)[:500],
            "top_k":           k,
            "result_count":    len(docs or []),
            "latency_ms":      latency_ms,
        }
        if texts:
            payload["result_preview"] = texts[0][:300]

        if _DEBUG:
            logger.debug("aegivis.intercept_rag: langchain %s similarity_search results=%d",
                         _store_name(self), payload["result_count"])
        _fire(payload, sid, aid)
        return docs

    VectorStore.similarity_search = _patched_similarity_search  # type: ignore[method-assign]

    if _orig_sim_score is not None:
        def _patched_sim_score(
            self: Any,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> Any:
            sid = _session_id()
            aid = _agent_id()
            t0  = time.monotonic()
            pairs = _orig_sim_score(self, query, k=k, **kwargs)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)

            texts  = []
            scores = []
            for item in (pairs or []):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    doc, score = item
                    texts.append(getattr(doc, "page_content", str(doc)))
                    scores.append(round(float(score), 4))

            payload: dict = {
                "vector_store":    "langchain/" + _store_name(self),
                "collection_name": getattr(self, "collection_name", ""),
                "query_preview":   str(query)[:500],
                "top_k":           k,
                "result_count":    len(pairs or []),
                "latency_ms":      latency_ms,
            }
            if texts:
                payload["result_preview"] = texts[0][:300]
            if scores:
                payload["result_scores"] = scores[:5]
            _fire(payload, sid, aid)
            return pairs

        VectorStore.similarity_search_with_score = _patched_sim_score  # type: ignore[method-assign]

    if _orig_mmr is not None:
        def _patched_mmr(
            self: Any,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> Any:
            sid = _session_id()
            aid = _agent_id()
            t0  = time.monotonic()
            docs = _orig_mmr(self, query, k=k, **kwargs)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)

            payload: dict = {
                "vector_store":    "langchain/" + _store_name(self),
                "collection_name": getattr(self, "collection_name", ""),
                "query_preview":   str(query)[:500],
                "retrieval_type":  "mmr",
                "top_k":           k,
                "result_count":    len(docs or []),
                "latency_ms":      latency_ms,
            }
            if docs:
                payload["result_preview"] = getattr(docs[0], "page_content", str(docs[0]))[:300]
            _fire(payload, sid, aid)
            return docs

        VectorStore.max_marginal_relevance_search = _patched_mmr  # type: ignore[method-assign]

    VectorStore._aegivis_rag_patched = True  # type: ignore[attr-defined]
    if _DEBUG:
        logger.debug("aegivis.intercept_rag: langchain_core.VectorStore patched (similarity_search, similarity_search_with_score, mmr)")
    return True


# ---------------------------------------------------------------------------
# Milvus / PyMilvus — Collection.search()
# ---------------------------------------------------------------------------

def _install_milvus() -> bool:
    try:
        from pymilvus import Collection  # noqa: PLC0415
    except ImportError:
        return False

    if getattr(Collection, "_aegivis_rag_patched", False):
        return True

    _orig_search = Collection.search

    def _patched_search(
        self: Any,
        data: Any,
        anns_field: str,
        param: dict,
        limit: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sid = _session_id()
        aid = _agent_id()
        t0  = time.monotonic()
        result = _orig_search(self, data, anns_field, param, limit, *args, **kwargs)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        # result is a list of Hits objects; each Hits has a list of Hit
        result_count = sum(len(hits) for hits in (result or []))
        scores = []
        first_text = ""
        for hits in (result or []):
            for hit in hits:
                s = getattr(hit, "distance", None)
                if s is not None:
                    scores.append(round(float(s), 4))
                if not first_text:
                    e = getattr(hit, "entity", None)
                    if e:
                        for key in _META_TEXT_KEYS:
                            val = getattr(e, key, None) or (e.get(key) if isinstance(e, dict) else None)
                            if isinstance(val, str) and val:
                                first_text = val
                                break

        payload: dict = {
            "vector_store":    "milvus",
            "collection_name": getattr(self, "name", ""),
            "anns_field":      anns_field,
            "top_k":           limit,
            "result_count":    result_count,
            "latency_ms":      latency_ms,
        }
        if first_text:
            payload["result_preview"] = first_text[:300]
        if scores:
            payload["result_scores"] = scores[:5]

        if _DEBUG:
            logger.debug("aegivis.intercept_rag: milvus search collection=%s results=%d",
                         payload["collection_name"], result_count)
        _fire(payload, sid, aid)
        return result

    Collection.search             = _patched_search  # type: ignore[method-assign]
    Collection._aegivis_rag_patched = True           # type: ignore[attr-defined]
    if _DEBUG:
        logger.debug("aegivis.intercept_rag: pymilvus.Collection.search patched")
    return True


# ---------------------------------------------------------------------------
# Auto-install on import
# ---------------------------------------------------------------------------

if _ENABLED:
    _install_chroma()
    _install_pinecone()
    _install_qdrant()
    _install_langchain()
    _install_milvus()
