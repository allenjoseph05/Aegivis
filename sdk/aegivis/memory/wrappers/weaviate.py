"""
Weaviate client wrapper for the Memory Commit Validator.

Auto-detects the Weaviate client version at ``wrap()`` time:

- **v3** (``weaviate-client < 4.0``): wraps ``client.data_object.create()`` and
  ``client.batch.add_data_object()``.
- **v4** (``weaviate-client >= 4.0``): wraps ``client.collections.get(name).data.insert()``
  and ``client.collections.get(name).data.insert_many()``.

Detection is duck-typed: if ``hasattr(client, 'collections')`` the v4 path is
used, otherwise v3.  No direct ``weaviate`` import — works with any version
that exposes the documented API surface.
"""
from __future__ import annotations

from typing import Any

from aegivis.memory._base import _scan_and_report
from aegivis.memory import ScanConfig


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _extract_strings(data_object: dict[str, Any]) -> list[str]:
    """Recursively extract all string values from a dict."""
    texts: list[str] = []
    for val in data_object.values():
        if isinstance(val, str) and val.strip():
            texts.append(val)
        elif isinstance(val, dict):
            texts.extend(_extract_strings(val))
    return texts


# ---------------------------------------------------------------------------
# v3 wrapper  (weaviate-client < 4.0)
# ---------------------------------------------------------------------------

class _WrappedDataObject:
    def __init__(self, real_data_object: Any, config: ScanConfig) -> None:
        self._real = real_data_object
        self._config = config

    def create(
        self,
        data_object: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if data_object:
            texts = _extract_strings(data_object)
            if texts:
                _scan_and_report(texts, self._config)
        return self._real.create(data_object, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedBatch:
    def __init__(self, real_batch: Any, config: ScanConfig) -> None:
        self._real = real_batch
        self._config = config

    def add_data_object(
        self,
        data_object: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if data_object:
            texts = _extract_strings(data_object)
            if texts:
                _scan_and_report(texts, self._config)
        return self._real.add_data_object(data_object, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedWeaviateClientV3:
    def __init__(self, real_client: Any, config: ScanConfig) -> None:
        self._real = real_client
        self._config = config
        self.data_object = _WrappedDataObject(real_client.data_object, config)
        self.batch = _WrappedBatch(real_client.batch, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# v4 wrapper  (weaviate-client >= 4.0)
# ---------------------------------------------------------------------------

def _extract_props(obj: Any) -> list[str]:
    """Extract string values from a v4 DataObject or plain dict."""
    if isinstance(obj, dict):
        props = obj.get("properties") or obj
    else:
        props = getattr(obj, "properties", None)
    if isinstance(props, dict):
        return _extract_strings(props)
    return []


class _WrappedDataV4:
    def __init__(self, real_data: Any, config: ScanConfig) -> None:
        self._real = real_data
        self._config = config

    def insert(self, properties: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if properties and isinstance(properties, dict):
            texts = _extract_strings(properties)
            if texts:
                _scan_and_report(texts, self._config)
        return self._real.insert(properties=properties, **kwargs)

    def insert_many(self, objects: list[Any] | None = None, **kwargs: Any) -> Any:
        if objects:
            texts: list[str] = []
            for obj in objects:
                texts.extend(_extract_props(obj))
            if texts:
                _scan_and_report(texts, self._config)
        return self._real.insert_many(objects=objects, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedCollectionV4:
    def __init__(self, real_collection: Any, config: ScanConfig) -> None:
        self._real = real_collection
        self._config = config
        self.data = _WrappedDataV4(real_collection.data, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedCollectionsV4:
    def __init__(self, real_collections: Any, config: ScanConfig) -> None:
        self._real = real_collections
        self._config = config

    def get(self, name: str, **kwargs: Any) -> _WrappedCollectionV4:
        col = self._real.get(name, **kwargs)
        return _WrappedCollectionV4(col, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedWeaviateClientV4:
    def __init__(self, real_client: Any, config: ScanConfig) -> None:
        self._real = real_client
        self._config = config
        self.collections = _WrappedCollectionsV4(real_client.collections, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def wrap(client: Any, *, config: ScanConfig) -> Any:
    """
    Wrap a Weaviate client and return a guarded proxy.

    Auto-detects v3 vs v4 by checking for the ``collections`` attribute
    (present only in weaviate-client >= 4.0).
    """
    if hasattr(client, "collections"):
        return _WrappedWeaviateClientV4(client, config)
    return _WrappedWeaviateClientV3(client, config)
