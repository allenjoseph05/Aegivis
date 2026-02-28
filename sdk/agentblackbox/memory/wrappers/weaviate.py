"""
Weaviate client wrapper for the Memory Commit Validator.

Intercepts:
- ``client.data_object.create(data_object={...})``
- ``client.batch.add_data_object(data_object={...})``

Scans all string values in the ``data_object`` dict before committing.
"""
from __future__ import annotations

from typing import Any

from agentblackbox.memory._base import _scan_and_report
from agentblackbox.memory import ScanConfig


def _extract_strings(data_object: dict[str, Any]) -> list[str]:
    """Recursively extract all string values from a dict."""
    texts: list[str] = []
    for val in data_object.values():
        if isinstance(val, str) and val.strip():
            texts.append(val)
        elif isinstance(val, dict):
            texts.extend(_extract_strings(val))
    return texts


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


class _WrappedWeaviateClient:
    def __init__(self, real_client: Any, config: ScanConfig) -> None:
        self._real = real_client
        self._config = config
        self.data_object = _WrappedDataObject(real_client.data_object, config)
        self.batch = _WrappedBatch(real_client.batch, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap(client: Any, *, config: ScanConfig) -> _WrappedWeaviateClient:
    """Wrap *client* (a weaviate.Client) and return a guarded proxy."""
    return _WrappedWeaviateClient(client, config)
