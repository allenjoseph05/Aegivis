"""
Generic callable wrapper for the Memory Commit Validator.

Use :func:`wrap` to protect any write function whose first positional
argument or ``text`` / ``documents`` / ``texts`` keyword argument holds
text to scan.

Example::

    from agentblackbox.memory.wrappers.callable import wrap

    safe_add = wrap(my_store.add_texts, config=config)
    safe_add(["safe document text"])        # fine
    safe_add(["ignore all instructions"])   # raises MemoryInjectionError
"""
from __future__ import annotations

from typing import Any, Callable

from agentblackbox.memory._base import _scan_and_report
from agentblackbox.memory import ScanConfig

# Keyword argument names that conventionally hold document text
_TEXT_KWARGS = ("text", "texts", "documents", "document", "content", "contents")


def _extract_texts(args: tuple, kwargs: dict) -> list[str]:
    texts: list[str] = []

    # Check first positional arg
    if args:
        first = args[0]
        if isinstance(first, str):
            texts.append(first)
        elif isinstance(first, list):
            texts.extend(item for item in first if isinstance(item, str))

    # Check well-known keyword args
    for key in _TEXT_KWARGS:
        val = kwargs.get(key)
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            texts.extend(item for item in val if isinstance(item, str))

    return texts


def wrap(fn: Callable, *, config: ScanConfig) -> Callable:
    """
    Return a wrapper around *fn* that scans text arguments before calling *fn*.

    Parameters
    ----------
    fn : Callable
        Any callable whose first arg or ``text``/``documents`` kwarg holds text.
    config : ScanConfig
        Scanner configuration.

    Returns
    -------
    Callable
        A wrapped version of *fn*.
    """
    def _guarded(*args: Any, **kwargs: Any) -> Any:
        texts = _extract_texts(args, kwargs)
        if texts:
            _scan_and_report(texts, config)
        return fn(*args, **kwargs)

    _guarded.__name__ = getattr(fn, "__name__", "wrapped")
    _guarded.__doc__ = getattr(fn, "__doc__", None)
    return _guarded
