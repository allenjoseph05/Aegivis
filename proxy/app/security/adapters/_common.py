"""
Shared utilities for all sandbox adapters.

Import from here instead of copy-pasting into each adapter module.
"""
from __future__ import annotations


def tokenize(name: str) -> frozenset[str]:
    """
    Split a name on non-alphanumeric characters and return a lowercased frozenset.

    Examples::

        tokenize("ec2_terminate_instances") → frozenset({"ec2", "terminate", "instances"})
        tokenize("CODE_EXECUTE")            → frozenset({"code", "execute"})
        tokenize("s3.delete.object")        → frozenset({"s3", "delete", "object"})
    """
    parts: list[str] = []
    buf: list[str] = []
    for ch in name:
        if ch.isalnum():
            buf.append(ch.lower())
        elif buf:
            parts.append("".join(buf))
            buf.clear()
    if buf:
        parts.append("".join(buf))
    return frozenset(parts)
