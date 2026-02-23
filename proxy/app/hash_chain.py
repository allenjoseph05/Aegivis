"""
SHA-256 hash chain implementation for AgentBlackBox.

Each event is linked to the previous via its hash, forming a tamper-evident chain.
Any modification to a historical event causes all subsequent hashes to diverge.

Chain structure:
  event[0].previous_hash = "GENESIS_<session_id>"
  event[0].current_hash  = SHA-256(canonical_json(event[0], exclude=current_hash))
  event[n].previous_hash = event[n-1].current_hash
  event[n].current_hash  = SHA-256(canonical_json(event[n], exclude=current_hash))
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _canonical_json(obj: dict, exclude_keys: set[str] | None = None) -> bytes:
    """Produce deterministic, sorted JSON bytes suitable for hashing."""
    if exclude_keys:
        obj = {k: v for k, v in obj.items() if k not in exclude_keys}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_event_hash(event_dict: dict) -> str:
    """
    Compute SHA-256 hash of an event envelope, excluding the current_hash field.
    The event_dict must include all fields except current_hash.
    Returns hex digest string (64 chars).
    """
    serialized = _canonical_json(event_dict, exclude_keys={"current_hash"})
    return hashlib.sha256(serialized).hexdigest()


def genesis_hash(session_id: str) -> str:
    """Return the sentinel previous_hash for the first event in a session."""
    return f"GENESIS_{session_id}"


def verify_chain(events: list[dict]) -> tuple[bool, int | None, str | None]:
    """
    Verify integrity of an ordered event chain.

    Args:
        events: List of event dicts ordered by sequence_number, all from same session.

    Returns:
        (valid, failed_sequence_number, error_message)
        valid=True, None, None if chain is intact.
    """
    if not events:
        return True, None, None

    session_id = events[0].get("session_id", "unknown")

    for i, event in enumerate(events):
        seq = event.get("sequence_number", i)
        stored_hash = event.get("current_hash")

        # Recompute hash
        computed = compute_event_hash(event)
        if computed != stored_hash:
            return False, seq, f"Hash mismatch at sequence {seq}: stored={stored_hash[:16]}… computed={computed[:16]}…"

        # Verify previous_hash linkage
        if i == 0:
            expected_prev = genesis_hash(session_id)
        else:
            expected_prev = events[i - 1].get("current_hash")

        actual_prev = event.get("previous_hash")
        if actual_prev != expected_prev:
            return False, seq, (
                f"Chain break at sequence {seq}: "
                f"previous_hash={actual_prev[:16] if actual_prev else None}… "
                f"expected={expected_prev[:16] if expected_prev else None}…"
            )

    return True, None, None


def merkle_root(hashes: list[str]) -> str:
    """
    Compute Merkle root of a list of hex-encoded SHA-256 hashes.
    Used for CHECKPOINT events covering 1000 events.
    """
    if not hashes:
        return hashlib.sha256(b"empty").hexdigest()

    layer = [bytes.fromhex(h) for h in hashes]

    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last if odd
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]

    return layer[0].hex()
