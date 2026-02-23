"""
Hash chain verification service.

Verifies the integrity of a session's event chain:
1. Recomputes each event's hash
2. Checks that previous_hash links are intact
3. Returns a detailed verification report
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _canonical_json(obj: dict, exclude_keys: set[str] | None = None) -> bytes:
    if exclude_keys:
        obj = {k: v for k, v in obj.items() if k not in exclude_keys}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def recompute_hash(event: dict) -> str:
    """Recompute the SHA-256 hash of an event (excluding current_hash field)."""
    serialized = _canonical_json(event, exclude_keys={"current_hash", "received_at"})
    return hashlib.sha256(serialized).hexdigest()


@dataclass
class VerificationResult:
    session_id: str
    valid: bool
    total_events: int
    first_failed_sequence: int | None
    error_message: str | None
    checked_at: str

    def to_dict(self) -> dict:
        import datetime
        return {
            "session_id": self.session_id,
            "valid": self.valid,
            "total_events": self.total_events,
            "first_failed_sequence": self.first_failed_sequence,
            "error_message": self.error_message,
            "checked_at": self.checked_at,
        }


def verify_session_chain(session_id: str, events: list[dict]) -> VerificationResult:
    """
    Verify the hash chain integrity for a session.

    Args:
        session_id: The session being verified
        events: List of event dicts ordered by sequence_number

    Returns:
        VerificationResult with valid=True if chain is intact
    """
    from datetime import datetime, timezone

    checked_at = datetime.now(timezone.utc).isoformat()

    if not events:
        return VerificationResult(
            session_id=session_id,
            valid=True,
            total_events=0,
            first_failed_sequence=None,
            error_message=None,
            checked_at=checked_at,
        )

    genesis = f"GENESIS_{session_id}"

    for i, event in enumerate(events):
        seq = event.get("sequence_number", i)
        stored_hash = event.get("current_hash")

        # Recompute hash
        computed = recompute_hash(event)
        if computed != stored_hash:
            return VerificationResult(
                session_id=session_id,
                valid=False,
                total_events=len(events),
                first_failed_sequence=seq,
                error_message=(
                    f"Hash mismatch at sequence {seq}: "
                    f"stored={stored_hash[:16] if stored_hash else 'null'}… "
                    f"computed={computed[:16]}…"
                ),
                checked_at=checked_at,
            )

        # Verify chain linkage
        expected_prev = genesis if i == 0 else events[i - 1].get("current_hash")
        actual_prev = event.get("previous_hash")
        if actual_prev != expected_prev:
            return VerificationResult(
                session_id=session_id,
                valid=False,
                total_events=len(events),
                first_failed_sequence=seq,
                error_message=(
                    f"Chain break at sequence {seq}: "
                    f"previous_hash={actual_prev[:16] if actual_prev else 'null'}… "
                    f"expected={expected_prev[:16] if expected_prev else 'null'}…"
                ),
                checked_at=checked_at,
            )

    return VerificationResult(
        session_id=session_id,
        valid=True,
        total_events=len(events),
        first_failed_sequence=None,
        error_message=None,
        checked_at=checked_at,
    )
