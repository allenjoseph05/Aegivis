"""
Local disk buffer for audit events and violations.

When the backend is unreachable, events are written to a local SQLite database
instead of being dropped. A background retry loop sends them when the backend
recovers.

Design:
- SQLite is stdlib (no extra dependency)
- Append-only inserts; rows deleted only after confirmed send
- Thread-safe: SQLite WAL mode + connection-per-operation
- Max size enforced: oldest events evicted when buffer_max_events exceeded
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalBuffer:
    """
    SQLite-backed disk buffer for unsent events and violations.

    Schema:
        buffered_events(id, kind, payload_json, buffered_at)
        kind: "event" | "violation"
    """

    def __init__(self, db_path: str, max_events: int = 50_000):
        self._path = db_path
        self._max = max_events
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS buffered_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind         TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    buffered_at  REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kind ON buffered_events(kind)")

    def write_events(self, events: list[dict]):
        """Buffer a list of audit events to disk."""
        if not events:
            return
        self._write_rows("event", events)

    def write_violations(self, violations: list[dict]):
        """Buffer a list of policy violations to disk."""
        if not violations:
            return
        self._write_rows("violation", violations)

    def _write_rows(self, kind: str, items: list[dict]):
        now = time.time()
        rows = [(kind, json.dumps(item, default=str), now) for item in items]
        try:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT INTO buffered_events (kind, payload_json, buffered_at) VALUES (?,?,?)",
                    rows,
                )
                # Evict oldest rows if over limit
                total = conn.execute("SELECT COUNT(*) FROM buffered_events").fetchone()[0]
                if total > self._max:
                    excess = total - self._max
                    conn.execute(
                        "DELETE FROM buffered_events WHERE id IN "
                        "(SELECT id FROM buffered_events ORDER BY id ASC LIMIT ?)",
                        (excess,),
                    )
                    logger.warning(
                        f"LocalBuffer: evicted {excess} old events (buffer at max {self._max})"
                    )
        except Exception as e:
            logger.error(f"LocalBuffer: failed to write {kind}s: {e}")

    def read_events(self, limit: int = 200) -> list[tuple[int, dict]]:
        """Return up to `limit` buffered events as (row_id, event_dict) pairs."""
        return self._read_rows("event", limit)

    def read_violations(self, limit: int = 200) -> list[tuple[int, dict]]:
        """Return up to `limit` buffered violations as (row_id, violation_dict) pairs."""
        return self._read_rows("violation", limit)

    def _read_rows(self, kind: str, limit: int) -> list[tuple[int, dict]]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT id, payload_json FROM buffered_events "
                    "WHERE kind=? ORDER BY id ASC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            return [(row_id, json.loads(payload)) for row_id, payload in rows]
        except Exception as e:
            logger.error(f"LocalBuffer: read failed: {e}")
            return []

    def delete(self, row_ids: list[int]):
        """Delete rows by ID (call after successful send)."""
        if not row_ids:
            return
        try:
            placeholders = ",".join("?" * len(row_ids))
            with self._conn() as conn:
                conn.execute(
                    f"DELETE FROM buffered_events WHERE id IN ({placeholders})",
                    row_ids,
                )
        except Exception as e:
            logger.error(f"LocalBuffer: delete failed: {e}")

    def size(self) -> dict[str, int]:
        """Return count of buffered events and violations."""
        try:
            with self._conn() as conn:
                ev = conn.execute(
                    "SELECT COUNT(*) FROM buffered_events WHERE kind='event'"
                ).fetchone()[0]
                viol = conn.execute(
                    "SELECT COUNT(*) FROM buffered_events WHERE kind='violation'"
                ).fetchone()[0]
            return {"events": ev, "violations": viol}
        except Exception:
            return {"events": 0, "violations": 0}


# Singleton
_buffer: LocalBuffer | None = None


def get_buffer(db_path: str = "", max_events: int = 50_000) -> LocalBuffer | None:
    """
    Return the LocalBuffer singleton, or None if buffering is disabled
    (db_path is empty string).
    """
    global _buffer
    if not db_path:
        return None
    if _buffer is None:
        _buffer = LocalBuffer(db_path, max_events)
        logger.info(f"LocalBuffer initialized at {db_path!r}")
    return _buffer
