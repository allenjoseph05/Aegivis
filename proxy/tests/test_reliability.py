"""
Tests for reliability features:
  - LocalBuffer: SQLite disk buffer for events + violations
  - SessionTracker: save/load session state across restarts
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from proxy.app.buffer import LocalBuffer
from proxy.app.session import SessionState, SessionTracker


# ── LocalBuffer tests ─────────────────────────────────────────────────────────

class TestLocalBuffer:
    def _make_buf(self, max_events: int = 100) -> tuple[LocalBuffer, str]:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        return LocalBuffer(path, max_events), path

    def test_write_and_read_events(self):
        buf, _ = self._make_buf()
        events = [{"event_id": f"evt{i}", "payload": {"x": i}} for i in range(5)]
        buf.write_events(events)

        rows = buf.read_events(limit=10)
        assert len(rows) == 5
        ids   = [r[0] for r in rows]
        items = [r[1] for r in rows]
        assert items[0]["event_id"] == "evt0"
        assert items[4]["event_id"] == "evt4"

    def test_write_and_read_violations(self):
        buf, _ = self._make_buf()
        viol = [{"rule_name": "test", "action": "ALERT", "session_id": "s1"}]
        buf.write_violations(viol)

        rows = buf.read_violations(limit=10)
        assert len(rows) == 1
        assert rows[0][1]["rule_name"] == "test"

    def test_delete_removes_rows(self):
        buf, _ = self._make_buf()
        buf.write_events([{"event_id": "a"}, {"event_id": "b"}])
        rows = buf.read_events(limit=10)
        assert len(rows) == 2

        buf.delete([rows[0][0]])  # delete first row by ID
        remaining = buf.read_events(limit=10)
        assert len(remaining) == 1
        assert remaining[0][1]["event_id"] == "b"

    def test_events_and_violations_are_separate(self):
        buf, _ = self._make_buf()
        buf.write_events([{"event_id": "e1"}])
        buf.write_violations([{"rule_name": "r1"}])

        ev_rows = buf.read_events(limit=10)
        viol_rows = buf.read_violations(limit=10)
        assert len(ev_rows) == 1
        assert len(viol_rows) == 1
        assert ev_rows[0][1]["event_id"] == "e1"
        assert viol_rows[0][1]["rule_name"] == "r1"

    def test_max_events_evicts_oldest(self):
        buf, _ = self._make_buf(max_events=5)
        # Write 8 events — should keep only the 5 newest
        buf.write_events([{"event_id": f"e{i}"} for i in range(8)])
        rows = buf.read_events(limit=20)
        assert len(rows) == 5
        # Oldest (e0, e1, e2) should be gone; newest (e3-e7) should remain
        ids_in_buf = [r[1]["event_id"] for r in rows]
        assert "e0" not in ids_in_buf
        assert "e7" in ids_in_buf

    def test_size_returns_correct_counts(self):
        buf, _ = self._make_buf()
        assert buf.size() == {"events": 0, "violations": 0}
        buf.write_events([{"e": 1}, {"e": 2}])
        buf.write_violations([{"v": 1}])
        assert buf.size() == {"events": 2, "violations": 1}

    def test_empty_write_is_noop(self):
        buf, _ = self._make_buf()
        buf.write_events([])
        buf.write_violations([])
        assert buf.size() == {"events": 0, "violations": 0}

    def test_delete_empty_list_is_noop(self):
        buf, _ = self._make_buf()
        buf.write_events([{"e": 1}])
        buf.delete([])   # no-op
        assert buf.size()["events"] == 1

    def test_persists_across_instances(self):
        """Data written by one LocalBuffer instance is readable by another on same file."""
        buf1, path = self._make_buf()
        buf1.write_events([{"event_id": "persistent"}])

        buf2 = LocalBuffer(path, 100)
        rows = buf2.read_events(limit=10)
        assert len(rows) == 1
        assert rows[0][1]["event_id"] == "persistent"


# ── SessionTracker persistence tests ─────────────────────────────────────────

def _make_session_state(session_id: str, agent_id: str = "test-agent") -> SessionState:
    state = SessionState(session_id, agent_id, first_user_hash="abc123")
    state.sequence_number = 5
    state.llm_call_count = 3
    state.tool_call_count = 2
    state.last_hash = "deadbeef" * 8
    return state


class TestSessionTrackerPersistence:
    def _tracker_with_file(self) -> tuple[SessionTracker, str]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        Path(path).unlink()   # file must not exist initially
        return SessionTracker(state_path=path), path

    def test_save_creates_file(self):
        tracker, path = self._tracker_with_file()
        tracker._sessions["sess_abc"] = _make_session_state("sess_abc")
        tracker._first_user_hash_index["abc123"] = "sess_abc"
        tracker.save()
        assert Path(path).exists()

    def test_load_restores_sessions(self):
        tracker1, path = self._tracker_with_file()
        tracker1._sessions["sess_abc"] = _make_session_state("sess_abc")
        tracker1._first_user_hash_index["abc123"] = "sess_abc"
        tracker1.save()

        tracker2 = SessionTracker(state_path=path)
        tracker2.load()

        assert "sess_abc" in tracker2._sessions
        state = tracker2._sessions["sess_abc"]
        assert state.sequence_number == 5
        assert state.llm_call_count == 3
        assert state.tool_call_count == 2
        assert state.last_hash == "deadbeef" * 8
        assert state.first_user_hash == "abc123"

    def test_load_restores_first_user_hash_index(self):
        tracker1, path = self._tracker_with_file()
        tracker1._sessions["sess_abc"] = _make_session_state("sess_abc")
        tracker1._first_user_hash_index["abc123"] = "sess_abc"
        tracker1.save()

        tracker2 = SessionTracker(state_path=path)
        tracker2.load()
        assert tracker2._first_user_hash_index.get("abc123") == "sess_abc"

    def test_stale_sessions_not_saved(self):
        tracker, path = self._tracker_with_file()
        state = _make_session_state("sess_old")
        # Simulate a session that was last seen 5 hours ago (> 4hr TTL)
        state.last_seen_ns = time.time_ns() - (5 * 3600 * 1_000_000_000)
        tracker._sessions["sess_old"] = state
        tracker.save()

        tracker2 = SessionTracker(state_path=path)
        tracker2.load()
        assert "sess_old" not in tracker2._sessions

    def test_stale_sessions_not_loaded(self):
        tracker1, path = self._tracker_with_file()
        state = _make_session_state("sess_old")
        # Make session just barely non-stale for saving
        tracker1._sessions["sess_old"] = state
        tracker1.save()

        # Manually corrupt the saved file so last_seen_ns is old
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["sessions"]["sess_old"]["last_seen_ns"] = (
            time.time_ns() - (5 * 3600 * 1_000_000_000)
        )
        Path(path).write_text(json.dumps(raw), encoding="utf-8")

        tracker2 = SessionTracker(state_path=path)
        tracker2.load()
        assert "sess_old" not in tracker2._sessions

    def test_load_nonexistent_file_is_noop(self):
        tracker = SessionTracker(state_path="/nonexistent/path/sessions.json")
        tracker.load()   # should not raise
        assert tracker.active_count() == 0

    def test_load_with_empty_path_is_noop(self):
        tracker = SessionTracker(state_path="")
        tracker.load()
        tracker.save()   # both are no-ops
        assert tracker.active_count() == 0

    def test_active_count(self):
        tracker = SessionTracker(state_path="")
        assert tracker.active_count() == 0
        tracker._sessions["s1"] = _make_session_state("s1")
        tracker._sessions["s2"] = _make_session_state("s2")
        assert tracker.active_count() == 2

    def test_session_continuity_after_reload(self):
        """A reloaded session should continue the hash chain exactly where it left off."""
        tracker1, path = self._tracker_with_file()

        # Simulate a session with known hash state
        state = _make_session_state("sess_chain")
        state.sequence_number = 10
        state.last_hash = "aabbccdd" * 8
        tracker1._sessions["sess_chain"] = state
        tracker1.save()

        tracker2 = SessionTracker(state_path=path)
        tracker2.load()

        reloaded = tracker2.get_state("sess_chain")
        assert reloaded.sequence_number == 10
        assert reloaded.last_hash == "aabbccdd" * 8
