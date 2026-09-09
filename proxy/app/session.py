"""
Session tracking: correlates multiple LLM calls into a single agent session.

Priority for session resolution:
1. Explicit X-Aegivis-Session-ID header -> use as-is
2. X-Aegivis-Agent-ID header only -> auto-generate session ID per conversation
3. Auto-detect from conversation structure:
   - New session: messages = [system?, user (first message)] with no prior assistant turns
   - Continuation: messages contain prior assistant messages -> same session
   - Key: SHA-256(first_user_message_content) as session anchor

Persistence:
  On shutdown, session state is written to a JSON file (aegivis_sessions.json by default).
  On startup, it is reloaded so that long multi-call sessions survive proxy restarts.
  Sessions idle for > TTL (4 hours) are evicted from memory and not persisted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

from .hash_chain import genesis_hash

logger = logging.getLogger(__name__)


def _make_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def _first_user_hash(messages: list[dict]) -> str | None:
    """Return a short SHA-256 of the first user message content, or None."""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            content = msg["content"]
            if isinstance(content, str):
                return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            elif isinstance(content, list):
                # OpenAI vision format: content is list of parts
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                combined = "".join(text_parts)
                if combined:
                    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
    return None


def _has_prior_assistant_turns(messages: list[dict]) -> bool:
    """True if conversation already has assistant responses (continuation)."""
    return any(msg.get("role") == "assistant" for msg in messages)


class SessionState:
    """Mutable per-session state held in memory by the proxy.

    Slots are divided into two groups:
    • **Persisted** — written to ``aegivis_sessions.json`` via ``to_dict()`` so that long
      sessions survive proxy restarts.  The canonical set is ``SessionState.PERSISTED_SLOTS``.
    • **Runtime-only** — ephemeral; re-initialised to zero/None/empty on every proxy start
      (or on session reload via ``from_dict()``).  Includes lazy-init objects that hold
      in-memory graphs and classifiers (TaintTracker, SessionPDG, IFCContext).

    Adding a new persisted field requires updating *both* ``to_dict()`` and ``from_dict()``;
    the test ``test_session.py::test_to_dict_key_set`` will catch any mismatch.
    """

    # ── Persisted field names (must stay in sync with to_dict / from_dict) ─────
    PERSISTED_SLOTS: frozenset[str] = frozenset({
        "session_id", "agent_id", "sequence_number", "last_hash",
        "llm_call_count", "tool_call_count", "pending_tool_calls",
        "started_at_ns", "last_seen_ns", "first_user_hash",
        "max_injection_score", "total_tokens", "error_count",
        "system_prompt_hash",
        "parent_agent_id", "parent_session_id", "spawn_depth",
    })

    __slots__ = (
        # ── Persisted ─────────────────────────────────────────────────────────
        "session_id",
        "agent_id",
        "sequence_number",
        "last_hash",
        "llm_call_count",
        "tool_call_count",
        "pending_tool_calls",
        "started_at_ns",
        "last_seen_ns",
        "first_user_hash",
        "max_injection_score",      # float: max injection score seen this session
        "total_tokens",             # int: cumulative token usage this session
        "error_count",              # int: count of SYSTEM_ERROR events (Isolation Forest)
        "system_prompt_hash",       # str | None: SHA-256[:16] of first system prompt (Phase 6)
        "parent_agent_id",          # str | None: agent_id of spawning agent (Phase 6)
        "parent_session_id",        # str | None: session_id of spawning agent (Phase 6)
        "spawn_depth",              # int: 0 = root agent, N = Nth delegation level (Phase 6)
        # ── Runtime-only (not persisted; reset to defaults on proxy restart) ──
        "active_canaries",          # dict[run_id → canary_token]: cleared after each response
        "event_type_sequence",      # list[str]: event types for Markov model
        "injection_score_history",  # list[float]: rolling per-turn injection scores
        "ml_injection_flag",        # bool: async ML classifier detected injection in prev turn
        "ml_injection_score",       # float: ML classifier score that set the flag
        "taint_tracker",            # TaintTracker | None: credential taint store (Phase 8)
        "pdg",                      # SessionPDG | None: data-flow graph (Phase 10)
        "ifc_context",              # IFCContext | None: IFC label store (Phase 11)
        "trust_score",              # float: 0.0–1.0; degraded by violations (Phase 9C)
        "hitl_pending_approval_id", # str | None: HITL approval UUID (Phase 9)
        "tools_hash",               # str | None: SHA-256[:16] of tools[] from first call
        "tools_set",                # frozenset[str] | None: tool names from first call
    )

    def __init__(
        self,
        session_id: str,
        agent_id: str,
        first_user_hash: str | None = None,
        *,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
        spawn_depth: int = 0,
    ):
        self.session_id = session_id
        self.agent_id = agent_id
        self.sequence_number = 0
        self.last_hash = genesis_hash(session_id)
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.pending_tool_calls: dict[str, dict] = {}
        self.started_at_ns = time.time_ns()
        self.last_seen_ns = self.started_at_ns
        self.first_user_hash = first_user_hash
        self.active_canaries: dict[str, str] = {}   # run_id -> canary token; not persisted
        self.event_type_sequence: list[str] = []    # Markov model; not persisted
        self.max_injection_score: float = 0.0       # persisted; updated on each scan
        self.injection_score_history: list[float] = []  # rolling per-turn scores; not persisted
        self.ml_injection_flag: bool = False         # Phase 4; async ML classifier; not persisted
        self.ml_injection_score: float = 0.0        # Phase 4; score that set the flag; not persisted
        self.total_tokens: int = 0                  # persisted; cumulative token usage
        self.error_count: int = 0                   # persisted; SYSTEM_ERROR count for IF features
        self.system_prompt_hash: str | None = None  # Phase 6; first system prompt hash; persisted
        self.parent_agent_id: str | None = parent_agent_id   # Phase 6; spawn chain
        self.parent_session_id: str | None = parent_session_id
        self.spawn_depth: int = spawn_depth
        self.taint_tracker = None                            # Phase 8; lazy-init TaintTracker; not persisted
        self.pdg = None                                      # Phase 10; lazy-init SessionPDG; not persisted
        self.ifc_context = None                              # Phase 11; lazy-init IFCContext; not persisted
        self.trust_score: float = 1.0                       # Phase 9C; starts fully trusted; not persisted
        self.hitl_pending_approval_id: str | None = None   # Phase 9; HITL approval UUID; not persisted
        self.tools_hash: str | None = None                  # tool baseline; hash of tools[] from first call
        self.tools_set: frozenset[str] | None = None        # tool baseline; names from first call

    def to_dict(self) -> dict:
        return {
            "session_id":         self.session_id,
            "agent_id":           self.agent_id,
            "sequence_number":    self.sequence_number,
            "last_hash":          self.last_hash,
            "llm_call_count":     self.llm_call_count,
            "tool_call_count":    self.tool_call_count,
            "pending_tool_calls": self.pending_tool_calls,
            "started_at_ns":      self.started_at_ns,
            "last_seen_ns":       self.last_seen_ns,
            "first_user_hash":    self.first_user_hash,
            "max_injection_score": self.max_injection_score,
            "total_tokens":        self.total_tokens,
            "error_count":         self.error_count,
            "system_prompt_hash":  self.system_prompt_hash,
            "parent_agent_id":     self.parent_agent_id,
            "parent_session_id":   self.parent_session_id,
            "spawn_depth":         self.spawn_depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        obj = cls.__new__(cls)
        obj.session_id        = data["session_id"]
        obj.agent_id          = data.get("agent_id", "unknown")
        obj.sequence_number   = data.get("sequence_number", 0)
        obj.last_hash         = data.get("last_hash", genesis_hash(data["session_id"]))
        obj.llm_call_count    = data.get("llm_call_count", 0)
        obj.tool_call_count   = data.get("tool_call_count", 0)
        obj.pending_tool_calls = data.get("pending_tool_calls", {})
        obj.started_at_ns     = data.get("started_at_ns", time.time_ns())
        obj.last_seen_ns      = data.get("last_seen_ns", time.time_ns())
        obj.first_user_hash   = data.get("first_user_hash")
        obj.active_canaries         = {}     # not persisted; cleared per response
        obj.event_type_sequence     = []     # not persisted; Markov model resets on restart
        obj.max_injection_score     = data.get("max_injection_score", 0.0)
        obj.injection_score_history = []     # not persisted; rolling per-turn scores
        obj.ml_injection_flag       = False  # not persisted; Phase 4 ML classifier flag
        obj.ml_injection_score      = 0.0   # not persisted; Phase 4 ML classifier score
        obj.total_tokens            = data.get("total_tokens", 0)
        obj.error_count             = data.get("error_count", 0)
        obj.system_prompt_hash      = data.get("system_prompt_hash")
        obj.parent_agent_id         = data.get("parent_agent_id")
        obj.parent_session_id       = data.get("parent_session_id")
        obj.spawn_depth             = data.get("spawn_depth", 0)
        obj.taint_tracker               = None   # not persisted; recreated lazily
        obj.pdg                         = None   # not persisted; recreated lazily (Phase 10)
        obj.ifc_context                 = None  # not persisted; recreated lazily (Phase 11)
        obj.trust_score                 = 1.0   # not persisted; resets on proxy restart
        obj.hitl_pending_approval_id    = None  # not persisted; ephemeral per-call
        obj.tools_hash                  = None  # not persisted
        obj.tools_set                   = None  # not persisted
        return obj

    def get_taint_tracker(self) -> "TaintTracker":
        """Return the session's TaintTracker, creating it on first access."""
        if self.taint_tracker is None:
            from .security.taint_tracker import TaintTracker
            self.taint_tracker = TaintTracker()
        return self.taint_tracker

    def get_pdg(self) -> "SessionPDG":
        """Return the session's SessionPDG, creating it on first access."""
        if self.pdg is None:
            from .security.session_pdg import SessionPDG
            self.pdg = SessionPDG()
        return self.pdg

    def get_ifc_context(self) -> "IFCContext":
        """Return the session's IFCContext, creating it on first access."""
        if self.ifc_context is None:
            from .security.ifc_labels import IFCContext
            self.ifc_context = IFCContext()
        return self.ifc_context


class SessionTracker:
    """
    In-memory session registry. One instance per proxy process.

    Persistence: call save() on shutdown and load() on startup to survive restarts.
    Sessions idle > TTL are evicted automatically every 30 minutes (via lifespan loop).
    """

    _TTL_NS = 4 * 3600 * 1_000_000_000  # 4 hours idle -> evict

    def __init__(self, state_path: str = ""):
        self._sessions: dict[str, SessionState] = {}
        self._first_user_hash_index: dict[str, str] = {}  # first_user_hash -> session_id
        self._state_path = state_path

    def load(self):
        """
        Restore session state from disk (call once at startup).
        Silently skips if the file doesn't exist yet.
        """
        if not self._state_path:
            return
        path = Path(self._state_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions_raw = data.get("sessions", {})
            index_raw    = data.get("first_user_hash_index", {})

            loaded = 0
            now = time.time_ns()
            for session_id, sdata in sessions_raw.items():
                state = SessionState.from_dict(sdata)
                # Skip sessions that were stale even before we reloaded
                if now - state.last_seen_ns < self._TTL_NS:
                    self._sessions[session_id] = state
                    loaded += 1

            # Rebuild index (only for sessions that survived the TTL check)
            for fuh, sid in index_raw.items():
                if sid in self._sessions:
                    self._first_user_hash_index[fuh] = sid

            if loaded > 0:
                logger.info(f"SessionTracker: restored {loaded} sessions from {self._state_path!r}")
        except Exception as e:
            logger.warning(f"SessionTracker: could not load state from {self._state_path!r}: {e}")

    def save(self):
        """
        Persist active session state to disk (call on shutdown).
        Only non-stale sessions are saved.
        """
        if not self._state_path:
            return
        now = time.time_ns()
        sessions_to_save = {
            sid: state.to_dict()
            for sid, state in self._sessions.items()
            if now - state.last_seen_ns < self._TTL_NS
        }
        index_to_save = {
            fuh: sid
            for fuh, sid in self._first_user_hash_index.items()
            if sid in sessions_to_save
        }
        try:
            path = Path(self._state_path)
            path.write_text(
                json.dumps(
                    {"sessions": sessions_to_save, "first_user_hash_index": index_to_save},
                    default=str,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(f"SessionTracker: saved {len(sessions_to_save)} sessions to {self._state_path!r}")
        except Exception as e:
            logger.warning(f"SessionTracker: could not save state to {self._state_path!r}: {e}")

    def resolve_session(
        self,
        *,
        explicit_session_id: str | None,
        messages: list[dict],
        agent_id: str,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        """
        Determine the session_id for this LLM call.
        Returns existing session_id or creates a new one.
        """
        # Priority 1: explicit header
        if explicit_session_id:
            if explicit_session_id not in self._sessions:
                self._create_session(
                    explicit_session_id, agent_id, None,
                    parent_agent_id=parent_agent_id,
                    parent_session_id=parent_session_id,
                )
            self._sessions[explicit_session_id].last_seen_ns = time.time_ns()
            return explicit_session_id

        # Priority 2 & 3: auto-detect from conversation
        fuh = _first_user_hash(messages)
        has_prior = _has_prior_assistant_turns(messages)

        if has_prior and fuh and fuh in self._first_user_hash_index:
            # Continuation of existing session.
            # Tie-breaking: _first_user_hash_index maps fuh → session_id and stores
            # only one entry per fuh (the most recently created session for that hash).
            # Concurrent agents that start with the same first user message will
            # therefore share the last-created session, which is intentional —
            # the session represents a conversation identity, not a parallel run.
            session_id = self._first_user_hash_index[fuh]
            if session_id in self._sessions:
                self._sessions[session_id].last_seen_ns = time.time_ns()
                return session_id

        # New session
        session_id = _make_session_id()
        self._create_session(
            session_id, agent_id, fuh,
            parent_agent_id=parent_agent_id,
            parent_session_id=parent_session_id,
        )
        if fuh:
            # Register fuh → session_id; overwrites any prior mapping for this hash
            self._first_user_hash_index[fuh] = session_id
        return session_id

    def _create_session(
        self,
        session_id: str,
        agent_id: str,
        fuh: str | None,
        *,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
    ):
        # Compute spawn depth: parent's depth + 1, or 1 if parent declared but unknown
        spawn_depth = 0
        if parent_session_id and parent_session_id in self._sessions:
            spawn_depth = self._sessions[parent_session_id].spawn_depth + 1
        elif parent_agent_id:
            spawn_depth = 1  # parent declared but not tracked here yet

        state = SessionState(
            session_id, agent_id, fuh,
            parent_agent_id=parent_agent_id,
            parent_session_id=parent_session_id,
            spawn_depth=spawn_depth,
        )
        # Register with trust graph (Phase 9C) — inherit parent trust score
        try:
            from .trust.graph import get_trust_graph
            entry = get_trust_graph().register(
                session_id=session_id,
                agent_id=agent_id,
                parent_session_id=parent_session_id,
                parent_agent_id=parent_agent_id,
            )
            state.trust_score = entry.trust_score
        except Exception:
            pass  # trust graph is best-effort

        self._sessions[session_id] = state
        if parent_agent_id:
            logger.debug(
                f"New session: {session_id} (agent={agent_id}, "
                f"parent={parent_agent_id}, depth={spawn_depth})"
            )
        else:
            logger.debug(f"New session: {session_id} (agent={agent_id})")

    def get_state(self, session_id: str) -> "SessionState":
        """Return the live SessionState object (mutations persist)."""
        if session_id not in self._sessions:
            self._create_session(session_id, "unknown", None)
        return self._sessions[session_id]

    def evict_stale(self):
        """Remove sessions idle for more than TTL (call periodically)."""
        now = time.time_ns()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_seen_ns > self._TTL_NS
        ]
        for sid in stale:
            state = self._sessions.pop(sid, None)
            if state and state.first_user_hash:
                self._first_user_hash_index.pop(state.first_user_hash, None)
        if stale:
            logger.debug(f"Evicted {len(stale)} stale sessions")

    def active_count(self) -> int:
        return len(self._sessions)


# Singleton
_tracker: SessionTracker | None = None


def get_session_tracker() -> SessionTracker:
    """Return in-memory SessionTracker (default / fallback)."""
    global _tracker
    if _tracker is None:
        from .config import settings
        _tracker = SessionTracker(state_path=settings.session_state_path)
    return _tracker


async def get_tracker() -> "SessionTracker":
    """
    Return the best available tracker.

    If AEGIVIS_REDIS_URL is set:  RedisSessionTracker (state shared across proxies).
    Otherwise:                 in-memory SessionTracker (default).
    """
    from .config import settings
    if settings.redis_url:
        try:
            from .redis_session import get_redis_session_tracker
            return await get_redis_session_tracker()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Redis session tracker unavailable (%s) — using in-memory", e
            )
    return get_session_tracker()
