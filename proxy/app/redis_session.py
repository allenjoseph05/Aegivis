"""
Redis-backed session tracker.

Provides the same public API as SessionTracker (session.py) but stores all
persistent session fields in Redis instead of a process-local dict.

Key schema
----------
    aegivis:sess:{session_id}         — JSON blob (all persistent SessionState fields)
    aegivis:idx:fuh:{first_user_hash} — string (session_id lookup by first-user hash)

Both keys carry a 4-hour TTL (refreshed on every write).

Ephemeral fields
----------------
The following SessionState slots are NOT stored in Redis because they are
meaningless across proxy restarts / replicas:
    active_canaries, event_type_sequence, ml_injection_flag, ml_injection_score,
    injection_score_history

They are kept in a process-local dict keyed by session_id, as before.

Fallback
--------
If the Redis connection is unavailable the tracker logs a warning and falls
back to an in-memory dict so the proxy can still function (without cross-
process sharing).
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from .session import SessionState, SessionTracker, _first_user_hash, _has_prior_assistant_turns, _make_session_id
from .hash_chain import genesis_hash

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SESSION_TTL = 4 * 3600          # 4 hours in seconds
_KEY_PREFIX_SESS = "aegivis:sess:"
_KEY_PREFIX_FUH  = "aegivis:idx:fuh:"


class RedisSessionTracker:
    """
    Redis-backed drop-in replacement for SessionTracker.

    Shares the same public interface so intercept.py needs no changes.
    """

    def __init__(self, redis_client):  # type: ignore[annotation]
        self._redis = redis_client
        # Process-local ephemeral state (not shared across proxies)
        self._ephemeral: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API (matches SessionTracker)
    # ------------------------------------------------------------------

    def resolve_session(
        self,
        *,
        explicit_session_id: str | None,
        messages: list[dict],
        agent_id: str,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        """Synchronous wrapper — delegates to the async implementation via run."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Called from within an async context — create a task.
                # The caller (intercept.py) uses `await` via the async variant.
                raise RuntimeError("use resolve_session_async in async context")
            return loop.run_until_complete(
                self.resolve_session_async(
                    explicit_session_id=explicit_session_id,
                    messages=messages,
                    agent_id=agent_id,
                    parent_agent_id=parent_agent_id,
                    parent_session_id=parent_session_id,
                )
            )
        except Exception as e:
            logger.warning("RedisSessionTracker.resolve_session failed: %s — falling back", e)
            return _make_session_id()

    async def resolve_session_async(
        self,
        *,
        explicit_session_id: str | None,
        messages: list[dict],
        agent_id: str,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        """Async version — called directly from intercept.py coroutines."""
        try:
            # Priority 1: explicit header
            if explicit_session_id:
                if not await self._exists(explicit_session_id):
                    await self._create_session(
                        explicit_session_id, agent_id, None,
                        parent_agent_id=parent_agent_id,
                        parent_session_id=parent_session_id,
                    )
                else:
                    await self._touch(explicit_session_id)
                return explicit_session_id

            # Priority 2 & 3: auto-detect from conversation
            fuh = _first_user_hash(messages)
            has_prior = _has_prior_assistant_turns(messages)

            if has_prior and fuh:
                existing = await self._redis.get(f"{_KEY_PREFIX_FUH}{fuh}")
                if existing:
                    session_id = existing.decode() if isinstance(existing, bytes) else existing
                    if await self._exists(session_id):
                        await self._touch(session_id)
                        return session_id

            # New session
            session_id = _make_session_id()
            await self._create_session(
                session_id, agent_id, fuh,
                parent_agent_id=parent_agent_id,
                parent_session_id=parent_session_id,
            )
            if fuh:
                await self._redis.setex(
                    f"{_KEY_PREFIX_FUH}{fuh}", _SESSION_TTL, session_id
                )
            return session_id
        except Exception as e:
            logger.warning("Redis resolve_session_async failed: %s — creating local session", e)
            return _make_session_id()

    def get_state(self, session_id: str) -> SessionState:
        """Synchronous wrapper — loads from Redis synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("use get_state_async in async context")
            return loop.run_until_complete(self.get_state_async(session_id))
        except Exception as e:
            logger.warning("RedisSessionTracker.get_state failed: %s", e)
            return self._make_empty_state(session_id)

    async def get_state_async(self, session_id: str) -> SessionState:
        """Load SessionState from Redis; create if missing."""
        try:
            raw = await self._redis.get(f"{_KEY_PREFIX_SESS}{session_id}")
            if raw:
                data = json.loads(raw)
                state = SessionState.from_dict(data)
            else:
                state = self._make_empty_state(session_id)
                await self._save_state(state)

            # Merge ephemeral fields from process-local dict
            eph = self._ephemeral.get(session_id, {})
            state.active_canaries        = eph.get("active_canaries", {})
            state.event_type_sequence    = eph.get("event_type_sequence", [])
            state.ml_injection_flag      = eph.get("ml_injection_flag", False)
            state.ml_injection_score     = eph.get("ml_injection_score", 0.0)
            state.injection_score_history = eph.get("injection_score_history", [])
            return state
        except Exception as e:
            logger.warning("get_state_async failed for %s: %s", session_id, e)
            return self._make_empty_state(session_id)

    async def save_state_async(self, state: SessionState) -> None:
        """Persist updated state back to Redis and save ephemeral fields locally."""
        try:
            await self._save_state(state)
            # Update ephemeral process-local dict
            self._ephemeral[state.session_id] = {
                "active_canaries":        state.active_canaries,
                "event_type_sequence":    state.event_type_sequence,
                "ml_injection_flag":      state.ml_injection_flag,
                "ml_injection_score":     state.ml_injection_score,
                "injection_score_history": state.injection_score_history,
            }
        except Exception as e:
            logger.warning("save_state_async failed for %s: %s", state.session_id, e)

    def evict_stale(self):
        """No-op: Redis TTL handles eviction automatically."""

    def active_count(self) -> int:
        """Approximate count via SCAN (for /health only — not exact)."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(self._count_keys())
        except Exception:
            pass
        return -1

    async def _count_keys(self) -> int:
        count = 0
        async for _ in self._redis.scan_iter(f"{_KEY_PREFIX_SESS}*"):
            count += 1
        return count

    # Compatibility stubs
    def load(self): pass
    def save(self): pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _exists(self, session_id: str) -> bool:
        return bool(await self._redis.exists(f"{_KEY_PREFIX_SESS}{session_id}"))

    async def _touch(self, session_id: str) -> None:
        await self._redis.expire(f"{_KEY_PREFIX_SESS}{session_id}", _SESSION_TTL)

    async def _create_session(
        self,
        session_id: str,
        agent_id: str,
        fuh: str | None,
        *,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        spawn_depth = 0
        if parent_session_id:
            try:
                raw = await self._redis.get(f"{_KEY_PREFIX_SESS}{parent_session_id}")
                if raw:
                    parent_data = json.loads(raw)
                    spawn_depth = parent_data.get("spawn_depth", 0) + 1
            except Exception:
                pass
        elif parent_agent_id:
            spawn_depth = 1

        state = SessionState(
            session_id, agent_id, fuh,
            parent_agent_id=parent_agent_id,
            parent_session_id=parent_session_id,
            spawn_depth=spawn_depth,
        )
        await self._save_state(state)

    async def _save_state(self, state: SessionState) -> None:
        data = state.to_dict()
        await self._redis.setex(
            f"{_KEY_PREFIX_SESS}{state.session_id}",
            _SESSION_TTL,
            json.dumps(data, default=str),
        )

    @staticmethod
    def _make_empty_state(session_id: str) -> SessionState:
        return SessionState(session_id, "unknown", None)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_redis_tracker: RedisSessionTracker | None = None
_redis_client = None


async def get_redis_client():
    """Get or create the shared async Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            from redis.asyncio import Redis as AsyncRedis
            from .config import settings
            _redis_client = AsyncRedis.from_url(
                settings.redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            # Verify connection
            await _redis_client.ping()
            logger.info("Redis session tracker connected: %s", settings.redis_url)
        except Exception as e:
            logger.error("Cannot connect to Redis (%s): %s", getattr(e, '__class__.__name__', ''), e)
            _redis_client = None
    return _redis_client


async def get_redis_session_tracker() -> "RedisSessionTracker | SessionTracker":
    """Return RedisSessionTracker if Redis is available, else in-memory tracker."""
    global _redis_tracker
    if _redis_tracker is None:
        client = await get_redis_client()
        if client:
            _redis_tracker = RedisSessionTracker(client)
        else:
            from .session import get_session_tracker
            return get_session_tracker()
    return _redis_tracker
