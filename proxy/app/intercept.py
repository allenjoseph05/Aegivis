"""
Core interception logic: orchestrates per-request event creation.

Flow for each LLM API call:
1. Parse incoming request (delegated to provider handler)
2. Detect/create session
3. Create LLM_CALL_START event → PII mask → hash-chain sign → enqueue
4. Infer TOOL_CALL_END events from tool result messages in request
5. Forward request to real LLM provider
6. Parse response (streaming-aware)
7. Create LLM_CALL_END event → PII mask → hash-chain sign → enqueue
8. Infer TOOL_CALL_START events from tool_calls in response
9. Emit AGENT_FINISH if finish_reason=stop and no tool_calls
10. Emit CHECKPOINT every 1000 events per session
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from .canonicalize import (
    make_agent_finish,
    make_checkpoint,
    make_llm_call_end,
    make_llm_call_start,
    make_system_error,
)
from .config import settings
from .hash_chain import compute_event_hash, genesis_hash, merkle_root
from .pii import mask_dict
from .reconstruct import (
    extract_tool_results_from_request,
    make_tool_call_end,
    make_tool_call_start,
)
from .policy import PolicyAction, PolicyViolation, get_policy_engine
from .session import SessionState, SessionTracker
from .transport import get_transport

logger = logging.getLogger(__name__)


class InterceptContext:
    """
    Stateless per-request context. Session state is held in SessionTracker.
    """

    def __init__(
        self,
        *,
        session_tracker: SessionTracker,
        org_id: str,
    ):
        self.session_tracker = session_tracker
        self.org_id = org_id
        self.transport = get_transport()
        self.policy_engine = get_policy_engine()

    def _sign_event(self, event: dict, previous_hash: str) -> dict:
        """Fill previous_hash and compute current_hash."""
        event["previous_hash"] = previous_hash
        event["current_hash"] = compute_event_hash(event)
        return event

    def _apply_pii(self, event: dict) -> dict:
        """Run Presidio on event payload in-place. Store original hash as payload_hash."""
        if not settings.pii_enabled:
            return event

        payload = event.get("payload", {})
        masked_payload, entity_types, original_hash = mask_dict(payload, settings.pii_language)
        event["payload"] = masked_payload
        event["payload_hash"] = original_hash
        event["pii_detected"] = entity_types
        return event

    def _evaluate_policy(
        self, event: dict, state: SessionState
    ) -> list[PolicyViolation]:
        """Run policy engine on an event. Enqueue violations to backend if enabled."""
        session_dict = {
            "tool_call_count": state.tool_call_count,
            "llm_call_count": state.llm_call_count,
            "started_at_ns": state.started_at_ns,
        }
        violations = self.policy_engine.evaluate(event, session_dict)
        for v in violations:
            if v.action in (PolicyAction.ALERT, PolicyAction.BLOCK):
                logger.warning(
                    f"[POLICY:{v.action.value}] rule={v.rule_name!r} "
                    f"session={v.session_id} agent={v.agent_id} "
                    f"event={v.event_type} reason={v.reason!r}"
                )
            if v.action == PolicyAction.ALERT and settings.violations_enabled:
                self.transport.enqueue_violation(v.to_dict())
        return violations

    async def process_request(
        self,
        *,
        request_data: dict,
        provider: str,
        model: str,
        agent_id: str,
        explicit_session_id: str | None,
    ) -> tuple[str, str, list[PolicyViolation]]:
        """
        Process an incoming LLM API request.

        Returns:
            (session_id, run_id, violations)
            Callers must check violations for BLOCK actions before forwarding.
        """
        messages = request_data.get("messages", [])
        tools = request_data.get("tools") or request_data.get("functions") or []

        # Resolve session — returns its ID
        session_id = self.session_tracker.resolve_session(
            explicit_session_id=explicit_session_id,
            messages=messages,
            agent_id=agent_id,
        )

        # get_state() returns the live SessionState object — mutations persist
        state: SessionState = self.session_tracker.get_state(session_id)
        seq = state.sequence_number
        prev_hash = state.last_hash

        # --- Infer TOOL_CALL_END from tool result messages ---
        tool_results = extract_tool_results_from_request(messages)
        for tr in tool_results:
            pending = state.pending_tool_calls.pop(tr["tool_call_id"], None)
            if pending:
                tool_end = make_tool_call_end(
                    session_id=session_id,
                    org_id=self.org_id,
                    agent_id=agent_id,
                    provider=provider,
                    model=model,
                    parent_run_id=pending.get("parent_run_id", ""),
                    tool_call_id=tr["tool_call_id"],
                    tool_name=pending["tool_name"],
                    tool_output=tr["content"],
                    sequence_number=seq,
                    previous_hash=prev_hash,
                )
                tool_end = self._apply_pii(tool_end)
                tool_end = self._sign_event(tool_end, prev_hash)
                seq += 1
                prev_hash = tool_end["current_hash"]
                self.transport.enqueue(tool_end)

        # --- LLM_CALL_START ---
        run_id = str(uuid.uuid4())

        start_event = make_llm_call_start(
            session_id=session_id,
            org_id=self.org_id,
            agent_id=agent_id,
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            temperature=request_data.get("temperature"),
            max_tokens=request_data.get("max_tokens"),
            stream=request_data.get("stream"),
            extra_params={
                k: v for k, v in request_data.items()
                if k not in {"messages", "tools", "functions", "temperature", "max_tokens", "stream", "model"}
            } or None,
            run_id=run_id,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        start_event = self._apply_pii(start_event)
        start_event = self._sign_event(start_event, prev_hash)

        # Policy evaluation BEFORE forwarding — BLOCK fires here
        violations = self._evaluate_policy(start_event, state)
        block = [v for v in violations if v.action == PolicyAction.BLOCK]
        if block:
            # Return early — do NOT enqueue event and do NOT forward
            return session_id, run_id, block

        seq += 1
        prev_hash = start_event["current_hash"]

        # Persist updated state back onto the SessionState object
        state.sequence_number = seq
        state.last_hash = prev_hash
        state.llm_call_count += 1
        state.last_seen_ns = time.time_ns()

        self.transport.enqueue(start_event)

        return session_id, run_id, violations

    async def process_response(
        self,
        *,
        session_id: str,
        run_id: str,
        provider: str,
        model: str,
        agent_id: str,
        response_data: dict,
        latency_ms: float,
        http_status: int,
    ) -> None:
        """
        Process a complete LLM API response (assembled from SSE chunks if streaming).
        """
        state: SessionState = self.session_tracker.get_state(session_id)
        seq = state.sequence_number
        prev_hash = state.last_hash

        if http_status >= 400:
            error_event = make_system_error(
                session_id=session_id,
                org_id=self.org_id,
                agent_id=agent_id,
                provider=provider,
                model=model,
                run_id=run_id,
                error_message=str(response_data),
                http_status=http_status,
                sequence_number=seq,
                previous_hash=prev_hash,
            )
            error_event = self._sign_event(error_event, prev_hash)
            seq += 1
            prev_hash = error_event["current_hash"]
            state.sequence_number = seq
            state.last_hash = prev_hash
            state.last_seen_ns = time.time_ns()
            self.transport.enqueue(error_event)
            return

        response_text = response_data.get("response_text")
        finish_reason = response_data.get("finish_reason")
        tool_calls = response_data.get("tool_calls", [])
        token_usage = response_data.get("token_usage")

        end_event = make_llm_call_end(
            session_id=session_id,
            org_id=self.org_id,
            agent_id=agent_id,
            provider=provider,
            model=model,
            run_id=run_id,
            response_text=response_text,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            token_usage=token_usage,
            latency_ms=latency_ms,
            http_status=http_status,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        end_event = self._apply_pii(end_event)
        end_event = self._sign_event(end_event, prev_hash)

        seq += 1
        prev_hash = end_event["current_hash"]
        self.transport.enqueue(end_event)

        # --- Infer TOOL_CALL_START from tool_calls in response ---
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
            tc_args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})

            # Parse arguments if string
            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except (json.JSONDecodeError, ValueError):
                    pass

            tool_start = make_tool_call_start(
                session_id=session_id,
                org_id=self.org_id,
                agent_id=agent_id,
                provider=provider,
                model=model,
                parent_run_id=run_id,
                tool_call_id=tc_id,
                tool_name=tc_name,
                tool_input=tc_args,
                tool_description=None,
                sequence_number=seq,
                previous_hash=prev_hash,
            )
            tool_start = self._apply_pii(tool_start)
            tool_start = self._sign_event(tool_start, prev_hash)

            # Increment tool_call_count before policy eval so count-based rules fire correctly
            state.tool_call_count += 1
            tool_violations = self._evaluate_policy(tool_start, state)
            tool_block = [v for v in tool_violations if v.action == PolicyAction.BLOCK]
            if tool_block:
                logger.warning(
                    f"Tool call BLOCKED by policy: tool={tc_name} rule={tool_block[0].rule_name}"
                )
                # Skip this tool call — don't emit or track it
                continue

            seq += 1
            prev_hash = tool_start["current_hash"]

            # Track as pending (resolved when tool result arrives in next request)
            state.pending_tool_calls[tc_id] = {
                "tool_name": tc_name,
                "parent_run_id": run_id,
            }

            self.transport.enqueue(tool_start)

        # --- AGENT_FINISH if no tool calls and finish_reason=stop ---
        if finish_reason in ("stop", "end_turn", "STOP") and not tool_calls:
            duration_ms = (time.time_ns() - state.started_at_ns) / 1_000_000

            finish_event = make_agent_finish(
                session_id=session_id,
                org_id=self.org_id,
                agent_id=agent_id,
                provider=provider,
                model=model,
                run_id=run_id,
                final_output=response_text,
                total_llm_calls=state.llm_call_count,
                total_tool_calls=state.tool_call_count,
                session_duration_ms=duration_ms,
                sequence_number=seq,
                previous_hash=prev_hash,
            )
            finish_event = self._sign_event(finish_event, prev_hash)
            seq += 1
            prev_hash = finish_event["current_hash"]
            self.transport.enqueue(finish_event)

        # --- CHECKPOINT every N events ---
        if seq > 0 and seq % settings.checkpoint_interval == 0:
            checkpoint_event = make_checkpoint(
                session_id=session_id,
                org_id=self.org_id,
                agent_id=agent_id,
                provider=provider,
                model=model,
                run_id=run_id,
                merkle_root="pending",  # backend computes actual Merkle root
                events_covered=settings.checkpoint_interval,
                from_sequence=max(0, seq - settings.checkpoint_interval),
                to_sequence=seq,
                sequence_number=seq,
                previous_hash=prev_hash,
            )
            checkpoint_event = self._sign_event(checkpoint_event, prev_hash)
            seq += 1
            prev_hash = checkpoint_event["current_hash"]
            self.transport.enqueue(checkpoint_event)

        # Persist final state
        state.sequence_number = seq
        state.last_hash = prev_hash
        state.last_seen_ns = time.time_ns()
