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
import hashlib
import json
import logging
import time
import uuid
from typing import Any

import httpx

from .canonicalize import (
    make_agent_finish,
    make_checkpoint,
    make_llm_call_end,
    make_llm_call_start,
    make_system_error,
)
from .config import settings
from .hash_chain import compute_event_hash, genesis_hash, merkle_root
from .security.remote_config import SecurityConfig, get_security_config
from .pii import mask_dict
from .reconstruct import (
    extract_tool_results_from_request,
    make_tool_call_end,
    make_tool_call_start,
)
from .policy import PolicyAction, PolicyViolation, get_policy_engine
from .session import SessionState, SessionTracker
from .tool_permissions import get_tool_permissions_engine
from .transport import get_transport, EventTransport
from .security.tool_output_scanner import scan as scan_tool_output

logger = logging.getLogger(__name__)


# ── System prompt integrity helpers ───────────────────────────────────────────

def _extract_system_prompt(body: dict) -> str | None:
    """Extract system prompt text from an LLM request body."""
    # Anthropic: top-level "system" key (string or list of content blocks)
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        return system
    if isinstance(system, list):
        texts = [b.get("text", "") for b in system if isinstance(b, dict)]
        combined = " ".join(texts).strip()
        if combined:
            return combined
    # OpenAI: first message with role=system
    for msg in body.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict)]
                combined = " ".join(texts).strip()
                if combined:
                    return combined
    return None



class InterceptContext:
    """
    Per-request context. Holds resolved config, transport, and org identity.

    Not stateless — caches _cfg (security config for this org+agent pair),
    agent_id, and transport for the duration of a single request.
    Session state is separate, held in SessionTracker and persisted across requests.
    """

    def __init__(
        self,
        *,
        session_tracker: SessionTracker,
        org_id: str,
        transport: EventTransport | None = None,
    ):
        self.session_tracker = session_tracker
        self.org_id = org_id
        self.agent_id: str = ""
        self.transport = transport if transport is not None else get_transport()
        self.policy_engine = get_policy_engine()
        self._cfg: SecurityConfig | None = None   # loaded once per request via _load_cfg()

    def _sign_event(self, event: dict, previous_hash: str) -> dict:
        """Fill previous_hash and compute current_hash."""
        event["previous_hash"] = previous_hash
        event["current_hash"] = compute_event_hash(event)
        return event

    async def _load_cfg(self) -> SecurityConfig:
        """Fetch (cached) org+agent security config. Called once at the start of each request."""
        if self._cfg is None:
            self._cfg = await get_security_config(
                self.org_id or "default-org", self.agent_id
            )
        return self._cfg

    def _apply_pii(self, event: dict) -> dict:
        """Run Presidio on event payload in-place. Store original hash as payload_hash."""
        pii_on = self._cfg.pii_detection_enabled if self._cfg else settings.pii_enabled
        if not pii_on:
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
            if v.action in (PolicyAction.ALERT, PolicyAction.BLOCK) and settings.violations_enabled:
                self.transport.enqueue_violation(v.to_dict())
        return violations

    # ── Private guard methods ──────────────────────────────────────────────────

    def _check_spawn_depth(
        self,
        state: SessionState,
        session_id: str,
        agent_id: str,
        max_depth: int,
    ) -> list[PolicyViolation] | None:
        """Block agents spawned beyond max_depth. Returns violation list or None to continue."""
        if max_depth <= 0 or state.spawn_depth <= max_depth:
            return None
        logger.warning(
            "[SPAWN-DEPTH:BLOCK] depth=%d max=%d session=%s agent=%s parent_agent=%s",
            state.spawn_depth, max_depth, session_id, agent_id,
            state.parent_agent_id or "none",
        )
        sv = PolicyViolation(
            rule_name="spawn-depth-exceeded",
            action=PolicyAction.BLOCK,
            reason=(
                f"Agent spawn depth {state.spawn_depth} exceeds maximum "
                f"allowed depth of {max_depth}. "
                f"Parent agent: {state.parent_agent_id or 'unknown'}. "
                f"Check for runaway delegation or compromised orchestrator."
            ),
            event_type="LLM_CALL_START",
            session_id=session_id,
            agent_id=agent_id,
            org_id=self.org_id,
        )
        if settings.violations_enabled:
            self.transport.enqueue_violation(sv.to_dict())
        return [sv]

    def _check_ml_flag(
        self,
        state: SessionState,
        session_id: str,
        agent_id: str,
    ) -> list[PolicyViolation] | None:
        """Block if async ML classifier flagged injection in the previous turn.
        Consumes the flag so it fires at most once. Returns violation list or None."""
        if not state.ml_injection_flag:
            return None
        state.ml_injection_flag = False  # consume — only fires once per turn
        logger.warning(
            "[ML-CLASSIFIER:BLOCK] Multi-turn injection detected in previous turn: "
            "score=%.3f session=%s agent=%s",
            state.ml_injection_score, session_id, agent_id,
        )
        v = PolicyViolation(
            rule_name="ml-classifier-injection-block",
            action=PolicyAction.BLOCK,
            reason=(
                f"Async ML classifier detected prompt injection in previous turn: "
                f"score={state.ml_injection_score:.3f} "
                f"model={settings.analysis_classifier_model}"
            ),
            event_type="LLM_CALL_START",
            session_id=session_id,
            agent_id=agent_id,
            org_id=self.org_id,
        )
        if settings.violations_enabled:
            self.transport.enqueue_violation(v.to_dict())
        return [v]

    async def _check_hitl_pending(
        self,
        state: SessionState,
        session_id: str,
        agent_id: str,
    ) -> list[PolicyViolation] | None:
        """Poll backend for HITL approval decision. Returns violation list if denied/expired,
        or None to continue (approved or no pending approval)."""
        if not state.hitl_pending_approval_id:
            return None
        approval_id = state.hitl_pending_approval_id
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.hitl_timeout_seconds
        decision = "pending"
        logger.info(
            "[HITL] Waiting for approval id=%s session=%s agent=%s",
            approval_id, session_id, agent_id,
        )
        async with httpx.AsyncClient(timeout=5.0) as _hc:
            while loop.time() < deadline:
                try:
                    _gr = await _hc.get(
                        f"{settings.backend_url}/v1/approvals/{approval_id}",
                        headers={"X-API-Key": settings.backend_api_key},
                    )
                    if _gr.status_code == 200:
                        decision = _gr.json().get("status", "pending")
                        if decision in ("approved", "denied", "expired"):
                            break
                except Exception as _he:
                    logger.debug("[HITL] Poll error: %s", _he)
                await asyncio.sleep(2.0)
        state.hitl_pending_approval_id = None
        if decision == "approved":
            return None
        logger.warning(
            "[HITL] LLM call blocked: approval=%s decision=%s session=%s",
            approval_id, decision, session_id,
        )
        hv = PolicyViolation(
            rule_name="hitl-denied",
            action=PolicyAction.BLOCK,
            reason=(
                f"HITL approval {decision}: tool call approval was not granted "
                f"(approval_id={approval_id})"
            ),
            event_type="LLM_CALL_START",
            session_id=session_id,
            agent_id=agent_id,
            org_id=self.org_id,
        )
        if settings.violations_enabled:
            self.transport.enqueue_violation(hv.to_dict())
        return [hv]

    async def maybe_hold_response(
        self,
        session_id: str,
        agent_id: str,
        provider: str,
        model: str,
    ) -> tuple[bool, dict | None]:
        """
        Phase 19 — Response-Hold HITL: pre-execution gate for non-streaming responses.

        Called by main.py after process_response() returns, before the LLM
        response is forwarded to the agent.  If a HITL approval was created
        during process_response() AND response_hold_enabled is True, this method
        holds execution, polls the approvals endpoint, and returns a decision.

        Returns:
            (should_deny=False, None)        — continue; return response to agent.
            (should_deny=True,  denial_dict) — replace response with synthetic denial.

        The caller is responsible for constructing the FastAPI Response from
        ``denial_dict``; this keeps FastAPI out of the security module.

        Preconditions:
            - ``_load_cfg()`` must have been called earlier in the same request
              (it will be; process_request() always calls it first).
            - ``state.hitl_pending_approval_id`` is set iff an approval exists.
        """
        cfg = await self._load_cfg()
        if not cfg.response_hold_enabled:
            return False, None

        state = self.session_tracker.get_state(session_id)
        approval_id = state.hitl_pending_approval_id
        if not approval_id:
            return False, None

        from .security.response_hold import poll_hold_gate, synthesize_denial
        decision = await poll_hold_gate(
            approval_id=approval_id,
            timeout_s=cfg.response_hold_timeout_s,
            backend_url=settings.backend_url,
            api_key=settings.backend_api_key,
        )

        # Always clear the pending ID — polling is complete regardless of outcome.
        # The existing HITL gate in process_request() checks this slot; clearing
        # here prevents the next request from being blocked a second time.
        state.hitl_pending_approval_id = None

        if decision.approved:
            logger.info(
                "[RESPONSE-HOLD] Approved: id=%s session=%s latency=%.0fms",
                approval_id, session_id, decision.latency_ms,
            )
            return False, None

        logger.warning(
            "[RESPONSE-HOLD] Denied: id=%s decision=%s session=%s latency=%.0fms",
            approval_id, decision.decision, session_id, decision.latency_ms,
        )
        return True, synthesize_denial(provider, model)

    def _build_llm_call_start(
        self,
        *,
        session_id: str,
        agent_id: str,
        provider: str,
        model: str,
        messages: list[dict],
        tools: list[dict],
        request_data: dict,
        run_id: str,
        seq: int,
        prev_hash: str,
        state: SessionState,
    ) -> dict:
        """
        Build, PII-mask, and hash-chain-sign the initial LLM_CALL_START event dict.

        Enriches with spawn chain metadata when applicable.
        Does NOT run any security scans — callers should apply those after receiving
        the returned event and before enqueuing it.
        """
        start_event = make_llm_call_start(
            session_id=session_id,
            org_id=self.org_id,
            agent_id=agent_id,
            provider=provider,
            model=model,
            messages=messages,  # original messages — canary is NOT in the audit log
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
        # Enrich with spawn chain metadata BEFORE signing so it is included in the hash
        if state.spawn_depth > 0:
            start_event.setdefault("payload", {}).update({
                "spawn_depth": state.spawn_depth,
                "parent_agent_id": state.parent_agent_id,
                "parent_session_id": state.parent_session_id,
            })

        start_event = self._apply_pii(start_event)
        start_event = self._sign_event(start_event, prev_hash)
        return start_event

    def _handle_error_response(
        self,
        state: SessionState,
        session_id: str,
        agent_id: str,
        provider: str,
        model: str,
        run_id: str,
        response_data: dict,
        http_status: int,
        seq: int,
        prev_hash: str,
    ) -> bool:
        """Emit SYSTEM_ERROR event for 4xx/5xx responses and update session state.
        Returns True if the response was an error (caller should return early)."""
        if http_status < 400:
            return False
        state.error_count += 1
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
        state.sequence_number = seq + 1
        state.last_hash = error_event["current_hash"]
        state.last_seen_ns = time.time_ns()
        self.transport.enqueue(error_event)
        return True

    async def process_request(
        self,
        *,
        request_data: dict,
        provider: str,
        model: str,
        agent_id: str,
        explicit_session_id: str | None,
        parent_agent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> tuple[str, str, list[PolicyViolation], dict | None]:
        """
        Process an incoming LLM API request.

        Returns:
            (session_id, run_id, violations, forward_body)
            - violations: check for PolicyAction.BLOCK before forwarding.
            - forward_body: if not None, serialize and use instead of the
              original request body (contains canary + spotlighting modifications).
        """
        # Store agent_id before loading config so _load_cfg uses the agent-specific cache key
        self.agent_id = agent_id

        # Load org+agent security config (cached, 60s TTL — dashboard changes take effect promptly)
        cfg = await self._load_cfg()

        messages = request_data.get("messages", [])
        tools = request_data.get("tools") or request_data.get("functions") or []

        # Resolve session — returns its ID
        session_id = self.session_tracker.resolve_session(
            explicit_session_id=explicit_session_id,
            messages=messages,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_session_id=parent_session_id,
        )

        # get_state() returns the live SessionState object — mutations persist
        state: SessionState = self.session_tracker.get_state(session_id)
        seq = state.sequence_number
        prev_hash = state.last_hash

        # --- Spawn depth enforcement (Phase 6) --------------------------------
        # Block agents spawned beyond the configured max depth.
        if block := self._check_spawn_depth(
            state, session_id, agent_id, cfg.spawn_depth_max
        ):
            return session_id, str(uuid.uuid4()), block, None

        # --- Phase 4: Check async ML injection flag from previous turn -------
        # Async classifier runs after forwarding; defends against multi-turn
        # attack chains (inject turn 1, exfil turn 2) with zero added latency.
        if cfg.injection_ml_async_enabled:
            if block := self._check_ml_flag(state, session_id, agent_id):
                return session_id, str(uuid.uuid4()), block, None

        # --- HITL hold: block LLM call until pending approval is resolved ----
        # Polls backend until reviewer decides or deadline passes.
        if cfg.hitl_enabled:
            if block := await self._check_hitl_pending(state, session_id, agent_id):
                return session_id, str(uuid.uuid4()), block, None

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

                # --- Tool output scanning (Phase 3.3) ---
                # Scan tool return values for injection before re-entering LLM context.
                # Closes indirect-injection gap (EchoLeak CVE-2025-32711).
                # Pure pattern matching — runs always (no security_enabled gate).
                if settings.security_tool_output_scanning_enabled and tr.get("content"):
                    try:
                        loop = asyncio.get_running_loop()
                        tool_out_result = await loop.run_in_executor(
                            None,
                            lambda name=pending["tool_name"], content=tr["content"]: scan_tool_output(name, content),
                        )
                        if "security" not in tool_end:
                            tool_end["security"] = {}
                        tool_end["security"]["tool_output"] = tool_out_result.to_dict()

                        if tool_out_result.score >= settings.security_injection_block_threshold:
                            logger.warning(
                                "[TOOL-OUTPUT-SCAN:BLOCK] tool=%s score=%.3f session=%s agent=%s",
                                pending["tool_name"], tool_out_result.score, session_id, agent_id,
                            )
                            v = PolicyViolation(
                                rule_name="tool-output-injection-block",
                                action=PolicyAction.BLOCK,
                                reason=(
                                    f"Tool output injection detected: tool={pending['tool_name']} "
                                    f"score={tool_out_result.score:.3f} label={tool_out_result.label}"
                                ),
                                event_type="TOOL_CALL_END",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            if settings.violations_enabled:
                                self.transport.enqueue_violation(v.to_dict())
                            # Drop this tool result -- don't emit event, skip LLM re-entry
                            seq += 1
                            prev_hash = tool_end["current_hash"]
                            continue
                        elif tool_out_result.score >= settings.security_injection_alert_threshold:
                            logger.warning(
                                "[TOOL-OUTPUT-SCAN:ALERT] tool=%s score=%.3f session=%s agent=%s",
                                pending["tool_name"], tool_out_result.score, session_id, agent_id,
                            )
                            if settings.violations_enabled:
                                ov = PolicyViolation(
                                    rule_name="tool-output-injection-alert",
                                    action=PolicyAction.ALERT,
                                    reason=(
                                        f"Tool output suspicious: tool={pending['tool_name']} "
                                        f"score={tool_out_result.score:.3f} label={tool_out_result.label}"
                                    ),
                                    event_type="TOOL_CALL_END",
                                    session_id=session_id,
                                    agent_id=agent_id,
                                    org_id=self.org_id,
                                )
                                self.transport.enqueue_violation(ov.to_dict())

                        # ── Tool output ML scan (deep scan, optional) ──────
                        # When enabled, also run the full ML classifier pipeline
                        # on tool outputs (web pages, emails, file contents).
                        # This catches indirect injection payloads that pattern
                        # matching alone misses — e.g. paraphrased instructions
                        # embedded in a search result or email body.
                        if cfg.tool_output_ml_scan_enabled and tr.get("content"):
                            try:
                                from .enforcement import scan_messages as _enf_scan_tool_out
                                synthetic_msgs = [{"role": "user", "content": str(tr["content"])[:4096]}]
                                ml_tool_result = await loop.run_in_executor(
                                    None,
                                    lambda msgs=synthetic_msgs: _enf_scan_tool_out(msgs),
                                )
                                # Fuse pattern score with ML score
                                fused_tool_score = max(tool_out_result.score, ml_tool_result.injection_score)
                                if ml_tool_result.injection_score > tool_out_result.score:
                                    tool_end.setdefault("security", {})["tool_output_ml"] = {
                                        "injection_score": round(ml_tool_result.injection_score, 4),
                                        "injection_label": ml_tool_result.injection_label,
                                        "fused_score":     round(fused_tool_score, 4),
                                    }
                                    if fused_tool_score >= settings.security_injection_block_threshold:
                                        logger.warning(
                                            "[TOOL-OUTPUT-ML:BLOCK] tool=%s ml_score=%.3f session=%s",
                                            pending["tool_name"], ml_tool_result.injection_score, session_id,
                                        )
                                        mlv = PolicyViolation(
                                            rule_name="tool-output-ml-injection-block",
                                            action=PolicyAction.BLOCK,
                                            reason=(
                                                f"ML classifier detected injection in tool output: "
                                                f"tool={pending['tool_name']} "
                                                f"score={ml_tool_result.injection_score:.3f}"
                                            ),
                                            event_type="TOOL_CALL_END",
                                            session_id=session_id,
                                            agent_id=agent_id,
                                            org_id=self.org_id,
                                        )
                                        if settings.violations_enabled:
                                            self.transport.enqueue_violation(mlv.to_dict())
                                        seq += 1
                                        prev_hash = tool_end["current_hash"]
                                        continue
                            except Exception as ml_exc:
                                logger.debug("Tool output ML scan error (skipped): %s", ml_exc)

                    except Exception as exc:
                        logger.warning("Tool output scan error (continuing): %s", exc)

                # Hook 2-SOURCE-ACCURACY: Accumulate tool result for source accuracy detection (Phase H)
                # Appended to the session-wide rolling window — never cleared between turns
                # so that multi-turn responses can be checked against ALL prior tool results.
                if settings.security_hallucination_enabled and tr.get("content"):
                    state.session_tool_results.append({
                        "tool_name": pending["tool_name"],
                        "content":   tr["content"],
                    })
                    # Bound memory: keep only the most recent N results per session
                    window = settings.security_hallucination_window
                    if len(state.session_tool_results) > window:
                        state.session_tool_results = state.session_tool_results[-window:]

                # Hook 2: Taint credentials returned by this tool result
                if cfg.taint_tracking_enabled and tr.get("content"):
                    try:
                        from .security.taint_tracker import extract_credentials_from_text
                        tool_source = f"tool_result:{pending.get('tool_name', 'unknown')}"
                        for cred_val, cred_label in extract_credentials_from_text(str(tr["content"])):
                            state.get_taint_tracker().taint(cred_val, cred_label, tool_source)
                    except Exception as _te:
                        logger.debug("Taint extraction from tool result failed (skipped): %s", _te)

                # Hook 2-IFC: Label tool result as EXTERNAL in IFC context (Phase 11)
                if cfg.ifc_enabled and tr.get("content"):
                    try:
                        from .security.ifc_labels import Label
                        _ifc_tool_name = pending.get("tool_name", "unknown")
                        _ifc_source_id = f"tool_result:{_ifc_tool_name}"
                        state.get_ifc_context().add_source(
                            _ifc_source_id, str(tr["content"]), Label.EXTERNAL
                        )
                    except Exception as _ie:
                        logger.debug("IFC tool result labeling failed (skipped): %s", _ie)

                # Hook 2-PDG: Register tool result as PDG source (Phase 10)
                if cfg.pdg_enabled and tr.get("content"):
                    try:
                        from .security.session_pdg import classify_tool_trust
                        from .security.capability_manifest import get_active_manifest as _get_manifest_pdg
                        _pdg_tool_name = pending.get("tool_name", "unknown")
                        # Build manifest_tools lookup (cached — no extra network cost)
                        _pdg_manifest_tools: dict[str, str] | None = None
                        if settings.manifest_signing_key:
                            try:
                                _pdg_m = await _get_manifest_pdg(
                                    agent_id, self.org_id,
                                    settings.backend_url, settings.backend_api_key,
                                    signing_key=settings.manifest_signing_key,
                                    cache_ttl_s=settings.manifest_cache_ttl_s,
                                )
                                if _pdg_m is not None:
                                    _pdg_manifest_tools = {
                                        t.name: t.trust_classification
                                        for t in _pdg_m.permitted_tools
                                    }
                            except Exception:
                                pass
                        _pdg_trust = classify_tool_trust(_pdg_tool_name, _pdg_manifest_tools)
                        _pdg_source_id = f"tool_result:{_pdg_tool_name}"
                        state.get_pdg().add_source_node(
                            _pdg_source_id, str(tr["content"]),
                            trust=_pdg_trust, tool_name=_pdg_tool_name,
                        )
                    except Exception as _pe:
                        logger.debug("PDG tool result source failed (skipped): %s", _pe)

                seq += 1
                prev_hash = tool_end["current_hash"]
                self.transport.enqueue(tool_end)

        # --- LLM_CALL_START ---
        run_id = str(uuid.uuid4())
        start_event = self._build_llm_call_start(
            session_id=session_id, agent_id=agent_id, provider=provider, model=model,
            messages=messages, tools=tools, request_data=request_data,
            run_id=run_id, seq=seq, prev_hash=prev_hash, state=state,
        )

        # --- System prompt mutation detection (Phase 6) ----------------------
        # Hash the system prompt on the first call (baseline) and compare on
        # every subsequent call within the same session.  A mid-session change
        # is a near-certain indicator of a successful prompt injection or a
        # compromised orchestrator swapping instructions between turns.
        system_prompt = _extract_system_prompt(request_data)
        if system_prompt:
            new_hash = hashlib.sha256(system_prompt.encode("utf-8", errors="replace")).hexdigest()[:16]
            if state.system_prompt_hash is None:
                # Establish baseline — no violation on first call
                state.system_prompt_hash = new_hash
                logger.debug(
                    "[SYS-PROMPT] baseline stored hash=%s session=%s",
                    new_hash, session_id,
                )
                # Hook 1: Taint credentials found in system prompt
                if cfg.taint_tracking_enabled:
                    try:
                        from .security.taint_tracker import extract_credentials_from_text
                        for cred_val, cred_label in extract_credentials_from_text(system_prompt):
                            state.get_taint_tracker().taint(cred_val, cred_label, "system_prompt")
                    except Exception as _te:
                        logger.debug("Taint extraction from system prompt failed (skipped): %s", _te)

                # Hook 1-IFC: Label system prompt as TRUSTED in IFC context (Phase 11)
                if cfg.ifc_enabled:
                    try:
                        from .security.ifc_labels import Label
                        state.get_ifc_context().add_source("system_prompt", system_prompt, Label.TRUSTED)
                    except Exception as _ie:
                        logger.debug("IFC system prompt labeling failed (skipped): %s", _ie)

                # Hook 1-PDG: Register system prompt as TRUSTED PDG source (Phase 10)
                if cfg.pdg_enabled:
                    try:
                        state.get_pdg().add_source_node(
                            "system_prompt", system_prompt, trust="trusted", tool_name=None
                        )
                    except Exception as _pe:
                        logger.debug("PDG system prompt source failed (skipped): %s", _pe)

                # Hook 1-AUTH: Register system prompt intents in authorization chain (Phase 18)
                try:
                    state.get_auth_chain().add_system_prompt(system_prompt)
                except Exception as _ac_sys_exc:
                    logger.debug("Auth chain system prompt failed (skipped): %s", _ac_sys_exc)

                # --- Trust propagation (Phase 9) ----------------------------------
                # If this child session's parent was compromised (ML flag or high
                # injection score), lower the effective classifier threshold so the
                # child is more sensitive. Only done once (trust_score == 1.0 guard).
                if (
                    settings.trust_propagation_enabled
                    and state.spawn_depth > 0
                    and state.parent_session_id
                    and state.trust_score == 1.0
                ):
                    try:
                        parent = self.session_tracker.get_state(state.parent_session_id)
                        if parent is not None and (
                            parent.ml_injection_flag
                            or parent.max_injection_score > 0.7
                        ):
                            state.trust_score = 1.0 - settings.trust_propagation_discount
                            logger.info(
                                "[TRUST] child %s inherits reduced trust score=%.2f "
                                "(parent=%s max_inj_score=%.3f ml_flag=%s)",
                                session_id, state.trust_score,
                                state.parent_session_id,
                                parent.max_injection_score, parent.ml_injection_flag,
                            )
                    except Exception as _te:
                        logger.debug("Trust propagation error (skipped): %s", _te)

            elif state.system_prompt_hash != new_hash:
                logger.warning(
                    "[SYS-PROMPT:BLOCK] mutation detected: %s→%s session=%s agent=%s",
                    state.system_prompt_hash, new_hash, session_id, agent_id,
                )
                spv = PolicyViolation(
                    rule_name="system-prompt-mutation",
                    action=PolicyAction.BLOCK,
                    reason=(
                        f"System prompt changed mid-session "
                        f"(hash {state.system_prompt_hash} → {new_hash}). "
                        f"Possible successful prompt injection or compromised orchestrator."
                    ),
                    event_type="LLM_CALL_START",
                    session_id=session_id,
                    agent_id=agent_id,
                    org_id=self.org_id,
                )
                if settings.violations_enabled:
                    self.transport.enqueue_violation(spv.to_dict())
                start_event["blocked"] = True
                start_event["block_reason"] = "system-prompt-mutation"
                seq += 1
                prev_hash = start_event["current_hash"]
                state.sequence_number = seq
                state.last_hash = prev_hash
                state.llm_call_count += 1
                state.last_seen_ns = time.time_ns()
                self.transport.enqueue(start_event)
                return session_id, run_id, [spv], None

        # --- Tool baseline: hash tools[] and detect mid-session mutations ----
        # The LLM API guarantees the model can only call tools in the tools[]
        # array sent by the client.  We fingerprint that array on the first
        # call of a session and BLOCK if it changes on any subsequent call.
        # This catches prompt-injection that convinces an orchestrator to
        # add extra tools mid-flight, with zero risk on the first call.
        if settings.tool_baseline_enabled and tools:
            try:
                from .security.tool_baseline import (
                    extract_tool_names as _tb_names,
                    hash_tool_set as _tb_hash,
                    report_tools_observed as _tb_report,
                )
                _tool_names_now = _tb_names(tools)
                _tool_hash_now  = _tb_hash(tools)

                if state.tools_hash is None:
                    # ── First call: establish session baseline ──────────────
                    state.tools_hash = _tool_hash_now
                    state.tools_set  = frozenset(_tool_names_now)
                    logger.debug(
                        "[TOOL-BASELINE] baseline set hash=%s tools=%s session=%s agent=%s",
                        _tool_hash_now, sorted(_tool_names_now), session_id, agent_id,
                    )
                    # Report observed tools to backend (async, fire-and-forget)
                    asyncio.create_task(
                        _tb_report(
                            agent_id, self.org_id, tools, session_id,
                            settings.backend_url, settings.backend_api_key,
                        ),
                        name=f"tb-observe-{session_id[:8]}",
                    )

                elif _tool_hash_now != state.tools_hash:
                    # ── Subsequent call: tool set changed → BLOCK ───────────
                    _new_tools = _tool_names_now - (state.tools_set or set())
                    _removed   = (state.tools_set or set()) - _tool_names_now
                    logger.warning(
                        "[TOOL-BASELINE:BLOCK] mid-session mutation "
                        "added=%s removed=%s session=%s agent=%s",
                        sorted(_new_tools), sorted(_removed), session_id, agent_id,
                    )
                    _tbm_v = PolicyViolation(
                        rule_name="tool-set-mutation",
                        action=PolicyAction.BLOCK,
                        reason=(
                            f"Tool set changed mid-session. "
                            f"Added: {sorted(_new_tools) or 'none'}. "
                            f"Removed: {sorted(_removed) or 'none'}."
                        ),
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                        timestamp_ns=time.time_ns(),
                    )
                    start_event.setdefault("security", {})["tool_baseline"] = {
                        "mutation_detected": True,
                        "new_tools": sorted(_new_tools),
                        "removed_tools": sorted(_removed),
                    }
                    start_event["blocked"] = True
                    seq += 1
                    prev_hash = start_event["current_hash"]
                    state.sequence_number = seq
                    state.last_hash = prev_hash
                    state.llm_call_count += 1
                    state.last_seen_ns = time.time_ns()
                    self.transport.enqueue(start_event)
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(_tbm_v.to_dict())
                    return session_id, run_id, [_tbm_v], None
            except Exception as _tbe:
                logger.debug("Tool baseline check error (skipped): %s", _tbe)

        # --- MCP tool definition scanning (Phase 3.3) ---
        # Scan tools[] array for malicious definitions: name traversal,
        # description injection, shadow overloading.
        # Pure Python, no ML — runs always regardless of security_enabled.
        mcp_block_violations: list[PolicyViolation] = []
        if cfg.mcp_scanning_enabled and tools:
            try:
                from .security.mcp_scanner import scan as _scan_mcp
                mcp_result = _scan_mcp(tools)
                if mcp_result.detected:
                    start_event.setdefault("security", {})["mcp"] = mcp_result.to_dict()
                    logger.warning(
                        "[MCP-SCAN:%s] findings=%d tools_scanned=%d session=%s agent=%s",
                        mcp_result.severity, len(mcp_result.findings),
                        mcp_result.tools_scanned, session_id, agent_id,
                    )
                    if mcp_result.severity == "high":
                        v = PolicyViolation(
                            rule_name="mcp-tool-definition-block",
                            action=PolicyAction.BLOCK,
                            reason=(
                                f"Malicious tool definition detected: "
                                f"{mcp_result.findings[0].finding_type if mcp_result.findings else 'unknown'} "
                                f"tools_scanned={mcp_result.tools_scanned}"
                            ),
                            event_type="LLM_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        mcp_block_violations.append(v)
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(v.to_dict())
                    elif mcp_result.severity == "medium" and settings.violations_enabled:
                        av = PolicyViolation(
                            rule_name="mcp-tool-definition-alert",
                            action=PolicyAction.ALERT,
                            reason=(
                                f"Suspicious tool definition: "
                                f"{mcp_result.findings[0].finding_type if mcp_result.findings else 'unknown'} "
                                f"tools_scanned={mcp_result.tools_scanned}"
                            ),
                            event_type="LLM_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        self.transport.enqueue_violation(av.to_dict())
            except Exception as exc:
                logger.warning("MCP scan error (continuing): %s", exc)

        # --- MCP rug pull detection -------------------------------------------
        # Detects tool definitions that changed after capability manifest approval.
        # Runs only when tool definitions have been observed at least once before.
        if cfg.mcp_scanning_enabled and tools and state.tool_definitions:
            try:
                from .security.mcp_scanner import (
                    hash_tool_definitions as _hash_defs,
                    detect_rug_pull as _detect_rug_pull,
                )
                _rug_pulls = _detect_rug_pull(tools, state.tool_definitions)
                if _rug_pulls:
                    _rp_names = [name for name, _, _ in _rug_pulls]
                    logger.warning(
                        "[MCP-RUG-PULL:BLOCK] changed_tools=%s session=%s agent=%s",
                        _rp_names, session_id, agent_id,
                    )
                    _rp_detail = "; ".join(
                        f"{n}: {o[:8]}→{h[:8]}" for n, o, h in _rug_pulls
                    )
                    _rp_violation = PolicyViolation(
                        rule_name="mcp-rug-pull",
                        action=PolicyAction.BLOCK,
                        reason=f"Tool definitions changed after approval: {_rp_detail}",
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    mcp_block_violations.append(_rp_violation)
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(_rp_violation.to_dict())
                    start_event.setdefault("security", {})["mcp_rug_pull"] = {
                        "detected": True,
                        "changed_tools": _rp_names,
                        "detail": _rp_detail,
                    }
                else:
                    # Update snapshot with current definitions (no changes)
                    state.tool_definitions = _hash_defs(tools)
            except Exception as exc:
                logger.warning("MCP rug pull check error (continuing): %s", exc)
        elif cfg.mcp_scanning_enabled and tools and state.tool_definitions is None:
            # First observation: record snapshot for future rug-pull comparison
            try:
                from .security.mcp_scanner import hash_tool_definitions as _hash_defs
                state.tool_definitions = _hash_defs(tools)
            except Exception:
                pass

        # Return early if MCP scan produced a BLOCK — store event for audit trail
        if mcp_block_violations:
            start_event["blocked"] = True
            start_event["block_reason"] = mcp_block_violations[0].rule_name
            seq += 1
            prev_hash = start_event["current_hash"]
            state.sequence_number = seq
            state.last_hash = prev_hash
            state.llm_call_count += 1
            state.last_seen_ns = time.time_ns()
            self.transport.enqueue(start_event)
            return session_id, run_id, mcp_block_violations, None

        # --- PII Redaction & Tokenization -----------------------------------
        # Replace PII in messages with reversible tokens before forwarding.
        # Must run BEFORE the enforcement scan (scan the redacted messages,
        # not the raw ones, so PII doesn't influence injection scoring).
        if cfg.pii_redaction_enabled:
            try:
                from .security.pii_redactor import (
                    redact_messages as _redact_messages,
                    PIIBlockError as _PIIBlockError,
                    get_redaction_config as _get_redaction_cfg,
                )
                _redaction_cfg = await _get_redaction_cfg(
                    self.org_id, settings.backend_url, settings.backend_api_key,
                )
                if _redaction_cfg.enabled:
                    _vault = state.get_pii_vault()
                    messages, _redaction_summary = _redact_messages(messages, _vault, _redaction_cfg)
                    if _redaction_summary["total_findings"] > 0:
                        start_event.setdefault("security", {})["pii_redaction"] = _redaction_summary
                        logger.info(
                            "[PII-REDACT] %d tokens created session=%s types=%s",
                            _redaction_summary["total_findings"], session_id,
                            list(_redaction_summary["by_type"].keys()),
                        )
            except Exception as _pii_exc:
                # PIIBlockError — block the request
                if hasattr(_pii_exc, "pii_type"):
                    _pii_v = PolicyViolation(
                        rule_name="pii-redaction-block",
                        action=PolicyAction.BLOCK,
                        reason=f"PII type {_pii_exc.pii_type!r} with mode=block found in request",
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    start_event["blocked"] = True
                    start_event["block_reason"] = "pii-redaction-block"
                    start_event.setdefault("security", {})["pii_redaction"] = {
                        "blocked": True, "pii_type": _pii_exc.pii_type,
                    }
                    seq += 1
                    prev_hash = start_event["current_hash"]
                    state.sequence_number = seq
                    state.last_hash = prev_hash
                    state.llm_call_count += 1
                    state.last_seen_ns = time.time_ns()
                    self.transport.enqueue(start_event)
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(_pii_v.to_dict())
                    return session_id, run_id, [_pii_v], None
                else:
                    logger.warning("PII redaction error (continuing): %s", _pii_exc)

        # --- Enforcement scan: structural + credential, always runs, <5ms ----
        # Zero ML dependencies. Provides injection_score + credential_detected
        # for policy engine evaluation regardless of security_enabled setting.
        violations: list[PolicyViolation] = []
        try:
            from .enforcement import scan_messages as _enforcement_scan
            enf_result = _enforcement_scan(messages)
            start_event["security"] = enf_result.to_event_dict()
            if enf_result.injection_score > state.max_injection_score:
                state.max_injection_score = enf_result.injection_score
            if enf_result.injection_score >= settings.security_injection_alert_threshold:
                logger.warning(
                    "[ENFORCEMENT] Injection risk: score=%.3f label=%s session=%s agent=%s",
                    enf_result.injection_score, enf_result.injection_label,
                    session_id, agent_id,
                )
            if enf_result.credential_detected:
                logger.warning(
                    "[ENFORCEMENT] Credential detected: %d match(es) session=%s agent=%s",
                    enf_result.credential_count, session_id, agent_id,
                )
        except Exception as exc:
            logger.warning("Enforcement scan error (continuing): %s", exc)

        # --- Auth chain: accumulate user message intents (Phase 18) ---------------
        # Track what the user has asked for this session so we can detect
        # TOOL_CALL_START events whose intent class was never requested.
        # Runs unconditionally — cheap frozenset lookups, no I/O.
        try:
            for _ac_msg in messages:
                if isinstance(_ac_msg, dict) and _ac_msg.get("role") == "user":
                    _ac_content = _ac_msg.get("content", "")
                    if isinstance(_ac_content, str) and _ac_content.strip():
                        state.get_auth_chain().add_user_message(_ac_content)
                    elif isinstance(_ac_content, list):
                        # OpenAI vision format: list of content parts
                        for _ac_part in _ac_content:
                            if isinstance(_ac_part, dict) and _ac_part.get("type") == "text":
                                _ac_text = _ac_part.get("text", "")
                                if _ac_text:
                                    state.get_auth_chain().add_user_message(_ac_text)
        except Exception as _ac_exc:
            logger.debug("Auth chain user message accumulation failed (skipped): %s", _ac_exc)

        # --- Multi-modal scan: visual prompt injection in images (Phase V2) ----
        # Scans base64 images embedded in the request for EXIF metadata injection,
        # LSB steganography, QR/barcode payloads, OCR-extracted injection text, and
        # low-contrast text overlays. Runs only when Pillow is installed; degrades
        # gracefully otherwise. Uses the enforcement scanner as injection_fn so
        # flagged image text is scored with the same model as normal request text.
        if cfg.multimodal_scan_enabled:
            try:
                from .security.multimodal import scan_request_images as _mm_scan
                from .enforcement import scan_messages as _mm_enf_scan

                def _mm_injection_fn(text: str) -> float:
                    try:
                        return _mm_enf_scan([{"role": "user", "content": text}]).injection_score
                    except Exception:
                        return 0.0

                mm_results = _mm_scan(request_data, _mm_injection_fn)
                detected = [r for r in mm_results if r.detected]
                if detected:
                    worst = max(detected, key=lambda r: r.score)
                    start_event.setdefault("security", {})["multimodal"] = {
                        "images_scanned": len(mm_results),
                        "threats_detected": len(detected),
                        "top_threat": worst.threat,
                        "top_score": round(worst.score, 3),
                        "flags": worst.flags,
                    }
                    logger.warning(
                        "[MULTIMODAL:%s] images=%d threats=%d top_threat=%s score=%.3f "
                        "session=%s agent=%s",
                        "BLOCK" if worst.score >= settings.security_injection_block_threshold else "ALERT",
                        len(mm_results), len(detected), worst.threat, worst.score,
                        session_id, agent_id,
                    )
                    mm_action = (
                        PolicyAction.BLOCK
                        if worst.score >= settings.security_injection_block_threshold
                        else PolicyAction.ALERT
                    )
                    mm_v = PolicyViolation(
                        rule_name="visual-prompt-injection",
                        action=mm_action,
                        reason=(
                            f"Visual prompt injection detected in image: "
                            f"threat={worst.threat} score={worst.score:.3f} "
                            f"flags={','.join(worst.flags)}"
                        ),
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(mm_v.to_dict())
                    if mm_action == PolicyAction.BLOCK:
                        start_event["blocked"] = True
                        start_event["block_reason"] = "visual-prompt-injection"
                        seq += 1
                        prev_hash = start_event["current_hash"]
                        state.sequence_number = seq
                        state.last_hash = prev_hash
                        state.llm_call_count += 1
                        state.last_seen_ns = time.time_ns()
                        self.transport.enqueue(start_event)
                        return session_id, run_id, [mm_v], None
                    else:
                        violations.append(mm_v)
            except Exception as exc:
                logger.debug("Multimodal scan error (skipped): %s", exc)

        # --- Context-aware injection scoring (multi-turn Crescendo detection) --
        # Track per-turn injection scores. If the rolling average over the last
        # N turns is elevated, fire even when no single turn exceeds the threshold.
        # This catches gradual injection attacks that stay under the per-call radar.
        try:
            if hasattr(state, "injection_score_history") and "security" in start_event:
                score_now = start_event["security"].get("injection_score", 0.0)
                state.injection_score_history.append(score_now)
                window = settings.security_context_injection_window
                if len(state.injection_score_history) > window:
                    state.injection_score_history = state.injection_score_history[-window:]
                if len(state.injection_score_history) >= 3:
                    rolling_avg = sum(state.injection_score_history) / len(state.injection_score_history)
                    if rolling_avg >= settings.security_context_injection_block_avg:
                        logger.warning(
                            "[CONTEXT-INJECT:BLOCK] rolling_avg=%.3f window=%d session=%s agent=%s",
                            rolling_avg, len(state.injection_score_history), session_id, agent_id,
                        )
                        cv = PolicyViolation(
                            rule_name="context-injection-block",
                            action=PolicyAction.BLOCK,
                            reason=(
                                f"Multi-turn injection detected: rolling avg injection score "
                                f"{rolling_avg:.3f} over {len(state.injection_score_history)} turns "
                                f">= block threshold {settings.security_context_injection_block_avg}"
                            ),
                            event_type="LLM_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        violations.append(cv)
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(cv.to_dict())
                    elif rolling_avg >= settings.security_context_injection_alert_avg:
                        logger.warning(
                            "[CONTEXT-INJECT:ALERT] rolling_avg=%.3f window=%d session=%s agent=%s",
                            rolling_avg, len(state.injection_score_history), session_id, agent_id,
                        )
                        if settings.violations_enabled:
                            ca = PolicyViolation(
                                rule_name="context-injection-alert",
                                action=PolicyAction.ALERT,
                                reason=(
                                    f"Elevated multi-turn injection signal: rolling avg "
                                    f"{rolling_avg:.3f} over {len(state.injection_score_history)} turns"
                                ),
                                event_type="LLM_CALL_START",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            self.transport.enqueue_violation(ca.to_dict())
        except Exception as exc:
            logger.debug("Context-aware injection scoring error (skipped): %s", exc)

        # --- Rate limiting: token budget + call budget -------------------------
        try:
            if cfg.rate_limit_tokens > 0:
                if state.total_tokens >= cfg.rate_limit_tokens:
                    logger.warning(
                        "[RATE-LIMIT:BLOCK] token budget exhausted: total=%d cap=%d session=%s",
                        state.total_tokens, cfg.rate_limit_tokens, session_id,
                    )
                    rv = PolicyViolation(
                        rule_name="token-budget-exceeded",
                        action=PolicyAction.BLOCK,
                        reason=(
                            f"Session token budget exhausted: {state.total_tokens} tokens used, "
                            f"cap is {cfg.rate_limit_tokens}"
                        ),
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    violations.append(rv)
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(rv.to_dict())
            if cfg.rate_limit_llm_calls > 0:
                if state.llm_call_count >= cfg.rate_limit_llm_calls:
                    logger.warning(
                        "[RATE-LIMIT:BLOCK] LLM call budget exhausted: calls=%d cap=%d session=%s",
                        state.llm_call_count, cfg.rate_limit_llm_calls, session_id,
                    )
                    rcv = PolicyViolation(
                        rule_name="llm-call-budget-exceeded",
                        action=PolicyAction.BLOCK,
                        reason=(
                            f"Session LLM call budget exhausted: {state.llm_call_count} calls, "
                            f"cap is {cfg.rate_limit_llm_calls}"
                        ),
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    violations.append(rcv)
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(rcv.to_dict())
        except Exception as exc:
            logger.debug("Rate limit check error (skipped): %s", exc)

        # Policy evaluation BEFORE forwarding — BLOCK fires here
        policy_violations = self._evaluate_policy(start_event, state)
        violations = violations + policy_violations
        block = [v for v in violations if v.action == PolicyAction.BLOCK]

        # --- Trust graph: update on solid BLOCK violations (Phase 9C) ----------
        # Only triggered by cryptographically solid signals (canary, taint, mutation).
        _inj_score_now = start_event.get("security", {}).get("injection_score", 0.0)
        if block:
            try:
                from .trust.graph import get_trust_graph
                new_trust = get_trust_graph().on_violation(
                    session_id, block[0].rule_name, "BLOCK", _inj_score_now
                )
                state.trust_score = new_trust
            except Exception as _tg_exc:
                logger.warning("Trust graph violation update failed (continuing): %s", _tg_exc)
            # Store blocked event in audit trail so blocks are fully auditable
            start_event["blocked"] = True
            start_event["block_reason"] = block[0].rule_name
            seq += 1
            prev_hash = start_event["current_hash"]
            state.sequence_number = seq
            state.last_hash = prev_hash
            state.llm_call_count += 1
            state.last_seen_ns = time.time_ns()
            self.transport.enqueue(start_event)
            return session_id, run_id, block, None

        seq += 1
        prev_hash = start_event["current_hash"]

        # Persist updated state back onto the SessionState object
        state.sequence_number = seq
        state.last_hash = prev_hash
        state.llm_call_count += 1
        state.last_seen_ns = time.time_ns()

        self.transport.enqueue(start_event)

        # ── Build forward_body: apply spotlighting + canary injection ─────────
        # These modifications are applied to the messages that are FORWARDED to
        # the LLM but NOT stored in the audit log (the audit captures originals).
        forward_body: dict | None = None
        forward_messages = messages

        # Spotlighting: wrap tool output in randomized delimiters to defend
        # against indirect prompt injection (Microsoft Research, 2024).
        if settings.security_spotlighting_enabled:
            try:
                from .security.spotlighting import (
                    spotlight_tool_messages,
                    add_spotlight_directive_to_system,
                )
                forward_messages = spotlight_tool_messages(forward_messages)
                forward_messages = add_spotlight_directive_to_system(forward_messages)
            except Exception as exc:
                logger.warning("Spotlighting error (skipped): %s", exc)

        # Canary injection: embed a 256-bit random token in the system prompt.
        # Any appearance of this token in the LLM response confirms exfiltration.
        if cfg.canary_enabled:
            try:
                from .security.canary import generate as _gen_canary, inject_into_messages
                canary_token = _gen_canary()
                state.active_canaries[run_id] = canary_token
                forward_messages = inject_into_messages(forward_messages, canary_token)
            except Exception as exc:
                logger.warning("Canary injection error (skipped): %s", exc)

        # Only build a new body dict if we actually modified the messages
        if forward_messages is not messages:
            forward_body = {**request_data, "messages": forward_messages}

        # --- Phase 4: Schedule async ML classification (non-blocking) ---------
        # Classify the current messages with DeBERTa in a background thread.
        # If injection is detected, session.ml_injection_flag is set True and
        # the NEXT call in this session will be BLOCKed (see check above).
        if cfg.injection_ml_async_enabled:
            asyncio.create_task(
                self._classify_messages_async(messages, state)
            )

        return session_id, run_id, violations, forward_body

    def _build_llm_call_end(
        self,
        *,
        session_id: str,
        agent_id: str,
        provider: str,
        model: str,
        run_id: str,
        response_data: dict,
        latency_ms: float,
        http_status: int,
        state: SessionState,
        seq: int,
        prev_hash: str,
    ) -> dict:
        """
        Accumulate token usage onto session state, then build, PII-mask, and
        hash-chain-sign the LLM_CALL_END event dict.

        Side effect: mutates state.total_tokens with the tokens from this response.
        """
        token_usage = response_data.get("token_usage")
        if token_usage and isinstance(token_usage, dict):
            try:
                used = int(
                    token_usage.get("total_tokens") or
                    token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0) or
                    0
                )
                if used > 0:
                    state.total_tokens += used
            except (TypeError, ValueError):
                pass

        end_event = make_llm_call_end(
            session_id=session_id,
            org_id=self.org_id,
            agent_id=agent_id,
            provider=provider,
            model=model,
            run_id=run_id,
            response_text=response_data.get("response_text"),
            finish_reason=response_data.get("finish_reason"),
            tool_calls=response_data.get("tool_calls", []),
            token_usage=token_usage,
            latency_ms=latency_ms,
            http_status=http_status,
            thinking_blocks=response_data.get("thinking_blocks") or None,
            sequence_number=seq,
            previous_hash=prev_hash,
        )
        end_event = self._apply_pii(end_event)
        end_event = self._sign_event(end_event, prev_hash)
        return end_event

    async def _classify_messages_async(
        self,
        messages: list[dict],
        state: SessionState,
    ) -> None:
        """
        Background ML injection classifier (Phase 4).

        Runs in a background task AFTER forwarding the current call. Never
        raises. Updates state.ml_injection_flag so the NEXT call in this
        session is blocked if injection is detected.
        """
        try:
            from .analysis.classifier import classify as _ml_classify

            # Only examine attacker-controlled segments (user + tool messages)
            segments = [
                str(m.get("content") or "")
                for m in messages
                if m.get("role") in ("user", "tool") and m.get("content")
            ]
            if not segments:
                return

            full_text = " ".join(segments)
            text = full_text[:2048]  # DeBERTa primary — cap at 512 tokens

            # Apply trust-adjusted threshold: child sessions of compromised
            # parents get a lower threshold (more sensitive detection).
            _async_cfg = await self._load_cfg()
            effective_threshold = _async_cfg.injection_ml_threshold * state.trust_score

            loop = asyncio.get_running_loop()

            # ── Primary async: DeBERTa (thorough, ~50ms) ──────────────────
            result = await loop.run_in_executor(
                None,
                lambda: _ml_classify(
                    text,
                    model_name=settings.analysis_classifier_model,
                    threshold=effective_threshold,
                ),
            )

            state.ml_injection_score = result.score

            if result.label == "malicious":
                state.ml_injection_flag = True
                logger.warning(
                    "[ANALYSIS:ML-INJECTION-FLAG] Session flagged for next-call block: "
                    "score=%.3f model=%s session=%s",
                    result.score, result.model, state.session_id,
                )
            elif result.label == "suspicious":
                # Suspicious: alert-only, do not block next call
                logger.info(
                    "[ANALYSIS:ML-INJECTION-ALERT] Suspicious injection signal: "
                    "score=%.3f model=%s session=%s",
                    result.score, result.model, state.session_id,
                )
                if settings.violations_enabled:
                    v = PolicyViolation(
                        rule_name="ml-classifier-injection-alert",
                        action=PolicyAction.ALERT,
                        reason=(
                            f"Async ML classifier suspicious injection signal: "
                            f"score={result.score:.3f} model={result.model}"
                        ),
                        event_type="LLM_CALL_START",
                        session_id=state.session_id,
                        agent_id=state.agent_id,
                        org_id=self.org_id,
                    )
                    self.transport.enqueue_violation(v.to_dict())

        except Exception as exc:
            logger.debug("Background ML classification error (ignored): %s", exc)

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
    ) -> list[PolicyViolation]:
        """
        Process a complete LLM API response (assembled from SSE chunks if streaming).

        Returns a list of PolicyViolation objects raised during TOOL_CALL_START
        processing (e.g. taint-tracking BLOCK violations). Callers should check
        for BLOCK violations and return 403 before forwarding the response to the
        agent (non-streaming only; streaming responses are already in flight).
        """
        cfg = await self._load_cfg()
        state: SessionState = self.session_tracker.get_state(session_id)
        _response_violations: list[PolicyViolation] = []
        seq = state.sequence_number
        prev_hash = state.last_hash

        if self._handle_error_response(
            state, session_id, agent_id, provider, model, run_id,
            response_data, http_status, seq, prev_hash,
        ):
            return []

        response_text = response_data.get("response_text")
        finish_reason = response_data.get("finish_reason")
        tool_calls = response_data.get("tool_calls", [])

        # --- PII token restore (post-response) --------------------------------
        # If we tokenized PII in the request, restore tokens in the LLM's
        # response so the application receives the original values.
        # Only applies to "tokenize" mode PII (redact/block are one-way).
        if cfg.pii_redaction_enabled and state.pii_vault and state.pii_vault.size() > 0:
            try:
                if isinstance(response_text, str):
                    response_text = state.pii_vault.restore(response_text)
                    response_data = {**response_data, "response_text": response_text}
            except Exception as _restore_exc:
                logger.warning("PII token restore error (continuing): %s", _restore_exc)

        end_event = self._build_llm_call_end(
            session_id=session_id, agent_id=agent_id, provider=provider, model=model,
            run_id=run_id, response_data=response_data, latency_ms=latency_ms,
            http_status=http_status, state=state, seq=seq, prev_hash=prev_hash,
        )

        # ── Output scanning (Phase 3.2) ────────────────────────────────────────
        # Scan the LLM response for canary leakage, relay injection, credential
        # exposure, and prompt echo. The canary for this run_id is consumed here
        # (popped from state.active_canaries) so it is never reused.
        if settings.security_output_scanning_enabled:
            try:
                canary_for_run = state.active_canaries.pop(run_id, None)
                from .security.output_scanner import scan as _scan_output
                output_result = _scan_output(response_text, canary=canary_for_run)
                if "security" not in end_event:
                    end_event["security"] = {}
                end_event["security"]["output"] = output_result.to_dict()
                if output_result.detected:
                    logger.warning(
                        "[OUTPUT-SCAN:%s] threats=%s session=%s agent=%s",
                        output_result.severity,
                        output_result.threats[:5],
                        session_id, agent_id,
                    )
                    if settings.violations_enabled:
                        ov = PolicyViolation(
                            rule_name="output-threat-detected",
                            action=PolicyAction.ALERT,
                            reason=(
                                f"Output scanner: {', '.join(output_result.threats[:3])} "
                                f"severity={output_result.severity}"
                            ),
                            event_type="LLM_CALL_END",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        self.transport.enqueue_violation(ov.to_dict())
            except Exception as exc:
                logger.warning("Output scan error (continuing): %s", exc)

        # ── Source Accuracy Detection (Phase H) ───────────────────────────────────
        # Compare the LLM response against the full session-wide tool result window.
        # Checks whether factual claims in the response are grounded in actual tool outputs.
        # Catches multi-turn inaccuracies (e.g. turn 5 referencing data fetched in turn 1).
        if settings.security_hallucination_enabled and state.session_tool_results and response_text:
            try:
                from .security.hallucination_detector import (
                    SourceAccuracyDetector, SourceAccuracyConfig,
                )
                h_cfg = SourceAccuracyConfig(
                    enabled=True,
                    action=settings.security_hallucination_action,
                    threshold=settings.security_hallucination_threshold,
                    use_minicheck=settings.security_hallucination_use_minicheck,
                )
                h_result = SourceAccuracyDetector().check(
                    tool_results=state.session_tool_results,
                    llm_response=response_text,
                    config=h_cfg,
                )
                # Store findings in end_event.security JSONB
                # Key kept as "hallucination" for DB backward compatibility
                if h_result.checked:
                    end_event.setdefault("security", {})["hallucination"] = h_result.to_dict()

                if h_result.detected:
                    logger.warning(
                        "[SOURCE_ACCURACY:%s] findings=%d session=%s agent=%s",
                        h_result.severity, len(h_result.findings), session_id, agent_id,
                    )
                    if settings.violations_enabled:
                        h_action = PolicyAction.BLOCK if settings.security_hallucination_action == "block" else PolicyAction.ALERT
                        hv = PolicyViolation(
                            rule_name="hallucination-detected",
                            action=h_action,
                            reason=(
                                f"LLM response contains {len(h_result.findings)} claim(s) not supported "
                                f"by tool outputs: severity={h_result.severity}"
                            ),
                            event_type="LLM_CALL_END",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        self.transport.enqueue_violation(hv.to_dict())
            except Exception as _he:
                logger.debug("Source accuracy detection error (skipped): %s", _he)
            # NOTE: session_tool_results is NOT cleared — accumulates for the full session

        # ── Refusal Detection (Phase RAG-DoS) ─────────────────────────────────

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

            # --- Enforcement: RCE + SSRF + schema validation (always runs) ----
            # Deterministic, no ML. Provides rce_detected/ssrf_detected for
            # policy engine regardless of security_enabled setting.
            tool_scan = None  # initialised before try; set inside on success
            try:
                from .enforcement import scan_tool_call as enf_scan_tool, validate_tool_args
                loop = asyncio.get_running_loop()
                tool_scan = await loop.run_in_executor(
                    None,
                    lambda name=tc_name, args=tc_args: enf_scan_tool(name, args),
                )
                tool_security_dict = tool_scan.to_event_dict()

                # Schema validation — catches args that violate the tool's declared schema
                schema_result = validate_tool_args(tc_name, tc_args, tools)
                if not schema_result.valid and schema_result.violations:
                    tool_security_dict["schema_violations"] = [
                        {"field": v.field, "expected": v.expected,
                         "actual": v.actual, "severity": v.severity}
                        for v in schema_result.violations[:5]
                    ]
                    logger.warning(
                        "[SCHEMA] Tool %r has %d schema violation(s) session=%s",
                        tc_name, len(schema_result.violations), session_id,
                    )

                tool_start["security"] = tool_security_dict

                if tool_scan.rce_detected:
                    logger.warning(
                        "[ENFORCEMENT] RCE attempt: tool=%s confidence=%.3f "
                        "patterns=%s session=%s agent=%s",
                        tc_name, tool_scan.rce.confidence,
                        tool_scan.rce.dangerous_patterns[:3], session_id, agent_id,
                    )
                if tool_scan.ssrf_detected:
                    top_host = tool_scan.ssrf.matches[0].host if tool_scan.ssrf.matches else "?"
                    logger.warning(
                        "[ENFORCEMENT] SSRF attempt: tool=%s host=%s "
                        "urls_scanned=%d session=%s agent=%s",
                        tc_name, top_host, tool_scan.ssrf.urls_scanned,
                        session_id, agent_id,
                    )
            except Exception as exc:
                logger.warning("Enforcement tool scan error (continuing): %s", exc)

            # --- Blast Radius Guard (Phase 15 — Excessive Agency Prevention) ----
            # Classify tool call by (reversibility × scope) BEFORE execution.
            # CRITICAL (≥0.85): BLOCK + policy violation immediately.
            # HIGH     (≥0.60): HITL — next LLM call held for human review.
            # MEDIUM   (≥0.30): ALERT — logged, not blocked.
            # Session accumulator: when cumulative score > budget, HIGH→HITL.
            _br_result = None
            if cfg.blast_radius_enabled:
                try:
                    from .security.blast_radius import score_blast_radius as _score_br
                    _br_args = tc_args if isinstance(tc_args, dict) else {}
                    _br_result = _score_br(
                        tc_name,
                        _br_args,
                        spawn_depth=state.spawn_depth,
                        manifest_tools=None,  # manifest_tools integration deferred to Phase 15.1
                    )
                    state.blast_radius_cumulative += _br_result.blast_score
                    tool_start.setdefault("security", {}).update(
                        {"blast_radius": _br_result.to_dict(),
                         "blast_radius_cumulative": round(state.blast_radius_cumulative, 4)}
                    )
                    if _br_result.risk_level == "CRITICAL":
                        _br_v = PolicyViolation(
                            rule_name="excessive-agency-block",
                            action=PolicyAction.BLOCK,
                            reason=(
                                f"Blast radius CRITICAL (score={_br_result.blast_score:.3f}): "
                                + "; ".join(_br_result.signals[:3])
                            ),
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        logger.warning(
                            "[BLAST-RADIUS:BLOCK] tool=%s score=%.3f signals=%s session=%s",
                            tc_name, _br_result.blast_score, _br_result.signals[:2], session_id,
                        )
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(_br_v.to_dict())
                        _response_violations.append(_br_v)
                    elif _br_result.risk_level == "MEDIUM":
                        _br_alert = PolicyViolation(
                            rule_name="excessive-agency-alert",
                            action=PolicyAction.ALERT,
                            reason=(
                                f"Blast radius MEDIUM (score={_br_result.blast_score:.3f}): "
                                + "; ".join(_br_result.signals[:3])
                            ),
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        logger.info(
                            "[BLAST-RADIUS:ALERT] tool=%s score=%.3f session=%s",
                            tc_name, _br_result.blast_score, session_id,
                        )
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(_br_alert.to_dict())
                except Exception as _br_exc:
                    logger.warning("Blast radius scan error (continuing): %s", _br_exc)

            # Early exit: if blast radius already produced a BLOCK, skip the remaining
            # expensive hooks (sandbox, compound, behavioral). Annotation-only hooks
            # (rollback gate, auth chain) and the taint/IFC checks still run below
            # because they annotate the event or may escalate independently.
            _already_blocked = any(v.action == PolicyAction.BLOCK for v in _response_violations)

            # --- Sandbox Dry-Run (Phase 16 — Pre-Execution Scope Estimation) ----
            # Runs the first matching sandbox adapter (email preview, git dry-run,
            # etc.) when the blast radius score meets the configured threshold.
            # Fail-open: adapter errors / missing git binary → proxy continues.
            if not _already_blocked and cfg.sandbox_enabled and _br_result is not None and \
                    _br_result.blast_score >= cfg.sandbox_min_blast_score:
                try:
                    from .security.sandbox import SandboxRegistry as _SB_REG
                    from .security.sandbox import SandboxContext as _SB_CTX
                    _sb_ctx = _SB_CTX(
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                        spawn_depth=state.spawn_depth,
                        timeout_ms=cfg.sandbox_timeout_ms,
                        blast_score=_br_result.blast_score,
                        verb_class=_br_result.risk_level.lower(),
                    )
                    _sb_args = tc_args if isinstance(tc_args, dict) else {}
                    _sb_result = await _SB_REG.run(tc_name, _sb_args, _sb_ctx)
                    if _sb_result is not None:
                        tool_start.setdefault("security", {})["sandbox"] = _sb_result.to_dict()
                        if not _sb_result.safe:
                            _sb_v = PolicyViolation(
                                rule_name="sandbox-scope-exceeded",
                                action=PolicyAction.BLOCK,
                                reason=(
                                    f"Sandbox dry-run ({_sb_result.adapter}) revealed "
                                    f"unacceptable scope: {'; '.join(_sb_result.signals[:3])}"
                                ),
                                event_type="TOOL_CALL_START",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            logger.warning(
                                "[SANDBOX:BLOCK] adapter=%s tool=%s safe=False signals=%s session=%s",
                                _sb_result.adapter, tc_name, _sb_result.signals[:2], session_id,
                            )
                            if settings.violations_enabled:
                                self.transport.enqueue_violation(_sb_v.to_dict())
                            _response_violations.append(_sb_v)
                except Exception as _sb_exc:
                    logger.warning("Sandbox dry-run error (continuing): %s", _sb_exc)

            # --- Compound Sequence Detector (Phase 23) ---------------------------
            # Detects dangerous multi-step action sequences across the session.
            # A single tool call may be benign; the combination reveals intent.
            # Runs on every TOOL_CALL_START; O(patterns × history) — both small.
            # Skip if already blocked — intent history is still updated below.
            try:
                from .security.compound_detector import (
                    classify_tool_intent as _classify_intent,
                    compound_detector as _compound_det,
                )
                _tc_intent = _classify_intent(tc_name)
                state.intent_history.append(_tc_intent)
                # Bound history to prevent O(n²) pattern matching on very long sessions
                if len(state.intent_history) > 500:
                    state.intent_history = state.intent_history[-500:]
                # Always update history even when blocked; only evaluate patterns when not blocked
                _compound_matches = [] if _already_blocked else _compound_det.check(_tc_intent, state.intent_history)
                if _compound_matches:
                    tool_start.setdefault("security", {})["compound_sequences"] = [
                        m.to_dict() for m in _compound_matches
                    ]
                    for _cm in _compound_matches:
                        _cm_v = PolicyViolation(
                            rule_name="compound-sequence-violation",
                            action=PolicyAction.ALERT if _cm.severity == "medium"
                                   else PolicyAction.BLOCK,
                            reason=(
                                f"Compound sequence '{_cm.pattern_name}' detected "
                                f"({_cm.severity}): {_cm.description[:150]}"
                            ),
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        logger.warning(
                            "[COMPOUND-SEQ:%s] pattern=%s tool=%s session=%s",
                            _cm.severity.upper(), _cm.pattern_name, tc_name, session_id,
                        )
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(_cm_v.to_dict())
                        if _cm.severity in ("critical", "high"):
                            _response_violations.append(_cm_v)
            except Exception as _csd_exc:
                logger.warning("Compound sequence detector error (continuing): %s", _csd_exc)

            # --- Behavioral Baseline (Phase 20 — Anomaly Detection) --------------
            # Track per-tool call counts and flag statistical anomalies vs baseline.
            # No ML — Welford z-score on historical per-session call counts.
            # Requires min_observations sessions before baseline activates.
            # Counter always incremented; anomaly scored only when not already blocked.
            try:
                from .security.behavioral_baseline import (
                    baseline_store as _bl_store,
                )
                _tc_count = state.tool_call_counts.get(tc_name, 0) + 1
                state.tool_call_counts[tc_name] = _tc_count
                if _already_blocked:
                    raise StopIteration  # skip anomaly scoring, still updates counter
                _bl_anomaly = _bl_store.score_anomaly(
                    org_id, agent_id, tc_name, _tc_count
                )
                if _bl_anomaly.has_baseline and _bl_anomaly.is_anomalous:
                    tool_start.setdefault("security", {})["behavioral_anomaly"] = (
                        _bl_anomaly.to_dict()
                    )
                    _bl_v = PolicyViolation(
                        rule_name="behavioral-baseline-violation",
                        action=PolicyAction.BLOCK if _bl_anomaly.should_block
                               else PolicyAction.ALERT,
                        reason=_bl_anomaly.reason[:300],
                        event_type="TOOL_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    logger.warning(
                        "[BASELINE:%s] tool=%s count=%d z=%.1f session=%s",
                        "BLOCK" if _bl_anomaly.should_block else "ALERT",
                        tc_name, _tc_count, _bl_anomaly.z_score, session_id,
                    )
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(_bl_v.to_dict())
                    if _bl_anomaly.should_block:
                        _response_violations.append(_bl_v)
            except StopIteration:
                pass  # already blocked — skip anomaly scoring
            except Exception as _bl_exc:
                logger.warning("Behavioral baseline error (continuing): %s", _bl_exc)

            # --- Rollback Readiness Gate (Phase 21 — Reversibility Assessment) --
            # Assess whether the tool call can be undone if it causes harm.
            # Structural: verb taxonomy + argument signal checks. No ML, no regex.
            # Annotates tool_start["security"]["rollback"] for dashboard display.
            try:
                from .security.rollback_gate import assess_reversibility as _assess_rb
                _rb = _assess_rb(tc_name, _br_args)
                tool_start.setdefault("security", {})["rollback"] = _rb.to_dict()
                if _rb.is_irreversible:
                    logger.debug(
                        "[ROLLBACK:IRREVERSIBLE] tool=%r verb=%r session=%s",
                        tc_name, _rb.matched_verb, session_id,
                    )
            except Exception as _rb_exc:
                logger.warning("Rollback gate error (continuing): %s", _rb_exc)

            # --- Authorization Chain check (Phase 18) ─────────────────────────
            # Detect high-risk tool calls with no authorization in conversation history.
            # Only fires ALERT — never BLOCK on its own (false positive risk).
            # Cheap: frozenset lookups, no I/O.
            try:
                _ac_result = state.get_auth_chain().check_tool_call(tc_name)
                tool_start.setdefault("security", {})["auth_chain"] = _ac_result.to_dict()
                if not _ac_result.is_authorized:
                    logger.warning(
                        "[AUTH-CHAIN:ALERT] unauthorized-tool-call tool=%s intent=%s session=%s agent=%s",
                        tc_name, _ac_result.intent_class, session_id, agent_id,
                    )
                    if settings.violations_enabled:
                        _ac_v = PolicyViolation(
                            rule_name="unauthorized-tool-call",
                            action=PolicyAction.ALERT,
                            reason=_ac_result.reason,
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        self.transport.enqueue_violation(_ac_v.to_dict())
            except Exception as _ac_tool_exc:
                logger.debug("Auth chain tool check failed (skipped): %s", _ac_tool_exc)

            # --- HITL gate: create approval request if this tool needs one ------
            # Fires when: tool is in hitl_required_tools list, enforcement
            # scanner flagged RCE/SSRF, tainted creds are heading to a
            # network-sink tool, or blast radius is HIGH/CRITICAL.
            # The agent sees the tool call proceed — but the NEXT LLM call
            # will be held until a reviewer decides.
            if cfg.hitl_enabled and not state.hitl_pending_approval_id:
                _hitl_trigger: str | None = None
                if tc_name in settings.hitl_required_tools_set:
                    _hitl_trigger = "hitl_tool_list"
                elif _br_result is not None and _br_result.should_hitl:
                    _hitl_trigger = f"blast_radius_{_br_result.risk_level.lower()}"
                elif (
                    _br_result is not None
                    and state.blast_radius_cumulative > cfg.blast_radius_session_budget
                    and _br_result.verb_risk >= 0.30
                ):
                    _hitl_trigger = "blast_radius_budget_exceeded"
                elif tool_scan is not None and (tool_scan.rce_detected or tool_scan.ssrf_detected):
                    _hitl_trigger = "hitl_score_threshold"
                elif settings.hitl_on_network_sink and state.taint_tracker:
                    try:
                        _hits = state.taint_tracker.check_tool_call(
                            tc_name, tc_args if isinstance(tc_args, dict) else {}
                        )
                        if any(h.is_network_sink for h in _hits):
                            _hitl_trigger = "hitl_network_sink"
                    except Exception:
                        pass

                if _hitl_trigger:
                    try:
                        import httpx as _httpx
                        import json as _json
                        async with _httpx.AsyncClient(timeout=5.0) as _hc:
                            _resp = await _hc.post(
                                f"{settings.backend_url}/v1/approvals",
                                json={
                                    "session_id": session_id,
                                    "agent_id": agent_id,
                                    "tool_name": tc_name,
                                    "tool_args": tc_args if isinstance(tc_args, dict) else {},
                                    "trigger": _hitl_trigger,
                                    "timeout_seconds": settings.hitl_timeout_seconds,
                                },
                                headers={"X-API-Key": settings.backend_api_key},
                            )
                            if _resp.status_code == 200:
                                state.hitl_pending_approval_id = _resp.json().get("id")
                                logger.info(
                                    "[HITL] Approval created: id=%s tool=%s trigger=%s session=%s",
                                    state.hitl_pending_approval_id, tc_name, _hitl_trigger, session_id,
                                )
                    except Exception as _he:
                        logger.warning("[HITL] Failed to create approval request: %s", _he)

            # --- Hook 3: Data flow taint check (Phase 8) -------------------------
            # Detect credential exfiltration: a tainted value (from system prompt
            # or tool result) appearing verbatim in a tool call argument.
            # Network-sink tools: BLOCK.  Other tools: ALERT.
            if cfg.taint_tracking_enabled and state.taint_tracker:
                try:
                    taint_hits = state.taint_tracker.check_tool_call(tc_name, tc_args if isinstance(tc_args, dict) else {})
                    for hit in taint_hits:
                        if hit.is_network_sink:
                            tv = PolicyViolation(
                                rule_name="data-exfiltration-attempt",
                                action=PolicyAction.BLOCK,
                                reason=(
                                    f"Tainted credential ({hit.label} from {hit.source}) "
                                    f"found in arg '{hit.arg_key}' of network-sink tool '{hit.tool_name}'"
                                ),
                                event_type="TOOL_CALL_START",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            logger.warning(
                                "[TAINT:BLOCK] data-exfiltration-attempt tool=%s arg=%s label=%s source=%s session=%s",
                                hit.tool_name, hit.arg_key, hit.label, hit.source, session_id,
                            )
                            if settings.violations_enabled:
                                self.transport.enqueue_violation(tv.to_dict())
                            _response_violations.append(tv)
                            continue  # Skip this tool call — violation recorded
                        else:
                            av = PolicyViolation(
                                rule_name="data-flow-suspicious",
                                action=PolicyAction.ALERT,
                                reason=(
                                    f"Tainted credential ({hit.label} from {hit.source}) "
                                    f"found in arg '{hit.arg_key}' of tool '{hit.tool_name}'"
                                ),
                                event_type="TOOL_CALL_START",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            logger.warning(
                                "[TAINT:ALERT] data-flow-suspicious tool=%s arg=%s label=%s source=%s session=%s",
                                hit.tool_name, hit.arg_key, hit.label, hit.source, session_id,
                            )
                            if settings.violations_enabled:
                                self.transport.enqueue_violation(av.to_dict())
                except Exception as _te:
                    logger.debug("Taint check error (skipped): %s", _te)

            # --- Hook 3-IFC: IFC label flow check (Phase 11) ----------------------
            # Deterministic check: did EXTERNAL-labeled content flow into a
            # privileged sink argument? No ML, no heuristics — pure label math.
            if cfg.ifc_enabled and state.ifc_context:
                try:
                    _extra_sinks: frozenset[str] | None = None
                    if settings.security_ifc_extra_sink_keys:
                        _extra_sinks = frozenset(
                            k.strip() for k in settings.security_ifc_extra_sink_keys.split(",")
                            if k.strip()
                        )
                    _ifc_violations = state.ifc_context.check_tool_call(
                        tc_name,
                        tc_args if isinstance(tc_args, dict) else {},
                        extra_sink_keys=_extra_sinks,
                    )
                    for _iv in _ifc_violations:
                        _ifc_action = PolicyAction.BLOCK if (cfg.ifc_mode == 'block') else PolicyAction.ALERT
                        _ifc_v = PolicyViolation(
                            rule_name="ifc-external-to-sink",
                            action=_ifc_action,
                            reason=(
                                f"IFC: EXTERNAL-labeled fragment (from {_iv.source_id}) "
                                f"found in privileged sink arg '{_iv.arg_key}' of tool '{_iv.tool_name}'. "
                                f"Possible injection-driven exfiltration."
                            ),
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        logger.warning(
                            "[IFC:%s] ifc-external-to-sink tool=%s arg=%s source=%s session=%s",
                            _ifc_action.value, _iv.tool_name, _iv.arg_key, _iv.source_id, session_id,
                        )
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(_ifc_v.to_dict())
                        if _ifc_action == PolicyAction.BLOCK:
                            _response_violations.append(_ifc_v)
                            tool_start.setdefault("security", {}).update(
                                state.ifc_context.to_summary_dict()
                            )
                            tool_start["security"]["ifc_violation"] = {
                                "arg_key": _iv.arg_key,
                                "source_id": _iv.source_id,
                                "fragment": _iv.fragment,
                            }
                        else:
                            tool_start.setdefault("security", {}).update(
                                state.ifc_context.to_summary_dict()
                            )
                except Exception as _ie:
                    logger.debug("IFC check error (skipped): %s", _ie)

            # --- Hook 3-PDG: Session PDG data-flow check (Phase 10) ---------------
            # Detect multi-hop exfiltration: untrusted fragment from a prior tool
            # result appearing in a sensitive argument of the current tool call.
            # This catches chains the taint tracker misses (non-credential values
            # like email addresses, URLs, and domain names from web search results).
            if cfg.pdg_enabled and state.pdg:
                try:
                    _pdg_edges = state.pdg.check_tool_call(
                        tc_name, tc_args if isinstance(tc_args, dict) else {}
                    )
                    if _pdg_edges:
                        _pdg_action = PolicyAction.BLOCK if cfg.pdg_mode == "block" else PolicyAction.ALERT
                        _pdg_v = PolicyViolation(
                            rule_name="data-flow-graph-violation",
                            action=_pdg_action,
                            reason=(
                                f"PDG: {len(_pdg_edges)} untrusted data-flow edge(s) detected. "
                                f"Tool '{tc_name}' received fragments from untrusted sources: "
                                + ", ".join(
                                    f"{e.source_fragment[:40]!r} → {e.sink_arg}"
                                    for e in _pdg_edges[:3]
                                )
                            ),
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        logger.warning(
                            "[PDG:%s] data-flow-graph-violation tool=%s edges=%d session=%s",
                            _pdg_action.value, tc_name, len(_pdg_edges), session_id,
                        )
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(_pdg_v.to_dict())
                        # Enrich tool_start security field with edge data
                        tool_start.setdefault("security", {}).update({
                            "pdg_edges":   [e.to_dict() for e in _pdg_edges],
                            "pdg_summary": state.pdg.to_summary_dict()["pdg_summary"],
                        })
                        if _pdg_action == PolicyAction.BLOCK:
                            _response_violations.append(_pdg_v)
                    else:
                        # No violations — still record PDG summary for observability
                        tool_start.setdefault("security", {}).update(
                            state.pdg.to_summary_dict()
                        )
                except Exception as _pdgerr:
                    logger.debug("PDG check error (skipped): %s", _pdgerr)

            # --- Hard Egress Enforcement (Phase 13) -------------------------------
            # Deterministic allowlist-based network destination check.
            # Any tool call whose destination arg is not in the allowlist → BLOCK.
            # Mode "audit" logs only; mode "allowlist" blocks; mode "off" skips.
            if settings.security_egress_mode != "off":
                try:
                    from .security.egress_enforcer import (
                        EgressEnforcer as _EgressEnforcer,
                        get_egress_allowlist as _get_egress_allowlist,
                    )
                    _egress_allowed = await _get_egress_allowlist(self.org_id or "default-org")
                    _egress = _EgressEnforcer(_egress_allowed)
                    # Skip if no allowlist configured and not in audit mode
                    if not _egress.is_empty() or settings.security_egress_mode == "audit":
                        _egress_violations = _egress.check_tool_call(
                            tc_name,
                            tc_args if isinstance(tc_args, dict) else {},
                        )
                        for _ev in _egress_violations:
                            _is_blocking = (
                                settings.security_egress_mode == "allowlist"
                                and not _egress.is_empty()
                            )
                            _eg_action = PolicyAction.BLOCK if _is_blocking else PolicyAction.ALERT
                            _eg_v = PolicyViolation(
                                rule_name="egress-destination-not-allowed",
                                action=_eg_action,
                                reason=(
                                    f"Egress enforcement: tool '{_ev.tool_name}' arg "
                                    f"'{_ev.arg_key}' targets '{_ev.destination}' "
                                    f"which is not in the approved network destination allowlist."
                                ),
                                event_type="TOOL_CALL_START",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            logger.warning(
                                "[EGRESS:%s] egress-destination-not-allowed tool=%s dest=%s session=%s",
                                _eg_action.value, _ev.tool_name, _ev.destination, session_id,
                            )
                            if settings.violations_enabled:
                                self.transport.enqueue_violation(_eg_v.to_dict())
                            if _is_blocking:
                                _response_violations.append(_eg_v)
                except Exception as _egresserr:
                    logger.debug("Egress enforcement error (skipped): %s", _egresserr)

            # --- Capability Manifest Validation (Phase 12) ------------------------
            # Cryptographically signed manifest defines exactly what this agent
            # is permitted to do. If a manifest is active, any tool call not in
            # the manifest, or with arguments violating permitted patterns, is blocked.
            # Agents with no manifest (learning mode) are not blocked here.
            if settings.manifest_signing_key is not None:  # always enabled if config present
                try:
                    from .security.capability_manifest import (
                        get_active_manifest as _get_manifest,
                        ManifestValidator as _ManifestValidator,
                    )
                    _manifest = await _get_manifest(
                        agent_id, self.org_id,
                        settings.backend_url, settings.backend_api_key,
                        signing_key=settings.manifest_signing_key,
                        cache_ttl_s=settings.manifest_cache_ttl_s,
                    )
                    if _manifest is not None:
                        _mv = _ManifestValidator(_manifest)
                        _manifest_violations = _mv.check_tool_call(
                            tc_name, tc_args if isinstance(tc_args, dict) else {}
                        )
                        for _mvi in _manifest_violations:
                            _mf_action = PolicyAction.BLOCK if cfg.manifest_enforce_enabled else PolicyAction.ALERT
                            _mf_v = PolicyViolation(
                                rule_name="manifest-violation",
                                action=_mf_action,
                                reason=(
                                    f"Manifest: {_mvi.detail} "
                                    f"[manifest {_manifest.manifest_id}]"
                                ),
                                event_type="TOOL_CALL_START",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            logger.warning(
                                "[MANIFEST:%s] %s tool=%s arg=%s session=%s manifest=%s",
                                _mf_action.value, _mvi.violation_type,
                                _mvi.tool_name, _mvi.arg_key, session_id, _manifest.manifest_id,
                            )
                            if settings.violations_enabled:
                                self.transport.enqueue_violation(_mf_v.to_dict())
                            if _mf_action == PolicyAction.BLOCK:
                                _response_violations.append(_mf_v)
                                tool_start.setdefault("security", {})["manifest_violation"] = {
                                    "type": _mvi.violation_type,
                                    "arg_key": _mvi.arg_key,
                                    "detail": _mvi.detail,
                                    "manifest_id": _manifest.manifest_id,
                                }
                except Exception as _mferr:
                    logger.debug("Manifest validation error (skipped): %s", _mferr)

            # --- Tool baseline enforcement ----------------------------------------
            # If the operator has approved a baseline for this agent, any tool
            # name not in that approved set is blocked immediately.
            # Agents with no approved baseline are in audit mode (not blocked).
            if settings.tool_baseline_enabled:
                try:
                    from .security.tool_baseline import get_approved_baseline as _get_bl
                    _approved = await _get_bl(
                        agent_id, self.org_id,
                        settings.backend_url, settings.backend_api_key,
                        cache_ttl_s=settings.tool_baseline_cache_ttl_s,
                    )
                    if _approved is not None and tc_name not in _approved:
                        logger.warning(
                            "[TOOL-BASELINE:BLOCK] unapproved tool=%s agent=%s session=%s",
                            tc_name, agent_id, session_id,
                        )
                        _bl_v = PolicyViolation(
                            rule_name="tool-not-in-baseline",
                            action=PolicyAction.BLOCK,
                            reason=(
                                f"Tool '{tc_name}' has not been approved for agent "
                                f"'{agent_id}'. Review and approve it in the Baselines "
                                f"dashboard to allow this tool."
                            ),
                            event_type="TOOL_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                            timestamp_ns=time.time_ns(),
                        )
                        if settings.violations_enabled:
                            self.transport.enqueue_violation(_bl_v.to_dict())
                        _response_violations.append(_bl_v)
                        continue  # Skip this tool call
                except Exception as _ble:
                    logger.debug("Tool baseline enforcement error (skipped): %s", _ble)

            # --- Tool permissions check (Phase 3.1 Iteration 3) ---
            # Evaluated BEFORE policy engine and BEFORE tool_call_count increment
            # so that blocked calls are not counted toward the session quota.
            tp_violations = get_tool_permissions_engine().check(
                tc_name, agent_id, tc_args,
                session_id=session_id,
                org_id=self.org_id,
            )
            for v in tp_violations:
                if v.action in (PolicyAction.ALERT, PolicyAction.BLOCK):
                    logger.warning(
                        "[TOOL-PERM:%s] rule=%r tool=%s session=%s agent=%s reason=%r",
                        v.action.value, v.rule_name, tc_name,
                        session_id, agent_id, v.reason,
                    )
                if v.action == PolicyAction.ALERT and settings.violations_enabled:
                    self.transport.enqueue_violation(v.to_dict())
            tp_block = [v for v in tp_violations if v.action == PolicyAction.BLOCK]
            if tp_block:
                # Enqueue the BLOCK violation record before skipping
                if settings.violations_enabled:
                    self.transport.enqueue_violation(tp_block[0].to_dict())
                continue  # Skip this tool call — don't emit or track it

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
        return _response_violations
