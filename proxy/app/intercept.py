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
from .tool_permissions import get_tool_permissions_engine
from .transport import get_transport, EventTransport

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


def _hash_prompt(text: str) -> str:
    """Return a short SHA-256 hex digest of a string."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class InterceptContext:
    """
    Stateless per-request context. Session state is held in SessionTracker.
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
        self.transport = transport if transport is not None else get_transport()
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
            if v.action in (PolicyAction.ALERT, PolicyAction.BLOCK) and settings.violations_enabled:
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
        # Block agents spawned beyond the configured max depth to prevent
        # runaway delegation chains and trust-boundary violations.
        max_depth = settings.max_spawn_depth
        if max_depth > 0 and state.spawn_depth > max_depth:
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
            return session_id, str(uuid.uuid4()), [sv], None

        # --- Phase 4: Check async ML injection flag from previous turn -------
        # The async classifier runs AFTER forwarding. If it found injection in
        # the previous turn it sets state.ml_injection_flag = True. We check
        # that flag here and BLOCK the current call before the LLM sees it.
        # This defends against multi-turn attack chains (e.g. inject turn 1 to
        # exfil data in turn 2) without adding latency to the current call.
        if settings.analysis_classifier_enabled and state.ml_injection_flag:
            state.ml_injection_flag = False   # consume — only fires once
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
            return session_id, str(uuid.uuid4()), [v], None

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
                        from .security.tool_output_scanner import scan as _scan_tool_output
                        loop = asyncio.get_running_loop()
                        tool_out_result = await loop.run_in_executor(
                            None,
                            lambda name=pending["tool_name"], content=tr["content"]: _scan_tool_output(name, content),
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
                        if settings.security_tool_output_ml_scan_enabled and tr.get("content"):
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

                # Hook 2: Taint credentials returned by this tool result
                if settings.security_taint_tracking_enabled and tr.get("content"):
                    try:
                        from .security.taint_tracker import extract_credentials_from_text
                        tool_source = f"tool_result:{pending.get('tool_name', 'unknown')}"
                        for cred_val, cred_label in extract_credentials_from_text(str(tr["content"])):
                            state.get_taint_tracker().taint(cred_val, cred_label, tool_source)
                    except Exception as _te:
                        logger.debug("Taint extraction from tool result failed (skipped): %s", _te)

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
            messages=messages,  # original messages -- canary is NOT in the audit log
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

        # Enrich event with spawn chain metadata so topology + audit can use it
        if state.spawn_depth > 0:
            start_event.setdefault("payload", {}).update({
                "spawn_depth": state.spawn_depth,
                "parent_agent_id": state.parent_agent_id,
                "parent_session_id": state.parent_session_id,
            })

        # --- System prompt mutation detection (Phase 6) ----------------------
        # Hash the system prompt on the first call (baseline) and compare on
        # every subsequent call within the same session.  A mid-session change
        # is a near-certain indicator of a successful prompt injection or a
        # compromised orchestrator swapping instructions between turns.
        system_prompt = _extract_system_prompt(request_data)
        if system_prompt:
            new_hash = _hash_prompt(system_prompt)
            if state.system_prompt_hash is None:
                # Establish baseline — no violation on first call
                state.system_prompt_hash = new_hash
                logger.debug(
                    "[SYS-PROMPT] baseline stored hash=%s session=%s",
                    new_hash, session_id,
                )
                # Hook 1: Taint credentials found in system prompt
                if settings.security_taint_tracking_enabled:
                    try:
                        from .security.taint_tracker import extract_credentials_from_text
                        for cred_val, cred_label in extract_credentials_from_text(system_prompt):
                            state.get_taint_tracker().taint(cred_val, cred_label, "system_prompt")
                    except Exception as _te:
                        logger.debug("Taint extraction from system prompt failed (skipped): %s", _te)
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
        if settings.security_mcp_scanning_enabled and tools:
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
            if settings.rate_limit_max_tokens_per_session > 0:
                if state.total_tokens >= settings.rate_limit_max_tokens_per_session:
                    logger.warning(
                        "[RATE-LIMIT:BLOCK] token budget exhausted: total=%d cap=%d session=%s",
                        state.total_tokens, settings.rate_limit_max_tokens_per_session, session_id,
                    )
                    rv = PolicyViolation(
                        rule_name="token-budget-exceeded",
                        action=PolicyAction.BLOCK,
                        reason=(
                            f"Session token budget exhausted: {state.total_tokens} tokens used, "
                            f"cap is {settings.rate_limit_max_tokens_per_session}"
                        ),
                        event_type="LLM_CALL_START",
                        session_id=session_id,
                        agent_id=agent_id,
                        org_id=self.org_id,
                    )
                    violations.append(rv)
                    if settings.violations_enabled:
                        self.transport.enqueue_violation(rv.to_dict())
            if settings.rate_limit_max_llm_calls_per_session > 0:
                if state.llm_call_count >= settings.rate_limit_max_llm_calls_per_session:
                    logger.warning(
                        "[RATE-LIMIT:BLOCK] LLM call budget exhausted: calls=%d cap=%d session=%s",
                        state.llm_call_count, settings.rate_limit_max_llm_calls_per_session, session_id,
                    )
                    rcv = PolicyViolation(
                        rule_name="llm-call-budget-exceeded",
                        action=PolicyAction.BLOCK,
                        reason=(
                            f"Session LLM call budget exhausted: {state.llm_call_count} calls, "
                            f"cap is {settings.rate_limit_max_llm_calls_per_session}"
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
            except Exception:
                pass
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

        # --- Markov sequence tracking (Phase 3.3) ---
        # Score the LLM_CALL_START transition, then record it.
        if settings.security_behavioral_enabled and state.event_type_sequence:
            try:
                from .security.markov import score_transition, observe_transition
                prev_evt = state.event_type_sequence[-1]
                markov_result = score_transition(
                    prev_evt, "LLM_CALL_START", agent_id,
                    threshold=settings.security_markov_alert_threshold,
                )
                observe_transition(prev_evt, "LLM_CALL_START", agent_id)
                if markov_result.is_anomaly:
                    logger.warning(
                        "[MARKOV:ALERT] %s->LLM_CALL_START prob=%.4f session=%s agent=%s",
                        prev_evt, markov_result.probability, session_id, agent_id,
                    )
                    if settings.violations_enabled:
                        mv = PolicyViolation(
                            rule_name="markov-anomalous-transition",
                            action=PolicyAction.ALERT,
                            reason=(
                                f"Anomalous event sequence: {prev_evt}->LLM_CALL_START "
                                f"P={markov_result.probability:.4f}"
                            ),
                            event_type="LLM_CALL_START",
                            session_id=session_id,
                            agent_id=agent_id,
                            org_id=self.org_id,
                        )
                        self.transport.enqueue_violation(mv.to_dict())
            except Exception as exc:
                logger.debug("Markov tracking error (skipped): %s", exc)
        state.event_type_sequence.append("LLM_CALL_START")

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
        if settings.security_canary_enabled:
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
        if settings.analysis_classifier_enabled:
            asyncio.create_task(
                self._classify_messages_async(messages, state)
            )

        return session_id, run_id, violations, forward_body

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

            text = " ".join(segments)[:2048]  # cap for speed

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: _ml_classify(
                    text,
                    model_name=settings.analysis_classifier_model,
                    threshold=settings.analysis_classifier_threshold,
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
        state: SessionState = self.session_tracker.get_state(session_id)
        _response_violations: list[PolicyViolation] = []
        seq = state.sequence_number
        prev_hash = state.last_hash

        if http_status >= 400:
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

        # Accumulate token usage for rate-limiting / budget enforcement
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

        # --- Markov: record LLM_CALL_END transition ---
        if settings.security_behavioral_enabled and state.event_type_sequence:
            try:
                from .security.markov import score_transition, observe_transition
                prev_evt = state.event_type_sequence[-1]
                observe_transition(prev_evt, "LLM_CALL_END", agent_id)
            except Exception as exc:
                logger.debug("Markov LLM_CALL_END tracking error (skipped): %s", exc)
        state.event_type_sequence.append("LLM_CALL_END")

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

            # --- Hook 3: Data flow taint check (Phase 8) -------------------------
            # Detect credential exfiltration: a tainted value (from system prompt
            # or tool result) appearing verbatim in a tool call argument.
            # Network-sink tools: BLOCK.  Other tools: ALERT.
            if settings.security_taint_tracking_enabled and state.taint_tracker:
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

            # --- Markov: record TOOL_CALL_START transition ---
            if settings.security_behavioral_enabled and state.event_type_sequence:
                try:
                    from .security.markov import score_transition, observe_transition
                    prev_evt = state.event_type_sequence[-1]
                    markov_tc = score_transition(
                        prev_evt, "TOOL_CALL_START", agent_id,
                        threshold=settings.security_markov_alert_threshold,
                    )
                    observe_transition(prev_evt, "TOOL_CALL_START", agent_id)
                    state.event_type_sequence.append("TOOL_CALL_START")
                    if markov_tc.is_anomaly:
                        logger.warning(
                            "[MARKOV:ALERT] %s->TOOL_CALL_START prob=%.4f session=%s agent=%s",
                            prev_evt, markov_tc.probability, session_id, agent_id,
                        )
                except Exception as exc:
                    logger.debug("Markov TOOL_CALL_START tracking error (skipped): %s", exc)

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

            # --- Markov: record AGENT_FINISH transition ---
            if settings.security_behavioral_enabled and state.event_type_sequence:
                try:
                    from .security.markov import score_transition, observe_transition
                    prev_evt = state.event_type_sequence[-1]
                    markov_result = score_transition(
                        prev_evt, "AGENT_FINISH", agent_id,
                        threshold=settings.security_markov_alert_threshold,
                    )
                    observe_transition(prev_evt, "AGENT_FINISH", agent_id)
                    state.event_type_sequence.append("AGENT_FINISH")
                    if markov_result.is_anomaly:
                        logger.warning(
                            "[MARKOV:ALERT] %s->AGENT_FINISH prob=%.4f session=%s agent=%s",
                            prev_evt, markov_result.probability, session_id, agent_id,
                        )
                        if settings.violations_enabled:
                            mv = PolicyViolation(
                                rule_name="markov-anomalous-transition",
                                action=PolicyAction.ALERT,
                                reason=(
                                    f"Anomalous event sequence: {prev_evt}->AGENT_FINISH "
                                    f"P={markov_result.probability:.4f}"
                                ),
                                event_type="AGENT_FINISH",
                                session_id=session_id,
                                agent_id=agent_id,
                                org_id=self.org_id,
                            )
                            self.transport.enqueue_violation(mv.to_dict())
                except Exception as exc:
                    logger.debug("Markov AGENT_FINISH tracking error (skipped): %s", exc)

            # --- Isolation Forest: score session at completion (Phase 3.3) ---
            if settings.security_behavioral_enabled:
                try:
                    from .security.isolation_forest import fit_and_score as _if_score
                    loop = asyncio.get_running_loop()
                    error_rate = (
                        float(state.error_count) / max(state.llm_call_count, 1)
                    )
                    if_features = {
                        "llm_call_count":       float(state.llm_call_count),
                        "tool_call_rate":       float(state.tool_call_count) / max(state.llm_call_count, 1),
                        "error_rate":           error_rate,
                        "session_duration_min": (time.time_ns() - state.started_at_ns) / 6e10,
                        "max_injection_score":  state.max_injection_score,
                    }
                    if_result = await loop.run_in_executor(
                        None,
                        lambda features=if_features: _if_score(features),
                    )
                    if if_result is not None:
                        finish_event.setdefault("security", {})["isolation_forest"] = if_result.to_dict()
                        if if_result.is_anomaly:
                            logger.warning(
                                "[ISOLATION-FOREST:ALERT] anomaly_score=%.3f session=%s agent=%s",
                                if_result.anomaly_score, session_id, agent_id,
                            )
                            if settings.violations_enabled:
                                ifv = PolicyViolation(
                                    rule_name="isolation-forest-anomaly",
                                    action=PolicyAction.ALERT,
                                    reason=(
                                        f"Session behavioral anomaly: "
                                        f"score={if_result.anomaly_score:.3f} "
                                        f"samples_seen={if_result.samples_seen}"
                                    ),
                                    event_type="AGENT_FINISH",
                                    session_id=session_id,
                                    agent_id=agent_id,
                                    org_id=self.org_id,
                                )
                                self.transport.enqueue_violation(ifv.to_dict())
                except Exception as exc:
                    logger.debug("Isolation Forest scoring error (skipped): %s", exc)

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
