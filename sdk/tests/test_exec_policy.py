"""
Tests for the Policy Engine (sdk/aegivis/security/exec_policy.py).

Covers: evaluate_policy, GateConfig, scan_return_value, ToolExecutionBlocked,
ToolExecutionTimeout, HITL mocking (no real network calls).
"""
from __future__ import annotations

import pytest

from aegivis.security.arg_classifier import ArgSignal, ClassifierConfig, Signal
from aegivis.security.exec_policy import (
    GateConfig,
    PolicyAction,
    PolicyDecision,
    ToolExecutionBlocked,
    ToolExecutionTimeout,
    evaluate_policy,
    scan_return_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_signal(signal_type: str, path: str = "args[0]") -> ArgSignal:
    return ArgSignal(signal=signal_type, path=path, preview="preview")


# ---------------------------------------------------------------------------
# ToolExecutionBlocked
# ---------------------------------------------------------------------------


class TestToolExecutionBlocked:
    def test_inherits_runtime_error(self):
        exc = ToolExecutionBlocked("my_tool", "test reason", [])
        assert isinstance(exc, RuntimeError)

    def test_attributes(self):
        sigs = [make_signal(Signal.CREDENTIAL)]
        exc = ToolExecutionBlocked("send_email", "blocked", sigs)
        assert exc.tool_name == "send_email"
        assert exc.reason == "blocked"
        assert exc.signals == sigs

    def test_message_contains_tool_name(self):
        exc = ToolExecutionBlocked("transfer_funds", "bad signal", [])
        assert "transfer_funds" in str(exc)

    def test_empty_signals_list(self):
        exc = ToolExecutionBlocked("tool", "reason", [])
        assert exc.signals == []


# ---------------------------------------------------------------------------
# ToolExecutionTimeout
# ---------------------------------------------------------------------------


class TestToolExecutionTimeout:
    def test_inherits_runtime_error(self):
        exc = ToolExecutionTimeout("slow_tool", 30.0)
        assert isinstance(exc, RuntimeError)

    def test_attributes(self):
        exc = ToolExecutionTimeout("fetch_data", 10.0)
        assert exc.tool_name == "fetch_data"
        assert exc.timeout_s == 10.0

    def test_message_contains_tool_name_and_timeout(self):
        exc = ToolExecutionTimeout("download", 5.5)
        assert "download" in str(exc)
        assert "5.5" in str(exc)


# ---------------------------------------------------------------------------
# evaluate_policy — ALLOW
# ---------------------------------------------------------------------------


class TestEvaluatePolicyAllow:
    def test_no_signals_allows(self):
        cfg = GateConfig(block_on=frozenset({"credential"}), alert_on=frozenset({"email_destination"}))
        result = evaluate_policy([], cfg)
        assert result.action == PolicyAction.ALLOW

    def test_irrelevant_signals_allow(self):
        cfg = GateConfig(block_on=frozenset({"credential"}), alert_on=frozenset({"email_destination"}))
        sigs = [make_signal(Signal.FILE_PATH)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.ALLOW

    def test_empty_config_always_allows(self):
        cfg = GateConfig()  # empty block_on, alert_on
        sigs = [make_signal(Signal.CREDENTIAL), make_signal(Signal.EMAIL_DESTINATION)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.ALLOW
        assert result.triggered_by == []

    def test_allow_has_empty_triggered_by(self):
        result = evaluate_policy([], GateConfig())
        assert result.triggered_by == []


# ---------------------------------------------------------------------------
# evaluate_policy — BLOCK
# ---------------------------------------------------------------------------


class TestEvaluatePolicyBlock:
    def test_block_on_credential(self):
        cfg = GateConfig(block_on=frozenset({Signal.CREDENTIAL}))
        sigs = [make_signal(Signal.CREDENTIAL)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.BLOCK

    def test_block_triggered_by_contains_matching_signal(self):
        cfg = GateConfig(block_on=frozenset({Signal.CREDENTIAL}))
        sigs = [make_signal(Signal.CREDENTIAL), make_signal(Signal.FILE_PATH)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.BLOCK
        assert any(s.signal == Signal.CREDENTIAL for s in result.triggered_by)

    def test_block_only_lists_matching_signals(self):
        cfg = GateConfig(block_on=frozenset({Signal.CREDENTIAL}))
        sigs = [make_signal(Signal.FILE_PATH), make_signal(Signal.CREDENTIAL)]
        result = evaluate_policy(sigs, cfg)
        # triggered_by should only contain the credential signal, not file_path
        assert all(s.signal == Signal.CREDENTIAL for s in result.triggered_by)

    def test_block_on_multiple_types(self):
        cfg = GateConfig(block_on=frozenset({Signal.CREDENTIAL, Signal.EMAIL_DESTINATION}))
        sigs = [make_signal(Signal.EMAIL_DESTINATION)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.BLOCK

    def test_block_reason_mentions_signal(self):
        cfg = GateConfig(block_on=frozenset({Signal.CREDENTIAL}))
        sigs = [make_signal(Signal.CREDENTIAL)]
        result = evaluate_policy(sigs, cfg)
        assert "credential" in result.reason.lower()

    def test_returns_policy_decision_type(self):
        result = evaluate_policy([], GateConfig())
        assert isinstance(result, PolicyDecision)


# ---------------------------------------------------------------------------
# evaluate_policy — ALERT
# ---------------------------------------------------------------------------


class TestEvaluatePolicyAlert:
    def test_alert_on_email(self):
        cfg = GateConfig(alert_on=frozenset({Signal.EMAIL_DESTINATION}))
        sigs = [make_signal(Signal.EMAIL_DESTINATION)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.ALERT

    def test_alert_triggered_by_contains_matching(self):
        cfg = GateConfig(alert_on=frozenset({Signal.NETWORK_DESTINATION}))
        sigs = [make_signal(Signal.NETWORK_DESTINATION)]
        result = evaluate_policy(sigs, cfg)
        assert len(result.triggered_by) == 1
        assert result.triggered_by[0].signal == Signal.NETWORK_DESTINATION

    def test_alert_does_not_fire_for_non_matching(self):
        cfg = GateConfig(alert_on=frozenset({Signal.EMAIL_DESTINATION}))
        sigs = [make_signal(Signal.FILE_PATH)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.ALLOW


# ---------------------------------------------------------------------------
# evaluate_policy — Priority (BLOCK > ALERT)
# ---------------------------------------------------------------------------


class TestEvaluatePolicyPriority:
    def test_block_wins_over_alert_same_signal(self):
        # Same signal in both block_on and alert_on → BLOCK
        cfg = GateConfig(
            block_on=frozenset({Signal.CREDENTIAL}),
            alert_on=frozenset({Signal.CREDENTIAL}),
        )
        sigs = [make_signal(Signal.CREDENTIAL)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.BLOCK

    def test_block_wins_when_two_signals_one_each(self):
        # credential → block, email → alert; should still block
        cfg = GateConfig(
            block_on=frozenset({Signal.CREDENTIAL}),
            alert_on=frozenset({Signal.EMAIL_DESTINATION}),
        )
        sigs = [make_signal(Signal.EMAIL_DESTINATION), make_signal(Signal.CREDENTIAL)]
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.BLOCK

    def test_alert_only_when_no_block_signals(self):
        cfg = GateConfig(
            block_on=frozenset({Signal.CREDENTIAL}),
            alert_on=frozenset({Signal.EMAIL_DESTINATION}),
        )
        sigs = [make_signal(Signal.EMAIL_DESTINATION)]  # no credential
        result = evaluate_policy(sigs, cfg)
        assert result.action == PolicyAction.ALERT


# ---------------------------------------------------------------------------
# GateConfig defaults
# ---------------------------------------------------------------------------


class TestGateConfig:
    def test_default_block_on_empty(self):
        cfg = GateConfig()
        assert len(cfg.block_on) == 0

    def test_default_alert_on_empty(self):
        cfg = GateConfig()
        assert len(cfg.alert_on) == 0

    def test_default_timeout_none(self):
        cfg = GateConfig()
        assert cfg.timeout_s is None

    def test_default_hitl_false(self):
        cfg = GateConfig()
        assert cfg.hitl is False

    def test_default_scan_return_true(self):
        cfg = GateConfig()
        assert cfg.scan_return is True

    def test_custom_values(self):
        cfg = GateConfig(
            block_on=frozenset({"credential"}),
            alert_on=frozenset({"email_destination"}),
            timeout_s=30.0,
            hitl=True,
            hitl_timeout_s=60.0,
            scan_return=False,
        )
        assert "credential" in cfg.block_on
        assert "email_destination" in cfg.alert_on
        assert cfg.timeout_s == 30.0
        assert cfg.hitl is True
        assert cfg.hitl_timeout_s == 60.0
        assert cfg.scan_return is False


# ---------------------------------------------------------------------------
# scan_return_value
# ---------------------------------------------------------------------------


class TestScanReturnValue:
    def setup_method(self):
        self.cfg = ClassifierConfig()

    def test_credential_in_string_return(self):
        secret = "sk-abcdefghijklmnopqrstuvwxy"
        findings = scan_return_value(secret, self.cfg)
        assert len(findings) >= 1
        assert findings[0]["path"] == "$"

    def test_credential_in_dict_return(self):
        result = {"api_key": "sk-abcdefghijklmnopqrstuvwxy", "status": "ok"}
        findings = scan_return_value(result, self.cfg)
        assert any("api_key" in f["path"] for f in findings)

    def test_credential_in_nested_dict(self):
        result = {"outer": {"inner": {"token": "sk-abcdefghijklmnopqrstuvwxy"}}}
        findings = scan_return_value(result, self.cfg)
        assert len(findings) >= 1

    def test_credential_in_list_return(self):
        result = ["normal string", "sk-abcdefghijklmnopqrstuvwxy"]
        findings = scan_return_value(result, self.cfg)
        assert len(findings) >= 1

    def test_no_credential_clean_return(self):
        result = {"status": "success", "count": 42}
        findings = scan_return_value(result, self.cfg)
        assert findings == []

    def test_none_return_no_crash(self):
        findings = scan_return_value(None, self.cfg)
        assert findings == []

    def test_integer_return_no_crash(self):
        findings = scan_return_value(42, self.cfg)
        assert findings == []

    def test_finding_has_path_and_preview(self):
        result = {"key": "sk-abcdefghijklmnopqrstuvwxy"}
        findings = scan_return_value(result, self.cfg)
        assert len(findings) >= 1
        f = findings[0]
        assert "path" in f
        assert "preview" in f

    def test_large_list_sampled(self):
        # scan_return_value samples up to 20 items from lists
        result = ["safe string"] * 100
        findings = scan_return_value(result, self.cfg)
        assert findings == []  # no credentials

    def test_preview_does_not_expose_full_secret(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz012345"
        findings = scan_return_value(secret, self.cfg)
        assert len(findings) >= 1
        # Preview should be much shorter than the secret
        assert len(findings[0]["preview"]) < len(secret)


# ---------------------------------------------------------------------------
# HITL — fail-open behavior (no real backend)
# ---------------------------------------------------------------------------


class TestHITLFailOpen:
    """
    All HITL tests use monkeypatching to simulate network failures.
    The gate must fail-open (not raise) when the backend is unreachable.
    """

    def test_check_hitl_sync_fails_open_on_no_backend(self):
        from aegivis.security.exec_policy import check_hitl_sync

        # Pass an empty backend_url — should return silently (fail open)
        check_hitl_sync(
            tool_name="test_tool",
            signals=[],
            backend_url="",
            api_key="key",
            session_id="s1",
            agent_id="a1",
            org_id="org1",
            timeout_s=5.0,
        )
        # No exception = pass

    @pytest.mark.asyncio
    async def test_check_hitl_async_fails_open_on_no_backend(self):
        from aegivis.security.exec_policy import check_hitl_async

        await check_hitl_async(
            tool_name="test_tool",
            signals=[],
            backend_url="",
            api_key="key",
            session_id="s1",
            agent_id="a1",
            org_id="org1",
            timeout_s=5.0,
        )

    def test_check_hitl_sync_fails_open_on_connection_refused(self):
        from aegivis.security.exec_policy import check_hitl_sync

        # Use a URL that will be refused immediately
        check_hitl_sync(
            tool_name="test_tool",
            signals=[make_signal(Signal.CREDENTIAL)],
            backend_url="http://127.0.0.1:19999",  # nothing listening here
            api_key="key",
            session_id="s1",
            agent_id="a1",
            org_id="org1",
            timeout_s=2.0,
        )
        # Fail-open: should return, not raise ToolExecutionBlocked

    def test_create_approval_returns_none_on_connection_error(self):
        from aegivis.security.exec_policy import _create_approval

        result = _create_approval(
            tool_name="t",
            signals=[],
            backend_url="http://127.0.0.1:19999",
            api_key="k",
            session_id="s",
            agent_id="a",
            org_id="o",
        )
        assert result is None  # fail-open

    def test_poll_approval_sync_raises_blocked_on_denied(self, monkeypatch):
        from aegivis.security.exec_policy import _poll_approval_sync
        import time

        # Simulate a backend that immediately returns "denied"
        class FakeResponse:
            status_code = 200
            def json(self): return {"status": "denied"}

        class FakeClient:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, headers): return FakeResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: FakeClient())

        with pytest.raises(ToolExecutionBlocked) as exc_info:
            _poll_approval_sync(
                approval_id="test-id",
                tool_name="my_tool",
                signals=[make_signal(Signal.CREDENTIAL)],
                backend_url="http://fake",
                api_key="key",
                deadline=time.monotonic() + 10.0,
                poll_interval_s=0.0,
            )

        assert exc_info.value.tool_name == "my_tool"
        assert "denied" in exc_info.value.reason

    def test_poll_approval_sync_passes_on_approved(self, monkeypatch):
        from aegivis.security.exec_policy import _poll_approval_sync
        import time

        class FakeResponse:
            status_code = 200
            def json(self): return {"status": "approved"}

        class FakeClient:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, headers): return FakeResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: FakeClient())

        # Should return without raising
        _poll_approval_sync(
            approval_id="test-id",
            tool_name="my_tool",
            signals=[],
            backend_url="http://fake",
            api_key="key",
            deadline=time.monotonic() + 10.0,
            poll_interval_s=0.0,
        )

    def test_poll_approval_sync_raises_on_timeout(self, monkeypatch):
        from aegivis.security.exec_policy import _poll_approval_sync
        import time

        class FakeResponse:
            status_code = 200
            def json(self): return {"status": "pending"}

        class FakeClient:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, headers): return FakeResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: FakeClient())

        # Deadline already passed
        with pytest.raises(ToolExecutionBlocked) as exc_info:
            _poll_approval_sync(
                approval_id="test-id",
                tool_name="my_tool",
                signals=[],
                backend_url="http://fake",
                api_key="key",
                deadline=time.monotonic() - 1.0,  # already expired
                poll_interval_s=0.0,
            )

        assert "timeout" in exc_info.value.reason.lower()
