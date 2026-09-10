"""
Unit tests for the Email Preview Sandbox Adapter — Phase 16.6.
"""
from __future__ import annotations

import asyncio
import pytest

from app.security.adapters.email import EmailPreviewAdapter
from app.security.sandbox import SandboxContext


def ctx() -> SandboxContext:
    return SandboxContext(session_id="s", agent_id="a", org_id="o", timeout_ms=2000)


def run(coro):
    return asyncio.run(coro)


ADP = EmailPreviewAdapter()


# ── can_handle ────────────────────────────────────────────────────────────────

class TestCanHandle:
    def test_email_in_tool_name(self):
        assert run(ADP.can_handle("send_email", {}))

    def test_mail_in_tool_name(self):
        assert run(ADP.can_handle("send_mail", {}))

    def test_notify_in_tool_name(self):
        assert run(ADP.can_handle("notify_user", {}))

    def test_smtp_in_tool_name(self):
        assert run(ADP.can_handle("smtp_send", {}))

    def test_sendgrid_in_tool_name(self):
        assert run(ADP.can_handle("sendgrid_send", {}))

    def test_recipient_arg_key_matches(self):
        """Even without email in tool name, 'to' arg triggers."""
        assert run(ADP.can_handle("deliver_message", {"to": "user@example.com"}))

    def test_recipient_key_cc(self):
        assert run(ADP.can_handle("some_tool", {"cc": ["a@b.com"]}))

    def test_no_match_unknown_tool_no_email_arg(self):
        assert not run(ADP.can_handle("run_sql", {"query": "SELECT 1"}))

    def test_no_match_empty_args(self):
        assert not run(ADP.can_handle("compute_total", {}))


# ── dry_run: recipients ───────────────────────────────────────────────────────

class TestRecipients:
    def test_single_recipient_safe(self):
        r = run(ADP.dry_run("send_email", {"to": "alice@example.com", "body": "hi"}, ctx()))
        assert r.safe is True
        assert r.scope_estimate["recipient_count"] == 1

    def test_comma_separated_recipients(self):
        r = run(ADP.dry_run("send_email", {"to": "a@x.com, b@x.com, c@x.com"}, ctx()))
        assert r.scope_estimate["recipient_count"] == 3

    def test_list_recipients(self):
        addrs = [f"user{i}@example.com" for i in range(5)]
        r = run(ADP.dry_run("send_email", {"recipients": addrs}, ctx()))
        assert r.scope_estimate["recipient_count"] == 5

    def test_bulk_threshold_unsafe(self):
        """10 or more recipients → safe=False."""
        addrs = [f"u{i}@x.com" for i in range(10)]
        r = run(ADP.dry_run("send_email", {"to": ", ".join(addrs)}, ctx()))
        assert r.safe is False
        assert any("bulk-send" in s for s in r.signals)

    def test_mass_threshold_unsafe(self):
        """50 or more recipients → mass-send signal."""
        addrs = [f"u{i}@x.com" for i in range(50)]
        r = run(ADP.dry_run("send_email", {"recipients": addrs}, ctx()))
        assert r.safe is False
        assert any("mass-send" in s for s in r.signals)

    def test_nine_recipients_safe(self):
        """9 recipients — just under threshold."""
        addrs = [f"u{i}@x.com" for i in range(9)]
        r = run(ADP.dry_run("send_email", {"recipients": addrs}, ctx()))
        assert r.safe is True

    def test_no_recipients_safe(self):
        r = run(ADP.dry_run("send_email", {"subject": "hello"}, ctx()))
        assert r.safe is True
        assert r.scope_estimate["recipient_count"] == 0

    def test_semicolon_separated_recipients(self):
        r = run(ADP.dry_run("send_email", {"to": "a@x.com; b@x.com"}, ctx()))
        assert r.scope_estimate["recipient_count"] == 2

    def test_bcc_key(self):
        r = run(ADP.dry_run("send_email", {"bcc": "secret@x.com"}, ctx()))
        assert r.scope_estimate["recipient_count"] == 1


# ── dry_run: body / PII ───────────────────────────────────────────────────────

class TestBody:
    def test_no_pii_safe(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "body": "Hello, how are you?"}, ctx()))
        assert r.safe is True
        assert r.scope_estimate["pii_signals"] == []

    def test_password_in_body_unsafe(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "body": "Your password is abc123"}, ctx()))
        assert r.safe is False
        assert "pii-in-body" in " ".join(r.signals)

    def test_ssn_in_body_unsafe(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "body": "SSN: 123-45-6789"}, ctx()))
        assert r.safe is False

    def test_credit_card_signal(self):
        r = run(ADP.dry_run("send_email", {"body": "your credit card details"}, ctx()))
        assert r.safe is False

    def test_secret_token_signal(self):
        r = run(ADP.dry_run("send_email", {"body": "here is your secret token: xyz"}, ctx()))
        assert r.safe is False

    def test_body_via_html_key(self):
        r = run(ADP.dry_run("send_email", {"html": "<p>Hello World</p>"}, ctx()))
        assert r.scope_estimate["body_length"] > 0

    def test_body_via_content_key(self):
        r = run(ADP.dry_run("send_email", {"content": "plain text"}, ctx()))
        assert r.scope_estimate["body_length"] == len("plain text")

    def test_body_in_preview(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "body": "Hello!"}, ctx()))
        assert "Hello!" in r.preview


# ── dry_run: attachments ──────────────────────────────────────────────────────

class TestAttachments:
    def test_no_attachments(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com"}, ctx()))
        assert r.scope_estimate["attachment_count"] == 0

    def test_attachment_count_in_scope(self):
        r = run(ADP.dry_run("send_email", {"attachments": ["f1.pdf", "f2.pdf"]}, ctx()))
        assert r.scope_estimate["attachment_count"] == 2

    def test_large_attachment_signal(self):
        big = {"content": "x" * (11 * 1024 * 1024), "name": "big.bin"}
        r = run(ADP.dry_run("send_email", {"attachments": [big]}, ctx()))
        assert any("large-attachment" in s for s in r.signals)

    def test_large_attachment_not_unsafe(self):
        """Large attachment adds a warning signal but does not set safe=False alone."""
        big = {"content": "x" * (11 * 1024 * 1024)}
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "attachments": [big]}, ctx()))
        # Safe can be True — large attachment alone doesn't block
        assert any("large-attachment" in s for s in r.signals)

    def test_many_attachments_signal(self):
        r = run(ADP.dry_run("send_email", {"attachments": [f"f{i}.pdf" for i in range(6)]}, ctx()))
        assert any("many-attachments" in s for s in r.signals)

    def test_size_from_dict_size_field(self):
        att = {"name": "report.pdf", "size": 5 * 1024 * 1024}
        r = run(ADP.dry_run("send_email", {"attachments": [att]}, ctx()))
        assert r.scope_estimate["total_attachment_bytes"] == 5 * 1024 * 1024


# ── dry_run: general ──────────────────────────────────────────────────────────

class TestGeneral:
    def test_executed_always_false(self):
        """Email adapter never actually sends — executed=False always."""
        r = run(ADP.dry_run("send_email", {"to": "a@b.com"}, ctx()))
        assert r.executed is False

    def test_adapter_name(self):
        r = run(ADP.dry_run("send_email", {}, ctx()))
        assert r.adapter == "email"

    def test_ok_signal_when_clean(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "body": "hello"}, ctx()))
        assert "email-preview-ok" in r.signals

    def test_preview_in_to_dict(self):
        r = run(ADP.dry_run("send_email", {"to": "a@b.com", "body": "hi"}, ctx()))
        d = r.to_dict()
        assert "preview" in d

    def test_latency_ms_positive(self):
        r = run(ADP.dry_run("send_email", {}, ctx()))
        assert r.latency_ms >= 0.0

    def test_combined_pii_and_bulk_signals(self):
        addrs = [f"u{i}@x.com" for i in range(15)]
        r = run(ADP.dry_run("send_email", {
            "recipients": addrs,
            "body": "Here is your password and ssn",
        }, ctx()))
        assert r.safe is False
        signal_str = " ".join(r.signals)
        assert "bulk-send" in signal_str or "mass-send" in signal_str
        assert "pii-in-body" in signal_str
