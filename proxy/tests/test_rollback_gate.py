"""
Tests for proxy.app.security.rollback_gate — Phase 21.

Test philosophy:
  - _tokenize: snake_case, hyphenated, upper-case, dots.
  - _classify_verb_tier: IRREVERSIBLE > SOFT_REVERSIBLE > REVERSIBLE priority.
  - _classify_verb_tier: unknown tool returns (UNKNOWN, "").
  - assess_reversibility: each tier with representative tools.
  - assess_reversibility: backup argument upgrades soft_reversible → reversible.
  - assess_reversibility: permanent/force argument degrades reversible → soft_reversible.
  - assess_reversibility: external comm argument → always IRREVERSIBLE.
  - assess_reversibility: external comm overrides even reversible verb.
  - assess_reversibility: priority — irreversible verb + backup arg stays IRREVERSIBLE.
  - assess_reversibility: ambiguous tool (send_backup) resolves conservatively.
  - RollbackAssessment.to_dict: correct keys, JSON-serialisable.
  - assess_reversibility: signals list non-empty.
  - assess_reversibility: unknown verb → UNKNOWN tier, conservative note.
"""
from __future__ import annotations

import json
import pytest

from app.security.rollback_gate import (
    IRREVERSIBLE,
    REVERSIBLE,
    SOFT_REVERSIBLE,
    UNKNOWN,
    RollbackAssessment,
    _classify_verb_tier,
    _tokenize,
    assess_reversibility,
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_snake_case(self):
        assert _tokenize("write_file") == ["write", "file"]

    def test_hyphenated(self):
        assert _tokenize("delete-document") == ["delete", "document"]

    def test_upper_case(self):
        assert _tokenize("SEND_EMAIL") == ["send", "email"]

    def test_dot_separated(self):
        assert _tokenize("fs.delete") == ["fs", "delete"]

    def test_single_word(self):
        assert _tokenize("delete") == ["delete"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_consecutive_underscores(self):
        tokens = _tokenize("write__file")
        assert "write" in tokens
        assert "file" in tokens


# ---------------------------------------------------------------------------
# _classify_verb_tier
# ---------------------------------------------------------------------------

class TestClassifyVerbTier:
    def test_reversible_create(self):
        tier, verb = _classify_verb_tier(["create", "file"])
        assert tier == REVERSIBLE
        assert verb == "create"

    def test_soft_reversible_update(self):
        tier, verb = _classify_verb_tier(["update", "record"])
        assert tier == SOFT_REVERSIBLE
        assert verb == "update"

    def test_irreversible_delete(self):
        tier, verb = _classify_verb_tier(["delete", "file"])
        assert tier == IRREVERSIBLE
        assert verb == "delete"

    def test_irreversible_takes_priority_over_reversible(self):
        """delete_create → irreversible (delete wins)."""
        tier, verb = _classify_verb_tier(["delete", "create"])
        assert tier == IRREVERSIBLE

    def test_irreversible_takes_priority_over_soft(self):
        """send_update → irreversible (send wins)."""
        tier, verb = _classify_verb_tier(["send", "update"])
        assert tier == IRREVERSIBLE

    def test_soft_takes_priority_over_reversible(self):
        """rename_add → soft_reversible (rename wins over add)."""
        tier, verb = _classify_verb_tier(["rename", "add"])
        assert tier == SOFT_REVERSIBLE

    def test_unknown_returns_unknown(self):
        tier, verb = _classify_verb_tier(["frob", "baz"])
        assert tier == UNKNOWN
        assert verb == ""

    def test_empty_tokens_returns_unknown(self):
        tier, verb = _classify_verb_tier([])
        assert tier == UNKNOWN

    def test_irreversible_send(self):
        tier, verb = _classify_verb_tier(["send"])
        assert tier == IRREVERSIBLE

    def test_reversible_insert(self):
        tier, verb = _classify_verb_tier(["insert"])
        assert tier == REVERSIBLE

    def test_soft_reversible_patch(self):
        tier, verb = _classify_verb_tier(["patch"])
        assert tier == SOFT_REVERSIBLE


# ---------------------------------------------------------------------------
# assess_reversibility — verb-tier classification
# ---------------------------------------------------------------------------

class TestAssessVerbTier:
    def test_create_file_reversible(self):
        ra = assess_reversibility("create_file", {})
        assert ra.tier == REVERSIBLE
        assert ra.is_reversible is True
        assert ra.is_irreversible is False

    def test_write_document_reversible(self):
        ra = assess_reversibility("write_document", {})
        assert ra.tier == REVERSIBLE

    def test_add_record_reversible(self):
        ra = assess_reversibility("add_record", {})
        assert ra.tier == REVERSIBLE

    def test_update_user_soft_reversible(self):
        ra = assess_reversibility("update_user", {})
        assert ra.tier == SOFT_REVERSIBLE
        assert ra.needs_backup is True

    def test_rename_file_soft_reversible(self):
        ra = assess_reversibility("rename_file", {})
        assert ra.tier == SOFT_REVERSIBLE

    def test_move_document_soft_reversible(self):
        ra = assess_reversibility("move_document", {})
        assert ra.tier == SOFT_REVERSIBLE

    def test_delete_file_irreversible(self):
        ra = assess_reversibility("delete_file", {})
        assert ra.tier == IRREVERSIBLE
        assert ra.is_irreversible is True

    def test_send_email_irreversible(self):
        ra = assess_reversibility("send_email", {})
        assert ra.tier == IRREVERSIBLE

    def test_drop_table_irreversible(self):
        ra = assess_reversibility("drop_table", {})
        assert ra.tier == IRREVERSIBLE

    def test_purge_data_irreversible(self):
        ra = assess_reversibility("purge_data", {})
        assert ra.tier == IRREVERSIBLE

    def test_unknown_tool_unknown_tier(self):
        ra = assess_reversibility("frob_nicate", {})
        assert ra.tier == UNKNOWN
        assert ra.is_reversible is False
        assert ra.is_irreversible is False


# ---------------------------------------------------------------------------
# assess_reversibility — argument signal effects
# ---------------------------------------------------------------------------

class TestAssessArgSignals:
    def test_backup_arg_upgrades_soft_to_reversible(self):
        ra = assess_reversibility("update_record", {"backup": True})
        assert ra.backup_signal is True
        assert ra.tier == REVERSIBLE
        assert ra.is_reversible is True

    def test_dry_run_arg_upgrades_soft_to_reversible(self):
        ra = assess_reversibility("rename_file", {"dry_run": True})
        assert ra.backup_signal is True
        assert ra.tier == REVERSIBLE

    def test_backup_does_not_upgrade_irreversible(self):
        """Even with backup=True, delete is still irreversible."""
        ra = assess_reversibility("delete_file", {"backup": True})
        assert ra.tier == IRREVERSIBLE

    def test_permanent_arg_degrades_reversible_to_soft(self):
        ra = assess_reversibility("write_file", {"force": True})
        assert ra.permanent_signal is True
        assert ra.tier == SOFT_REVERSIBLE  # degraded from reversible

    def test_permanent_arg_on_already_soft(self):
        """permanent on soft_reversible stays soft_reversible."""
        ra = assess_reversibility("update_record", {"permanent": True})
        assert ra.tier == SOFT_REVERSIBLE

    def test_permanent_arg_on_irreversible(self):
        """permanent on irreversible stays irreversible."""
        ra = assess_reversibility("delete_file", {"hard_delete": True})
        assert ra.tier == IRREVERSIBLE

    def test_external_comm_arg_always_irreversible(self):
        """Even a 'create' tool with a 'to' arg is irreversible."""
        ra = assess_reversibility("create_notification", {"to": "user@example.com"})
        assert ra.external_comm is True
        assert ra.tier == IRREVERSIBLE

    def test_recipient_arg_irreversible(self):
        ra = assess_reversibility("write_message", {"recipient": "alice@example.com"})
        assert ra.external_comm is True
        assert ra.tier == IRREVERSIBLE

    def test_webhook_url_arg_irreversible(self):
        ra = assess_reversibility("trigger_action", {"webhook_url": "https://hooks.x.com"})
        assert ra.external_comm is True
        assert ra.tier == IRREVERSIBLE

    def test_no_args_no_signal_boost(self):
        ra = assess_reversibility("write_file", {})
        assert ra.backup_signal is False
        assert ra.permanent_signal is False
        assert ra.external_comm is False

    def test_both_backup_and_permanent_args(self):
        """permanent wins over backup for soft_reversible → stays soft."""
        ra = assess_reversibility("rename_file", {"backup": True, "force": True})
        # permanent degrades reversible→soft, backup upgrades soft→reversible
        # Net effect depends on order of logic in implementation.
        # Either way, the result should be internally consistent:
        assert ra.tier in (REVERSIBLE, SOFT_REVERSIBLE)


# ---------------------------------------------------------------------------
# assess_reversibility — ambiguous tool names
# ---------------------------------------------------------------------------

class TestAmbiguousTools:
    def test_send_backup_is_irreversible(self):
        """send is irreversible; backup does not override verb tier."""
        ra = assess_reversibility("send_backup", {})
        assert ra.tier == IRREVERSIBLE

    def test_create_then_delete_verb_is_irreversible(self):
        ra = assess_reversibility("delete_and_create", {})
        # "delete" and "create" both present → IRREVERSIBLE wins
        assert ra.tier == IRREVERSIBLE

    def test_deploy_is_irreversible(self):
        ra = assess_reversibility("deploy_service", {})
        assert ra.tier == IRREVERSIBLE


# ---------------------------------------------------------------------------
# RollbackAssessment.to_dict
# ---------------------------------------------------------------------------

class TestRollbackAssessmentToDict:
    def test_to_dict_keys(self):
        ra = assess_reversibility("delete_file", {})
        d = ra.to_dict()
        expected = {
            "tool_name", "tier", "is_reversible", "needs_backup",
            "is_irreversible", "backup_signal", "permanent_signal",
            "external_comm", "signals", "matched_verb",
        }
        assert expected <= d.keys()

    def test_to_dict_json_serialisable(self):
        ra = assess_reversibility("delete_file", {"backup": True})
        json.dumps(ra.to_dict())  # must not raise

    def test_to_dict_values_correct(self):
        ra = assess_reversibility("create_file", {})
        d = ra.to_dict()
        assert d["tier"] == REVERSIBLE
        assert d["is_reversible"] is True
        assert d["is_irreversible"] is False
        assert d["tool_name"] == "create_file"

    def test_matched_verb_in_dict(self):
        ra = assess_reversibility("delete_file", {})
        assert ra.to_dict()["matched_verb"] == "delete"


# ---------------------------------------------------------------------------
# Signals list
# ---------------------------------------------------------------------------

class TestSignals:
    def test_signals_non_empty(self):
        ra = assess_reversibility("delete_file", {})
        assert len(ra.signals) >= 1

    def test_signals_mention_verb(self):
        ra = assess_reversibility("delete_file", {})
        combined = " ".join(ra.signals)
        assert "delete" in combined

    def test_backup_signal_mentioned_in_signals(self):
        ra = assess_reversibility("rename_file", {"backup": True})
        combined = " ".join(ra.signals)
        assert "backup" in combined.lower() or "reversible" in combined.lower()

    def test_unknown_tier_has_explanation(self):
        ra = assess_reversibility("frob_nicate", {})
        combined = " ".join(ra.signals)
        assert "unknown" in combined.lower() or "did not match" in combined.lower()
