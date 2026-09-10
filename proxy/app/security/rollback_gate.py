"""
Rollback Readiness Gate — Phase 21.

Before executing a tool call, assess whether it can be rolled back if something
goes wrong.  Reversibility is a key safety property: an irreversible tool call
that causes harm cannot be undone.

This gate provides a second dimension beyond the blast radius guard:
  - Blast radius: HOW MUCH damage could this cause?
  - Rollback gate: CAN the damage be undone?

Both HIGH blast radius AND irreversible → strongest case for HITL / BLOCK.

Architecture:
  ``assess_reversibility(tool_name, args)`` → RollbackAssessment

Reversibility tiers (structural — frozenset lookups, no regex):
  REVERSIBLE       — action can be undone without external state
                     (create → delete, write → overwrite/delete)
  SOFT_REVERSIBLE  — undoable with a prior backup or snapshot
                     (update → restore backup, rename → rename back)
  IRREVERSIBLE     — cannot be undone once executed
                     (delete/drop, send/email, deploy/publish)

Argument signals (structural — frozenset key-name checks):
  Pro-reversibility:   backup=True, dry_run=True, snapshot=True, rollback_plan=...
  Anti-reversibility:  force=True, permanent=True, overwrite=True, no_backup=True
  External comm:       to=..., recipient=..., webhook=..., notify=... → always IRREVERSIBLE

Integration in intercept.py (TOOL_CALL_START, after behavioral baseline):
    ra = assess_reversibility(tc_name, tc_args)
    tool_start["security"]["rollback"] = ra.to_dict()
    if not ra.is_reversible and blast_score >= HIGH_THRESHOLD:
        # Escalate to HITL — irreversible + high blast radius
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reversibility taxonomy — frozensets, O(1) lookup
# ---------------------------------------------------------------------------

#: Tier 1 — Reversible: action creates new state; inverse exists and is simple.
#: Creating something that was not there before can always be undone by deleting it.
_VERBS_REVERSIBLE: frozenset[str] = frozenset({
    "create", "add", "insert", "write", "append", "save", "store",
    "put", "upload", "touch", "make", "init", "initialise", "initialize",
    "register", "enqueue", "schedule", "set",
})

#: Tier 2 — Soft-reversible: requires a pre-existing backup or snapshot.
#: These modify existing state; original can be restored from backup if it exists.
_VERBS_SOFT_REVERSIBLE: frozenset[str] = frozenset({
    "update", "modify", "patch", "edit", "change", "rename",
    "move", "copy", "replace", "overwrite", "migrate", "rewrite",
    "reset", "rollback", "restore", "rotate", "swap", "transfer",
    "archive",
})

#: Tier 3 — Irreversible: destructive or external; cannot be undone.
_VERBS_IRREVERSIBLE: frozenset[str] = frozenset({
    # Destructive
    "delete", "remove", "rm", "unlink", "destroy", "terminate",
    "purge", "wipe", "drop", "truncate", "format", "erase", "clear",
    "flush", "kill", "revoke", "deactivate", "drain", "nuke",
    # External communication (sent, cannot be un-sent)
    "send", "email", "mail", "notify", "publish", "broadcast",
    "relay", "forward", "push", "emit", "dispatch", "webhook",
    "post", "submit", "upload",   # "upload" — also REVERSIBLE for local; external upload is not
    # Deployment (live changes in production)
    "deploy", "release", "promote", "launch",
    # Privilege escalation (hard to undo)
    "grant", "escalate", "elevate",
})

# Note: "upload" appears in both _VERBS_REVERSIBLE and _VERBS_IRREVERSIBLE.
# The tier classification resolves this by checking tiers in order (irreversible first).
# An upload to an external system is irreversible.

#: Argument KEYS that indicate a backup/safety mechanism is in place.
#: Presence of any of these with a truthy value boosts reversibility.
_BACKUP_ARG_KEYS: frozenset[str] = frozenset({
    "backup", "backup_first", "create_backup", "save_backup",
    "dry_run", "dryrun", "dry", "simulate", "simulation",
    "snapshot", "checkpoint",
    "rollback_plan", "rollback", "recovery_plan", "undo",
    "reversible", "safe_mode",
})

#: Argument KEYS that indicate the action is intentionally permanent.
_PERMANENT_ARG_KEYS: frozenset[str] = frozenset({
    "force", "permanent", "permanently", "no_backup", "skip_backup",
    "hard_delete", "hard", "no_recovery", "irrecoverable",
    "purge", "overwrite_existing",
})

#: Argument KEYS that indicate external communication (always irreversible).
_EXTERNAL_COMM_ARG_KEYS: frozenset[str] = frozenset({
    "to", "recipient", "recipients", "cc", "bcc", "email", "mail",
    "webhook", "webhook_url", "notify", "notification",
    "slack_channel", "slack", "teams", "sms",
})


# ---------------------------------------------------------------------------
# Reversibility tier constants
# ---------------------------------------------------------------------------

REVERSIBLE = "reversible"
SOFT_REVERSIBLE = "soft_reversible"
IRREVERSIBLE = "irreversible"
UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# RollbackAssessment
# ---------------------------------------------------------------------------

@dataclass
class RollbackAssessment:
    """
    Result of assessing rollback readiness for a tool call.

    Attributes:
        tool_name:          Tool being assessed.
        tier:               One of ``reversible``, ``soft_reversible``,
                            ``irreversible``, ``unknown``.
        is_reversible:      True for tier ``reversible``.
        needs_backup:       True for tier ``soft_reversible``.
        is_irreversible:    True for tier ``irreversible``.
        backup_signal:      True if a backup argument was found → boost.
        permanent_signal:   True if a permanent/force argument was found → downgrade.
        external_comm:      True if external communication args were detected.
        signals:            Human-readable explanation list.
        matched_verb:       The verb token that drove the tier classification.
    """
    tool_name: str
    tier: str = UNKNOWN
    is_reversible: bool = False
    needs_backup: bool = False
    is_irreversible: bool = False
    backup_signal: bool = False
    permanent_signal: bool = False
    external_comm: bool = False
    signals: list[str] = field(default_factory=list)
    matched_verb: str = ""

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tier": self.tier,
            "is_reversible": self.is_reversible,
            "needs_backup": self.needs_backup,
            "is_irreversible": self.is_irreversible,
            "backup_signal": self.backup_signal,
            "permanent_signal": self.permanent_signal,
            "external_comm": self.external_comm,
            "signals": self.signals,
            "matched_verb": self.matched_verb,
        }


# ---------------------------------------------------------------------------
# Core assessment function
# ---------------------------------------------------------------------------

def assess_reversibility(tool_name: str, args: dict[str, Any]) -> RollbackAssessment:
    """
    Assess whether a tool call can be rolled back.

    Detection is purely structural:
      1. Tokenise the tool name (split on ``_`` / ``-``).
      2. Match tokens against tier frozensets (irreversible → soft → reversible).
      3. Check argument keys for backup/permanent/external signals.
      4. Combine: backup signal boosts tier; permanent/external signal degrades.

    Args:
        tool_name: Name of the tool.
        args:      Tool argument dict.

    Returns:
        RollbackAssessment with tier and signal annotations.
    """
    result = RollbackAssessment(tool_name=tool_name)

    # ── Step 1: classify verb tier from tool name ──────────────────────────
    tokens = _tokenize(tool_name)
    tier, matched_verb = _classify_verb_tier(tokens)
    result.matched_verb = matched_verb

    # ── Step 2: check argument keys for signals ────────────────────────────
    arg_keys_lower = {k.lower() for k in args.keys() if isinstance(k, str)}

    backup_keys = arg_keys_lower & _BACKUP_ARG_KEYS
    if backup_keys:
        result.backup_signal = True
        result.signals.append(
            f"Backup/safety argument detected: {', '.join(sorted(backup_keys))}"
        )

    permanent_keys = arg_keys_lower & _PERMANENT_ARG_KEYS
    if permanent_keys:
        result.permanent_signal = True
        result.signals.append(
            f"Permanent/force argument detected: {', '.join(sorted(permanent_keys))} "
            "— rollback explicitly disabled"
        )

    ext_keys = arg_keys_lower & _EXTERNAL_COMM_ARG_KEYS
    if ext_keys:
        result.external_comm = True
        result.signals.append(
            f"External communication argument detected: {', '.join(sorted(ext_keys))} "
            "— sent data cannot be recalled"
        )

    # ── Step 3: resolve final tier with argument adjustments ──────────────
    final_tier = tier

    if result.external_comm:
        # External comms always irreversible regardless of verb tier
        final_tier = IRREVERSIBLE

    if result.permanent_signal and final_tier in (REVERSIBLE, SOFT_REVERSIBLE):
        # force/permanent overrides reversible assessment
        final_tier = SOFT_REVERSIBLE  # degrade by one tier at most

    if result.backup_signal and final_tier == SOFT_REVERSIBLE:
        # Backup argument makes soft-reversible effectively reversible
        final_tier = REVERSIBLE
        result.signals.append(
            "Backup mechanism present — soft-reversible upgraded to reversible"
        )

    # Permanent on already-irreversible → no change (stay irreversible)

    # ── Step 4: populate result flags ─────────────────────────────────────
    result.tier = final_tier
    result.is_reversible = final_tier == REVERSIBLE
    result.needs_backup = final_tier == SOFT_REVERSIBLE
    result.is_irreversible = final_tier == IRREVERSIBLE

    if tier == UNKNOWN:
        result.signals.append(
            f"Tool name '{tool_name}' did not match any known reversibility verb — "
            "treating as unknown (conservative: assume soft-reversible)"
        )
    else:
        result.signals.insert(
            0,
            f"Verb '{matched_verb}' classified as {tier} "
            + (f"→ adjusted to {final_tier}" if final_tier != tier else ""),
        )

    logger.debug(
        "[ROLLBACK] tool=%r verb=%r tier=%s→%s backup=%s permanent=%s ext=%s",
        tool_name, matched_verb, tier, final_tier,
        result.backup_signal, result.permanent_signal, result.external_comm,
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(tool_name: str) -> list[str]:
    """Split a tool name into lowercase tokens on ``_`` and ``-``."""
    normalized = tool_name.lower().replace("-", "_").replace(".", "_")
    return [t for t in normalized.split("_") if t]


def _classify_verb_tier(tokens: list[str]) -> tuple[str, str]:
    """
    Return (tier, matched_verb) for the first matching token.

    Priority: IRREVERSIBLE > SOFT_REVERSIBLE > REVERSIBLE.
    This ensures that a tool like ``send_backup`` (both send=irreversible and
    backup-like) is classified conservatively as IRREVERSIBLE.

    Returns (UNKNOWN, "") if no token matches any tier.
    """
    # Check all tiers for all tokens; pick highest-risk match
    found_irreversible = ""
    found_soft = ""
    found_reversible = ""

    for token in tokens:
        if token in _VERBS_IRREVERSIBLE and not found_irreversible:
            found_irreversible = token
        if token in _VERBS_SOFT_REVERSIBLE and not found_soft:
            found_soft = token
        if token in _VERBS_REVERSIBLE and not found_reversible:
            found_reversible = token

    if found_irreversible:
        return IRREVERSIBLE, found_irreversible
    if found_soft:
        return SOFT_REVERSIBLE, found_soft
    if found_reversible:
        return REVERSIBLE, found_reversible

    return UNKNOWN, ""
