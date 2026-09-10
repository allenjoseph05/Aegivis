"""
Authorization Chain Verification — Phase 18.

Tracks which "intent classes" appear in user messages and system prompts.
At TOOL_CALL_START, checks whether the tool's intent class was ever authorized
by a human message or is covered by the system prompt.

Core concept:
  High-risk tool calls (network_send, auth_modify, code_exec, credential_access)
  should not be executed if NO prior message in the conversation expressed intent
  for that class of operation.  A ``send_email`` tool call should only happen if
  a user (or the system prompt) said something like "send", "email", "notify", etc.
  A ``get_secret`` call that appears from nowhere — without any user request
  mentioning credentials — is suspicious.

Authorization logic:
  - LOW-RISK intents  (data_read, recon, env_read, …): always authorized.
  - HIGH-RISK intents (network_send, auth_modify, code_exec, credential_access):
    require EXPLICIT mention of a related word in a user message OR system prompt.
  - UNKNOWN intent: always authorized (no false positives for unrecognised tools).

Note on auth_source:
  - "user_message"   — a user turn mentioned a token matching this intent class
  - "system_prompt"  — the system prompt mentioned a token (broader implicit grant)
  - "implicit"       — low-risk: no explicit mention needed
  - "none"           — high-risk with no authorization found

Integration in intercept.py:
  1. LLM_CALL_START hook: state.get_auth_chain().add_user_message(user_content)
  2. System prompt hook (Hook 1 block): state.get_auth_chain().add_system_prompt(sys_prompt)
  3. TOOL_CALL_START (after rollback gate):
       result = state.get_auth_chain().check_tool_call(tc_name)
       if not result.is_authorized → fire "unauthorized-tool-call" ALERT violation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .compound_detector import (
    classify_tool_intent,
    _INTENT_CLASSES,
    _CREDENTIAL_ACCESS,
    _ENV_READ,
    _DATA_READ,
    _RECON,
    _MEMORY_ACCESS,
    _ENCODE,
    _FILE_WRITE,
    _CODE_EXEC,
    _PROCESS_SPAWN,
    _NETWORK_SEND,
    _AUTH_MODIFY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent → token set mapping (for text scanning)
# ---------------------------------------------------------------------------

#: Combined mapping of intent class name → frozenset of matching tokens.
#: Built from the same frozensets used by compound_detector — single source of truth.
_INTENT_TOKEN_MAP: dict[str, frozenset[str]] = {
    name: token_set for name, token_set in _INTENT_CLASSES
}

#: High-risk intents that require explicit authorization in the conversation.
_HIGH_RISK_INTENTS: frozenset[str] = frozenset({
    "network_send",
    "auth_modify",
    "code_exec",
    "credential_access",
})


# ---------------------------------------------------------------------------
# Public API: extract_intents_from_text
# ---------------------------------------------------------------------------

def extract_intents_from_text(text: str) -> set[str]:
    """
    Tokenise ``text`` and return the set of intent class names that have
    at least one matching token in the text.

    Structural: lower-cases the text, splits on whitespace (and other
    non-alphanumeric boundary characters via multi-word scanning), then
    checks each word against all intent class frozensets.

    Returns an empty set for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return set()

    # Split on whitespace to get words; also split each word on common
    # punctuation boundaries so that "send-email" tokenises to ["send", "email"].
    raw_words = text.lower().split()
    words: set[str] = set()
    for raw in raw_words:
        # Strip common punctuation from word edges and split on non-alpha chars
        parts = _split_on_non_alnum(raw)
        words.update(parts)

    found: set[str] = set()
    for intent_name, token_set in _INTENT_CLASSES:
        if words & token_set:
            found.add(intent_name)
    return found


def _split_on_non_alnum(word: str) -> list[str]:
    """
    Split a word on any non-alphanumeric character and return the non-empty parts.

    E.g. "send-email" → ["send", "email"]
         "api_key"    → ["api", "key"]
         "https://..."→ ["https", "..."] (URL tokens)
    """
    current: list[str] = []
    parts: list[str] = []
    for ch in word:
        if ch.isalnum() or ch == "_":
            current.append(ch)
        else:
            if current:
                parts.append("".join(current))
                current = []
    if current:
        parts.append("".join(current))
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# AuthCheckResult
# ---------------------------------------------------------------------------

@dataclass
class AuthCheckResult:
    """
    Result of checking whether a tool call is authorized by the conversation history.

    Attributes:
        tool_name:     Name of the tool being checked.
        intent_class:  Intent class assigned to the tool (from compound_detector).
        is_authorized: True if the tool call is authorized.
        is_high_risk:  True if this intent class requires explicit authorization.
        auth_source:   Where authorization was found:
                         - "user_message"  — matched a token in a user turn
                         - "system_prompt" — matched a token in the system prompt
                         - "implicit"      — low-risk; implicitly authorized
                         - "none"          — high-risk with no authorization found
        reason:        Human-readable explanation.
    """

    tool_name: str
    intent_class: str
    is_authorized: bool
    is_high_risk: bool
    auth_source: str
    reason: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict suitable for JSONB storage."""
        return {
            "tool_name":     self.tool_name,
            "intent_class":  self.intent_class,
            "is_authorized": self.is_authorized,
            "is_high_risk":  self.is_high_risk,
            "auth_source":   self.auth_source,
            "reason":        self.reason,
        }


# ---------------------------------------------------------------------------
# AuthorizationChain
# ---------------------------------------------------------------------------

class AuthorizationChain:
    """
    Tracks intent classes mentioned in user messages and the system prompt.

    Usage (from intercept.py)::

        chain = state.get_auth_chain()

        # Called once per user message received
        chain.add_user_message(user_content)

        # Called once when system prompt is seen (Hook 1)
        chain.add_system_prompt(system_prompt)

        # Called at TOOL_CALL_START
        result = chain.check_tool_call(tool_name)
        if not result.is_authorized:
            # fire unauthorized-tool-call ALERT
    """

    def __init__(self) -> None:
        self._user_intents: set[str] = set()     # from user messages
        self._system_intents: set[str] = set()   # from system prompt (broader trust)

    def add_user_message(self, text: str) -> None:
        """
        Extract intent classes from a user message and add to user_intents.

        Called once for each user turn in the conversation.
        """
        if not text:
            return
        new_intents = extract_intents_from_text(text)
        if new_intents:
            self._user_intents.update(new_intents)
            logger.debug(
                "[AUTH-CHAIN] user_message → intents: %s",
                sorted(new_intents),
            )

    def add_system_prompt(self, text: str) -> None:
        """
        Extract intent classes from the system prompt and add to system_intents.

        The system prompt is set by the operator and thus has broader implicit
        authority — any intent class mentioned in the system prompt is treated
        as implicitly authorized for the session.
        """
        if not text:
            return
        new_intents = extract_intents_from_text(text)
        if new_intents:
            self._system_intents.update(new_intents)
            logger.debug(
                "[AUTH-CHAIN] system_prompt → intents: %s",
                sorted(new_intents),
            )

    @property
    def authorized_intents(self) -> set[str]:
        """Union of user-message and system-prompt intents."""
        return self._user_intents | self._system_intents

    def check_tool_call(self, tool_name: str) -> AuthCheckResult:
        """
        Check whether a tool call is authorized by the conversation history.

        High-risk tools (network_send, auth_modify, code_exec, credential_access)
        require that at least one user message OR the system prompt mentioned a
        token matching the tool's intent class.

        Low-risk tools are always authorized (implicit grant).
        Unknown-intent tools are always authorized (no false positives).

        Returns an AuthCheckResult with full authorization context.
        """
        intent = classify_tool_intent(tool_name)
        is_high_risk = intent in _HIGH_RISK_INTENTS

        # Determine authorization
        is_authorized: bool
        auth_source: str
        reason: str

        if intent == "unknown":
            is_authorized = True
            auth_source = "implicit"
            reason = (
                f"Tool '{tool_name}' has unknown intent class — "
                "no false positive: implicitly authorized."
            )
        elif not is_high_risk:
            is_authorized = True
            auth_source = "implicit"
            reason = (
                f"Tool '{tool_name}' intent class '{intent}' is low-risk — "
                "implicitly authorized without explicit user mention."
            )
        elif intent in self._user_intents:
            is_authorized = True
            auth_source = "user_message"
            reason = (
                f"Tool '{tool_name}' (intent: '{intent}') is authorized — "
                "a user message contained an explicit authorization signal."
            )
        elif intent in self._system_intents:
            is_authorized = True
            auth_source = "system_prompt"
            reason = (
                f"Tool '{tool_name}' (intent: '{intent}') is authorized — "
                "the system prompt contains an implicit authorization signal."
            )
        else:
            is_authorized = False
            auth_source = "none"
            reason = (
                f"Tool '{tool_name}' (intent: '{intent}') is HIGH-RISK and was NOT "
                "authorized by any user message or system prompt in this session. "
                "No conversation context requested this class of operation."
            )

        return AuthCheckResult(
            tool_name=tool_name,
            intent_class=intent,
            is_authorized=is_authorized,
            is_high_risk=is_high_risk,
            auth_source=auth_source,
            reason=reason,
        )
