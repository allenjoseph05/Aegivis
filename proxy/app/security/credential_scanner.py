"""
Credential and secret leak scanner.

Detection uses three independent, complementary signals:

  1. Shannon entropy  — secrets are high-entropy random-looking strings.
                        English prose averages ~3.5 bits/char; secrets
                        typically range 4.5-6.0 bits/char.

  2. Context proximity — words near a high-entropy token that semantically
                         indicate credential usage (key, token, bearer, ...).
                         Even one such word significantly raises confidence.

  3. Structural format — length range and known prefix patterns that are
                         characteristic of structured secrets (JWT, GitHub
                         tokens, etc.).  We check STRUCTURE, not content —
                         we never store or match actual secret values.

Confidence is derived from all three signals together.  A token must pass
the entropy threshold before the other two are even evaluated, keeping the
false-positive rate low for ordinary high-entropy text (e.g. base64 images).

No regex patterns encoding specific service formats are used.  New secret
formats are caught automatically as long as they are high-entropy and appear
in a credential context.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants  (all overridable via settings in future iterations)
# ---------------------------------------------------------------------------

# Minimum / maximum token character length to consider.
_MIN_LEN: int = 16
_MAX_LEN: int = 512

# Minimum Shannon entropy (bits/char) to flag a token as a candidate.
# English text: ~3.0-3.8.  Secrets: typically 4.2-6.0.
_ENTROPY_THRESHOLD: float = 4.2

# Maximum plausible entropy for normalisation (theoretical max for 95-char printable ASCII = 6.57).
_ENTROPY_MAX: float = 6.0

# Number of words on each side of a token to search for context keywords.
_CONTEXT_WINDOW: int = 8


# ---------------------------------------------------------------------------
# Context keyword set
# This is a semantically stable, small cluster of words that unambiguously
# signal credential usage regardless of programming language or service.
# ---------------------------------------------------------------------------

_CONTEXT_KEYWORDS: frozenset[str] = frozenset({
    "key", "token", "secret", "password", "passwd", "credential", "credentials",
    "auth", "authorization", "authentication", "apikey", "api_key", "access_token",
    "bearer", "private", "signing", "signature", "cert", "certificate",
    "passphrase", "passkey", "masterkey", "master_key", "client_secret",
    "client_id", "refresh_token", "id_token", "session_token",
})

# ---------------------------------------------------------------------------
# Structural prefix patterns
# These identify the TYPE of a secret (its format), not its value.
# We do NOT match or store actual secret material.
# ---------------------------------------------------------------------------

_KNOWN_PREFIXES: tuple[str, ...] = (
    "sk-",         # OpenAI-style API keys
    "ghp_",        # GitHub personal access tokens
    "ghs_",        # GitHub server tokens
    "gho_",        # GitHub OAuth tokens
    "xoxb-",       # Slack bot tokens
    "xoxp-",       # Slack user tokens
    "xoxa-",       # Slack app tokens
    "eyJ",         # JWT header (base64url of '{"')
    "AKIA",        # AWS access key ID
    "AIza",        # Google API key
    "ya29.",       # Google OAuth access token
    "SG.",         # SendGrid API key
    "AC",          # Twilio Account SID (followed by 32 hex chars)
    "npm_",        # npm access tokens
    "pat_",        # Generic PAT prefix
    "glpat-",      # GitLab PAT
    "dp.",         # DigitalOcean token
)

# Pattern for things that look like UUIDs / content hashes — typically not secrets.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HASH_RE = re.compile(r"^[0-9a-f]{32,64}$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CredentialMatch:
    value_preview:  str    # first 8 chars + '...'  (never expose the full secret)
    entropy:        float  # bits/char
    context_score:  float  # 0.0-1.0
    format_score:   float  # 0.0-1.0
    confidence:     float  # combined 0.0-1.0
    token_length:   int


@dataclass
class CredentialScanResult:
    detected:         bool
    matches:          list[CredentialMatch] = field(default_factory=list)
    scan_text_length: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of *s* in bits per character."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _context_words(text: str, token_start: int, token_end: int) -> list[str]:
    """Return lowercase stripped words within ±_CONTEXT_WINDOW of the token."""
    before = text[:token_start].split()[-_CONTEXT_WINDOW:]
    after  = text[token_end:].split()[:_CONTEXT_WINDOW]
    clean  = re.compile(r"[^a-z0-9_]")
    return [
        w for raw in before + after
        if (w := clean.sub("", raw.lower()))
    ]


def _context_score(words: list[str]) -> float:
    """Return [0.0, 1.0] score from credential-context keyword proximity."""
    hits = sum(1 for w in words if w in _CONTEXT_KEYWORDS)
    return min(1.0, hits / 2.0)   # 2+ context keywords => 1.0


def _format_score(token: str) -> float:
    """
    Return [0.0, 1.0] structural plausibility as a formatted secret.

    Checks:
      - Known prefix patterns (strong signal)
      - Token length in expected secret range (medium)
      - Character diversity (digits + mixed case, common in secrets)
    """
    score = 0.0

    for prefix in _KNOWN_PREFIXES:
        if token.startswith(prefix):
            score = max(score, 0.90)
            break

    tlen = len(token)
    if 20 <= tlen <= 200:
        score = max(score, 0.50)
    elif 16 <= tlen < 20:
        score = max(score, 0.30)

    # Character diversity heuristic
    diversity = sum([
        any(c.isupper() for c in token),
        any(c.islower() for c in token),
        any(c.isdigit() for c in token),
    ])
    if diversity == 3:
        score = max(score, 0.45)
    elif diversity == 2:
        score = max(score, 0.30)

    return score


def _is_likely_non_secret(token: str) -> bool:
    """Return True for tokens that look like content hashes or UUIDs, not secrets."""
    return bool(_UUID_RE.match(token) or _HASH_RE.match(token))


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan(text: str) -> CredentialScanResult:
    """
    Scan *text* for potential credential / secret leaks.

    Returns a CredentialScanResult.  ``detected`` is True only when at least
    one token has a confidence value above the configured threshold across
    all three signals.
    """
    from ..config import settings

    threshold = settings.security_credential_confidence_threshold

    # Tokenise on whitespace boundaries
    tokens = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    matches: list[CredentialMatch] = []

    for raw_token, start, end in tokens:
        # Strip common wrapping punctuation
        core = raw_token.strip("\"'`()[]{},:;=")

        if not (_MIN_LEN <= len(core) <= _MAX_LEN):
            continue

        # Skip obvious non-secrets early
        if core.startswith(("http://", "https://", "ftp://", "/", "./")):
            continue
        if _is_likely_non_secret(core):
            continue

        entropy = _shannon_entropy(core)
        if entropy < _ENTROPY_THRESHOLD:
            continue   # fast path: not high-entropy => not a secret

        ctx_words  = _context_words(text, start, end)
        ctx_score  = _context_score(ctx_words)
        fmt_score  = _format_score(core)

        # Normalise entropy to [0, 1] above the threshold
        entropy_norm = min(1.0, (entropy - _ENTROPY_THRESHOLD) / (_ENTROPY_MAX - _ENTROPY_THRESHOLD))

        # Confidence: product of three signals, each clamped to a floor so
        # that even a token with no context keywords can still be flagged if
        # entropy + format are very strong.
        confidence = (
            entropy_norm
            * max(0.25, ctx_score)   # floor: high-entropy token in any context
            * max(0.20, fmt_score)   # floor: unknown-format secrets still caught
        )
        # Scale into [0, 1] — the product of three [0,1] values is small;
        # a token that is high-entropy + has context + known format should
        # reach ~0.9-1.0.
        confidence = min(1.0, confidence * 4.0)

        if confidence >= threshold:
            matches.append(CredentialMatch(
                value_preview=core[:8] + "...",
                entropy=round(entropy, 3),
                context_score=round(ctx_score, 3),
                format_score=round(fmt_score, 3),
                confidence=round(confidence, 3),
                token_length=len(core),
            ))
            logger.warning(
                "Potential credential detected: preview=%r entropy=%.2f confidence=%.2f",
                core[:8] + "...", entropy, confidence,
            )

    return CredentialScanResult(
        detected=len(matches) > 0,
        matches=matches,
        scan_text_length=len(text),
    )
