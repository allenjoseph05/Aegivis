"""
Shared fragment extraction utility — used by session_pdg.py (Phase 10) and
ifc_labels.py (Phase 11).

A "fragment" is any sub-string of agent context content that is likely to be
a meaningful data identifier: URL, email address, domain name, quoted string,
or a long opaque token (hash, API key, UUID, base64 string).

Both the PDG and IFC systems use these fragments to detect when data from an
untrusted / external source flows verbatim into a privileged sink argument of
a subsequent tool call.
"""
from __future__ import annotations

import re


# Minimum lengths — keep in sync with tests/test_fragments.py
_MIN_FRAGMENT_LEN: int = 8    # base filter (URLs, domains, general tokens)
_MIN_QUOTED_LEN:   int = 15   # minimum chars inside quoted strings
_MIN_TOKEN_LEN:    int = 25   # minimum chars for bare long-token detection


def extract_fragments(content: str) -> list[str]:
    """
    Extract meaningful data fragments from *content*.

    Returns a deduplicated list of strings that are likely data identifiers:
    URLs, email addresses, domain names, quoted strings ≥ _MIN_QUOTED_LEN,
    and long continuous tokens ≥ _MIN_TOKEN_LEN.

    Empty string or content with only short tokens returns [].
    """
    if not content:
        return []

    fragments: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if len(s) >= _MIN_FRAGMENT_LEN and s not in seen:
            seen.add(s)
            fragments.append(s)

    # URLs
    for m in re.finditer(r'https?://[^\s"\'<>]{8,}', content):
        _add(m.group(0))

    # Email addresses — always significant identifiers; skip the length check
    for m in re.finditer(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', content):
        val = m.group(0).strip()
        if val not in seen:
            seen.add(val)
            fragments.append(val)

    # Domain names (bare, at least 2 parts, no spaces)
    for m in re.finditer(r'\b(?:[a-zA-Z0-9\-]+\.){1,}[a-zA-Z]{2,}\b', content):
        val = m.group(0).strip()
        # Skip very short or trivial tokens (e.g. "e.g", "i.e")
        if len(val) >= _MIN_FRAGMENT_LEN and '.' in val and val not in seen:
            seen.add(val)
            fragments.append(val)

    # Double-quoted strings (≥ _MIN_QUOTED_LEN chars inside quotes)
    for m in re.finditer(r'"([^"\n]{%d,})"' % _MIN_QUOTED_LEN, content):
        _add(m.group(1))

    # Single-quoted strings (≥ _MIN_QUOTED_LEN chars)
    for m in re.finditer(r"'([^'\n]{%d,})'" % _MIN_QUOTED_LEN, content):
        _add(m.group(1))

    # Long continuous tokens (no whitespace, ≥ _MIN_TOKEN_LEN chars)
    # Catches tokens, hashes, base64 strings, API keys, UUIDs
    for m in re.finditer(r'[^\s]{%d,}' % _MIN_TOKEN_LEN, content):
        _add(m.group(0))

    return fragments
