"""
enforcement/canonicalizer.py — Input encoding normalization.

Thin re-export of security.normalizer with a stable enforcement/ API name.
The normalizer is pure stdlib and is the first step in the sync scan path:

    canonicalize(text) → NormalizationResult
        .normalized_text   — text ready for structural scanning
        .encoding_score    — [0.0, 1.0] obfuscation confidence
        .encoding_detected — list of detected techniques

Detected techniques:
    base64, hex, hex_spaced, rot13, reversed, url_encoded,
    fullwidth_unicode, combining_diacritics, high_cpt_density

All detections are soft signals that add to the injection score.
Nothing is blocked here — that is the policy engine's job.
"""
from __future__ import annotations

# Re-export everything from the existing normalizer so callers use
# the enforcement/ namespace while we migrate the file in a later phase.
from ..security.normalizer import (   # noqa: F401
    NormalizationResult,
    normalize as canonicalize,
    normalize,                        # keep original name for compatibility
)

__all__ = ["NormalizationResult", "canonicalize", "normalize"]
