"""
proxy/app/enforcement — Sync-path security enforcement.

This package contains ALL security checks that run INLINE in the request path.
Every module here must:
  - Have no ML dependencies (no torch, no sentence-transformers, no sklearn)
  - Complete in <5ms per request on typical agent traffic
  - Never raise exceptions that block the request (all scanners catch errors)
  - Be independently testable without Docker or LLM APIs

Behavioral analysis (Markov, Isolation Forest) lives in proxy/app/security/
and runs in a background executor — it never blocks the request path.

Public API
----------
scan_messages(messages)      → MessageScanResult
    Runs canonicalizer + structural scan on all user/tool-result messages.
    Call this at LLM_CALL_START before forwarding to the LLM.

scan_tool_call(name, args)   → ToolScanResult
    Runs RCE + SSRF scan on tool call arguments.
    Call this at TOOL_CALL_START after the LLM emits a tool_call.

validate_tool_args(name, args, tools) → SchemaCheckResult
    Validates tool arguments against the declared JSON Schema.
    Call this at TOOL_CALL_START alongside scan_tool_call.

Encoding normalization (canonicalize) is applied inside scan_messages
and is also available directly:

canonicalize(text)            → NormalizationResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .canonicalizer import NormalizationResult, canonicalize
from .content_policy import ToolScanResult, scan_tool_call
from .schema_check import SchemaCheckResult, validate_tool_args
from .structural import MessageScanResult, StructuralScanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combined message scan: canonicalizer → structural
# ---------------------------------------------------------------------------

@dataclass
class EnforcementScanResult:
    """
    Full sync-path scan result for an LLM call's messages.

    The ``injection_score`` is the primary value for policy evaluation.
    It fuses (in order of application):
      1. structural_score — LLM delimiter / phrase injection signal
      2. encoding_score   — obfuscation signal from canonicalizer
      3. classifier_score — DeBERTa ML injection probability (sync mode only)
    All fused via additive residual space:
      score = a + b * (1 - a)

    ``credential_detected`` mirrors the field the policy engine expects so
    that credential-leak rules fire without any ML pipeline.
    """
    injection_score:    float                       # fused [0.0, 1.0]
    injection_label:    str                         # "safe" | "suspicious" | "malicious"
    structural:         MessageScanResult           # L1 structural result
    encoding_detected:  list[str] = field(default_factory=list)
    encoding_score:     float = 0.0                # normalizer output
    segments_scanned:   int = 0
    credential_detected: bool = False              # from credential_scanner
    credential_count:   int = 0
    classifier_score:   float = 0.0                # ML classifier (0.0 when not run)
    classifier_label:   str = ""                   # "safe"|"suspicious"|"malicious"|""

    def to_event_dict(self) -> dict:
        """Serialize to a dict for inclusion in the event payload."""
        d: dict = {
            "injection_score":    round(self.injection_score, 4),
            "injection_label":    self.injection_label,
            "credential_detected": self.credential_detected,
            "credential_count":   self.credential_count,
            "encoding_detected":  self.encoding_detected,
            "encoding_score":     round(self.encoding_score, 4),
            "structural_score":   round(self.structural.score, 4),
            "segments_scanned":   self.segments_scanned,
            # Per-segment breakdown capped at 5 for storage
            "segments": [
                {
                    "score": round(s.score, 4),
                    "label": s.label,
                    "delimiter_hits": s.delimiter_hits,
                    "matched_phrases": s.matched_phrases[:3],
                }
                for s in self.structural.per_segment[:5]
            ],
        }
        if self.classifier_score > 0.0:
            d["classifier_score"] = round(self.classifier_score, 4)
            d["classifier_label"] = self.classifier_label
        return d


def scan_messages(messages: list[dict]) -> EnforcementScanResult:
    """
    Run the full sync-path scan on LLM request messages.

    Steps:
      1. Canonicalize each segment (Base64, Hex, ROT13, invisible Unicode, etc.)
      2. Run structural L1 scan on canonicalized text
      3. Run credential scanner on ALL messages (incl. system — operators leak keys)
      4. Fuse encoding_score + structural_score

    This is the function intercept.py calls at LLM_CALL_START.
    Always runs — no feature flag required.
    Never raises — failures degrade gracefully (score stays at 0.0).
    """
    from .structural import _extract_segments, scan as structural_scan
    from .canonicalizer import canonicalize
    from ..security.credential_scanner import scan as cred_scan

    try:
        segments = _extract_segments(messages)
        if not segments:
            # Still run credential scan on all messages even if no user segments
            all_text = " ".join(
                str(m.get("content") or "") for m in messages if m.get("content")
            )
            cred = cred_scan(all_text)
            return EnforcementScanResult(
                injection_score=0.0,
                injection_label="safe",
                structural=MessageScanResult(score=0.0, label="safe"),
                credential_detected=cred.detected,
                credential_count=len(cred.matches),
            )

        max_encoding_score = 0.0
        all_encodings: list[str] = []
        canonical_segments: list[str] = []

        for seg in segments:
            norm = canonicalize(seg)
            if norm.encoding_score > max_encoding_score:
                max_encoding_score = norm.encoding_score
            all_encodings.extend(norm.encoding_detected)
            canonical_segments.append(norm.normalized_text)

        # Run structural scan on canonicalized segments
        structural_results = [structural_scan(seg) for seg in canonical_segments]
        worst_structural = max(structural_results, key=lambda r: r.score)

        # Credential scan over ALL message text (including system messages)
        all_text = " ".join(
            str(m.get("content") or "") for m in messages if m.get("content")
        )
        cred = cred_scan(all_text)

        from ..config import settings
        # Choose structural base score for fusion:
        #
        # When the sync ML classifier is enabled, use token_score (delimiter +
        # invisible unicode only — no phrase matching). The classifier handles
        # semantic detection, so phrase-based scoring is bypassed to prevent
        # it from dominating the fused score and causing false positives on
        # benign text that happens to contain injection-like phrases.
        #
        # When running structural-only (no sync classifier), use the full
        # combined score (token + phrases) as the primary injection signal.
        if settings.analysis_classifier_sync_enabled:
            s = worst_structural.token_score   # delimiter + unicode only
        else:
            s = worst_structural.score         # token + phrase (full structural)
        e = max_encoding_score
        fused = s + e * (1.0 - s)
        fused = float(min(1.0, fused))

        # ── Sync ML classifier (optional, adds ~50ms) ──────────────────────
        # When analysis_classifier_sync_enabled=True, run DeBERTa inline.
        # This is the primary high-accuracy detection layer (88-99% PINT).
        # Score fused in same residual space: fused = fused + clf * (1 - fused)
        # We scan the canonicalized text (encoding-normalized) so the classifier
        # also catches obfuscated attacks that the structural layer decoded.
        clf_score = 0.0
        clf_label = ""
        if settings.analysis_classifier_sync_enabled:
            try:
                from ..analysis.classifier import classify, is_available as clf_avail
                if clf_avail():
                    # Build text window(s) for classification.
                    # DeBERTa handles up to ~512 tokens ≈ 2048 chars.
                    # For long texts, scan BOTH start and tail — attacks are
                    # often appended after legitimate-looking preamble content.
                    joined = " ".join(canonical_segments)
                    if len(joined) <= 2048:
                        windows = [joined]
                    else:
                        windows = [joined[:2048], joined[-2048:]]

                    clf_score = 0.0
                    clf_label = ""
                    for window in windows:
                        _r = classify(
                            window,
                            model_name=settings.analysis_classifier_model,
                            threshold=settings.analysis_classifier_threshold,
                        )
                        if _r.score > clf_score:
                            clf_score = _r.score
                            clf_label = _r.label

                    # ── Ensemble: second model (e.g. PromptGuard) ──────────
                    # If a second classifier is configured, run it on the same
                    # windows and take max(primary, secondary).  Different
                    # architecture catches attacks the primary misses.
                    if settings.analysis_classifier_second_enabled:
                        try:
                            from ..analysis.classifier import classify_second
                            for window in windows:
                                _r2 = classify_second(
                                    window,
                                    model_name=settings.analysis_classifier_second_model,
                                    threshold=settings.analysis_classifier_threshold,
                                )
                                if _r2.score > clf_score:
                                    clf_score = _r2.score
                                    clf_label = _r2.label
                                    logger.debug(
                                        "Ensemble second model fired: score=%.3f model=%s",
                                        _r2.score, _r2.model,
                                    )
                        except Exception as exc2:
                            logger.warning("Ensemble second classifier skipped: %s", exc2)

                    # ── Phrase corroboration ────────────────────────────────
                    # When the classifier already sees genuine injection signal
                    # (≥0.40), explicit phrase matches act as corroborating
                    # evidence and boost the score toward certainty.
                    # Below 0.40: phrases are ignored entirely to prevent FP
                    # on benign roleplay / creative-writing prompts.
                    # Formula: effective = max(clf, (clf + phrase) / 2)
                    # Example: clf=0.45, phrase=0.85 → effective=0.65
                    phrase_s = worst_structural.phrase_score
                    if clf_score >= 0.40 and phrase_s > 0.0:
                        effective_clf = max(clf_score, (clf_score + phrase_s) / 2.0)
                        clf_score = float(min(1.0, effective_clf))

                    fused = min(1.0, fused + clf_score * (1.0 - fused))
                    fused = float(fused)
                    logger.debug(
                        "Sync classifier: score=%.3f phrase_s=%.3f label=%s fused=%.3f",
                        clf_score, phrase_s, clf_label, fused,
                    )
            except Exception as exc:
                logger.warning("Sync classifier degraded (skipping): %s", exc)

        block_t = settings.security_injection_block_threshold
        alert_t = settings.security_injection_alert_threshold
        if fused >= block_t:
            label = "malicious"
        elif fused >= alert_t:
            label = "suspicious"
        else:
            label = "safe"

        return EnforcementScanResult(
            injection_score=fused,
            injection_label=label,
            structural=MessageScanResult(
                score=worst_structural.score,
                label=worst_structural.label,
                segments_scanned=len(segments),
                per_segment=structural_results,
            ),
            encoding_detected=list(dict.fromkeys(all_encodings)),  # dedup, preserve order
            encoding_score=round(max_encoding_score, 4),
            segments_scanned=len(segments),
            credential_detected=cred.detected,
            credential_count=len(cred.matches),
            classifier_score=round(clf_score, 4),
            classifier_label=clf_label,
        )

    except Exception as exc:
        logger.warning("enforcement.scan_messages error (degrading to safe): %s", exc)
        return EnforcementScanResult(
            injection_score=0.0,
            injection_label="safe",
            structural=MessageScanResult(score=0.0, label="safe"),
        )


__all__ = [
    # Functions
    "scan_messages",
    "scan_tool_call",
    "validate_tool_args",
    "canonicalize",
    # Result types
    "EnforcementScanResult",
    "ToolScanResult",
    "SchemaCheckResult",
    "NormalizationResult",
    "MessageScanResult",
    "StructuralScanResult",
]
