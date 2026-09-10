"""
Source Accuracy Detection Engine.

Detects when an LLM response contradicts or fabricates information that was
not present in the tool results it received.  Works by comparing the LLM's
response text against the actual raw tool outputs using NLI (Natural Language
Inference) — specifically MiniCheck-DeBERTa.

Detection layer — MiniCheck NLI (50–150ms on CPU, optional):
    Checks each factual sentence in the LLM response against the serialised
    tool result.  A score < threshold means the tool result does NOT support
    that sentence — i.e. the model fabricated or contradicted the tool output.

Note: Logprob-based scoring was removed. Average logprob is a weak signal —
the threshold is arbitrary, logprob magnitude varies by model/temperature, and
it produces FPs on legitimately uncertain but correct responses. MiniCheck NLI
is a solid grounded-fact check with a well-defined scoring interpretation.

Usage:
    detector = SourceAccuracyDetector()
    result = detector.check(
        tool_results=[{"tool_name": "get_weather", "content": '{"error": "not_found"}'}],
        llm_response="The weather in Paris is 18°C.",
        config=SourceAccuracyConfig(),
    )
    if result.findings:
        # fire violation
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Sentences shorter than this are not factual claims worth checking
_MIN_SENTENCE_LEN = 20

# Truncate tool result to this many chars before NLI (DeBERTa context limit)
_MAX_TOOL_RESULT_CHARS = 1500



# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SourceAccuracyFinding:
    """A single sentence detected as not grounded in tool output."""
    sentence: str          # The LLM sentence that is not supported
    tool_name: str         # Which tool result it contradicts
    tool_snippet: str      # First 200 chars of the tool result (for display)
    score: float           # 0.0 = completely fabricated, 1.0 = fully grounded
    method: str            # "minicheck"
    severity: str          # "LOW" | "MEDIUM" | "HIGH"

    def to_dict(self) -> dict:
        return {
            "sentence":     self.sentence[:300],
            "tool_name":    self.tool_name,
            "tool_snippet": self.tool_snippet[:200],
            "score":        round(self.score, 4),
            "method":       self.method,
            "severity":     self.severity,
        }


@dataclass
class SourceAccuracyResult:
    """Result of running source accuracy detection on one LLM response."""
    findings: list[SourceAccuracyFinding] = field(default_factory=list)
    checked: bool = False          # False = no tool context available, skipped
    minicheck_available: bool = False

    @property
    def detected(self) -> bool:
        return len(self.findings) > 0

    @property
    def severity(self) -> str:
        if not self.findings:
            return "NONE"
        if any(f.severity == "HIGH" for f in self.findings):
            return "HIGH"
        if any(f.severity == "MEDIUM" for f in self.findings):
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict:
        return {
            "checked":              self.checked,
            "detected":             self.detected,
            "finding_count":        len(self.findings),
            "severity":             self.severity,
            "minicheck_available":  self.minicheck_available,
            "findings":             [f.to_dict() for f in self.findings],
        }


@dataclass
class SourceAccuracyConfig:
    enabled: bool = True
    action: str = "alert"          # "alert" | "block"
    threshold: float = 0.35        # minicheck score below this = source inaccuracy
    use_minicheck: bool = True


# ── Sentence extraction ────────────────────────────────────────────────────────

def extract_factual_sentences(text: str) -> list[str]:
    """
    Split the LLM response into sentences and return all sufficiently long ones.
    MiniCheck handles non-factual sentences gracefully (returns high scores).
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LEN]


def _serialize_tool_result(content: str | dict | list | None) -> str:
    """Serialize a tool result to a plain string for NLI input."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:_MAX_TOOL_RESULT_CHARS]
    try:
        return json.dumps(content, default=str)[:_MAX_TOOL_RESULT_CHARS]
    except Exception:
        return str(content)[:_MAX_TOOL_RESULT_CHARS]


# ── MiniCheck NLI scorer ───────────────────────────────────────────────────────

class _MiniCheckScorer:
    """
    Singleton wrapper around MiniCheck-DeBERTa.
    Loaded once at first use, then cached for the lifetime of the process.
    Returns None gracefully if the minicheck package is not installed.
    """
    _instance: "_MiniCheckScorer | None" = None
    _failed: bool = False

    def __init__(self) -> None:
        from minicheck.minicheck import MiniCheck  # type: ignore[import]
        self._model = MiniCheck(model_name="deberta-v3-large", enable_prefix_caching=False)
        logger.info("MiniCheck-DeBERTa loaded for source accuracy detection")

    def score(self, doc: str, claims: list[str]) -> list[float]:
        """
        Score each claim against the document.
        Returns raw_prob per claim: 0.0 = not supported, 1.0 = fully supported.
        """
        if not claims or not doc.strip():
            return [1.0] * len(claims)
        try:
            _, raw_probs, _, _ = self._model.score(
                docs=[doc] * len(claims),
                claims=claims,
            )
            return list(raw_probs)
        except Exception as exc:
            logger.warning("MiniCheck scoring error: %s", exc)
            return [1.0] * len(claims)  # fail open — don't false-positive on errors

    @classmethod
    def get(cls) -> "_MiniCheckScorer | None":
        if cls._failed:
            return None
        if cls._instance is None:
            try:
                cls._instance = cls()
            except ImportError:
                cls._failed = True
                logger.info(
                    "minicheck not installed — source accuracy detection disabled. "
                    "Install with: pip install minicheck"
                )
            except Exception as exc:
                cls._failed = True
                logger.warning("MiniCheck load failed: %s — source accuracy detection disabled", exc)
        return cls._instance


# ── Main detector ──────────────────────────────────────────────────────────────

class SourceAccuracyDetector:
    """
    Orchestrates both detection layers and returns a SourceAccuracyResult.

    Checks whether each factual sentence in the LLM response is actually
    supported by the tool results the model received. A low MiniCheck score
    means the tool output does NOT support the claim — the model went beyond
    its source data.

    Usage — call once per LLM response, in process_response() after the
    LLM response is assembled:

        detector = SourceAccuracyDetector()
        result = detector.check(
            tool_results=state.last_tool_results,
            llm_response=response_text,
            config=config,
        )
    """

    def check(
        self,
        tool_results: list[dict],          # [{tool_name, content}, ...] from session state
        llm_response: str | None,
        config: SourceAccuracyConfig,
    ) -> SourceAccuracyResult:

        if not config.enabled:
            return SourceAccuracyResult(checked=False)

        if not llm_response or not tool_results:
            return SourceAccuracyResult(checked=False)

        scorer = _MiniCheckScorer.get() if config.use_minicheck else None
        minicheck_available = scorer is not None

        if scorer is None:
            # MiniCheck not installed — cannot perform source accuracy detection. Fail open.
            return SourceAccuracyResult(checked=False, minicheck_available=False)

        sentences = extract_factual_sentences(llm_response)
        if not sentences:
            return SourceAccuracyResult(checked=True, minicheck_available=minicheck_available)

        findings: list[SourceAccuracyFinding] = []

        for tool in tool_results:
            tool_name = tool.get("tool_name", "unknown")
            doc = _serialize_tool_result(tool.get("content"))
            if not doc.strip():
                continue

            scores = scorer.score(doc, sentences)

            for sentence, score in zip(sentences, scores):
                if score < config.threshold:
                    severity = "HIGH" if score < 0.15 else ("MEDIUM" if score < 0.30 else "LOW")
                    findings.append(SourceAccuracyFinding(
                        sentence=sentence,
                        tool_name=tool_name,
                        tool_snippet=doc[:200],
                        score=score,
                        method="minicheck",
                        severity=severity,
                    ))
                    logger.debug(
                        "[SOURCE_ACCURACY] sentence=%r score=%.3f tool=%s severity=%s",
                        sentence[:80], score, tool_name, severity,
                    )

        # Deduplicate: if same sentence flagged by multiple tools, keep lowest score
        seen: dict[str, SourceAccuracyFinding] = {}
        for f in findings:
            key = f.sentence[:100]
            if key not in seen or f.score < seen[key].score:
                seen[key] = f
        findings = list(seen.values())

        return SourceAccuracyResult(
            findings=findings,
            checked=True,
            minicheck_available=minicheck_available,
        )


# ── Backward-compatibility aliases ─────────────────────────────────────────────
# Kept so that existing imports (intercept.py, tests) continue to work during
# the rename transition.  Will be removed once all callers are updated.
HallucinationFinding  = SourceAccuracyFinding
HallucinationResult   = SourceAccuracyResult
HallucinationConfig   = SourceAccuracyConfig
HallucinationDetector = SourceAccuracyDetector
