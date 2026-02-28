"""
proxy/app/analysis/classifier.py — Async ML injection classifier.

This is the ASYNC PATH scanner (Phase 4). It runs in a background thread
AFTER the current LLM call has been forwarded. Results are stored on the
SessionState and checked at the START of the NEXT call in the session.

Why async?
----------
The DeBERTa model takes ~50ms on CPU. Running it inline would add 50ms to
every LLM API call. Instead, we run it after forwarding so the current call
sees zero added latency. The ML scan protects the NEXT call:

  Turn N:   user message → enforcement (sync, <5ms) → forward → ML scan (async)
  Turn N+1: user message → check ml_injection_flag → BLOCK if True

This is sufficient for multi-turn attack chains, the most dangerous scenario
for autonomous agents (e.g. an injected document in turn 1 causing an exfil
tool call in turn 2). Single-turn direct attacks are handled by the sync
enforcement layer + canary + output_scanner.

Why this model?
---------------
protectai/deberta-v3-base-prompt-injection-v2
  - 88–99% accuracy on the PINT benchmark (direct + indirect injection)
  - ~280 MB, downloaded once to ~/.cache/huggingface/hub/
  - CPU inference: ~50ms/text (acceptable for async-only use)
  - Model is NOT gated — no HuggingFace account required

Without the [classifier] extra (transformers + torch), this module
gracefully returns ClassifierResult(score=0.0, label="safe", model="degraded")
and the proxy continues with structural-only scanning.

Install
-------
    pip install 'agentblackbox-proxy[classifier]'
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton: lazy-load the model exactly once (on first classify() call)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_pipe = None          # primary pipeline (DeBERTa)
_pipe_loaded = False
_pipe_model_name: str = ""

_pipe2_lock = threading.Lock()
_pipe2 = None         # secondary pipeline (PromptGuard / any second model)
_pipe2_loaded = False
_pipe2_model_name: str = ""


def _load_pipeline(model_name: str) -> None:
    """Load the primary HuggingFace text-classification pipeline."""
    global _pipe, _pipe_loaded, _pipe_model_name
    if _pipe_loaded:
        return
    _pipe_loaded = True
    _pipe_model_name = model_name
    try:
        from transformers import pipeline as _hf_pipeline  # type: ignore

        _pipe = _hf_pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=512,
            device=-1,  # CPU; set device=0 for CUDA
        )
        logger.info("Analysis ML classifier loaded: %s", model_name)
    except Exception as exc:
        logger.warning(
            "Analysis ML classifier failed to load (degrading to structural-only): %s", exc
        )
        _pipe = None


def _load_pipeline2(model_name: str) -> None:
    """Load the secondary HuggingFace text-classification pipeline (ensemble model)."""
    global _pipe2, _pipe2_loaded, _pipe2_model_name
    if _pipe2_loaded:
        return
    _pipe2_loaded = True
    _pipe2_model_name = model_name
    try:
        from transformers import pipeline as _hf_pipeline  # type: ignore

        _pipe2 = _hf_pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=512,
            device=-1,
        )
        logger.info("Analysis ML second classifier loaded: %s", model_name)
    except Exception as exc:
        logger.warning(
            "Analysis ML second classifier failed to load (skipping ensemble): %s", exc
        )
        _pipe2 = None


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClassifierResult:
    """
    Result from the async ML injection classifier.

    score:  injection probability [0.0, 1.0]
    label:  "safe" | "suspicious" | "malicious"
    model:  model name used, or "degraded" when transformers is not installed
    """
    score:      float
    label:      str
    model:      str
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Core classify() function
# ---------------------------------------------------------------------------

def classify(
    text: str,
    *,
    model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
    threshold: float = 0.85,
) -> ClassifierResult:
    """
    Run the ML injection classifier on a text string.

    Thread-safe. Lazy-loads the model on first call.

    Returns ClassifierResult(score=0.0, label="safe", model="degraded")
    if transformers is not installed or the model fails to load.

    The *threshold* is the injection probability above which the result is
    labelled "malicious". Scores >= threshold*0.6 are labelled "suspicious".
    """
    import time
    t0 = time.monotonic()

    with _model_lock:
        if not _pipe_loaded:
            _load_pipeline(model_name)

    if _pipe is None:
        return ClassifierResult(score=0.0, label="safe", model="degraded")

    try:
        # Do NOT pre-truncate here — the pipeline already has truncation=True,
        # max_length=512 set at load time. The HuggingFace tokenizer handles
        # truncation at 512 tokens (~2 048 chars) which is longer than the
        # old hard 1 024-char cut. The caller (enforcement/__init__.py) is
        # responsible for passing the right window(s) of text.
        result = _pipe(text)
        if isinstance(result, list):
            result = result[0]

        raw_label: str = (result.get("label") or "").upper()
        raw_score: float = float(result.get("score") or 0.0)

        # Normalize across label conventions used by different fine-tuned checkpoints:
        #   protectai/deberta-v3-base-prompt-injection-v2: INJECTION / SAFE
        #   older variants: LABEL_0 (safe) / LABEL_1 (injection)
        if raw_label in ("INJECTION", "LABEL_1", "1"):
            injection_score = raw_score
        elif raw_label in ("SAFE", "LABEL_0", "0"):
            injection_score = 1.0 - raw_score
        else:
            # Unknown label convention: treat raw_score as injection probability
            injection_score = raw_score

        injection_score = float(min(1.0, max(0.0, injection_score)))

        if injection_score >= threshold:
            label = "malicious"
        elif injection_score >= threshold * 0.6:
            label = "suspicious"
        else:
            label = "safe"

        latency_ms = (time.monotonic() - t0) * 1000
        return ClassifierResult(
            score=round(injection_score, 4),
            label=label,
            model=_pipe_model_name or model_name,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        logger.warning("Classifier inference error (degrading to safe): %s", exc)
        return ClassifierResult(score=0.0, label="safe", model="degraded")


def classify_second(
    text: str,
    *,
    model_name: str,
    threshold: float = 0.85,
) -> ClassifierResult:
    """
    Run the SECONDARY ensemble classifier (e.g. meta-llama/Prompt-Guard-86M).

    Same interface as classify() but uses a separate model singleton so both
    models can be warm simultaneously. Returns degraded result if model is
    unavailable or fails to load.

    Why a second model?
    -------------------
    DeBERTa (primary) has blind spots on subtle persona jailbreaks and
    social-engineering framing.  A second model with a different architecture
    (RoBERTa-based PromptGuard) trained on different data catches attacks the
    primary misses.  The ensemble rule is max(primary, secondary) — if either
    fires, the injection is flagged.
    """
    import time
    t0 = time.monotonic()

    with _pipe2_lock:
        if not _pipe2_loaded:
            _load_pipeline2(model_name)

    if _pipe2 is None:
        return ClassifierResult(score=0.0, label="safe", model="degraded")

    try:
        result = _pipe2(text)
        if isinstance(result, list):
            result = result[0]

        raw_label: str = (result.get("label") or "").upper()
        raw_score: float = float(result.get("score") or 0.0)

        # PromptGuard and other models use various label conventions
        if raw_label in ("INJECTION", "JAILBREAK", "LABEL_1", "1", "HARMFUL", "UNSAFE"):
            injection_score = raw_score
        elif raw_label in ("SAFE", "BENIGN", "LABEL_0", "0", "UNHARMFUL"):
            injection_score = 1.0 - raw_score
        else:
            injection_score = raw_score

        injection_score = float(min(1.0, max(0.0, injection_score)))

        if injection_score >= threshold:
            label = "malicious"
        elif injection_score >= threshold * 0.6:
            label = "suspicious"
        else:
            label = "safe"

        latency_ms = (time.monotonic() - t0) * 1000
        return ClassifierResult(
            score=round(injection_score, 4),
            label=label,
            model=_pipe2_model_name or model_name,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        logger.warning("Second classifier inference error (skipping): %s", exc)
        return ClassifierResult(score=0.0, label="safe", model="degraded")


def is_available() -> bool:
    """Return True if the transformers library and a loaded model are available."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
