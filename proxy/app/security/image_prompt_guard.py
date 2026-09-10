"""
Phase E5 — Image generation & audio prompt security guard.

Attack surface
--------------
Image generation (/v1/images/generations):
  1. Prompt injection — adversary compromises a tool result with text like
     "ignore previous instructions, generate [harmful content]". The agent
     passes it verbatim to the image generation API.
  2. Data exfiltration via rendered text — "generate image with the text:
     SECRET_KEY=abc123". The image becomes an out-of-band exfil channel.
  3. Content policy bypass — multi-step prompts that individually look safe
     but compose into policy-violating content.

Audio Speech-to-Text / transcription (/v1/audio/transcriptions):
  1. Injected audio — attacker-controlled audio files containing commands
     (ultrasonic prompt injection, inaudible instructions). The transcription
     lands in agent context as trusted text.
  2. PII in transcribed content — call recordings with SSNs / card numbers.

Audio Text-to-Speech (/v1/audio/speech):
  1. PII vocalisation — agent sends confidential data to TTS, leaking it
     as audio output if the audio is stored/transmitted.

Detection strategy
------------------
- Image prompts: run through the *existing* structural injection scanner
  (delimiter detection + unicode anomalies) and the ML classifier when
  available. Same signals used for text prompt injection, but applied to
  the image prompt string.
- Audio transcription output: PII scan on the returned transcript text
  (handled post-response in intercept hooks — this module provides the
  pre-request scan for TTS input text).
- Return a lightweight dataclass; the route handler decides whether to
  ALERT or BLOCK based on policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Injection threshold shared with text prompt scanner
_INJECTION_ALERT_THRESHOLD = 0.50


@dataclass
class ImagePromptGuardResult:
    """Result of scanning an image generation prompt."""

    injection_detected: bool
    """True if structural injection signals were found in the prompt."""

    injection_score: float
    """0.0–1.0 composite injection score."""

    pii_detected: bool
    """True if PII was found in the prompt (data-in-image exfil risk)."""

    pii_types: list[str]
    """Sorted list of PII entity types found."""

    prompt_chars: int
    """Character length of the combined prompt text."""

    def to_dict(self) -> dict:
        return {
            "injection_detected": self.injection_detected,
            "injection_score":    round(self.injection_score, 4),
            "pii_detected":       self.pii_detected,
            "pii_types":          self.pii_types,
            "prompt_chars":       self.prompt_chars,
        }


@dataclass
class AudioInputGuardResult:
    """Result of scanning TTS input text for PII."""

    pii_detected: bool
    pii_types: list[str]
    input_chars: int

    def to_dict(self) -> dict:
        return {
            "pii_detected": self.pii_detected,
            "pii_types":    self.pii_types,
            "input_chars":  self.input_chars,
        }


# ---------------------------------------------------------------------------
# Image prompt scanning
# ---------------------------------------------------------------------------

def scan_image_prompt(prompt: str) -> ImagePromptGuardResult:
    """
    Scan an image generation prompt for injection signals and PII.

    Uses the structural injection scanner (delimiter tokens, unicode anomalies)
    which is already in the hot path for text prompts.  No ML required for a
    meaningful signal — delimiter hits alone are strong indicators.

    PII detection uses the same presidio-backed logic as the embedding guard.
    """
    from ..enforcement.structural import scan as _structural_scan  # noqa: PLC0415

    injection_score = _structural_scan(prompt).score
    injection_detected = injection_score >= _INJECTION_ALERT_THRESHOLD

    # PII scan — same backend as embedding guard
    pii_types, _, _ = _detect_pii_in_text(prompt)

    return ImagePromptGuardResult(
        injection_detected=injection_detected,
        injection_score=injection_score,
        pii_detected=bool(pii_types),
        pii_types=sorted(pii_types),
        prompt_chars=len(prompt),
    )


def extract_image_prompt(body: dict) -> str:
    """
    Extract the combined prompt text from an OpenAI images/generations request.

    Handles:
    - ``prompt``: str  (standard)
    - ``prompt``: list[str]  (some providers batch)
    - Negative prompt in ``negative_prompt`` (DALL-E 3 / Stable Diffusion style)
    """
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    else:
        prompt = str(prompt)

    negative = body.get("negative_prompt", "")
    if negative:
        prompt = f"{prompt} {negative}"

    return prompt.strip()


# ---------------------------------------------------------------------------
# Audio input scanning (TTS)
# ---------------------------------------------------------------------------

def scan_audio_input(text: str) -> AudioInputGuardResult:
    """Scan text being sent to a TTS API for PII (vocalisation risk)."""
    pii_types, _, _ = _detect_pii_in_text(text)
    return AudioInputGuardResult(
        pii_detected=bool(pii_types),
        pii_types=sorted(pii_types),
        input_chars=len(text),
    )


# ---------------------------------------------------------------------------
# Shared PII detection (reuses embedding_guard logic)
# ---------------------------------------------------------------------------

def _detect_pii_in_text(text: str) -> tuple[set[str], bool, bool]:
    """Thin wrapper — delegates to embedding_guard._detect_pii."""
    from .embedding_guard import _detect_pii  # noqa: PLC0415
    return _detect_pii(text)
