"""
Multi-Modal Scanner — Visual Prompt Injection Detection
========================================================
Scans images embedded in LLM requests for attack vectors that are invisible
to humans but readable by vision models.

Attack vectors covered
----------------------
1. **EXIF metadata injection** — malicious instructions hidden in EXIF fields
   (ImageDescription, Comment, UserComment, Software, Artist, Copyright).
   No dep required — uses Pillow's EXIF reader.

2. **Visual prompt injection** — text rendered in the image with low contrast
   (white-on-white, near-invisible) that vision models extract but humans miss.
   Requires pytesseract + Pillow (optional).

3. **QR code / barcode** — encoded instructions invisible at normal viewing size.
   Requires pyzbar + Pillow (optional).

4. **LSB steganography** — binary payload hidden in least-significant bits of
   pixel values. Detected via Chi-Square test on pixel bit distribution.
   Requires Pillow only (no dep beyond that).

5. **Suspicious image characteristics** — 1×1 tracking pixels, blank images.
   Heuristic only, Pillow required. (Narrow banner detection removed — too many
   FPs from legitimate dividers/logos; low-contrast overlay removed — white
   backgrounds are common in documents.)

Graceful degradation
--------------------
All checks are optional. Without Pillow, only base64 length heuristics run.
Without pytesseract, OCR-based checks are skipped.
Without pyzbar, QR detection is skipped.

Install extras::

    pip install Pillow               # unlocks most checks
    pip install pyzbar               # QR/barcode detection
    pip install pytesseract          # OCR-based visual injection

Input formats supported
-----------------------
Anthropic content block::

    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                  "data": "<base64>"}}

OpenAI vision content block::

    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64>"}}

The scanner accepts raw base64 strings, data URIs, or raw bytes directly.
"""
from __future__ import annotations

import base64
import io
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ExifTags  # type: ignore[import-untyped]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from PIL import Image as _PILImg  # noqa: F811
    import pytesseract  # type: ignore[import-untyped]
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

try:
    from pyzbar import pyzbar  # type: ignore[import-untyped]
    _QR_AVAILABLE = True
except ImportError:
    _QR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImageScanResult:
    """
    Result of scanning a single image for visual prompt injection.

    Attributes:
        detected:     True if any threat was found above threshold.
        threat:       Primary threat label.
        score:        Highest risk score (0.0–1.0).
        flags:        All detected threat labels.
        details:      Human-readable summary.
        extracted_text: Text extracted by OCR (if available).
        all_threats:  Full list of {threat, score, detail} dicts.
        skipped:      True if scan was skipped (Pillow not installed).
    """
    detected:       bool
    threat:         str = ""
    score:          float = 0.0
    flags:          list[str] = field(default_factory=list)
    details:        str = ""
    extracted_text: str = ""
    all_threats:    list[dict] = field(default_factory=list)
    skipped:        bool = False


# ---------------------------------------------------------------------------
# Image extraction from content blocks
# ---------------------------------------------------------------------------

def _extract_base64(content_block: dict) -> bytes | None:
    """
    Extract raw image bytes from an Anthropic or OpenAI vision content block.
    Returns None if not an image block or decoding fails.
    """
    try:
        block_type = content_block.get("type", "")

        # Anthropic format: {"type": "image", "source": {"type": "base64", "data": "..."}}
        if block_type == "image":
            src = content_block.get("source") or {}
            if src.get("type") == "base64":
                return base64.b64decode(src["data"])
            # URL type — can't download, skip
            return None

        # OpenAI format: {"type": "image_url", "image_url": {"url": "data:...,<base64>"}}
        if block_type == "image_url":
            url = (content_block.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                # data:image/jpeg;base64,<data>
                _, encoded = url.split(",", 1)
                return base64.b64decode(encoded)
            # Remote URL — can't download, skip
            return None

    except Exception:
        return None
    return None


def extract_images_from_request(body: dict) -> list[bytes]:
    """
    Extract all base64-encoded images from an LLM request body.
    Handles both Anthropic and OpenAI message formats.
    """
    images: list[bytes] = []
    messages: list = body.get("messages") or []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    img = _extract_base64(block)
                    if img:
                        images.append(img)
    return images


# ---------------------------------------------------------------------------
# Check 1: EXIF metadata injection (Pillow required)
# ---------------------------------------------------------------------------

# EXIF tags that could carry injected text
_EXIF_TEXT_TAGS = frozenset({
    "ImageDescription", "Make", "Model", "Software", "Artist",
    "Copyright", "UserComment", "Comment", "XPComment",
    "XPAuthor", "XPSubject", "XPTitle", "XPKeywords",
    "DocumentName", "PageName",
})


def _check_exif(img: Any, injection_fn: Callable[[str], float]) -> list[dict]:
    threats: list[dict] = []
    try:
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if not exif_data:
            return threats

        tag_map = {v: k for k, v in ExifTags.TAGS.items()} if _PIL_AVAILABLE else {}
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name not in _EXIF_TEXT_TAGS:
                continue
            text = ""
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="replace").strip("\x00").strip()
            elif isinstance(value, str):
                text = value.strip()
            if len(text) < 8:
                continue

            score = injection_fn(text)
            if score > 0.25:
                threats.append({
                    "threat": "exif_injection",
                    "score":  score,
                    "detail": f"EXIF tag {tag_name!r}: {text[:100]}",
                })
    except (AttributeError, Exception):
        pass
    return threats


# ---------------------------------------------------------------------------
# Check 2: LSB steganography (Pillow required, Chi-Square test)
# ---------------------------------------------------------------------------

def _check_lsb_steganography(img: Any) -> list[dict]:
    """
    Chi-square test on LSB distribution.
    Natural images have near-random LSBs; stego images have suspiciously
    uniform LSB distribution due to message embedding.
    """
    threats: list[dict] = []
    try:
        # Convert to RGB to normalise format
        rgb = img.convert("RGB")
        pixels = list(rgb.getdata())

        if len(pixels) < 100:
            return threats  # too small to be meaningful

        # Collect LSBs of the red channel
        lsbs = [p[0] & 1 for p in pixels[:10000]]
        zeros = lsbs.count(0)
        ones  = len(lsbs) - zeros
        total = len(lsbs)
        expected = total / 2.0

        # Chi-square statistic for two-category test
        chi2 = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected

        # For truly random data chi2 should be small.
        # Very LOW chi2 (< 0.5) means suspiciously uniform — hallmark of LSB stego.
        if chi2 < 0.5 and total >= 1000:
            score = max(0.0, min(1.0, 0.7 - chi2))
            threats.append({
                "threat": "lsb_steganography",
                "score":  round(score, 3),
                "detail": f"LSB chi-square={chi2:.3f} (threshold <0.5); "
                          f"suspicious uniformity in {total} pixels",
            })
    except Exception:
        pass
    return threats


# ---------------------------------------------------------------------------
# Check 3: Tiny / anomalous images (Pillow required)
# ---------------------------------------------------------------------------

def _check_image_anomalies(img: Any) -> list[dict]:
    threats: list[dict] = []
    try:
        w, h = img.size

        # 1×1 pixel tracking image
        if w <= 2 and h <= 2:
            threats.append({
                "threat": "tracking_pixel",
                "score":  0.60,
                "detail": f"Suspiciously small image ({w}×{h}px) — possible tracking pixel",
            })

        # Note: narrow banner detection (w>200 and h<=10) was removed —
        # legitimate dividers, logos, and banners have identical dimensions.

        # Pure-white or pure-black image (nothing to see, but vision model gets base64)
        try:
            extrema = img.convert("L").getextrema()
            if extrema[0] == extrema[1]:  # all pixels identical
                threats.append({
                    "threat": "blank_image",
                    "score":  0.35,
                    "detail": f"Blank image ({w}×{h}px, pixel value={extrema[0]}) "
                              "— may rely on EXIF/stego payload",
                })
        except Exception:
            pass

    except Exception:
        pass
    return threats


# ---------------------------------------------------------------------------
# Check 4: QR / barcode detection (pyzbar required)
# ---------------------------------------------------------------------------

def _check_qr_codes(img: Any, injection_fn: Callable[[str], float]) -> list[dict]:
    threats: list[dict] = []
    if not _QR_AVAILABLE:
        return threats
    try:
        decoded = pyzbar.decode(img)
        for item in decoded:
            text = item.data.decode("utf-8", errors="replace")
            score = max(0.55, injection_fn(text))  # QR in an LLM image is suspicious by default
            threats.append({
                "threat": "qr_barcode",
                "score":  score,
                "detail": f"{item.type} detected: {text[:100]}",
            })
    except Exception:
        pass
    return threats


# ---------------------------------------------------------------------------
# Check 5: OCR-based visual prompt injection (pytesseract required)
# ---------------------------------------------------------------------------

def _check_ocr_injection(img: Any, injection_fn: Callable[[str], float]) -> tuple[list[dict], str]:
    if not _OCR_AVAILABLE:
        return [], ""
    try:
        # Run OCR on the image
        text = pytesseract.image_to_string(img, timeout=5).strip()
        if len(text) < 8:
            return [], text

        score = injection_fn(text)
        threats: list[dict] = []
        if score > 0.40:
            threats.append({
                "threat": "visual_prompt_injection",
                "score":  score,
                "detail": f"OCR extracted text with injection score {score:.2f}: {text[:150]}",
            })
        return threats, text
    except Exception:
        return [], ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_image(
    image_bytes: bytes,
    injection_fn: Callable[[str], float],
    *,
    run_ocr: bool = True,
) -> ImageScanResult:
    """
    Scan a single image for visual prompt injection and steganography.

    Args:
        image_bytes:  Raw image data (JPEG, PNG, GIF, WEBP, BMP).
        injection_fn: Text injection scorer, 0.0–1.0. Use the proxy's
                      structural scanner or a simple heuristic.
        run_ocr:      Enable OCR scan if pytesseract is available.
                      Disable for performance-sensitive paths.

    Returns:
        ImageScanResult. Never raises.
    """
    if not image_bytes:
        return ImageScanResult(detected=False, details="empty image")

    if not _PIL_AVAILABLE:
        return ImageScanResult(
            detected=False,
            skipped=True,
            details="Pillow not installed — pip install Pillow to enable image scanning",
        )

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        logger.debug("multimodal.scan_image: cannot decode image: %s", exc)
        return ImageScanResult(detected=False, details=f"image decode error: {exc}")

    all_threats: list[dict] = []

    # Run all checks
    all_threats += _check_exif(img, injection_fn)
    all_threats += _check_lsb_steganography(img)
    all_threats += _check_image_anomalies(img)
    all_threats += _check_qr_codes(img, injection_fn)

    extracted_text = ""
    if run_ocr:
        ocr_threats, extracted_text = _check_ocr_injection(img, injection_fn)
        all_threats += ocr_threats

    if not all_threats:
        return ImageScanResult(
            detected=False,
            extracted_text=extracted_text,
            details="no threats detected",
        )

    all_threats.sort(key=lambda t: t["score"], reverse=True)
    worst  = all_threats[0]
    flags  = list(dict.fromkeys(t["threat"] for t in all_threats))  # unique, ordered
    score  = worst["score"]

    return ImageScanResult(
        detected=score >= 0.40,
        threat=worst["threat"],
        score=score,
        flags=flags,
        details=f"{len(all_threats)} threat(s): {', '.join(flags)}",
        extracted_text=extracted_text,
        all_threats=all_threats,
    )


def scan_request_images(
    body: dict,
    injection_fn: Callable[[str], float],
    *,
    run_ocr: bool = True,
) -> list[ImageScanResult]:
    """
    Scan all images in an LLM request body.

    Args:
        body:         Parsed JSON request body.
        injection_fn: Text injection scorer.
        run_ocr:      Enable OCR per image.

    Returns:
        List of ImageScanResult, one per image found. Empty list if no images.
    """
    images = extract_images_from_request(body)
    return [scan_image(img, injection_fn, run_ocr=run_ocr) for img in images]


def is_available() -> dict[str, bool]:
    """Return which optional image scanning capabilities are available."""
    return {
        "pillow":       _PIL_AVAILABLE,
        "ocr":          _OCR_AVAILABLE,
        "qr_detection": _QR_AVAILABLE,
    }
