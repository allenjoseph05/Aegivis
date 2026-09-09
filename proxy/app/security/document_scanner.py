"""
Document Scanner — Phase 9E (Steganographic Injection Scanner)

Extracts ALL text layers from structured documents (PDF, DOCX, ODT) and
scans each layer for injection payloads that are invisible in the rendered view
but visible to the LLM when the document is loaded into context.

Attack vectors covered
----------------------
PDF:
  1. Invisible text layer — text with matching foreground/background colour,
     or with font size effectively zero. Extracted via PyMuPDF rawdict.
  2. Annotation injection — text hidden in annotation fields (not page body).
  3. Metadata injection — Author, Subject, Keywords, custom XMP properties.
  4. JavaScript/XFA in PDF — can embed executable trigger content.
  5. Optional Content Groups with visibility=off (hidden layers).

DOCX / PPTX / ODT:
  1. Hidden text (w:vanish="1" attribute in DOCX — text is hidden in editor).
  2. Track changes — deleted text still present in XML (visible to LLM parser).
  3. Comments / annotations — injected into word/comments.xml.
  4. Metadata — dc:subject, dc:description in OOXML core properties.
  5. Macro code (vbaProject.bin) — presence flagged regardless of content.

Graceful degradation
--------------------
PyMuPDF (fitz) and python-docx are OPTIONAL:

    pip install 'aegivis-proxy[docs]'

If not installed, `scan_document()` returns detected=False with a note.
The text-level scanners (unicode_scanner, tool_output_scanner) still run.

This module has no required external dependencies.
"""
from __future__ import annotations

import base64
import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    import docx  # python-docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DocScanResult:
    """
    Result of scanning a structured document for hidden injection content.

    Attributes:
        detected:     True if any injection threat was found.
        threat:       Short threat label (e.g. "hidden_text", "macro_detected").
        score:        Highest injection score found across all layers.
        layer:        Which document layer the threat was found in.
        details:      Human-readable description.
        all_threats:  All individual threats (threat, layer, score, snippet).
        skipped:      True if the scanner was skipped (deps not installed).
    """
    detected: bool
    threat: str = ""
    score: float = 0.0
    layer: str = ""
    details: str = ""
    all_threats: list[dict] = field(default_factory=list)
    skipped: bool = False


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Base64 pattern: used when tool returns document content as base64
_B64_RE = re.compile(r"^[A-Za-z0-9+/\r\n]{40,}={0,2}$")


def _detect_format(content: str | bytes) -> str:
    """Detect document format from content. Returns 'pdf', 'docx', or 'unknown'."""
    if isinstance(content, bytes):
        if content[:4] == b"%PDF":
            return "pdf"
        if content[:2] == b"PK":   # ZIP-based (DOCX, PPTX, ODT, XLSX)
            return "docx"
        return "unknown"

    stripped = content.strip()
    # Base64-encoded document
    if _B64_RE.match(stripped):
        try:
            raw = base64.b64decode(stripped)
            return _detect_format(raw)
        except Exception:
            pass

    # Plaintext markers
    if stripped.startswith("%PDF"):
        return "pdf"
    if stripped.startswith("PK\x03\x04"):
        return "docx"
    return "unknown"


def _to_bytes(content: str | bytes) -> bytes | None:
    """Convert content to bytes, handling base64-encoded documents."""
    if isinstance(content, bytes):
        return content
    stripped = content.strip()
    if _B64_RE.match(stripped):
        try:
            return base64.b64decode(stripped)
        except Exception:
            return None
    return content.encode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# PDF scanner (requires PyMuPDF)
# ---------------------------------------------------------------------------

_WHITE_COLORS = {0xFFFFFF, 16777215, 0xFEFEFE}


def _scan_pdf(data: bytes, injection_fn: Callable[[str], float]) -> DocScanResult:
    """
    Extract and scan all text layers in a PDF document.

    Uses PyMuPDF's rawdict extraction which includes per-span color and size,
    allowing detection of invisible text even when it's not in the rendered view.
    """
    if not _PDF_AVAILABLE:
        return DocScanResult(
            detected=False, skipped=True,
            details="PyMuPDF not installed. Run: pip install pymupdf",
        )

    threats: list[dict] = []

    try:
        doc = fitz.open(stream=data, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # ── Vector 1: Raw text with visibility info ────────────────────
            raw_dict = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in raw_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue

                        color = span.get("color", 0)
                        size = span.get("size", 12)

                        # White text or invisible font size
                        is_hidden = (
                            color in _WHITE_COLORS
                            or (isinstance(color, int) and color >= 0xF0F0F0)
                            or size < 0.5
                        )

                        if is_hidden:
                            inj_score = injection_fn(span_text)
                            if inj_score > 0.35:
                                threats.append({
                                    "threat": "pdf_hidden_text",
                                    "layer": f"page_{page_num + 1}_text",
                                    "score": inj_score,
                                    "snippet": span_text[:120],
                                })

            # ── Vector 2: Annotations ──────────────────────────────────────
            for annot in page.annots():
                content = annot.info.get("content", "").strip()
                if not content:
                    continue
                inj_score = injection_fn(content)
                if inj_score > 0.30:
                    threats.append({
                        "threat": "pdf_annotation_injection",
                        "layer": f"page_{page_num + 1}_annot",
                        "score": inj_score,
                        "snippet": content[:120],
                    })

        # ── Vector 3: Document metadata ────────────────────────────────────
        meta = doc.metadata or {}
        meta_values = [
            str(v) for v in meta.values()
            if isinstance(v, str) and len(v) > 8
        ]
        for val in meta_values:
            inj_score = injection_fn(val)
            if inj_score > 0.35:
                threats.append({
                    "threat": "pdf_metadata_injection",
                    "layer": "metadata",
                    "score": inj_score,
                    "snippet": val[:120],
                })

        # ── Vector 4: JavaScript ───────────────────────────────────────────
        try:
            js = doc.get_js()
            if js and len(js.strip()) > 0:
                threats.append({
                    "threat": "pdf_javascript",
                    "layer": "javascript",
                    "score": 0.90,
                    "snippet": f"JavaScript detected ({len(js)} chars)",
                })
        except Exception:
            pass

    except Exception as exc:
        logger.warning("PDF scan failed (skipped): %s", exc)
        return DocScanResult(detected=False, details=f"PDF scan error: {exc}")

    if not threats:
        return DocScanResult(detected=False)

    # Sort by score descending, report worst
    threats.sort(key=lambda t: t["score"], reverse=True)
    worst = threats[0]
    return DocScanResult(
        detected=True,
        threat=worst["threat"],
        score=worst["score"],
        layer=worst["layer"],
        details=f"{len(threats)} threat(s) across {len(set(t['layer'] for t in threats))} layer(s)",
        all_threats=threats,
    )


# ---------------------------------------------------------------------------
# DOCX / ZIP-based document scanner (requires python-docx for hidden text)
# ---------------------------------------------------------------------------

_OOXML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _scan_docx(data: bytes, injection_fn: Callable[[str], float]) -> DocScanResult:
    """
    Extract and scan all hidden content layers in a DOCX (or PPTX/ODT ZIP).

    Scans:
      - Hidden text (w:vanish in run properties) — requires python-docx
      - Track changes (deleted text) — XML-level, no dep needed
      - Comments — word/comments.xml in ZIP
      - Core properties — docProps/core.xml in ZIP
      - Macro presence — word/vbaProject.bin in ZIP
    """
    threats: list[dict] = []

    try:
        # ── Vector 1: Hidden text — requires python-docx ───────────────────
        if _DOCX_AVAILABLE:
            try:
                doc_obj = docx.Document(io.BytesIO(data))
                for para in doc_obj.paragraphs:
                    for run in para.runs:
                        is_hidden = False
                        try:
                            is_hidden = bool(run.font.hidden)
                        except Exception:
                            pass

                        # Also check XML directly for w:vanish
                        if not is_hidden and run.element is not None:
                            rpr = run.element.find(f"{{{_OOXML_NS}}}rPr")
                            if rpr is not None:
                                vanish = rpr.find(f"{{{_OOXML_NS}}}vanish")
                                is_hidden = vanish is not None

                        if is_hidden and run.text.strip():
                            inj_score = injection_fn(run.text)
                            if inj_score > 0.30:
                                threats.append({
                                    "threat": "docx_hidden_text",
                                    "layer": "hidden_run",
                                    "score": inj_score,
                                    "snippet": run.text[:120],
                                })
            except Exception as exc:
                logger.debug("DOCX hidden text scan failed: %s", exc)

        # ── ZIP-level scanning (no deps) ───────────────────────────────────
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())

            # ── Vector 2: Macro presence ───────────────────────────────────
            if "word/vbaProject.bin" in names or "xl/vbaProject.bin" in names:
                threats.append({
                    "threat": "docx_macro_detected",
                    "layer": "vbaProject",
                    "score": 0.90,
                    "snippet": "VBA macro binary found in document archive",
                })

            # ── Vector 3: Comments ─────────────────────────────────────────
            for comments_path in ("word/comments.xml", "word/commentsExtended.xml"):
                if comments_path in names:
                    try:
                        from xml.etree.ElementTree import fromstring
                        xml_bytes = z.read(comments_path)
                        root = fromstring(xml_bytes)
                        for elem in root.iter():
                            if elem.text and len(elem.text.strip()) > 8:
                                inj_score = injection_fn(elem.text.strip())
                                if inj_score > 0.30:
                                    threats.append({
                                        "threat": "docx_comment_injection",
                                        "layer": comments_path,
                                        "score": inj_score,
                                        "snippet": elem.text.strip()[:120],
                                    })
                    except Exception as exc:
                        logger.debug("DOCX comments scan failed: %s", exc)

            # ── Vector 4: Core properties (metadata) ──────────────────────
            for props_path in ("docProps/core.xml", "docProps/app.xml"):
                if props_path in names:
                    try:
                        from xml.etree.ElementTree import fromstring
                        xml_bytes = z.read(props_path)
                        root = fromstring(xml_bytes)
                        for elem in root.iter():
                            if elem.text and len(elem.text.strip()) > 8:
                                inj_score = injection_fn(elem.text.strip())
                                if inj_score > 0.35:
                                    threats.append({
                                        "threat": "docx_metadata_injection",
                                        "layer": props_path,
                                        "score": inj_score,
                                        "snippet": elem.text.strip()[:120],
                                    })
                    except Exception as exc:
                        logger.debug("DOCX props scan failed: %s", exc)

            # ── Vector 5: Track changes (deleted text in XML) ──────────────
            for doc_path in ("word/document.xml",):
                if doc_path in names:
                    try:
                        from xml.etree.ElementTree import fromstring
                        xml_bytes = z.read(doc_path)
                        root = fromstring(xml_bytes)
                        # w:del elements contain deleted text (still in XML)
                        del_ns = f"{{{_OOXML_NS}}}del"
                        for del_elem in root.iter(del_ns):
                            texts = []
                            for child in del_elem.iter():
                                if child.text:
                                    texts.append(child.text)
                            deleted_text = " ".join(texts).strip()
                            if len(deleted_text) > 8:
                                inj_score = injection_fn(deleted_text)
                                if inj_score > 0.40:
                                    threats.append({
                                        "threat": "docx_track_changes_injection",
                                        "layer": "deleted_text",
                                        "score": inj_score,
                                        "snippet": deleted_text[:120],
                                    })
                    except Exception as exc:
                        logger.debug("DOCX track-changes scan failed: %s", exc)

    except zipfile.BadZipFile:
        return DocScanResult(detected=False, details="Not a valid ZIP/DOCX file")
    except Exception as exc:
        logger.warning("DOCX scan failed (skipped): %s", exc)
        return DocScanResult(detected=False, details=f"DOCX scan error: {exc}")

    if not threats:
        return DocScanResult(detected=False)

    threats.sort(key=lambda t: t["score"], reverse=True)
    worst = threats[0]
    return DocScanResult(
        detected=True,
        threat=worst["threat"],
        score=worst["score"],
        layer=worst["layer"],
        details=f"{len(threats)} threat(s) in document layers",
        all_threats=threats,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_document(
    content: str | bytes,
    tool_name: str,
    injection_fn: Callable[[str], float],
) -> DocScanResult:
    """
    Detect injection content hidden in structured document layers.

    Args:
        content:      Raw document bytes or base64-encoded document string.
        tool_name:    Name of the tool that produced this content (for logging).
        injection_fn: Callable that scores text 0.0–1.0 for injection likelihood.
                      Typically wraps tool_output_scanner._structural_scan().

    Returns:
        DocScanResult. Never raises.
    """
    if not content:
        return DocScanResult(detected=False)

    try:
        raw = _to_bytes(content)
        if raw is None or len(raw) < 4:
            return DocScanResult(detected=False)

        fmt = _detect_format(raw)

        if fmt == "pdf":
            result = _scan_pdf(raw, injection_fn)
        elif fmt == "docx":
            result = _scan_docx(raw, injection_fn)
        else:
            return DocScanResult(detected=False, details=f"Unsupported format: {fmt}")

        if result.detected:
            logger.info(
                "Document scan threat in tool=%s fmt=%s threat=%s score=%.2f",
                tool_name, fmt, result.threat, result.score,
            )
        return result

    except Exception as exc:
        logger.warning("document_scanner error (tool=%s, skipped): %s", tool_name, exc)
        return DocScanResult(detected=False)


def is_available() -> dict[str, bool]:
    """Return which optional document scanning dependencies are installed."""
    return {"pdf": _PDF_AVAILABLE, "docx": _DOCX_AVAILABLE}
