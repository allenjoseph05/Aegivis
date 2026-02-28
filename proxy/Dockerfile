# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Optional extras installed at build time.
# - pii:        Presidio PII masking (requires spaCy model download below)
# - security:   RCE/SSRF scanner enhancements (bashlex, sqlglot)
# - classifier: Async ML injection classifier (transformers, torch ~280 MB on first run)
# - ml:         Behavioral Isolation Forest (scikit-learn)
# Strip for lightweight CI/dev: PROXY_BUILD_EXTRAS=pii,security docker compose build proxy
ARG BUILD_EXTRAS=pii,security,classifier,ml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Phase 1: Install all Python dependencies ──────────────────────────────
# Cached until pyproject.toml changes. Code changes do NOT bust this layer.
# Empty stub satisfies hatchling package discovery without the real source.
COPY pyproject.toml .
RUN mkdir -p app && touch app/__init__.py

# Pre-install CPU-only torch BEFORE the main extras step.
# Without this, pip pulls torch from PyPI which includes the full CUDA
# distribution (~2.8 GB: torch 915 MB + cublas 594 MB + cudnn 706 MB + ...).
# The CPU wheel from pytorch.org is ~180 MB — sufficient for DeBERTa inference.
# When torch is already present, the subsequent ".[classifier]" install keeps it.
RUN --mount=type=cache,target=/root/.cache/pip \
    if echo "${BUILD_EXTRAS}" | grep -q "classifier"; then \
        pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$BUILD_EXTRAS" ]; then pip install ".[$BUILD_EXTRAS]"; else pip install "."; fi

# spaCy NER model — cached here, never re-downloaded on source changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_sm 2>/dev/null || \
    echo "spaCy model not available - PII detection will be disabled"

# ── Phase 2: Overlay real source (runs on every code change, ~10s) ────────
COPY app/ app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
