FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source + metadata together (hatchling needs app/ to build the package)
COPY pyproject.toml .
COPY app/ app/

# Install with PII extras; fall back to base install if presidio unavailable
RUN pip install --no-cache-dir ".[pii]" || pip install --no-cache-dir .

# Download spaCy NER model for Presidio (best-effort; proxy works without it)
RUN python -m spacy download en_core_web_lg 2>/dev/null || \
    python -m spacy download en_core_web_sm 2>/dev/null || \
    echo "spaCy model not available - PII detection will be disabled"

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
