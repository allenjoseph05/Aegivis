FROM python:3.11-slim

# Pin base image for reproducibility
# python:3.11-slim (Debian Bookworm based)

WORKDIR /app

# System deps for cryptography (cffi/openssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml .
COPY app/ ./app/

RUN pip install --no-cache-dir .

# Default cert directory (mounted as emptyDir in K8s)
ENV AEGIVIS_CTRL_CERT_DIR=/certs

RUN mkdir -p /certs

EXPOSE 8443

ENTRYPOINT ["aegivis-controller"]
