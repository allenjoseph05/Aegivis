FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source + metadata together (hatchling needs app/ to build the package)
COPY pyproject.toml alembic.ini ./
COPY app/ app/
COPY migrations/ migrations/

RUN pip install --no-cache-dir .

EXPOSE 8000

# Run Alembic migrations then start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2"]
