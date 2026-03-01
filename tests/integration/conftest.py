"""
Fixtures for AgentBlackBox end-to-end integration smoke tests.

Requires the Docker Compose stack to be running:
    docker compose up -d

Override URLs via environment variables:
    ABB_PROXY_URL     (default: http://localhost:8080)
    ABB_BACKEND_URL   (default: http://localhost:8000)
    ABB_API_KEY       (default: dev-dashboard-key)
"""
import os
import pytest
import httpx

PROXY_URL = os.getenv("ABB_PROXY_URL", "http://localhost:8080")
BACKEND_URL = os.getenv("ABB_BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("ABB_API_KEY", "dev-dashboard-key")


@pytest.fixture(scope="session")
def proxy_url() -> str:
    return PROXY_URL


@pytest.fixture(scope="session")
def backend_client() -> httpx.Client:
    with httpx.Client(
        base_url=BACKEND_URL,
        headers={"X-API-Key": API_KEY},
        timeout=15,
    ) as client:
        yield client


@pytest.fixture(scope="session")
def proxy_client() -> httpx.Client:
    with httpx.Client(base_url=PROXY_URL, timeout=15) as client:
        yield client
