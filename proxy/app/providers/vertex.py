"""
Google Vertex AI API format handler (Phase E3).

Vertex AI uses the same generateContent request/response format as Google
Gemini (AI Studio) — so this module is a thin shim that delegates all
parsing to the existing GoogleProvider.  The only difference is:

  1. The URL path structure:
       Gemini AI Studio  : /v1/models/{model}:generateContent
       Vertex AI         : /v1beta1/projects/{project}/locations/{location}/
                            publishers/google/models/{model}:generateContent

  2. Authentication:
       Gemini AI Studio  : API key in ?key= query param or x-goog-api-key header
       Vertex AI         : GCP OAuth2 Bearer token in Authorization header

Neither difference affects parsing.  The Authorization header passes through
unchanged (same as the standard Gemini proxy route).

How agents route through the proxy
-----------------------------------
Set the GOOGLE_CLOUD_API_ENDPOINT environment variable to point at the proxy
before initialising the Vertex AI SDK:

    # google-cloud-aiplatform
    import vertexai
    vertexai.init(
        project="my-project",
        location="us-central1",
        api_endpoint="localhost:8080",          # proxy host (no https)
    )

    # google-genai unified SDK (v1.0+)
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=my-project
    GOOGLE_CLOUD_LOCATION=us-central1
    GOOGLE_GENAI_API_ENDPOINT=http://localhost:8080/vertex

The proxy route /vertex/{path:path} captures all traffic and constructs
the correct upstream Vertex AI URL dynamically from the path's {location}
segment, so no per-org configuration is needed.

Upstream URL construction
--------------------------
  path  = "v1beta1/projects/proj/locations/us-central1/publishers/..."
  → upstream = https://us-central1-aiplatform.googleapis.com
  → forward to  https://us-central1-aiplatform.googleapis.com/{path}

Falls back to AEGIVIS_VERTEX_LOCATION (default "us-central1") when the
location cannot be extracted from the path.
"""
from __future__ import annotations

import re

from .google import (          # reuse all parsing logic from Gemini provider
    extract_request_params,
    parse_response,
    parse_sse_chunk,
    SSEAssembler,
)

PROVIDER_NAME = "vertex"

# Regex to extract location from Vertex AI path
# e.g. "v1beta1/projects/proj/locations/us-central1/publishers/..."
_LOCATION_RE = re.compile(r"/locations/([^/]+)/")


def extract_location_from_path(path: str) -> str | None:
    """Extract the GCP region/location from a Vertex AI path."""
    m = _LOCATION_RE.search(path)
    return m.group(1) if m else None


def extract_model_from_path(path: str) -> str:
    """Extract the model ID from a Vertex AI path."""
    # e.g. ".../models/gemini-1.5-pro:generateContent"
    m = re.search(r"/models/([^/:]+)", path)
    return m.group(1) if m else "unknown"


def build_upstream_url(path: str, default_location: str = "us-central1") -> str:
    """Construct the real Vertex AI endpoint URL from the request path."""
    location = extract_location_from_path(path) or default_location
    return f"https://{location}-aiplatform.googleapis.com"


class VertexProvider:
    """Vertex AI provider — delegates all parsing to GoogleProvider."""
    name = PROVIDER_NAME

    @staticmethod
    def extract_request_params(body: dict, model: str = "unknown") -> dict:
        return extract_request_params(body, model)

    @staticmethod
    def parse_response(body: dict) -> dict:
        return parse_response(body)

    @staticmethod
    def parse_sse_chunk(chunk_data: str) -> dict | None:
        return parse_sse_chunk(chunk_data)

    @staticmethod
    def new_assembler() -> SSEAssembler:
        return SSEAssembler()
