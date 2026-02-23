"""
Azure OpenAI API format handler.

Azure OpenAI uses the same request/response format as OpenAI,
but with a different URL structure:
  POST https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}

We reuse the OpenAI provider for parsing; this handler just manages routing.
"""
from __future__ import annotations

from .openai import (
    SSEAssembler,
    extract_request_params,
    parse_response,
    parse_sse_chunk,
)

PROVIDER_NAME = "azure"


class AzureProvider:
    name = PROVIDER_NAME

    @staticmethod
    def extract_request_params(body: dict) -> dict:
        return extract_request_params(body)

    @staticmethod
    def parse_response(body: dict) -> dict:
        return parse_response(body)

    @staticmethod
    def parse_sse_chunk(chunk_data: str) -> dict | None:
        return parse_sse_chunk(chunk_data)

    @staticmethod
    def new_assembler() -> SSEAssembler:
        return SSEAssembler()
