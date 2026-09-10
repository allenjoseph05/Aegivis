"""
Generic OpenAI-compatible provider factory.

All providers below use the OpenAI Chat Completions API format exactly —
same request shape, same response shape, same SSE streaming protocol.
We reuse the OpenAI parser; only the provider name and upstream URL differ.

Supported providers:
  groq, mistral, together, perplexity, deepseek, xai,
  fireworks, openrouter, cerebras, sambanova, nvidia, cohere
"""
from __future__ import annotations

from .openai import SSEAssembler, extract_request_params, parse_response, parse_sse_chunk


def make_compat_provider(provider_name: str):
    """Return a provider class for an OpenAI-compatible API."""

    class _CompatProvider:
        name = provider_name

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

    _CompatProvider.__name__ = f"{provider_name.capitalize()}Provider"
    _CompatProvider.__qualname__ = _CompatProvider.__name__
    return _CompatProvider


# Pre-built provider classes — one per OpenAI-compatible service
GroqProvider       = make_compat_provider("groq")
MistralProvider    = make_compat_provider("mistral")
TogetherProvider   = make_compat_provider("together")
PerplexityProvider = make_compat_provider("perplexity")
DeepSeekProvider   = make_compat_provider("deepseek")
XAIProvider        = make_compat_provider("xai")
FireworksProvider  = make_compat_provider("fireworks")
OpenRouterProvider = make_compat_provider("openrouter")
CerebrasProvider   = make_compat_provider("cerebras")
SambanovaProvider  = make_compat_provider("sambanova")
NvidiaProvider     = make_compat_provider("nvidia")
CohereCompatProvider = make_compat_provider("cohere")

__all__ = [
    "make_compat_provider",
    "GroqProvider",
    "MistralProvider",
    "TogetherProvider",
    "PerplexityProvider",
    "DeepSeekProvider",
    "XAIProvider",
    "FireworksProvider",
    "OpenRouterProvider",
    "CerebrasProvider",
    "SambanovaProvider",
    "NvidiaProvider",
    "CohereCompatProvider",
]
