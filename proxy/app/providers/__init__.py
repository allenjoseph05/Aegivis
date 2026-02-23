"""Provider-specific request/response parsers for the AgentBlackBox proxy."""
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .azure import AzureProvider
from .ollama import OllamaProvider

__all__ = [
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "AzureProvider",
    "OllamaProvider",
]
