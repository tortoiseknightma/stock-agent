"""
LLM Client Factory
====================
Creates the appropriate LLM client based on provider name.

Supported providers:
  OpenAI-compatible: openai, xai, deepseek, qwen, glm, openrouter, ollama
  Native:            anthropic, google

Usage:
    client = create_llm_client("openai", "gpt-4o")
    client = create_llm_client("anthropic", "claude-sonnet-4-20250514")
    client = create_llm_client("google", "gemini-2.0-flash")
    client = create_llm_client("deepseek", "deepseek-chat")
    client = create_llm_client("ollama", "llama3")
"""

from typing import Optional
from .base import BaseLLMClient

# Providers that use OpenAI-compatible chat completions API
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek", "qwen", "glm", "ollama", "openrouter",
)


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Create an LLM client for the specified provider.

    Args:
        provider: LLM provider name (openai, anthropic, google, deepseek, etc.)
        model: Model name/identifier
        base_url: Optional custom API endpoint
        **kwargs: Additional provider-specific args (api_key, etc.)

    Returns:
        Configured BaseLLMClient instance

    Examples:
        client = create_llm_client("openai", "gpt-4o")
        client = create_llm_client("anthropic", "claude-sonnet-4-20250514")
        client = create_llm_client("google", "gemini-2.0-flash")
        client = create_llm_client("deepseek", "deepseek-chat")
        client = create_llm_client("ollama", "llama3")
        client = create_llm_client("openai", "my-model", base_url="http://localhost:8000/v1")
    """
    provider_lower = provider.lower()

    if provider_lower in _OPENAI_COMPATIBLE:
        from .openai_compat import OpenAICompatibleClient
        return OpenAICompatibleClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        f"Supported: {', '.join(list(_OPENAI_COMPATIBLE) + ['anthropic', 'google'])}"
    )
