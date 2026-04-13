"""
Anthropic LLM Client
=====================
Supports: Claude models via Anthropic API.
"""

import os
import time
from typing import List, Optional

from .base import BaseLLMClient, LLMMessage, LLMResponse


class AnthropicClient(BaseLLMClient):
    """
    LLM client for Anthropic Claude models.

    Usage:
        client = AnthropicClient("claude-sonnet-4-20250514")
        response = client.chat([
            LLMMessage("system", "You are a financial analyst."),
            LLMMessage("user", "Analyze AAPL"),
        ])
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            client_kwargs = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            api_key = self.kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                client_kwargs["api_key"] = api_key

            self._client = Anthropic(**client_kwargs)

        return self._client

    def chat(self, messages: List[LLMMessage],
             temperature: float = 0.0,
             max_tokens: int = 4096) -> LLMResponse:
        """Send chat completion via Anthropic Messages API."""
        client = self._get_client()

        # Anthropic requires system message separate
        system = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        start = time.time()

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        latency = (time.time() - start) * 1000

        # Extract text content
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        content = "\n".join(text_blocks)

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            model=response.model,
            provider="anthropic",
            usage=usage,
            latency_ms=round(latency, 1),
            raw=response,
        )
