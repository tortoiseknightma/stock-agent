"""
OpenAI-Compatible LLM Client
==============================
Supports: OpenAI, DeepSeek, Qwen (DashScope), GLM, Ollama, OpenRouter, xAI

All of these providers expose an OpenAI-compatible chat completions API.
This client handles them all with provider-specific base URLs and env vars.
"""

import os
import time
from typing import List, Optional

from .base import BaseLLMClient, LLMMessage, LLMResponse


# Provider -> (base_url, api_key_env_var)
PROVIDER_CONFIG = {
    "openai":    ("https://api.openai.com/v1",        "OPENAI_API_KEY"),
    "xai":       ("https://api.x.ai/v1",              "XAI_API_KEY"),
    "deepseek":  ("https://api.deepseek.com",          "DEEPSEEK_API_KEY"),
    "qwen":      ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm":       ("https://api.z.ai/api/paas/v4/",    "ZHIPU_API_KEY"),
    "openrouter":("https://openrouter.ai/api/v1",     "OPENROUTER_API_KEY"),
    "ollama":    ("http://localhost:11434/v1",         None),
}


class OpenAICompatibleClient(BaseLLMClient):
    """
    LLM client for any provider with an OpenAI-compatible API.

    Usage:
        # OpenAI
        client = OpenAICompatibleClient("gpt-4o", provider="openai")

        # DeepSeek
        client = OpenAICompatibleClient("deepseek-chat", provider="deepseek")

        # Ollama (local)
        client = OpenAICompatibleClient("llama3", provider="ollama")

        # Custom endpoint
        client = OpenAICompatibleClient("my-model", base_url="http://localhost:8000/v1")
    """

    def __init__(self, model: str, base_url: Optional[str] = None,
                 provider: str = "openai", **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()
        self._client = None

    def _get_client(self):
        """Lazy-init the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            client_kwargs = {}

            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            elif self.provider in PROVIDER_CONFIG:
                url, key_env = PROVIDER_CONFIG[self.provider]
                client_kwargs["base_url"] = url
                if key_env:
                    api_key = os.environ.get(key_env, "")
                    if api_key:
                        client_kwargs["api_key"] = api_key
                else:
                    # Ollama doesn't need a real key
                    client_kwargs["api_key"] = "ollama"

            # Allow override via kwargs
            if "api_key" in self.kwargs:
                client_kwargs["api_key"] = self.kwargs["api_key"]

            self._client = OpenAI(**client_kwargs)

        return self._client

    def chat(self, messages: List[LLMMessage],
             temperature: float = 0.0,
             max_tokens: int = 4096) -> LLMResponse:
        """Send chat completion via OpenAI-compatible API."""
        client = self._get_client()

        formatted = [{"role": m.role, "content": m.content} for m in messages]

        start = time.time()

        completion = client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency = (time.time() - start) * 1000

        choice = completion.choices[0]
        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=completion.model,
            provider=self.provider,
            usage=usage,
            latency_ms=round(latency, 1),
            raw=completion,
        )

    @property
    def provider_name(self) -> str:
        return self.provider
