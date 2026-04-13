"""
StockAgent LLM Client Layer
=============================
Multi-provider LLM abstraction supporting:
  - OpenAI (GPT-4o, GPT-4.1, o3, etc.)
  - Anthropic (Claude Sonnet, Opus, etc.)
  - Google (Gemini 2.0, 2.5, etc.)
  - DeepSeek
  - Qwen (via DashScope)
  - GLM (via Zhipu)
  - xAI (Grok)
  - OpenRouter
  - Ollama (local models)

Usage:
    from core.llm import create_llm_client, LLMMessage

    client = create_llm_client("openai", "gpt-4o")

    # Simple usage
    answer = client.chat_simple(
        system="You are a financial analyst.",
        user="What is the fair value of AAPL?"
    )

    # Multi-turn
    response = client.chat([
        LLMMessage("system", "You are a financial analyst."),
        LLMMessage("user", "Analyze AAPL fundamentals."),
    ])
    print(response.content)
    print(f"Tokens: {response.usage}")
"""

from .base import BaseLLMClient, LLMMessage, LLMResponse
from .factory import create_llm_client

__all__ = ["BaseLLMClient", "LLMMessage", "LLMResponse", "create_llm_client"]
