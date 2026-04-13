"""
Base LLM Client
================
Abstract interface for all LLM provider implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str          # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    model: str = ""
    provider: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0
    raw: Any = None


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All provider implementations must implement:
    - chat(messages) -> LLMResponse

    Usage:
        client = create_llm_client("openai", "gpt-4o")
        response = client.chat([
            LLMMessage("system", "You are a financial analyst."),
            LLMMessage("user", "Analyze AAPL"),
        ])
        print(response.content)
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def chat(self, messages: List[LLMMessage],
             temperature: float = 0.0,
             max_tokens: int = 4096) -> LLMResponse:
        """Send a chat completion request."""
        pass

    def chat_simple(self, system: str, user: str,
                    temperature: float = 0.0,
                    max_tokens: int = 4096) -> str:
        """Shortcut: send system + user message, return content string."""
        response = self.chat(
            [LLMMessage("system", system), LLMMessage("user", user)],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.removesuffix("Client").lower()
