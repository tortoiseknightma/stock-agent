"""
Base Agent
===========
Abstract base class for all multi-agent system agents.
Each agent reads from and writes to AgentState via the run() method.
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.llm.base import BaseLLMClient
from agents.state import AgentState


class BaseAgent(ABC):
    """
    Abstract base for all agents in the multi-agent pipeline.

    Subclasses must implement ``run(state) -> state``.
    LLM calls go through ``_call_llm()`` which handles graceful degradation.
    """

    def __init__(
        self,
        name: str,
        llm_client: Optional[BaseLLMClient] = None,
        system_prompt: str = "",
    ):
        self.name = name
        self._llm = llm_client
        self._system_prompt = system_prompt

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Execute agent logic, mutate and return state."""

    def _call_llm(
        self,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Safe LLM call with graceful degradation.

        Returns the LLM response string, or None if the client is
        unavailable or the call fails.
        """
        if self._llm is None:
            return None
        try:
            return self._llm.chat_simple(
                system=self._system_prompt,
                user=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}")
            return None
