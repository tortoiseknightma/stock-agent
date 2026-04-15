"""Tests for agents.base — BaseAgent ABC."""

import pytest
from unittest.mock import MagicMock
from agents.base import BaseAgent
from agents.state import AgentState


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing."""
    def run(self, state: AgentState) -> AgentState:
        result = self._call_llm("test prompt")
        state.technical_report = result or "no llm"
        return state


class TestBaseAgent:
    def test_call_llm_returns_response(self):
        client = MagicMock()
        client.chat_simple.return_value = "LLM response"
        agent = ConcreteAgent("Test", llm_client=client, system_prompt="sys")
        result = agent._call_llm("hello")
        assert result == "LLM response"
        client.chat_simple.assert_called_once_with(
            system="sys", user="hello", temperature=0.0, max_tokens=1024
        )

    def test_call_llm_none_client_returns_none(self):
        agent = ConcreteAgent("Test", llm_client=None)
        assert agent._call_llm("hello") is None

    def test_call_llm_exception_returns_none(self):
        client = MagicMock()
        client.chat_simple.side_effect = RuntimeError("API error")
        agent = ConcreteAgent("Test", llm_client=client)
        assert agent._call_llm("hello") is None

    def test_call_llm_passes_temperature_and_max_tokens(self):
        client = MagicMock()
        client.chat_simple.return_value = "ok"
        agent = ConcreteAgent("Test", llm_client=client, system_prompt="sys")
        agent._call_llm("hello", max_tokens=500, temperature=0.7)
        client.chat_simple.assert_called_once_with(
            system="sys", user="hello", temperature=0.7, max_tokens=500
        )

    def test_run_with_llm(self):
        client = MagicMock()
        client.chat_simple.return_value = "analysis result"
        agent = ConcreteAgent("Test", llm_client=client)
        state = AgentState(ticker="AAPL")
        state = agent.run(state)
        assert state.technical_report == "analysis result"

    def test_run_without_llm(self):
        agent = ConcreteAgent("Test", llm_client=None)
        state = AgentState(ticker="AAPL")
        state = agent.run(state)
        assert state.technical_report == "no llm"

    def test_name_stored(self):
        agent = ConcreteAgent("MyAgent")
        assert agent.name == "MyAgent"
