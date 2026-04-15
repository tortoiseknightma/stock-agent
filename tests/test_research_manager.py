"""Tests for ResearchManagerAgent."""

import json
import pytest
from unittest.mock import MagicMock

from agents.state import AgentState, DebateState
from agents.managers.research_manager import ResearchManagerAgent


def make_state_with_debate(ticker="AAPL") -> AgentState:
    state = AgentState(ticker=ticker, trade_date="2024-01-15")
    state.technical_report = "Technical: bullish"
    state.investment_debate_state.rounds = [
        {"role": "bull", "content": "Revenue growing 20% YoY"},
        {"role": "bear", "content": "Valuation at 30x earnings is stretched"},
    ]
    return state


class TestResearchManager:
    def test_writes_investment_plan(self):
        llm = MagicMock()
        llm.chat_simple.return_value = json.dumps({
            "signal": "BUY",
            "conviction": 0.72,
            "reasoning": "Bull case prevails — strong revenue growth justifies valuation",
            "key_factors": ["revenue growth", "competitive moat"],
        })
        agent = ResearchManagerAgent(llm)
        state = make_state_with_debate()
        state = agent.run(state)

        assert state.investment_plan != ""
        assert len(state.agent_log) == 1

    def test_parses_json_verdict(self):
        verdict = {"signal": "BUY", "conviction": 0.75, "reasoning": "bull wins", "key_factors": []}
        llm = MagicMock()
        llm.chat_simple.return_value = json.dumps(verdict)
        agent = ResearchManagerAgent(llm)
        state = make_state_with_debate()
        state = agent.run(state)

        plan = json.loads(state.investment_plan)
        assert plan["signal"] == "BUY"
        assert plan["conviction"] == pytest.approx(0.75)

    def test_parses_json_in_code_fence(self):
        verdict = {"signal": "HOLD", "conviction": 0.45, "reasoning": "mixed signals"}
        raw = f"```json\n{json.dumps(verdict)}\n```"
        llm = MagicMock()
        llm.chat_simple.return_value = raw
        agent = ResearchManagerAgent(llm)
        state = make_state_with_debate()
        state = agent.run(state)

        plan = json.loads(state.investment_plan)
        assert plan["signal"] == "HOLD"

    def test_fallback_without_llm(self):
        agent = ResearchManagerAgent(None)
        state = make_state_with_debate()
        state = agent.run(state)
        assert state.investment_plan != ""
        assert "unavailable" in state.investment_plan.lower() or "fallback" in state.investment_plan.lower()

    def test_fallback_uses_composite_signal_when_available(self):
        agent = ResearchManagerAgent(None)
        state = make_state_with_debate()
        composite = MagicMock()
        composite.signal = "buy"
        composite.composite_score = 0.6
        state.composite_signal = composite
        state = agent.run(state)
        assert "buy" in state.investment_plan.lower() or "composite" in state.investment_plan.lower()

    def test_verdict_stored_in_debate_state(self):
        llm = MagicMock()
        llm.chat_simple.return_value = '{"signal": "SELL", "conviction": 0.65, "reasoning": "bear wins"}'
        agent = ResearchManagerAgent(llm)
        state = make_state_with_debate()
        state = agent.run(state)
        assert state.investment_debate_state.verdict != ""

    def test_prompt_includes_debate_rounds(self):
        llm = MagicMock()
        llm.chat_simple.return_value = '{"signal": "HOLD", "conviction": 0.5, "reasoning": "mixed"}'
        agent = ResearchManagerAgent(llm)
        state = make_state_with_debate()
        agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "BULL" in call_args or "bull" in call_args
        assert "BEAR" in call_args or "bear" in call_args
