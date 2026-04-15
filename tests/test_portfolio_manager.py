"""Tests for PortfolioManagerAgent."""

import json
import pytest
from unittest.mock import MagicMock

from agents.state import AgentState, DebateState
from agents.managers.portfolio_manager import PortfolioManagerAgent


def make_state(ticker="AAPL") -> AgentState:
    state = AgentState(ticker=ticker, trade_date="2024-01-15")
    state.investment_plan = '{"signal": "BUY", "conviction": 0.75}'
    state.trade_proposal = "BUY 100 shares at market"
    state.risk_debate_state.rounds = [
        {"role": "aggressive", "content": "Strong upside potential."},
        {"role": "conservative", "content": "Capital at risk."},
        {"role": "neutral", "content": "60% bull, 40% bear probability."},
    ]
    return state


class TestPortfolioManagerAgent:
    def test_writes_decision_signal_and_conviction(self):
        llm = MagicMock()
        llm.chat_simple.return_value = json.dumps({
            "signal": "BUY",
            "conviction": 0.82,
            "reasoning": "Strong fundamentals justify entry.",
        })
        agent = PortfolioManagerAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.decision_signal == "BUY"
        assert abs(state.decision_conviction - 0.82) < 0.001
        assert state.final_decision != ""

    def test_parses_code_fenced_json(self):
        llm = MagicMock()
        llm.chat_simple.return_value = (
            "Here is my decision:\n"
            "```json\n"
            '{"signal": "SELL", "conviction": 0.6, "reasoning": "Overvalued."}\n'
            "```"
        )
        agent = PortfolioManagerAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.decision_signal == "SELL"
        assert abs(state.decision_conviction - 0.6) < 0.001

    def test_keyword_fallback_for_malformed_json(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "After careful review, I recommend to HOLD this position."
        agent = PortfolioManagerAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.decision_signal == "HOLD"

    def test_fallback_without_llm_uses_composite(self):
        agent = PortfolioManagerAgent(None)
        state = make_state()
        composite = MagicMock()
        composite.signal = "strong_buy"
        composite.confidence = 0.9
        state.composite_signal = composite
        state = agent.run(state)

        assert state.decision_signal == "BUY"
        assert state.decision_conviction == 0.9

    def test_fallback_without_llm_no_composite_defaults_hold(self):
        agent = PortfolioManagerAgent(None)
        state = make_state()
        state = agent.run(state)

        assert state.decision_signal == "HOLD"
        assert state.decision_conviction == 0.5

    def test_signal_normalized_to_uppercase(self):
        llm = MagicMock()
        llm.chat_simple.return_value = json.dumps({
            "signal": "buy",
            "conviction": 0.7,
            "reasoning": "Good outlook.",
        })
        agent = PortfolioManagerAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.decision_signal == "BUY"

    def test_logged_to_agent_log(self):
        llm = MagicMock()
        llm.chat_simple.return_value = json.dumps({
            "signal": "HOLD",
            "conviction": 0.5,
            "reasoning": "Uncertain.",
        })
        agent = PortfolioManagerAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert len(state.agent_log) == 1
        assert state.agent_log[0]["agent"] == "PortfolioManager"

    def test_sell_signal_from_composite(self):
        agent = PortfolioManagerAgent(None)
        state = make_state()
        composite = MagicMock()
        composite.signal = "strong_sell"
        composite.confidence = 0.85
        state.composite_signal = composite
        state = agent.run(state)

        assert state.decision_signal == "SELL"
