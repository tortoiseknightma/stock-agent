"""Tests for TraderAgent."""

import pytest
from unittest.mock import MagicMock

from agents.state import AgentState
from agents.trader.trader import TraderAgent


def make_state(ticker="AAPL") -> AgentState:
    state = AgentState(ticker=ticker, trade_date="2024-01-15")
    state.investment_plan = '{"signal": "BUY", "conviction": 0.75, "reasoning": "strong growth"}'
    return state


class TestTraderAgent:
    def test_writes_trade_proposal(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "BUY 100 shares at market, 5% of portfolio"
        agent = TraderAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.trade_proposal == "BUY 100 shares at market, 5% of portfolio"
        assert len(state.agent_log) == 1

    def test_fallback_without_llm(self):
        agent = TraderAgent(None)
        state = make_state()
        composite = MagicMock()
        composite.signal = "buy"
        composite.composite_score = 0.6
        state.composite_signal = composite
        state = agent.run(state)

        assert "Rule-based" in state.trade_proposal
        assert "buy" in state.trade_proposal.lower()

    def test_fallback_no_composite(self):
        agent = TraderAgent(None)
        state = make_state()
        state = agent.run(state)

        assert state.trade_proposal != ""
        assert "No trade proposal" in state.trade_proposal

    def test_includes_broker_context_in_prompt(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "proposal"
        broker = MagicMock()
        account = MagicMock()
        account.net_liquidation = 100_000.0
        account.total_cash = 80_000.0
        broker.get_account_info.return_value = account
        broker.get_current_price.return_value = 180.0
        broker.get_position.return_value = None

        agent = TraderAgent(llm, broker=broker)
        state = make_state()
        agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "180" in call_args

    def test_prompt_includes_investment_plan(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "proposal"
        agent = TraderAgent(llm)
        state = make_state()
        agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "Investment Plan" in call_args
