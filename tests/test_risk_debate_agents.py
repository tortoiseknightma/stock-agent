"""Tests for AggressiveDebatorAgent, ConservativeDebatorAgent, NeutralDebatorAgent."""

import pytest
from unittest.mock import MagicMock

from agents.state import AgentState, DebateState
from agents.risk.aggressive import AggressiveDebatorAgent
from agents.risk.conservative import ConservativeDebatorAgent
from agents.risk.neutral import NeutralDebatorAgent


def make_state(ticker="AAPL") -> AgentState:
    state = AgentState(ticker=ticker, trade_date="2024-01-15")
    state.trade_proposal = "BUY 100 shares of AAPL at market price"
    state.investment_plan = '{"signal": "BUY", "conviction": 0.75}'
    return state


class TestAggressiveDebatorAgent:
    def test_appends_round_with_correct_role(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Strong upside potential justifies aggressive sizing."
        agent = AggressiveDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 1
        assert state.risk_debate_state.rounds[0]["role"] == "aggressive"
        assert "upside" in state.risk_debate_state.rounds[0]["content"].lower()

    def test_increments_count(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Go big."
        agent = AggressiveDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.risk_debate_state.count == 1
        assert state.risk_debate_state.last_speaker == "aggressive"

    def test_fallback_without_llm(self):
        agent = AggressiveDebatorAgent(None)
        state = make_state()
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 1
        assert state.risk_debate_state.rounds[0]["role"] == "aggressive"
        assert state.risk_debate_state.rounds[0]["content"] != ""

    def test_includes_prior_arguments_in_prompt(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Aggressive rebuttal."
        agent = AggressiveDebatorAgent(llm)
        state = make_state()
        # Add a prior conservative round
        state.risk_debate_state.rounds.append({
            "role": "conservative",
            "content": "Risk is too high, position limits apply.",
        })
        state = agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "CONSERVATIVE" in call_args or "conservative" in call_args.lower()

    def test_logged_to_agent_log(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Aggressive case."
        agent = AggressiveDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert len(state.agent_log) == 1
        assert state.agent_log[0]["agent"] == "AggressiveDebator"


class TestConservativeDebatorAgent:
    def test_appends_round_with_correct_role(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Capital preservation requires caution."
        agent = ConservativeDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 1
        assert state.risk_debate_state.rounds[0]["role"] == "conservative"

    def test_increments_count_and_last_speaker(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Reduce position size."
        agent = ConservativeDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.risk_debate_state.count == 1
        assert state.risk_debate_state.last_speaker == "conservative"

    def test_fallback_without_llm(self):
        agent = ConservativeDebatorAgent(None)
        state = make_state()
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 1
        assert state.risk_debate_state.rounds[0]["role"] == "conservative"

    def test_includes_investment_plan_in_prompt(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Conservative view."
        agent = ConservativeDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "Investment Plan" in call_args or "investment_plan" in call_args.lower() or "BUY" in call_args

    def test_multiple_rounds_accumulate(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Stay cautious."
        agent = ConservativeDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 2
        assert state.risk_debate_state.count == 2


class TestNeutralDebatorAgent:
    def test_appends_round_with_correct_role(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Bull scenario 60%, bear scenario 40%."
        agent = NeutralDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 1
        assert state.risk_debate_state.rounds[0]["role"] == "neutral"

    def test_increments_count_and_last_speaker(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Balanced analysis."
        agent = NeutralDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert state.risk_debate_state.count == 1
        assert state.risk_debate_state.last_speaker == "neutral"

    def test_fallback_without_llm(self):
        agent = NeutralDebatorAgent(None)
        state = make_state()
        state = agent.run(state)

        assert len(state.risk_debate_state.rounds) == 1
        assert state.risk_debate_state.rounds[0]["role"] == "neutral"
        assert state.risk_debate_state.rounds[0]["content"] != ""

    def test_includes_prior_non_neutral_arguments(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Balanced view."
        agent = NeutralDebatorAgent(llm)
        state = make_state()
        state.risk_debate_state.rounds.append({
            "role": "aggressive",
            "content": "This is a high-upside opportunity.",
        })
        state.risk_debate_state.rounds.append({
            "role": "conservative",
            "content": "Downside risk is significant.",
        })
        state = agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "AGGRESSIVE" in call_args or "aggressive" in call_args.lower()

    def test_logged_to_agent_log(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Neutral analysis."
        agent = NeutralDebatorAgent(llm)
        state = make_state()
        state = agent.run(state)

        assert len(state.agent_log) == 1
        assert state.agent_log[0]["agent"] == "NeutralDebator"
