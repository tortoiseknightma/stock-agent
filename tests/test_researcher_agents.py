"""Tests for Bull and Bear researcher agents."""

import pytest
from unittest.mock import MagicMock

from agents.state import AgentState, DebateState
from agents.researchers.bull import BullResearcherAgent
from agents.researchers.bear import BearResearcherAgent


def make_state_with_reports(ticker="AAPL") -> AgentState:
    state = AgentState(ticker=ticker, trade_date="2024-01-15")
    state.technical_report = "Technical: bullish trend, RSI=55"
    state.fundamental_report = "Fundamental: P/E=25, strong ROE"
    state.sentiment_report = "Sentiment: positive news coverage"
    state.macro_report = "Macro: stable rates environment"
    return state


class TestBullResearcher:
    def test_appends_to_rounds(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Strong bull case: 1) Revenue growth..."
        agent = BullResearcherAgent(llm)
        state = make_state_with_reports()
        state = agent.run(state)

        assert len(state.investment_debate_state.rounds) == 1
        assert state.investment_debate_state.rounds[0]["role"] == "bull"
        assert state.investment_debate_state.last_speaker == "bull"
        assert state.investment_debate_state.count == 1

    def test_logs_entry(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Bull argument"
        agent = BullResearcherAgent(llm)
        state = make_state_with_reports()
        state = agent.run(state)
        assert any(e["agent"] == "BullResearcher" for e in state.agent_log)

    def test_fallback_without_llm(self):
        agent = BullResearcherAgent(None)
        state = make_state_with_reports()
        state = agent.run(state)
        assert len(state.investment_debate_state.rounds) == 1
        assert "no LLM" in state.investment_debate_state.rounds[0]["content"]

    def test_prompt_includes_analyst_reports(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "bull"
        agent = BullResearcherAgent(llm)
        state = make_state_with_reports()
        agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "Technical" in call_args
        assert "Fundamental" in call_args

    def test_prompt_includes_bear_counter_when_available(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "rebuttal"
        agent = BullResearcherAgent(llm)
        state = make_state_with_reports()
        state.investment_debate_state.rounds.append(
            {"role": "bear", "content": "Bear counter argument"}
        )
        agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "Bear" in call_args

    def test_multiple_rounds_accumulate(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "argument"
        agent = BullResearcherAgent(llm)
        state = make_state_with_reports()
        state = agent.run(state)
        state = agent.run(state)
        bull_rounds = [r for r in state.investment_debate_state.rounds if r["role"] == "bull"]
        assert len(bull_rounds) == 2


class TestBearResearcher:
    def test_appends_to_rounds(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Bear case: valuation stretched..."
        agent = BearResearcherAgent(llm)
        state = make_state_with_reports()
        state = agent.run(state)

        assert len(state.investment_debate_state.rounds) == 1
        assert state.investment_debate_state.rounds[0]["role"] == "bear"
        assert state.investment_debate_state.last_speaker == "bear"

    def test_fallback_without_llm(self):
        agent = BearResearcherAgent(None)
        state = make_state_with_reports()
        state = agent.run(state)
        assert "no LLM" in state.investment_debate_state.rounds[0]["content"]

    def test_prompt_includes_bull_arguments(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "bear"
        agent = BearResearcherAgent(llm)
        state = make_state_with_reports()
        state.investment_debate_state.rounds.append(
            {"role": "bull", "content": "Bull opening argument"}
        )
        agent.run(state)

        call_args = llm.chat_simple.call_args[1]["user"]
        assert "Bull" in call_args

    def test_alternating_debate_order(self):
        bull_llm = MagicMock()
        bull_llm.chat_simple.return_value = "bull arg"
        bear_llm = MagicMock()
        bear_llm.chat_simple.return_value = "bear arg"

        bull = BullResearcherAgent(bull_llm)
        bear = BearResearcherAgent(bear_llm)
        state = make_state_with_reports()

        state = bull.run(state)
        state = bear.run(state)

        roles = [r["role"] for r in state.investment_debate_state.rounds]
        assert roles == ["bull", "bear"]
