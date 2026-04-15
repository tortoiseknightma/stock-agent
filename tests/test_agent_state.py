"""Tests for agents.state — AgentState and DebateState."""

import time
import pytest
from agents.state import AgentState, DebateState


class TestDebateState:
    def test_defaults(self):
        ds = DebateState()
        assert ds.rounds == []
        assert ds.count == 0
        assert ds.last_speaker == ""
        assert ds.verdict == ""

    def test_append_round(self):
        ds = DebateState()
        ds.rounds.append({"role": "bull", "content": "argument"})
        ds.count += 1
        ds.last_speaker = "bull"
        assert len(ds.rounds) == 1
        assert ds.count == 1


class TestAgentState:
    def test_defaults(self):
        state = AgentState(ticker="AAPL", trade_date="2024-01-15")
        assert state.ticker == "AAPL"
        assert state.trade_date == "2024-01-15"
        assert state.decision_signal == "HOLD"
        assert state.decision_conviction == 0.5
        assert state.technical_signal is None
        assert state.technical_report == ""
        assert state.agent_log == []

    def test_log(self):
        state = AgentState(ticker="AAPL", trade_date="2024-01-15")
        before = time.time()
        state.log("TestAgent", "analyze", "did something")
        after = time.time()

        assert len(state.agent_log) == 1
        entry = state.agent_log[0]
        assert entry["agent"] == "TestAgent"
        assert entry["action"] == "analyze"
        assert entry["content"] == "did something"
        assert before <= entry["timestamp"] <= after

    def test_multiple_logs(self):
        state = AgentState()
        state.log("A", "x", "1")
        state.log("B", "y", "2")
        assert len(state.agent_log) == 2
        assert state.agent_log[0]["agent"] == "A"
        assert state.agent_log[1]["agent"] == "B"

    def test_debate_states_independent(self):
        state = AgentState()
        state.investment_debate_state.rounds.append({"role": "bull", "content": "buy!"})
        state.risk_debate_state.rounds.append({"role": "aggressive", "content": "go big!"})
        assert len(state.investment_debate_state.rounds) == 1
        assert len(state.risk_debate_state.rounds) == 1

    def test_signals_default_to_none(self):
        state = AgentState()
        assert state.technical_signal is None
        assert state.fundamental_signal is None
        assert state.sentiment_signal is None
        assert state.composite_signal is None
