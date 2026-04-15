"""Tests for AgentReflector."""

import pytest
from unittest.mock import MagicMock

from agents.state import AgentState, DebateState
from agents.memory.reflector import AgentReflector


def make_state_with_debates() -> AgentState:
    state = AgentState(ticker="AAPL", trade_date="2024-01-15")
    state.investment_debate_state.rounds = [
        {"role": "bull", "content": "Strong earnings growth supports BUY."},
        {"role": "bear", "content": "Valuation looks stretched at 30x PE."},
    ]
    state.risk_debate_state.rounds = [
        {"role": "aggressive", "content": "Momentum justifies full position."},
        {"role": "conservative", "content": "Macro risk too high, limit to 2%."},
        {"role": "neutral", "content": "60% upside scenario, 40% flat."},
    ]
    return state


class TestAgentReflector:
    def test_reflect_stores_lessons_with_llm(self, tmp_path):
        llm = MagicMock()
        llm.chat_simple.return_value = "Strong earnings alone don't justify ignoring valuation."
        state = make_state_with_debates()
        reflector = AgentReflector(
            llm_client=llm,
            memory_db=str(tmp_path / "mem.db"),
        )
        count = reflector.reflect("AAPL", state, outcome="Lost 3% in 2 weeks")
        assert count == 5  # all 5 roles stored a lesson

    def test_reflect_without_llm_stores_nothing(self, tmp_path):
        state = make_state_with_debates()
        reflector = AgentReflector(
            llm_client=None,
            memory_db=str(tmp_path / "mem.db"),
        )
        count = reflector.reflect("AAPL", state, outcome="Lost 3%")
        assert count == 0

    def test_reflect_skips_roles_with_no_arguments(self, tmp_path):
        llm = MagicMock()
        llm.chat_simple.return_value = "Lesson learned."
        state = AgentState(ticker="TSLA", trade_date="2024-01-15")
        # Only bull has an argument
        state.investment_debate_state.rounds = [
            {"role": "bull", "content": "TSLA autonomy potential."},
        ]
        reflector = AgentReflector(
            llm_client=llm,
            memory_db=str(tmp_path / "mem.db"),
        )
        count = reflector.reflect("TSLA", state, outcome="Flat")
        assert count == 1  # only bull stored

    def test_reflect_handles_llm_exception(self, tmp_path):
        llm = MagicMock()
        llm.chat_simple.side_effect = RuntimeError("LLM unavailable")
        state = make_state_with_debates()
        reflector = AgentReflector(
            llm_client=llm,
            memory_db=str(tmp_path / "mem.db"),
        )
        # Should not raise, just store 0
        count = reflector.reflect("AAPL", state, outcome="Gained 5%")
        assert count == 0

    def test_lessons_are_retrievable_after_reflect(self, tmp_path):
        from agents.memory.situation_memory import SituationMemory

        llm = MagicMock()
        llm.chat_simple.return_value = "Earnings beats don't override valuation risk."
        state = make_state_with_debates()
        db = str(tmp_path / "mem.db")
        reflector = AgentReflector(llm_client=llm, memory_db=db)
        reflector.reflect("AAPL", state, outcome="Lost 3%")

        bull_mem = SituationMemory(db_path=db, role="bull")
        lessons = bull_mem.retrieve("AAPL earnings valuation", top_k=3)
        assert len(lessons) >= 1
