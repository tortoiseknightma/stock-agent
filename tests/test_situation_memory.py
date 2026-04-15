"""Tests for SituationMemory."""

import pytest
import tempfile
import os

from agents.memory.situation_memory import SituationMemory


@pytest.fixture
def mem(tmp_path):
    db = str(tmp_path / "test_agent_memory.db")
    return SituationMemory(db_path=db, role="bull")


class TestSituationMemory:
    def test_store_and_count(self, mem):
        assert mem.count() == 0
        mem.store(
            situation="AAPL earnings beat",
            decision="argued strong upside",
            outcome="gained 5%",
            lesson="earnings beats with high margin expansion are reliable buy signals",
        )
        assert mem.count() == 1

    def test_retrieve_returns_relevant_lesson(self, mem):
        mem.store(
            situation="AAPL Q4 earnings strong",
            decision="pushed for full allocation",
            outcome="gained 6%",
            lesson="strong earnings with margin expansion justify larger position sizes",
        )
        lessons = mem.retrieve("AAPL earnings margin")
        assert len(lessons) >= 1
        assert "margin" in lessons[0].lower() or "earnings" in lessons[0].lower()

    def test_retrieve_empty_query_returns_empty(self, mem):
        mem.store(
            situation="TSLA momentum",
            decision="bullish",
            outcome="flat",
            lesson="momentum alone is insufficient",
        )
        lessons = mem.retrieve("")
        assert lessons == []

    def test_retrieve_respects_top_k(self, mem):
        for i in range(5):
            mem.store(
                situation=f"AAPL scenario {i}",
                decision=f"bull argument {i}",
                outcome=f"outcome {i}",
                lesson=f"AAPL lesson number {i}",
            )
        lessons = mem.retrieve("AAPL lesson", top_k=2)
        assert len(lessons) <= 2

    def test_role_isolation(self, tmp_path):
        db = str(tmp_path / "shared.db")
        bull_mem = SituationMemory(db_path=db, role="bull")
        bear_mem = SituationMemory(db_path=db, role="bear")

        bull_mem.store("AAPL rally", "buy signal strong", "profit", "bull lesson here")
        bear_mem.store("AAPL rally", "overvalued", "pullback", "bear lesson here")

        bull_lessons = bull_mem.retrieve("AAPL rally", top_k=5)
        bear_lessons = bear_mem.retrieve("AAPL rally", top_k=5)

        assert any("bull" in l.lower() for l in bull_lessons)
        assert any("bear" in l.lower() for l in bear_lessons)
        # Each role only sees its own lessons
        assert not any("bear" in l.lower() for l in bull_lessons)
        assert not any("bull" in l.lower() for l in bear_lessons)

    def test_special_chars_in_query_dont_crash(self, mem):
        mem.store("AAPL", "buy", "profit", "lesson about AAPL")
        # Should not raise even with FTS5 special characters
        lessons = mem.retrieve('AAPL "earnings" (beat) -risk')
        # Just verify it returns without error
        assert isinstance(lessons, list)

    def test_multiple_stores_accumulate(self, mem):
        for i in range(3):
            mem.store(f"sit {i}", f"dec {i}", f"out {i}", f"lesson {i}")
        assert mem.count() == 3
