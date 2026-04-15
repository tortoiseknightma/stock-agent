"""Tests for ReflectionEngine — LLM-driven post-trade lesson extraction."""

import json
import time
import pytest
from unittest.mock import MagicMock, call
from analysis.llm.reflection_engine import ReflectionEngine, ReflectionLesson


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_llm(responses):
    """Mock BaseLLMClient whose chat_simple returns successive strings."""
    client = MagicMock()
    client.chat_simple.side_effect = list(responses)
    return client


def make_lesson_json(lessons=None):
    """Build a valid JSON lesson array string."""
    if lessons is None:
        lessons = [
            {"lesson": "Cut losses faster when momentum breaks.", "importance": 7,
             "tags": ["stop_loss", "momentum"], "confidence": 0.8},
        ]
    return json.dumps(lessons)


def make_trade(id=1, ticker="AAPL", side="sell", price=180.0, quantity=50,
               decision_id=None, timestamp=None, realized_pnl=None):
    return {
        "id": id,
        "ticker": ticker,
        "side": side,
        "price": price,
        "quantity": quantity,
        "decision_id": decision_id,
        "timestamp": timestamp or time.time(),
        "realized_pnl": realized_pnl,
        "order_type": "market",
        "commission": 1.0,
        "slippage": 0.0,
        "portfolio_value_before": 100000.0,
        "portfolio_value_after": 109000.0,
        "current_price": None,
        "unrealized_pnl": None,
        "closed_at": None,
    }


def make_buy_trade(**kwargs):
    defaults = dict(id=2, ticker="AAPL", side="buy", price=160.0, quantity=50,
                    timestamp=time.time() - 7 * 86400)
    defaults.update(kwargs)
    return make_trade(**defaults)


def make_decision(id=1):
    return {
        "id": id,
        "ticker": "AAPL",
        "signal": "buy",
        "confidence": 0.72,
        "analysis": {
            "technical_score": 0.6,
            "fundamental_score": 0.5,
            "sentiment_score": 0.3,
            "reasoning": "Strong technical breakout with positive sentiment.",
        },
        "intended_action": "buy 50 shares",
        "portfolio_value": 100000.0,
        "current_position": 0,
        "current_price": 160.0,
        "executed": True,
        "execution_reason": "Within risk limits",
        "risk_assessment": "",
        "timestamp": time.time() - 7 * 86400,
    }


def make_engine(llm_responses=None, existing_memories=None):
    """Return (engine, mock_memory, mock_journal)."""
    llm = make_llm(llm_responses or [make_lesson_json()])

    memory = MagicMock()
    memory.search.return_value = existing_memories or []
    memory.add_lesson.return_value = 42

    journal = MagicMock()
    sell = make_trade()
    buy = make_buy_trade()
    journal.get_trade_pair.return_value = {"sell": sell, "buy": buy}
    journal.get_decision_by_id.return_value = None
    journal.get_recent_sell_trades.return_value = [sell]

    engine = ReflectionEngine(llm, memory, journal, max_lessons_per_trade=3)
    return engine, memory, journal


# ---------------------------------------------------------------------------
# ReflectionLesson dataclass
# ---------------------------------------------------------------------------

class TestReflectionLesson:
    def test_defaults(self):
        lesson = ReflectionLesson(content="Test lesson.", importance=5)
        assert lesson.tags == []
        assert lesson.trade_ticker == ""
        assert lesson.confidence == 0.5
        assert lesson.memory_id is None

    def test_fields(self):
        lesson = ReflectionLesson(
            content="Buy dips in strong uptrends.",
            importance=7,
            tags=["AAPL", "uptrend"],
            trade_ticker="AAPL",
            trade_pnl=500.0,
            confidence=0.85,
        )
        assert lesson.importance == 7
        assert "AAPL" in lesson.tags
        assert lesson.trade_pnl == 500.0


# ---------------------------------------------------------------------------
# reflect_on_trade — winning trade
# ---------------------------------------------------------------------------

class TestReflectOnTrade:
    def test_winning_trade_extracts_lesson(self):
        """A profitable sell should produce stored lessons."""
        engine, memory, journal = make_engine()
        journal.get_trade_pair.return_value = {
            "sell": make_trade(price=200.0),
            "buy": make_buy_trade(price=160.0),
        }

        lessons = engine.reflect_on_trade(1)

        assert len(lessons) == 1
        assert lessons[0].memory_id == 42
        memory.add_lesson.assert_called_once()

    def test_losing_trade_extracts_lesson(self):
        """A losing sell should also produce lessons (risk management focus)."""
        lesson_json = make_lesson_json([
            {"lesson": "Exit sooner when stop-loss triggers are ignored.",
             "importance": 9, "tags": ["stop_loss"], "confidence": 0.9},
        ])
        engine, memory, journal = make_engine(llm_responses=[lesson_json])
        journal.get_trade_pair.return_value = {
            "sell": make_trade(price=130.0),
            "buy": make_buy_trade(price=160.0),
        }

        lessons = engine.reflect_on_trade(1)

        assert len(lessons) == 1
        assert lessons[0].importance == 9

    def test_includes_decision_context_in_prompt(self):
        """When a decision record exists, its reasoning should appear in the LLM prompt."""
        engine, memory, journal = make_engine()
        sell_with_decision = make_trade(decision_id=5)
        journal.get_trade_pair.return_value = {
            "sell": sell_with_decision,
            "buy": make_buy_trade(),
        }
        journal.get_decision_by_id.return_value = make_decision()

        engine.reflect_on_trade(1)

        prompt = engine._client.chat_simple.call_args[1]["user"]
        assert "Decision context at entry" in prompt

    def test_no_decision_context_when_missing(self):
        """When no decision record, prompt should still work without it."""
        engine, memory, journal = make_engine()
        journal.get_decision_by_id.return_value = None

        lessons = engine.reflect_on_trade(1)
        # Should still work — no decision context, but lesson extracted
        assert isinstance(lessons, list)

    def test_returns_empty_when_no_trade_pair(self):
        """When trade pair lookup returns None, reflect_on_trade returns []."""
        engine, memory, journal = make_engine()
        journal.get_trade_pair.return_value = None

        lessons = engine.reflect_on_trade(99)

        assert lessons == []
        memory.add_lesson.assert_not_called()

    def test_ticker_added_to_tags(self):
        """The ticker should be prepended to tags when not already present."""
        engine, memory, journal = make_engine()
        journal.get_trade_pair.return_value = {
            "sell": make_trade(ticker="NVDA", price=500.0),
            "buy": make_buy_trade(ticker="NVDA", price=400.0),
        }

        lessons = engine.reflect_on_trade(1)

        stored_tags = memory.add_lesson.call_args[1]["tags"]
        assert stored_tags[0] == "NVDA"


# ---------------------------------------------------------------------------
# reflect_batch
# ---------------------------------------------------------------------------

class TestReflectBatch:
    def test_batch_processes_single_trade(self):
        """A single sell trade in window should run per-trade reflection."""
        engine, memory, journal = make_engine()

        lessons = engine.reflect_batch(days=1)

        assert len(lessons) >= 1
        journal.get_recent_sell_trades.assert_called_with(days=1)

    def test_batch_with_two_trades_adds_portfolio_reflection(self):
        """With ≥2 trades, a portfolio-pattern LLM call should fire."""
        sell1 = make_trade(id=1, ticker="AAPL", price=200.0)
        sell2 = make_trade(id=2, ticker="MSFT", price=350.0)

        pattern_lesson = make_lesson_json([
            {"lesson": "Winners are being cut too early across positions.",
             "importance": 8, "tags": ["portfolio", "exits"], "confidence": 0.75},
        ])
        per_trade_lesson = make_lesson_json()

        llm = make_llm([per_trade_lesson, per_trade_lesson, pattern_lesson])
        memory = MagicMock()
        memory.search.return_value = []
        memory.add_lesson.return_value = 10
        journal = MagicMock()
        journal.get_recent_sell_trades.return_value = [sell1, sell2]
        journal.get_trade_pair.return_value = {
            "sell": sell1, "buy": make_buy_trade()
        }
        journal.get_decision_by_id.return_value = None

        engine = ReflectionEngine(llm, memory, journal)
        lessons = engine.reflect_batch(days=1)

        # Per-trade (2) + portfolio (1) = 3 LLM calls total
        assert llm.chat_simple.call_count == 3
        assert len(lessons) >= 1

    def test_batch_returns_empty_when_no_sells(self):
        """No sell trades → no lessons, no LLM calls."""
        engine, memory, journal = make_engine()
        journal.get_recent_sell_trades.return_value = []

        lessons = engine.reflect_batch(days=1)

        assert lessons == []
        engine._client.chat_simple.assert_not_called()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_lesson_not_stored(self):
        """A lesson with ≥80% word overlap with an existing memory is skipped."""
        existing = MagicMock()
        existing.content = "Cut losses faster when momentum breaks down."

        engine, memory, journal = make_engine(existing_memories=[existing])

        # This lesson is ~80% overlap with the existing one
        dup_json = make_lesson_json([
            {"lesson": "Cut losses faster when momentum breaks.",
             "importance": 7, "tags": [], "confidence": 0.8},
        ])
        engine._client.chat_simple.side_effect = [dup_json]

        lessons = engine.reflect_on_trade(1)

        assert lessons == []
        memory.add_lesson.assert_not_called()

    def test_unique_lesson_is_stored(self):
        """A lesson with no overlap to existing memories is stored."""
        engine, memory, journal = make_engine(existing_memories=[])
        lessons = engine.reflect_on_trade(1)
        assert len(lessons) == 1
        memory.add_lesson.assert_called_once()


# ---------------------------------------------------------------------------
# LLM failure handling
# ---------------------------------------------------------------------------

class TestGracefulFailures:
    def test_llm_exception_returns_empty_list(self):
        """If the LLM raises an exception, reflect_on_trade returns [] without crashing."""
        engine, memory, journal = make_engine()
        engine._client.chat_simple.side_effect = RuntimeError("API timeout")

        lessons = engine.reflect_on_trade(1)

        assert lessons == []
        memory.add_lesson.assert_not_called()

    def test_malformed_json_returns_empty_list(self):
        """If the LLM returns non-JSON, no lessons are stored."""
        engine, memory, journal = make_engine(
            llm_responses=["This is not JSON at all, just prose."]
        )

        lessons = engine.reflect_on_trade(1)

        assert lessons == []
        memory.add_lesson.assert_not_called()

    def test_empty_json_array_returns_empty_list(self):
        """LLM returning [] is valid — no lessons to store."""
        engine, memory, journal = make_engine(llm_responses=["[]"])

        lessons = engine.reflect_on_trade(1)

        assert lessons == []

    def test_partial_json_lesson_skipped(self):
        """A JSON item missing 'lesson' key is skipped."""
        bad_json = json.dumps([{"importance": 5, "tags": ["x"], "confidence": 0.5}])
        engine, memory, journal = make_engine(llm_responses=[bad_json])

        lessons = engine.reflect_on_trade(1)

        assert lessons == []

    def test_batch_llm_failure_returns_empty_list(self):
        """If LLM raises on every call during batch, reflect_batch returns []."""
        engine, memory, journal = make_engine()
        engine._client.chat_simple.side_effect = RuntimeError("quota exceeded")

        lessons = engine.reflect_batch(days=1)

        assert lessons == []

    def test_json_in_markdown_fences_is_parsed(self):
        """LLM wrapping JSON in ```json ... ``` should still be parsed."""
        fenced = "```json\n" + make_lesson_json() + "\n```"
        engine, memory, journal = make_engine(llm_responses=[fenced])

        lessons = engine.reflect_on_trade(1)

        assert len(lessons) == 1


# ---------------------------------------------------------------------------
# Trade pair lookup (journal integration)
# ---------------------------------------------------------------------------

class TestTradePairLookup:
    def test_get_trade_pair_called_with_sell_id(self):
        engine, memory, journal = make_engine()
        engine.reflect_on_trade(sell_trade_id=7)
        journal.get_trade_pair.assert_called_once_with(7)

    def test_pnl_computed_from_buy_sell_prices(self):
        """Lesson content prompt should reflect correct P&L computation."""
        engine, memory, journal = make_engine()
        journal.get_trade_pair.return_value = {
            "sell": make_trade(price=200.0, quantity=100),
            "buy": make_buy_trade(price=150.0, quantity=100),
        }

        engine.reflect_on_trade(1)

        prompt = engine._client.chat_simple.call_args[1]["user"]
        # P&L = (200 - 150) * 100 = +5000
        assert "+5000" in prompt or "+$5000" in prompt.replace(",", "")

    def test_hold_duration_in_prompt(self):
        """Hold duration (days) should appear in the LLM prompt."""
        now = time.time()
        engine, memory, journal = make_engine()
        journal.get_trade_pair.return_value = {
            "sell": make_trade(price=180.0, timestamp=now),
            "buy": make_buy_trade(price=160.0, timestamp=now - 5 * 86400),
        }

        engine.reflect_on_trade(1)

        prompt = engine._client.chat_simple.call_args[1]["user"]
        assert "5 day" in prompt


# ---------------------------------------------------------------------------
# Max lessons cap
# ---------------------------------------------------------------------------

class TestMaxLessonsCap:
    def test_max_lessons_per_trade_respected(self):
        """Engine should not store more than max_lessons_per_trade."""
        many_lessons = make_lesson_json([
            {"lesson": f"Lesson {i}.", "importance": 5, "tags": [], "confidence": 0.7}
            for i in range(10)
        ])
        engine, memory, journal = make_engine(llm_responses=[many_lessons])
        engine._max_lessons = 2

        lessons = engine.reflect_on_trade(1)

        assert len(lessons) <= 2
