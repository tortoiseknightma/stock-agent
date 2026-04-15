"""
LLM Reflection Engine
======================
Analyzes completed trades and extracts actionable lessons that are stored in
long-term memory.  This closes the feedback loop: the agent not only makes
decisions but learns from their outcomes.

Flow (per trade)
----------------
1. A sell trade completes (real-time) or postmarket batch triggers reflection.
2. ``ReflectionEngine`` fetches the sell trade + matching buy trade from the
   journal to compute hold duration, entry/exit prices, and realized P&L.
3. The original decision record (if available) is fetched to provide the
   reasoning context that existed *at the time of entry*.
4. A single LLM call produces 1–3 actionable lessons as JSON.
5. Each lesson is deduplicated against existing LESSON memories before storage.

Batch flow
----------
``reflect_batch()`` aggregates all sell trades from a time window, then makes
one additional LLM call asking for portfolio-level pattern observations (not
just per-trade lessons).

All LLM calls are wrapped in try/except — a reflection failure must never
interrupt trading.  When the LLM is unavailable, all methods return [].
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from core.llm.base import BaseLLMClient
from core.memory.long_term import LongTermMemory
from core.memory.trade_journal import TradeJournal


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM = """You are a trading journal analyst reviewing completed trades.
Your job is to extract actionable lessons that will improve future trading decisions.

Rules:
- Focus on ACTIONABLE insights, not just observations ("Next time X, do Y" not just "X happened").
- Be specific to the ticker or sector when relevant; be general when the lesson applies broadly.
- Rate importance 8-10 for insights that could prevent significant losses or capture larger gains,
  5-7 for refinements, 1-4 for minor observations.
- Extract 1-3 lessons per trade. Quality over quantity.
- Output valid JSON only — no markdown, no explanation outside the JSON."""

_BATCH_SYSTEM = """You are a trading performance coach reviewing a portfolio's recent trade history.
Your job is to identify recurring patterns across multiple trades and extract high-level lessons.

Rules:
- Focus on PATTERNS across trades, not per-trade specifics.
- A pattern requires at least 2 supporting data points from the trades provided.
- Rate importance 7-10 for systematic issues, 5-6 for tendencies to watch.
- Extract 1-4 portfolio-level lessons. Quality over quantity.
- Output valid JSON only — no markdown, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReflectionLesson:
    """A single lesson extracted from a completed trade."""
    content: str
    importance: int          # 1-10
    tags: List[str] = field(default_factory=list)
    trade_ticker: str = ""
    trade_pnl: float = 0.0
    confidence: float = 0.5  # LLM's stated confidence 0-1
    memory_id: Optional[int] = None  # Set after storing in LongTermMemory


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReflectionEngine:
    """
    Extracts and stores lessons from completed trades.

    Usage::

        engine = ReflectionEngine(llm_client, memory, journal)

        # After a sell trade completes:
        lessons = engine.reflect_on_trade(sell_trade_id=42)

        # During postmarket batch:
        lessons = engine.reflect_batch(days=1)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory: LongTermMemory,
        journal: TradeJournal,
        max_lessons_per_trade: int = 3,
    ):
        self._client = llm_client
        self._memory = memory
        self._journal = journal
        self._max_lessons = max_lessons_per_trade

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect_on_trade(self, sell_trade_id: int) -> List[ReflectionLesson]:
        """
        Reflect on a single completed sell trade.

        Returns a (possibly empty) list of lessons that were extracted and
        stored in long-term memory.  Never raises; returns [] on any error.
        """
        try:
            pair = self._journal.get_trade_pair(sell_trade_id)
            if not pair:
                return []

            sell = pair["sell"]
            buy = pair["buy"]

            decision = None
            if sell.get("decision_id"):
                decision = self._journal.get_decision_by_id(sell["decision_id"])

            context = self._build_trade_context(sell, buy, decision)
            raw = self._client.chat_simple(
                system=_REFLECTION_SYSTEM,
                user=context,
                max_tokens=800,
            )
            raw_lessons = self._parse_lessons(raw, sell.get("ticker", ""))
            return self._store_unique_lessons(raw_lessons)
        except Exception as e:
            print(f"[ReflectionEngine] reflect_on_trade({sell_trade_id}) failed: {e}")
            return []

    def reflect_batch(self, days: int = 1) -> List[ReflectionLesson]:
        """
        Reflect on all sell trades in the given window.

        Runs per-trade reflection for each sell, then an additional batch call
        looking for portfolio-level patterns.  Returns all stored lessons.
        """
        try:
            sell_trades = self._journal.get_recent_sell_trades(days=days)
            if not sell_trades:
                return []

            all_lessons: List[ReflectionLesson] = []

            # Per-trade reflection
            for trade in sell_trades:
                lessons = self.reflect_on_trade(trade["id"])
                all_lessons.extend(lessons)

            # Portfolio-level pattern reflection (only when ≥2 trades)
            if len(sell_trades) >= 2:
                pattern_lessons = self._reflect_portfolio_patterns(sell_trades)
                all_lessons.extend(pattern_lessons)

            return all_lessons
        except Exception as e:
            print(f"[ReflectionEngine] reflect_batch(days={days}) failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Private: context builders
    # ------------------------------------------------------------------

    def _build_trade_context(
        self,
        sell: Dict[str, Any],
        buy: Dict[str, Any],
        decision: Optional[Dict[str, Any]],
    ) -> str:
        ticker = sell.get("ticker", "UNKNOWN")
        entry_price = buy.get("price", 0.0)
        exit_price = sell.get("price", 0.0)
        quantity = sell.get("quantity", 0)
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0.0

        buy_ts = buy.get("timestamp", 0)
        sell_ts = sell.get("timestamp", time.time())
        hold_days = max(0, int((sell_ts - buy_ts) / 86400))

        lines = [
            f"Review this completed trade and extract actionable lessons.",
            f"",
            f"Ticker: {ticker}",
            f"Entry price: ${entry_price:.2f}  →  Exit price: ${exit_price:.2f}",
            f"Quantity: {quantity}",
            f"Realized P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)",
            f"Hold duration: {hold_days} day(s)",
        ]

        if decision:
            analysis = decision.get("analysis", {})
            lines += [
                f"",
                f"Decision context at entry:",
                f"  Signal: {decision.get('signal', 'N/A')} "
                f"(confidence: {decision.get('confidence', 0):.0%})",
                f"  Technical score: {analysis.get('technical_score', 'N/A')}",
                f"  Fundamental score: {analysis.get('fundamental_score', 'N/A')}",
                f"  Sentiment score: {analysis.get('sentiment_score', 'N/A')}",
                f"  Reasoning: {analysis.get('reasoning', 'N/A')}",
            ]

        lines += [
            f"",
            f"Output a JSON array of lessons (1–{self._max_lessons} max):",
            f'[{{"lesson": "<actionable lesson>", "importance": <1-10>, '
            f'"tags": ["<tag>"], "confidence": <0.0-1.0>}}]',
        ]
        return "\n".join(lines)

    def _build_batch_context(self, sell_trades: List[Dict[str, Any]]) -> str:
        lines = [
            "Review the following recent sell trades and identify portfolio-level patterns.",
            "",
        ]
        for i, trade in enumerate(sell_trades, 1):
            ticker = trade.get("ticker", "?")
            price = trade.get("price", 0.0)
            pnl = trade.get("realized_pnl")
            pnl_str = f"${pnl:+.2f}" if pnl is not None else "unknown"
            lines.append(f"{i}. {ticker}: exit ${price:.2f}, realized P&L {pnl_str}")

        lines += [
            "",
            f"Output a JSON array of portfolio-level lessons (1–4 max):",
            f'[{{"lesson": "<pattern-based lesson>", "importance": <1-10>, '
            f'"tags": ["<tag>"], "confidence": <0.0-1.0>}}]',
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private: LLM + parsing
    # ------------------------------------------------------------------

    def _reflect_portfolio_patterns(
        self, sell_trades: List[Dict[str, Any]]
    ) -> List[ReflectionLesson]:
        try:
            context = self._build_batch_context(sell_trades)
            raw = self._client.chat_simple(
                system=_BATCH_SYSTEM,
                user=context,
                max_tokens=600,
            )
            raw_lessons = self._parse_lessons(raw, ticker="portfolio")
            return self._store_unique_lessons(raw_lessons)
        except Exception as e:
            print(f"[ReflectionEngine] Portfolio pattern reflection failed: {e}")
            return []

    def _parse_lessons(
        self, raw: str, ticker: str
    ) -> List[ReflectionLesson]:
        """Parse JSON lessons from LLM response. Returns [] on any parse error."""
        # Strip markdown code fences if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        json_str = m.group(1) if m else raw.strip()

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Try extracting a JSON array from anywhere in the text
            m2 = re.search(r"(\[[\s\S]*?\])", raw)
            if not m2:
                return []
            try:
                data = json.loads(m2.group(1))
            except (json.JSONDecodeError, ValueError):
                return []

        if not isinstance(data, list):
            return []

        lessons = []
        for item in data[: self._max_lessons]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("lesson", "")).strip()
            if not content:
                continue
            lessons.append(ReflectionLesson(
                content=content,
                importance=max(1, min(10, int(item.get("importance", 5)))),
                tags=list(item.get("tags", [])),
                trade_ticker=ticker,
                confidence=float(item.get("confidence", 0.5)),
            ))
        return lessons

    # ------------------------------------------------------------------
    # Private: deduplication + storage
    # ------------------------------------------------------------------

    def _is_duplicate(self, lesson_content: str) -> bool:
        """
        Return True if a sufficiently similar lesson already exists in memory.

        Uses keyword overlap: if ≥80% of the new lesson's content words
        appear in any existing lesson, it's considered a duplicate.
        """
        words = set(lesson_content.lower().split())
        if len(words) < 3:
            return False

        existing = self._memory.search(list(words)[:5], max_results=5)
        for entry in existing:
            existing_words = set(entry.content.lower().split())
            overlap = len(words & existing_words) / len(words)
            if overlap >= 0.8:
                return True
        return False

    def _store_unique_lessons(
        self, lessons: List[ReflectionLesson]
    ) -> List[ReflectionLesson]:
        """Store non-duplicate lessons in long-term memory. Returns stored lessons."""
        stored = []
        for lesson in lessons:
            if self._is_duplicate(lesson.content):
                continue
            tags = lesson.tags or []
            if lesson.trade_ticker and lesson.trade_ticker not in tags:
                tags = [lesson.trade_ticker] + tags
            memory_id = self._memory.add_lesson(
                content=lesson.content,
                importance=lesson.importance,
                tags=tags or None,
            )
            lesson.memory_id = memory_id
            stored.append(lesson)
        return stored
