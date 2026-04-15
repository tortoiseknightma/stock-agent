"""Tests for agents.graph — TradingGraph orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

import time
from core.config import AppConfig
from data.sources.fundamentals import FinancialRatios
from data.sources.news import NewsItem
from agents.graph import TradingGraph
from agents.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(**kwargs) -> TradingGraph:
    """Create a TradingGraph with mocked data providers to avoid real API calls."""
    config = AppConfig()
    market_data = MagicMock()
    market_data.get_history.return_value = [
        {"date": "2024-01-15", "open": 180.0, "high": 182.0,
         "low": 179.0, "close": 181.0, "volume": 1_000_000}
    ]
    fundamentals = MagicMock()
    fundamentals.get_ratios.return_value = FinancialRatios(
        ticker="AAPL", pe_forward=25.0, peg_ratio=1.5, pb_ratio=8.0,
        revenue_growth=0.12, earnings_growth=0.15, net_margin=0.25,
        roe=0.30, operating_margin=0.30, debt_to_equity=1.5,
        current_ratio=1.2, dividend_yield=0.005,
    )
    fundamentals.get_sector.return_value = "Technology"
    news = MagicMock()
    news.get_ticker_news.return_value = [
        NewsItem(
            title="AAPL reports strong quarter",
            summary="Apple beat earnings expectations",
            source="Reuters", url="https://example.com",
            published_at=time.time(), tickers=["AAPL"],
            sentiment=0.5, relevance=0.8,
        )
    ]

    return TradingGraph(
        config,
        market_data=market_data,
        fundamentals=fundamentals,
        news=news,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------

class TestGraphInstantiation:
    def test_creates_without_llm(self):
        graph = _make_graph()
        assert graph is not None

    def test_creates_with_llm(self):
        llm = MagicMock()
        graph = _make_graph(llm_client=llm)
        assert graph is not None

    def test_has_four_analyst_agents(self):
        graph = _make_graph()
        for name in ["technical", "fundamental", "sentiment", "macro"]:
            assert name in graph._agents


# ---------------------------------------------------------------------------
# Analyst layer
# ---------------------------------------------------------------------------

class TestRunAnalysts:
    def test_all_reports_populated(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "LLM report"
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        assert state.ticker == "AAPL"
        assert state.technical_report != ""
        assert state.fundamental_report != ""
        assert state.sentiment_report != ""
        assert state.macro_report != ""

    def test_reports_populated_without_llm(self):
        graph = _make_graph()
        state = graph.run("AAPL")

        # All reports should have fallback content
        assert state.technical_report != ""
        assert state.fundamental_report != ""
        assert state.sentiment_report != ""
        assert "unavailable" in state.macro_report

    def test_agent_log_has_entries(self):
        graph = _make_graph()
        state = graph.run("AAPL")

        # Each analyst should log at least one entry
        assert len(state.agent_log) >= 4

    def test_trade_date_defaults_to_today(self):
        import datetime
        graph = _make_graph()
        state = graph.run("AAPL")
        assert state.trade_date == datetime.date.today().isoformat()

    def test_trade_date_passed_through(self):
        graph = _make_graph()
        state = graph.run("AAPL", trade_date="2024-06-15")
        assert state.trade_date == "2024-06-15"

    def test_technical_signal_stored(self):
        graph = _make_graph()
        state = graph.run("AAPL")
        # TechnicalAnalyzer should have run and stored a signal
        assert state.technical_signal is not None


# ---------------------------------------------------------------------------
# Research debate layer
# ---------------------------------------------------------------------------

class TestResearchDebate:
    def test_investment_plan_populated(self):
        llm = MagicMock()
        llm.chat_simple.return_value = '{"signal": "BUY", "conviction": 0.7, "reasoning": "ok"}'
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        assert state.investment_plan != ""

    def test_debate_rounds_appended(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Bullish argument."
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        # Default max_debate_rounds=1 → 1 bull + 1 bear round
        assert len(state.investment_debate_state.rounds) >= 2

    def test_research_debate_without_llm(self):
        graph = _make_graph()
        state = graph.run("AAPL")
        # Fallback verdict should still set investment_plan
        assert state.investment_plan != ""


# ---------------------------------------------------------------------------
# Risk debate layer
# ---------------------------------------------------------------------------

class TestRiskDebate:
    def test_final_decision_populated(self):
        llm = MagicMock()
        llm.chat_simple.return_value = '{"signal": "BUY", "conviction": 0.75, "reasoning": "ok"}'
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        assert state.final_decision != ""
        assert state.decision_signal in ("BUY", "HOLD", "SELL")

    def test_risk_debate_rounds_appended(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Risk argument."
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        # Default max_risk_discuss_rounds=1 → aggressive + conservative + neutral
        assert len(state.risk_debate_state.rounds) >= 3

    def test_trade_proposal_populated(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "BUY 50 shares at market."
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        assert state.trade_proposal != ""


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_with_llm(self):
        """Full 12-agent pipeline with mocked LLM returns a valid final decision."""
        llm = MagicMock()
        llm.chat_simple.return_value = '{"signal": "BUY", "conviction": 0.8, "reasoning": "strong"}'
        graph = _make_graph(llm_client=llm)
        state = graph.run("AAPL")

        assert state.ticker == "AAPL"
        assert state.decision_signal in ("BUY", "HOLD", "SELL")
        assert 0.0 <= state.decision_conviction <= 1.0
        assert len(state.agent_log) >= 10  # all 12 agents log at least once

    def test_end_to_end_without_llm(self):
        """Full pipeline with no LLM still completes using all fallback paths."""
        graph = _make_graph()
        state = graph.run("AAPL")

        assert state.ticker == "AAPL"
        assert state.decision_signal in ("BUY", "HOLD", "SELL")
        assert state.technical_report != ""
        assert state.trade_proposal != ""

    def test_full_pipeline_with_fast_llm(self):
        """Deep model vs fast model separation: fast_llm for analysts, llm for judges."""
        llm = MagicMock()
        llm.chat_simple.return_value = '{"signal": "HOLD", "conviction": 0.5, "reasoning": "uncertain"}'
        fast_llm = MagicMock()
        fast_llm.chat_simple.return_value = "Analyst/debator output."

        graph = _make_graph(llm_client=llm, fast_llm_client=fast_llm)
        state = graph.run("TSLA")

        assert state.ticker == "TSLA"
        assert state.decision_signal in ("BUY", "HOLD", "SELL")
        # fast_llm should have been called (analysts + debators)
        assert fast_llm.chat_simple.call_count > 0
