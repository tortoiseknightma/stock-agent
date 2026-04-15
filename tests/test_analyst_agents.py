"""Tests for analyst layer agents."""

import pytest
from unittest.mock import MagicMock, patch

from agents.state import AgentState
from agents.analysts.technical import TechnicalAnalystAgent
from agents.analysts.fundamental import FundamentalAnalystAgent
from agents.analysts.sentiment import SentimentAnalystAgent
from agents.analysts.macro import MacroAnalystAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_technical_signal():
    sig = MagicMock()
    sig.score = 0.5
    sig.signal = "buy"
    sig.ma_score = 0.3
    sig.rsi_score = 0.4
    sig.macd_score = 0.2
    sig.bb_score = 0.1
    sig.volume_score = 0.3
    sig.trend_score = 0.4
    sig.indicators = {
        "rsi": 55.0, "macd_line": 0.5, "bb_pct_b": 0.6,
        "volume_ratio": 1.2, "current_price": 180.0,
    }
    return sig


def _make_fundamental_signal():
    sig = MagicMock()
    sig.score = 0.4
    sig.signal = "buy"
    sig.valuation_score = 0.3
    sig.growth_score = 0.5
    sig.profitability_score = 0.4
    sig.health_score = 0.3
    sig.ratios = {"pe_forward": 25.0, "roe": 0.3}
    return sig


def _make_sentiment_signal():
    sig = MagicMock()
    sig.score = 0.2
    sig.signal = "buy"
    sig.news_sentiment = 0.3
    sig.news_volume = 5
    sig.vix_signal = "normal"
    return sig


# ---------------------------------------------------------------------------
# TechnicalAnalystAgent
# ---------------------------------------------------------------------------

class TestTechnicalAnalyst:
    def test_writes_signal_and_report(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Technical report: bullish trend"
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_technical_signal()
        data = MagicMock()
        data.get_history.return_value = [{"close": 150.0}]

        agent = TechnicalAnalystAgent(llm, analyzer, data)
        state = AgentState(ticker="AAPL", trade_date="2024-01-15")
        state = agent.run(state)

        assert state.technical_signal is not None
        assert state.technical_report == "Technical report: bullish trend"
        assert len(state.agent_log) == 1

    def test_fallback_when_no_llm(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_technical_signal()
        data = MagicMock()
        data.get_history.return_value = [{"close": 150.0}]

        agent = TechnicalAnalystAgent(None, analyzer, data)
        state = agent.run(AgentState(ticker="AAPL"))

        assert state.technical_signal is not None
        assert "Rule-based" in state.technical_report

    def test_no_data_skips_analysis(self):
        data = MagicMock()
        data.get_history.return_value = []
        agent = TechnicalAnalystAgent(None, MagicMock(), data)
        state = agent.run(AgentState(ticker="AAPL"))

        assert state.technical_signal is None
        assert "No price data" in state.technical_report


# ---------------------------------------------------------------------------
# FundamentalAnalystAgent
# ---------------------------------------------------------------------------

class TestFundamentalAnalyst:
    def test_writes_signal_and_report(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Fundamental report: undervalued"
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_fundamental_signal()
        data = MagicMock()
        data.get_ratios.return_value = MagicMock()
        data.get_sector.return_value = "Technology"

        agent = FundamentalAnalystAgent(llm, analyzer, data)
        state = agent.run(AgentState(ticker="AAPL"))

        assert state.fundamental_signal is not None
        assert state.fundamental_report == "Fundamental report: undervalued"

    def test_fallback_when_no_llm(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_fundamental_signal()
        data = MagicMock()
        data.get_ratios.return_value = MagicMock()
        data.get_sector.return_value = "Technology"

        agent = FundamentalAnalystAgent(None, analyzer, data)
        state = agent.run(AgentState(ticker="AAPL"))

        assert "Rule-based" in state.fundamental_report

    def test_no_ratios_skips(self):
        data = MagicMock()
        data.get_ratios.return_value = None
        data.get_sector.return_value = None
        agent = FundamentalAnalystAgent(None, MagicMock(), data)
        state = agent.run(AgentState(ticker="AAPL"))

        assert "No fundamental data" in state.fundamental_report

    def test_data_exception_skips(self):
        data = MagicMock()
        data.get_ratios.side_effect = RuntimeError("API down")
        agent = FundamentalAnalystAgent(None, MagicMock(), data)
        state = agent.run(AgentState(ticker="AAPL"))

        assert "No fundamental data" in state.fundamental_report


# ---------------------------------------------------------------------------
# SentimentAnalystAgent
# ---------------------------------------------------------------------------

class TestSentimentAnalyst:
    def test_writes_signal_and_report(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Sentiment report: cautiously optimistic"
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_sentiment_signal()
        news = MagicMock()
        news_item = MagicMock()
        news_item.title = "AAPL beats earnings"
        news.get_ticker_news.return_value = [news_item]

        agent = SentimentAnalystAgent(llm, analyzer, news)
        state = agent.run(AgentState(ticker="AAPL"))

        assert state.sentiment_signal is not None
        assert state.sentiment_report == "Sentiment report: cautiously optimistic"

    def test_fallback_when_no_llm(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_sentiment_signal()
        news = MagicMock()
        news.get_ticker_news.return_value = [MagicMock(title="headline")]

        agent = SentimentAnalystAgent(None, analyzer, news)
        state = agent.run(AgentState(ticker="AAPL"))

        assert "Rule-based" in state.sentiment_report

    def test_no_news_skips(self):
        news = MagicMock()
        news.get_ticker_news.return_value = []
        agent = SentimentAnalystAgent(None, MagicMock(), news)
        state = agent.run(AgentState(ticker="AAPL"))

        assert "No recent news" in state.sentiment_report


# ---------------------------------------------------------------------------
# MacroAnalystAgent
# ---------------------------------------------------------------------------

class TestMacroAnalyst:
    def test_writes_report_with_llm(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "Macro report: rising rates headwind"
        agent = MacroAnalystAgent(llm)
        state = agent.run(AgentState(ticker="AAPL", trade_date="2024-01-15"))

        assert state.macro_report == "Macro report: rising rates headwind"
        assert len(state.agent_log) == 1

    def test_fallback_without_llm(self):
        agent = MacroAnalystAgent(None)
        state = agent.run(AgentState(ticker="AAPL"))

        assert "unavailable" in state.macro_report

    def test_prompt_includes_ticker(self):
        llm = MagicMock()
        llm.chat_simple.return_value = "report"
        agent = MacroAnalystAgent(llm)
        agent.run(AgentState(ticker="NVDA", trade_date="2024-06-01"))

        call_args = llm.chat_simple.call_args
        assert "NVDA" in call_args[1]["user"]
