"""Tests for SentimentAnalyzer — rule-based path, LLM integration, VIX logic."""

import pytest
from unittest.mock import MagicMock, patch
from analysis.sentiment.sentiment import SentimentAnalyzer, SentimentSignal
from data.sources.news import NewsItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_news(titles_and_sentiments: list) -> list:
    """Build NewsItem list from [(title, sentiment_score)] pairs."""
    items = []
    for i, (title, sent) in enumerate(titles_and_sentiments):
        items.append(NewsItem(
            title=title, summary=title,
            source="Test", url="",
            published_at=float(1000 - i),  # most recent first
            tickers=["TEST"],
            sentiment=sent,
        ))
    return items


def make_llm_result(overall_score: float, confidence: float = 0.8,
                    signal: str = None):
    """Mock LLM NewsAnalysisResult."""
    from analysis.llm.news_analyzer import NewsAnalysisResult
    if signal is None:
        signal = "bullish" if overall_score > 0 else "bearish" if overall_score < 0 else "neutral"
    return NewsAnalysisResult(
        ticker="TEST",
        overall_score=overall_score,
        overall_signal=signal,
        confidence=confidence,
        key_themes=["growth", "earnings"],
        summary="Test summary.",
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_no_llm(self):
        sa = SentimentAnalyzer()
        assert sa._llm is None

    def test_accepts_llm_analyzer(self):
        mock_llm = MagicMock()
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        assert sa._llm is mock_llm


# ---------------------------------------------------------------------------
# Empty / no news
# ---------------------------------------------------------------------------

class TestEmptyNews:
    def test_no_news_returns_neutral(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([], ticker="AAPL")
        assert result.score == pytest.approx(0.0)
        assert result.signal == "hold"

    def test_no_news_zero_volume(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([])
        assert result.news_volume == 0


# ---------------------------------------------------------------------------
# Rule-based path (no LLM)
# ---------------------------------------------------------------------------

class TestRuleBased:
    def test_positive_news_positive_score(self):
        sa = SentimentAnalyzer()
        news = make_news([("Great earnings beat", 0.8), ("Record revenue", 0.7)])
        result = sa.analyze(news, ticker="AAPL")
        assert result.score > 0

    def test_negative_news_negative_score(self):
        sa = SentimentAnalyzer()
        news = make_news([("Massive layoffs", -0.8), ("Earnings miss", -0.7)])
        result = sa.analyze(news, ticker="AAPL")
        assert result.score < 0

    def test_score_clamped_to_bounds(self):
        sa = SentimentAnalyzer()
        # Extreme positive sentiment
        news = make_news([("Best results ever", 1.0)] * 20)
        result = sa.analyze(news)
        assert -1.0 <= result.score <= 1.0

    def test_recency_weighting(self):
        """Most recent article (index 0) gets highest weight."""
        sa = SentimentAnalyzer()
        # First article very positive, rest negative
        news = make_news([("Great news", 1.0)] + [("Bad news", -0.5)] * 5)
        result_first_positive = sa.analyze(news)

        # Reverse: first very negative, rest positive
        news2 = make_news([("Bad news", -1.0)] + [("Great news", 0.5)] * 5)
        result_first_negative = sa.analyze(news2)

        assert result_first_positive.score > result_first_negative.score

    def test_news_sentiment_stored(self):
        sa = SentimentAnalyzer()
        news = make_news([("Good news", 0.5), ("More good news", 0.4)])
        result = sa.analyze(news)
        assert result.news_volume == 2
        assert result.news_sentiment > 0

    def test_ticker_stored_in_result(self):
        sa = SentimentAnalyzer()
        result = sa.analyze(make_news([("Test", 0.5)]), ticker="MSFT")
        assert result.ticker == "MSFT"


# ---------------------------------------------------------------------------
# VIX analysis
# ---------------------------------------------------------------------------

class TestVIX:
    def test_high_vix_adds_negative_component(self):
        sa = SentimentAnalyzer()
        no_vix = sa.analyze([], vix_level=None)
        high_vix = sa.analyze([], vix_level=35)
        assert high_vix.score < no_vix.score

    def test_high_vix_sets_signal(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([], vix_level=35)
        assert result.vix_signal == "high_fear"
        assert result.vix_level == pytest.approx(35)

    def test_elevated_vix(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([], vix_level=25)
        assert result.vix_signal == "elevated"

    def test_low_vix_adds_positive_component(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([], vix_level=12)
        assert result.vix_signal == "low_vol"
        assert result.score > 0

    def test_normal_vix_neutral(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([], vix_level=18)
        assert result.vix_signal == "normal"

    def test_vix_appears_in_reasoning(self):
        sa = SentimentAnalyzer()
        result = sa.analyze([], vix_level=35)
        assert "VIX" in result.reasoning


# ---------------------------------------------------------------------------
# LLM integration path
# ---------------------------------------------------------------------------

class TestLLMPath:
    def test_llm_called_when_configured_with_ticker(self):
        mock_llm = MagicMock()
        mock_llm.analyze_batch.return_value = make_llm_result(0.6)
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        news = make_news([("Good news", 0.5)])
        sa.analyze(news, ticker="AAPL")
        mock_llm.analyze_batch.assert_called_once_with("AAPL", news)

    def test_llm_not_called_without_ticker(self):
        mock_llm = MagicMock()
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        news = make_news([("Good news", 0.5)])
        sa.analyze(news, ticker="")
        mock_llm.analyze_batch.assert_not_called()

    def test_llm_not_called_for_empty_news(self):
        mock_llm = MagicMock()
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        sa.analyze([], ticker="AAPL")
        mock_llm.analyze_batch.assert_not_called()

    def test_high_confidence_llm_blends_overall_score(self):
        """LLM overall score (confidence >= 0.6) is blended 60/40."""
        mock_llm = MagicMock()
        mock_llm.analyze_batch.return_value = make_llm_result(0.9, confidence=0.8)
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        # Rule-based sentiment is 0 (neutral title)
        news = make_news([("Company updates policy", 0.0)])
        result = sa.analyze(news, ticker="AAPL")
        # LLM score (0.9) blended with article score (≈0) → should be notably positive
        assert result.score > 0.3

    def test_low_confidence_llm_does_not_dominate(self):
        """LLM result with confidence < 0.6 is not blended — rule-based prevails."""
        mock_llm = MagicMock()
        mock_llm.analyze_batch.return_value = make_llm_result(0.9, confidence=0.4)
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        # Rule-based sentiment is negative
        news = make_news([("Earnings miss disaster", -0.8)])
        result = sa.analyze(news, ticker="AAPL")
        # LLM not blended; rule-based drives negative score
        assert result.score < 0

    def test_llm_failure_falls_back_to_rule_based(self):
        mock_llm = MagicMock()
        mock_llm.analyze_batch.return_value = None  # simulate failure
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        news = make_news([("Great quarter", 0.7)])
        result = sa.analyze(news, ticker="AAPL")
        # Rule-based should still produce positive score
        assert result.score > 0

    def test_llm_reasoning_tag_in_result(self):
        mock_llm = MagicMock()
        mock_llm.analyze_batch.return_value = make_llm_result(0.6, confidence=0.8)
        sa = SentimentAnalyzer(llm_analyzer=mock_llm)
        news = make_news([("Strong quarter", 0.6)])
        result = sa.analyze(news, ticker="AAPL")
        assert "LLM-scored" in result.reasoning


# ---------------------------------------------------------------------------
# Score → signal mapping
# ---------------------------------------------------------------------------

class TestScoreToSignal:
    @pytest.mark.parametrize("score,expected", [
        (0.6, "strong_buy"),
        (0.3, "buy"),
        (0.0, "hold"),
        (-0.3, "sell"),
        (-0.6, "strong_sell"),
    ])
    def test_thresholds(self, score, expected):
        assert SentimentAnalyzer._score_to_signal(score) == expected


# ---------------------------------------------------------------------------
# Portfolio analysis
# ---------------------------------------------------------------------------

class TestPortfolioSentiment:
    def test_multi_ticker(self):
        sa = SentimentAnalyzer()
        ticker_news = {
            "AAPL": make_news([("Apple beats earnings", 0.7)]),
            "TSLA": make_news([("Tesla misses", -0.6)]),
        }
        results = sa.analyze_portfolio_sentiment(ticker_news)
        assert "AAPL" in results
        assert "TSLA" in results
        assert results["AAPL"].score > results["TSLA"].score
