"""
Sentiment Analyst Agent
========================
Wraps SentimentAnalyzer with LLM interpretation to produce a detailed report.
"""

from typing import Optional

from core.llm.base import BaseLLMClient
from analysis.sentiment.sentiment import SentimentAnalyzer, SentimentSignal
from data.sources.news import NewsProvider
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior market sentiment analyst. Given news sentiment data and "
    "scores for a stock, produce a concise report (150-250 words) covering: "
    "overall news tone, key headlines and their impact, market fear/greed "
    "indicators, and a clear bullish/bearish/neutral sentiment assessment. "
    "Be specific — cite the sentiment scores and news volume provided."
)


class SentimentAnalystAgent(BaseAgent):
    """Run sentiment analysis and produce an LLM-interpreted report."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient],
        analyzer: SentimentAnalyzer,
        news_provider: NewsProvider,
    ):
        super().__init__(name="SentimentAnalyst", llm_client=llm_client,
                         system_prompt=_SYSTEM)
        self._analyzer = analyzer
        self._news = news_provider

    def run(self, state: AgentState) -> AgentState:
        # 1. Get news
        try:
            news_items = self._news.get_ticker_news(state.ticker, max_items=10)
        except Exception:
            news_items = []

        if not news_items:
            state.sentiment_report = f"No recent news found for {state.ticker}."
            state.log(self.name, "skip", "no news data")
            return state

        # 2. Run rule-based analyzer
        signal = self._analyzer.analyze(news_items, ticker=state.ticker)
        signal.ticker = state.ticker
        state.sentiment_signal = signal

        # 3. LLM interpretation
        headlines = [getattr(n, "title", str(n)) for n in news_items[:5]]
        report = self._call_llm(self._build_prompt(state.ticker, signal, headlines))
        state.sentiment_report = report or self._fallback_report(signal)
        state.log(self.name, "report", state.sentiment_report[:200])
        return state

    @staticmethod
    def _build_prompt(ticker: str, signal: SentimentSignal, headlines: list) -> str:
        hl_text = "\n".join(f"  - {h}" for h in headlines)
        return (
            f"Ticker: {ticker}\n"
            f"Overall sentiment score: {signal.score:+.2f} ({signal.signal})\n"
            f"News sentiment: {signal.news_sentiment:+.2f}, "
            f"News volume: {signal.news_volume} articles\n"
            f"VIX signal: {signal.vix_signal}\n"
            f"Recent headlines:\n{hl_text}\n\n"
            f"Produce your sentiment analysis report."
        )

    @staticmethod
    def _fallback_report(signal: SentimentSignal) -> str:
        direction = "bullish" if signal.score > 0.1 else "bearish" if signal.score < -0.1 else "neutral"
        return (
            f"Sentiment outlook: {direction} (score {signal.score:+.2f}). "
            f"News sentiment {signal.news_sentiment:+.2f} across {signal.news_volume} articles. "
            f"VIX: {signal.vix_signal}. "
            f"[Rule-based summary — LLM unavailable]"
        )
