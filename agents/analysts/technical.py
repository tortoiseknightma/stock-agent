"""
Technical Analyst Agent
========================
Wraps TechnicalAnalyzer with LLM interpretation to produce a detailed report.
"""

from typing import Optional

from core.llm.base import BaseLLMClient
from analysis.technical.technical import TechnicalAnalyzer, TechnicalSignal
from data.sources.market_data import MarketDataProvider
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior technical analyst at a quantitative hedge fund. "
    "Given raw technical indicator data for a stock, produce a concise "
    "report (150-250 words) covering: trend direction, momentum signals, "
    "key support/resistance levels, and a clear bullish/bearish/neutral "
    "assessment. Be specific — cite the indicator values provided."
)


class TechnicalAnalystAgent(BaseAgent):
    """Run technical analysis and produce an LLM-interpreted report."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient],
        analyzer: TechnicalAnalyzer,
        data_provider: MarketDataProvider,
    ):
        super().__init__(name="TechnicalAnalyst", llm_client=llm_client,
                         system_prompt=_SYSTEM)
        self._analyzer = analyzer
        self._data = data_provider

    def run(self, state: AgentState) -> AgentState:
        # 1. Get price history
        history = self._data.get_history(state.ticker, period="6mo")
        if not history:
            state.technical_report = f"No price data available for {state.ticker}."
            state.log(self.name, "skip", "no price data")
            return state

        # 2. Run rule-based analyzer
        signal = self._analyzer.analyze(history)
        signal.ticker = state.ticker
        state.technical_signal = signal

        # 3. LLM interpretation
        report = self._call_llm(self._build_prompt(state.ticker, signal))
        state.technical_report = report or self._fallback_report(signal)
        state.log(self.name, "report", state.technical_report[:200])
        return state

    @staticmethod
    def _build_prompt(ticker: str, signal: TechnicalSignal) -> str:
        ind = signal.indicators
        return (
            f"Ticker: {ticker}\n"
            f"Overall score: {signal.score:+.2f} ({signal.signal})\n"
            f"MA score: {signal.ma_score:+.2f}, RSI score: {signal.rsi_score:+.2f}, "
            f"MACD score: {signal.macd_score:+.2f}, BB score: {signal.bb_score:+.2f}\n"
            f"Volume score: {signal.volume_score:+.2f}, Trend score: {signal.trend_score:+.2f}\n"
            f"Indicators: RSI={ind.get('rsi', 'N/A')}, "
            f"MACD={ind.get('macd_line', 'N/A')}, "
            f"BB %B={ind.get('bb_pct_b', 'N/A')}, "
            f"Volume ratio={ind.get('volume_ratio', 'N/A')}, "
            f"Price={ind.get('current_price', 'N/A')}\n\n"
            f"Produce your technical analysis report."
        )

    @staticmethod
    def _fallback_report(signal: TechnicalSignal) -> str:
        direction = "bullish" if signal.score > 0.1 else "bearish" if signal.score < -0.1 else "neutral"
        return (
            f"Technical outlook: {direction} (score {signal.score:+.2f}). "
            f"MA {signal.ma_score:+.2f}, RSI {signal.rsi_score:+.2f}, "
            f"MACD {signal.macd_score:+.2f}. "
            f"[Rule-based summary — LLM unavailable]"
        )
