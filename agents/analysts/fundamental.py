"""
Fundamental Analyst Agent
==========================
Wraps FundamentalAnalyzer with LLM interpretation to produce a detailed report.
"""

from typing import Optional

from core.llm.base import BaseLLMClient
from analysis.fundamental.fundamental import FundamentalAnalyzer, FundamentalSignal
from data.sources.fundamentals import FundamentalsProvider
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior fundamental analyst at an institutional asset manager. "
    "Given financial ratios and fundamental scores for a stock, produce a "
    "concise report (150-250 words) covering: valuation assessment, growth "
    "trajectory, profitability quality, balance sheet health, and a clear "
    "bullish/bearish/neutral assessment. Be specific — cite the numbers."
)


class FundamentalAnalystAgent(BaseAgent):
    """Run fundamental analysis and produce an LLM-interpreted report."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient],
        analyzer: FundamentalAnalyzer,
        data_provider: FundamentalsProvider,
    ):
        super().__init__(name="FundamentalAnalyst", llm_client=llm_client,
                         system_prompt=_SYSTEM)
        self._analyzer = analyzer
        self._data = data_provider

    def run(self, state: AgentState) -> AgentState:
        # 1. Get financial data
        try:
            ratios = self._data.get_ratios(state.ticker)
            sector = self._data.get_sector(state.ticker)
        except Exception:
            state.fundamental_report = f"No fundamental data available for {state.ticker}."
            state.log(self.name, "skip", "no fundamental data")
            return state

        if ratios is None:
            state.fundamental_report = f"No fundamental data available for {state.ticker}."
            state.log(self.name, "skip", "no fundamental data")
            return state

        # 2. Run rule-based analyzer
        signal = self._analyzer.analyze(ratios, sector or "")
        signal.ticker = state.ticker
        state.fundamental_signal = signal

        # 3. LLM interpretation
        report = self._call_llm(self._build_prompt(state.ticker, signal))
        state.fundamental_report = report or self._fallback_report(signal)
        state.log(self.name, "report", state.fundamental_report[:200])
        return state

    @staticmethod
    def _build_prompt(ticker: str, signal: FundamentalSignal) -> str:
        ratios = signal.ratios if hasattr(signal, "ratios") and signal.ratios else {}
        return (
            f"Ticker: {ticker}\n"
            f"Overall score: {signal.score:+.2f} ({signal.signal})\n"
            f"Valuation: {signal.valuation_score:+.2f}, "
            f"Growth: {signal.growth_score:+.2f}, "
            f"Profitability: {signal.profitability_score:+.2f}, "
            f"Health: {signal.health_score:+.2f}\n"
            f"Key ratios: {ratios}\n\n"
            f"Produce your fundamental analysis report."
        )

    @staticmethod
    def _fallback_report(signal: FundamentalSignal) -> str:
        direction = "bullish" if signal.score > 0.1 else "bearish" if signal.score < -0.1 else "neutral"
        return (
            f"Fundamental outlook: {direction} (score {signal.score:+.2f}). "
            f"Valuation {signal.valuation_score:+.2f}, Growth {signal.growth_score:+.2f}, "
            f"Profitability {signal.profitability_score:+.2f}, Health {signal.health_score:+.2f}. "
            f"[Rule-based summary — LLM unavailable]"
        )
