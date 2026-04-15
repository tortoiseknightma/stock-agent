"""
Macro Analyst Agent
====================
Purely LLM-driven macro analysis — interest rates, geopolitical,
sector rotation, and macroeconomic factors relevant to a ticker.
"""

from typing import Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior macro strategist at a global investment bank. "
    "Given a stock ticker, produce a concise macro analysis report "
    "(150-250 words) covering: current interest rate environment and "
    "its impact on this company/sector, relevant geopolitical factors, "
    "sector rotation trends, inflation/consumer spending outlook, and "
    "an overall macro-level bullish/bearish/neutral assessment for this "
    "stock. Be specific and actionable."
)


class MacroAnalystAgent(BaseAgent):
    """Produce a macro analysis report via LLM. No rule-based analyzer to wrap."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="MacroAnalyst", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = (
            f"Ticker: {state.ticker}\n"
            f"Date: {state.trade_date}\n\n"
            f"Produce your macro environment analysis for {state.ticker}."
        )
        report = self._call_llm(prompt)
        state.macro_report = report or "Macro analysis unavailable (no LLM configured)."
        state.log(self.name, "report", state.macro_report[:200])
        return state
