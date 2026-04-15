"""
Trader Agent
=============
Converts the investment plan into a concrete trade proposal.
"""

import json
from typing import Optional

from core.llm.base import BaseLLMClient
from core.broker.base import BaseBroker
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior equity trader at a hedge fund. Given an investment plan "
    "and current market context, produce a concrete trade proposal.\n\n"
    "Rules:\n"
    "- Specify: direction (BUY/SELL/HOLD), suggested position size (% of portfolio), "
    "and timing rationale.\n"
    "- Consider current account balance and existing positions.\n"
    "- Be decisive and specific. Keep it to 3-5 sentences."
)


class TraderAgent(BaseAgent):
    """Convert investment plan into a concrete trade proposal."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient],
        broker: Optional[BaseBroker] = None,
    ):
        super().__init__(name="Trader", llm_client=llm_client,
                         system_prompt=_SYSTEM)
        self._broker = broker

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        raw = self._call_llm(prompt)
        state.trade_proposal = raw or self._rule_based_proposal(state)
        state.log(self.name, "proposal", state.trade_proposal[:200])
        return state

    def _build_prompt(self, state: AgentState) -> str:
        parts = [f"Ticker: {state.ticker}"]

        # Account context
        if self._broker:
            try:
                account = self._broker.get_account_info()
                price = self._broker.get_current_price(state.ticker)
                position = self._broker.get_position(state.ticker)
                parts.append(f"Portfolio: ${account.net_liquidation:,.0f} total, "
                              f"${account.total_cash:,.0f} cash")
                if price:
                    parts.append(f"Current price: ${price:.2f}")
                if position:
                    parts.append(f"Current position: {position.quantity} shares")
                else:
                    parts.append("Current position: none")
            except Exception:
                pass

        # Investment plan
        parts.append(f"\nInvestment Plan:\n{state.investment_plan}")
        parts.append("\nProduce your trade proposal.")
        return "\n".join(parts)

    @staticmethod
    def _rule_based_proposal(state: AgentState) -> str:
        """Derive a simple proposal from composite_signal when LLM is unavailable."""
        if state.composite_signal:
            sig = state.composite_signal.signal
            score = state.composite_signal.composite_score
            return (
                f"Rule-based proposal for {state.ticker}: {sig.upper()} "
                f"(composite score {score:+.2f}). "
                f"Suggested position: 5% of portfolio at market. "
                f"[LLM unavailable]"
            )
        return (
            f"No trade proposal available for {state.ticker} — "
            f"no investment plan and no LLM."
        )
