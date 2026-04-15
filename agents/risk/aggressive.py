"""Aggressive risk debator — argues the risk-tolerant perspective."""

from typing import Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are the aggressive risk analyst on the portfolio risk committee. "
    "Your perspective: capture upside, avoid opportunity cost. "
    "Evaluate the proposed trade from a growth-oriented lens.\n\n"
    "Rules:\n"
    "- Emphasize upside potential and why the opportunity justifies the risk.\n"
    "- Address specific concerns about position sizing being too conservative.\n"
    "- Cite the trade proposal details and analyst reports.\n"
    "- Write 3-4 concise points (100-150 words)."
)


class AggressiveDebatorAgent(BaseAgent):
    """Argues the risk-tolerant perspective in the risk debate."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="AggressiveDebator", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        argument = self._call_llm(prompt) or "Aggressive view: no LLM available."

        state.risk_debate_state.rounds.append({
            "role": "aggressive",
            "content": argument,
        })
        state.risk_debate_state.last_speaker = "aggressive"
        state.risk_debate_state.count += 1
        state.log(self.name, "argue", argument[:200])
        return state

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        parts = [
            f"Ticker: {state.ticker}",
            f"Trade Proposal:\n{state.trade_proposal}",
        ]
        if state.investment_plan:
            parts.append(f"Investment Plan:\n{state.investment_plan}")
        if state.technical_report:
            parts.append(f"Technical context:\n{state.technical_report[:300]}")

        prior = [r for r in state.risk_debate_state.rounds if r["role"] != "aggressive"]
        if prior:
            parts.append("Prior arguments:")
            for r in prior[-2:]:
                parts.append(f"[{r['role'].upper()}]: {r['content'][:200]}")

        parts.append("\nPresent your aggressive risk assessment.")
        return "\n".join(parts)
