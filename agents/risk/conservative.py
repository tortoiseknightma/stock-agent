"""Conservative risk debator — argues the risk-averse perspective."""

from typing import Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are the conservative risk analyst on the portfolio risk committee. "
    "Your perspective: capital preservation, avoid tail risks. "
    "Evaluate the proposed trade from a risk-management lens.\n\n"
    "Rules:\n"
    "- Highlight downside scenarios and why position sizing should be cautious.\n"
    "- Identify specific risks that could materially harm the portfolio.\n"
    "- Cite the trade proposal details and analyst reports.\n"
    "- Write 3-4 concise points (100-150 words)."
)


class ConservativeDebatorAgent(BaseAgent):
    """Argues the risk-averse perspective in the risk debate."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="ConservativeDebator", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        argument = self._call_llm(prompt) or "Conservative view: no LLM available."

        state.risk_debate_state.rounds.append({
            "role": "conservative",
            "content": argument,
        })
        state.risk_debate_state.last_speaker = "conservative"
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
        if state.fundamental_report:
            parts.append(f"Fundamental context:\n{state.fundamental_report[:300]}")

        prior = [r for r in state.risk_debate_state.rounds if r["role"] != "conservative"]
        if prior:
            parts.append("Prior arguments:")
            for r in prior[-2:]:
                parts.append(f"[{r['role'].upper()}]: {r['content'][:200]}")

        parts.append("\nPresent your conservative risk assessment.")
        return "\n".join(parts)
