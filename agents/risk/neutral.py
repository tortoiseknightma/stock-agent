"""Neutral risk debator — balanced scenario analysis."""

from typing import Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are the balanced risk analyst on the portfolio risk committee. "
    "Your perspective: weigh probabilities across scenarios objectively. "
    "Evaluate the proposed trade from a scenario-analysis lens.\n\n"
    "Rules:\n"
    "- Present 2-3 scenarios (bull, base, bear) with rough probability estimates.\n"
    "- Identify what the market is pricing in and where the edge lies.\n"
    "- Cite the trade proposal and debate arguments so far.\n"
    "- Write 3-4 concise points (100-150 words)."
)


class NeutralDebatorAgent(BaseAgent):
    """Balanced scenario analysis in the risk debate."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="NeutralDebator", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        argument = self._call_llm(prompt) or "Neutral view: no LLM available."

        state.risk_debate_state.rounds.append({
            "role": "neutral",
            "content": argument,
        })
        state.risk_debate_state.last_speaker = "neutral"
        state.risk_debate_state.count += 1
        state.log(self.name, "argue", argument[:200])
        return state

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        parts = [
            f"Ticker: {state.ticker}",
            f"Trade Proposal:\n{state.trade_proposal}",
        ]
        if state.sentiment_report:
            parts.append(f"Sentiment context:\n{state.sentiment_report[:300]}")

        prior = [r for r in state.risk_debate_state.rounds if r["role"] != "neutral"]
        if prior:
            parts.append("Prior arguments:")
            for r in prior[-2:]:
                parts.append(f"[{r['role'].upper()}]: {r['content'][:200]}")

        parts.append("\nPresent your balanced scenario analysis.")
        return "\n".join(parts)
