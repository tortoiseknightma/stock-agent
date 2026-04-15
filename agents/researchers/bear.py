"""
Bear Researcher Agent
======================
Argues the short thesis / stress-tests the bull case.
"""

from typing import Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior short-seller and risk analyst. Your mandate: "
    "stress-test the long thesis by finding every flaw. Present the "
    "bear case based on the analyst reports provided.\n\n"
    "Rules:\n"
    "- Be specific and evidence-based — use the numbers from the reports.\n"
    "- Directly refute the bull's specific claims where you can.\n"
    "- Do NOT manufacture risks; only raise objections grounded in data.\n"
    "- If the bull has already argued, directly counter their strongest points.\n"
    "- Write 3-5 concise points (150-250 words total)."
)


class BearResearcherAgent(BaseAgent):
    """Argue the short thesis / stress-test the bull case."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="BearResearcher", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        argument = self._call_llm(prompt) or "Bear case: no LLM available."

        state.investment_debate_state.rounds.append({
            "role": "bear",
            "content": argument,
        })
        state.investment_debate_state.last_speaker = "bear"
        state.investment_debate_state.count += 1
        state.log(self.name, "argue", argument[:200])
        return state

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        parts = [f"Ticker: {state.ticker}\n"]

        if state.technical_report:
            parts.append(f"--- Technical Report ---\n{state.technical_report}\n")
        if state.fundamental_report:
            parts.append(f"--- Fundamental Report ---\n{state.fundamental_report}\n")
        if state.sentiment_report:
            parts.append(f"--- Sentiment Report ---\n{state.sentiment_report}\n")
        if state.macro_report:
            parts.append(f"--- Macro Report ---\n{state.macro_report}\n")

        # Include prior bull arguments (to counter them)
        bull_rounds = [r for r in state.investment_debate_state.rounds
                       if r["role"] == "bull"]
        if bull_rounds:
            parts.append("--- Bull's Arguments (counter these) ---")
            for r in bull_rounds:
                parts.append(r["content"])
            parts.append("")

        parts.append("Present your bear case.")
        return "\n".join(parts)
