"""
Bull Researcher Agent
======================
Argues the long thesis using analyst reports and prior debate rounds.
"""

from typing import Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are a senior equity analyst at a long-only fund. Your mandate: "
    "identify and articulate the strongest arguments for buying or holding "
    "this stock based on the analyst reports provided.\n\n"
    "Rules:\n"
    "- Be specific and evidence-based — cite the numbers from the reports.\n"
    "- Anticipate the obvious bear objections and pre-empt them.\n"
    "- Do NOT hedge excessively; your job is to steelman the bull case.\n"
    "- If the bear has already argued, directly address their strongest points.\n"
    "- Write 3-5 concise points (150-250 words total)."
)


class BullResearcherAgent(BaseAgent):
    """Argue the long thesis based on analyst reports."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="BullResearcher", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        argument = self._call_llm(prompt) or "Bull case: no LLM available."

        state.investment_debate_state.rounds.append({
            "role": "bull",
            "content": argument,
        })
        state.investment_debate_state.last_speaker = "bull"
        state.investment_debate_state.count += 1
        state.log(self.name, "argue", argument[:200])
        return state

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        parts = [f"Ticker: {state.ticker}\n"]

        # Include analyst reports
        if state.technical_report:
            parts.append(f"--- Technical Report ---\n{state.technical_report}\n")
        if state.fundamental_report:
            parts.append(f"--- Fundamental Report ---\n{state.fundamental_report}\n")
        if state.sentiment_report:
            parts.append(f"--- Sentiment Report ---\n{state.sentiment_report}\n")
        if state.macro_report:
            parts.append(f"--- Macro Report ---\n{state.macro_report}\n")

        # Include prior bear arguments (for rebuttal rounds)
        bear_rounds = [r for r in state.investment_debate_state.rounds
                       if r["role"] == "bear"]
        if bear_rounds:
            parts.append("--- Bear's Arguments (respond to these) ---")
            for r in bear_rounds:
                parts.append(r["content"])
            parts.append("")

        parts.append("Present your bull case.")
        return "\n".join(parts)
