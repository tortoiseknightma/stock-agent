"""
Research Manager Agent
=======================
Judges the Bull/Bear debate and produces an investment plan.
Uses the DEEP model (not fast_model).
"""

import json
import re
from typing import Any, Dict, Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are the research director and final arbiter of an investment debate. "
    "A Bull analyst and Bear analyst have debated whether to invest in a stock. "
    "Your mandate: weigh the arguments objectively and produce an investment plan.\n\n"
    "Rules:\n"
    "- Identify which side won each key argument and why.\n"
    "- Your recommendation must follow logically from the debate.\n"
    "- Assign conviction honestly: 0.2-0.4 uncertain, 0.5-0.65 moderate, "
    "0.7-0.9 high-conviction. Reserve 0.9+ for exceptional clarity.\n"
    "- Output valid JSON with keys: signal (BUY/HOLD/SELL), conviction (float), "
    "reasoning (string), key_factors (list of strings)."
)


class ResearchManagerAgent(BaseAgent):
    """Judge the research debate and produce an investment plan."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="ResearchManager", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        raw = self._call_llm(prompt, max_tokens=1500)

        if raw:
            parsed = self._parse_verdict(raw)
            state.investment_plan = parsed.get("reasoning", raw)
            state.investment_debate_state.verdict = raw
            # Store parsed signal for downstream use
            if "signal" in parsed:
                state.investment_plan = json.dumps(parsed)
        else:
            state.investment_plan = self._fallback_plan(state)

        state.log(self.name, "verdict", state.investment_plan[:200])
        return state

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        parts = [f"Ticker: {state.ticker}\n"]

        # Analyst summaries
        for label, report in [
            ("Technical", state.technical_report),
            ("Fundamental", state.fundamental_report),
            ("Sentiment", state.sentiment_report),
            ("Macro", state.macro_report),
        ]:
            if report:
                parts.append(f"--- {label} Report ---\n{report}\n")

        # Debate transcript
        parts.append("--- Debate Transcript ---")
        for r in state.investment_debate_state.rounds:
            parts.append(f"[{r['role'].upper()}]: {r['content']}\n")

        parts.append("Render your verdict as JSON.")
        return "\n".join(parts)

    @staticmethod
    def _parse_verdict(raw: str) -> Dict[str, Any]:
        """Extract JSON from the LLM response (handles code fences)."""
        # Try code-fence extraction first
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        text = m.group(1) if m else raw

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"reasoning": raw}

    @staticmethod
    def _fallback_plan(state: AgentState) -> str:
        if state.composite_signal:
            return (
                f"Fallback plan based on composite signal: "
                f"{state.composite_signal.signal} "
                f"(score {state.composite_signal.composite_score:+.2f})"
            )
        return "No investment plan — debate and composite signal unavailable."
