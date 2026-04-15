"""
Portfolio Manager Agent
========================
Final judge of the risk debate. Outputs the definitive BUY/HOLD/SELL decision.
Uses the DEEP model (not fast_model).
"""

import json
import re
from typing import Any, Dict, Optional

from core.llm.base import BaseLLMClient
from agents.base import BaseAgent
from agents.state import AgentState


_SYSTEM = (
    "You are the Chief Investment Officer making the final portfolio decision. "
    "You have reviewed the full analysis pipeline: analyst reports, "
    "investment debate, trade proposal, and risk debate.\n\n"
    "Rules:\n"
    "- Synthesize all inputs and render a final, definitive decision.\n"
    "- Assign conviction honestly: 0.2-0.4 uncertain, 0.5-0.65 moderate, "
    "0.7-0.9 high-conviction.\n"
    "- Output valid JSON with keys: signal (BUY/HOLD/SELL), "
    "conviction (float 0-1), reasoning (string 2-3 sentences)."
)


class PortfolioManagerAgent(BaseAgent):
    """Final judge — produces the definitive BUY/HOLD/SELL decision."""

    def __init__(self, llm_client: Optional[BaseLLMClient]):
        super().__init__(name="PortfolioManager", llm_client=llm_client,
                         system_prompt=_SYSTEM)

    def run(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        raw = self._call_llm(prompt, max_tokens=1500)

        if raw:
            parsed = self._parse_decision(raw)
            state.final_decision = raw
            state.decision_signal = parsed.get("signal", "HOLD").upper()
            state.decision_conviction = float(parsed.get("conviction", 0.5))
        else:
            state.decision_signal = self._signal_from_state(state)
            state.decision_conviction = (
                state.composite_signal.confidence
                if state.composite_signal else 0.5
            )
            state.final_decision = (
                f"Fallback: {state.decision_signal} based on composite signal."
            )

        state.log(self.name, "final_decision",
                  f"{state.decision_signal} @ {state.decision_conviction:.2f}")
        return state

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        parts = [f"Ticker: {state.ticker}\n"]

        # Analyst summary (brief)
        for label, report in [
            ("Technical", state.technical_report),
            ("Fundamental", state.fundamental_report),
            ("Sentiment", state.sentiment_report),
        ]:
            if report:
                parts.append(f"--- {label} ---\n{report[:300]}\n")

        # Investment plan
        if state.investment_plan:
            parts.append(f"--- Investment Plan ---\n{state.investment_plan}\n")

        # Trade proposal
        if state.trade_proposal:
            parts.append(f"--- Trade Proposal ---\n{state.trade_proposal}\n")

        # Risk debate
        if state.risk_debate_state.rounds:
            parts.append("--- Risk Debate ---")
            for r in state.risk_debate_state.rounds:
                parts.append(f"[{r['role'].upper()}]: {r['content'][:250]}\n")

        parts.append("Render your final decision as JSON.")
        return "\n".join(parts)

    @staticmethod
    def _parse_decision(raw: str) -> Dict[str, Any]:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        text = m.group(1) if m else raw
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract signal keyword
            for sig in ("BUY", "SELL", "HOLD"):
                if sig in raw.upper():
                    return {"signal": sig, "conviction": 0.5, "reasoning": raw}
            return {"signal": "HOLD", "conviction": 0.5, "reasoning": raw}

    @staticmethod
    def _signal_from_state(state: AgentState) -> str:
        if state.composite_signal:
            sig = state.composite_signal.signal
            if "buy" in sig:
                return "BUY"
            if "sell" in sig:
                return "SELL"
        return "HOLD"
