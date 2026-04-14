"""
LLM Risk Assessor
==================
Uses a BaseLLMClient (any provider) to generate nuanced, qualitative risk
assessments that complement the rule-based RiskEngine.

Identifies risks that quantitative models miss: regulatory, competitive,
macro-narrative, execution, and tail risks.
"""

import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from core.llm.base import BaseLLMClient


_SYSTEM_PROMPT = """You are a risk analyst at a quantitative equity fund.
You identify and evaluate investment risks that pure quantitative models miss.

Risk categories you assess:
- Regulatory / legal: pending investigations, compliance exposure, policy risk
- Competitive: new entrants, pricing pressure, technology disruption
- Macro / geopolitical: interest rate sensitivity, FX exposure, trade policy
- Execution: management transitions, strategy shifts, integration risk
- Narrative: valuation story breakdowns, consensus re-rating risk
- Tail risks: low-probability, high-impact scenarios

Be specific and stock-specific. "Market risk" is not useful — name the actual risk.
Calibrate carefully: most stocks have medium risk, not extreme risk.
Mitigation factors are equally important — note what protects the investment case."""


@dataclass
class RiskAssessment:
    """
    Qualitative risk assessment produced by LLMRiskAssessor.

    Attributes:
        ticker:             Stock ticker symbol.
        risk_level:         Overall risk category.
        risk_score:         0.0 (minimal) to 1.0 (extreme).
        primary_risks:      Top 2–4 specific risk factors.
        regulatory_risk:    Regulatory/legal risk level.
        competitive_risk:   Competitive risk level.
        macro_risk:         Macro/geopolitical risk level.
        execution_risk:     Execution/management risk level.
        mitigation_factors: Factors that reduce downside risk.
        summary:            2–3 sentence risk narrative.
    """

    ticker: str
    risk_level: str             # "low" | "medium" | "high" | "very_high"
    risk_score: float           # 0.0 – 1.0

    primary_risks: List[str] = field(default_factory=list)

    regulatory_risk: str = "medium"
    competitive_risk: str = "medium"
    macro_risk: str = "medium"
    execution_risk: str = "low"

    mitigation_factors: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class LLMRiskAssessor:
    """
    Generates qualitative risk assessments using an LLM (any provider).

    Complements the rule-based ``RiskEngine`` with a nuanced narrative risk
    evaluation used for sizing and recommendation generation.

    Usage::

        from core.llm import create_llm_client
        client = create_llm_client("anthropic", "claude-opus-4-6")
        assessor = LLMRiskAssessor(client)
        assessment = assessor.assess(
            ticker="TSLA",
            composite_signal=composite,
            portfolio_context={"position_weight_pct": 8.0},
        )
        print(assessment.risk_level)        # "high"
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._client = llm_client

    def assess(
        self,
        ticker: str,
        composite_signal: Any = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[RiskAssessment]:
        """
        Generate a risk assessment for *ticker*.

        Args:
            ticker:             Stock ticker symbol.
            composite_signal:   ``CompositeSignal`` from the analysis pipeline.
            portfolio_context:  Optional dict with portfolio-level context,
                                e.g. ``{"position_weight_pct": 8.0}``.

        Returns ``None`` when the call fails.
        """
        context_parts = [f"Stock: {ticker}"]

        if composite_signal:
            context_parts.append(
                f"Composite signal: {composite_signal.signal} "
                f"(score: {composite_signal.composite_score:+.2f}, "
                f"confidence: {composite_signal.confidence:.0%})"
            )

            if composite_signal.technical:
                t = composite_signal.technical
                rsi = t.indicators.get("rsi", "N/A")
                context_parts.append(
                    f"Technical: RSI={rsi}, signal={t.signal}, score={t.score:+.2f}"
                )

            if composite_signal.fundamental:
                f_ = composite_signal.fundamental
                context_parts.append(f"Fundamental: {f_.reasoning}")

            if composite_signal.sentiment:
                s = composite_signal.sentiment
                context_parts.append(f"Sentiment: {s.reasoning}")

            if composite_signal.disagreeing_signals:
                context_parts.append(
                    "Signal disagreement: "
                    + ", ".join(composite_signal.disagreeing_signals)
                    + " conflict with composite direction"
                )

        if portfolio_context:
            weight = portfolio_context.get("position_weight_pct", 0)
            context_parts.append(f"Current portfolio weight: {weight:.1f}%")

        context = "\n".join(context_parts)

        user_prompt = (
            f"Assess the investment risk for {ticker} given the following analysis:\n\n"
            f"{context}\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "risk_level": <"low" | "medium" | "high" | "very_high">,\n'
            '  "risk_score": <float 0.0–1.0>,\n'
            '  "primary_risks": [<2–4 specific, named risk factors>],\n'
            '  "regulatory_risk": <"low" | "medium" | "high">,\n'
            '  "competitive_risk": <"low" | "medium" | "high">,\n'
            '  "macro_risk": <"low" | "medium" | "high">,\n'
            '  "execution_risk": <"low" | "medium" | "high">,\n'
            '  "mitigation_factors": [<1–3 factors that reduce risk>],\n'
            '  "summary": <2–3 sentence risk narrative>\n'
            "}\n\n"
            "Return only the JSON object."
        )

        try:
            raw = self._client.chat_simple(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=1024,
            )

            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if m:
                raw = m.group(1)

            data = json.loads(raw)

            return RiskAssessment(
                ticker=ticker,
                risk_level=data.get("risk_level", "medium"),
                risk_score=float(data.get("risk_score", 0.5)),
                primary_risks=data.get("primary_risks", []),
                regulatory_risk=data.get("regulatory_risk", "medium"),
                competitive_risk=data.get("competitive_risk", "medium"),
                macro_risk=data.get("macro_risk", "medium"),
                execution_risk=data.get("execution_risk", "low"),
                mitigation_factors=data.get("mitigation_factors", []),
                summary=data.get("summary", ""),
            )

        except Exception as e:
            print(f"[LLMRiskAssessor] Assessment failed for {ticker}: {e}")
            return None
