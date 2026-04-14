"""
Investment Thesis Generator
============================
Uses a BaseLLMClient (any provider) to synthesise technical, fundamental,
and sentiment signals into a coherent, human-readable investment thesis.

This is the core LLM feature that elevates StockAgent from a rule-based
scoring system to a genuine AI investment agent.
"""

import json
import re
from typing import List, Optional, Any
from dataclasses import dataclass, field

from core.llm.base import BaseLLMClient


_SYSTEM_PROMPT = """You are a senior equity research analyst at a quantitative hedge fund.
Your role is to synthesise multi-dimensional quantitative signals into a clear,
actionable investment thesis for the portfolio manager.

Standards:
- Integrate technical, fundamental, and sentiment evidence coherently
- Articulate the primary bull case and primary bear case with equal rigor
- Identify specific, named risk factors — not generic disclaimers
- Propose a signal (strong_buy → strong_sell) that logically follows from the evidence
- Assign conviction honestly: 0.3 for uncertain, 0.7+ for high-confidence setups
- Write like a professional analyst: precise, evidence-based, no hype

If signals conflict, lower conviction and explain the disagreement."""


@dataclass
class InvestmentThesis:
    """
    Comprehensive investment thesis produced by ThesisGenerator.

    Attributes:
        ticker:             Stock ticker symbol.
        signal:             Recommended action (strong_buy / buy / hold / sell / strong_sell).
        conviction:         Confidence level 0–1 (0 = no view, 1 = very high conviction).
        thesis_summary:     2–3 sentence executive summary.
        bull_case:          Arguments for buying / holding.
        bear_case:          Arguments against or for selling.
        key_risks:          Specific downside risk factors.
        catalysts:          Near-term positive catalysts.
        fair_value_comment: Qualitative valuation comment.
    """

    ticker: str
    signal: str                         # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    conviction: float                   # 0.0 – 1.0

    thesis_summary: str = ""
    bull_case: str = ""
    bear_case: str = ""
    key_risks: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    fair_value_comment: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class ThesisGenerator:
    """
    Generates investment theses using an LLM (any provider via BaseLLMClient).

    Usage::

        from core.llm import create_llm_client
        client = create_llm_client("anthropic", "claude-opus-4-6")
        gen = ThesisGenerator(client)
        thesis = gen.generate(
            ticker="AAPL",
            technical_signal=tech_signal,
            fundamental_signal=fund_signal,
            sentiment_signal=sent_signal,
            price=182.50,
        )
        print(thesis.thesis_summary)
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._client = llm_client

    def generate(
        self,
        ticker: str,
        technical_signal: Any = None,
        fundamental_signal: Any = None,
        sentiment_signal: Any = None,
        price: float = None,
    ) -> Optional[InvestmentThesis]:
        """
        Generate an investment thesis for *ticker*.

        Returns ``None`` when the call fails.
        """
        context_parts = [f"Stock: {ticker}"]

        if price:
            context_parts.append(f"Current Price: ${price:.2f}")

        if technical_signal:
            t = technical_signal
            rsi = t.indicators.get("rsi", "N/A")
            bb = t.indicators.get("bb_pct_b", "N/A")
            vol_ratio = t.indicators.get("volume_ratio", "N/A")
            context_parts.append(
                f"\nTechnical Analysis (composite score: {t.score:+.2f}, signal: {t.signal}):"
                f"\n  RSI: {rsi}"
                f"\n  Bollinger %B: {bb}"
                f"\n  Volume ratio: {vol_ratio}"
                f"\n  MA score: {t.ma_score:+.2f} | "
                f"RSI score: {t.rsi_score:+.2f} | "
                f"MACD score: {t.macd_score:+.2f}"
            )

        if fundamental_signal:
            f_ = fundamental_signal
            context_parts.append(
                f"\nFundamental Analysis (composite score: {f_.score:+.2f}, signal: {f_.signal}):"
                f"\n  {f_.reasoning}"
                f"\n  Valuation: {f_.valuation_score:+.2f} | "
                f"Growth: {f_.growth_score:+.2f} | "
                f"Profitability: {f_.profitability_score:+.2f} | "
                f"Health: {f_.health_score:+.2f}"
            )

        if sentiment_signal:
            s = sentiment_signal
            context_parts.append(
                f"\nSentiment Analysis (composite score: {s.score:+.2f}, signal: {s.signal}):"
                f"\n  {s.reasoning}"
                f"\n  Articles analysed: {s.news_volume}"
            )

        context = "\n".join(context_parts)

        user_prompt = (
            "Based on the following multi-dimensional quantitative analysis, "
            f"generate a professional investment thesis for {ticker}.\n\n"
            f"{context}\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "signal": <"strong_buy" | "buy" | "hold" | "sell" | "strong_sell">,\n'
            '  "conviction": <float 0.0–1.0>,\n'
            '  "thesis_summary": <2–3 sentence executive summary>,\n'
            '  "bull_case": <paragraph of bull case arguments>,\n'
            '  "bear_case": <paragraph of bear case arguments>,\n'
            '  "key_risks": [<list of 2–4 specific risk factors>],\n'
            '  "catalysts": [<list of 2–3 near-term positive catalysts>],\n'
            '  "fair_value_comment": <qualitative valuation comment>\n'
            "}\n\n"
            "Return only the JSON object."
        )

        try:
            raw = self._client.chat_simple(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=2048,
            )

            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if m:
                raw = m.group(1)

            data = json.loads(raw)

            return InvestmentThesis(
                ticker=ticker,
                signal=data.get("signal", "hold"),
                conviction=float(data.get("conviction", 0.5)),
                thesis_summary=data.get("thesis_summary", ""),
                bull_case=data.get("bull_case", ""),
                bear_case=data.get("bear_case", ""),
                key_risks=data.get("key_risks", []),
                catalysts=data.get("catalysts", []),
                fair_value_comment=data.get("fair_value_comment", ""),
            )

        except Exception as e:
            print(f"[ThesisGenerator] Generation failed for {ticker}: {e}")
            return None
