"""
Earnings Call Analyzer
=======================
Uses a BaseLLMClient (any provider) to extract actionable insights from
earnings call transcripts and SEC filings (10-Q, 10-K, 8-K).

Surfaces: EPS/revenue vs. expectations, management tone, forward guidance,
and key risks that quantitative models miss.
"""

import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from core.llm.base import BaseLLMClient


_SYSTEM_PROMPT = """You are a financial analyst specialising in earnings call analysis.
You extract actionable, investor-relevant insights from earnings transcripts and SEC filings.

Focus on:
1. Actual results vs. prior guidance / consensus expectations
2. Management tone and language — overconfidence or hedging matters
3. Forward guidance: raised / maintained / lowered / withdrawn
4. Key business momentum: new contracts, customer growth, margin trajectory
5. Risks explicitly flagged by management (not boilerplate)
6. Changes from prior quarters that signal trend shifts

Be precise. Cite specific numbers when the transcript contains them.
Avoid generic commentary that could apply to any company."""


@dataclass
class EarningsAnalysis:
    """
    Analysis result from an earnings call transcript.

    Attributes:
        ticker:                 Stock ticker symbol.
        sentiment_score:        -1.0 (very bearish) to 1.0 (very bullish).
        beat_or_miss:           EPS/revenue outcome vs. expectations.
        management_tone:        Inferred tone of management commentary.
        eps_commentary:         Brief summary of EPS discussion.
        revenue_commentary:     Brief summary of revenue discussion.
        guidance_commentary:    Forward guidance summary.
        key_positives:          Bullish highlights from the call.
        key_negatives:          Concerns or disappointments.
        key_risks:              Specific risks management mentioned.
        summary:                2–3 sentence overall assessment.
    """

    ticker: str
    sentiment_score: float          # -1.0 to 1.0
    beat_or_miss: str               # "beat" | "miss" | "in-line" | "unknown"
    management_tone: str            # "optimistic" | "cautious" | "neutral" | "concerned"

    eps_commentary: str = ""
    revenue_commentary: str = ""
    guidance_commentary: str = ""

    key_positives: List[str] = field(default_factory=list)
    key_negatives: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)

    summary: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class EarningsAnalyzer:
    """
    Analyzes earnings call transcripts using an LLM (any provider).

    Long transcripts are automatically trimmed (keeping the prepared remarks
    start and Q&A end) to stay within reasonable token budgets.

    Usage::

        from core.llm import create_llm_client
        client = create_llm_client("anthropic", "claude-opus-4-6")
        analyzer = EarningsAnalyzer(client)
        result = analyzer.analyze("AAPL", transcript_text)
        print(result.beat_or_miss)          # "beat"
        print(result.guidance_commentary)   # "Management raised FY guidance..."
    """

    _MAX_CHARS = 8_000   # ~2K tokens; covers most prepared remarks

    def __init__(self, llm_client: BaseLLMClient):
        self._client = llm_client

    def analyze(
        self,
        ticker: str,
        transcript: str,
        prior_guidance: Optional[Dict[str, Any]] = None,
    ) -> Optional[EarningsAnalysis]:
        """
        Analyze an earnings call transcript for *ticker*.

        Args:
            ticker:          Stock ticker symbol.
            transcript:      Full or partial transcript text.
            prior_guidance:  Optional dict of prior-quarter guidance values
                             (e.g. {"eps": 1.25, "revenue_bn": 89.5}) for
                             beat/miss comparison.

        Returns ``None`` when the call fails.
        """
        if not transcript:
            return None

        # Trim long transcripts: keep the opening and closing halves
        if len(transcript) > self._MAX_CHARS:
            half = self._MAX_CHARS // 2
            transcript = (
                transcript[:half]
                + "\n...[transcript trimmed for length]...\n"
                + transcript[-half:]
            )

        guidance_note = ""
        if prior_guidance:
            items = ", ".join(f"{k}={v}" for k, v in prior_guidance.items())
            guidance_note = f"\nPrior guidance for comparison: {items}\n"

        user_prompt = (
            f"Analyze the following earnings call transcript for {ticker}:{guidance_note}\n\n"
            "---\n"
            f"{transcript}\n"
            "---\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "sentiment_score": <float -1.0 to 1.0>,\n'
            '  "beat_or_miss": <"beat" | "miss" | "in-line" | "unknown">,\n'
            '  "management_tone": <"optimistic" | "cautious" | "neutral" | "concerned">,\n'
            '  "eps_commentary": <brief EPS discussion>,\n'
            '  "revenue_commentary": <brief revenue discussion>,\n'
            '  "guidance_commentary": <forward guidance summary>,\n'
            '  "key_positives": [<2–4 bullish highlights>],\n'
            '  "key_negatives": [<1–3 concerns or disappointments>],\n'
            '  "key_risks": [<1–3 specific risks management mentioned>],\n'
            '  "summary": <2–3 sentence overall assessment>\n'
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

            return EarningsAnalysis(
                ticker=ticker,
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                beat_or_miss=data.get("beat_or_miss", "unknown"),
                management_tone=data.get("management_tone", "neutral"),
                eps_commentary=data.get("eps_commentary", ""),
                revenue_commentary=data.get("revenue_commentary", ""),
                guidance_commentary=data.get("guidance_commentary", ""),
                key_positives=data.get("key_positives", []),
                key_negatives=data.get("key_negatives", []),
                key_risks=data.get("key_risks", []),
                summary=data.get("summary", ""),
            )

        except Exception as e:
            print(f"[EarningsAnalyzer] Analysis failed for {ticker}: {e}")
            return None
