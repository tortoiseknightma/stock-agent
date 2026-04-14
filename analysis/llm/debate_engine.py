"""
Bull/Bear Debate Engine
========================
Implements a structured adversarial debate between a Bull analyst, a Bear analyst,
and an impartial Judge.  This elevates signal synthesis from a single-analyst
opinion to a multi-perspective deliberation — the core AI-agent differentiator.

Flow
----
1. Bull Opening  — argues for the long thesis (3–5 points)
2. Bear Counter  — refutes Bull and presents the short thesis (3–5 points)
3. Bull Rebuttal — responds to Bear's strongest objections (2–3 points)
4. Bear Rebuttal — responds to Bull's rebuttal (2–3 points)
5. Judge Verdict — weighs the debate and produces a final signal + conviction

All roles are played by the same ``BaseLLMClient`` with different system prompts,
so any configured LLM provider (Claude, GPT-4o, DeepSeek, Ollama…) works.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm.base import BaseLLMClient


# ---------------------------------------------------------------------------
# System prompts — each role has a distinct persona and mandate
# ---------------------------------------------------------------------------

_BULL_SYSTEM = """You are a senior equity analyst at a long-only fund who has just
built a bullish case for a position. Your mandate: identify and articulate the
strongest arguments for buying or holding this stock.

Rules:
- Be specific and evidence-based — cite the numbers in the data provided.
- Anticipate the obvious bear objections and pre-empt them where possible.
- Do NOT hedge excessively; your job is to steelman the bull case.
- Write like a professional: precise, no hype, no filler."""

_BEAR_SYSTEM = """You are a senior short-seller and risk analyst whose mandate is
to stress-test long thesis by finding every flaw.  You are presenting the bear case
for a stock you believe is overvalued or facing headwinds.

Rules:
- Be specific and evidence-based — use the numbers in the data to expose weaknesses.
- Directly refute the bull's specific claims where you can.
- Do NOT manufacture risks; only raise objections grounded in the provided data.
- Write like a professional: precise, no doom-saying, no filler."""

_JUDGE_SYSTEM = """You are the portfolio manager and final decision-maker. You have
observed a structured debate between a Bull analyst and a Bear analyst for a stock.
Your mandate: weigh the arguments objectively and render a final investment verdict.

Rules:
- Identify which side won each key argument and why.
- Your final signal must follow logically from your assessment of the debate.
- Assign conviction honestly: 0.2–0.4 for uncertain, 0.5–0.65 for moderate,
  0.7–0.9 for high-conviction setups. Reserve 0.9+ for exceptional clarity.
- Summarise the key deciding factors in 2–3 sentences.
- Output valid JSON only."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DebateRound:
    """A single round of the debate."""
    role: str           # "bull" | "bear" | "judge"
    round_name: str     # "opening" | "counter" | "rebuttal" | "verdict"
    content: str        # Full text of the argument
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateResult:
    """
    Full result of a Bull/Bear debate session.

    Attributes:
        ticker:         Stock ticker debated.
        rounds:         All debate rounds in order.
        judge_verdict:  Parsed judge verdict text.
        final_signal:   Judge's recommended signal.
        final_conviction: Judge's conviction score 0–1.
        deciding_factors: Key reasons behind the verdict.
        bull_score:     Judge's rating of Bull's case 0–1.
        bear_score:     Judge's rating of Bear's case 0–1.
        debate_summary: One-paragraph summary of the debate.
        timestamp:      When the debate was generated.
    """
    ticker: str
    rounds: List[DebateRound] = field(default_factory=list)

    judge_verdict: str = ""
    final_signal: str = "hold"
    final_conviction: float = 0.5
    deciding_factors: List[str] = field(default_factory=list)
    bull_score: float = 0.5
    bear_score: float = 0.5
    debate_summary: str = ""
    timestamp: float = field(default_factory=time.time)

    def get_round(self, role: str, round_name: str) -> Optional[DebateRound]:
        for r in self.rounds:
            if r.role == role and r.round_name == round_name:
                return r
        return None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "final_signal": self.final_signal,
            "final_conviction": self.final_conviction,
            "bull_score": self.bull_score,
            "bear_score": self.bear_score,
            "debate_summary": self.debate_summary,
            "deciding_factors": self.deciding_factors,
            "rounds": [
                {"role": r.role, "round_name": r.round_name, "content": r.content}
                for r in self.rounds
            ],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DebateEngine:
    """
    Runs a structured Bull/Bear debate to produce an investment verdict.

    Usage::

        from core.llm.factory import create_llm_client
        llm = create_llm_client("anthropic", "claude-opus-4-6")
        engine = DebateEngine(llm)
        result = engine.debate(
            ticker="AAPL",
            technical_signal=tech,
            fundamental_signal=fund,
            sentiment_signal=sent,
        )
        print(result.final_signal, result.final_conviction)
        print(result.debate_summary)

    The debate consists of five LLM calls in sequence; total cost is roughly
    5× the cost of a single thesis-generation call.  Only enable when LLM
    capacity allows (e.g., not for every intraday tick — once per analysis cycle).
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def debate(
        self,
        ticker: str,
        technical_signal: Any = None,
        fundamental_signal: Any = None,
        sentiment_signal: Any = None,
        price: float = None,
        context: str = "",
    ) -> Optional["DebateResult"]:
        """
        Run the full Bull/Bear debate for *ticker*.

        Returns ``None`` when any LLM call fails.  The caller should fall
        back to the rule-based ``CompositeSignal`` in that case.
        """
        try:
            data_summary = self._build_data_summary(
                ticker, technical_signal, fundamental_signal,
                sentiment_signal, price, context
            )

            result = DebateResult(ticker=ticker)

            # Round 1: Bull opening
            bull_opening = self._bull_opening(ticker, data_summary)
            result.rounds.append(
                DebateRound("bull", "opening", bull_opening)
            )

            # Round 2: Bear counter
            bear_counter = self._bear_counter(ticker, data_summary, bull_opening)
            result.rounds.append(
                DebateRound("bear", "counter", bear_counter)
            )

            # Round 3: Bull rebuttal
            bull_rebuttal = self._bull_rebuttal(ticker, bear_counter)
            result.rounds.append(
                DebateRound("bull", "rebuttal", bull_rebuttal)
            )

            # Round 4: Bear rebuttal
            bear_rebuttal = self._bear_rebuttal(ticker, bull_rebuttal)
            result.rounds.append(
                DebateRound("bear", "rebuttal", bear_rebuttal)
            )

            # Round 5: Judge verdict
            verdict_data = self._judge_verdict(
                ticker, data_summary, result.rounds
            )
            result.rounds.append(
                DebateRound("judge", "verdict", verdict_data.get("raw_text", ""))
            )

            # Populate result from parsed verdict
            result.judge_verdict = verdict_data.get("raw_text", "")
            result.final_signal = verdict_data.get("signal", "hold")
            result.final_conviction = float(verdict_data.get("conviction", 0.5))
            result.deciding_factors = verdict_data.get("deciding_factors", [])
            result.bull_score = float(verdict_data.get("bull_score", 0.5))
            result.bear_score = float(verdict_data.get("bear_score", 0.5))
            result.debate_summary = verdict_data.get("debate_summary", "")

            return result

        except Exception as e:
            print(f"[DebateEngine] Debate failed for {ticker}: {e}")
            return None

    # ------------------------------------------------------------------
    # Private: debate rounds
    # ------------------------------------------------------------------

    def _bull_opening(self, ticker: str, data_summary: str) -> str:
        """Bull analyst presents the long thesis."""
        prompt = (
            f"Present the bull case for {ticker}.\n\n"
            f"Available data:\n{data_summary}\n\n"
            "Structure your argument as 3–5 numbered points. "
            "Be specific — cite the numbers. ~200–300 words."
        )
        return self._client.chat_simple(
            system=_BULL_SYSTEM, user=prompt, max_tokens=600
        )

    def _bear_counter(self, ticker: str, data_summary: str,
                      bull_opening: str) -> str:
        """Bear analyst counters the bull case."""
        prompt = (
            f"You are presenting the bear case for {ticker}.\n\n"
            f"Available data:\n{data_summary}\n\n"
            f"The bull analyst argued:\n\"\"\"\n{bull_opening}\n\"\"\"\n\n"
            "Directly refute their key points AND add 2–3 bear arguments "
            "they did not address. Structure as 3–5 numbered points. ~200–300 words."
        )
        return self._client.chat_simple(
            system=_BEAR_SYSTEM, user=prompt, max_tokens=600
        )

    def _bull_rebuttal(self, ticker: str, bear_counter: str) -> str:
        """Bull analyst responds to the bear counter."""
        prompt = (
            f"The bear analyst countered your {ticker} thesis with:\n"
            f"\"\"\"\n{bear_counter}\n\"\"\"\n\n"
            "Rebut their 2–3 strongest objections. Be concise and precise. "
            "~150–200 words."
        )
        return self._client.chat_simple(
            system=_BULL_SYSTEM, user=prompt, max_tokens=400
        )

    def _bear_rebuttal(self, ticker: str, bull_rebuttal: str) -> str:
        """Bear analyst responds to the bull rebuttal."""
        prompt = (
            f"The bull analyst rebutted your {ticker} counter with:\n"
            f"\"\"\"\n{bull_rebuttal}\n\"\"\"\n\n"
            "Address their 2–3 key rebuttal points. Maintain your bear stance "
            "only where the data supports it. ~150–200 words."
        )
        return self._client.chat_simple(
            system=_BEAR_SYSTEM, user=prompt, max_tokens=400
        )

    def _judge_verdict(
        self, ticker: str, data_summary: str, rounds: List[DebateRound]
    ) -> Dict:
        """Judge weighs the debate and delivers a verdict as JSON."""
        debate_transcript = "\n\n".join(
            f"=== {r.role.upper()} {r.round_name.upper()} ===\n{r.content}"
            for r in rounds
        )

        prompt = (
            f"You have observed the following debate about {ticker}:\n\n"
            f"{debate_transcript}\n\n"
            f"Underlying data:\n{data_summary}\n\n"
            "Render your verdict as a JSON object:\n"
            "{\n"
            '  "signal": <"strong_buy"|"buy"|"hold"|"sell"|"strong_sell">,\n'
            '  "conviction": <float 0.0–1.0>,\n'
            '  "bull_score": <float 0.0–1.0 — how well Bull argued>,\n'
            '  "bear_score": <float 0.0–1.0 — how well Bear argued>,\n'
            '  "deciding_factors": [<2–4 key reasons for your verdict>],\n'
            '  "debate_summary": <2–3 sentence summary of the debate and verdict>\n'
            "}\n\n"
            "Return only the JSON object."
        )

        raw = self._client.chat_simple(
            system=_JUDGE_SYSTEM, user=prompt, max_tokens=800
        )

        # Extract JSON
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        json_str = m.group(1) if m else raw
        try:
            data = json.loads(json_str)
            data["raw_text"] = raw
            return data
        except json.JSONDecodeError:
            # Fallback: return partial data so the result is still useful
            return {
                "signal": "hold",
                "conviction": 0.5,
                "bull_score": 0.5,
                "bear_score": 0.5,
                "deciding_factors": ["JSON parse failed — debate recorded but verdict unstructured"],
                "debate_summary": raw[:500],
                "raw_text": raw,
            }

    # ------------------------------------------------------------------
    # Private: data summary builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_data_summary(
        ticker: str,
        technical_signal: Any = None,
        fundamental_signal: Any = None,
        sentiment_signal: Any = None,
        price: float = None,
        context: str = "",
    ) -> str:
        parts = [f"Ticker: {ticker}"]

        if price:
            parts.append(f"Current price: ${price:.2f}")

        if technical_signal:
            t = technical_signal
            ind = t.indicators
            parts.append(
                f"\nTechnical (score {t.score:+.2f}, signal: {t.signal}):"
                f"\n  RSI: {ind.get('rsi', 'N/A')}"
                f"\n  MACD: {ind.get('macd', 'N/A')}"
                f"\n  Bollinger %B: {ind.get('bb_pct_b', 'N/A')}"
                f"\n  Volume ratio: {ind.get('volume_ratio', 'N/A')}"
                f"\n  MA score: {t.ma_score:+.2f} | RSI score: {t.rsi_score:+.2f}"
                f" | MACD score: {t.macd_score:+.2f}"
            )

        if fundamental_signal:
            f_ = fundamental_signal
            parts.append(
                f"\nFundamental (score {f_.score:+.2f}, signal: {f_.signal}):"
                f"\n  {f_.reasoning}"
                f"\n  Valuation: {f_.valuation_score:+.2f}"
                f" | Growth: {f_.growth_score:+.2f}"
                f" | Profitability: {f_.profitability_score:+.2f}"
                f" | Health: {f_.health_score:+.2f}"
            )

        if sentiment_signal:
            s = sentiment_signal
            parts.append(
                f"\nSentiment (score {s.score:+.2f}, signal: {s.signal}):"
                f"\n  {s.reasoning}"
                f"\n  Volume: {s.news_volume} articles"
            )

        if context:
            parts.append(f"\nAdditional context:\n{context}")

        return "\n".join(parts)
