"""
LLM-Powered News Sentiment Analyzer
=====================================
Uses a BaseLLMClient (any provider) to perform deep semantic sentiment
analysis on news articles, replacing keyword-based matching.

Falls back gracefully when no LLM client is provided.
"""

import json
import re
from typing import List, Optional
from dataclasses import dataclass, field

from data.sources.news import NewsItem
from core.llm.base import BaseLLMClient


_SYSTEM_PROMPT = """You are a financial news sentiment analyst specializing in equity markets.
Your task is to analyze news articles about specific stocks and assess their sentiment impact.

Guidelines:
- Evaluate whether each article is genuinely bullish, bearish, or neutral for the company
- Consider magnitude: a minor product update vs. an earnings beat are very different
- Distinguish company-specific news from macro/sector noise
- Be calibrated: most news is neutral; avoid over-reading minor items
- Focus on what matters to investors, not just what sounds dramatic

Always return valid JSON matching the requested schema."""


@dataclass
class ArticleSentiment:
    """Sentiment assessment for a single article."""
    title: str
    score: float                        # -1.0 to 1.0
    reasoning: str
    key_themes: List[str] = field(default_factory=list)


@dataclass
class NewsAnalysisResult:
    """Aggregated result of LLM news analysis for one ticker."""
    ticker: str
    overall_score: float                # -1.0 to 1.0
    overall_signal: str                 # "bullish" | "bearish" | "neutral"
    article_sentiments: List[ArticleSentiment] = field(default_factory=list)
    key_themes: List[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.5


class LLMNewsAnalyzer:
    """
    LLM-powered news sentiment analyzer.

    Sends a batch of news headlines/summaries to an LLM (any provider via
    BaseLLMClient) and returns per-article scores plus an overall sentiment.

    Usage::

        from core.llm import create_llm_client
        client = create_llm_client("anthropic", "claude-opus-4-6")
        analyzer = LLMNewsAnalyzer(client)
        result = analyzer.analyze_batch("AAPL", news_items)
    """

    _MAX_ARTICLES = 10

    def __init__(self, llm_client: BaseLLMClient):
        self._client = llm_client

    def analyze_batch(self, ticker: str,
                      news_items: List[NewsItem]) -> Optional[NewsAnalysisResult]:
        """
        Analyze up to 10 news articles for *ticker* in a single LLM call.

        Returns ``None`` when the call fails — callers should fall back to
        the rule-based sentiment scores already attached to each NewsItem.
        """
        if not news_items:
            return None

        batch = news_items[:self._MAX_ARTICLES]

        articles_text = "\n\n".join(
            f"Article {i + 1}:\n"
            f"Title: {item.title}\n"
            f"Summary: {item.summary or item.title}\n"
            f"Source: {item.source}"
            for i, item in enumerate(batch)
        )

        user_prompt = (
            f"Analyze the following {len(batch)} news articles about {ticker} stock.\n\n"
            f"{articles_text}\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "overall_score": <float -1.0 to 1.0>,\n'
            '  "overall_signal": <"bullish" | "bearish" | "neutral">,\n'
            '  "confidence": <float 0.0 to 1.0>,\n'
            '  "key_themes": [<2-4 key theme strings>],\n'
            '  "summary": <one paragraph overall summary>,\n'
            '  "articles": [\n'
            "    {\n"
            '      "index": <1-based article number>,\n'
            '      "score": <float -1.0 to 1.0>,\n'
            '      "key_themes": [<1-2 themes>],\n'
            '      "reasoning": <one-sentence rationale>\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return only the JSON object, no other text."
        )

        try:
            raw = self._client.chat_simple(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=1024,
            )

            # Strip markdown code fence if present
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if m:
                raw = m.group(1)

            data = json.loads(raw)

            article_sentiments = []
            for art in data.get("articles", []):
                idx = art.get("index", 1) - 1
                if 0 <= idx < len(batch):
                    article_sentiments.append(ArticleSentiment(
                        title=batch[idx].title,
                        score=float(art.get("score", 0.0)),
                        reasoning=art.get("reasoning", ""),
                        key_themes=art.get("key_themes", []),
                    ))

            return NewsAnalysisResult(
                ticker=ticker,
                overall_score=float(data.get("overall_score", 0.0)),
                overall_signal=data.get("overall_signal", "neutral"),
                article_sentiments=article_sentiments,
                key_themes=data.get("key_themes", []),
                summary=data.get("summary", ""),
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception as e:
            print(f"[LLMNewsAnalyzer] Analysis failed for {ticker}: {e}")
            return None
