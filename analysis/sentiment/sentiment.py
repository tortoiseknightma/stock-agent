"""
Sentiment Analysis Module
==========================
Analyzes news, social media, and market sentiment.

Sources:
- News article sentiment (rule-based or LLM-augmented)
- Put/Call ratio (market fear gauge)
- VIX (volatility index)
- Analyst consensus
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from data.sources.news import NewsItem


@dataclass
class SentimentSignal:
    """Sentiment analysis result."""
    ticker: str
    score: float              # -1 (bearish) to 1 (bullish)
    signal: str
    
    news_sentiment: float = 0.0
    news_volume: int = 0
    avg_sentiment: float = 0.0
    
    # Market-wide sentiment
    vix_level: Optional[float] = None
    vix_signal: str = ""
    
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return self.__dict__


class SentimentAnalyzer:
    """
    Sentiment analysis engine.
    
    Combines:
    1. News sentiment from multiple articles
    2. News volume (high volume + negative = bearish)
    3. Market-wide fear indicators (VIX)
    """
    
    def analyze(self, news_items: List[NewsItem],
                vix_level: float = None) -> SentimentSignal:
        """
        Analyze sentiment from news and market indicators.
        """
        # News sentiment
        sentiments = [n.sentiment for n in news_items if n.sentiment is not None]
        avg_news = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Volume-weighted sentiment (recent articles matter more)
        if news_items:
            weighted_sum = 0
            weight_total = 0
            for i, item in enumerate(news_items):
                if item.sentiment is not None:
                    recency_weight = 1 / (i + 1)  # First article (most recent) gets highest weight
                    weighted_sum += item.sentiment * recency_weight
                    weight_total += recency_weight
            
            weighted_sentiment = weighted_sum / weight_total if weight_total > 0 else 0
        else:
            weighted_sentiment = 0
        
        # VIX analysis
        vix_signal = ""
        vix_score = 0
        if vix_level:
            if vix_level > 30:
                vix_signal = "high_fear"
                vix_score = -0.5  # High VIX = market fear, contrarian buy
            elif vix_level > 20:
                vix_signal = "elevated"
                vix_score = -0.2
            elif vix_level < 15:
                vix_signal = "low_vol"
                vix_score = 0.2   # Complacency
            else:
                vix_signal = "normal"
                vix_score = 0
        
        # News volume effect
        # Negative news with high volume is more impactful
        volume_multiplier = min(len(news_items) / 10, 1.5)  # Cap at 1.5x
        
        # Composite score
        news_component = weighted_sentiment * 0.6
        vix_component = vix_score * 0.4
        
        composite = news_component + vix_component
        
        # Adjust for news volume
        if len(news_items) > 5 and abs(weighted_sentiment) > 0.3:
            composite *= 1.2  # Amplify signal when there's strong consensus
        
        composite = max(-1, min(1, composite))
        
        # Build reasoning
        reasons = []
        if abs(weighted_sentiment) > 0.2:
            direction = "positive" if weighted_sentiment > 0 else "negative"
            reasons.append(f"{direction} news flow ({len(news_items)} articles)")
        
        if vix_signal:
            reasons.append(f"VIX {vix_level:.1f} ({vix_signal})")
        
        reasoning = f"Sentiment: {'; '.join(reasons) if reasons else 'neutral'}"
        
        return SentimentSignal(
            ticker="",
            score=round(composite, 3),
            signal=self._score_to_signal(composite),
            news_sentiment=round(weighted_sentiment, 3),
            news_volume=len(news_items),
            avg_sentiment=round(avg_news, 3),
            vix_level=vix_level,
            vix_signal=vix_signal,
            reasoning=reasoning,
        )
    
    def analyze_portfolio_sentiment(self, ticker_news: Dict[str, List[NewsItem]]) -> Dict[str, SentimentSignal]:
        """Analyze sentiment for multiple tickers."""
        return {ticker: self.analyze(news) for ticker, news in ticker_news.items()}
    
    @staticmethod
    def _score_to_signal(score: float) -> str:
        if score >= 0.5:
            return "strong_buy"
        elif score >= 0.2:
            return "buy"
        elif score <= -0.5:
            return "strong_sell"
        elif score <= -0.2:
            return "sell"
        else:
            return "hold"
