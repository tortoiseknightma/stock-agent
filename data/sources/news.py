"""
News & Sentiment Provider
==========================
Aggregates news from multiple free sources:
- Yahoo Finance RSS (via yfinance)
- Finnhub free tier (60 calls/min)
- NewsAPI free tier (100 calls/day)

Also provides basic sentiment scoring.
"""

import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class NewsItem:
    """A single news article."""
    title: str
    summary: str
    source: str
    url: str
    published_at: float
    tickers: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None     # -1 to 1
    relevance: float = 0.0


class NewsProvider:
    """
    Multi-source news aggregator.
    
    Usage:
        news = NewsProvider()
        
        # Get news for a specific stock
        articles = news.get_ticker_news("AAPL", limit=10)
        
        # Get market-wide news
        market_news = news.get_market_news(limit=20)
    """
    
    def __init__(self, finnhub_key: str = None, newsapi_key: str = None):
        self.finnhub_key = finnhub_key
        self.newsapi_key = newsapi_key
        self._yf = None
    
    def _ensure_yf(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
    
    def get_ticker_news(self, ticker: str, limit: int = 10) -> List[NewsItem]:
        """Get news articles for a specific ticker."""
        articles = []
        
        # Source 1: Yahoo Finance
        try:
            self._ensure_yf()
            tk = self._yf.Ticker(ticker)
            yf_news = tk.news or []
            
            for item in yf_news[:limit]:
                pub_time = item.get("providerPublishTime", time.time())
                articles.append(NewsItem(
                    title=item.get("title", ""),
                    summary=item.get("title", ""),  # yfinance news often has no summary
                    source=item.get("publisher", "Yahoo Finance"),
                    url=item.get("link", ""),
                    published_at=pub_time if isinstance(pub_time, (int, float)) else time.time(),
                    tickers=[ticker],
                ))
        except Exception as e:
            print(f"Yahoo news error for {ticker}: {e}")
        
        # Source 2: Finnhub (if key available)
        if self.finnhub_key:
            try:
                import requests
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                
                resp = requests.get(
                    f"https://finnhub.io/api/v1/company-news",
                    params={"symbol": ticker, "from": start_date, "to": end_date,
                            "token": self.finnhub_key},
                    timeout=10
                )
                
                if resp.ok:
                    for item in resp.json()[:limit]:
                        articles.append(NewsItem(
                            title=item.get("headline", ""),
                            summary=item.get("summary", ""),
                            source=item.get("source", "Finnhub"),
                            url=item.get("url", ""),
                            published_at=item.get("datetime", time.time()),
                            tickers=[ticker],
                        ))
            except Exception as e:
                print(f"Finnhub news error for {ticker}: {e}")
        
        # Deduplicate by title similarity
        seen_titles = set()
        unique = []
        for a in articles:
            key = re.sub(r'\W+', '', a.title.lower())[:50]
            if key not in seen_titles:
                seen_titles.add(key)
                # Basic sentiment from title keywords
                a.sentiment = self._basic_sentiment(a.title + " " + a.summary)
                unique.append(a)
        
        # Sort by recency
        unique.sort(key=lambda x: x.published_at, reverse=True)
        return unique[:limit]
    
    def get_market_news(self, limit: int = 20) -> List[NewsItem]:
        """Get general market news."""
        articles = []
        
        # Yahoo Finance general news
        try:
            self._ensure_yf()
            # Use SPY as proxy for market news
            tk = self._yf.Ticker("SPY")
            yf_news = tk.news or []
            
            for item in yf_news[:limit]:
                pub_time = item.get("providerPublishTime", time.time())
                articles.append(NewsItem(
                    title=item.get("title", ""),
                    summary=item.get("title", ""),
                    source=item.get("publisher", "Yahoo Finance"),
                    url=item.get("link", ""),
                    published_at=pub_time if isinstance(pub_time, (int, float)) else time.time(),
                    tickers=[],
                ))
        except Exception:
            pass
        
        # Finnhub general news
        if self.finnhub_key:
            try:
                import requests
                resp = requests.get(
                    "https://finnhub.io/api/v1/news",
                    params={"category": "general", "token": self.finnhub_key},
                    timeout=10
                )
                if resp.ok:
                    for item in resp.json()[:limit]:
                        articles.append(NewsItem(
                            title=item.get("headline", ""),
                            summary=item.get("summary", ""),
                            source=item.get("source", "Finnhub"),
                            url=item.get("url", ""),
                            published_at=item.get("datetime", time.time()),
                            tickers=[],
                        ))
            except Exception:
                pass
        
        articles.sort(key=lambda x: x.published_at, reverse=True)
        return articles[:limit]
    
    def get_sector_news(self, sector: str, limit: int = 10) -> List[NewsItem]:
        """Get news related to a sector."""
        # Use sector ETF as proxy
        sector_etfs = {
            "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
            "Energy": "XLE", "Consumer": "XLY", "Industrial": "XLI",
            "Utilities": "XLU", "Real Estate": "XLRE", "Materials": "XLB",
            "Communication": "XLC",
        }
        
        etf = sector_etfs.get(sector, "SPY")
        return self.get_ticker_news(etf, limit)
    
    def aggregate_sentiment(self, articles: List[NewsItem]) -> float:
        """
        Aggregate sentiment across multiple articles.
        Returns a score from -1 (very bearish) to 1 (very bullish).
        """
        if not articles:
            return 0.0
        
        sentiments = [a.sentiment for a in articles if a.sentiment is not None]
        if not sentiments:
            return 0.0
        
        return sum(sentiments) / len(sentiments)
    
    @staticmethod
    def _basic_sentiment(text: str) -> float:
        """
        Rule-based sentiment scoring.
        Returns -1 to 1. Not as good as an LLM but fast and free.
        """
        text_lower = text.lower()
        
        positive_words = {
            'beat', 'beats', 'exceeds', 'surge', 'surges', 'surged', 'rally',
            'rallies', 'gains', 'gain', 'rise', 'rises', 'rising', 'bull',
            'bullish', 'upgrade', 'upgraded', 'outperform', 'buy', 'strong',
            'growth', 'record', 'high', 'higher', 'positive', 'optimistic',
            'breakthrough', 'innovative', 'profit', 'profitable', 'dividend',
            'buyback', 'expansion', 'partnership', 'deal', 'approval',
            'breakthrough', 'momentum', 'opportunity', 'recovery',
        }
        
        negative_words = {
            'miss', 'misses', 'missed', 'fall', 'falls', 'falling', 'fell',
            'drop', 'drops', 'dropped', 'decline', 'declines', 'bear', 'bearish',
            'downgrade', 'downgraded', 'underperform', 'sell', 'weak', 'weakness',
            'loss', 'losses', 'debt', 'bankruptcy', 'layoff', 'layoffs',
            'lawsuit', 'investigation', 'fraud', 'scandal', 'risk', 'warning',
            'concern', 'concerns', 'negative', 'pessimistic', 'recession',
            'crash', 'crisis', 'restructuring', 'cuts', 'uncertainty',
        }
        
        words = set(re.findall(r'\w+', text_lower))
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return round((pos_count - neg_count) / total, 2)
