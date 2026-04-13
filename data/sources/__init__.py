"""
StockAgent Data Sources
=======================
Unified data acquisition layer. Each source is a class that knows how to
fetch data from a specific provider (yfinance, Finnhub, SEC EDGAR, etc.).

All sources return data in normalized formats for downstream consumption.
"""

from .market_data import MarketDataProvider
from .news import NewsProvider
from .fundamentals import FundamentalsProvider
from .sec_filings import SECFilingsProvider

__all__ = [
    "MarketDataProvider", "NewsProvider",
    "FundamentalsProvider", "SECFilingsProvider",
]
