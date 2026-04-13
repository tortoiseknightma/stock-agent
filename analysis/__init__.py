"""
StockAgent Analysis Engine
==========================
Multi-dimensional stock analysis combining:
1. Technical Analysis (price patterns, indicators)
2. Fundamental Analysis (financials, valuation)
3. Sentiment Analysis (news, social)
4. Momentum Analysis (relative strength, trends)

Each analyzer produces a score from -1 to 1.
The composite engine combines them using configurable weights.
"""

from .technical import TechnicalAnalyzer, TechnicalSignal
from .fundamental import FundamentalAnalyzer, FundamentalSignal
from .sentiment import SentimentAnalyzer, SentimentSignal
from .composite import CompositeAnalyzer, CompositeSignal

__all__ = [
    "TechnicalAnalyzer", "TechnicalSignal",
    "FundamentalAnalyzer", "FundamentalSignal",
    "SentimentAnalyzer", "SentimentSignal",
    "CompositeAnalyzer", "CompositeSignal",
]
