"""
LLM Analysis Modules
=====================
Claude-powered analysis components for StockAgent.

All modules are optional: they check for ANTHROPIC_API_KEY at instantiation
and fall back gracefully when unavailable.
"""

from .news_analyzer import LLMNewsAnalyzer, NewsAnalysisResult, ArticleSentiment
from .thesis_generator import ThesisGenerator, InvestmentThesis
from .earnings_analyzer import EarningsAnalyzer, EarningsAnalysis
from .risk_assessor import LLMRiskAssessor, RiskAssessment
from .debate_engine import DebateEngine, DebateResult, DebateRound

__all__ = [
    "LLMNewsAnalyzer", "NewsAnalysisResult", "ArticleSentiment",
    "ThesisGenerator", "InvestmentThesis",
    "EarningsAnalyzer", "EarningsAnalysis",
    "LLMRiskAssessor", "RiskAssessment",
    "DebateEngine", "DebateResult", "DebateRound",
]
