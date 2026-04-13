"""
Fundamental Analysis Module
=============================
Evaluates stocks based on financial health, valuation, and growth metrics.

Produces a score from -1 (fundamentally weak) to 1 (fundamentally strong).
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from data.sources.fundamentals import FinancialRatios


@dataclass
class FundamentalSignal:
    """Fundamental analysis result."""
    ticker: str
    score: float              # -1 to 1
    signal: str               # "strong_buy", "buy", "hold", "sell", "strong_sell"
    
    valuation_score: float    # P/E, P/B relative to peers
    growth_score: float       # Revenue/earnings growth
    profitability_score: float  # Margins, ROE
    health_score: float       # Debt levels, liquidity
    
    ratios: Dict[str, Any] = None
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return self.__dict__


class FundamentalAnalyzer:
    """
    Fundamental analysis engine.
    
    Evaluates stocks on 4 dimensions:
    1. Valuation — Is the stock cheap or expensive?
    2. Growth — Is revenue/earnings growing?
    3. Profitability — How efficient is the business?
    4. Financial health — Can the company survive downturns?
    """
    
    # Sector-specific benchmarks for comparison
    SECTOR_BENCHMARKS = {
        "Technology": {"pe_forward": 25, "revenue_growth": 0.15, "net_margin": 0.20, "roe": 0.20},
        "Healthcare": {"pe_forward": 20, "revenue_growth": 0.10, "net_margin": 0.15, "roe": 0.15},
        "Financials": {"pe_forward": 12, "revenue_growth": 0.05, "net_margin": 0.25, "roe": 0.10},
        "Consumer Cyclical": {"pe_forward": 18, "revenue_growth": 0.10, "net_margin": 0.10, "roe": 0.15},
        "Energy": {"pe_forward": 10, "revenue_growth": 0.05, "net_margin": 0.10, "roe": 0.12},
        "default": {"pe_forward": 18, "revenue_growth": 0.08, "net_margin": 0.12, "roe": 0.12},
    }
    
    def analyze(self, ratios: FinancialRatios, sector: str = "") -> FundamentalSignal:
        """
        Run fundamental analysis.
        
        Args:
            ratios: FinancialRatios object with calculated metrics
            sector: Company's sector for peer comparison
        """
        benchmarks = self.SECTOR_BENCHMARKS.get(sector, self.SECTOR_BENCHMARKS["default"])
        
        # Score each dimension
        val_score = self._score_valuation(ratios, benchmarks)
        growth_score = self._score_growth(ratios, benchmarks)
        profit_score = self._score_profitability(ratios, benchmarks)
        health_score = self._score_health(ratios)
        
        # Weighted composite
        composite = (
            0.30 * val_score +
            0.25 * growth_score +
            0.25 * profit_score +
            0.20 * health_score
        )
        
        # Build reasoning
        reasons = []
        if val_score > 0.3:
            reasons.append("attractive valuation")
        elif val_score < -0.3:
            reasons.append("expensive valuation")
        
        if growth_score > 0.3:
            reasons.append("strong growth")
        elif growth_score < -0.3:
            reasons.append("weak growth")
        
        if profit_score > 0.3:
            reasons.append("high profitability")
        
        if health_score < -0.3:
            reasons.append("balance sheet concerns")
        
        reasoning = f"Fundamentals: {'; '.join(reasons) if reasons else 'mixed signals'}"
        
        return FundamentalSignal(
            ticker=ratios.ticker,
            score=round(composite, 3),
            signal=self._score_to_signal(composite),
            valuation_score=round(val_score, 3),
            growth_score=round(growth_score, 3),
            profitability_score=round(profit_score, 3),
            health_score=round(health_score, 3),
            ratios=ratios.to_dict(),
            reasoning=reasoning,
        )
    
    def _score_valuation(self, ratios: FinancialRatios, benchmarks: dict) -> float:
        """Score valuation attractiveness (lower is better for buyers)."""
        scores = []
        
        # P/E Forward
        if ratios.pe_forward and ratios.pe_forward > 0:
            benchmark = benchmarks["pe_forward"]
            relative = ratios.pe_forward / benchmark
            if relative < 0.7:
                scores.append(0.8)    # Very cheap
            elif relative < 1.0:
                scores.append(0.4)    # Cheap
            elif relative < 1.5:
                scores.append(-0.2)   # Fairly valued
            else:
                scores.append(-0.7)   # Expensive
        
        # PEG Ratio (lower is better, <1 is undervalued)
        if ratios.peg_ratio and ratios.peg_ratio > 0:
            if ratios.peg_ratio < 0.5:
                scores.append(0.9)
            elif ratios.peg_ratio < 1.0:
                scores.append(0.5)
            elif ratios.peg_ratio < 2.0:
                scores.append(0.0)
            else:
                scores.append(-0.5)
        
        # P/B Ratio
        if ratios.pb_ratio and ratios.pb_ratio > 0:
            if ratios.pb_ratio < 1.0:
                scores.append(0.6)    # Below book value
            elif ratios.pb_ratio < 3.0:
                scores.append(0.1)
            else:
                scores.append(-0.3)
        
        return sum(scores) / len(scores) if scores else 0
    
    def _score_growth(self, ratios: FinancialRatios, benchmarks: dict) -> float:
        """Score growth trajectory."""
        scores = []
        
        # Revenue growth
        if ratios.revenue_growth is not None:
            benchmark = benchmarks["revenue_growth"]
            if ratios.revenue_growth > benchmark * 2:
                scores.append(0.9)
            elif ratios.revenue_growth > benchmark:
                scores.append(0.5)
            elif ratios.revenue_growth > 0:
                scores.append(0.1)
            else:
                scores.append(-0.6)
        
        # Earnings growth
        if ratios.earnings_growth is not None:
            if ratios.earnings_growth > 0.25:
                scores.append(0.8)
            elif ratios.earnings_growth > 0.10:
                scores.append(0.4)
            elif ratios.earnings_growth > 0:
                scores.append(0.1)
            else:
                scores.append(-0.5)
        
        return sum(scores) / len(scores) if scores else 0
    
    def _score_profitability(self, ratios: FinancialRatios, benchmarks: dict) -> float:
        """Score profitability metrics."""
        scores = []
        
        # Net margin
        if ratios.net_margin is not None:
            benchmark = benchmarks["net_margin"]
            if ratios.net_margin > benchmark * 1.5:
                scores.append(0.8)
            elif ratios.net_margin > benchmark:
                scores.append(0.4)
            elif ratios.net_margin > 0:
                scores.append(0.0)
            else:
                scores.append(-0.7)
        
        # ROE
        if ratios.roe is not None:
            benchmark = benchmarks["roe"]
            if ratios.roe > benchmark * 2:
                scores.append(0.8)
            elif ratios.roe > benchmark:
                scores.append(0.4)
            elif ratios.roe > 0:
                scores.append(0.0)
            else:
                scores.append(-0.5)
        
        # Operating margin
        if ratios.operating_margin is not None:
            if ratios.operating_margin > 0.25:
                scores.append(0.6)
            elif ratios.operating_margin > 0.15:
                scores.append(0.3)
            elif ratios.operating_margin > 0:
                scores.append(0.0)
            else:
                scores.append(-0.5)
        
        return sum(scores) / len(scores) if scores else 0
    
    def _score_health(self, ratios: FinancialRatios) -> float:
        """Score financial health (debt, liquidity)."""
        scores = []
        
        # Debt to equity
        if ratios.debt_to_equity is not None:
            if ratios.debt_to_equity < 30:
                scores.append(0.7)    # Low debt
            elif ratios.debt_to_equity < 80:
                scores.append(0.3)    # Moderate debt
            elif ratios.debt_to_equity < 150:
                scores.append(-0.2)   # High debt
            else:
                scores.append(-0.7)   # Very high debt
        
        # Current ratio
        if ratios.current_ratio is not None:
            if ratios.current_ratio > 2.0:
                scores.append(0.5)    # Very liquid
            elif ratios.current_ratio > 1.5:
                scores.append(0.3)
            elif ratios.current_ratio > 1.0:
                scores.append(0.0)
            else:
                scores.append(-0.5)   # Liquidity concerns
        
        # Dividend yield (income reliability)
        if ratios.dividend_yield is not None and ratios.dividend_yield > 0:
            if ratios.dividend_yield > 0.04:
                scores.append(0.3)    # High yield
            elif ratios.dividend_yield > 0.02:
                scores.append(0.2)
            else:
                scores.append(0.1)
        
        return sum(scores) / len(scores) if scores else 0
    
    @staticmethod
    def _score_to_signal(score: float) -> str:
        if score >= 0.7:
            return "strong_buy"
        elif score >= 0.4:
            return "buy"
        elif score <= -0.7:
            return "strong_sell"
        elif score <= -0.4:
            return "sell"
        else:
            return "hold"
