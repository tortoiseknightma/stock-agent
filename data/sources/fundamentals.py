"""
Fundamentals Provider
======================
Fetches fundamental financial data for stock analysis.

Data sources:
- Yahoo Finance (via yfinance): income statements, balance sheets, cash flow
- Calculated ratios: P/E, P/B, ROE, debt/equity, etc.

All data is normalized into a standard format.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class FinancialRatios:
    """Standardized financial ratios."""
    ticker: str
    
    # Valuation
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    fcf_growth: Optional[float] = None
    
    # Financial Health
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # Efficiency
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    
    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class FundamentalsProvider:
    """
    Fundamental data provider using Yahoo Finance.
    
    Usage:
        provider = FundamentalsProvider()
        
        # Get comprehensive ratios
        ratios = provider.get_ratios("AAPL")
        
        # Get income statement
        income = provider.get_income_statement("AAPL")
        
        # Get earnings history
        earnings = provider.get_earnings("AAPL")
    """
    
    def __init__(self):
        self._yf = None
    
    def _ensure_yf(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
    
    def get_ratios(self, ticker: str) -> FinancialRatios:
        """Calculate standard financial ratios."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            info = tk.info
            
            return FinancialRatios(
                ticker=ticker,
                pe_trailing=info.get("trailingPE"),
                pe_forward=info.get("forwardPE"),
                peg_ratio=info.get("pegRatio"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                pb_ratio=info.get("priceToBook"),
                ev_ebitda=info.get("enterpriseToEbitda"),
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                net_margin=info.get("profitMargins"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                revenue_growth=info.get("revenueGrowth"),
                earnings_growth=info.get("earningsGrowth"),
                current_ratio=info.get("currentRatio"),
                debt_to_equity=info.get("debtToEquity"),
                dividend_yield=info.get("dividendYield"),
                payout_ratio=info.get("payoutRatio"),
            )
        except Exception as e:
            print(f"Error calculating ratios for {ticker}: {e}")
            return FinancialRatios(ticker=ticker)
    
    def get_income_statement(self, ticker: str) -> Dict[str, Any]:
        """Get income statement data."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            income = tk.income_stmt
            
            if income is None or income.empty:
                return {"ticker": ticker, "error": "No data"}
            
            # Get most recent quarter
            latest = income.iloc[:, 0]
            
            return {
                "ticker": ticker,
                "period": str(income.columns[0]),
                "total_revenue": float(latest.get("Total Revenue", 0)),
                "cost_of_revenue": float(latest.get("Cost Of Revenue", 0)),
                "gross_profit": float(latest.get("Gross Profit", 0)),
                "operating_income": float(latest.get("Operating Income", 0)),
                "net_income": float(latest.get("Net Income", 0)),
                "ebitda": float(latest.get("EBITDA", 0)),
                "diluted_eps": float(latest.get("Diluted EPS", 0)),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def get_balance_sheet(self, ticker: str) -> Dict[str, Any]:
        """Get balance sheet data."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            bs = tk.balance_sheet
            
            if bs is None or bs.empty:
                return {"ticker": ticker, "error": "No data"}
            
            latest = bs.iloc[:, 0]
            
            return {
                "ticker": ticker,
                "period": str(bs.columns[0]),
                "total_assets": float(latest.get("Total Assets", 0)),
                "total_liabilities": float(latest.get("Total Liabilities Net Minority Interest", 0)),
                "total_equity": float(latest.get("Stockholders Equity", 0)),
                "cash": float(latest.get("Cash And Cash Equivalents", 0)),
                "short_term_investments": float(latest.get("Other Short Term Investments", 0)),
                "total_debt": float(latest.get("Total Debt", 0)),
                "net_debt": float(latest.get("Net Debt", 0)),
                "working_capital": float(latest.get("Working Capital", 0)),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def get_cash_flow(self, ticker: str) -> Dict[str, Any]:
        """Get cash flow statement data."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            cf = tk.cashflow
            
            if cf is None or cf.empty:
                return {"ticker": ticker, "error": "No data"}
            
            latest = cf.iloc[:, 0]
            
            return {
                "ticker": ticker,
                "period": str(cf.columns[0]),
                "operating_cash_flow": float(latest.get("Operating Cash Flow", 0)),
                "capital_expenditure": float(latest.get("Capital Expenditure", 0)),
                "free_cash_flow": float(latest.get("Free Cash Flow", 0)),
                "dividends_paid": float(latest.get("Dividend Paid", 0)),
                "share_repurchase": float(latest.get("Repurchase Of Capital Stock", 0)),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def get_earnings(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data and estimates."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            
            earnings = tk.earnings
            earnings_dates = tk.earnings_dates
            earnings_estimate = tk.earnings_estimate
            
            result = {"ticker": ticker}
            
            if earnings is not None and not earnings.empty:
                result["annual_earnings"] = earnings.to_dict()
            
            if earnings_dates is not None and not earnings_dates.empty:
                upcoming = earnings_dates.head(1)
                if not upcoming.empty:
                    result["next_earnings_date"] = str(upcoming.index[0])
                    row = upcoming.iloc[0]
                    result["eps_estimate"] = float(row.get("EPS Estimate", 0)) if "EPS Estimate" in row else None
                    result["reported_eps"] = float(row.get("Reported EPS", 0)) if "Reported EPS" in row else None
                    result["surprise_pct"] = float(row.get("Surprise(%)", 0)) if "Surprise(%)" in row else None
            
            return result
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def score_fundamentals(self, ratios: FinancialRatios) -> float:
        """
        Score fundamentals from -1 (weak) to 1 (strong).
        Based on standard value investing criteria.
        """
        scores = []
        
        # Valuation (lower P/E is better, but not negative)
        if ratios.pe_forward and ratios.pe_forward > 0:
            if ratios.pe_forward < 15:
                scores.append(0.8)   # Cheap
            elif ratios.pe_forward < 25:
                scores.append(0.3)   # Fair
            elif ratios.pe_forward < 40:
                scores.append(-0.3)  # Expensive
            else:
                scores.append(-0.7)  # Very expensive
        
        # Growth (higher is better)
        if ratios.revenue_growth is not None:
            if ratios.revenue_growth > 0.20:
                scores.append(0.8)
            elif ratios.revenue_growth > 0.10:
                scores.append(0.4)
            elif ratios.revenue_growth > 0:
                scores.append(0.1)
            else:
                scores.append(-0.5)
        
        # Profitability (higher margins are better)
        if ratios.net_margin is not None:
            if ratios.net_margin > 0.20:
                scores.append(0.7)
            elif ratios.net_margin > 0.10:
                scores.append(0.3)
            elif ratios.net_margin > 0:
                scores.append(0.0)
            else:
                scores.append(-0.6)
        
        # Return on equity
        if ratios.roe is not None:
            if ratios.roe > 0.20:
                scores.append(0.7)
            elif ratios.roe > 0.10:
                scores.append(0.3)
            else:
                scores.append(-0.3)
        
        # Financial health (debt/equity)
        if ratios.debt_to_equity is not None:
            if ratios.debt_to_equity < 50:
                scores.append(0.5)
            elif ratios.debt_to_equity < 100:
                scores.append(0.0)
            else:
                scores.append(-0.5)
        
        if not scores:
            return 0.0
        
        return round(sum(scores) / len(scores), 2)
