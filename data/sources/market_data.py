"""
Market Data Provider
=====================
Fetches historical and real-time market data using yfinance.

Free, no API key required. Provides:
- Historical OHLCV data
- Real-time quotes (15-min delayed)
- Options chains
- Basic company info
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Quote:
    """Real-time quote data."""
    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    avg_volume: int
    market_cap: float
    pe_ratio: Optional[float]
    week_52_high: float
    week_52_low: float
    timestamp: float


class MarketDataProvider:
    """
    Market data from Yahoo Finance (via yfinance).
    
    Usage:
        provider = MarketDataProvider()
        
        # Get historical data as DataFrame
        history = provider.get_history("AAPL", period="6mo", interval="1d")
        
        # Get current quote
        quote = provider.get_quote("AAPL")
        
        # Get multiple quotes
        quotes = provider.get_quotes(["AAPL", "MSFT", "GOOGL"])
    """
    
    def __init__(self):
        self._yf = None
    
    def _ensure_yf(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
    
    def get_history(self, ticker: str, period: str = "6mo",
                    interval: str = "1d") -> List[Dict[str, Any]]:
        """
        Get historical OHLCV data.
        
        Args:
            ticker: Stock symbol
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval)
            
            if df.empty:
                return []
            
            result = []
            for date, row in df.iterrows():
                result.append({
                    "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": int(row["Volume"]),
                })
            
            return result
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            return []
    
    def get_quote(self, ticker: str) -> Optional[Quote]:
        """Get current quote for a single ticker."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            info = tk.info
            
            if not info or "regularMarketPrice" not in info:
                return None
            
            price = info.get("regularMarketPrice", 0)
            prev_close = info.get("regularMarketPreviousClose", price)
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            return Quote(
                ticker=ticker,
                price=round(price, 2),
                change=round(change, 2),
                change_pct=round(change_pct, 2),
                volume=info.get("regularMarketVolume", 0),
                avg_volume=info.get("averageDailyVolume10Day", 0),
                market_cap=info.get("marketCap", 0),
                pe_ratio=info.get("trailingPE"),
                week_52_high=info.get("fiftyTwoWeekHigh", 0),
                week_52_low=info.get("fiftyTwoWeekLow", 0),
                timestamp=time.time(),
            )
        except Exception as e:
            print(f"Error fetching quote for {ticker}: {e}")
            return None
    
    def get_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple tickers."""
        return {t: q for t in tickers if (q := self.get_quote(t))}
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company fundamentals from Yahoo Finance."""
        self._ensure_yf()
        
        try:
            tk = self._yf.Ticker(ticker)
            info = tk.info
            
            return {
                "ticker": ticker,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "free_cash_flow": info.get("freeCashflow"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "50d_avg": info.get("fiftyDayAverage"),
                "200d_avg": info.get("twoHundredDayAverage"),
            }
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_multiple_histories(self, tickers: List[str],
                                period: str = "6mo") -> Dict[str, List[Dict]]:
        """Get historical data for multiple tickers."""
        return {t: self.get_history(t, period) for t in tickers}
