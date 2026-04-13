"""
SEC Filings Provider
=====================
Fetches SEC EDGAR filings for fundamental analysis.

Free, no API key required. Provides:
- 10-K (annual reports)
- 10-Q (quarterly reports)
- 8-K (material events)
- Insider trading (Form 4)
- Proxy statements (DEF 14A)

Rate limit: 10 requests/sec (SEC requires User-Agent header).
"""

import re
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests


@dataclass
class SECFiling:
    """A single SEC filing."""
    ticker: str
    form_type: str       # 10-K, 10-Q, 8-K, 4, etc.
    filing_date: str
    accession_number: str
    url: str
    description: str = ""
    items: List[str] = None  # For 8-K: which items were reported
    
    def to_dict(self) -> dict:
        return self.__dict__


class SECFilingsProvider:
    """
    SEC EDGAR filings provider.
    
    Usage:
        provider = SECFilingsProvider(user_agent="StockAgent/1.0 (your@email.com)")
        
        # Get recent 10-K and 10-Q filings
        filings = provider.get_filings("AAPL", form_types=["10-K", "10-Q"])
        
        # Get insider trading activity
        insider = provider.get_insider_trading("AAPL")
        
        # Get risk factors from latest 10-K
        risks = provider.get_risk_factors("AAPL")
    """
    
    BASE_URL = "https://efts.sec.gov/LATEST"
    EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
    
    def __init__(self, user_agent: str = "StockAgent/1.0 (stockagent@example.com)"):
        self.headers = {"User-Agent": user_agent}
        self._cache = {}
    
    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """Make rate-limited request to SEC EDGAR."""
        cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        time.sleep(0.11)  # Stay under 10 req/sec
        
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.ok:
                data = resp.json() if 'json' in resp.headers.get('content-type', '') else resp.text
                self._cache[cache_key] = data
                return data
        except Exception as e:
            print(f"SEC EDGAR request failed: {e}")
        
        return None
    
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker symbol."""
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&enddt=2025-12-31&forms=10-K"
        
        # Alternative: use the company tickers JSON
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(tickers_url)
        
        if data and isinstance(data, dict):
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        
        return None
    
    def get_filings(self, ticker: str, form_types: List[str] = None,
                    limit: int = 10) -> List[SECFiling]:
        """
        Get recent SEC filings for a company.
        
        Args:
            ticker: Stock symbol
            form_types: List of form types to fetch (default: ["10-K", "10-Q", "8-K"])
            limit: Maximum number of filings to return
        """
        if form_types is None:
            form_types = ["10-K", "10-Q", "8-K"]
        
        cik = self._get_cik(ticker)
        if not cik:
            return []
        
        # Use EDGAR full-text search
        forms_param = ",".join(form_types)
        url = f"https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{ticker}"',
            "forms": forms_param,
            "dateRange": "custom",
            "startdt": "2023-01-01",
            "enddt": "2030-12-31",
        }
        
        data = self._get(url, params)
        if not data:
            # Fallback: use EDGAR company filings API
            return self._get_filings_fallback(ticker, cik, form_types, limit)
        
        filings = []
        hits = data.get("hits", {}).get("hits", [])
        
        for hit in hits[:limit]:
            source = hit.get("_source", {})
            filings.append(SECFiling(
                ticker=ticker,
                form_type=source.get("form_type", ""),
                filing_date=source.get("file_date", ""),
                accession_number=source.get("accession_no", ""),
                url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{source.get('accession_no', '').replace('-', '')}/",
                description=source.get("display_names", [""])[0] if source.get("display_names") else "",
            ))
        
        return filings
    
    def _get_filings_fallback(self, ticker: str, cik: str,
                               form_types: List[str], limit: int) -> List[SECFiling]:
        """Fallback filing fetch using EDGAR company API."""
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = self._get(url)
        
        if not data:
            return []
        
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        if not recent:
            return []
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        
        for i in range(min(len(forms), 50)):
            if forms[i] in form_types:
                acc = accessions[i].replace("-", "")
                filings.append(SECFiling(
                    ticker=ticker,
                    form_type=forms[i],
                    filing_date=dates[i],
                    accession_number=accessions[i],
                    url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/",
                ))
                if len(filings) >= limit:
                    break
        
        return filings
    
    def get_insider_trading(self, ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent insider trading (Form 4) filings."""
        cik = self._get_cik(ticker)
        if not cik:
            return []
        
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = self._get(url)
        
        if not data:
            return []
        
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])
        
        insider_filings = []
        for i in range(min(len(forms), 100)):
            if forms[i] == "4":
                insider_filings.append({
                    "ticker": ticker,
                    "form": forms[i],
                    "date": dates[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                })
                if len(insider_filings) >= limit:
                    break
        
        return insider_filings
    
    def get_risk_factors(self, ticker: str) -> Optional[str]:
        """
        Extract risk factors section from latest 10-K.
        This is a simplified extraction — full parsing would need XBRL.
        """
        filings = self.get_filings(ticker, form_types=["10-K"], limit=1)
        if not filings:
            return None
        
        filing = filings[0]
        
        # Note: Full text extraction would require downloading and parsing
        # the actual filing document. This returns the filing URL for
        # the agent to reference.
        return f"Risk factors available at: {filing.url}\nFiling date: {filing.filing_date}"
    
    def get_recent_material_events(self, ticker: str, limit: int = 10) -> List[SECFiling]:
        """Get recent 8-K filings (material events)."""
        return self.get_filings(ticker, form_types=["8-K"], limit=limit)
