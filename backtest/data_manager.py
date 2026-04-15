"""
Historical Data Manager
========================
Fetches OHLCV data from yfinance and caches it locally as CSV files.
Serves date-sliced windows to the backtest engine and broker.

Design:
- One CSV file per ticker: data/backtest_cache/AAPL.csv
- Cache covers the full fetched range; re-fetches if range is insufficient
- get_bars_up_to() returns List[Dict] matching TechnicalAnalyzer.analyze() input format
- No look-ahead: get_bars_up_to(date) only returns bars ≤ date
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class HistoricalDataManager:
    """
    Fetch and cache historical OHLCV data for backtesting.

    Usage::

        dm = HistoricalDataManager()
        dm.load("AAPL", "2022-01-01", "2024-01-01")
        bars = dm.get_bars_up_to("AAPL", datetime.date(2023, 6, 30), lookback=252)
        close = dm.get_close("AAPL", datetime.date(2023, 6, 30))
        dates = dm.get_trading_dates("AAPL", "2023-01-01", "2023-12-31")
    """

    def __init__(self, cache_dir: str = "data/backtest_cache"):
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for backtesting: pip install pandas")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # In-memory store: ticker -> DataFrame indexed by datetime.date
        self._data: Dict[str, "pd.DataFrame"] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, ticker: str, start: str, end: str) -> "pd.DataFrame":
        """
        Load OHLCV data for *ticker* between *start* and *end* (inclusive).

        Checks CSV cache first; fetches from yfinance if the cache does not
        cover the requested range.

        Returns a DataFrame with columns: open, high, low, close, volume,
        indexed by datetime.date.
        """
        ticker = ticker.upper()

        # Try cache
        cached = self._load_from_cache(ticker, start, end)
        if cached is not None:
            self._data[ticker] = cached
            return cached

        # Fetch from yfinance and cache
        df = self._fetch_and_cache(ticker, start, end)
        self._data[ticker] = df
        return df

    def get_bars_up_to(
        self,
        ticker: str,
        date: datetime.date,
        lookback: int = 252,
    ) -> List[Dict[str, Any]]:
        """
        Return up to *lookback* OHLCV bars ending on *date* (inclusive).

        Output format matches MarketDataProvider.get_history() and
        TechnicalAnalyzer.analyze() input: List[Dict] with keys:
        date, open, high, low, close, volume.

        No look-ahead: bars after *date* are never included.
        """
        ticker = ticker.upper()
        df = self._data.get(ticker)
        if df is None or df.empty:
            return []

        subset = df[df.index <= date].tail(lookback)
        return self._df_to_list(subset)

    def get_close(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Return closing price for *ticker* on *date*, or None if not available."""
        ticker = ticker.upper()
        df = self._data.get(ticker)
        if df is None or date not in df.index:
            return None
        return float(df.loc[date, "close"])

    def get_trading_dates(
        self, ticker: str, start: str, end: str
    ) -> List[datetime.date]:
        """
        Return sorted list of trading dates for *ticker* between *start* and *end*.
        Data must have been loaded with load() first.
        """
        ticker = ticker.upper()
        df = self._data.get(ticker)
        if df is None or df.empty:
            return []

        start_d = _parse_date(start)
        end_d = _parse_date(end)
        mask = (df.index >= start_d) & (df.index <= end_d)
        return sorted(df.index[mask].tolist())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.csv"

    def _fetch_and_cache(self, ticker: str, start: str, end: str) -> "pd.DataFrame":
        """Fetch from yfinance, normalise, and save to CSV cache."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required: pip install yfinance")

        df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = self._normalise(df)
        df.to_csv(self._cache_path(ticker))
        return df

    def _load_from_cache(
        self, ticker: str, start: str, end: str
    ) -> Optional["pd.DataFrame"]:
        """
        Load CSV if it exists and covers the full requested range.
        Returns None if cache is missing or insufficient.
        """
        path = self._cache_path(ticker)
        if not path.exists():
            return None

        df = pd.read_csv(path, index_col=0)
        # Explicitly parse index to datetime.date
        df.index = pd.to_datetime(df.index).map(lambda x: x.date())

        if df.empty:
            return None

        start_d = _parse_date(start)
        end_d = _parse_date(end)

        # The cache must cover the requested range.
        # Allow ±7 days tolerance for weekends and holidays at both ends.
        tolerance = datetime.timedelta(days=7)
        if (df.index.max() < end_d - tolerance
                or df.index.min() > start_d + tolerance):
            return None

        return df

    @staticmethod
    def _normalise(df: "pd.DataFrame") -> "pd.DataFrame":
        """Rename yfinance columns, convert index to datetime.date."""
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = df.index.map(lambda x: x.date() if hasattr(x, "date") else x)
        df.index.name = "date"
        df = df.round({"open": 4, "high": 4, "low": 4, "close": 4})
        df["volume"] = df["volume"].astype(int)
        return df

    @staticmethod
    def _df_to_list(df: "pd.DataFrame") -> List[Dict[str, Any]]:
        """Convert DataFrame to List[Dict] matching TechnicalAnalyzer format."""
        rows = []
        for date, row in df.iterrows():
            rows.append({
                "date": str(date),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })
        return rows


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _parse_date(s: str) -> datetime.date:
    """Parse 'YYYY-MM-DD' string to datetime.date."""
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()
