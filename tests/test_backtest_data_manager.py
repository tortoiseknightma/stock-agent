"""Tests for backtest.HistoricalDataManager."""

import datetime
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from backtest.data_manager import HistoricalDataManager, _parse_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv_df(start="2023-01-03", days=260) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with `days` trading rows."""
    dates = pd.bdate_range(start=start, periods=days)
    prices = [150.0 + i * 0.5 for i in range(days)]
    df = pd.DataFrame({
        "open":   [p - 0.5 for p in prices],
        "high":   [p + 1.0 for p in prices],
        "low":    [p - 1.0 for p in prices],
        "close":  prices,
        "volume": [1_000_000] * days,
    }, index=dates)
    df.index = [d.date() for d in df.index]
    df.index.name = "date"
    return df


def make_dm(tmp_path) -> HistoricalDataManager:
    return HistoricalDataManager(cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# _parse_date helper
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_parses_iso_string(self):
        d = _parse_date("2023-06-15")
        assert d == datetime.date(2023, 6, 15)


# ---------------------------------------------------------------------------
# load() — fetching and caching
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_calls_yfinance_when_no_cache(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df.copy()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = dm.load("AAPL", "2023-01-01", "2023-12-31")

        assert not result.empty
        mock_ticker.history.assert_called_once()

    def test_load_writes_csv_cache(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df.copy()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            dm.load("AAPL", "2023-01-01", "2023-12-31")

        cache_file = Path(tmp_path / "cache" / "AAPL.csv")
        assert cache_file.exists()

    def test_load_from_cache_skips_yfinance(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df(start="2022-01-03", days=522)  # ~522 bdays covers 2022-01-03→2023-12-31

        # Prime the cache
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df.copy()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            dm.load("AAPL", "2022-01-01", "2023-12-31")

        call_count_before = mock_ticker.history.call_count

        # Second load should hit cache, not yfinance
        dm2 = make_dm(tmp_path)
        with patch("yfinance.Ticker", return_value=mock_ticker):
            dm2.load("AAPL", "2022-01-01", "2023-12-31")

        assert mock_ticker.history.call_count == call_count_before

    def test_load_returns_dataframe_with_required_columns(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()

        with patch("yfinance.Ticker", return_value=MagicMock(history=MagicMock(return_value=df))):
            result = dm.load("AAPL", "2023-01-01", "2023-12-31")

        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# get_bars_up_to()
# ---------------------------------------------------------------------------

class TestGetBarsUpTo:
    def _load(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()
        with patch("yfinance.Ticker", return_value=MagicMock(history=MagicMock(return_value=df))):
            dm.load("AAPL", "2023-01-01", "2023-12-31")
        return dm

    def test_returns_list_of_dicts(self, tmp_path):
        dm = self._load(tmp_path)
        bars = dm.get_bars_up_to("AAPL", datetime.date(2023, 6, 30))
        assert isinstance(bars, list)
        assert isinstance(bars[0], dict)

    def test_dict_has_required_keys(self, tmp_path):
        dm = self._load(tmp_path)
        bars = dm.get_bars_up_to("AAPL", datetime.date(2023, 6, 30))
        required = {"date", "open", "high", "low", "close", "volume"}
        assert required.issubset(set(bars[0].keys()))

    def test_no_lookahead_bars_all_before_or_on_date(self, tmp_path):
        dm = self._load(tmp_path)
        cutoff = datetime.date(2023, 3, 31)
        bars = dm.get_bars_up_to("AAPL", cutoff)
        for b in bars:
            assert b["date"] <= str(cutoff)

    def test_lookback_limits_result_length(self, tmp_path):
        dm = self._load(tmp_path)
        bars = dm.get_bars_up_to("AAPL", datetime.date(2023, 12, 29), lookback=20)
        assert len(bars) <= 20

    def test_returns_empty_for_unknown_ticker(self, tmp_path):
        dm = make_dm(tmp_path)
        bars = dm.get_bars_up_to("ZZZZ", datetime.date(2023, 6, 30))
        assert bars == []


# ---------------------------------------------------------------------------
# get_close()
# ---------------------------------------------------------------------------

class TestGetClose:
    def test_returns_float(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()
        first_date = df.index[0]
        with patch("yfinance.Ticker", return_value=MagicMock(history=MagicMock(return_value=df))):
            dm.load("AAPL", "2023-01-01", "2023-12-31")
        close = dm.get_close("AAPL", first_date)
        assert isinstance(close, float)
        assert close > 0

    def test_returns_none_for_non_trading_day(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()
        with patch("yfinance.Ticker", return_value=MagicMock(history=MagicMock(return_value=df))):
            dm.load("AAPL", "2023-01-01", "2023-12-31")
        # Saturday — not a trading day
        saturday = datetime.date(2023, 1, 7)
        close = dm.get_close("AAPL", saturday)
        assert close is None


# ---------------------------------------------------------------------------
# get_trading_dates()
# ---------------------------------------------------------------------------

class TestGetTradingDates:
    def test_returns_sorted_list_of_dates(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()
        with patch("yfinance.Ticker", return_value=MagicMock(history=MagicMock(return_value=df))):
            dm.load("AAPL", "2023-01-01", "2023-12-31")
        dates = dm.get_trading_dates("AAPL", "2023-01-01", "2023-12-31")
        assert isinstance(dates, list)
        assert len(dates) > 200  # roughly 252 trading days in a year
        assert dates == sorted(dates)
        assert all(isinstance(d, datetime.date) for d in dates)

    def test_dates_are_within_requested_range(self, tmp_path):
        dm = make_dm(tmp_path)
        df = make_ohlcv_df()
        with patch("yfinance.Ticker", return_value=MagicMock(history=MagicMock(return_value=df))):
            dm.load("AAPL", "2023-01-01", "2023-12-31")
        start = datetime.date(2023, 3, 1)
        end = datetime.date(2023, 6, 30)
        dates = dm.get_trading_dates("AAPL", str(start), str(end))
        assert all(start <= d <= end for d in dates)
