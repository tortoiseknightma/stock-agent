"""Tests for backtest.BacktestEngine."""

import datetime
import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from backtest.data_manager import HistoricalDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv_df(start="2023-01-03", days=260, base_price=150.0) -> pd.DataFrame:
    """Linearly increasing prices to produce a clear buy signal."""
    dates = pd.bdate_range(start=start, periods=days)
    prices = [base_price + i * 0.2 for i in range(days)]
    df = pd.DataFrame({
        "open":   [p - 0.1 for p in prices],
        "high":   [p + 0.2 for p in prices],
        "low":    [p - 0.2 for p in prices],
        "close":  prices,
        "volume": [1_500_000] * days,
    }, index=dates)
    df.index = [d.date() for d in df.index]
    df.index.name = "date"
    return df


def make_config(**kwargs) -> BacktestConfig:
    defaults = dict(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100_000.0,
        slippage_pct=0.0,
        commission_per_trade=0.0,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def make_engine_with_mock_data(tmp_path, **config_kwargs) -> BacktestEngine:
    """
    Create a BacktestEngine whose DataManager is pre-loaded with
    synthetic data (no real yfinance call).
    """
    cfg = make_config(cache_dir=str(tmp_path / "cache"), **config_kwargs)
    engine = BacktestEngine(cfg)

    df = make_ohlcv_df()
    engine.data_manager._data["AAPL"] = df
    return engine


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestEngineSmoke:
    def test_engine_instantiates(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        assert engine is not None

    def test_run_returns_backtest_result(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_run_with_no_data_returns_empty_result(self, tmp_path):
        cfg = make_config(cache_dir=str(tmp_path / "cache"))
        engine = BacktestEngine(cfg)
        # Patch yfinance to return empty data so no trading dates are available
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = empty_df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = engine.run()
        assert result.daily_values == []


# ---------------------------------------------------------------------------
# Daily values
# ---------------------------------------------------------------------------

class TestDailyValues:
    def test_daily_values_have_required_keys(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        assert len(result.daily_values) > 0
        for entry in result.daily_values:
            assert "date" in entry
            assert "portfolio_value" in entry
            assert "cash" in entry

    def test_daily_values_count_near_trading_days(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        # Roughly 252 trading days in a year
        assert 200 <= len(result.daily_values) <= 270

    def test_start_and_end_values_populated(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        assert result.start_value == pytest.approx(100_000.0)
        assert result.end_value > 0


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class TestSignals:
    def test_signals_populated(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        # After warm-up (50 bars), signals should appear
        assert len(result.signals) > 0

    def test_signals_have_required_fields(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        for sig in result.signals:
            assert "date" in sig
            assert "ticker" in sig
            assert "composite_score" in sig
            assert "signal" in sig

    def test_signal_ticker_matches_config(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        for sig in result.signals:
            assert sig["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Config used correctly
# ---------------------------------------------------------------------------

class TestConfig:
    def test_custom_initial_capital(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path, initial_capital=50_000.0)
        result = engine.run()
        assert result.start_value == pytest.approx(50_000.0)

    def test_result_config_matches_input(self, tmp_path):
        engine = make_engine_with_mock_data(tmp_path)
        result = engine.run()
        assert result.config.ticker == "AAPL"
