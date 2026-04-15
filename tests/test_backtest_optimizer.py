"""Tests for backtest.GridOptimizer."""

import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.engine import BacktestConfig
from backtest.optimizer import GridOptimizer, OptimizationResult, _cartesian
from backtest.metrics import PerformanceMetrics


# ---------------------------------------------------------------------------
# _cartesian helper
# ---------------------------------------------------------------------------

class TestCartesian:
    def test_single_param(self):
        combos = list(_cartesian({"a": [1, 2, 3]}))
        assert len(combos) == 3
        assert all("a" in c for c in combos)

    def test_two_params(self):
        combos = list(_cartesian({"a": [1, 2], "b": [10, 20]}))
        assert len(combos) == 4

    def test_values_present(self):
        combos = list(_cartesian({"x": [5]}))
        assert combos == [{"x": 5}]

    def test_empty_grid(self):
        combos = list(_cartesian({}))
        assert combos == [{}]


# ---------------------------------------------------------------------------
# _apply_params
# ---------------------------------------------------------------------------

class TestApplyParams:
    def _base_config(self) -> BacktestConfig:
        return BacktestConfig(
            ticker="AAPL", start_date="2023-01-01", end_date="2023-12-31"
        )

    def test_overrides_buy_threshold(self):
        opt = GridOptimizer(self._base_config())
        new_cfg = opt._apply_params({"buy_threshold": 0.3})
        assert new_cfg.buy_threshold == pytest.approx(0.3)

    def test_overrides_sell_threshold(self):
        opt = GridOptimizer(self._base_config())
        new_cfg = opt._apply_params({"sell_threshold": -0.5})
        assert new_cfg.sell_threshold == pytest.approx(-0.5)

    def test_original_config_unchanged(self):
        base = self._base_config()
        opt = GridOptimizer(base)
        opt._apply_params({"buy_threshold": 0.1})
        assert base.buy_threshold == pytest.approx(0.4)  # unchanged

    def test_other_fields_preserved(self):
        opt = GridOptimizer(self._base_config())
        new_cfg = opt._apply_params({"buy_threshold": 0.3})
        assert new_cfg.ticker == "AAPL"
        assert new_cfg.start_date == "2023-01-01"


# ---------------------------------------------------------------------------
# optimize — sorted results
# ---------------------------------------------------------------------------

class TestOptimize:
    def _make_opt(self) -> GridOptimizer:
        base = BacktestConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            cache_dir="data/backtest_cache",
        )
        return GridOptimizer(base)

    def _mock_engine_run(self, sharpe_val):
        """Return a mock BacktestResult whose compute_metrics gives a specific Sharpe."""
        from backtest.engine import BacktestResult
        result = MagicMock(spec=BacktestResult)
        result.daily_values = [
            {"date": "2023-01-03", "portfolio_value": 100_000 + i * 100,
             "cash": 100_000.0, "positions_value": 0.0}
            for i in range(50)
        ]
        result.trades = []
        result.start_value = 100_000.0
        result.end_value = 100_000.0 + 50 * 100
        result.config = MagicMock()
        return result

    def test_results_sorted_by_sharpe(self, tmp_path):
        """Results should come back sorted highest Sharpe first."""
        import pandas as pd
        df = pd.DataFrame({
            "open":   [150.0 + i for i in range(260)],
            "high":   [151.0 + i for i in range(260)],
            "low":    [149.0 + i for i in range(260)],
            "close":  [150.0 + i * 0.5 for i in range(260)],
            "volume": [1_000_000] * 260,
        }, index=[pd.Timestamp("2023-01-03") + pd.Timedelta(days=i) for i in range(260)])
        df.index = [d.date() for d in df.index]
        df.index.name = "date"

        base = BacktestConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            cache_dir=str(tmp_path / "cache"),
        )
        opt = GridOptimizer(base)

        # Pre-load data into the shared DM so no yfinance call
        from backtest.data_manager import HistoricalDataManager
        with patch.object(HistoricalDataManager, "load", return_value=df):
            with patch.object(HistoricalDataManager, "get_trading_dates",
                              return_value=[df.index[i] for i in range(len(df))]):
                results = opt.optimize(
                    param_grid={"buy_threshold": [0.3, 0.4, 0.5]},
                    rank_by="sharpe_ratio",
                )

        if len(results) >= 2:
            sharpe_values = [r.metrics.sharpe_ratio for r in results]
            assert sharpe_values == sorted(sharpe_values, reverse=True)

    def test_result_count_matches_combos(self, tmp_path):
        """Number of OptimizationResult should equal number of param combos (when all succeed)."""
        import pandas as pd
        df = pd.DataFrame({
            "open":   [150.0] * 260, "high": [152.0] * 260,
            "low":    [148.0] * 260, "close": [151.0] * 260,
            "volume": [1_000_000] * 260,
        }, index=[pd.Timestamp("2023-01-03") + pd.Timedelta(days=i) for i in range(260)])
        df.index = [d.date() for d in df.index]
        df.index.name = "date"

        base = BacktestConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            cache_dir=str(tmp_path / "cache"),
        )
        opt = GridOptimizer(base)

        from backtest.data_manager import HistoricalDataManager
        with patch.object(HistoricalDataManager, "load", return_value=df):
            with patch.object(HistoricalDataManager, "get_trading_dates",
                              return_value=[df.index[i] for i in range(len(df))]):
                results = opt.optimize(
                    param_grid={"buy_threshold": [0.4, 0.5]},
                )

        assert len(results) == 2
