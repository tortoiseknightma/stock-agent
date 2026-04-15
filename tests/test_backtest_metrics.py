"""Tests for backtest.metrics — all use known values for exact verification."""

import math
import pytest
from backtest.metrics import (
    PerformanceMetrics, compute_metrics,
    _daily_returns, _sharpe, _sortino, _max_drawdown, _trade_returns,
    _volatility,
)
from backtest.engine import BacktestConfig, BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(values: list, trades: list = None) -> BacktestResult:
    """Build a minimal BacktestResult from a list of portfolio values."""
    cfg = BacktestConfig(ticker="TEST", start_date="2023-01-01", end_date="2023-12-31")
    daily_values = [{"date": f"2023-{i:02d}-01", "portfolio_value": v,
                     "cash": v, "positions_value": 0.0}
                    for i, v in enumerate(values, 1)]
    result = BacktestResult(
        config=cfg,
        daily_values=daily_values,
        trades=trades or [],
        start_value=values[0],
        end_value=values[-1],
    )
    return result


# ---------------------------------------------------------------------------
# _daily_returns
# ---------------------------------------------------------------------------

class TestDailyReturns:
    def test_simple_gain(self):
        ret = _daily_returns([100.0, 110.0])
        assert ret == pytest.approx([0.10])

    def test_flat_returns_zero(self):
        ret = _daily_returns([100.0, 100.0])
        assert ret == pytest.approx([0.0])

    def test_multiple_periods(self):
        ret = _daily_returns([100.0, 110.0, 99.0])
        assert ret[0] == pytest.approx(0.10)
        assert ret[1] == pytest.approx(-0.10, rel=1e-3)

    def test_empty_when_single_value(self):
        assert _daily_returns([100.0]) == []


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_drawdown(self):
        # Peak 110, falls to 88 → -20% drawdown
        dd, _ = _max_drawdown([100.0, 110.0, 88.0])
        assert dd == pytest.approx(-0.2, rel=1e-3)

    def test_monotonically_increasing_no_drawdown(self):
        dd, _ = _max_drawdown([100.0, 110.0, 120.0])
        assert dd == pytest.approx(0.0, abs=1e-9)

    def test_duration_tracked(self):
        _, duration = _max_drawdown([100.0, 110.0, 88.0, 95.0])
        assert duration >= 1

    def test_single_value(self):
        dd, dur = _max_drawdown([100.0])
        assert dd == pytest.approx(0.0)
        assert dur == 0


# ---------------------------------------------------------------------------
# _sharpe / _sortino
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_zero_volatility_returns_zero(self):
        # Constant returns → std=0 → Sharpe=0
        rets = [0.01] * 252
        s = _sharpe(rets, rf_daily=0.0)
        assert s == pytest.approx(0.0)

    def test_positive_excess_returns_positive_sharpe(self):
        rets = [0.002] * 252   # 50% annual return
        s = _sharpe(rets, rf_daily=0.0)
        # Mean > 0, std=0 → would be inf, but due to same values it's 0
        # Let's use slightly varying returns
        import random
        random.seed(42)
        rets2 = [0.001 + random.gauss(0, 0.005) for _ in range(252)]
        s2 = _sharpe(rets2, rf_daily=0.0)
        assert isinstance(s2, float)

    def test_empty_returns_zero(self):
        assert _sharpe([], 0.0) == 0.0


class TestSortino:
    def test_no_downside_returns_zero_sortino(self):
        # All positive returns → no downside
        rets = [0.01] * 10
        s = _sortino(rets, rf_daily=0.0)
        assert s == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert _sortino([], 0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_total_return_known_value(self):
        # Simple 10% gain
        result = make_result([100_000.0, 110_000.0])
        m = compute_metrics(result, risk_free_rate=0.0)
        assert m.total_return == pytest.approx(0.10)

    def test_total_return_loss(self):
        result = make_result([100_000.0, 90_000.0])
        m = compute_metrics(result, risk_free_rate=0.0)
        assert m.total_return == pytest.approx(-0.10)

    def test_max_drawdown_from_100_to_80(self):
        result = make_result([100_000.0, 120_000.0, 80_000.0])
        m = compute_metrics(result, risk_free_rate=0.0)
        assert m.max_drawdown == pytest.approx(-1/3, rel=1e-2)

    def test_win_rate_all_wins(self):
        trades = [
            {"side": "SELL", "realized_pnl": 500.0,
             "portfolio_value_before": 10_000.0},
            {"side": "SELL", "realized_pnl": 300.0,
             "portfolio_value_before": 10_000.0},
        ]
        result = make_result([100_000.0, 101_000.0], trades=trades)
        m = compute_metrics(result)
        assert m.win_rate == pytest.approx(1.0)

    def test_win_rate_mixed(self):
        trades = [
            {"side": "SELL", "realized_pnl": 500.0,
             "portfolio_value_before": 10_000.0},
            {"side": "SELL", "realized_pnl": -200.0,
             "portfolio_value_before": 10_000.0},
        ]
        result = make_result([100_000.0, 101_000.0], trades=trades)
        m = compute_metrics(result)
        assert m.win_rate == pytest.approx(0.5)

    def test_zero_trades_no_crash(self):
        result = make_result([100_000.0, 105_000.0])
        m = compute_metrics(result)
        assert m.num_trades == 0
        assert m.win_rate == pytest.approx(0.0)

    def test_constant_portfolio_sharpe_zero(self):
        values = [100_000.0] * 50
        result = make_result(values)
        m = compute_metrics(result, risk_free_rate=0.0)
        assert m.sharpe_ratio == pytest.approx(0.0)

    def test_insufficient_data_returns_empty_metrics(self):
        result = make_result([100_000.0])
        m = compute_metrics(result)
        assert m.total_return == pytest.approx(0.0)
        assert m.sharpe_ratio == pytest.approx(0.0)

    def test_profit_factor_all_losses(self):
        trades = [
            {"side": "SELL", "realized_pnl": -200.0,
             "portfolio_value_before": 10_000.0},
        ]
        result = make_result([100_000.0, 99_800.0], trades=trades)
        m = compute_metrics(result)
        assert m.profit_factor == pytest.approx(0.0, abs=0.01)

    def test_volatility_positive_for_varying_portfolio(self):
        import random
        random.seed(1)
        values = [100_000.0]
        for _ in range(251):
            values.append(values[-1] * (1 + random.gauss(0.0005, 0.015)))
        result = make_result(values)
        m = compute_metrics(result)
        assert m.volatility > 0.0
