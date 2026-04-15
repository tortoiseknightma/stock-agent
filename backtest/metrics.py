"""
Backtest Performance Metrics
=============================
Compute standard quantitative finance metrics from a BacktestResult.

Metrics computed:
- Total return, Annualized return (CAGR)
- Sharpe ratio, Sortino ratio
- Max drawdown (depth and duration)
- Win rate, Profit factor
- Volatility (annualized), Calmar ratio
- Per-trade statistics (avg, best, worst)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """All performance statistics for a backtest run."""
    # Returns
    total_return: float           # e.g. 0.15 = +15%
    annualized_return: float      # CAGR
    # Risk-adjusted
    sharpe_ratio: float           # (mean_daily - rf/252) / std * sqrt(252)
    sortino_ratio: float          # uses downside deviation only
    # Drawdown
    max_drawdown: float           # e.g. -0.12 = -12%
    max_drawdown_duration_days: int
    # Volatility
    volatility: float             # annualized std dev of daily returns
    calmar_ratio: float           # CAGR / |max_drawdown|
    # Trades
    num_trades: int
    win_rate: float               # profitable trades / total trades
    profit_factor: float          # gross_profit / gross_loss
    avg_trade_return: float
    best_trade: float
    worst_trade: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def compute_metrics(
    result: Any,  # BacktestResult (forward ref)
    risk_free_rate: float = 0.04,
) -> PerformanceMetrics:
    """
    Compute all performance metrics from a BacktestResult.

    Args:
        result: BacktestResult with daily_values and trades fields
        risk_free_rate: Annual risk-free rate (default 4%)
    """
    daily_values = result.daily_values
    trades = result.trades

    # Require at least 2 data points
    if len(daily_values) < 2:
        return _empty_metrics()

    values = [d["portfolio_value"] for d in daily_values]
    start_value = values[0]
    end_value = values[-1]

    # --- Returns ---
    total_return = (end_value - start_value) / start_value if start_value > 0 else 0.0
    n_days = len(values)
    years = n_days / _TRADING_DAYS_PER_YEAR
    annualized_return = (
        (end_value / start_value) ** (1 / years) - 1
        if years > 0 and start_value > 0
        else 0.0
    )

    # --- Daily returns ---
    daily_returns = _daily_returns(values)
    vol = _volatility(daily_returns)

    # --- Sharpe & Sortino ---
    rf_daily = risk_free_rate / _TRADING_DAYS_PER_YEAR
    sharpe = _sharpe(daily_returns, rf_daily)
    sortino = _sortino(daily_returns, rf_daily)

    # --- Max drawdown ---
    max_dd, max_dd_days = _max_drawdown(values)

    # --- Calmar ---
    calmar = (
        annualized_return / abs(max_dd)
        if max_dd < 0
        else (annualized_return if annualized_return != 0 else 0.0)
    )

    # --- Trade stats ---
    trade_returns = _trade_returns(trades)
    n_trades = len(trade_returns)
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit == 0 else float("inf"))
    )
    avg_tr = sum(trade_returns) / n_trades if n_trades > 0 else 0.0
    best = max(trade_returns) if trade_returns else 0.0
    worst = min(trade_returns) if trade_returns else 0.0

    return PerformanceMetrics(
        total_return=round(total_return, 4),
        annualized_return=round(annualized_return, 4),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        max_drawdown_duration_days=max_dd_days,
        volatility=round(vol, 4),
        calmar_ratio=round(calmar, 3),
        num_trades=n_trades,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3),
        avg_trade_return=round(avg_tr, 4),
        best_trade=round(best, 4),
        worst_trade=round(worst, 4),
    )


def format_report(metrics: PerformanceMetrics, result: Any) -> str:
    """Format a CLI-friendly backtest summary."""
    cfg = result.config
    lines = [
        "",
        f"{'='*55}",
        f"  BACKTEST RESULTS — {cfg.ticker}",
        f"  {cfg.start_date}  →  {cfg.end_date}",
        f"{'='*55}",
        "",
        "  RETURNS",
        f"    Total return:         {metrics.total_return:+.2%}",
        f"    Annualized return:    {metrics.annualized_return:+.2%}",
        f"    Start value:          ${result.start_value:,.0f}",
        f"    End value:            ${result.end_value:,.0f}",
        "",
        "  RISK-ADJUSTED",
        f"    Sharpe ratio:         {metrics.sharpe_ratio:.3f}",
        f"    Sortino ratio:        {metrics.sortino_ratio:.3f}",
        f"    Calmar ratio:         {metrics.calmar_ratio:.3f}",
        f"    Annualized volatility:{metrics.volatility:.2%}",
        "",
        "  DRAWDOWN",
        f"    Max drawdown:         {metrics.max_drawdown:.2%}",
        f"    Max DD duration:      {metrics.max_drawdown_duration_days} days",
        "",
        "  TRADES",
        f"    Total trades:         {metrics.num_trades}",
        f"    Win rate:             {metrics.win_rate:.1%}",
        f"    Profit factor:        {metrics.profit_factor:.2f}",
        f"    Avg trade return:     {metrics.avg_trade_return:.2%}",
        f"    Best trade:           {metrics.best_trade:.2%}",
        f"    Worst trade:          {metrics.worst_trade:.2%}",
        f"{'='*55}",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _daily_returns(values: List[float]) -> List[float]:
    """Compute day-over-day percentage returns."""
    if len(values) < 2:
        return []
    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        ret = (values[i] - prev) / prev if prev > 0 else 0.0
        returns.append(ret)
    return returns


def _volatility(daily_returns: List[float]) -> float:
    """Annualized standard deviation of daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    return math.sqrt(variance) * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _sharpe(daily_returns: List[float], rf_daily: float) -> float:
    """Sharpe = (mean_return - rf_daily) / std * sqrt(252)."""
    if len(daily_returns) < 2:
        return 0.0
    n = len(daily_returns)
    mean = sum(daily_returns) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in daily_returns) / (n - 1))
    if std < 1e-15:  # handle floating-point imprecision
        return 0.0
    return (mean - rf_daily) / std * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _sortino(daily_returns: List[float], rf_daily: float) -> float:
    """Sortino = (mean_return - rf_daily) / downside_std * sqrt(252)."""
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    downside = [r for r in daily_returns if r < rf_daily]
    if not downside:
        return 0.0
    downside_std = math.sqrt(
        sum((r - rf_daily) ** 2 for r in downside) / len(downside)
    )
    if downside_std == 0:
        return 0.0
    return (mean - rf_daily) / downside_std * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _max_drawdown(values: List[float]) -> Tuple[float, int]:
    """
    Compute maximum drawdown and its duration in days.
    Returns (max_drawdown, duration_days) where max_drawdown ≤ 0.
    """
    if len(values) < 2:
        return 0.0, 0

    peak = values[0]
    max_dd = 0.0
    peak_idx = 0
    max_dd_start = 0
    max_dd_duration = 0
    current_dd_start = 0

    for i, v in enumerate(values):
        if v > peak:
            peak = v
            peak_idx = i
            current_dd_start = i

        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            max_dd_duration = i - current_dd_start

    return max_dd, max_dd_duration


def _trade_returns(trades: List[Dict]) -> List[float]:
    """
    Extract round-trip (buy→sell) returns from the trades list.
    Uses realized_pnl and portfolio_value_before from each trade record.
    For sell trades where realized_pnl is available, computes return directly.
    """
    returns = []
    for t in trades:
        side = t.get("side", "")
        if side.upper() not in ("SELL", "BUY"):
            continue
        pnl = t.get("realized_pnl")
        value_before = t.get("portfolio_value_before") or t.get("portfolio_value")
        if pnl is not None and value_before and value_before > 0:
            returns.append(float(pnl) / float(value_before))
    return returns


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.0, annualized_return=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0,
        max_drawdown=0.0, max_drawdown_duration_days=0,
        volatility=0.0, calmar_ratio=0.0,
        num_trades=0, win_rate=0.0, profit_factor=1.0,
        avg_trade_return=0.0, best_trade=0.0, worst_trade=0.0,
    )
