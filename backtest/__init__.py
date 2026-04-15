"""
Backtesting Framework
======================
Historical data replay, performance metrics, and parameter optimization
for the StockAgent analysis pipeline.
"""

from .data_manager import HistoricalDataManager
from .broker import BacktestBroker
from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import compute_metrics, format_report, PerformanceMetrics
from .optimizer import GridOptimizer, OptimizationResult

__all__ = [
    "HistoricalDataManager",
    "BacktestBroker",
    "BacktestEngine", "BacktestConfig", "BacktestResult",
    "compute_metrics", "format_report", "PerformanceMetrics",
    "GridOptimizer", "OptimizationResult",
]
