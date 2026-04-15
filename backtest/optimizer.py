"""
Parameter Optimizer
====================
Grid search over BacktestConfig parameters to find the combination that
maximises a chosen performance metric (default: Sharpe ratio).

Usage::

    from backtest.engine import BacktestConfig
    from backtest.optimizer import GridOptimizer

    base_config = BacktestConfig(
        ticker="AAPL",
        start_date="2022-01-01",
        end_date="2024-01-01",
    )
    optimizer = GridOptimizer(base_config)
    results = optimizer.optimize(
        param_grid={
            "buy_threshold":  [0.3, 0.4, 0.5],
            "sell_threshold": [-0.3, -0.4, -0.5],
        },
        rank_by="sharpe_ratio",
    )
    print(optimizer.format_results(results, top_n=5))
"""

import copy
import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import PerformanceMetrics, compute_metrics


@dataclass
class OptimizationResult:
    """Result for one parameter combination."""
    params: Dict[str, Any]
    metrics: PerformanceMetrics
    config: BacktestConfig


class GridOptimizer:
    """
    Exhaustive grid search over BacktestConfig scalar parameters.

    Supported param_grid keys (all BacktestConfig fields are eligible):
        buy_threshold, sell_threshold, target_position_pct,
        slippage_pct, commission_per_trade, etc.

    For nested config keys (e.g., analysis weights), use dotted notation:
        "analysis_config.weight_technical": [0.3, 0.4, 0.5]
    """

    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config

    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        rank_by: str = "sharpe_ratio",
    ) -> List[OptimizationResult]:
        """
        Run a backtest for every combination in *param_grid*.

        Returns results sorted by *rank_by* (highest first).
        The data is fetched once and cached, so subsequent engine runs
        skip the yfinance API call.
        """
        # Pre-load data (shared cache across all runs)
        cfg = self.base_config
        from backtest.data_manager import HistoricalDataManager
        dm = HistoricalDataManager(cfg.cache_dir)
        dm.load(cfg.ticker, cfg.start_date, cfg.end_date)

        combos = list(_cartesian(param_grid))
        print(f"[Optimizer] Running {len(combos)} parameter combinations…")

        results = []
        for i, params in enumerate(combos, 1):
            run_config = self._apply_params(params)
            try:
                engine = BacktestEngine(run_config)
                bt_result = engine.run()
                metrics = compute_metrics(bt_result)
                results.append(OptimizationResult(
                    params=params, metrics=metrics, config=run_config
                ))
            except Exception as e:
                print(f"  [Optimizer] Combo {i}/{len(combos)} failed: {e}")

        # Sort descending by rank_by metric
        results.sort(
            key=lambda r: getattr(r.metrics, rank_by, 0.0),
            reverse=True,
        )
        print(f"[Optimizer] Done. Best {rank_by}: "
              f"{getattr(results[0].metrics, rank_by, 0.0):.3f}"
              if results else "[Optimizer] No results.")
        return results

    def _apply_params(self, params: Dict[str, Any]) -> BacktestConfig:
        """Clone base_config and override fields from params."""
        cfg_dict = dataclasses.asdict(self.base_config)

        for key, value in params.items():
            if "." in key:
                # Nested: e.g. "analysis_config.weight_technical"
                parent_key, child_key = key.split(".", 1)
                if cfg_dict.get(parent_key) is None:
                    cfg_dict[parent_key] = {}
                if isinstance(cfg_dict[parent_key], dict):
                    cfg_dict[parent_key][child_key] = value
            else:
                cfg_dict[key] = value

        # Reconstruct dataclass (analysis_config and risk_config stay as dicts here;
        # BacktestConfig stores them as Optional[AnalysisConfig/RiskConfig] objects)
        from core.config import AnalysisConfig, RiskConfig
        ac = cfg_dict.pop("analysis_config", None)
        rc = cfg_dict.pop("risk_config", None)
        new_cfg = BacktestConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in {f.name for f in dataclasses.fields(BacktestConfig)}
        })
        if ac and isinstance(ac, dict):
            a = AnalysisConfig()
            for k, v in ac.items():
                if hasattr(a, k):
                    setattr(a, k, v)
            new_cfg.analysis_config = a
        if rc and isinstance(rc, dict):
            r = RiskConfig()
            for k, v in rc.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            new_cfg.risk_config = r
        return new_cfg

    def format_results(
        self, results: List[OptimizationResult], top_n: int = 10
    ) -> str:
        """Format top-N results as a readable table."""
        if not results:
            return "No optimization results."

        lines = [
            "",
            f"{'='*65}",
            f"  OPTIMIZATION RESULTS — {self.base_config.ticker}",
            f"{'='*65}",
        ]

        for i, r in enumerate(results[:top_n], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            m = r.metrics
            lines += [
                f"",
                f"  #{i}  {params_str}",
                f"       Sharpe={m.sharpe_ratio:.3f}  "
                f"Return={m.total_return:+.1%}  "
                f"MaxDD={m.max_drawdown:.1%}  "
                f"WinRate={m.win_rate:.1%}  "
                f"Trades={m.num_trades}",
            ]

        lines.append(f"{'='*65}\n")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _cartesian(param_grid: Dict[str, List[Any]]):
    """Yield all Cartesian-product combinations as dicts."""
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        yield dict(zip(keys, values))
