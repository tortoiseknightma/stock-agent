"""
Backtest Engine
================
Orchestrates a day-by-day backtest loop, feeding historical OHLCV through
the existing analysis pipeline and execution layer.

Design:
- BacktestBroker replays real prices (no GBM noise)
- TechnicalAnalyzer runs on history-up-to-date (no look-ahead)
- TradeExecutor, RiskEngine, TradeJournal used unchanged
- No LLM calls — technical-only analysis for determinism and speed
- Single-ticker for simplicity; extend to multi-ticker as a follow-up
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import AppConfig, AnalysisConfig, RiskConfig, TradingMode
from core.memory.trade_journal import TradeJournal
from analysis.technical.technical import TechnicalAnalyzer
from analysis.composite.composite import CompositeAnalyzer
from execution.risk_engine import RiskEngine
from execution.executor import TradeExecutor
from backtest.data_manager import HistoricalDataManager
from backtest.broker import BacktestBroker


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    ticker: str
    start_date: str                           # "YYYY-MM-DD"
    end_date: str                             # "YYYY-MM-DD"
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.001
    commission_per_trade: float = 1.0
    analysis_config: Optional[AnalysisConfig] = None
    risk_config: Optional[RiskConfig] = None
    buy_threshold: float = 0.4               # composite_score ≥ this → buy
    sell_threshold: float = -0.4             # composite_score ≤ this → sell
    target_position_pct: float = 0.95        # fraction of capital to deploy
    cache_dir: str = "data/backtest_cache"
    min_history_bars: int = 50               # skip days with < this many bars


@dataclass
class BacktestResult:
    """Output of a completed backtest run."""
    config: BacktestConfig
    daily_values: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    start_value: float = 0.0
    end_value: float = 0.0


class BacktestEngine:
    """
    Runs a day-by-day backtest using the full StockAgent analysis and
    execution stack, powered by historical OHLCV data.

    Usage::

        config = BacktestConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
        )
        engine = BacktestEngine(config)
        result = engine.run()

        from backtest.metrics import compute_metrics, format_report
        metrics = compute_metrics(result)
        print(format_report(metrics, result))
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # --- Data ---
        self.data_manager = HistoricalDataManager(config.cache_dir)

        # --- Broker ---
        self.broker = BacktestBroker(
            data_manager=self.data_manager,
            initial_capital=config.initial_capital,
            slippage_pct=config.slippage_pct,
            commission_per_trade=config.commission_per_trade,
        )

        # --- Analysis (no LLM, no debate, no thesis) ---
        analysis_cfg = config.analysis_config or AnalysisConfig()
        self.technical = TechnicalAnalyzer(analysis_cfg)
        self.composite = CompositeAnalyzer(
            analysis_cfg,
            thesis_generator=None,
            debate_engine=None,
        )

        # --- Execution ---
        risk_cfg = config.risk_config or RiskConfig()
        # Disable daily loss limit for backtest (prevents early-exit on volatile days)
        # Keep all other risk rules active
        app_config = AppConfig(
            trading_mode=TradingMode.SIMULATED,
            analysis=analysis_cfg,
            risk=risk_cfg,
        )
        self.risk = RiskEngine(risk_cfg, self.broker)
        self.journal = TradeJournal(":memory:")
        self.executor = TradeExecutor(app_config, self.broker, self.risk, self.journal)

    def run(self) -> BacktestResult:
        """
        Execute the backtest. Returns a BacktestResult with daily portfolio
        values, all signals generated, and all trades executed.
        """
        cfg = self.config

        print(f"[Backtest] Loading data for {cfg.ticker} "
              f"({cfg.start_date} → {cfg.end_date})…")
        self.data_manager.load(cfg.ticker, cfg.start_date, cfg.end_date)

        trading_dates = self.data_manager.get_trading_dates(
            cfg.ticker, cfg.start_date, cfg.end_date
        )
        if not trading_dates:
            print(f"[Backtest] No trading data found for {cfg.ticker}.")
            return BacktestResult(config=cfg)

        self.broker.connect()
        self.risk.reset_daily()

        result = BacktestResult(config=cfg)
        result.start_value = cfg.initial_capital

        print(f"[Backtest] Running {len(trading_dates)} trading days…")

        for date in trading_dates:
            # Advance time
            self.broker.set_date(date)
            self.broker.reset_daily()
            self.risk.reset_daily()

            # Get price history UP TO this date (no look-ahead)
            price_history = self.broker.get_price_history(cfg.ticker, period="6M")

            # Skip if insufficient history for indicators (need at least ma_long bars)
            if len(price_history) < cfg.min_history_bars:
                account = self.broker.get_account_info()
                result.daily_values.append({
                    "date": str(date),
                    "portfolio_value": account.net_liquidation,
                    "cash": account.total_cash,
                    "positions_value": account.gross_position_value,
                })
                continue

            # Technical analysis
            tech_signal = self.technical.analyze(price_history)
            tech_signal.ticker = cfg.ticker

            # Composite (technical-only)
            composite_signal = self.composite.analyze(
                cfg.ticker, technical=tech_signal
            )

            # Record signal
            result.signals.append({
                "date": str(date),
                "ticker": cfg.ticker,
                "composite_score": composite_signal.composite_score,
                "signal": composite_signal.signal,
                "confidence": composite_signal.confidence,
            })

            # Trade decision
            position = self.broker.get_position(cfg.ticker)
            score = composite_signal.composite_score

            if score >= cfg.buy_threshold and not position:
                # No position and bullish signal → buy
                self.executor.execute_buy(
                    cfg.ticker, composite_signal,
                    target_pct=cfg.target_position_pct,
                )

            elif score <= cfg.sell_threshold and position:
                # Have position and bearish signal → sell all
                self.executor.execute_sell(cfg.ticker, composite_signal)

            # Record daily portfolio snapshot
            account = self.broker.get_account_info()
            result.daily_values.append({
                "date": str(date),
                "portfolio_value": round(account.net_liquidation, 2),
                "cash": round(account.total_cash, 2),
                "positions_value": round(account.gross_position_value, 2),
            })

        # Close any open position at end of backtest
        position = self.broker.get_position(cfg.ticker)
        if position and position.quantity > 0:
            final_signal = self.composite.analyze(cfg.ticker)
            self.executor.execute_sell(cfg.ticker, final_signal)
            # Update final portfolio value
            account = self.broker.get_account_info()
            if result.daily_values:
                result.daily_values[-1]["portfolio_value"] = round(
                    account.net_liquidation, 2
                )

        result.end_value = (
            result.daily_values[-1]["portfolio_value"]
            if result.daily_values
            else cfg.initial_capital
        )
        result.trades = self.journal.get_trades()

        pct = (result.end_value / result.start_value - 1) * 100
        print(
            f"[Backtest] Done. "
            f"Return: {pct:+.1f}%  "
            f"({len(result.trades)} trades, {len(result.daily_values)} days)"
        )

        return result
