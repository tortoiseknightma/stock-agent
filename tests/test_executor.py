"""Tests for TradeExecutor — buy/sell lifecycle, risk gate, advisory mode."""

import pytest
from unittest.mock import MagicMock, patch
from core.config import AppConfig, TradingMode, load_config
from core.broker.simulated import SimulatedBroker
from core.broker.base import OrderSide as BrokerOrderSide, OrderType as BrokerOrderType
from core.memory.trade_journal import TradeJournal
from analysis.composite.composite import CompositeSignal
from execution.risk_engine import RiskEngine
from execution.executor import TradeExecutor, ExecutionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(mode: TradingMode = TradingMode.SIMULATED) -> AppConfig:
    cfg = AppConfig()
    cfg.trading_mode = mode
    return cfg


def make_broker(tmp_path, capital=100_000.0, portfolio=None):
    seed = {"AAPL": 0}
    if portfolio:
        seed.update(portfolio)
    broker = SimulatedBroker(
        initial_capital=capital,
        default_portfolio=seed,
        slippage_pct=0.0,
        commission_per_trade=0.0,
        state_file=str(tmp_path / "broker_state.json"),
    )
    broker.connect()
    return broker


def make_journal(tmp_path):
    return TradeJournal(str(tmp_path / "journal.db"))


def make_signal(ticker="AAPL", score=0.6, signal="buy",
                confidence=0.7) -> CompositeSignal:
    return CompositeSignal(
        ticker=ticker, composite_score=score,
        signal=signal, confidence=confidence,
        recommendation="Test recommendation",
    )


def make_executor(tmp_path, capital=100_000.0, portfolio=None,
                  mode=TradingMode.SIMULATED):
    cfg = make_config(mode)
    broker = make_broker(tmp_path, capital=capital, portfolio=portfolio)
    risk = RiskEngine(cfg.risk, broker)
    risk.reset_daily()
    journal = make_journal(tmp_path)
    return TradeExecutor(cfg, broker, risk, journal), broker, journal


# ---------------------------------------------------------------------------
# Buy lifecycle
# ---------------------------------------------------------------------------

class TestBuyExecution:
    def test_buy_succeeds_with_sufficient_cash(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("AAPL", score=0.7, signal="buy")
        # target_pct=0.03 keeps trade within risk limits (max_trade_pct=5%)
        result = executor.execute_buy("AAPL", signal, target_pct=0.03)
        assert result.success is True

    def test_buy_reduces_cash(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        cash_before = broker.get_account_info().total_cash
        signal = make_signal("AAPL", score=0.7)
        executor.execute_buy("AAPL", signal, target_pct=0.03)
        cash_after = broker.get_account_info().total_cash
        assert cash_after < cash_before

    def test_buy_creates_position(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("AAPL", score=0.7)
        executor.execute_buy("AAPL", signal, target_pct=0.03)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity > 0

    def test_buy_records_decision(self, tmp_path):
        executor, broker, journal = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("AAPL", score=0.7)
        result = executor.execute_buy("AAPL", signal, target_pct=0.03)
        assert result.journal_recorded is True
        assert result.decision_id is not None

    def test_buy_result_message_contains_ticker(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("AAPL", score=0.7)
        result = executor.execute_buy("AAPL", signal, target_pct=0.03)
        assert "AAPL" in result.message

    def test_buy_fails_with_no_price(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("ZZZZ", score=0.7)  # unknown ticker, no price
        result = executor.execute_buy("ZZZZ", signal)
        assert result.success is False

    def test_buy_with_target_pct(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("AAPL", score=0.7)
        result = executor.execute_buy("AAPL", signal, target_pct=0.03)  # 3% of portfolio
        assert result.success is True


# ---------------------------------------------------------------------------
# Sell lifecycle
# ---------------------------------------------------------------------------

class TestSellExecution:
    def test_sell_succeeds_with_position(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0,
                                            portfolio={"AAPL": 20})
        signal = make_signal("AAPL", score=-0.6, signal="sell")
        result = executor.execute_sell("AAPL", signal, quantity=10)
        assert result.success is True

    def test_sell_increases_cash(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0,
                                            portfolio={"AAPL": 20})
        cash_before = broker.get_account_info().total_cash
        signal = make_signal("AAPL", score=-0.5)
        executor.execute_sell("AAPL", signal, quantity=5)
        cash_after = broker.get_account_info().total_cash
        assert cash_after > cash_before

    def test_sell_all_when_no_quantity(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0,
                                            portfolio={"AAPL": 15})
        signal = make_signal("AAPL", score=-0.5)
        result = executor.execute_sell("AAPL", signal)
        assert result.success is True
        assert broker.get_position("AAPL") is None

    def test_sell_without_position_fails(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        signal = make_signal("AAPL", score=-0.5)
        result = executor.execute_sell("AAPL", signal)
        assert result.success is False
        assert "No position" in result.message

    def test_sell_records_journal(self, tmp_path):
        executor, broker, journal = make_executor(tmp_path, capital=100_000.0,
                                                  portfolio={"AAPL": 10})
        signal = make_signal("AAPL", score=-0.5)
        result = executor.execute_sell("AAPL", signal, quantity=5)
        assert result.journal_recorded is True


# ---------------------------------------------------------------------------
# Risk gate
# ---------------------------------------------------------------------------

class TestRiskGate:
    def test_buy_blocked_by_daily_loss_limit(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0)
        # Artificially trigger daily loss limit
        executor.risk._daily_pnl = -9999.0
        signal = make_signal("AAPL", score=0.9)
        result = executor.execute_buy("AAPL", signal)
        assert result.success is False

    def test_blocked_decision_still_recorded(self, tmp_path):
        executor, broker, journal = make_executor(tmp_path, capital=100_000.0)
        executor.risk._daily_pnl = -9999.0  # trigger daily loss
        signal = make_signal("AAPL", score=0.9)
        result = executor.execute_buy("AAPL", signal)
        assert result.journal_recorded is True


# ---------------------------------------------------------------------------
# Advisory mode
# ---------------------------------------------------------------------------

class TestAdvisoryMode:
    def test_advisory_buy_does_not_create_position(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0,
                                            mode=TradingMode.ADVISORY)
        signal = make_signal("AAPL", score=0.8)
        # Use target_pct within risk limits so the advisory gate is reached
        result = executor.execute_buy("AAPL", signal, target_pct=0.03)
        # advisory: success=False (not executed), no position created
        assert result.success is False
        assert "ADVISORY" in result.message
        pos = broker.get_position("AAPL")
        assert pos is None or pos.quantity == 0  # seed creates qty=0 placeholder

    def test_advisory_sell_does_not_close_position(self, tmp_path):
        executor, broker, _ = make_executor(tmp_path, capital=100_000.0,
                                            portfolio={"AAPL": 10},
                                            mode=TradingMode.ADVISORY)
        signal = make_signal("AAPL", score=-0.5)
        result = executor.execute_sell("AAPL", signal)
        assert "ADVISORY" in result.message
        assert broker.get_position("AAPL") is not None  # position untouched
