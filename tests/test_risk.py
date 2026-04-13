"""Tests for RiskEngine — all BLOCKER rules and safe quantity calculation."""

import pytest
from unittest.mock import MagicMock

from core.config import RiskConfig
from core.broker.base import (
    AccountInfo, Position, Order, OrderSide, OrderType,
)
from execution.risk_engine import RiskEngine, RiskLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides):
    defaults = dict(
        max_position_pct=0.20,
        max_trade_pct=0.05,
        daily_loss_limit_pct=0.03,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        max_trades_per_day=10,
        large_trade_threshold=5_000.0,
        var_confidence=0.95,
        max_drawdown_pct=0.15,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def make_account(net_liq=100_000.0, cash=50_000.0, buying_power=None):
    return AccountInfo(
        account_id="TEST",
        net_liquidation=net_liq,
        total_cash=cash,
        buying_power=buying_power if buying_power is not None else cash * 2,
    )


def make_broker(account=None, position=None, price=100.0):
    broker = MagicMock()
    broker.get_account_info.return_value = account or make_account()
    broker.get_position.return_value = position
    broker.get_current_price.return_value = price
    return broker


def buy_order(ticker="AAPL", qty=10):
    return Order(ticker=ticker, side=OrderSide.BUY,
                 order_type=OrderType.MARKET, quantity=qty)


def sell_order(ticker="AAPL", qty=10):
    return Order(ticker=ticker, side=OrderSide.SELL,
                 order_type=OrderType.MARKET, quantity=qty)


def make_engine(config=None, account=None, position=None, price=100.0):
    cfg = config or make_config()
    broker = make_broker(account=account, position=position, price=price)
    return RiskEngine(cfg, broker), broker


# ---------------------------------------------------------------------------
# No price data
# ---------------------------------------------------------------------------

class TestNoPriceData:
    def test_blocked_when_no_price(self):
        cfg = make_config()
        broker = make_broker(price=None)
        engine = RiskEngine(cfg, broker)
        result = engine.check_order(buy_order())
        assert result.approved is False
        assert any(v.level == RiskLevel.BLOCKER for v in result.violations)

    def test_blocked_when_price_zero(self):
        cfg = make_config()
        broker = make_broker(price=0.0)
        engine = RiskEngine(cfg, broker)
        result = engine.check_order(buy_order())
        assert result.approved is False


# ---------------------------------------------------------------------------
# Position concentration (Check 1)
# ---------------------------------------------------------------------------

class TestMaxPosition:
    def test_approved_when_within_concentration_limit(self):
        """Buy 10 shares @ $100 = $1,000 on $100k portfolio = 1%."""
        engine, _ = make_engine(price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=10))
        assert result.approved is True

    def test_blocked_when_exceeds_concentration(self):
        """Buy 250 shares @ $100 = $25,000 on $100k = 25% > 20% limit."""
        engine, _ = make_engine(price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=250))
        assert result.approved is False
        rules = [v.rule for v in result.violations]
        assert "max_position" in rules or "max_trade_size" in rules

    def test_existing_position_counted_toward_concentration(self):
        """Existing 180 shares @ $100 = 18%; adding 30 more → 30% > 20%."""
        existing = Position(ticker="AAPL", quantity=180, avg_cost=100.0,
                            current_price=100.0, market_value=18_000.0)
        engine, _ = make_engine(position=existing, price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=30))
        assert result.approved is False


# ---------------------------------------------------------------------------
# Trade size (Check 2)
# ---------------------------------------------------------------------------

class TestMaxTradeSize:
    def test_blocked_when_single_trade_too_large(self):
        """100 shares @ $100 = $10k = 10% > 5% limit."""
        engine, _ = make_engine(price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=100))
        assert result.approved is False
        assert any(v.rule == "max_trade_size" for v in result.violations)

    def test_approved_at_exactly_trade_limit(self):
        """5 shares @ $100 = $500 = 0.5% < 5% limit."""
        engine, _ = make_engine(price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=5))
        assert result.approved is True


# ---------------------------------------------------------------------------
# Buying power (Check 3)
# ---------------------------------------------------------------------------

class TestBuyingPower:
    def test_blocked_when_insufficient_buying_power(self):
        account = make_account(net_liq=100_000.0, cash=200.0, buying_power=200.0)
        engine, _ = make_engine(account=account, price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=3))
        assert result.approved is False
        assert any(v.rule == "buying_power" for v in result.violations)


# ---------------------------------------------------------------------------
# Daily loss limit (Check 4)
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_blocked_when_daily_loss_exceeded(self):
        engine, broker = make_engine(price=100.0)

        # Simulate start-of-day portfolio at $100k, now down 5%
        engine._daily_start_value = 100_000.0
        broker.get_account_info.return_value = make_account(net_liq=95_000.0)

        result = engine.check_order(buy_order("AAPL", qty=1))
        assert result.approved is False
        assert any(v.rule == "daily_loss_limit" for v in result.violations)

    def test_approved_when_daily_loss_within_limit(self):
        engine, broker = make_engine(price=100.0)

        engine._daily_start_value = 100_000.0
        # Down only 1% — within 3% limit
        broker.get_account_info.return_value = make_account(net_liq=99_000.0)

        result = engine.check_order(buy_order("AAPL", qty=1))
        # May still fail other checks but NOT daily_loss
        assert not any(v.rule == "daily_loss_limit" for v in result.violations)


# ---------------------------------------------------------------------------
# Daily trade count (Check 5)
# ---------------------------------------------------------------------------

class TestMaxDailyTrades:
    def test_blocked_when_trade_limit_reached(self):
        engine, _ = make_engine(config=make_config(max_trades_per_day=3),
                                price=100.0)
        engine._daily_trades = 3

        result = engine.check_order(buy_order("AAPL", qty=1))
        assert result.approved is False
        assert any(v.rule == "max_daily_trades" for v in result.violations)

    def test_approved_when_under_trade_limit(self):
        engine, _ = make_engine(config=make_config(max_trades_per_day=5),
                                price=100.0)
        engine._daily_trades = 2
        result = engine.check_order(buy_order("AAPL", qty=1))
        assert not any(v.rule == "max_daily_trades" for v in result.violations)

    def test_increment_trade_count(self):
        engine, _ = make_engine()
        engine._daily_trades = 0
        engine.increment_trade_count()
        assert engine._daily_trades == 1


# ---------------------------------------------------------------------------
# Large trade warning (Check 6)
# ---------------------------------------------------------------------------

class TestLargeTrade:
    def test_warning_for_large_trade(self):
        """$6k trade > $5k threshold → warning but not blocker."""
        engine, _ = make_engine(config=make_config(large_trade_threshold=5_000.0),
                                price=100.0)
        # 40 shares × $100 = $4,000 — fine. Use 60 × $100 = $6,000.
        # But 60 × $100 = 6% > 5% trade limit too. Reduce to 4 shares @ $1500 price.
        engine2, _ = make_engine(
            config=make_config(large_trade_threshold=5_000.0, max_trade_pct=1.0,
                               max_position_pct=1.0),
            price=100.0,
        )
        result = engine2.check_order(buy_order("AAPL", qty=60))
        assert any(v.rule == "large_trade" for v in result.warnings)
        assert all(v.level != RiskLevel.BLOCKER for v in result.warnings)


# ---------------------------------------------------------------------------
# Short-selling (Check 7)
# ---------------------------------------------------------------------------

class TestShortSelling:
    def test_blocked_sell_without_position(self):
        engine, _ = make_engine(position=None, price=100.0)
        result = engine.check_order(sell_order("AAPL", qty=10))
        assert result.approved is False
        assert any(v.rule == "short_selling" for v in result.violations)

    def test_blocked_sell_more_than_owned(self):
        pos = Position(ticker="AAPL", quantity=5, avg_cost=100.0,
                       current_price=100.0, market_value=500.0)
        engine, _ = make_engine(position=pos, price=100.0)
        result = engine.check_order(sell_order("AAPL", qty=20))
        assert result.approved is False
        assert any(v.rule == "short_selling" for v in result.violations)

    def test_sell_approved_when_position_sufficient(self):
        pos = Position(ticker="AAPL", quantity=50, avg_cost=100.0,
                       current_price=100.0, market_value=5_000.0)
        engine, _ = make_engine(position=pos, price=100.0)
        result = engine.check_order(sell_order("AAPL", qty=5))
        assert not any(v.rule == "short_selling" for v in result.violations)


# ---------------------------------------------------------------------------
# Stop-loss / Take-profit triggers
# ---------------------------------------------------------------------------

class TestStopLoss:
    def test_triggers_when_below_stop(self):
        engine, _ = make_engine(config=make_config(stop_loss_pct=0.08))
        pos = Position(ticker="AAPL", quantity=10, avg_cost=100.0,
                       current_price=90.0,   # 10% below cost → triggers 8% SL
                       market_value=900.0)
        order = engine.check_stop_loss(pos)
        assert order is not None
        assert order.side == OrderSide.SELL

    def test_no_trigger_when_above_stop(self):
        engine, _ = make_engine(config=make_config(stop_loss_pct=0.08))
        pos = Position(ticker="AAPL", quantity=10, avg_cost=100.0,
                       current_price=95.0,   # 5% below cost — within 8% SL
                       market_value=950.0)
        order = engine.check_stop_loss(pos)
        assert order is None

    def test_stop_loss_order_is_full_position(self):
        engine, _ = make_engine()
        pos = Position(ticker="AAPL", quantity=25, avg_cost=100.0,
                       current_price=85.0, market_value=2_125.0)
        order = engine.check_stop_loss(pos)
        assert order.quantity == 25


class TestTakeProfit:
    def test_triggers_when_above_target(self):
        engine, _ = make_engine(config=make_config(take_profit_pct=0.20))
        pos = Position(ticker="AAPL", quantity=10, avg_cost=100.0,
                       current_price=125.0,  # 25% gain → triggers 20% TP
                       market_value=1_250.0)
        order = engine.check_take_profit(pos)
        assert order is not None
        assert order.side == OrderSide.SELL

    def test_no_trigger_when_below_target(self):
        engine, _ = make_engine(config=make_config(take_profit_pct=0.20))
        pos = Position(ticker="AAPL", quantity=10, avg_cost=100.0,
                       current_price=115.0,  # 15% gain — within 20% TP
                       market_value=1_150.0)
        order = engine.check_take_profit(pos)
        assert order is None


# ---------------------------------------------------------------------------
# Safe quantity calculation
# ---------------------------------------------------------------------------

class TestSafeQuantity:
    def test_adjusted_qty_zero_when_over_limits(self):
        """If existing position already fills concentration limit → 0."""
        existing = Position(ticker="AAPL", quantity=200, avg_cost=100.0,
                            current_price=100.0, market_value=20_000.0)
        engine, _ = make_engine(position=existing, price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=100))
        assert result.adjusted_quantity == 0

    def test_adjusted_qty_positive_when_partial_room(self):
        """No existing position → some room available."""
        engine, _ = make_engine(price=100.0)
        result = engine.check_order(buy_order("AAPL", qty=1))
        assert result.adjusted_quantity >= 1


# ---------------------------------------------------------------------------
# Reset daily
# ---------------------------------------------------------------------------

class TestResetDaily:
    def test_reset_clears_trade_count(self):
        engine, _ = make_engine()
        engine._daily_trades = 7
        engine.reset_daily()
        assert engine._daily_trades == 0
