"""Tests for backtest.BacktestBroker."""

import datetime
import pytest
from unittest.mock import MagicMock

from backtest.broker import BacktestBroker
from backtest.data_manager import HistoricalDataManager
from core.broker.base import Order, OrderSide, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dates(start="2023-01-03", days=10):
    import pandas as pd
    return [d.date() for d in pd.bdate_range(start=start, periods=days)]


def make_dm(prices: dict) -> HistoricalDataManager:
    """Build a mock HistoricalDataManager with canned price lookups."""
    dm = MagicMock(spec=HistoricalDataManager)
    dm.get_close.side_effect = lambda ticker, date: prices.get(str(date))
    dm.get_bars_up_to.return_value = [
        {"date": str(d), "open": 150.0, "high": 152.0,
         "low": 149.0, "close": 151.0, "volume": 1_000_000}
        for d in make_dates()
    ]
    return dm


def make_broker(prices=None, capital=100_000.0) -> BacktestBroker:
    if prices is None:
        prices = {str(d): 150.0 + i for i, d in enumerate(make_dates())}
    dm = make_dm(prices)
    broker = BacktestBroker(dm, initial_capital=capital,
                            slippage_pct=0.0, commission_per_trade=0.0)
    return broker


def buy_order(ticker="AAPL", qty=10) -> Order:
    return Order(ticker=ticker, side=OrderSide.BUY,
                 order_type=OrderType.MARKET, quantity=qty)


def sell_order(ticker="AAPL", qty=10) -> Order:
    return Order(ticker=ticker, side=OrderSide.SELL,
                 order_type=OrderType.MARKET, quantity=qty)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_returns_true(self):
        broker = make_broker()
        assert broker.connect() is True

    def test_is_connected_after_connect(self):
        broker = make_broker()
        broker.connect()
        assert broker.is_connected() is True

    def test_is_disconnected_initially(self):
        broker = make_broker()
        assert broker.is_connected() is False

    def test_submit_order_fails_when_not_connected(self):
        broker = make_broker()
        result = broker.submit_order(buy_order())
        assert result.success is False


# ---------------------------------------------------------------------------
# set_date / get_current_price
# ---------------------------------------------------------------------------

class TestSetDate:
    def test_get_current_price_returns_none_before_set_date(self):
        broker = make_broker()
        broker.connect()
        assert broker.get_current_price("AAPL") is None

    def test_get_current_price_returns_correct_close(self):
        dates = make_dates()
        prices = {str(d): 150.0 + i for i, d in enumerate(dates)}
        broker = make_broker(prices)
        broker.connect()
        broker.set_date(dates[0])
        assert broker.get_current_price("AAPL") == pytest.approx(150.0)

    def test_set_date_updates_position_price(self):
        dates = make_dates()
        prices = {str(d): 150.0 + i for i, d in enumerate(dates)}
        broker = make_broker(prices, capital=100_000)
        broker.connect()

        broker.set_date(dates[0])
        broker.submit_order(buy_order(qty=10))
        pos_before = broker.get_position("AAPL")
        assert pos_before.current_price == pytest.approx(150.0)

        broker.set_date(dates[5])   # price = 155.0
        pos_after = broker.get_position("AAPL")
        assert pos_after.current_price == pytest.approx(155.0)


# ---------------------------------------------------------------------------
# submit_order — buy
# ---------------------------------------------------------------------------

class TestBuyOrder:
    def test_buy_reduces_cash(self):
        dates = make_dates()
        broker = make_broker()
        broker.connect()
        broker.set_date(dates[0])
        cash_before = broker.get_account_info().total_cash
        broker.submit_order(buy_order(qty=10))
        cash_after = broker.get_account_info().total_cash
        assert cash_after < cash_before

    def test_buy_creates_position(self):
        dates = make_dates()
        broker = make_broker()
        broker.connect()
        broker.set_date(dates[0])
        broker.submit_order(buy_order(qty=10))
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 10

    def test_buy_insufficient_cash_rejected(self):
        dates = make_dates()
        broker = make_broker(capital=100.0)  # tiny capital
        broker.connect()
        broker.set_date(dates[0])
        result = broker.submit_order(buy_order(qty=1000))
        assert result.success is False

    def test_buy_returns_success(self):
        dates = make_dates()
        broker = make_broker()
        broker.connect()
        broker.set_date(dates[0])
        result = broker.submit_order(buy_order(qty=5))
        assert result.success is True


# ---------------------------------------------------------------------------
# submit_order — sell
# ---------------------------------------------------------------------------

class TestSellOrder:
    def _buy_first(self, broker, dates, qty=10):
        broker.set_date(dates[0])
        broker.submit_order(buy_order(qty=qty))

    def test_sell_increases_cash(self):
        dates = make_dates()
        prices = {str(d): 150.0 for d in dates}
        broker = make_broker(prices)
        broker.connect()
        self._buy_first(broker, dates)
        cash_before = broker.get_account_info().total_cash
        broker.submit_order(sell_order(qty=5))
        assert broker.get_account_info().total_cash > cash_before

    def test_sell_removes_position_when_selling_all(self):
        dates = make_dates()
        broker = make_broker()
        broker.connect()
        self._buy_first(broker, dates)
        broker.submit_order(sell_order(qty=10))
        assert broker.get_position("AAPL") is None

    def test_sell_without_position_rejected(self):
        dates = make_dates()
        broker = make_broker()
        broker.connect()
        broker.set_date(dates[0])
        result = broker.submit_order(sell_order(qty=5))
        assert result.success is False

    def test_sell_insufficient_shares_rejected(self):
        dates = make_dates()
        broker = make_broker()
        broker.connect()
        self._buy_first(broker, dates, qty=5)
        result = broker.submit_order(sell_order(qty=10))
        assert result.success is False


# ---------------------------------------------------------------------------
# get_price_history — no look-ahead
# ---------------------------------------------------------------------------

class TestPriceHistory:
    def test_get_price_history_no_lookahead(self):
        """All returned bars must be on or before the current date."""
        dates = make_dates(days=60)
        cutoff = dates[30]
        bars_before_cutoff = [
            {"date": str(d), "open": 150.0, "high": 152.0,
             "low": 149.0, "close": 151.0, "volume": 1_000_000}
            for d in dates[:31]  # 31 bars ≤ cutoff
        ]
        dm = MagicMock(spec=HistoricalDataManager)
        dm.get_close.return_value = 150.0
        dm.get_bars_up_to.return_value = bars_before_cutoff

        broker = BacktestBroker(dm, initial_capital=100_000,
                                slippage_pct=0.0, commission_per_trade=0.0)
        broker.connect()
        broker.set_date(cutoff)

        bars = broker.get_price_history("AAPL", period="6M")
        for b in bars:
            assert b["date"] <= str(cutoff)


# ---------------------------------------------------------------------------
# get_account_info
# ---------------------------------------------------------------------------

class TestAccountInfo:
    def test_net_liquidation_equals_cash_when_no_positions(self):
        broker = make_broker(capital=50_000)
        broker.connect()
        info = broker.get_account_info()
        assert info.net_liquidation == pytest.approx(50_000, abs=1.0)
        assert info.total_cash == pytest.approx(50_000, abs=1.0)

    def test_net_liquidation_includes_position_value(self):
        dates = make_dates()
        prices = {str(d): 100.0 for d in dates}
        broker = make_broker(prices, capital=10_000)
        broker.connect()
        broker.set_date(dates[0])
        broker.submit_order(buy_order("AAPL", qty=10))  # costs 1000
        info = broker.get_account_info()
        # 9000 cash + 10 * 100 = 10000
        assert info.net_liquidation == pytest.approx(10_000, abs=5.0)
