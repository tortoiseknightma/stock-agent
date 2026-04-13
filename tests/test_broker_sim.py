"""Tests for SimulatedBroker — buy/sell mechanics, account state, persistence."""

import pytest

from core.broker.simulated import SimulatedBroker
from core.broker.base import Order, OrderSide, OrderType, OrderStatus


def make_broker(tmp_path, capital=100_000.0, portfolio=None):
    """Factory: fresh broker with no saved state.

    Always seeds AAPL price data so get_current_price("AAPL") works
    without network calls. Pass portfolio={} for no positions.
    """
    state_file = str(tmp_path / "sim_state.json")
    # Use the default_portfolio to seed prices. qty=0 seeds the price
    # without creating a real share position (cash deduction is $0).
    seed = {"AAPL": 0}
    if portfolio:
        seed.update(portfolio)
    broker = SimulatedBroker(
        initial_capital=capital,
        default_portfolio=seed,
        slippage_pct=0.0,
        commission_per_trade=0.0,
        state_file=state_file,
    )
    return broker


def buy_order(ticker, qty):
    return Order(ticker=ticker, side=OrderSide.BUY,
                 order_type=OrderType.MARKET, quantity=qty)


def sell_order(ticker, qty):
    return Order(ticker=ticker, side=OrderSide.SELL,
                 order_type=OrderType.MARKET, quantity=qty)


class TestConnection:
    def test_connect_returns_true(self, tmp_path):
        broker = make_broker(tmp_path)
        assert broker.connect() is True
        assert broker.is_connected() is True

    def test_disconnect_sets_not_connected(self, tmp_path):
        broker = make_broker(tmp_path)
        broker.connect()
        broker.disconnect()
        assert broker.is_connected() is False


class TestAccountInfo:
    def test_initial_cash_equals_capital(self, tmp_path):
        broker = make_broker(tmp_path, capital=50_000.0)
        broker.connect()
        info = broker.get_account_info()
        assert info.total_cash == pytest.approx(50_000.0, rel=0.01)

    def test_net_liquidation_includes_positions(self, tmp_path):
        # Start with AAPL at $178 × 100 shares = $17,800 position
        broker = make_broker(tmp_path, capital=100_000.0,
                             portfolio={"AAPL": 100})
        broker.connect()
        info = broker.get_account_info()
        # Net liquidation should be > cash (positions add value)
        assert info.net_liquidation > info.total_cash

    def test_buying_power_is_2x_cash(self, tmp_path):
        broker = make_broker(tmp_path, capital=10_000.0)
        broker.connect()
        info = broker.get_account_info()
        assert info.buying_power == pytest.approx(info.total_cash * 2, rel=0.01)


class TestBuyOrder:
    def test_buy_reduces_cash(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()
        price = broker.get_current_price("AAPL")
        cash_before = broker.get_account_info().total_cash

        result = broker.submit_order(buy_order("AAPL", 10))

        assert result.success is True
        cash_after = broker.get_account_info().total_cash
        assert cash_after < cash_before

    def test_buy_creates_position(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()

        broker.submit_order(buy_order("AAPL", 5))
        pos = broker.get_position("AAPL")

        assert pos is not None
        assert pos.quantity == 5

    def test_buy_adds_to_existing_position(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0,
                             portfolio={"AAPL": 10})
        broker.connect()

        broker.submit_order(buy_order("AAPL", 5))
        pos = broker.get_position("AAPL")
        assert pos.quantity == 15

    def test_buy_rejected_when_insufficient_cash(self, tmp_path):
        broker = make_broker(tmp_path, capital=100.0)
        broker.connect()

        result = broker.submit_order(buy_order("AAPL", 1000))
        assert result.success is False
        assert result.order.status == OrderStatus.REJECTED

    def test_buy_order_gets_order_id(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()
        result = broker.submit_order(buy_order("AAPL", 1))
        assert result.order.order_id is not None

    def test_order_status_is_filled(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()
        result = broker.submit_order(buy_order("AAPL", 1))
        assert result.order.status == OrderStatus.FILLED
        assert result.order.filled_quantity == 1


class TestSellOrder:
    def test_sell_increases_cash(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0,
                             portfolio={"AAPL": 50})
        broker.connect()
        cash_before = broker.get_account_info().total_cash

        broker.submit_order(sell_order("AAPL", 10))
        cash_after = broker.get_account_info().total_cash

        assert cash_after > cash_before

    def test_sell_reduces_position(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0,
                             portfolio={"AAPL": 50})
        broker.connect()

        broker.submit_order(sell_order("AAPL", 20))
        pos = broker.get_position("AAPL")
        assert pos.quantity == 30

    def test_sell_all_removes_position(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0,
                             portfolio={"AAPL": 10})
        broker.connect()

        broker.submit_order(sell_order("AAPL", 10))
        pos = broker.get_position("AAPL")
        assert pos is None

    def test_sell_more_than_owned_rejected(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0,
                             portfolio={"AAPL": 5})
        broker.connect()

        result = broker.submit_order(sell_order("AAPL", 100))
        assert result.success is False

    def test_sell_without_position_rejected(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()

        result = broker.submit_order(sell_order("MSFT", 10))
        assert result.success is False


class TestOrderWithoutConnection:
    def test_submit_order_fails_when_disconnected(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        # Do NOT connect
        result = broker.submit_order(buy_order("AAPL", 1))
        assert result.success is False


class TestPriceHistory:
    def test_price_history_returns_bars(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()
        bars = broker.get_price_history("AAPL", period="1M")
        assert len(bars) > 0

    def test_price_history_bar_has_ohlcv(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()
        bars = broker.get_price_history("AAPL", period="5D")
        for bar in bars:
            for key in ("open", "high", "low", "close", "volume"):
                assert key in bar

    def test_high_gte_low_in_all_bars(self, tmp_path):
        broker = make_broker(tmp_path, capital=100_000.0)
        broker.connect()
        bars = broker.get_price_history("AAPL", period="1M")
        for bar in bars:
            assert bar["high"] >= bar["low"]


class TestStatePersistence:
    def test_state_saved_and_reloaded(self, tmp_path):
        state_file = str(tmp_path / "state.json")

        # Seed AAPL price so the buy order is accepted
        broker1 = SimulatedBroker(
            initial_capital=100_000.0,
            default_portfolio={"AAPL": 0},  # seeds price without position
            slippage_pct=0.0,
            commission_per_trade=0.0,
            state_file=state_file,
        )
        broker1.connect()
        result = broker1.submit_order(buy_order("AAPL", 5))
        assert result.success, f"Buy failed: {result.error_message}"
        broker1.disconnect()

        # Second broker loads state; also seed AAPL so _load_state can restore
        broker2 = SimulatedBroker(
            initial_capital=100_000.0,
            default_portfolio={"AAPL": 0},
            slippage_pct=0.0,
            commission_per_trade=0.0,
            state_file=state_file,
        )
        broker2.connect()
        pos = broker2.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 5
