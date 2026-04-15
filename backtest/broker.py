"""
Backtest Broker
================
A deterministic broker that replays historical OHLCV data day by day.
Implements BaseBroker so TradeExecutor, RiskEngine, and the rest of the
execution layer work unchanged.

Key difference from SimulatedBroker:
- Prices are deterministic (from cached OHLCV), not random (GBM)
- No state file persistence — ephemeral per backtest run
- set_date(date) is the time-control hook; the engine calls it each day
- No look-ahead: get_current_price() and get_price_history() only see
  data available on or before the current backtest date
"""

import datetime
import uuid
import time
from typing import Any, Dict, List, Optional

from core.broker.base import (
    BaseBroker, BaseBroker, AccountInfo, Order, OrderResult,
    OrderSide, OrderStatus, OrderType, Position,
)
from backtest.data_manager import HistoricalDataManager


# Period string → approximate trading-day lookback
_PERIOD_TO_DAYS: Dict[str, int] = {
    "1D": 1, "5D": 5,
    "1M": 22, "3M": 66, "6M": 132, "1Y": 252, "2Y": 504,
    # yfinance-style aliases
    "1mo": 22, "3mo": 66, "6mo": 132, "1y": 252, "2y": 504,
}


class BacktestBroker(BaseBroker):
    """
    A deterministic broker that replays historical prices for backtesting.

    Usage (via BacktestEngine — do not use directly)::

        broker = BacktestBroker(data_manager, initial_capital=100_000)
        broker.connect()

        for date in trading_dates:
            broker.set_date(date)          # advance time
            price = broker.get_current_price("AAPL")
            # ... run analysis, submit orders via TradeExecutor ...
    """

    def __init__(
        self,
        data_manager: HistoricalDataManager,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.001,
        commission_per_trade: float = 1.0,
    ):
        self._data = data_manager
        self.initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._current_date: Optional[datetime.date] = None
        self._connected = False
        self.slippage_pct = slippage_pct
        self.commission = commission_per_trade
        # Track realized P&L for daily reporting
        self._realized_pnl: float = 0.0
        self._daily_start_value: float = initial_capital

    # ------------------------------------------------------------------
    # Time control (called by BacktestEngine, not part of BaseBroker)
    # ------------------------------------------------------------------

    def set_date(self, date: datetime.date):
        """
        Advance the broker to *date*.
        Updates all open position prices to that day's closing price.
        """
        self._current_date = date
        # Refresh position prices
        for ticker, pos in self._positions.items():
            close = self._data.get_close(ticker, date)
            if close is not None:
                pos.update_price(close)

    def get_date(self) -> Optional[datetime.date]:
        return self._current_date

    def reset_daily(self):
        """Record portfolio value at start of day (for daily P&L tracking)."""
        self._daily_start_value = self._calculate_total_value()

    # ------------------------------------------------------------------
    # BaseBroker — connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # BaseBroker — account & positions
    # ------------------------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        total_value = self._calculate_total_value()
        positions_value = total_value - self._cash
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        daily_pnl = total_value - self._daily_start_value
        return AccountInfo(
            account_id="backtest",
            net_liquidation=round(total_value, 2),
            total_cash=round(self._cash, 2),
            buying_power=round(self._cash, 2),
            gross_position_value=round(positions_value, 2),
            daily_pnl=round(daily_pnl, 2),
            unrealized_pnl=round(unrealized, 2),
            realized_pnl=round(self._realized_pnl, 2),
        )

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._positions.get(ticker.upper())

    # ------------------------------------------------------------------
    # BaseBroker — market data
    # ------------------------------------------------------------------

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Return closing price for *ticker* on the current backtest date."""
        if self._current_date is None:
            return None
        return self._data.get_close(ticker.upper(), self._current_date)

    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        price = self.get_current_price(ticker) or 0.0
        return {
            "ticker": ticker.upper(),
            "last": price,
            "bid": price,
            "ask": price,
            "volume": 0,
            "timestamp": time.time(),
        }

    def get_price_history(
        self, ticker: str, period: str = "1M", bar_size: str = "1 day"
    ) -> List[Dict[str, Any]]:
        """
        Return historical bars up to (and including) the current backtest date.
        Never returns data after the current date — no look-ahead bias.
        """
        if self._current_date is None:
            return []
        lookback = _PERIOD_TO_DAYS.get(period, 132)
        return self._data.get_bars_up_to(ticker.upper(), self._current_date, lookback)

    # ------------------------------------------------------------------
    # BaseBroker — orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> OrderResult:
        """Fill immediately at today's close with slippage."""
        if not self._connected:
            return OrderResult(success=False, error_message="Not connected")

        order.order_id = str(uuid.uuid4())[:8]
        order.timestamp = time.time()

        price = self.get_current_price(order.ticker)
        if not price:
            order.status = OrderStatus.REJECTED
            return OrderResult(
                success=False, order=order,
                error_message=f"No price data for {order.ticker} on {self._current_date}"
            )

        # Apply slippage
        fill_price = (
            price * (1 + self.slippage_pct)
            if order.side == OrderSide.BUY
            else price * (1 - self.slippage_pct)
        )
        fill_price = round(fill_price, 2)
        trade_value = order.quantity * fill_price

        # Validate
        if order.side == OrderSide.BUY:
            if trade_value + self.commission > self._cash:
                order.status = OrderStatus.REJECTED
                return OrderResult(
                    success=False, order=order,
                    error_message=(
                        f"Insufficient cash: need ${trade_value:.2f}, "
                        f"have ${self._cash:.2f}"
                    ),
                )

        if order.side == OrderSide.SELL:
            pos = self._positions.get(order.ticker.upper())
            if not pos or pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                have = pos.quantity if pos else 0
                return OrderResult(
                    success=False, order=order,
                    error_message=(
                        f"Insufficient shares: need {order.quantity}, have {have}"
                    ),
                )

        # Execute
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = self.commission

        if order.side == OrderSide.BUY:
            self._buy(order.ticker.upper(), order.quantity, fill_price)
        else:
            self._sell(order.ticker.upper(), order.quantity, fill_price)

        self._cash -= self.commission
        self._orders[order.order_id] = order
        return OrderResult(success=True, order=order)

    def cancel_order(self, order_id: str) -> OrderResult:
        order = self._orders.get(order_id)
        if not order:
            return OrderResult(success=False, error_message="Order not found")
        if order.is_active:
            order.status = OrderStatus.CANCELLED
        return OrderResult(success=True, order=order)

    def get_open_orders(self) -> List[Order]:
        return []  # Backtest broker fills instantly — no open orders

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    # ------------------------------------------------------------------
    # Internal helpers (logic from SimulatedBroker lines 321-354)
    # ------------------------------------------------------------------

    def _buy(self, ticker: str, quantity: int, price: float):
        cost = quantity * price
        self._cash -= cost

        if ticker in self._positions:
            pos = self._positions[ticker]
            total_cost = pos.avg_cost * pos.quantity + cost
            pos.quantity += quantity
            pos.avg_cost = round(total_cost / pos.quantity, 2)
            pos.update_price(price)
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                market_value=quantity * price,
            )

    def _sell(self, ticker: str, quantity: int, price: float):
        proceeds = quantity * price
        self._cash += proceeds

        pos = self._positions[ticker]
        pnl = (price - pos.avg_cost) * quantity
        self._realized_pnl += pnl
        pos.realized_pnl += pnl
        pos.quantity -= quantity

        if pos.quantity <= 0:
            del self._positions[ticker]
        else:
            pos.update_price(price)

    def _calculate_total_value(self) -> float:
        positions_value = sum(
            p.quantity * (self._data.get_close(p.ticker, self._current_date) or p.current_price)
            for p in self._positions.values()
        ) if self._current_date else sum(
            p.market_value for p in self._positions.values()
        )
        return self._cash + positions_value
