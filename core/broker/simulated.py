"""
Simulated Broker
=================
Local mock broker for development and backtesting.

Features:
- Geometric Brownian Motion (GBM) price simulation
- Realistic slippage and commission modeling
- Persistent state across sessions (JSON file)
- Configurable initial portfolio

This is the DEFAULT mode — use it to test strategies before
connecting to real money.
"""

import json
import math
import random
import time
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from .base import (
    BaseBroker, Position, Order, OrderResult, AccountInfo,
    OrderSide, OrderType, OrderStatus
)


class SimulatedBroker(BaseBroker):
    """
    Simulated broker with realistic price dynamics.
    
    Price model: Geometric Brownian Motion
        dS = mu * S * dt + sigma * S * dW
    
    Where:
        mu = drift (slight upward bias, configurable)
        sigma = volatility (calibrated per ticker)
    """
    
    # Default volatility per ticker (annualized)
    DEFAULT_VOLATILITY = {
        "AAPL": 0.25, "MSFT": 0.22, "GOOGL": 0.28, "NVDA": 0.45,
        "AMZN": 0.30, "JPM": 0.20, "V": 0.18, "TSLA": 0.55,
        "META": 0.35, "SPY": 0.15, "QQQ": 0.20, "AMD": 0.40,
    }
    DEFAULT_VOL = 0.30  # Fallback volatility
    
    # Default starting prices (approximate)
    DEFAULT_PRICES = {
        "AAPL": 178.0, "MSFT": 378.0, "GOOGL": 141.0, "NVDA": 875.0,
        "AMZN": 178.0, "JPM": 195.0, "V": 275.0, "TSLA": 248.0,
        "META": 505.0, "SPY": 510.0, "QQQ": 438.0, "AMD": 172.0,
    }
    DEFAULT_PRICE = 100.0
    
    def __init__(self, initial_capital: float = 100_000.0,
                 default_portfolio: Dict[str, int] = None,
                 slippage_pct: float = 0.001,
                 commission_per_trade: float = 1.0,
                 state_file: str = "data/sim_state.json",
                 drift: float = 0.05):     # 5% annual drift
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission = commission_per_trade
        self.state_file = Path(state_file)
        self.drift = drift  # Annual drift rate
        
        self._positions: Dict[str, Position] = {}
        self._cash: float = initial_capital
        self._orders: Dict[str, Order] = {}
        self._price_history: Dict[str, List[float]] = {}
        self._current_prices: Dict[str, float] = {}
        self._connected = False
        self._daily_start_value: float = initial_capital
        
        # Initialize with default portfolio
        if default_portfolio:
            for ticker, qty in default_portfolio.items():
                price = self.DEFAULT_PRICES.get(ticker, self.DEFAULT_PRICE)
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_cost=price,
                    current_price=price,
                    market_value=qty * price,
                )
                self._cash -= qty * price
                self._current_prices[ticker] = price
                self._price_history[ticker] = [price] * 30  # 30-day history

            # Guard: if portfolio cost exceeds initial_capital, adjust so cash >= 0.
            # This keeps accounting consistent regardless of price changes.
            if self._cash < 0:
                shortfall = -self._cash
                print(f"Warning: default portfolio costs ${initial_capital - self._cash:,.0f} "
                      f"but initial_capital is ${initial_capital:,.0f}. "
                      f"Adjusting initial_capital by ${shortfall:,.0f} to avoid negative cash.")
                self.initial_capital += shortfall
                self._cash = 0.0

        # Try to load saved state
        self._load_state()
    
    def connect(self) -> bool:
        """Connect to simulated broker."""
        self._connected = True
        self._daily_start_value = self._calculate_total_value()
        return True
    
    def disconnect(self):
        """Disconnect and save state."""
        self._save_state()
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected
    
    # --- Account ---
    
    def get_account_info(self) -> AccountInfo:
        total_value = self._calculate_total_value()
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        
        return AccountInfo(
            account_id="SIM-001",
            net_liquidation=total_value,
            total_cash=self._cash,
            buying_power=self._cash * 2,  # Simulate 2x margin
            gross_position_value=total_value - self._cash,
            daily_pnl=total_value - self._daily_start_value,
            unrealized_pnl=unrealized,
            realized_pnl=total_value - self.initial_capital - unrealized,
        )
    
    # --- Positions ---
    
    def get_positions(self) -> List[Position]:
        # Update prices with random walk
        self._simulate_price_tick()
        return list(self._positions.values())
    
    def get_position(self, ticker: str) -> Optional[Position]:
        self._simulate_price_tick()
        return self._positions.get(ticker)
    
    # --- Market Data ---
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        self._simulate_price_tick()
        return self._current_prices.get(ticker)
    
    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        price = self.get_current_price(ticker)
        if not price:
            return {}
        
        spread = price * 0.0005  # 5 bps spread
        return {
            "ticker": ticker,
            "last": price,
            "bid": round(price - spread / 2, 2),
            "ask": round(price + spread / 2, 2),
            "volume": random.randint(100_000, 10_000_000),
            "timestamp": time.time(),
        }
    
    # --- Orders ---
    
    def submit_order(self, order: Order) -> OrderResult:
        """Submit and immediately fill a simulated order."""
        if not self._connected:
            return OrderResult(success=False, error_message="Not connected")
        
        # Generate order ID
        order.order_id = str(uuid.uuid4())[:8]
        order.timestamp = time.time()
        
        # Get current price
        price = self.get_current_price(order.ticker)
        if not price:
            order.status = OrderStatus.REJECTED
            return OrderResult(success=False, order=order,
                             error_message=f"No price data for {order.ticker}")
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)
        
        fill_price = round(fill_price, 2)
        trade_value = order.quantity * fill_price
        
        # Validate
        if order.side == OrderSide.BUY:
            if trade_value + self.commission > self._cash:
                order.status = OrderStatus.REJECTED
                return OrderResult(success=False, order=order,
                                 error_message=f"Insufficient cash: need ${trade_value:.2f}, have ${self._cash:.2f}")
        
        if order.side == OrderSide.SELL:
            pos = self._positions.get(order.ticker)
            if not pos or pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return OrderResult(success=False, order=order,
                                 error_message=f"Insufficient shares: need {order.quantity}, have {pos.quantity if pos else 0}")
        
        # Execute
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = self.commission
        
        # Update positions
        if order.side == OrderSide.BUY:
            self._buy(order.ticker, order.quantity, fill_price)
        else:
            self._sell(order.ticker, order.quantity, fill_price)
        
        # Deduct commission
        self._cash -= self.commission
        
        self._orders[order.order_id] = order
        self._save_state()
        
        return OrderResult(success=True, order=order)
    
    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order (in simulated mode, orders fill instantly)."""
        order = self._orders.get(order_id)
        if not order:
            return OrderResult(success=False, error_message="Order not found")
        
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            return OrderResult(success=True, order=order)
        
        return OrderResult(success=False, order=order,
                         error_message=f"Cannot cancel order in status {order.status.value}")
    
    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_active]
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)
    
    # --- History ---
    
    def get_price_history(self, ticker: str, period: str = "1M",
                          bar_size: str = "1 day") -> List[Dict[str, Any]]:
        """Generate simulated price history using GBM."""
        # Determine number of bars
        num_bars = {"1D": 1, "5D": 5, "1M": 22, "3M": 66, "6M": 132, "1Y": 252}.get(period, 22)
        
        current_price = self._current_prices.get(
            ticker, self.DEFAULT_PRICES.get(ticker, self.DEFAULT_PRICE))
        volatility = self.DEFAULT_VOLATILITY.get(ticker, self.DEFAULT_VOL)
        
        # Generate prices using GBM
        dt = 1 / 252  # Daily
        prices = [current_price]
        
        for _ in range(num_bars - 1):
            z = random.gauss(0, 1)
            new_price = prices[-1] * math.exp(
                (self.drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * z
            )
            prices.append(round(max(new_price, 0.01), 2))
        
        # Reverse so oldest first
        prices = prices[::-1]
        
        # Build OHLCV bars
        bars = []
        base_time = time.time() - num_bars * 86400
        for i, close in enumerate(prices):
            high = close * (1 + abs(random.gauss(0, 0.005)))
            low = close * (1 - abs(random.gauss(0, 0.005)))
            open_p = prices[i-1] if i > 0 else close
            bars.append({
                "date": base_time + i * 86400,
                "open": round(open_p, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": random.randint(500_000, 15_000_000),
            })
        
        return bars
    
    # --- Internal ---
    
    def _simulate_price_tick(self):
        """Simulate a single price tick for all positions."""
        dt = 1 / (252 * 390)  # 1 minute in trading year
        
        for ticker in list(self._current_prices.keys()):
            volatility = self.DEFAULT_VOLATILITY.get(ticker, self.DEFAULT_VOL)
            price = self._current_prices[ticker]
            
            z = random.gauss(0, 1)
            new_price = price * math.exp(
                (self.drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * z
            )
            new_price = round(max(new_price, 0.01), 2)
            
            self._current_prices[ticker] = new_price
            
            # Update position
            if ticker in self._positions:
                self._positions[ticker].update_price(new_price)
            
            # Append to history
            if ticker in self._price_history:
                self._price_history[ticker].append(new_price)
                if len(self._price_history[ticker]) > 390:  # Keep ~1 trading day
                    self._price_history[ticker] = self._price_history[ticker][-390:]
    
    def _buy(self, ticker: str, quantity: int, price: float):
        """Execute a buy."""
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
        """Execute a sell."""
        proceeds = quantity * price
        self._cash += proceeds
        
        pos = self._positions[ticker]
        pnl = (price - pos.avg_cost) * quantity
        pos.realized_pnl += pnl
        pos.quantity -= quantity
        
        if pos.quantity <= 0:
            del self._positions[ticker]
        else:
            pos.update_price(price)
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value."""
        self._simulate_price_tick()
        positions_value = sum(
            p.quantity * self._current_prices.get(p.ticker, p.current_price)
            for p in self._positions.values()
        )
        return self._cash + positions_value
    
    def _save_state(self):
        """Save state to disk for persistence across sessions."""
        state = {
            "cash": self._cash,
            "initial_capital": self.initial_capital,
            "daily_start_value": self._daily_start_value,
            "positions": {
                t: {
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "realized_pnl": p.realized_pnl,
                }
                for t, p in self._positions.items()
            },
            "prices": self._current_prices,
            "orders": {
                oid: {
                    "ticker": o.ticker, "side": o.side.value,
                    "order_type": o.order_type.value, "quantity": o.quantity,
                    "filled_price": o.filled_price, "status": o.status.value,
                    "commission": o.commission, "timestamp": o.timestamp,
                }
                for oid, o in self._orders.items()
            },
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            self._cash = state.get("cash", self._cash)
            self._daily_start_value = state.get("daily_start_value", self._daily_start_value)
            
            for ticker, pdata in state.get("positions", {}).items():
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=pdata["quantity"],
                    avg_cost=pdata["avg_cost"],
                    current_price=pdata.get("current_price", pdata["avg_cost"]),
                    realized_pnl=pdata.get("realized_pnl", 0),
                )
            
            self._current_prices = state.get("prices", {})
            
            for oid, odata in state.get("orders", {}).items():
                order = Order(
                    ticker=odata["ticker"],
                    side=OrderSide(odata["side"]),
                    order_type=OrderType(odata["order_type"]),
                    quantity=odata["quantity"],
                )
                order.order_id = oid
                order.status = OrderStatus(odata["status"])
                order.filled_price = odata.get("filled_price")
                order.commission = odata.get("commission", 0)
                order.timestamp = odata.get("timestamp", 0)
                self._orders[oid] = order
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load state: {e}")
