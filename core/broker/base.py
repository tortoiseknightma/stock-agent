"""
Base Broker Interface
=====================
Abstract interface that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """Represents a position in the portfolio."""
    ticker: str
    quantity: int                # Negative for short positions
    avg_cost: float             # Average cost basis
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, price: float):
        """Update current price and derived fields."""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.avg_cost) * self.quantity


@dataclass
class Order:
    """Represents a trading order."""
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"     # DAY, GTC, IOC, FOK
    
    # Filled in by broker
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    commission: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
    
    @property
    def estimated_value(self) -> float:
        price = self.limit_price or self.stop_price or 0.0
        return abs(self.quantity * price)


@dataclass
class OrderResult:
    """Result of an order submission or query."""
    success: bool
    order: Optional[Order] = None
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class AccountInfo:
    """Brokerage account information."""
    account_id: str = ""
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    gross_position_value: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    currency: str = "USD"
    timestamp: float = field(default_factory=time.time)


class BaseBroker(ABC):
    """
    Abstract base class for all broker implementations.
    
    Every method that touches the market must go through this interface.
    This ensures the rest of the system is broker-agnostic.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker. Returns success."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close the connection."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the broker."""
        pass
    
    # --- Account ---
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account summary (cash, buying power, P&L, etc.)."""
        pass
    
    # --- Positions ---
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        pass
    
    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker."""
        pass
    
    # --- Market Data ---
    
    @abstractmethod
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current market price for a ticker."""
        pass
    
    @abstractmethod
    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """Get detailed market data (bid, ask, volume, etc.)."""
        pass
    
    # --- Orders ---
    
    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit a trading order."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel a pending order."""
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""
        pass
    
    # --- History ---
    
    @abstractmethod
    def get_price_history(self, ticker: str, period: str = "1M",
                          bar_size: str = "1 day") -> List[Dict[str, Any]]:
        """
        Get historical price data.
        
        Returns list of dicts with keys: date, open, high, low, close, volume
        """
        pass
    
    # --- Utilities ---
    
    def calculate_position_size(self, ticker: str, target_pct: float) -> int:
        """
        Calculate number of shares for a target portfolio percentage.
        
        Args:
            ticker: Stock symbol
            target_pct: Target position as fraction of portfolio (0.05 = 5%)
        
        Returns:
            Number of shares to buy (positive) or sell (negative)
        """
        account = self.get_account_info()
        price = self.get_current_price(ticker)
        
        if not price or price <= 0:
            return 0
        
        target_value = account.net_liquidation * target_pct
        current = self.get_position(ticker)
        current_value = (current.quantity * current.current_price) if current else 0
        
        delta_value = target_value - current_value
        shares = int(delta_value / price)
        
        return shares
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a complete portfolio summary."""
        account = self.get_account_info()
        positions = self.get_positions()
        
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        position_details = []
        
        for p in positions:
            weight = (p.market_value / account.net_liquidation * 100) if account.net_liquidation > 0 else 0
            position_details.append({
                "ticker": p.ticker,
                "quantity": p.quantity,
                "avg_cost": round(p.avg_cost, 2),
                "current_price": round(p.current_price, 2),
                "market_value": round(p.market_value, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "weight_pct": round(weight, 1),
            })
        
        return {
            "account_id": account.account_id,
            "net_liquidation": round(account.net_liquidation, 2),
            "total_cash": round(account.total_cash, 2),
            "buying_power": round(account.buying_power, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "daily_pnl": round(account.daily_pnl, 2),
            "num_positions": len(positions),
            "positions": position_details,
            "timestamp": account.timestamp,
        }
