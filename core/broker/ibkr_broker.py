"""
IBKR Broker Implementation
============================
Connects to Interactive Brokers via ib_insync.

Supports both Paper Trading (port 7497) and Live Trading (port 7496).
Requires TWS or IB Gateway running locally.

IMPORTANT: Live trading requires setting trading_mode='live' in config
AND passing --confirm-live flag on the command line.
"""

import time
from typing import List, Optional, Dict, Any
from decimal import Decimal

from .base import (
    BaseBroker, Position, Order, OrderResult, AccountInfo,
    OrderSide, OrderType, OrderStatus
)


class IBKRBroker(BaseBroker):
    """
    Interactive Brokers broker implementation using ib_insync.
    
    Prerequisites:
        1. Install: pip install ib_insync
        2. Run TWS or IB Gateway
        3. Enable API connections in TWS settings
        4. Set correct port: 7497 (paper) or 7496 (live)
    
    Usage:
        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker.connect()
        
        account = broker.get_account_info()
        positions = broker.get_positions()
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, 
                                  order_type=OrderType.MARKET, quantity=10))
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, timeout: int = 30,
                 readonly: bool = True):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self._ib = None
        self._connected = False
    
    def _ensure_ib(self):
        """Lazy import ib_insync."""
        if self._ib is None:
            try:
                from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
                self._ib = IB()
                self._Stock = Stock
                self._MarketOrder = MarketOrder
                self._LimitOrder = LimitOrder
                self._StopOrder = StopOrder
            except ImportError:
                raise ImportError(
                    "ib_insync is required for IBKR integration. "
                    "Install with: pip install ib_insync"
                )
    
    def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        self._ensure_ib()
        
        try:
            self._ib.connect(
                self.host, self.port, clientId=self.client_id,
                timeout=self.timeout, readonly=self.readonly
            )
            self._connected = True
            return True
        except Exception as e:
            print(f"IBKR connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected and self._ib and self._ib.isConnected()
    
    # --- Account ---
    
    def get_account_info(self) -> AccountInfo:
        if not self.is_connected():
            return AccountInfo()
        
        account_values = self._ib.accountValues()
        summary = {v.tag: v.value for v in account_values}
        
        return AccountInfo(
            account_id=summary.get("AccountId", ""),
            net_liquidation=float(summary.get("NetLiquidation", 0)),
            total_cash=float(summary.get("TotalCashValue", 0)),
            buying_power=float(summary.get("BuyingPower", 0)),
            gross_position_value=float(summary.get("GrossPositionValue", 0)),
            daily_pnl=float(summary.get("DailyPnL", 0)),
            unrealized_pnl=float(summary.get("UnrealizedPnL", 0)),
            realized_pnl=float(summary.get("RealizedPnL", 0)),
            margin_used=float(summary.get("MaintMarginReq", 0)),
            currency=summary.get("Currency", "USD"),
        )
    
    # --- Positions ---
    
    def get_positions(self) -> List[Position]:
        if not self.is_connected():
            return []
        
        ib_positions = self._ib.positions()
        result = []
        
        for ib_pos in ib_positions:
            contract = ib_pos.contract
            ticker = contract.symbol
            
            # Get current price
            ticker_data = self._ib.reqMktData(contract, "", True, False)
            self._ib.sleep(1)  # Wait for data
            current_price = ticker_data.last if ticker_data.last and ticker_data.last > 0 else ticker_data.close
            
            pos = Position(
                ticker=ticker,
                quantity=int(ib_pos.position),
                avg_cost=float(ib_pos.avgCost),
                current_price=float(current_price) if current_price else 0,
                market_value=float(ib_pos.position) * float(current_price) if current_price else 0,
                unrealized_pnl=float(ib_pos.unrealizedPNL) if ib_pos.unrealizedPNL else 0,
            )
            result.append(pos)
        
        return result
    
    def get_position(self, ticker: str) -> Optional[Position]:
        positions = self.get_positions()
        for p in positions:
            if p.ticker == ticker:
                return p
        return None
    
    # --- Market Data ---
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        if not self.is_connected():
            return None
        
        contract = self._Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        
        ticker_data = self._ib.reqMktData(contract, "", True, False)
        self._ib.sleep(2)
        
        if ticker_data.last and ticker_data.last > 0:
            return float(ticker_data.last)
        elif ticker_data.close and ticker_data.close > 0:
            return float(ticker_data.close)
        return None
    
    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        if not self.is_connected():
            return {}
        
        contract = self._Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        
        ticker_data = self._ib.reqMktData(contract, "", False, False)
        self._ib.sleep(2)
        
        return {
            "ticker": ticker,
            "bid": float(ticker_data.bid) if ticker_data.bid else None,
            "ask": float(ticker_data.ask) if ticker_data.ask else None,
            "last": float(ticker_data.last) if ticker_data.last else None,
            "close": float(ticker_data.close) if ticker_data.close else None,
            "volume": int(ticker_data.volume) if ticker_data.volume else None,
            "high": float(ticker_data.high) if ticker_data.high else None,
            "low": float(ticker_data.low) if ticker_data.low else None,
        }
    
    # --- Orders ---
    
    def submit_order(self, order: Order) -> OrderResult:
        if not self.is_connected():
            return OrderResult(success=False, error_message="Not connected")
        
        if self.readonly:
            return OrderResult(success=False, error_message="Connection is read-only")
        
        try:
            contract = self._Stock(order.ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            
            action = "BUY" if order.side == OrderSide.BUY else "SELL"
            
            if order.order_type == OrderType.MARKET:
                ib_order = self._MarketOrder(action, order.quantity)
            elif order.order_type == OrderType.LIMIT:
                ib_order = self._LimitOrder(action, order.quantity, order.limit_price)
            elif order.order_type == OrderType.STOP:
                ib_order = self._StopOrder(action, order.quantity, order.stop_price)
            else:
                return OrderResult(success=False, error_message=f"Unsupported order type: {order.order_type}")
            
            ib_order.tif = order.time_in_force
            
            trade = self._ib.placeOrder(contract, ib_order)
            self._ib.sleep(2)
            
            # Map result
            order.order_id = str(trade.order.orderId)
            order.status = self._map_order_status(trade.orderStatus.status)
            
            if trade.orderStatus.status == "Filled":
                order.filled_quantity = int(trade.orderStatus.filled)
                order.filled_price = float(trade.orderStatus.avgFillPrice)
                order.commission = float(trade.orderStatus.commission or 0)
            
            return OrderResult(success=True, order=order)
            
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))
    
    def cancel_order(self, order_id: str) -> OrderResult:
        if not self.is_connected():
            return OrderResult(success=False, error_message="Not connected")
        
        try:
            open_trades = self._ib.openTrades()
            for trade in open_trades:
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return OrderResult(success=True)
            
            return OrderResult(success=False, error_message="Order not found")
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))
    
    def get_open_orders(self) -> List[Order]:
        if not self.is_connected():
            return []
        
        open_trades = self._ib.openTrades()
        orders = []
        
        for trade in open_trades:
            order = Order(
                ticker=trade.contract.symbol,
                side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
                order_type=self._map_order_type(trade.order),
                quantity=int(trade.order.totalQuantity),
                order_id=str(trade.order.orderId),
                status=self._map_order_status(trade.orderStatus.status),
            )
            if hasattr(trade.order, 'lmtPrice') and trade.order.lmtPrice:
                order.limit_price = float(trade.order.lmtPrice)
            if hasattr(trade.order, 'auxPrice') and trade.order.auxPrice:
                order.stop_price = float(trade.order.auxPrice)
            orders.append(order)
        
        return orders
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        open_orders = self.get_open_orders()
        for o in open_orders:
            if o.order_id == order_id:
                return o
        return None
    
    # --- History ---
    
    def get_price_history(self, ticker: str, period: str = "1M",
                          bar_size: str = "1 day") -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        
        contract = self._Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        
        # Map period
        duration = {"1D": "1 D", "5D": "5 D", "1M": "1 M", "3M": "3 M",
                     "6M": "6 M", "1Y": "1 Y"}.get(period, "1 M")
        
        bar_size_str = {"1 min": "1 min", "5 mins": "5 mins", "1 hour": "1 hour",
                         "1 day": "1 day"}.get(bar_size, "1 day")
        
        bars = self._ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting=bar_size_str, whatToShow="TRADES", useRTH=True
        )
        
        result = []
        for bar in bars:
            result.append({
                "date": bar.date.timestamp() if hasattr(bar.date, 'timestamp') else 0,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })
        
        return result
    
    # --- Helpers ---
    
    @staticmethod
    def _map_order_status(status: str) -> OrderStatus:
        mapping = {
            "PendingSubmit": OrderStatus.PENDING,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
        }
        return mapping.get(status, OrderStatus.PENDING)
    
    @staticmethod
    def _map_order_type(ib_order) -> OrderType:
        from ib_insync import MarketOrder, LimitOrder, StopOrder
        if isinstance(ib_order, MarketOrder):
            return OrderType.MARKET
        elif isinstance(ib_order, LimitOrder):
            return OrderType.LIMIT
        elif isinstance(ib_order, StopOrder):
            return OrderType.STOP
        return OrderType.MARKET
