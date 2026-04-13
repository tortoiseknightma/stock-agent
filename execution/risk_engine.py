"""
Risk Management Engine
=======================
Enforces trading rules and position limits BEFORE any order is submitted.

Risk controls (all configurable):
- Max position concentration (e.g., no more than 20% in one stock)
- Max single trade size (e.g., no more than 5% of portfolio)
- Daily loss limit (e.g., stop trading if down 3% today)
- Maximum trades per day
- Large trade confirmation threshold
- Stop-loss and take-profit levels

The risk engine is the LAST line of defense. If it says no, the trade
does not happen — regardless of how good the analysis looks.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.config import RiskConfig
from core.broker.base import BaseBroker, AccountInfo, Position, Order, OrderSide, OrderType


class RiskLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


@dataclass
class RiskViolation:
    """A specific risk rule violation."""
    rule: str
    level: RiskLevel
    message: str
    current_value: float = 0
    limit_value: float = 0
    
    def __str__(self):
        prefix = {"info": "ℹ", "warning": "⚠", "blocker": "🚫"}[self.level.value]
        return f"{prefix} [{self.rule}] {self.message}"


@dataclass
class RiskCheck:
    """Result of a pre-trade risk check."""
    approved: bool
    violations: List[RiskViolation] = field(default_factory=list)
    warnings: List[RiskViolation] = field(default_factory=list)
    adjusted_quantity: int = 0     # Risk-adjusted position size
    reasoning: str = ""
    
    @property
    def has_blockers(self) -> bool:
        return any(v.level == RiskLevel.BLOCKER for v in self.violations)
    
    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


class RiskEngine:
    """
    Pre-trade risk management engine.
    
    Usage:
        risk = RiskEngine(config, broker)
        
        # Check if a trade is allowed
        check = risk.check_order(Order(
            ticker="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=50
        ))
        
        if check.approved:
            # Execute with risk-adjusted quantity
            order.quantity = check.adjusted_quantity
            broker.submit_order(order)
        else:
            print("Trade blocked:", check.violations)
    """
    
    def __init__(self, config: RiskConfig, broker: BaseBroker):
        self.config = config
        self.broker = broker
        
        # Track daily state
        self._daily_trades = 0
        self._daily_start_value = None
        self._daily_start_time = None
        self._reset_daily_tracking()
    
    def check_order(self, order: Order) -> RiskCheck:
        """
        Run all risk checks on a proposed order.
        
        Returns RiskCheck with:
        - approved: whether the trade can proceed
        - violations: list of blocking violations
        - warnings: list of non-blocking warnings
        - adjusted_quantity: recommended position size after risk adjustment
        """
        violations = []
        warnings = []
        
        # Get current state
        account = self.broker.get_account_info()
        position = self.broker.get_position(order.ticker)
        current_price = self.broker.get_current_price(order.ticker)
        
        if not current_price or current_price <= 0:
            return RiskCheck(
                approved=False,
                violations=[RiskViolation(
                    "price_data", RiskLevel.BLOCKER,
                    f"Cannot get price for {order.ticker}"
                )]
            )
        
        trade_value = order.quantity * current_price
        
        # --- Check 1: Position concentration ---
        if order.side == OrderSide.BUY:
            existing_value = (position.quantity * position.current_price) if position else 0
            new_total = existing_value + trade_value
            position_pct = new_total / account.net_liquidation if account.net_liquidation > 0 else 0
            
            if position_pct > self.config.max_position_pct:
                violations.append(RiskViolation(
                    "max_position", RiskLevel.BLOCKER,
                    f"Position would be {position_pct:.1%} of portfolio (limit: {self.config.max_position_pct:.0%})",
                    position_pct, self.config.max_position_pct
                ))
        
        # --- Check 2: Trade size ---
        trade_pct = trade_value / account.net_liquidation if account.net_liquidation > 0 else 0
        if trade_pct > self.config.max_trade_pct:
            violations.append(RiskViolation(
                "max_trade_size", RiskLevel.BLOCKER,
                f"Trade is {trade_pct:.1%} of portfolio (limit: {self.config.max_trade_pct:.0%})",
                trade_pct, self.config.max_trade_pct
            ))
        
        # --- Check 3: Buying power ---
        if order.side == OrderSide.BUY:
            if trade_value > account.buying_power:
                violations.append(RiskViolation(
                    "buying_power", RiskLevel.BLOCKER,
                    f"Insufficient buying power: need ${trade_value:.2f}, have ${account.buying_power:.2f}",
                    trade_value, account.buying_power
                ))
        
        # --- Check 4: Daily loss limit ---
        if self._daily_start_value:
            daily_pnl_pct = (account.net_liquidation - self._daily_start_value) / self._daily_start_value
            if daily_pnl_pct < -self.config.daily_loss_limit_pct:
                violations.append(RiskViolation(
                    "daily_loss_limit", RiskLevel.BLOCKER,
                    f"Daily loss {daily_pnl_pct:.1%} exceeds limit ({self.config.daily_loss_limit_pct:.0%})",
                    daily_pnl_pct, self.config.daily_loss_limit_pct
                ))
        
        # --- Check 5: Daily trade count ---
        if self._daily_trades >= self.config.max_trades_per_day:
            violations.append(RiskViolation(
                "max_daily_trades", RiskLevel.BLOCKER,
                f"Daily trade limit reached ({self._daily_trades}/{self.config.max_trades_per_day})",
                self._daily_trades, self.config.max_trades_per_day
            ))
        
        # --- Check 6: Large trade warning ---
        if trade_value > self.config.large_trade_threshold:
            warnings.append(RiskViolation(
                "large_trade", RiskLevel.WARNING,
                f"Large trade (${trade_value:.2f}) — consider splitting or confirming",
                trade_value, self.config.large_trade_threshold
            ))
        
        # --- Check 7: Short selling (if applicable) ---
        if order.side == OrderSide.SELL:
            if not position or position.quantity < order.quantity:
                violations.append(RiskViolation(
                    "short_selling", RiskLevel.BLOCKER,
                    f"Cannot sell {order.quantity} shares — only own {position.quantity if position else 0}",
                    order.quantity, position.quantity if position else 0
                ))
        
        # --- Calculate adjusted quantity ---
        adjusted_qty = self._calculate_safe_quantity(
            order, account, position, current_price, trade_value
        )
        
        # Build reasoning
        if violations:
            reasoning = f"BLOCKED: {'; '.join(v.message for v in violations)}"
        elif warnings:
            reasoning = f"APPROVED with warnings: {'; '.join(v.message for v in warnings)}"
        else:
            reasoning = "APPROVED — all risk checks passed"
        
        return RiskCheck(
            approved=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            adjusted_quantity=adjusted_qty,
            reasoning=reasoning,
        )
    
    def check_stop_loss(self, position: Position) -> Optional[Order]:
        """
        Check if a position has hit its stop-loss level.
        Returns a sell order if triggered, None otherwise.
        """
        if position.quantity <= 0:
            return None
        
        stop_price = position.avg_cost * (1 - self.config.stop_loss_pct)
        
        if position.current_price <= stop_price:
            return Order(
                ticker=position.ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
            )
        
        return None
    
    def check_take_profit(self, position: Position) -> Optional[Order]:
        """
        Check if a position has hit its take-profit level.
        Returns a sell order if triggered, None otherwise.
        """
        if position.quantity <= 0:
            return None
        
        target_price = position.avg_cost * (1 + self.config.take_profit_pct)
        
        if position.current_price >= target_price:
            return Order(
                ticker=position.ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
            )
        
        return None
    
    def check_portfolio_health(self) -> List[RiskViolation]:
        """
        Check overall portfolio health.
        Call periodically (e.g., every 30 minutes) to catch deteriorating positions.
        """
        warnings = []
        account = self.broker.get_account_info()
        positions = self.broker.get_positions()
        
        # Daily P&L check
        if self._daily_start_value:
            daily_pnl_pct = (account.net_liquidation - self._daily_start_value) / self._daily_start_value
            if daily_pnl_pct < -self.config.daily_loss_limit_pct * 0.7:
                warnings.append(RiskViolation(
                    "daily_loss_warning", RiskLevel.WARNING,
                    f"Approaching daily loss limit: {daily_pnl_pct:.1%}",
                    daily_pnl_pct, -self.config.daily_loss_limit_pct
                ))
        
        # Position stop-loss checks
        for pos in positions:
            if pos.quantity > 0:
                pnl_pct = (pos.current_price - pos.avg_cost) / pos.avg_cost
                if pnl_pct < -self.config.stop_loss_pct * 0.8:
                    warnings.append(RiskViolation(
                        f"stop_loss_warning_{pos.ticker}", RiskLevel.WARNING,
                        f"{pos.ticker} approaching stop-loss: {pnl_pct:.1%} loss",
                        pnl_pct, -self.config.stop_loss_pct
                    ))
        
        # Max drawdown check
        max_value = max(account.net_liquidation, self._daily_start_value or account.net_liquidation)
        drawdown = (max_value - account.net_liquidation) / max_value if max_value > 0 else 0
        if drawdown > self.config.max_drawdown_pct * 0.7:
            warnings.append(RiskViolation(
                "max_drawdown", RiskLevel.WARNING,
                f"Portfolio drawdown {drawdown:.1%} approaching limit ({self.config.max_drawdown_pct:.0%})",
                drawdown, self.config.max_drawdown_pct
            ))
        
        return warnings
    
    def reset_daily(self):
        """Reset daily tracking counters. Call at start of each trading day."""
        self._reset_daily_tracking()
    
    def increment_trade_count(self):
        """Increment daily trade counter after successful execution."""
        self._daily_trades += 1
    
    def _calculate_safe_quantity(self, order: Order, account: AccountInfo,
                                  position: Optional[Position],
                                  current_price: float,
                                  trade_value: float) -> int:
        """
        Calculate a safe position size that respects all risk limits.
        """
        if order.side == OrderSide.SELL:
            return order.quantity  # No adjustment needed for sells
        
        # Max position size based on concentration limit
        existing_value = (position.quantity * position.current_price) if position else 0
        max_total_value = account.net_liquidation * self.config.max_position_pct
        max_additional = max_total_value - existing_value
        
        # Max trade size based on trade limit
        max_trade_value = account.net_liquidation * self.config.max_trade_pct
        
        # Use the minimum of: requested, position limit, trade limit, buying power
        safe_value = min(
            trade_value,
            max(max_additional, 0),
            max_trade_value,
            account.buying_power * 0.95  # Leave 5% buffer
        )
        
        safe_qty = int(safe_value / current_price)
        return max(0, safe_qty)
    
    def _reset_daily_tracking(self):
        """Reset daily state."""
        self._daily_trades = 0
        account = self.broker.get_account_info()
        self._daily_start_value = account.net_liquidation
        self._daily_start_time = time.time()
