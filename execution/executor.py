"""
Trade Executor
===============
Orchestrates the full trade lifecycle:
1. Analysis signal → Risk check → Order submission → Journal recording

Handles different trading modes:
- SIMULATED: Execute in mock broker
- PAPER: Execute in IBKR paper account
- ADVISORY: Generate signal, don't execute
- LIVE: Execute with confirmation
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.config import AppConfig, TradingMode
from core.broker.base import BaseBroker, Order, OrderResult, OrderSide, OrderType
from core.memory.trade_journal import TradeJournal, TradeRecord, DecisionRecord, AnalysisSnapshot, Signal
from analysis.composite.composite import CompositeSignal
from .risk_engine import RiskEngine, RiskCheck


@dataclass
class ExecutionResult:
    """Complete result of a trade execution attempt."""
    success: bool
    signal: CompositeSignal
    risk_check: RiskCheck
    order_result: Optional[OrderResult] = None
    decision_id: Optional[int] = None
    journal_recorded: bool = False
    message: str = ""


class TradeExecutor:
    """
    Main execution interface.
    
    Usage:
        executor = TradeExecutor(config, broker, risk_engine, journal)
        
        # Execute a buy recommendation
        result = executor.execute_buy("AAPL", signal, portfolio_value=100000)
        
        # Execute a sell recommendation
        result = executor.execute_sell("AAPL", signal)
        
        # Check and execute stop-losses
        triggered = executor.check_stop_losses()
    """
    
    def __init__(self, config: AppConfig, broker: BaseBroker,
                 risk_engine: RiskEngine, journal: TradeJournal):
        self.config = config
        self.broker = broker
        self.risk = risk_engine
        self.journal = journal
    
    def execute_buy(self, ticker: str, signal: CompositeSignal,
                    portfolio_value: float = 0,
                    target_pct: float = None) -> ExecutionResult:
        """
        Execute a buy order based on analysis signal.
        
        Args:
            ticker: Stock symbol
            signal: Composite analysis signal
            portfolio_value: Current portfolio value (for journal)
            target_pct: Target position size as % of portfolio (overrides default)
        """
        # Get current state
        account = self.broker.get_account_info()
        position = self.broker.get_position(ticker)
        current_price = self.broker.get_current_price(ticker)
        
        if not current_price:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=RiskCheck(approved=False),
                message=f"Cannot get price for {ticker}"
            )
        
        # Calculate position size
        if target_pct:
            # Use specified target percentage
            target_value = account.net_liquidation * target_pct
            existing_value = (position.quantity * current_price) if position else 0
            trade_value = max(0, target_value - existing_value)
            quantity = int(trade_value / current_price)
        else:
            # Use default sizing based on signal strength
            confidence_factor = signal.confidence * 0.5  # Max 50% for strong signals
            quantity = self.broker.calculate_position_size(ticker, confidence_factor)
        
        if quantity <= 0:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=RiskCheck(approved=False),
                message="Calculated quantity is 0"
            )
        
        # Create order
        order = Order(
            ticker=ticker,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        
        # Risk check
        risk_check = self.risk.check_order(order)
        
        # Record decision (even if blocked)
        analysis_snapshot = AnalysisSnapshot(
            technical_score=signal.technical.score if signal.technical else 0,
            fundamental_score=signal.fundamental.score if signal.fundamental else 0,
            sentiment_score=signal.sentiment.score if signal.sentiment else 0,
            composite_score=signal.composite_score,
            signal=Signal.BUY if signal.composite_score > 0 else Signal.HOLD,
            reasoning=signal.recommendation,
        )
        
        decision = DecisionRecord(
            ticker=ticker,
            signal=Signal.BUY if signal.composite_score > 0 else Signal.HOLD,
            analysis=analysis_snapshot,
            portfolio_value=account.net_liquidation,
            current_position=position.quantity if position else 0,
            current_price=current_price,
            intended_action=f"buy {quantity} shares at market",
            confidence=signal.confidence,
            risk_assessment=risk_check.reasoning,
            executed=risk_check.approved,
            execution_reason=risk_check.reasoning,
        )
        
        decision_id = self.journal.record_decision(decision)
        
        # Check if we should execute
        if not risk_check.approved:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=risk_check,
                decision_id=decision_id,
                journal_recorded=True,
                message=f"Risk check failed: {risk_check.reasoning}"
            )
        
        # Advisory mode: don't execute
        if self.config.trading_mode == TradingMode.ADVISORY:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=risk_check,
                decision_id=decision_id,
                journal_recorded=True,
                message=f"ADVISORY mode — would buy {quantity} {ticker}"
            )
        
        # Use risk-adjusted quantity
        order.quantity = risk_check.adjusted_quantity
        if order.quantity <= 0:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=risk_check,
                decision_id=decision_id,
                journal_recorded=True,
                message="Risk engine reduced quantity to 0"
            )
        
        # Execute
        order_result = self.broker.submit_order(order)
        
        if order_result.success and order_result.order:
            self.risk.increment_trade_count()
            
            # Record trade
            trade_record = TradeRecord(
                ticker=ticker,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=order.quantity,
                price=order_result.order.filled_price or current_price,
                portfolio_value_before=account.net_liquidation,
                decision_id=decision_id,
                commission=order_result.order.commission,
            )
            self.journal.record_trade(trade_record)
            
            return ExecutionResult(
                success=True, signal=signal,
                risk_check=risk_check,
                order_result=order_result,
                decision_id=decision_id,
                journal_recorded=True,
                message=f"Bought {order.quantity} {ticker} @ ${order_result.order.filled_price:.2f}"
            )
        else:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=risk_check,
                order_result=order_result,
                decision_id=decision_id,
                journal_recorded=True,
                message=f"Order failed: {order_result.error_message}"
            )
    
    def execute_sell(self, ticker: str, signal: CompositeSignal = None,
                     quantity: int = None,
                     reason: str = "analysis") -> ExecutionResult:
        """
        Execute a sell order.
        
        Args:
            ticker: Stock symbol
            signal: Analysis signal (optional)
            quantity: Shares to sell (None = sell all)
            reason: Reason for the sell
        """
        account = self.broker.get_account_info()
        position = self.broker.get_position(ticker)
        current_price = self.broker.get_current_price(ticker)
        
        if not position or position.quantity <= 0:
            return ExecutionResult(
                success=False, signal=signal or CompositeSignal(ticker=ticker, composite_score=0, signal="hold", confidence=0),
                risk_check=RiskCheck(approved=False),
                message=f"No position in {ticker}"
            )
        
        sell_qty = quantity or position.quantity
        sell_qty = min(sell_qty, position.quantity)
        
        # Create order
        order = Order(
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=sell_qty,
        )
        
        # Risk check (minimal for sells)
        risk_check = self.risk.check_order(order)
        
        # Record decision
        analysis_snapshot = AnalysisSnapshot(
            technical_score=signal.technical.score if signal and signal.technical else 0,
            fundamental_score=signal.fundamental.score if signal and signal.fundamental else 0,
            sentiment_score=signal.sentiment.score if signal and signal.sentiment else 0,
            composite_score=signal.composite_score if signal else 0,
            signal=Signal.SELL if signal and signal.composite_score < 0 else Signal.HOLD,
            reasoning=signal.recommendation if signal else reason,
        )
        
        decision = DecisionRecord(
            ticker=ticker,
            signal=Signal.SELL,
            analysis=analysis_snapshot,
            portfolio_value=account.net_liquidation,
            current_position=position.quantity,
            current_price=current_price or 0,
            intended_action=f"sell {sell_qty} shares",
            confidence=signal.confidence if signal else 0.5,
            risk_assessment=risk_check.reasoning,
            executed=True,
            execution_reason=reason,
        )
        
        decision_id = self.journal.record_decision(decision)
        
        # Advisory mode
        if self.config.trading_mode == TradingMode.ADVISORY:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=risk_check,
                decision_id=decision_id,
                journal_recorded=True,
                message=f"ADVISORY mode — would sell {sell_qty} {ticker}"
            )
        
        # Execute
        order_result = self.broker.submit_order(order)
        
        if order_result.success:
            self.risk.increment_trade_count()
            
            pnl = (current_price - position.avg_cost) * sell_qty if current_price else 0
            
            trade_record = TradeRecord(
                ticker=ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=sell_qty,
                price=order_result.order.filled_price or current_price,
                portfolio_value_before=account.net_liquidation,
                decision_id=decision_id,
                commission=order_result.order.commission,
                realized_pnl=pnl,
            )
            self.journal.record_trade(trade_record)
            
            return ExecutionResult(
                success=True, signal=signal,
                risk_check=risk_check,
                order_result=order_result,
                decision_id=decision_id,
                journal_recorded=True,
                message=f"Sold {sell_qty} {ticker} @ ${order_result.order.filled_price:.2f} (P&L: ${pnl:.2f})"
            )
        else:
            return ExecutionResult(
                success=False, signal=signal,
                risk_check=risk_check,
                order_result=order_result,
                decision_id=decision_id,
                message=f"Sell failed: {order_result.error_message}"
            )
    
    def check_stop_losses(self) -> list:
        """
        Check all positions for stop-loss triggers.
        Returns list of ExecutionResults for triggered stops.
        """
        results = []
        positions = self.broker.get_positions()
        
        for position in positions:
            stop_order = self.risk.check_stop_loss(position)
            if stop_order:
                result = self.execute_sell(
                    position.ticker,
                    quantity=stop_order.quantity,
                    reason=f"stop_loss triggered at ${position.current_price:.2f}"
                )
                results.append(result)
        
        return results
    
    def check_take_profits(self) -> list:
        """
        Check all positions for take-profit triggers.
        Returns list of ExecutionResults for triggered profits.
        """
        results = []
        positions = self.broker.get_positions()
        
        for position in positions:
            tp_order = self.risk.check_take_profit(position)
            if tp_order:
                result = self.execute_sell(
                    position.ticker,
                    quantity=tp_order.quantity,
                    reason=f"take_profit triggered at ${position.current_price:.2f}"
                )
                results.append(result)
        
        return results
