"""
StockAgent Execution Layer
==========================
Risk management + trade execution.

The execution layer sits between the analysis engine and the broker.
It:
1. Validates trades against risk limits
2. Calculates position sizes
3. Manages stop-loss and take-profit orders
4. Tracks daily P&L for circuit breakers
5. Records all decisions to the trade journal
"""

from .risk_engine import RiskEngine, RiskCheck, RiskViolation
from .executor import TradeExecutor, ExecutionResult

__all__ = [
    "RiskEngine", "RiskCheck", "RiskViolation",
    "TradeExecutor", "ExecutionResult",
]
