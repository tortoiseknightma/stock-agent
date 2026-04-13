"""
StockAgent Broker Layer
=======================
Abstract broker interface with two implementations:
1. SimulatedBroker — local mock with geometric Brownian motion
2. IBKRBroker — real Interactive Brokers via ib_insync

Factory function selects implementation based on config.

Design principle: ALL code that interacts with the market goes through
this abstraction. The rest of the system never touches IBKR directly.
"""

from .base import BaseBroker, Position, Order, OrderResult, AccountInfo
from .simulated import SimulatedBroker
from .factory import create_broker

__all__ = [
    "BaseBroker", "Position", "Order", "OrderResult", "AccountInfo",
    "SimulatedBroker", "create_broker",
]
