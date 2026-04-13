"""
StockAgent — AI-Powered Investment Agent
========================================
An intelligent stock investment agent that combines technical analysis,
fundamental analysis, and sentiment analysis to generate trading signals.

Supports IBKR (Interactive Brokers) for live/paper trading and includes
a simulated broker for development and backtesting.
"""

try:
    from .agent import StockAgentAgent
    from .core.config import AppConfig, TradingMode, load_config
except ImportError:
    pass

__version__ = "0.1.0"
__all__ = ["StockAgentAgent", "AppConfig", "TradingMode", "load_config"]
