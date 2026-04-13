"""
Broker Factory
==============
Creates the appropriate broker implementation based on configuration.
"""

from ..config import AppConfig, TradingMode
from .base import BaseBroker
from .simulated import SimulatedBroker


def create_broker(config: AppConfig) -> BaseBroker:
    """
    Factory: create the right broker based on trading mode.
    
    - simulated: SimulatedBroker (local mock, no external dependencies)
    - paper/live: IBKRBroker (requires ib_insync + TWS/Gateway)
    - advisory: SimulatedBroker (analysis only, uses sim for data)
    """
    if config.trading_mode in (TradingMode.SIMULATED, TradingMode.ADVISORY):
        return SimulatedBroker(
            initial_capital=config.simulated.initial_capital,
            default_portfolio=config.simulated.default_portfolio,
            slippage_pct=config.simulated.slippage_pct,
            commission_per_trade=config.simulated.commission_per_trade,
            state_file="data/sim_state.json",
        )
    
    elif config.trading_mode in (TradingMode.PAPER, TradingMode.LIVE):
        from .ibkr_broker import IBKRBroker
        readonly = (config.trading_mode == TradingMode.ADVISORY)
        
        return IBKRBroker(
            host=config.ibkr.host,
            port=config.ibkr.port,
            client_id=config.ibkr.client_id,
            timeout=config.ibkr.timeout,
            readonly=readonly or config.ibkr.readonly,
        )
    
    else:
        raise ValueError(f"Unknown trading mode: {config.trading_mode}")
